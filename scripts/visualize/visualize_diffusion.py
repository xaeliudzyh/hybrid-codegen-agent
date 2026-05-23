"""Visualize LLaDA diffusion denoising for a multi-function-call task.

Runs on the GPU server.  Captures the full sequence at every diffusion step
(or every N-th step) so that later -- locally -- one can inspect:
  * which tokens were unmasked at which step,
  * whether (and when) `<function_call>` blocks crystallise,
  * how different remasking strategies differ visually.

Output is a single self-contained JSON file (see `--out`).
Pair it with `plot_visualization.py` (run locally on the JSON).

Notes on the implementation
---------------------------
* Re-uses `DiffusionEngine._llada_generate`'s `step_callback` interface.
* The callback (`StateSnapshotter`) only *observes* the running tensor `x`
  and writes a Python list of dicts; it does not mutate state and never
  returns ``True`` (which would stop generation early).
* Function-call detection inside snapshots is identical in spirit to
  `EarlyFunctionDetector` (regex + mask-aware), but reports *all* fully
  crystallised FC blocks at the current step, not just the first.
* No code in `src/` is modified; this is a pure-add tool.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# expose src/ to imports
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch  # noqa: E402

from diffusion import DiffusionEngine  # noqa: E402
from diffusion.diffusion_engine import LLADA_MASK_ID  # noqa: E402
from function_calling.parser import parse_function_calls  # noqa: E402


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

GEOMETRY_TWO_SYS = """You are a function-calling assistant with two available tools:

- circle_area(radius)    returns the area of a circle (pi * r^2)
- sphere_volume(radius)  returns the volume of a sphere ((4/3) * pi * r^3)

To call a tool, emit EXACTLY this format:

<function_call>
{"name": "<tool_name>", "arguments": {"radius": <number>}}
</function_call>

ALWAYS start your response with exactly one sentence:
"I will call <tool1> and <tool2> with radius=<value>."
Right after that sentence, emit the required <function_call> blocks back-to-back.
Each block must contain exactly one JSON object with "name" and "arguments" keys.
Do NOT put any text between or after the function_call blocks.

Example.
User: For r=2, give the circle area and the sphere volume.
Assistant:
I will call circle_area and sphere_volume with radius=2.
<function_call>
{"name": "circle_area", "arguments": {"radius": 2}}
</function_call>
<function_call>
{"name": "sphere_volume", "arguments": {"radius": 2}}
</function_call>"""

GEOMETRY_TWO_TASK = (
    "For r=3, compute the circle area and the sphere volume. "
    "Start with: \"I will call circle_area and sphere_volume with radius=3.\" "
    "Then emit EXACTLY two <function_call> blocks: circle_area, then sphere_volume. "
    "Do not put any text between or after the function_call blocks."
)

GEOMETRY_THREE_SYS = """You are a function-calling assistant with three available tools:

- circle_area(radius)    returns the area of a circle (pi * r^2)
- sphere_volume(radius)  returns the volume of a sphere ((4/3) * pi * r^3)
- sphere_surface(radius) returns the surface area of a sphere (4 * pi * r^2)

To call a tool, emit EXACTLY this format:

<function_call>
{"name": "<tool_name>", "arguments": {"radius": <number>}}
</function_call>

Begin your response with 1 to 3 short sentences (no more than 30 words total)
that briefly explain your approach. Right after that explanation, emit the
required <function_call> blocks back-to-back. Each block must contain exactly
one JSON object with "name" and "arguments" keys. Do NOT put any text
between or after the function_call blocks.

Example.
User: For r=2, give the circle area and the sphere volume.
Assistant:
The circle area is pi*r^2 and the sphere volume is (4/3)*pi*r^3. I will call each tool with radius=2.
<function_call>
{"name": "circle_area", "arguments": {"radius": 2}}
</function_call>
<function_call>
{"name": "sphere_volume", "arguments": {"radius": 2}}
</function_call>"""

GEOMETRY_THREE_TASK = (
    "For r=5, compute the circle area, the sphere volume and the sphere surface area. "
    "Begin your response with 1 to 3 short sentences explaining your approach. "
    "Then emit EXACTLY three <function_call> blocks back-to-back, one per quantity, "
    "in the order: circle_area, sphere_volume, sphere_surface. "
    "Do not put any text between or after the function_call blocks."
)

ARITHMETIC_TWO_SYS = """You are an arithmetic assistant with two tools:

- add(a, b)       returns a + b
- multiply(a, b)  returns a * b

To call a tool, emit EXACTLY this format:

<function_call>
{"name": "<tool_name>", "arguments": {"a": <number>, "b": <number>}}
</function_call>

Begin your response with 1 to 3 short sentences (no more than 30 words total)
that briefly explain your approach. Right after that explanation, emit the
required <function_call> blocks back-to-back. Each block must contain exactly
one JSON object with "name" and "arguments" keys. Do NOT put any text
between or after the function_call blocks.

Example.
User: Compute (1 + 2) * 4.
Assistant:
First add 1 and 2, then multiply the result by 4. I will use add() and multiply() in sequence.
<function_call>
{"name": "add", "arguments": {"a": 1, "b": 2}}
</function_call>
<function_call>
{"name": "multiply", "arguments": {"a": 3, "b": 4}}
</function_call>"""

ARITHMETIC_TWO_TASK = (
    "Compute (3 + 4) * 5 step by step. "
    "Begin your response with 1 to 3 short sentences explaining your approach. "
    "Then emit one <function_call> for the addition (add 3 and 4), "
    "then one <function_call> for the multiplication (multiply 7 by 5). "
    "Do not put any text between or after the function_call blocks."
)

PRESETS: dict[str, tuple[str, str]] = {
    "geometry_two":   (GEOMETRY_TWO_SYS,   GEOMETRY_TWO_TASK),
    "geometry_three": (GEOMETRY_THREE_SYS, GEOMETRY_THREE_TASK),
    "arithmetic_two": (ARITHMETIC_TWO_SYS, ARITHMETIC_TWO_TASK),
}


def build_prompt(system_prompt: str, task: str) -> str:
    """`DiffusionEngine._format_prompt` splits the prompt on `'\\nTask:\\n'`."""
    return f"{system_prompt}\n\nTask:\n{task}"


# ---------------------------------------------------------------------------
# State snapshotter
# ---------------------------------------------------------------------------


class StateSnapshotter:
    """`step_callback` for `DiffusionEngine._llada_generate`.

    Records the full state of the *generated* sub-sequence at every step
    (and a more verbose "snapshot" every ``save_every`` steps).
    """

    def __init__(
        self,
        *,
        tokenizer,
        prompt_len: int,
        save_every: int = 1,
        fc_anchor_ids: Optional[torch.Tensor] = None,
    ) -> None:
        self._tok = tokenizer
        self._prompt_len = prompt_len
        self._save_every = max(1, int(save_every))
        if fc_anchor_ids is not None:
            self._fc_anchor_set: set[int] = set(int(t) for t in fc_anchor_ids.tolist())
        else:
            self._fc_anchor_set = set()
        self._mask_text: str = tokenizer.decode(
            [LLADA_MASK_ID], skip_special_tokens=False
        )

        # populated lazily on first call:
        self.gen_length: Optional[int] = None
        self._unmask_step: list[Optional[int]] = []
        self._prev_mask: list[bool] = []

        # outputs
        self.snapshots: list[dict[str, Any]] = []
        self.steps_seen: int = 0

    # ------------------------------------------------------------------

    def __call__(self, x: torch.Tensor, step: int) -> bool:
        """Observe state at this step; never request early stop."""
        gen_part = x[0, self._prompt_len:].detach().cpu()
        ids: list[int] = gen_part.tolist()

        if self.gen_length is None:
            self.gen_length = len(ids)
            self._unmask_step = [None] * self.gen_length
            self._prev_mask = [True] * self.gen_length

        self.steps_seen += 1
        cur_mask: list[bool] = [tid == LLADA_MASK_ID for tid in ids]

        # update first-unmask-step for each position
        newly_unmasked: list[int] = []
        for i, (was_masked, is_masked) in enumerate(zip(self._prev_mask, cur_mask)):
            if was_masked and not is_masked:
                self._unmask_step[i] = int(step)
                newly_unmasked.append(i)
        self._prev_mask = cur_mask

        # always record a thin "step record" for the first/last step;
        # otherwise only every save_every-th step.
        full_snapshot = (step % self._save_every == 0)
        if not full_snapshot and step != 1:
            return False

        decoded_with_masks = self._tok.decode(ids, skip_special_tokens=False)
        decoded_clean = decoded_with_masks.replace(self._mask_text, "\u00b7")

        fc_anchor_positions = [
            i for i, (tid, is_m) in enumerate(zip(ids, cur_mask))
            if (not is_m) and (int(tid) in self._fc_anchor_set)
        ]

        # detect any fully-crystallised function_call blocks in the current
        # decoded text. (Same regex the parser uses; mask-aware: skip if
        # mask token text appears inside the match.)
        try:
            calls = parse_function_calls(decoded_with_masks)
        except Exception:
            calls = []
        calls_serialized = [
            {
                "name": c.name,
                "arguments": c.arguments,
                "char_start": int(c.start_position),
                "char_end": int(c.end_position),
                "fully_unmasked": (
                    self._mask_text
                    not in decoded_with_masks[c.start_position : c.end_position]
                ),
            }
            for c in calls
        ]

        snap: dict[str, Any] = {
            "step": int(step),
            "num_masked": int(sum(cur_mask)),
            "num_unmasked": int(self.gen_length - sum(cur_mask)),
            "newly_unmasked_positions": newly_unmasked,
            "fc_anchor_positions_unmasked": fc_anchor_positions,
            "decoded_with_masks": decoded_with_masks,
            "decoded_clean": decoded_clean,
            "function_calls_detected": calls_serialized,
        }
        self.snapshots.append(snap)
        return False

    @property
    def unmask_steps(self) -> list[Optional[int]]:
        return self._unmask_step


# ---------------------------------------------------------------------------
# Token-role classification (post-generation)
# ---------------------------------------------------------------------------


_FC_SPAN_RE = re.compile(
    r"<function_call>\s*\{.*?\}\s*</function_call>",
    re.DOTALL,
)


def compute_token_roles(
    gen_ids: list[int],
    tokenizer,
    fc_anchor_set: set[int],
) -> dict[str, Any]:
    """Classify each generated-token position as one of:

    - ``"explanation"`` - the token's character span lies *before* the first
      ``<function_call>`` block in the final decoded text.
    - ``"fc_anchor"``   - the token's id is one of the FC-anchor ids AND its
      character span lies inside a ``<function_call>...</function_call>``
      block.
    - ``"fc_body"``     - any other token whose char span lies inside an FC
      block.
    - ``"other"``       - everything else (text between/after FC blocks,
      EOS/padding, etc.).

    The classification is computed *after generation* using the final
    decoded text - we cannot know FC span boundaries before generation
    completes.

    If no FC blocks were emitted at all, every position is labelled
    ``"explanation"`` so that the stabilisation curve still has a
    meaningful denominator.

    Returns a dict with keys ``roles``, ``char_starts``, ``char_ends``,
    ``fc_spans``, ``full_decoded``.
    """
    n = len(gen_ids)
    char_ends: list[int] = [0] * n
    full_decoded = ""
    for i in range(n):
        full_decoded = tokenizer.decode(gen_ids[: i + 1], skip_special_tokens=False)
        char_ends[i] = len(full_decoded)
    char_starts: list[int] = [0] + char_ends[:-1]

    fc_spans: list[list[int]] = [
        [int(m.start()), int(m.end())]
        for m in _FC_SPAN_RE.finditer(full_decoded)
    ]
    first_fc_start: Optional[int] = fc_spans[0][0] if fc_spans else None

    roles: list[str] = []
    for i in range(n):
        s, e = char_starts[i], char_ends[i]
        mid = (s + e) / 2 if e > s else s
        in_fc = any(fs <= mid < fe for fs, fe in fc_spans)
        if in_fc:
            roles.append("fc_anchor" if int(gen_ids[i]) in fc_anchor_set else "fc_body")
        elif first_fc_start is None:
            roles.append("explanation")
        elif mid < first_fc_start:
            roles.append("explanation")
        else:
            roles.append("other")

    return {
        "roles": roles,
        "char_starts": char_starts,
        "char_ends": char_ends,
        "fc_spans": fc_spans,
        "full_decoded": full_decoded,
    }


def _augment_snapshots_with_role_progress(
    snapshots: list[dict[str, Any]],
    roles: list[str],
    unmask_steps: list[Optional[int]],
) -> tuple[int, int]:
    """For every snapshot, fill in:

      - ``num_unmasked_explanation``   running count of explanation positions revealed
      - ``num_unmasked_fc``            running count of FC positions revealed
      - ``total_explanation``          denominator (constant across snapshots)
      - ``total_fc``                   denominator (constant)

    Returns ``(total_explanation, total_fc)``.
    """
    fc_indices = [i for i, r in enumerate(roles) if r in ("fc_anchor", "fc_body")]
    exp_indices = [i for i, r in enumerate(roles) if r == "explanation"]
    total_fc = len(fc_indices)
    total_exp = len(exp_indices)

    for snap in snapshots:
        k = int(snap["step"])
        n_exp = sum(
            1 for i in exp_indices
            if unmask_steps[i] is not None and unmask_steps[i] <= k
        )
        n_fc = sum(
            1 for i in fc_indices
            if unmask_steps[i] is not None and unmask_steps[i] <= k
        )
        snap["num_unmasked_explanation"] = n_exp
        snap["num_unmasked_fc"] = n_fc
        snap["total_explanation"] = total_exp
        snap["total_fc"] = total_fc
    return total_exp, total_fc


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture LLaDA denoising trajectory for a multi-FC task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--device", default="auto")

    # remasking-strategy parameters (match DiffusionEngine defaults)
    parser.add_argument(
        "--remasking",
        default="low_confidence",
        choices=["low_confidence", "random", "entropy", "fc_priority", "structural_boost"],
    )
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=128)
    parser.add_argument("--max_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--fc_boost", type=float, default=0.15)
    parser.add_argument("--structural_boost", type=float, default=0.15)
    parser.add_argument("--structural_window", type=int, default=2)

    # snapshot frequency
    parser.add_argument(
        "--save_every", type=int, default=1,
        help="Save full snapshot every N diffusion steps. "
             "1 = every step. The per-token first-unmask-step record is always kept.",
    )

    # task / prompt
    parser.add_argument(
        "--task_preset", default="geometry_two", choices=list(PRESETS.keys()) + ["custom"],
        help="Built-in (system_prompt, task) pair to use, or 'custom' to read from files.",
    )
    parser.add_argument("--system_prompt_file", default=None)
    parser.add_argument("--task_file", default=None)

    # output
    parser.add_argument("--out", default=None,
                        help="Output JSON path. "
                             "Default: scripts/visualize/out/visualize_<remasking>_<timestamp>.json")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # ---- pick prompt --------------------------------------------------
    if args.task_preset == "custom":
        if args.system_prompt_file is None or args.task_file is None:
            parser.error("--task_preset custom requires --system_prompt_file and --task_file")
        sys_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
        task = Path(args.task_file).read_text(encoding="utf-8").strip()
    else:
        sys_prompt, task = PRESETS[args.task_preset]
        if args.system_prompt_file is not None:
            sys_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8").strip()
        if args.task_file is not None:
            task = Path(args.task_file).read_text(encoding="utf-8").strip()

    full_prompt = build_prompt(sys_prompt, task)

    print("=" * 60)
    print("VISUALIZE LLaDA DENOISING")
    print(f"  model       : {args.model}")
    print(f"  remasking   : {args.remasking}")
    print(f"  steps       : {args.steps} | block_length: {args.block_length}")
    print(f"  temperature : {args.temperature} | save_every : {args.save_every}")
    print(f"  task_preset : {args.task_preset}")
    print("=" * 60)

    # ---- engine -------------------------------------------------------
    engine = DiffusionEngine(
        model_name_or_path=args.model,
        device=args.device,
        steps=args.steps,
        block_length=args.block_length,
        remasking=args.remasking,
        fc_boost=args.fc_boost,
        structural_boost=args.structural_boost,
        structural_window=args.structural_window,
    )

    # ---- tokenize -----------------------------------------------------
    formatted = engine._format_prompt(full_prompt)
    enc = engine.tokenizer(
        formatted,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(engine._device)
    attention_mask = enc["attention_mask"].to(engine._device)
    prompt_len = int(input_ids.shape[1])

    # round gen_length up to a multiple of block_length, and steps to a
    # multiple of num_blocks (replicates `engine.generate()`'s behaviour).
    gen_length = args.max_tokens
    if gen_length % args.block_length != 0:
        gen_length = ((gen_length // args.block_length) + 1) * args.block_length
    num_blocks = gen_length // args.block_length
    eff_steps = args.steps
    if eff_steps % num_blocks != 0:
        eff_steps = ((eff_steps // num_blocks) + 1) * num_blocks

    snap = StateSnapshotter(
        tokenizer=engine.tokenizer,
        prompt_len=prompt_len,
        save_every=args.save_every,
        fc_anchor_ids=engine._fc_anchor_ids,
    )

    # ---- generate -----------------------------------------------------
    t0 = time.perf_counter()
    output_ids = engine._llada_generate(
        prompt_ids=input_ids,
        attention_mask=attention_mask,
        gen_length=gen_length,
        temperature=args.temperature,
        steps=eff_steps,
        step_callback=snap,
    )
    wall = time.perf_counter() - t0

    # ---- decode final + parse FC --------------------------------------
    full_gen_ids = output_ids[0, prompt_len:].detach().cpu().tolist()
    truncated_ids = engine._truncate_at_eos(output_ids[0, prompt_len:]).detach().cpu().tolist()
    final_text = engine.tokenizer.decode(truncated_ids, skip_special_tokens=True)
    final_calls = parse_function_calls(final_text)

    # ---- per-token role classification (explanation / fc / other) -----
    fc_anchor_set: set[int] = set(int(t) for t in engine._fc_anchor_ids.tolist())
    roles_info = compute_token_roles(full_gen_ids, engine.tokenizer, fc_anchor_set)
    roles: list[str] = roles_info["roles"]
    fc_spans_in_final: list[list[int]] = roles_info["fc_spans"]
    final_decoded_full: str = roles_info["full_decoded"]

    total_exp, total_fc = _augment_snapshots_with_role_progress(
        snap.snapshots, roles, snap.unmask_steps,
    )

    # ---- write JSON ---------------------------------------------------
    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(__file__).resolve().parent / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"visualize_{args.remasking}_{ts}.json"
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "meta": {
            "model": args.model,
            "remasking": args.remasking,
            "steps_configured": args.steps,
            "steps_effective": eff_steps,
            "block_length": args.block_length,
            "gen_length": gen_length,
            "max_tokens_arg": args.max_tokens,
            "temperature": args.temperature,
            "fc_boost": args.fc_boost,
            "structural_boost": args.structural_boost,
            "structural_window": args.structural_window,
            "save_every": args.save_every,
            "wall_clock_seconds": round(wall, 4),
            "prompt_len": prompt_len,
            "fc_anchor_strings": list(DiffusionEngine._FC_ANCHOR_STRINGS),
            "system_prompt": sys_prompt,
            "task": task,
            "task_preset": args.task_preset,
            "seed": args.seed,
        },
        "prompt_decoded": engine.tokenizer.decode(
            input_ids[0].cpu().tolist(), skip_special_tokens=False,
        ),
        "final_full_gen_token_ids": full_gen_ids,
        "final_text": final_text,
        "final_decoded_full": final_decoded_full,
        "final_function_calls": [
            {
                "name": c.name,
                "arguments": c.arguments,
                "char_start": int(c.start_position),
                "char_end": int(c.end_position),
            }
            for c in final_calls
        ],
        "fc_spans_in_final": fc_spans_in_final,
        # length == gen_length; per-position role classification used by the
        # plotter to colour the bar chart and compute the stabilisation
        # curves (explanation vs FC).
        "token_role": roles,
        "token_role_totals": {
            "explanation": total_exp,
            "fc":          total_fc,
            "other":       len(roles) - total_exp - total_fc,
        },
        # length == gen_length; per-position step at which the position was
        # first unmasked (None means it stayed masked, e.g. trailing block
        # was skipped via early-block termination).
        "token_unmasked_at_step": snap.unmask_steps,
        # per-step records
        "snapshots": snap.snapshots,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---- console summary ---------------------------------------------
    print()
    print(f"Generation completed in {wall:.2f}s ({snap.steps_seen} step callbacks).")
    print(f"Final text:")
    print("-" * 60)
    print(final_text.strip() or "<empty>")
    print("-" * 60)
    print(f"Detected {len(final_calls)} function call(s) in final output:")
    for c in final_calls:
        print(f"  - {c.name}({c.arguments})")
    print()
    print(f"Token role breakdown (gen_length={len(roles)}): "
          f"explanation={total_exp}, fc={total_fc}, "
          f"other={len(roles) - total_exp - total_fc}")
    print(f"Saved {len(snap.snapshots)} snapshot(s) to: {out_path}")
    print(f"  File size: {out_path.stat().st_size} bytes")
    if len(final_calls) < 2 and args.task_preset != "custom":
        print()
        print("WARNING: fewer than 2 function calls were emitted in the final output.")
        print("         Try a different temperature, seed, or `--task_preset arithmetic_two`.")
    if total_exp == 0 and args.task_preset != "custom":
        print()
        print("WARNING: no explanation tokens were emitted before the first function_call.")
        print("         The model may have ignored the 'short explanation' instruction; "
              "consider rerunning with a different seed or temperature.")


if __name__ == "__main__":
    main()
