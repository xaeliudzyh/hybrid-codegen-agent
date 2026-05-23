"""Render denoising-trajectory figures from a `visualize_diffusion.py` JSON.

Runs locally (matplotlib only). Each figure has four panels:

  1. **Per-token "first-unmask step" bar chart** (top-left)
     X = generated token position (0..gen_length-1)
     Y = step at which the position was first unmasked
     Bars are colour-coded by the token's *role* in the final output:
     explanation prose, FC-anchor, FC-body, or other.

  2. **Step-vs-position mask heatmap** (middle-left)
     X = generated token position
     Y = diffusion step
     dark cell = masked, light cell = already revealed.
     Vertical bands mark explanation columns (green) and FC columns (orange).

  3. **Stabilisation curves** (bottom, full width)
     For every step, the % of explanation tokens already revealed and the
     % of `<function_call>` tokens already revealed. Directly answers the
     question: "what crystallises faster, the explanation or the function
     calls?"

  4. **Decoded text snapshots** (right column, spans rows 0-1)
     The decoded sequence at ~25 %, ~50 %, ~75 %, 100 % of the schedule
     (mask-token replaced with `\u00b7`).

Usage
-----

    python scripts/visualize/plot_visualization.py \\
        scripts/visualize/out/visualize_fc_priority_xxx.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ---------------------------------------------------------------------------
# Role-colour palette
# ---------------------------------------------------------------------------

ROLE_COLOR: dict[str, str] = {
    "explanation": "#4daf4a",  # green
    "fc_anchor":   "#e6550d",  # dark orange
    "fc_body":     "#3182bd",  # blue
    "other":       "#bbbbbb",  # light grey
}

ROLE_LABEL: dict[str, str] = {
    "explanation": "explanation prose",
    "fc_anchor":   "FC-anchor token",
    "fc_body":     "FC-body token",
    "other":       "other / EOS / pad",
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def load_run(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_mask_matrix(unmask_steps: list[int | None], n_steps: int) -> np.ndarray:
    """Return ``M[step, pos]`` where True means *masked* at that step.

    ``unmask_steps[pos]`` is the step at which position ``pos`` was first
    unmasked, or ``None`` if it was never unmasked.
    """
    n_pos = len(unmask_steps)
    # rows correspond to "after step k", k = 0..n_steps  (row 0 == fully masked)
    m = np.ones((n_steps + 1, n_pos), dtype=bool)
    for pos, k in enumerate(unmask_steps):
        if k is None:
            continue
        m[k:, pos] = False
    return m


def contiguous_runs(positions: list[int]) -> list[tuple[int, int]]:
    """Return list of (start, end_inclusive) ranges of contiguous integer runs."""
    if not positions:
        return []
    runs: list[tuple[int, int]] = []
    sp = positions[0]
    pp = sp
    for p in positions[1:]:
        if p == pp + 1:
            pp = p
        else:
            runs.append((sp, pp))
            sp, pp = p, p
    runs.append((sp, pp))
    return runs


def stabilisation_curves(
    snapshots: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Pull the (step, %explanation_revealed, %fc_revealed) curves from snapshots.

    Returns ``(steps, exp_pct, fc_pct, total_explanation, total_fc)``.
    Falls back to (zero-length arrays, 0, 0) if the snapshots predate the
    role-augmentation feature.
    """
    if not snapshots or "num_unmasked_explanation" not in snapshots[0]:
        return np.array([]), np.array([]), np.array([]), 0, 0
    steps = np.array([int(s["step"]) for s in snapshots])
    total_exp = int(snapshots[0]["total_explanation"])
    total_fc  = int(snapshots[0]["total_fc"])
    exp_pct = np.array([
        100.0 * int(s["num_unmasked_explanation"]) / total_exp if total_exp else 0.0
        for s in snapshots
    ])
    fc_pct = np.array([
        100.0 * int(s["num_unmasked_fc"]) / total_fc if total_fc else 0.0
        for s in snapshots
    ])
    return steps, exp_pct, fc_pct, total_exp, total_fc


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_run(run: dict[str, Any], out_path: Path, also_pdf: bool = False) -> None:
    meta = run["meta"]
    n_steps = int(meta["steps_effective"])
    gen_length = int(meta["gen_length"])
    unmask_steps_raw = run["token_unmasked_at_step"]
    unmask_steps: list[int | None] = list(unmask_steps_raw)

    # ---- per-position roles ------------------------------------------
    # Falls back gracefully on JSONs that predate role labels.
    raw_roles = run.get("token_role")
    if not raw_roles or len(raw_roles) != gen_length:
        roles = ["other"] * gen_length
    else:
        roles = list(raw_roles)

    role_idx: dict[str, list[int]] = {k: [] for k in ROLE_COLOR}
    for i, r in enumerate(roles):
        role_idx.setdefault(r, []).append(i)

    final_calls = run.get("final_function_calls", [])
    n_calls = len(final_calls)
    n_exp = sum(1 for r in roles if r == "explanation")
    n_fc = sum(1 for r in roles if r in ("fc_anchor", "fc_body"))

    # -------------------------------------------------------------------
    # Figure layout
    #   row 0: bar chart        |  text panel (rows 0-1)
    #   row 1: heatmap          |
    #   row 2: stabilisation curves (full width)
    # -------------------------------------------------------------------
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(
        3, 2,
        width_ratios=[3.4, 1.0],
        height_ratios=[1.0, 1.3, 0.85],
        hspace=0.36, wspace=0.06,
    )
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[1, 0], sharex=ax_bar)
    ax_text = fig.add_subplot(gs[0:2, 1])
    ax_text.axis("off")
    ax_curve = fig.add_subplot(gs[2, :])

    # ---- 1. per-token bar chart --------------------------------------
    xs = np.arange(gen_length)
    ys = np.array([s if s is not None else (n_steps + 1) for s in unmask_steps], dtype=float)
    never_unmasked = np.array([s is None for s in unmask_steps])

    bar_colors = [ROLE_COLOR.get(r, ROLE_COLOR["other"]) for r in roles]
    ax_bar.bar(xs, ys, width=1.0, color=bar_colors, edgecolor="none")
    if never_unmasked.any():
        ax_bar.bar(
            xs[never_unmasked], np.full(never_unmasked.sum(), n_steps + 1),
            width=1.0, color="#777777", alpha=0.55, edgecolor="none",
        )
    ax_bar.set_ylabel("step at which\ntoken was first unmasked", fontsize=10)
    ax_bar.set_ylim(0, n_steps + 2)
    ax_bar.set_xlim(-0.5, gen_length - 0.5)
    ax_bar.set_title(
        f"Diffusion denoising trajectory  \u00b7  remasking = {meta['remasking']}  \u00b7  "
        f"final FC blocks = {n_calls}  \u00b7  explanation tokens = {n_exp}  \u00b7  "
        f"FC tokens = {n_fc}",
        fontsize=11.5,
    )
    ax_bar.grid(axis="y", linestyle="--", alpha=0.3)
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=ROLE_COLOR["explanation"], label=ROLE_LABEL["explanation"]),
        mpatches.Patch(color=ROLE_COLOR["fc_anchor"],   label=ROLE_LABEL["fc_anchor"]),
        mpatches.Patch(color=ROLE_COLOR["fc_body"],     label=ROLE_LABEL["fc_body"]),
        mpatches.Patch(color=ROLE_COLOR["other"],       label=ROLE_LABEL["other"]),
    ]
    if never_unmasked.any():
        legend_handles.append(
            mpatches.Patch(color="#777777", alpha=0.6, label="never unmasked")
        )
    ax_bar.legend(handles=legend_handles, loc="upper right", fontsize=8.0,
                  ncol=2, frameon=True)

    # ---- 2. mask heatmap --------------------------------------------
    M = build_mask_matrix(unmask_steps, n_steps)  # (steps+1, gen_length), True == masked
    ax_heat.imshow(
        M, aspect="auto", cmap="Greys", origin="lower",
        interpolation="nearest", vmin=0, vmax=1,
    )
    # band highlights for explanation / FC columns (use contiguous runs)
    for s, e in contiguous_runs(role_idx.get("explanation", [])):
        ax_heat.axvspan(s - 0.5, e + 0.5, color=ROLE_COLOR["explanation"],
                        alpha=0.16, linewidth=0)
    for s, e in contiguous_runs(
        sorted(role_idx.get("fc_anchor", []) + role_idx.get("fc_body", []))
    ):
        ax_heat.axvspan(s - 0.5, e + 0.5, color=ROLE_COLOR["fc_anchor"],
                        alpha=0.16, linewidth=0)
    ax_heat.set_xlabel("token position in generated portion")
    ax_heat.set_ylabel("diffusion step")
    ax_heat.set_xlim(-0.5, gen_length - 0.5)
    ax_heat.set_ylim(0, n_steps + 1)
    ax_heat.spines["top"].set_visible(False)
    ax_heat.spines["right"].set_visible(False)

    # ---- 3. stabilisation curves -------------------------------------
    steps_arr, exp_pct, fc_pct, t_exp, t_fc = stabilisation_curves(run.get("snapshots", []))
    if steps_arr.size > 0:
        ax_curve.plot(
            steps_arr, exp_pct,
            color=ROLE_COLOR["explanation"], linewidth=2.0,
            marker="o", markersize=2.5, markevery=max(1, len(steps_arr) // 32),
            label=f"explanation  ({t_exp} tokens)",
        )
        ax_curve.plot(
            steps_arr, fc_pct,
            color=ROLE_COLOR["fc_anchor"], linewidth=2.0, linestyle="--",
            marker="s", markersize=2.5, markevery=max(1, len(steps_arr) // 32),
            label=f"function calls  ({t_fc} tokens)",
        )
        # crossover annotation
        ann = _crossover_annotation(steps_arr, exp_pct, fc_pct)
        if ann is not None:
            cx, cy, msg = ann
            ax_curve.annotate(
                msg,
                xy=(cx, cy),
                xytext=(cx + max(2, n_steps * 0.04), min(95, cy + 12)),
                fontsize=8.5,
                arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#bbb", lw=0.6),
            )
    else:
        ax_curve.text(
            0.5, 0.5,
            "no role-augmented snapshots in JSON\n(rerun visualize_diffusion.py)",
            ha="center", va="center", fontsize=10, color="#888",
            transform=ax_curve.transAxes,
        )
    ax_curve.set_xlabel("diffusion step")
    ax_curve.set_ylabel("% of role tokens revealed")
    ax_curve.set_xlim(0, n_steps)
    ax_curve.set_ylim(0, 102)
    ax_curve.grid(linestyle="--", alpha=0.35)
    ax_curve.spines["top"].set_visible(False)
    ax_curve.spines["right"].set_visible(False)
    if steps_arr.size > 0:
        ax_curve.legend(loc="lower right", fontsize=9, frameon=True)
    ax_curve.set_title(
        "Stabilisation curves: % of explanation vs FC tokens revealed at each step",
        fontsize=11,
    )

    # ---- 4. text snapshots side panel --------------------------------
    snaps = run.get("snapshots", [])
    if snaps:
        steps_to_show = pick_steps_for_panel(snaps, n_steps)
        text_lines: list[str] = []
        for snap_ in steps_to_show:
            decoded = snap_["decoded_clean"]
            shown = squeeze_text(decoded, max_chars=240)
            line_top = (
                f"step {snap_['step']:>3d}/{n_steps}   "
                f"({snap_['num_unmasked']}/{snap_['num_unmasked'] + snap_['num_masked']} unmasked)"
            )
            if "num_unmasked_explanation" in snap_:
                line_top += (
                    f"\n      exp {snap_['num_unmasked_explanation']:>3d}/"
                    f"{snap_['total_explanation']:<3d}   "
                    f"fc {snap_['num_unmasked_fc']:>3d}/{snap_['total_fc']:<3d}"
                )
            text_lines.append(line_top)
            text_lines.append("\u2500" * 30)
            text_lines.append(shown)
            text_lines.append("")
        ax_text.text(
            0.0, 1.0, "\n".join(text_lines),
            family="monospace", fontsize=7.4,
            transform=ax_text.transAxes, va="top", ha="left",
        )

    # ---- title -------------------------------------------------------
    fig.suptitle(
        f"LLaDA denoising  \u00b7  task = {meta['task_preset']}  \u00b7  "
        f"steps = {n_steps}  \u00b7  block_length = {meta['block_length']}",
        fontsize=13, y=0.995,
    )

    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    print(f"wrote {out_path}")
    if also_pdf:
        pdf_path = out_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"wrote {pdf_path}")
    plt.close(fig)


def _crossover_annotation(
    steps: np.ndarray, exp_pct: np.ndarray, fc_pct: np.ndarray,
) -> tuple[float, float, str] | None:
    """Find the step at which one role first reaches 50 % unmasked, and
    annotate which role got there first by how many steps."""
    if exp_pct.size == 0 or fc_pct.size == 0:
        return None
    half_exp_idx = int(np.argmax(exp_pct >= 50.0)) if (exp_pct >= 50.0).any() else None
    half_fc_idx = int(np.argmax(fc_pct >= 50.0)) if (fc_pct >= 50.0).any() else None
    if half_exp_idx is None or half_fc_idx is None:
        return None
    s_exp = int(steps[half_exp_idx])
    s_fc = int(steps[half_fc_idx])
    if s_exp < s_fc:
        winner = ("explanation", s_exp, exp_pct[half_exp_idx], s_fc - s_exp)
    elif s_fc < s_exp:
        winner = ("function calls", s_fc, fc_pct[half_fc_idx], s_exp - s_fc)
    else:
        return None
    name, x, y, delta = winner
    return float(x), float(y), f"{name} reach 50% first\n(by {delta} steps)"


def pick_steps_for_panel(
    snapshots: list[dict[str, Any]], n_steps: int,
) -> list[dict[str, Any]]:
    """Pick four snapshots: ~25%, ~50%, ~75%, 100% of the schedule."""
    if not snapshots:
        return []
    targets = [n_steps // 4, n_steps // 2, (3 * n_steps) // 4, n_steps]
    picked: list[dict[str, Any]] = []
    used: set[int] = set()
    for t in targets:
        # nearest available step
        best = min(snapshots, key=lambda s: abs(int(s["step"]) - t))
        if int(best["step"]) in used:
            continue
        used.add(int(best["step"]))
        picked.append(best)
    return picked


_WS_RE = re.compile(r"\s+")


def squeeze_text(text: str, max_chars: int = 240) -> str:
    """Compact whitespace + truncate.  Mask-replacement char `\u00b7` is preserved."""
    flat = _WS_RE.sub(" ", text).strip()
    if len(flat) > max_chars:
        flat = flat[: max_chars - 1] + "\u2026"
    return flat


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render denoising figures from a visualize_diffusion JSON.",
    )
    parser.add_argument("json_path", type=Path,
                        help="Path to JSON produced by visualize_diffusion.py")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output figure path (.png). Defaults to <json_path>.png")
    parser.add_argument("--pdf", action="store_true",
                        help="Also write a .pdf next to the .png")
    args = parser.parse_args()

    if not args.json_path.exists():
        parser.error(f"file not found: {args.json_path}")

    run = load_run(args.json_path)
    out_path = args.out or args.json_path.with_suffix(".png")
    plot_run(run, out_path, also_pdf=args.pdf)


if __name__ == "__main__":
    main()
