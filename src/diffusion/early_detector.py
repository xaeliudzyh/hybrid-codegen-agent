"""
EarlyFunctionDetector - early function call detection in diffusion intermediate states.
During LLaDA's denoising process, we want to inspect partially-generated sequences at each step. If a valid
function call pattern appears before generation completes, we notice it.

Usage as step_callback for DiffusionEngine._llada_generate:
    detector = EarlyFunctionDetector(tokenizer, total_steps=64)
    output = engine._llada_generate(
        ...,
        step_callback=detector,
    )
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional
from collections.abc import Callable

import torch


LLADA_MASK_ID = 126336  # <|mdm_mask|> token

_FC_PATTERN = re.compile(
    r'<function_call>\s*(\{.*?\})\s*</function_call>',
    re.DOTALL,
)


@dataclass
class DetectionEvent:
    """Record of a single detection event."""
    step: int
    total_steps: int
    decoded_text: str
    function_call_json: Optional[dict] = None

    @property
    def steps_saved(self) -> int:
        return self.total_steps - self.step

    @property
    def savings_percent(self) -> float:
        if self.total_steps > 0:
            return (self.steps_saved / self.total_steps) * 100
        return 0.0


class EarlyFunctionDetector:
    """
    Detect function calls in intermediate diffusion states.

    Implements the step_callback(x, step) -> bool protocol
    expected by DiffusionEngine._llada_generate.

    The detector decodes the current (partially-masked) sequence,
    searches for <function_call>...</function_call> patterns
    and optionally validates the JSON inside.

    Args:
        tokenizer: HuggingFace tokenizer (needs pad_token_id and decode).
        total_steps: Total number of diffusion steps (needed for ratio check).
        min_step_ratio: Skip the first N% of steps (too noisy to decode).
            Default 0.1 = ignore first 10%.
        require_valid_json: If True, the JSON inside the tags must parse
            and contain a "name key.  Reduces false positives.
        check_interval: Check every N-th step instead of every step.
            Default 1 = check every step.  Larger values trade detection
            latency for less decoding overhead.
    """

    def __init__(
        self,
        tokenizer,
        total_steps: int,
        *,
        prompt_len: int = 0,
        min_step_ratio: float = 0.1,
        require_valid_json: bool = True,
        check_interval: int = 1,
        on_detected: Callable[[DetectionEvent], None] = None
    ):
        if total_steps <= 0:
            raise ValueError(f"total_steps must be positive, got {total_steps}")
        if not 0.0 <= min_step_ratio < 1.0:
            raise ValueError(f"min_step_ratio must be in [0, 1), got {min_step_ratio}")
        if check_interval < 1:
            raise ValueError(f"check_interval must be >= 1, got {check_interval}")

        self._tokenizer = tokenizer
        self._total_steps = total_steps
        self._prompt_len = prompt_len
        self._min_step_ratio = min_step_ratio
        self._require_valid_json = require_valid_json
        self._check_interval = check_interval

        # State
        self._detection_event: Optional[DetectionEvent] = None
        self._steps_checked: int = 0
        self._steps_skipped: int = 0
        self._on_detected = on_detected

        # Precompute visible text of MASK token for FC completeness checks
        self._mask_token_text = tokenizer.decode([LLADA_MASK_ID], skip_special_tokens=False)

    def __call__(self, x: torch.Tensor, step: int) -> bool:
        """
        Called by _llada_generate on each diffusion step.

        Args:
            x: Current sequence tensor (batch, seq_len), may contain MASK tokens.
            step: Current step number).

        Returns:
            True  → stop generation early (function call detected).
            False → continue generation.
        """
        if self._detection_event is not None:
            return False
        min_step = int(self._total_steps * self._min_step_ratio)
        if step <= min_step:
            self._steps_skipped += 1
            return False

        # Check interval - skip intermediate steps for performance
        if self._check_interval > 1 and step % self._check_interval != 0:
            self._steps_skipped += 1
            return False

        self._steps_checked += 1
        text = self._decode_partial(x)
        match = _FC_PATTERN.search(text)
        if match is None:
            return False
        # Skip if MASK tokens remain inside the FC span (incomplete FC)
        if self._mask_token_text in match.group(0):
            return False
        fc_json = None
        if self._require_valid_json:
            fc_json = self._try_parse_function_call(match.group(1))
            if fc_json is None:
                return False

        self._detection_event = DetectionEvent(
            step=step,
            total_steps=self._total_steps,
            decoded_text=text,
            function_call_json=fc_json,
        )
        if self._on_detected is not None:
            self._on_detected(self._detection_event)
        return False

    @property
    def detected(self) -> bool:
        """Whether a function call was detected."""
        return self._detection_event is not None

    @property
    def detection_event(self) -> Optional[DetectionEvent]:
        return self._detection_event

    @property
    def detection_step(self) -> Optional[int]:
        """Step at which the function call was detected, or None."""
        if self._detection_event is not None:
            return self._detection_event.step
        return None

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def steps_saved(self) -> int:
        """Number of diffusion steps saved by early stopping."""
        if self._detection_event is not None:
            return self._detection_event.steps_saved
        return 0

    @property
    def savings_percent(self) -> float:
        """Percentage of steps saved (0–100)."""
        if self._detection_event is not None:
            return self._detection_event.savings_percent
        return 0.0

    @property
    def steps_checked(self) -> int:
        """How many steps actually ran the decode+search logic."""
        return self._steps_checked

    @property
    def steps_skipped(self) -> int:
        """Steps skipped due to min_step_ratio or check_interval."""
        return self._steps_skipped

    # Metadata for GenerationResult
    def get_metadata(self) -> dict:
        """Return a dict suitable for merging into GenerationResult.metadata."""
        return {
            "early_detection": {
                "enabled": True,
                "detected": self.detected,
                "detection_step": self.detection_step,
                "total_steps": self._total_steps,
                "steps_saved": self.steps_saved,
                "savings_percent": round(self.savings_percent, 2),
                "steps_checked": self._steps_checked,
                "steps_skipped": self._steps_skipped,
                "min_step_ratio": self._min_step_ratio,
                "require_valid_json": self._require_valid_json,
                "check_interval": self._check_interval,
            }
        }

    # Internals
    def _decode_partial(self, x: torch.Tensor) -> str:
        """
        Decode only the generated portion of the tensor (skip prompt tokens).
        MASK tokens are decoded visibly (not stripped) so that the caller
        can verify FC completeness by checking for mask text in the output.
        """
        gen_tokens = x[0, self._prompt_len:].clone()
        return self._tokenizer.decode(gen_tokens, skip_special_tokens=False)

    @staticmethod
    def _try_parse_function_call(json_str: str) -> Optional[dict]:
        """
        Try to parse JSON and validate it has a "name" key.

        Returns the parsed dict on success, None on failure.
        Mirrors the validation logic of function_calling.parser.
        """
        try:
            data = json.loads(json_str)
            if isinstance(data, dict) and isinstance(data.get("name"), str):
                return data
            return None
        except (json.JSONDecodeError, TypeError):
            return None
