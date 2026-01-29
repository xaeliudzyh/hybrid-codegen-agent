"""
Diffusion engine implementation (placeholder).

This will implement the DiffusionEngine using LLaDA.
Key feature: access to intermediate generation steps for early function call detection.

NOTE: This is a placeholder for future implementation.
Current stage focuses on autoregressive baseline.
"""

from typing import Optional

from engines.base import GenerativeEngine, GenerationResult


class DiffusionEngine(GenerativeEngine):
    """
    Diffusion generative engine (LLaDA-based).
    
    This is the research implementation that provides access to
    intermediate generation steps for early function call detection.
    
    NOT IMPLEMENTED YET - placeholder for future development.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct",
        device: str = "auto",
    ):
        self._model_name = model_name_or_path
        self._device = device
        raise NotImplementedError(
            "DiffusionEngine is not yet implemented. "
            "Current stage focuses on autoregressive baseline."
        )
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop_sequences: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate text using diffusion-based decoding."""
        raise NotImplementedError()
    
    @property
    def engine_type(self) -> str:
        return "diffusion"
    
    @property
    def model_name(self) -> str:
        return self._model_name
