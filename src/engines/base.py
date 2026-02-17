"""
Base abstractions for generative engines.

GenerativeEngine is the abstract interface that all engines must implement.
Models are treated strictly as generative engines - they don't make decisions
about tools or control flow.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationResult:
    """Result of text generation."""
    text: str
    tokens_generated: int
    generation_time_seconds: float
    metadata: dict = field(default_factory=dict)


class GenerativeEngine(ABC):
    """
    Abstract interface for text generation.
    
    All generative engines (autoregressive, diffusion) must implement this interface.
    The engine is responsible ONLY for text generation - no decision making.
    """
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop_sequences: Optional[list[str]] = None,
    ) -> GenerationResult:
        """
        Generate text given a prompt.
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            stop_sequences: Optional list of sequences that stop generation
            
        Returns:
            GenerationResult with generated text and metadata
        """
        pass
    
    @property
    @abstractmethod
    def engine_type(self) -> str:
        """Return the type of engine (e.g., 'autoregressive', 'diffusion')."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name/identifier of the underlying model."""
        pass
