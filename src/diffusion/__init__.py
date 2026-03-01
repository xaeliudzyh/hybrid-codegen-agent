""" Diffusion module - diffusion-specific logic. """
from .early_detector import EarlyFunctionDetector, DetectionEvent
from .speculative_executor import SpeculativeExecutor, SpeculativeResult
from .diffusion_engine import DiffusionEngine

__all__ = ["DiffusionEngine", "EarlyFunctionDetector", "DetectionEvent", "SpeculativeExecutor", "SpeculativeResult"]
