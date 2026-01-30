"""Tests for generative engines."""

import pytest
from engines import AutoregressiveEngine, GenerationResult


class TestAutoregressiveEngine:
    """Tests for the autoregressive engine."""
    
    def test_engine_properties(self):
        engine = AutoregressiveEngine(use_stub=True)
        
        assert engine.engine_type == "autoregressive"
        assert "llama" in engine.model_name.lower()
    
    def test_stub_generation(self):
        engine = AutoregressiveEngine(use_stub=True)
        
        result = engine.generate("Write a sort function")
        
        assert isinstance(result, GenerationResult)
        assert len(result.text) > 0
        assert result.generation_time_seconds >= 0
        assert "quicksort" in result.text.lower() or "def" in result.text
    
    def test_stub_with_function_call(self):
        engine = AutoregressiveEngine(use_stub=True)
        
        result = engine.generate("Write a fibonacci function")
        
        assert "fibonacci" in result.text.lower()
        assert "function_call" in result.text.lower()
    
    def test_generation_metadata(self):
        engine = AutoregressiveEngine(use_stub=True)
        
        result = engine.generate("test prompt")
        
        assert "engine" in result.metadata
        assert result.metadata["engine"] == "autoregressive"
