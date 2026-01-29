"""
Autoregressive engine implementation.

This serves as the baseline implementation using autoregressive models (e.g., LLaMA).
Currently implemented as a stub for testing the pipeline.
"""

import time
from typing import Optional

from .base import GenerativeEngine, GenerationResult


class AutoregressiveEngine(GenerativeEngine):
    """
    Autoregressive generative engine (LLaMA-based).
    
    This is the baseline implementation. Currently a stub that can be
    replaced with actual LLaMA integration.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Llama-2-7b-hf",
        device: str = "auto",
        use_stub: bool = True,
    ):
        """
        Initialize the autoregressive engine.
        
        Args:
            model_name_or_path: HuggingFace model identifier or local path
            device: Device to run on ('auto', 'cuda', 'cpu')
            use_stub: If True, use stub implementation instead of real model
        """
        self._model_name = model_name_or_path
        self._device = device
        self._use_stub = use_stub
        
        self._model = None
        self._tokenizer = None
        
        if not use_stub:
            self._load_model()
    
    def _load_model(self):
        """Load the actual model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_name,
            torch_dtype=torch.float16,
            device_map=self._device,
        )
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop_sequences: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate text using autoregressive decoding."""
        
        start_time = time.perf_counter()
        
        if self._use_stub:
            generated_text = self._stub_generate(prompt, max_tokens)
            tokens_generated = len(generated_text.split())  # Approximate
        else:
            generated_text, tokens_generated = self._real_generate(
                prompt, max_tokens, temperature, stop_sequences
            )
        
        end_time = time.perf_counter()
        
        return GenerationResult(
            text=generated_text,
            tokens_generated=tokens_generated,
            generation_time_seconds=end_time - start_time,
            metadata={"engine": "autoregressive", "model": self._model_name},
        )
    
    def _stub_generate(self, prompt: str, max_tokens: int) -> str:
        """Stub generation for testing the pipeline."""
        if "fibonacci" in prompt.lower():
            return '''Here's a Python function to calculate Fibonacci numbers:

```python
def fibonacci(n: int) -> int:
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
```

<function_call>
{"name": "execute_code", "arguments": {"code": "print(fibonacci(10))"}}
</function_call>
'''
        
        if "sort" in prompt.lower():
            return '''```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```'''
        
        return '''```python
def solution():
    pass
```'''
    
    def _real_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop_sequences: Optional[list[str]],
    ) -> tuple[str, int]:
        """Real generation using the loaded model."""
        import torch
        
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        input_len = inputs.input_ids.shape[1]
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        
        generated_tokens = outputs[0][input_len:]
        generated_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text[:generated_text.index(stop_seq)]
                    break
        
        return generated_text, len(generated_tokens)
    
    @property
    def engine_type(self) -> str:
        return "autoregressive"
    
    @property
    def model_name(self) -> str:
        return self._model_name
