"""
Autoregressive engine implementation.

This serves as the baseline implementation using autoregressive models (e.g., LLaMA).
Currently implemented as a stub for testing the pipeline.
"""

import time
from typing import Optional

from .base import GenerativeEngine, GenerationResult


def format_prompt_llama2_chat(prompt: str) -> str:
    task_marker = "\nTask:\n"
    idx = prompt.find(task_marker)
    if idx != -1:
        system = prompt[:idx].strip()
        user = prompt[idx + len(task_marker):].strip()
        return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"
    return f"<s>[INST] {prompt} [/INST]"


def format_prompt_llama3_instruct(prompt: str) -> str:
    """Format prompt for LLaMA-3-Instruct models."""
    return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"


def format_prompt_qwen(prompt: str) -> str:
    """Format prompt for Qwen models."""
    return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def format_prompt_mistral(prompt: str) -> str:
    """Format prompt for Mistral-Instruct models."""
    return f"<s>[INST] {prompt} [/INST]"


def detect_prompt_format(model_name: str):
    """Detect the appropriate prompt formatter based on model name."""
    model_lower = model_name.lower()
    
    if "llama-2" in model_lower and "chat" in model_lower:
        return format_prompt_llama2_chat
    elif "llama-3" in model_lower and "instruct" in model_lower:
        return format_prompt_llama3_instruct
    elif "qwen" in model_lower:
        return format_prompt_qwen
    elif "mistral" in model_lower and "instruct" in model_lower:
        return format_prompt_mistral
    else:
        return None


class AutoregressiveEngine(GenerativeEngine):
    """
    Autoregressive generative engine (LLaMA-based).
    
    This is the baseline implementation. Currently a stub that can be
    replaced with actual LLaMA integration.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "unsloth/llama-2-7b-chat",
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
        self._prompt_formatter = detect_prompt_format(model_name_or_path)
        
        if not use_stub:
            self._load_model()
    
    def _load_model(self):
        """Load the actual model and tokenizer."""
        import warnings
        import logging
        warnings.filterwarnings("ignore", message=".*resume_download.*is deprecated.*")
        warnings.filterwarnings("ignore", message=".*Special tokens have been added.*")
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
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
            return '''I'll compute Fibonacci numbers using recursion with memoization.

<function_call>
{"name": "execute_code", "arguments": {"code": "def fibonacci(n, memo={}):\\n    if n <= 1:\\n        return n\\n    if n not in memo:\\n        memo[n] = fibonacci(n-1, memo) + fibonacci(n-2, memo)\\n    return memo[n]\\n\\nfor x in [10, 20, 30]:\\n    print(f'fibonacci({x}) = {fibonacci(x)}')"}}
</function_call>
'''
        
        if "sort" in prompt.lower():
            return '''I'll implement quicksort and test it.

<function_call>
{"name": "execute_code", "arguments": {"code": "def quicksort(arr):\\n    if len(arr) <= 1:\\n        return arr\\n    pivot = arr[len(arr) // 2]\\n    left = [x for x in arr if x < pivot]\\n    middle = [x for x in arr if x == pivot]\\n    right = [x for x in arr if x > pivot]\\n    return quicksort(left) + middle + quicksort(right)\\n\\nprint(quicksort([3,1,4,1,5,9,2,6]))"}}
</function_call>
'''
        
        return '''I'll write a solution for this task.

<function_call>
{"name": "execute_code", "arguments": {"code": "def solution():\\n    return 'done'\\n\\nprint(solution())"}}
</function_call>
'''
    
    def _real_generate(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float,
        stop_sequences: Optional[list[str]],
    ) -> tuple[str, int]:
        """Real generation using the loaded model."""
        import torch
        
        if self._prompt_formatter:
            formatted_prompt = self._prompt_formatter(prompt)
        else:
            formatted_prompt = prompt
        
        inputs = self._tokenizer(formatted_prompt, return_tensors="pt").to(self._model.device)
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
