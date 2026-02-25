"""
Diffusion engine implementation using LLaDA (Large Language Diffusion with mAsking).

This implements the DiffusionEngine using GSAI-ML/LLaDA-8B-Instruct.
Key feature: access to intermediate generation steps for early function call detection.

LLaDA generates all tokens in parallel and iteratively refines them through
multiple denoising steps, unlike autoregressive models that generate one token at a time.

Reference: https://github.com/ML-GSAI/LLaDA
"""

import time
from typing import Optional, Callable

import torch
import torch.nn.functional as F
import numpy as np

from engines.base import GenerativeEngine, GenerationResult


LLADA_MASK_ID = 126336  # <|mdm_mask|> token


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """
    Apply Gumbel noise for sampling from categorical distribution.
    
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves
    perplexity score but reduces generation quality. Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Calculate how many tokens to unmask at each step.
    
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Because LLaDA employs a linear noise schedule, the expected number of tokens 
    transitioned at each step should be consistent.
    
    This function precomputes the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    
    base = mask_num // steps
    remainder = mask_num % steps
    
    num_transfer_tokens = torch.zeros(
        mask_num.size(0), steps, 
        device=mask_index.device, 
        dtype=torch.int64
    ) + base
    
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1
    
    return num_transfer_tokens


class DiffusionEngine(GenerativeEngine):
    """
    Diffusion generative engine using LLaDA.
    
    This is the research implementation that provides access to
    intermediate generation steps for early function call detection.
    
    LLaDA uses masked diffusion: starts with all [MASK] tokens and
    iteratively reveals tokens based on model confidence.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct",
        device: str = "auto",
        # diffusion-specific parameters
        steps: int = 64,
        block_length: int = 32,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",
    ):
        """
        Initialize the diffusion engine.
        
        Args:
            model_name_or_path: HuggingFace model identifier
            device: Device to run on ('auto', 'cuda', 'cpu')
            steps: Number of diffusion denoising steps
            block_length: Block size for semi-autoregressive generation
            cfg_scale: Classifier-free guidance scale (0 = disabled)
            remasking: Strategy for remasking ('low_confidence' or 'random')
        """
        self._model_name = model_name_or_path
        self._device = device
        self._model = None
        self._tokenizer = None
        
        # diffusion parameters
        self._steps = steps
        self._block_length = block_length
        self._cfg_scale = cfg_scale
        self._remasking = remasking
        
        self._load_model()

    @property
    def tokenizer(self):
        return self._tokenizer
    
    def _load_model(self):
        """Load the LLaDA model and tokenizer."""
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=True,
        )
        
        # LLaDA requires left padding for correct generation
        if self._tokenizer.padding_side != 'left':
            self._tokenizer.padding_side = 'left'
        if self._device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self._device
        config = AutoConfig.from_pretrained(
            self._model_name,
            trust_remote_code=True,
        )
        
        model_class = config.__class__.model_type

        self._model = AutoModel.from_pretrained(
            self._model_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            #_fast_init=False,
        )
        
        # add missing attribute if not present (compatibility fix)
        if not hasattr(self._model, 'all_tied_weights_keys'):
            self._model.all_tied_weights_keys = {}
        
        self._model = self._model.to(device).eval()
        self._device = device
        assert self._tokenizer.pad_token_id != LLADA_MASK_ID, \
            "Padding ID equals mask ID - this will cause incorrect generation"
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for LLaDA-8B-Instruct (LLaMA-3 style)."""
        messages = [{"role": "user", "content": prompt}]
        return self._tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
    
    @torch.no_grad()
    def _llada_generate(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        gen_length: int,
        temperature: float,
        steps: Optional[int] = None,
        step_callback: Optional[Callable[[torch.Tensor, int], bool]] = None,
    ) -> torch.Tensor:
        """
        LLaDA diffusion generation.
        
        This is the core diffusion sampling loop adapted from LLaDA's official generate.py.
        
        Args:
            prompt_ids: Tokenized prompt (batch, seq_len)
            attention_mask: Attention mask for padding
            gen_length: Number of tokens to generate
            temperature: Sampling temperature (0 = greedy)
            steps: Number of diffusion steps (defaults to self._steps)
            step_callback: Optional callback(x, step) called each step.
                          Return True to stop early.
        
        Returns:
            Full sequence including prompt and generated tokens
        """
        if steps is None:
            steps = self._steps
        device = self._model.device
        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        
        # Initialize: prompt + [MASK] * gen_length
        x = torch.full(
            (batch_size, prompt_len + gen_length), 
            LLADA_MASK_ID, 
            dtype=torch.long, 
            device=device
        )
        x[:, :prompt_len] = prompt_ids.clone()
        
        # Extend attention mask if provided
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, gen_length), dtype=attention_mask.dtype, device=device)
            ], dim=-1)
        
        prompt_index = (x != LLADA_MASK_ID)
        
        # Validate block configuration
        assert gen_length % self._block_length == 0, \
            f"gen_length ({gen_length}) must be divisible by block_length ({self._block_length})"
        num_blocks = gen_length // self._block_length
        
        assert steps % num_blocks == 0, \
            f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        steps_per_block = steps // num_blocks
        
        current_step = 0
        
        for num_block in range(num_blocks):
            block_start = prompt_len + num_block * self._block_length
            block_end = prompt_len + (num_block + 1) * self._block_length
            
            # Get mask indices for this block
            block_mask_index = (x[:, block_start:block_end] == LLADA_MASK_ID)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
            
            for i in range(steps_per_block):
                mask_index = (x == LLADA_MASK_ID)
                
                # Forward pass with optional CFG
                if self._cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = LLADA_MASK_ID
                    x_ = torch.cat([x, un_x], dim=0)
                    if attention_mask is not None:
                        attn_ = torch.cat([attention_mask, attention_mask], dim=0)
                        logits = self._model(x_, attention_mask=attn_).logits
                    else:
                        logits = self._model(x_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (self._cfg_scale + 1) * (logits - un_logits)
                else:
                    if attention_mask is not None:
                        logits = self._model(x, attention_mask=attention_mask).logits
                    else:
                        logits = self._model(x).logits
                
                # Sample tokens with Gumbel noise
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                
                # Calculate confidence for remasking
                if self._remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                elif self._remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=device)
                else:
                    raise ValueError(f"Unknown remasking strategy: {self._remasking}")
                
                # Don't unmask tokens in future blocks yet
                x0_p[:, block_end:] = -np.inf
                
                # Keep prompt tokens unchanged
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                
                # Transfer top-k confident tokens
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
                for j in range(confidence.shape[0]):
                    k = num_transfer_tokens[j, i].item()
                    if k > 0:
                        _, select_index = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_index] = True
                
                x[transfer_index] = x0[transfer_index]
                
                current_step += 1
                
                # Optional callback for early stopping (e.g., function call detection)
                if step_callback is not None:
                    if step_callback(x, current_step):
                        return x
        
        return x
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
        stop_sequences: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate text using diffusion-based decoding."""
        start_time = time.perf_counter()
        
        # Format and tokenize prompt
        formatted_prompt = self._format_prompt(prompt)
        encoded = self._tokenizer(
            formatted_prompt,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt"
        )
        input_ids = encoded['input_ids'].to(self._device)
        attention_mask = encoded['attention_mask'].to(self._device)
        prompt_len = input_ids.shape[1]
        
        # Ensure gen_length is divisible by block_length
        gen_length = max_tokens
        if gen_length % self._block_length != 0:
            gen_length = ((gen_length // self._block_length) + 1) * self._block_length
        
        # Compute effective steps for this call without mutating self._steps
        num_blocks = gen_length // self._block_length
        effective_steps = self._steps
        if effective_steps % num_blocks != 0:
            effective_steps = ((effective_steps // num_blocks) + 1) * num_blocks
        
        # Run diffusion generation
        output_ids = self._llada_generate(
            prompt_ids=input_ids,
            attention_mask=attention_mask,
            gen_length=gen_length,
            temperature=temperature,
            steps=effective_steps,
        )
        
        # Decode generated tokens (skip prompt)
        generated_ids = output_ids[0, prompt_len:]
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Apply stop sequences
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text[:generated_text.index(stop_seq)]
                    break
        
        end_time = time.perf_counter()
        
        return GenerationResult(
            text=generated_text,
            tokens_generated=len(generated_ids),
            generation_time_seconds=end_time - start_time,
            metadata={
                "engine": "diffusion",
                "model": self._model_name,
                "steps_configured": self._steps,
                "steps_effective": effective_steps,
                "gen_length": gen_length,
                "block_length": self._block_length,
                "remasking": self._remasking,
            },
        )
    
    @property
    def engine_type(self) -> str:
        return "diffusion"
    
    @property
    def model_name(self) -> str:
        return self._model_name
