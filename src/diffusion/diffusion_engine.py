"""
Diffusion engine implementation using LLaDA.
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

from diffusion.speculative_executor import SpeculativeExecutor
from engines.base import GenerativeEngine, GenerationResult
from function_calling import FunctionRegistry
from diffusion.early_detector import EarlyFunctionDetector

LLADA_MASK_ID = 126336  # <|mdm_mask|> token


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Calculate how many tokens to unmask at each step.
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
    LLaDA uses masked diffusion: starts with all [MASK] tokens and
    iteratively reveals tokens based on model confidence.
    """
    
    _FC_ANCHOR_STRINGS: list[str] = [
        "<function_call>", "</function_call>",
        "{", "}", '"name"', '"arguments"',
    ]

    def __init__(
        self,
        model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct",
        device: str = "auto",
        steps: int = 128,
        block_length: int = 128,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",
        # new remask. strategies params
        fc_boost: float = 0.15,
        structural_boost: float = 0.15,
        structural_window: int = 2,
    ):
        """
        Args:
            model_name_or_path: HuggingFace model identifier
            device: Device to run on ('auto', 'cuda', 'cpu')
            steps: Number of diffusion denoising steps
            block_length: Block size for semi-autoregressive generation
            cfg_scale: Classifier-free guidance scale (0 = disabled)
            remasking: Strategy for remasking
                ('low_confidence', 'random', 'fc_priority', 'structural_boost')
            fc_boost: Additive confidence bonus for FC-anchor tokens (fc_priority)
            structural_boost: Additive proximity bonus near fixed FC clusters (structural_boost)
            structural_window: Half-window size for proximity detection (structural_boost)
        """
        self._model_name = model_name_or_path
        self._device = device
        self._model = None
        self._tokenizer = None
        self._steps = steps
        self._block_length = block_length
        self._cfg_scale = cfg_scale
        self._remasking = remasking
        self._fc_boost = fc_boost
        self._structural_boost_value = structural_boost
        self._structural_window = structural_window
        
        self._load_model()
        self._build_fc_anchor_ids()

    def _build_fc_anchor_ids(self):
        """Precompute unique token IDs that form FC-structural patterns."""
        ids: set[int] = set()
        for s in self._FC_ANCHOR_STRINGS:
            ids.update(self._tokenizer.encode(s, add_special_tokens=False))
        self._fc_anchor_ids = torch.tensor(sorted(ids), dtype=torch.long)

    @property
    def tokenizer(self):
        return self._tokenizer
    
    def _load_model(self):
        import warnings
        import logging
        warnings.filterwarnings("ignore", message=".*resume_download.*is deprecated.*")
        warnings.filterwarnings("ignore", message=".*Special tokens have been added.*")
        logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=True,
        )
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
        if not hasattr(self._model, 'all_tied_weights_keys'):
            self._model.all_tied_weights_keys = {}
        
        self._model = self._model.to(device).eval()
        self._device = device
    
    def _truncate_at_eos(self, generated_ids: torch.Tensor) -> torch.Tensor:
        """Truncate generated token ids at the first EOS token"""
        eos_id = self._tokenizer.eos_token_id
        if eos_id is None:
            return generated_ids
        eos_positions = (generated_ids == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            return generated_ids[:eos_positions[0]]
        return generated_ids

    def _format_prompt(self, prompt: str) -> str:
        task_marker = "\nTask:\n"
        idx = prompt.find(task_marker)
        if idx != -1:
            system = prompt[:idx].strip()
            user = prompt[idx + len(task_marker):].strip()
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ]
        else:
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
        Args:
            prompt_ids: tokenized prompt (batch, seq_len)
            attention_mask: attention mask for padding
            gen_length: number of tokens to generate
            temperature: sampling temperature (0 = greedy)
            steps: num of diffusion steps (defaults to self._steps)
            step_callback: ptional callback(x, step) called each step. Return True to stop early.
        
        Returns:
            x: full sequence including prompt and generated tokens
        """
        if steps is None:
            steps = self._steps
        device = self._model.device
        batch_size = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]

        x = torch.full(
            (batch_size, prompt_len + gen_length), 
            LLADA_MASK_ID, 
            dtype=torch.long, 
            device=device
        )
        x[:, :prompt_len] = prompt_ids.clone()
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, gen_length), dtype=attention_mask.dtype, device=device)
            ], dim=-1)
        
        prompt_index = (x != LLADA_MASK_ID)
        assert gen_length % self._block_length == 0, \
            f"gen_length ({gen_length}) must be divisible by block_length ({self._block_length})"
        num_blocks = gen_length // self._block_length
        assert steps % num_blocks == 0, \
            f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        steps_per_block = steps // num_blocks
        current_step = 0
        
        # Pre-compute FC anchor ids on device and conv kernel (avoid per-step allocation)
        if self._remasking in ('fc_priority', 'structural_boost'):
            anchor_ids = self._fc_anchor_ids.to(device)
        if self._remasking == 'structural_boost':
            w = self._structural_window
            kernel = torch.ones(1, 1, 2 * w + 1, device=device) / (2 * w + 1)
        
        for num_block in range(num_blocks):
            block_start = prompt_len + num_block * self._block_length
            block_end = prompt_len + (num_block + 1) * self._block_length
            block_mask_index = (x[:, block_start:block_end] == LLADA_MASK_ID)
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
            
            for i in range(steps_per_block):
                mask_index = (x == LLADA_MASK_ID)
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
                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)
                if self._remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                elif self._remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=device)
                elif self._remasking == 'entropy':
                    p = F.softmax(logits, dim=-1)
                    log_p = torch.log(p.clamp(min=1e-10))
                    x0_p = (p * log_p).sum(dim=-1)
                elif self._remasking == 'fc_priority':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                    is_fc = (x0.unsqueeze(-1) == anchor_ids).any(dim=-1)
                    x0_p = x0_p + self._fc_boost * is_fc.float()
                elif self._remasking == 'structural_boost':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                    is_fixed_fc = (
                        (x.unsqueeze(-1) == anchor_ids).any(dim=-1)
                        & ~mask_index
                    ).float()
                    proximity = F.conv1d(
                        is_fixed_fc.unsqueeze(1), kernel, padding=self._structural_window
                    ).squeeze(1)  # (B, L)
                    x0_p = x0_p + self._structural_boost_value * proximity
                else:
                    raise ValueError(f"Unknown remasking strategy: {self._remasking}")
                
                x0_p[:, block_end:] = -np.inf
                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)
                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=device)
                for j in range(confidence.shape[0]):
                    k = num_transfer_tokens[j, i].item()
                    if k > 0:
                        _, select_index = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_index] = True
                
                x[transfer_index] = x0[transfer_index]
                current_step += 1
                if step_callback is not None:
                    if step_callback(x, current_step):
                        return x
            
            # Early block termination: if the just-completed block is entirely EOS tokens, remaining blocks would be pure padding.
            eos_id = self._tokenizer.eos_token_id
            if eos_id is not None:
                block_tokens = x[0, block_start:block_end]
                if (block_tokens == eos_id).all():
                    break
        
        return x
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.3,
        stop_sequences: Optional[list[str]] = None,
    ) -> GenerationResult:
        """Generate text using diffusion-based decoding."""
        start_time = time.perf_counter()
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
        gen_length = max_tokens
        if gen_length % self._block_length != 0:
            gen_length = ((gen_length // self._block_length) + 1) * self._block_length
        num_blocks = gen_length // self._block_length
        effective_steps = self._steps
        if effective_steps % num_blocks != 0:
            effective_steps = ((effective_steps // num_blocks) + 1) * num_blocks
        output_ids = self._llada_generate(
            prompt_ids=input_ids,
            attention_mask=attention_mask,
            gen_length=gen_length,
            temperature=temperature,
            steps=effective_steps,
        )
        
        generated_ids = self._truncate_at_eos(output_ids[0, prompt_len:])
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
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

    def generate_with_speculative_execution(self,
        function_registry: FunctionRegistry,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        stop_sequences: Optional[list[str]] = None,
        min_step_ratio: float = 0.1,
        require_valid_json: bool = True,
        check_interval: int = 1,
    ) -> tuple[GenerationResult, SpeculativeExecutor]:
        """Generate text using diffusion-based decoding + execute detected function in the background"""
        start_time = time.perf_counter()
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
        gen_length = max_tokens
        if gen_length % self._block_length != 0:
            gen_length = ((gen_length // self._block_length) + 1) * self._block_length
        num_blocks = gen_length // self._block_length
        effective_steps = self._steps
        if effective_steps % num_blocks != 0:
            effective_steps = ((effective_steps // num_blocks) + 1) * num_blocks
        
        executor = SpeculativeExecutor(function_registry)
        detector = EarlyFunctionDetector(
            tokenizer=self._tokenizer,
            total_steps=effective_steps,
            prompt_len=prompt_len,
            min_step_ratio=min_step_ratio,
            require_valid_json=require_valid_json,
            check_interval=check_interval,
            on_detected=executor.on_function_detected,
        )
        
        output_ids = self._llada_generate(
            prompt_ids=input_ids,
            attention_mask=attention_mask,
            gen_length=gen_length,
            temperature=temperature,
            steps=effective_steps,
            step_callback=detector
        )
        generated_ids = self._truncate_at_eos(output_ids[0, prompt_len:])
        generated_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text[:generated_text.index(stop_seq)]
                    break
        
        end_time = time.perf_counter()
        
        return (GenerationResult(
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
                "detector": detector.get_metadata()
            },
        ), executor)
    
    @property
    def engine_type(self) -> str:
        return "diffusion"
    
    @property
    def model_name(self) -> str:
        return self._model_name
