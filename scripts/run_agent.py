#!/usr/bin/env python3
"""
Script to run the code generation agent.

This is a simple entry point for testing the end-to-end pipeline.
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent import CodeGenAgent
from engines import AutoregressiveEngine
from diffusion import DiffusionEngine
from function_calling.builtin_functions import create_default_registry


def load_task(task_arg: str) -> str:
    """Load task from file if path exists, otherwise return as text."""
    task_path = Path(task_arg)
    if task_path.exists() and task_path.is_file():
        return task_path.read_text(encoding="utf-8").strip()
    return task_arg


def main():
    parser = argparse.ArgumentParser(description="Run the code generation agent")
    parser.add_argument(
        "--task",
        type=str,
        default="Write a Python function to calculate the factorial of a number",
        help="The code generation task (text or path to .txt file)",
    )
    parser.add_argument(
        "--engine",
        type=str,
        choices=["autoregressive", "diffusion"],
        default="autoregressive",
        help="Which generative engine to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/llama-2-7b-chat",
        help="Model name or path",
    )
    parser.add_argument(
        "--with-early-detection",
        dest="with_early_detection",
        action="store_true",
        default=False,
        help="Enable speculative execution with early function detection for diffusion engine",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=5,
        help="Max number of generation-execution cycles. Default is 5",
    )
    parser.add_argument(
        "--no-require-valid-json",
        dest="require_valid_json",
        action="store_false",
        default=True,
        help="Disable JSON validation in early detector (accept any text matching <function_call> tags)",
    )
    parser.add_argument(
        "--min_step_ratio",
        type=float,
        default=0.1,
        help="Fraction of total diffusion steps to skip before starting detection checks (0.0–1.0). Default is 0.1",
    )
    parser.add_argument(
        "--check_interval",
        type=int,
        default=1,
        help="Run the detection check every N-th diffusion step instead of every step. Default is 1",
    )
    parser.add_argument(
        "--use-stub",
        dest="use_stub",
        action="store_true",
        help="Use stub implementation instead of real model",
    )
    parser.add_argument(
        "--no-use-stub",
        dest="use_stub",
        action="store_false",
        help="Do not use stub; use real model",
    )
    parser.set_defaults(use_stub=False)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run model on (e.g., 'cuda', 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=1024,
        help="Maximum number of tokens to generate per call. "
             "Both LLaDA and Llama-2 have 4096 context window; prompt + max_tokens must fit. "
             "Default is 1024",
    )
    # Diffusion engine parameters
    parser.add_argument(
        "--remasking",
        type=str,
        choices=["low_confidence", "random", "fc_priority", "structural_boost"],
        default="low_confidence",
        help="Remasking strategy for diffusion engine. Default is 'low_confidence'",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=128,
        help="Number of diffusion denoising steps. Default is 128",
    )
    parser.add_argument(
        "--block_length",
        type=int,
        default=32,
        help="Block size for semi-autoregressive generation. Default is 32",
    )
    parser.add_argument(
        "--fc_boost",
        type=float,
        default=0.2,
        help="Additive confidence bonus for FC-anchor tokens (fc_priority strategy). Default is 0.2",
    )
    parser.add_argument(
        "--structural_boost",
        type=float,
        default=0.3,
        help="Additive proximity bonus near fixed FC clusters (structural_boost strategy). Default is 0.3",
    )
    parser.add_argument(
        "--structural_window",
        type=int,
        default=5,
        help="Half-window size for proximity detection in structural_boost strategy. Default is 5",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature. Default: 0.7 for autoregressive, 0.3 for diffusion. "
             "Higher = more random, 0 = greedy (causes degeneration in LLaDA)",
    )
    
    args = parser.parse_args()
    
    if args.engine == "autoregressive":
        engine = AutoregressiveEngine(
            model_name_or_path=args.model,
            device=args.device,
            use_stub=args.use_stub,
        )
    else:
        engine = DiffusionEngine(
            model_name_or_path=args.model,
            device=args.device,
            steps=args.steps,
            block_length=args.block_length,
            remasking=args.remasking,
            fc_boost=args.fc_boost,
            structural_boost=args.structural_boost,
            structural_window=args.structural_window,
        )
    
    registry = create_default_registry()
    
    task = load_task(args.task)
    
    agent = CodeGenAgent(
        engine=engine,
        function_registry=registry,
    )
    
    print(f"=" * 60)
    print(f"Code Generation Agent")
    print(f"Engine: {engine.engine_type} ({engine.model_name})")
    if args.engine == "diffusion":
        print(f"Remasking: {args.remasking} | Steps: {args.steps} | Block: {args.block_length}")
        if args.remasking == "fc_priority":
            print(f"  fc_boost: {args.fc_boost}")
        elif args.remasking == "structural_boost":
            print(f"  structural_boost: {args.structural_boost} | window: {args.structural_window}")
    if args.temperature is not None:
        print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Task: {task[:100]}{'...' if len(task) > 100 else ''}")
    print(f"=" * 60)
    print()
    
    if args.engine == "diffusion" and args.with_early_detection == True:
        result = agent.run_with_speculative_execution(
            task, 
            max_iterations=args.max_iterations,
            max_tokens=args.max_tokens,
            require_valid_json=args.require_valid_json,
            min_step_ratio=args.min_step_ratio,
            check_interval=args.check_interval,
            temperature=args.temperature,
            )
    else:
        result = agent.run(task, max_iterations=args.max_iterations, max_tokens=args.max_tokens, temperature=args.temperature)
    
    print("Generated Code:")
    print("-" * 40)
    print(result.generated_code)
    print("-" * 40)
    print()
    
    if result.function_calls:
        print(f"Function Calls ({len(result.function_calls)}):")
        for call in result.function_calls:
            print(f"  - {call}")
        print()
        
        print("Function Results:")
        for call, res in result.function_results:
            print(f"  - {call.name}: {res}")
        print()
    
    print("Metrics:")
    print(f"  - Generation time: {result.metrics.generation_duration:.4f}s")
    if result.metrics.time_to_function_detection:
        print(f"  - Time to function detection: {result.metrics.time_to_function_detection:.4f}s")
    for name, start, end in result.metrics.function_execution_times:
        print(f"  - Function '{name}' execution: {end - start:.4f}s")
    
    for i, it in enumerate(result.metrics.iterations):
        if it.speculative_hit is not None:
            print()
            print(f"Speculative Execution (iteration {i + 1}):")
            print(f"  - Detection step: {it.speculative_detection_step} / {it.speculative_total_steps}")
            if it.speculative_total_steps and it.speculative_detection_step:
                pct = (1 - it.speculative_detection_step / it.speculative_total_steps) * 100
                print(f"  - Steps saved: {it.speculative_total_steps - it.speculative_detection_step} ({pct:.1f}%)")
            print(f"  - Result: {'HIT' if it.speculative_hit else 'MISS'}")
            print(f"  - Time saved: {it.speculative_time_saved:.4f}s" if it.speculative_time_saved else "  - Time saved: 0.0000s")
        elif args.with_early_detection:
            print()
            print(f"Speculative Execution (iteration {i + 1}): detector did not fire")


if __name__ == "__main__":
    main()
