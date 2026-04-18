#!/usr/bin/env python3
"""CLI for HumanEval benchmark."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from engines import AutoregressiveEngine
from diffusion import DiffusionEngine
from agent import CodeGenAgent
from function_calling.builtin_functions import create_default_registry
from experiments.humaneval_runner import (
    load_humaneval,
    run_engine_mode,
    run_agent_mode,
    save_results,
    print_summary,
    AGENT_SYSTEM_PROMPT,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HumanEval benchmark")

    p.add_argument("--data", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--mode", choices=["engine", "agent"], default="engine")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--engine", choices=["autoregressive", "diffusion"], default="autoregressive")
    p.add_argument("--model", type=str, default="unsloth/llama-2-7b-chat")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--use-stub", dest="use_stub", action="store_true")
    p.add_argument("--max_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=None)
    
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--block_length", type=int, default=128)
    p.add_argument("--remasking", type=str, default="low_confidence",
                   choices=["low_confidence", "random", "entropy", "fc_priority", "structural_boost"])
    p.add_argument("--fc_boost", type=float, default=0.15)
    p.add_argument("--structural_boost", type=float, default=0.15)
    p.add_argument("--structural_window", type=int, default=2)
    p.add_argument("--max_iterations", type=int, default=1)
    p.add_argument("--speculative", action="store_true")
    p.add_argument("--min_step_ratio", type=float, default=0.1)
    p.add_argument("--check_interval", type=int, default=1)

    p.add_argument("--output", "-o", type=str, default=None)

    return p


def main():
    args = build_parser().parse_args()

    print("Loading HumanEval dataset...")
    tasks = load_humaneval(args.data)
    if args.limit:
        tasks = tasks[:args.limit]
    print(f"Loaded {len(tasks)} tasks")

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
    print(f"\nEngine: {engine.engine_type} ({engine.model_name})")
    if args.engine == "diffusion":
        print(f"Remasking: {args.remasking} | Steps: {args.steps}")
    print(f"Mode: {args.mode} | k={args.k}")
    print()

    if args.mode == "engine":
        temp = args.temperature if args.temperature is not None else 0.2
        bench = run_engine_mode(
            engine=engine,
            tasks=tasks,
            k=args.k,
            max_tokens=args.max_tokens,
            temperature=temp,
        )
    else:
        registry = create_default_registry()
        agent = CodeGenAgent(engine=engine, function_registry=registry,
                             system_prompt=AGENT_SYSTEM_PROMPT)
        bench = run_agent_mode(
            agent=agent,
            tasks=tasks,
            k=args.k,
            max_tokens=args.max_tokens,
            max_iterations=args.max_iterations,
            temperature=args.temperature,
            speculative=args.speculative,
            min_step_ratio=args.min_step_ratio,
            check_interval=args.check_interval,
        )

    print_summary(bench)

    if args.output:
        save_results(bench, args.output)
    else:
        suffix = f"{args.engine}_{args.remasking}_{args.mode}_k{args.k}"
        default_path = str(Path(__file__).parent / "results" / f"humaneval_{suffix}.json")
        save_results(bench, default_path)


if __name__ == "__main__":
    main()
