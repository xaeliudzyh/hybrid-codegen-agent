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
from function_calling.builtin_functions import create_default_registry


def main():
    parser = argparse.ArgumentParser(description="Run the code generation agent")
    parser.add_argument(
        "--task",
        type=str,
        default="Write a Python function to calculate the factorial of a number",
        help="The code generation task",
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
        default="meta-llama/Llama-2-7b-hf",
        help="Model name or path",
    )
    parser.add_argument(
        "--use-stub",
        action="store_true",
        default=True,
        help="Use stub implementation instead of real model",
    )
    
    args = parser.parse_args()
    
    if args.engine == "autoregressive":
        engine = AutoregressiveEngine(
            model_name_or_path=args.model,
            use_stub=args.use_stub,
        )
    else:
        print("Error: Diffusion engine is not yet implemented")
        print("Current stage focuses on autoregressive baseline.")
        sys.exit(1)
    
    registry = create_default_registry()
    
    agent = CodeGenAgent(
        engine=engine,
        function_registry=registry,
    )
    
    print(f"=" * 60)
    print(f"Code Generation Agent")
    print(f"Engine: {engine.engine_type} ({engine.model_name})")
    print(f"Task: {args.task}")
    print(f"=" * 60)
    print()
    
    result = agent.run(args.task)
    
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


if __name__ == "__main__":
    main()
