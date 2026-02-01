"""
CodeGenAgent - main orchestration class for code generation.

The agent is NOT autonomous. All control flow is deterministic and implemented in code.
Language models are treated strictly as generative engines.
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from engines.base import GenerativeEngine, GenerationResult
from function_calling.registry import FunctionRegistry
from function_calling.parser import parse_function_calls
from function_calling.types import FunctionCall


@dataclass
class AgentMetrics:
    """Timing metrics for agent execution."""
    generation_start_time: Optional[float] = None
    generation_end_time: Optional[float] = None
    function_detection_time: Optional[float] = None
    function_execution_times: list[tuple[str, float, float]] = field(default_factory=list)
    
    @property
    def generation_duration(self) -> Optional[float]:
        if self.generation_start_time and self.generation_end_time:
            return self.generation_end_time - self.generation_start_time
        return None
    
    @property
    def time_to_function_detection(self) -> Optional[float]:
        if self.generation_start_time and self.function_detection_time:
            return self.function_detection_time - self.generation_start_time
        return None


@dataclass
class AgentResult:
    """Result of agent execution."""
    generated_code: str
    function_calls: list[FunctionCall]
    function_results: list[tuple[FunctionCall, any]]
    metrics: AgentMetrics
    raw_output: str


class CodeGenAgent:
    """
    Agent for code generation with function calling support.
    
    The agent:
    - Accepts a GenerativeEngine (chosen BEFORE execution)
    - Builds prompts
    - Calls engine.generate()
    - Detects function calls
    - Executes them via FunctionRegistry
    """
    
    def __init__(
        self,
        engine: GenerativeEngine,
        function_registry: FunctionRegistry,
        system_prompt: Optional[str] = None,
    ):
        self.engine = engine
        self.function_registry = function_registry
        self.system_prompt = system_prompt or self._default_system_prompt()
    
    def _default_system_prompt(self) -> str:
        return """You are a code generation assistant. 
You can generate code and call functions when needed.
Available functions will be provided in the prompt."""
    
    def _build_prompt(self, task: str) -> str:
        """Build the full prompt including system prompt, available functions, and task."""
        functions_description = self.function_registry.get_functions_description()
        
        prompt_parts = [
            self.system_prompt,
            "",
            "Available functions:",
            functions_description,
            "",
            # few-shot examples:
            "When you need to call a function, use this format:",
            "<function_call>",
            '{"name": "function_name", "arguments": {"arg": "value"}}',
            "</function_call>",
            "",
            "Example 1: Simple code execution",
            "User: Execute print('Hello, World!')",
            "Assistant: I'll execute this code.",
            "<function_call>",
            '{"name": "execute_code", "arguments": {"code": "print(\'Hello, World!\')"}}',
            "</function_call>",
            "",
            "Example 2: Writing and executing a function",
            "User: Write a function to calculate factorial of 5",
            "Assistant: I'll write the factorial function and execute it.",
            "```python",
            "def factorial(n):",
            "    if n <= 1:",
            "        return 1",
            "    return n * factorial(n-1)",
            "",
            "result = factorial(5)",
            "print(f'Factorial of 5 is {result}')",
            "```",
            "",
            "Now I'll execute this code:",
            "<function_call>",
            '{"name": "execute_code", "arguments": {"code": "def factorial(n):\\n    if n <= 1:\\n        return 1\\n    return n * factorial(n-1)\\n\\nresult = factorial(5)\\nprint(f\'Factorial of 5 is {result}\')"}}',
            "</function_call>",
            "",
            "Task:",
            task,
        ]
        return "\n".join(prompt_parts)
    
    def run(self, task: str, max_iterations: int = 5) -> AgentResult:
        """
        Execute the agent on a given task.
        
        Args:
            task: The code generation task description
            max_iterations: Maximum number of generation-execution cycles
            
        Returns:
            AgentResult with generated code, function calls, and metrics
        """
        metrics = AgentMetrics()
        all_function_calls: list[FunctionCall] = []
        all_function_results: list[tuple[FunctionCall, any]] = []
        
        prompt = self._build_prompt(task)
        final_output = ""
        
        for iteration in range(max_iterations):
            metrics.generation_start_time = time.perf_counter()
            result: GenerationResult = self.engine.generate(prompt)
            metrics.generation_end_time = time.perf_counter()
            
            raw_output = result.text
            final_output = raw_output
            
            detection_start = time.perf_counter()
            function_calls = parse_function_calls(raw_output)
            metrics.function_detection_time = detection_start
            
            if not function_calls:
                break
            
            all_function_calls.extend(function_calls)
            
            for fc in function_calls:
                exec_start = time.perf_counter()
                result = self.function_registry.execute(fc)
                exec_end = time.perf_counter()
                
                all_function_results.append((fc, result))
                metrics.function_execution_times.append((fc.name, exec_start, exec_end))
            
            prompt = self._build_continuation_prompt(prompt, raw_output, all_function_results)
        
        return AgentResult(
            generated_code=self._extract_code(final_output),
            function_calls=all_function_calls,
            function_results=all_function_results,
            metrics=metrics,
            raw_output=final_output,
        )
    
    def _build_continuation_prompt(
        self,
        original_prompt: str,
        last_output: str,
        function_results: list[tuple[FunctionCall, any]],
    ) -> str:
        """Build prompt for continuation after function execution."""
        results_text = "\n".join(
            f"Function {fc.name} returned: {result}"
            for fc, result in function_results[-1:]
        )
        return f"{original_prompt}\n\nAssistant: {last_output}\n\nFunction results:\n{results_text}\n\nContinue:"
    
    def _extract_code(self, output: str) -> str:
        """Extract code blocks from the output."""
        import re
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', output, re.DOTALL)
        if code_blocks:
            return "\n\n".join(code_blocks)
        return output
