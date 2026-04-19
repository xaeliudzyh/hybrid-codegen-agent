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
from diffusion import DiffusionEngine, EarlyFunctionDetector, SpeculativeExecutor, SpeculativeResult


@dataclass
class IterationMetrics:
    """Timing metrics for a single generation-execution iteration."""
    generation_start_time: float
    generation_end_time: float
    function_detection_time: Optional[float] = None
    function_execution_times: list[tuple[str, float, float]] = field(default_factory=list)
    # callback's functional
    speculative_detection_step: Optional[int] = None #on which step funciton_call was detected
    speculative_total_steps: Optional[int] = None #total diffusion steps
    speculative_execution_started: Optional[float] = None
    speculative_execution_finished: Optional[float] = None
    speculative_hit: Optional[bool] = None #did the speculative results ,matched with the final results(None = speculative wasn't used)
    speculative_time_saved: Optional[float] = None




    @property
    def generation_duration(self) -> float:
        return self.generation_end_time - self.generation_start_time

    @property
    def time_to_function_detection(self) -> Optional[float]:
        if self.function_detection_time is not None:
            return self.function_detection_time - self.generation_start_time
        return None


@dataclass
class AgentMetrics:
    """Accumulated timing metrics across all agent iterations."""
    iterations: list[IterationMetrics] = field(default_factory=list)

    @property
    def generation_start_time(self) -> Optional[float]:
        """Start time of the first iteration."""
        if self.iterations:
            return self.iterations[0].generation_start_time
        return None

    @property
    def generation_end_time(self) -> Optional[float]:
        """End time of the last iteration."""
        if self.iterations:
            return self.iterations[-1].generation_end_time
        return None

    @property
    def function_detection_time(self) -> Optional[float]:
        """Detection time of the first iteration that found function calls."""
        for it in self.iterations:
            if it.function_detection_time is not None:
                return it.function_detection_time
        return None

    @property
    def function_execution_times(self) -> list[tuple[str, float, float]]:
        """All function execution times across iterations."""
        result = []
        for it in self.iterations:
            result.extend(it.function_execution_times)
        return result

    @property
    def generation_duration(self) -> Optional[float]:
        """Total generation time across all iterations."""
        if not self.iterations:
            return None
        return sum(it.generation_duration for it in self.iterations)

    @property
    def time_to_function_detection(self) -> Optional[float]:
        """Time from first generation start to first function call detection."""
        if self.iterations and self.iterations[0].function_detection_time is not None:
            return self.iterations[0].function_detection_time - self.iterations[0].generation_start_time
        return None

    @property
    def num_iterations(self) -> int:
        return len(self.iterations)


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
    Agent for code generation with function calling support:
    - Accepts a GenerativeEngine
    - Builds prompts
    - Calls engine.generate()
    - Detects function calls
    - Executes them
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
        return """You are a Python code assistant with access to an execute_code tool.
You MUST use the execute_code function to run any code you write. NEVER output bare code without wrapping it in a function call.
Always put ALL code (imports, function definitions, calls, print statements) in a SINGLE execute_code call."""
    
    def _build_prompt(self, task: str) -> str:
        """Build the full prompt including system prompt, available functions, and task."""
        functions_description = self.function_registry.get_functions_description()
        
        prompt_parts = [
            self.system_prompt,
            "",
            "Available functions:",
            functions_description,
            "",
            "To call a function, use EXACTLY this format:",
            "<function_call>",
            '{"name": "execute_code", "arguments": {"code": "<your code here>"}}',
            "</function_call>",
            "",
            "Example:",
            "User: Complete the body of the following Python function.",
            "def factorial(n: int) -> int:",
            '    """Return factorial of n."""',
            "",
            "Assistant: I'll use recursion for this task: return 1 for the base case, otherwise multiply n by factorial(n-1).",
            "<function_call>",
            '{"name": "execute_code", "arguments": {"code": "    if n <= 1:\\n        return 1\\n    return n * factorial(n - 1)"}}',
            "</function_call>",
            "",
            "IMPORTANT:",
            "- Write 2-3 sentences explaining your approach BEFORE the function call.",
            "- Wrap your code inside <function_call> tags using the execute_code function.",
            "- Output ONLY the function body (indented implementation lines).",
            "- Do NOT include the function signature, docstring or test calls.",
            "",
            "Task:",
            task,
        ]
        return "\n".join(prompt_parts)
    
    def run(self, task: str, max_iterations: int = 5, max_tokens: int = 1024, temperature: float = None) -> AgentResult:
        """
        Execute the agent on a given task.
        Args:
            task: Task description
            max_iterations: Max number of generation-execution cycles
            max_tokens: Maximum number of tokens to generate per call
            temperature: Sampling temperature (None = use engine default)
        Returns:
            AgentResult with generated code, function calls, and their's metrics
        """
        metrics = AgentMetrics()
        all_function_calls: list[FunctionCall] = []
        all_function_results: list[tuple[FunctionCall, any]] = []
        
        prompt = self._build_prompt(task)
        final_output = ""
        
        for iteration in range(max_iterations):
            gen_start = time.perf_counter()
            gen_kwargs = {"max_tokens": max_tokens}
            if temperature is not None:
                gen_kwargs["temperature"] = temperature
            gen_result: GenerationResult = self.engine.generate(prompt, **gen_kwargs)
            gen_end = time.perf_counter()
            
            raw_output = gen_result.text
            final_output = raw_output
            
            function_calls = parse_function_calls(raw_output)
            detection_end = time.perf_counter()
            
            iter_metrics = IterationMetrics(
                generation_start_time=gen_start,
                generation_end_time=gen_end,
                function_detection_time=detection_end if function_calls else None,
            )
            
            if not function_calls:
                metrics.iterations.append(iter_metrics)
                break
            
            all_function_calls.extend(function_calls)
            
            iteration_results = []
            for fc in function_calls:
                exec_start = time.perf_counter()
                result = self.function_registry.execute(fc)
                exec_end = time.perf_counter()
                
                all_function_results.append((fc, result))
                iteration_results.append((fc, result))
                iter_metrics.function_execution_times.append((fc.name, exec_start, exec_end))
            
            metrics.iterations.append(iter_metrics)
            
            # if execute_code succeeded and returned output, task is likely done
            if self._should_stop_after_execution(iteration_results):
                break
            
            prompt = self._build_continuation_prompt(prompt, raw_output, all_function_results)
        
        return AgentResult(
            generated_code=self._extract_code(final_output),
            function_calls=all_function_calls,
            function_results=all_function_results,
            metrics=metrics,
            raw_output=final_output,
        )
    
    def if_fc_equal(self, spec_result: SpeculativeResult, function_call: FunctionCall) -> bool:
        if spec_result.function_call.name == function_call.name and spec_result.function_call.arguments == function_call.arguments:
            return True
        return False
        

    def run_with_speculative_execution(self, task: str, max_iterations: int = 5, max_tokens: int = 1024, require_valid_json = True, min_step_ratio = 0.1, check_interval=1, temperature: float = None) -> AgentResult:
        """
        Execute the agent on a given task, but with a speculative execution of early detected function call.
        Args:
            task: Task description
            max_iterations: Max number of generation-execution cycles
            max_tokens: Maximum number of tokens to generate per call
            require_valid_json: Whether the early detector should validate that the JSON
                inside <function_call> tags is syntactically correct and contains a "name" key.
                When True, partial or malformed JSON is ignored, reducing false positives
                at the cost of slightly later detection. When False, any text matching
                the <function_call>...</function_call> pattern triggers detection immediately.
            min_step_ratio: Fraction of total diffusion steps to skip before starting
                detection checks (0.0–1.0).
            check_interval: Run the detection check every N-th step of diffusion instead of
                every step.
        """
        if self.engine.engine_type != "diffusion":
            raise ValueError('run_with_speculative_execution is only available for the models with engine_type == "diffusion"')
        metrics = AgentMetrics()
        all_function_calls: list[FunctionCall] = []
        all_function_results: list[tuple[FunctionCall, any]] = []
        
        prompt = self._build_prompt(task)
        final_output = ""
        
        for iteration in range(max_iterations):
            gen_start = time.perf_counter()
            spec_kwargs = {}
            if temperature is not None:
                spec_kwargs["temperature"] = temperature
            gen_result, executor = self.engine.generate_with_speculative_execution(prompt = prompt, function_registry = self.function_registry, 
            max_tokens = max_tokens, min_step_ratio = min_step_ratio, require_valid_json = require_valid_json, check_interval = check_interval, **spec_kwargs)
            gen_end = time.perf_counter()
            
            raw_output = gen_result.text
            final_output = raw_output
            
            function_calls = parse_function_calls(raw_output)
            detection_end = time.perf_counter()
            
            good_detection = False
            iter_metrics = IterationMetrics(
                generation_start_time=gen_start,
                generation_end_time=gen_end,
                function_detection_time=detection_end if function_calls else None,
            )
            
            if not function_calls:
                metrics.iterations.append(iter_metrics)
                break
            first_fc = function_calls[0]
            spec_result = executor.get_result(timeout = 30)
            if spec_result is not None and function_calls:
                if self.if_fc_equal(spec_result, function_calls[0]):
                    first_fc = spec_result.function_call
                    good_detection = True
                else:
                    executor.cancel()
                iter_metrics.speculative_detection_step = spec_result.detection_event.step
                iter_metrics.speculative_total_steps = spec_result.detection_event.total_steps
                iter_metrics.speculative_execution_started = spec_result.execution_started_at
                iter_metrics.speculative_execution_finished = spec_result.execution_finished_at
                iter_metrics.speculative_hit = good_detection
                iter_metrics.speculative_time_saved = spec_result.time_saved if good_detection else 0.0
                    
            function_calls[0] = first_fc
            
            all_function_calls.extend(function_calls)
            iteration_results = []
            if good_detection:
                function_calls.pop(0)
                all_function_results.append((first_fc, spec_result.function_result))
                iteration_results.append((first_fc, spec_result.function_result))
                iter_metrics.function_execution_times.append((first_fc.name, spec_result.execution_started_at, spec_result.execution_finished_at))
            for fc in function_calls:
                exec_start = time.perf_counter()
                result = self.function_registry.execute(fc)
                exec_end = time.perf_counter()
                
                all_function_results.append((fc, result))
                iteration_results.append((fc, result))
                iter_metrics.function_execution_times.append((fc.name, exec_start, exec_end))
            
            metrics.iterations.append(iter_metrics)
            
            # if execute_code succeeded and returned output, task is likely done
            if self._should_stop_after_execution(iteration_results):
                break
            
            prompt = self._build_continuation_prompt(prompt, raw_output, all_function_results)
        
        return AgentResult(
            generated_code=self._extract_code(final_output),
            function_calls=all_function_calls,
            function_results=all_function_results,
            metrics=metrics,
            raw_output=final_output,
        )
    
    def _should_stop_after_execution(
        self,
        iteration_results: list[tuple[FunctionCall, any]],
    ) -> bool:
        """
        Deciding if we should stop after this iteration.
        Returns True if: execute_code was called and succeeded with output, all function calls in this iteration succeeded
        """
        if not iteration_results:
            return False
        
        for fc, result in iteration_results:
            if fc.name == "execute_code" and result.success:
                if isinstance(result.result, dict) and result.result.get("stdout"):
                    return True
        
        return False
    
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
        return f"{original_prompt}\n\nAssistant: {last_output}\n\nFunction results:\n{results_text}\n\nIf the task is complete, summarize the result. Otherwise, continue working on it."
    
    def _extract_code(self, output: str) -> str:
        """Extract code blocks from the output."""
        import re
        code_blocks = re.findall(r'```(?:\w+)?\n(.*?)```', output, re.DOTALL)
        if code_blocks:
            return "\n\n".join(code_blocks)
        return output
