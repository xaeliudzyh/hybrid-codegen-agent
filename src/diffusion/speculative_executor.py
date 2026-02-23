"""
SpeculativeExecutor - stores metrics for early detected functions, and stores its status.

"""
from dataclasses import dataclass, field
from typing import Optional
from function_calling import FunctionRegistry, FunctionCall, FunctionResult, parse_function_calls, execute_code
from diffusion import DetectionEvent
import time
from concurrent.futures import ThreadPoolExecutor, Future


@dataclass
class SpeculativeResult:
    def __init__(
        self, function_call: FunctionCall, function_result: FunctionResult,
         detection_event: DetectionEvent, execution_started_at: float,
     execution_finished_at: float,  was_ready_before_generation_end: bool, 
     generation_finished_at: Optional[float] = None,
     ):
     self._function_call = function_call
     self._function_result = function_result
     self._detection_event = detection_event
     self._execution_started_at = execution_started_at
     self._execution_finished_at = execution_finished_at
     self._generation_finished_at = generation_finished_at
     self._was_ready_before_generation_end = was_ready_before_generation_end

    @property
    def time_saved(self) -> float:
        """Time that we saved by using early function detectioning"""
        if self._generation_finished_at == None:
            return 0.0
        ex_time = self._execution_finished_at - self._execution_started_at
        if self._was_ready_before_generation_end == True:
            return ex_time
        return max(0.0, self._generation_finished_at - self._execution_finished_at)
        
class SpeculativeExecutor:
    def __init__(self, function_registry: FunctionRegistry):
        self._executor = ThreadPoolExecutor(max_workers = 1)
        self._function_registry = function_registry
        self._future : Optional[Future] = None
        self._detection_event: Optional[DetectionEvent] = None
        
        self._execution_started_at: Optional[float] = None
        self._execution_finished_at: Optional[float] = None
        
        self._function_call: Optional[FunctionCall] = None

    def _execute(self, fun_call) -> FunctionResult:
        result = self._registry.execute(fun_call)
        self._execution_finished_at = time.perf_counter()
        return result

    def on_function_detected(self, event: DetectionEvent) -> None:
        """Callback for EarlyFunctionDetector, it starts a function's execution in the background."""
        if self._future is not None:
            return
        fun_call_raw = event.decoded_text
        fun_call = parse_function_calls(fun_call_raw)[0]
        self._detection_event = event
        self._function_call = fun_call
        self._execution_started_at = time.perf_counter()
        self._future = self._executor.submit(self._execute, fun_call)
        
        return

    def get_result(self, timeout: float = None) -> Optional[SpeculativeResult]:
        """
        Get result of speculative execution. Runs after end of generation. 
        Wait for result until timeout.
        """
        if self._future is None:
            return None
        generation_finished_at = time.perf_counter()
        if self._future.done():
            res = self._future.result()
            was_ready_before_gen_end = True
        else:
            try:
                res = self._future.result(timeout=timeout)
                was_ready_before_gen_end = True
            except TimeoutError:
                return None
        res = SpeculativeResult(function_call=self._function_call, 
                                function_result=res,
                                detection_event=self._detection_event,
                                execution_started_at=self._execution_started_at,
                                execution_finished_at=self._execution_finished_at,
                                was_ready_before_generation_end=was_ready_before_gen_end,
                                generation_finished_at=generation_finished_at,
                            )
        
        return res   
        
            

    @property
    def is_running(self) -> bool:
        return ((self._future is not None) and (not self._future.done()))
    
    @property
    def is_done(self) -> bool:
        return (self._future is not None) and (self._future.done())
    
    def cancel(self) -> None:
        """Stops speculative execution."""
        self._future.cancel()
        self._executor.shutdown(wait=False)
        return 

