"""
SpeculativeExecutor - stores metrics for early detected functions, and stores its status.

"""
from dataclasses import dataclass, field
from typing import Optional



class SpeculativeExecutor:
    def __init__(self, function_registry: FunctionRegistry):
        ...

    def on_function_detected(self, event: DetectionEvent) -> None:
        """Callback азщ EarlyFunctionDetector. Starts a function's execution in background."""
        ...      ...

    def get_result(self, timeout: float = None) -> Optional[SpeculativeResult]:
        """
        Get result of speculative execution. Runs after end of generation. 
        Wait for result until timeout.
        """
        ...

    @property
    def is_running(self) -> bool:
        ...
    
    @property
    def is_done(self) -> bool:
        ...
    
    def cancel(self) -> None:
        """Stops speculative execution."""
        ...

@dataclass
class SpeculativeResult:
    def __init__(
        self, function_call: FunctionCall, function_result: FunctionResult,
         detection_event: DetectionEvent, execution_started_at: float,
     execution_finished_at: float, generation_finished_at: 
     float, was_ready_before_generation_end: bool
     ):
     ...

    @property
    def time_saved(self) -> float:
        """Time that we saved by using early function detectioning"""
        ...

