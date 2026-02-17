"""
Function registry for explicit function execution.

The registry holds all available functions and executes them explicitly.
No implicit or automatic execution - all execution goes through this registry.
"""

import time
from typing import Any, Callable, Optional

from .types import FunctionCall, FunctionDefinition, FunctionResult


class FunctionRegistry:
    """
    Registry of functions available to the agent.
    
    All function execution is explicit and goes through this registry.
    """
    
    def __init__(self):
        self._functions: dict[str, FunctionDefinition] = {}
    
    def register(
        self,
        name: str,
        handler: Callable[..., Any],
        description: str,
        parameters: Optional[dict[str, dict]] = None,
    ) -> None:
        """
        Register a function.
        
        Args:
            name: Function name (must be unique)
            handler: The callable that implements the function
            description: Human-readable description
            parameters: JSON Schema-like parameter definitions
        """
        if name in self._functions:
            raise ValueError(f"Function '{name}' is already registered")
        
        self._functions[name] = FunctionDefinition(
            name=name,
            description=description,
            parameters=parameters or {},
            handler=handler,
        )
    
    def register_function(self, definition: FunctionDefinition) -> None:
        """Register a function from a FunctionDefinition object."""
        if definition.name in self._functions:
            raise ValueError(f"Function '{definition.name}' is already registered")
        self._functions[definition.name] = definition
    
    def get(self, name: str) -> Optional[FunctionDefinition]:
        """Get a function definition by name."""
        return self._functions.get(name)
    
    def list_functions(self) -> list[str]:
        """List all registered function names."""
        return list(self._functions.keys())
    
    def execute(self, call: FunctionCall) -> FunctionResult:
        """
        Execute a function call.
        
        Args:
            call: The FunctionCall to execute
            
        Returns:
            FunctionResult with execution outcome
        """
        func_def = self._functions.get(call.name)
        
        if func_def is None:
            return FunctionResult(
                call=call,
                success=False,
                result=None,
                error=f"Unknown function: {call.name}",
            )
        
        start_time = time.perf_counter()
        try:
            result = func_def.handler(**call.arguments)
            end_time = time.perf_counter()
            return FunctionResult(
                call=call,
                success=True,
                result=result,
                execution_time_seconds=end_time - start_time,
            )
        except Exception as e:
            end_time = time.perf_counter()
            return FunctionResult(
                call=call,
                success=False,
                result=None,
                error=str(e),
                execution_time_seconds=end_time - start_time,
            )
    
    def get_functions_description(self) -> str:
        """Get a formatted description of all available functions for prompts."""
        if not self._functions:
            return "No functions available."
        
        lines = []
        for func_def in self._functions.values():
            lines.append(func_def.format_for_prompt())
        return "\n".join(lines)
    
    def get_functions_schema(self) -> list[dict]:
        """Get JSON Schema representations of all functions."""
        return [func_def.get_schema() for func_def in self._functions.values()]
