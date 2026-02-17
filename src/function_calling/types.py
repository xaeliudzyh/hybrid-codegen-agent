"""
Type definitions for function calling.

Function calls are first-class structured objects, not raw text.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class FunctionCall:
    """
    Represents a parsed function call.
    
    This is a structured object, not raw text.
    """
    name: str
    arguments: dict[str, Any]
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    
    def __repr__(self) -> str:
        return f"FunctionCall(name={self.name!r}, arguments={self.arguments})"


@dataclass
class FunctionDefinition:
    """
    Definition of a function that can be called by the agent.
    """
    name: str
    description: str
    parameters: dict[str, dict]
    handler: Callable[..., Any]
    
    def get_schema(self) -> dict:
        """Return a JSON Schema-like description of the function."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
            },
        }
    
    def format_for_prompt(self) -> str:
        """Format function definition for inclusion in prompt."""
        params_str = ", ".join(
            f"{name}: {info.get('type', 'any')}"
            for name, info in self.parameters.items()
        )
        return f"- {self.name}({params_str}): {self.description}"


@dataclass
class FunctionResult:
    """Result of function execution."""
    call: FunctionCall
    success: bool
    result: Any
    error: Optional[str] = None
    execution_time_seconds: float = 0.0
