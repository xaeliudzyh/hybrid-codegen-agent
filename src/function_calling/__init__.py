"""Function calling subsystem - parsing, representation, execution."""

from .types import FunctionCall, FunctionDefinition
from .registry import FunctionRegistry
from .parser import parse_function_calls

__all__ = ["FunctionResult","FunctionCall", "FunctionDefinition", "FunctionRegistry", "parse_function_calls"]
