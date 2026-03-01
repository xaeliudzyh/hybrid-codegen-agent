"""Function calling subsystem - parsing, representation, execution."""

from .types import FunctionCall, FunctionDefinition, FunctionResult
from .registry import FunctionRegistry
from .parser import parse_function_calls
from .builtin_functions import execute_code

__all__ = ["FunctionResult", "FunctionCall", "FunctionDefinition", "FunctionRegistry", "parse_function_calls", "execute_code"]
