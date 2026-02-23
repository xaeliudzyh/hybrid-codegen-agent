"""
Built-in functions for the code generation agent.

These are the functions available to the agent.
"""

import sys
import io
from typing import Any

from .registry import FunctionRegistry


def execute_code(code: str) -> dict[str, Any]:
    """
    Execute Python code and return the result.
    
    Args:
        code: Python code to execute
        
    Returns:
        Dictionary with stdout, stderr, and any exception info
    """
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    result = {
        "stdout": "",
        "stderr": "",
        "exception": None,
        "success": True,
    }
    
    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        
        namespace = {"__builtins__": __builtins__}
        exec(code, namespace)
        
        result["stdout"] = stdout_capture.getvalue()
        result["stderr"] = stderr_capture.getvalue()
        
    except Exception as e:
        result["exception"] = f"{type(e).__name__}: {str(e)}"
        result["success"] = False
        result["stderr"] = stderr_capture.getvalue()
        
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    
    return result


def search_documentation(query: str) -> str:
    """
    Search documentation for a given query.
    This is a stub that can be replaced with actual documentation search.
    
    Args:
        query: Search query
        
    Returns:
        Documentation text
    """
    return f"Documentation search results for: {query}\n(No results - stub implementation)"


def get_function_signature(function_name: str) -> str:
    """
    Get the signature of a Python built-in or standard library function.
    Args:
        function_name: Name of the function
        
    Returns:
        Function signature as a string
    """
    import inspect
    
    try:
        if hasattr(__builtins__, function_name):
            obj = getattr(__builtins__, function_name)
            if callable(obj):
                try:
                    sig = inspect.signature(obj)
                    return f"{function_name}{sig}"
                except (ValueError, TypeError):
                    return f"{function_name}(...)"
        
        return f"Function '{function_name}' not found"
    except Exception as e:
        return f"Error getting signature: {e}"


def create_default_registry() -> FunctionRegistry:
    """
    Create a FunctionRegistry with default built-in functions.
    
    Returns:
        FunctionRegistry with built-in functions registered
    """
    registry = FunctionRegistry()
    
    registry.register(
        name="execute_code",
        handler=execute_code,
        description="Execute Python code and return stdout/stderr",
        parameters={
            "code": {
                "type": "string",
                "description": "Python code to execute",
            }
        },
    )
    
    registry.register(
        name="search_documentation",
        handler=search_documentation,
        description="Search documentation for a given query",
        parameters={
            "query": {
                "type": "string",
                "description": "Search query",
            }
        },
    )
    
    registry.register(
        name="get_function_signature",
        handler=get_function_signature,
        description="Get the signature of a Python function",
        parameters={
            "function_name": {
                "type": "string",
                "description": "Name of the function",
            }
        },
    )
    
    return registry
