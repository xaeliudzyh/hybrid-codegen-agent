"""
Function call parser.

Parses function calls from generated text. The format can be extended
as needed for different models or use cases.
"""

import json
import re
from typing import Optional

from .types import FunctionCall


FUNCTION_CALL_PATTERN = re.compile(
    r'<function_call>\s*(\{.*?\})\s*</function_call>',
    re.DOTALL
)

INLINE_FUNCTION_PATTERN = re.compile(
    r'\[FUNCTION_CALL:\s*(\{.*?\})\s*\]',
    re.DOTALL
)


def parse_function_calls(text: str) -> list[FunctionCall]:
    """
    Parse function calls from generated text.
    
    Supports multiple formats:
    1. XML-style: <function_call>{"name": "...", "arguments": {...}}</function_call>
    2. Inline: [FUNCTION_CALL: {"name": "...", "arguments": {...}}]
    
    Args:
        text: Generated text that may contain function calls
        
    Returns:
        List of parsed FunctionCall objects
    """
    calls = []
    
    for match in FUNCTION_CALL_PATTERN.finditer(text):
        call = _parse_json_call(match.group(1), match.start(), match.end())
        if call:
            calls.append(call)
    
    for match in INLINE_FUNCTION_PATTERN.finditer(text):
        call = _parse_json_call(match.group(1), match.start(), match.end())
        if call:
            calls.append(call)
    
    return calls


def _parse_json_call(
    json_str: str,
    start_pos: int,
    end_pos: int,
) -> Optional[FunctionCall]:
    """Parse a JSON function call string."""
    try:
        data = json.loads(json_str)
        
        if not isinstance(data, dict):
            return None
        
        name = data.get("name")
        if not name or not isinstance(name, str):
            return None
        
        arguments = data.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        
        return FunctionCall(
            name=name,
            arguments=arguments,
            start_position=start_pos,
            end_position=end_pos,
        )
    except json.JSONDecodeError:
        return None


def extract_text_without_function_calls(text: str) -> str:
    """
    Remove function call markers from text, leaving only the regular content.
    
    Args:
        text: Text that may contain function calls
        
    Returns:
        Text with function call markers removed
    """
    result = FUNCTION_CALL_PATTERN.sub('', text)
    result = INLINE_FUNCTION_PATTERN.sub('', result)
    return result.strip()
