"""Tests for function calling subsystem."""

import pytest
from function_calling import FunctionCall, FunctionRegistry, parse_function_calls


class TestFunctionCallParser:
    """Tests for the function call parser."""
    
    def test_parse_xml_style_function_call(self):
        text = '''Here is some code:
```python
def hello():
    pass
```

<function_call>
{"name": "execute_code", "arguments": {"code": "print('hello')"}}
</function_call>
'''
        calls = parse_function_calls(text)
        
        assert len(calls) == 1
        assert calls[0].name == "execute_code"
        assert calls[0].arguments == {"code": "print('hello')"}
    
    def test_parse_inline_function_call(self):
        text = 'Let me run this: [FUNCTION_CALL: {"name": "test", "arguments": {"x": 1}}]'
        calls = parse_function_calls(text)
        
        assert len(calls) == 1
        assert calls[0].name == "test"
        assert calls[0].arguments == {"x": 1}
    
    def test_parse_multiple_function_calls(self):
        text = '''
<function_call>
{"name": "func1", "arguments": {"a": 1}}
</function_call>

Some text

<function_call>
{"name": "func2", "arguments": {"b": 2}}
</function_call>
'''
        calls = parse_function_calls(text)
        
        assert len(calls) == 2
        assert calls[0].name == "func1"
        assert calls[1].name == "func2"
    
    def test_parse_no_function_calls(self):
        text = "Just some regular text without any function calls."
        calls = parse_function_calls(text)
        
        assert len(calls) == 0
    
    def test_parse_invalid_json(self):
        text = '<function_call>not valid json</function_call>'
        calls = parse_function_calls(text)
        
        assert len(calls) == 0


class TestFunctionRegistry:
    """Tests for the function registry."""
    
    def test_register_and_execute(self):
        registry = FunctionRegistry()
        
        def add(a: int, b: int) -> int:
            return a + b
        
        registry.register(
            name="add",
            handler=add,
            description="Add two numbers",
            parameters={
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            }
        )
        
        call = FunctionCall(name="add", arguments={"a": 2, "b": 3})
        result = registry.execute(call)
        
        assert result.success
        assert result.result == 5
    
    def test_execute_unknown_function(self):
        registry = FunctionRegistry()
        
        call = FunctionCall(name="unknown", arguments={})
        result = registry.execute(call)
        
        assert not result.success
        assert "Unknown function" in result.error
    
    def test_execute_with_exception(self):
        registry = FunctionRegistry()
        
        def failing_func():
            raise ValueError("Test error")
        
        registry.register(
            name="fail",
            handler=failing_func,
            description="A function that fails",
        )
        
        call = FunctionCall(name="fail", arguments={})
        result = registry.execute(call)
        
        assert not result.success
        assert "Test error" in result.error
    
    def test_list_functions(self):
        registry = FunctionRegistry()
        
        registry.register(name="func1", handler=lambda: None, description="")
        registry.register(name="func2", handler=lambda: None, description="")
        
        names = registry.list_functions()
        
        assert set(names) == {"func1", "func2"}
    
    def test_get_functions_description(self):
        registry = FunctionRegistry()
        
        registry.register(
            name="greet",
            handler=lambda name: f"Hello, {name}",
            description="Greet someone",
            parameters={"name": {"type": "string"}},
        )
        
        desc = registry.get_functions_description()
        
        assert "greet" in desc
        assert "Greet someone" in desc
