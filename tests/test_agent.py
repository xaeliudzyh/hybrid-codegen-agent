"""Tests for the CodeGenAgent."""

import pytest
from agent import CodeGenAgent
from engines import AutoregressiveEngine
from function_calling import FunctionRegistry, FunctionCall


class TestCodeGenAgent:
    """Tests for the code generation agent."""
    
    def test_agent_creation(self):
        engine = AutoregressiveEngine(use_stub=True)
        registry = FunctionRegistry()
        
        agent = CodeGenAgent(engine=engine, function_registry=registry)
        
        assert agent.engine is engine
        assert agent.function_registry is registry
    
    def test_agent_run_simple(self):
        engine = AutoregressiveEngine(use_stub=True)
        registry = FunctionRegistry()
        
        agent = CodeGenAgent(engine=engine, function_registry=registry)
        result = agent.run("Write a sort function")
        
        assert result.generated_code is not None
        assert result.metrics.generation_duration is not None
        assert result.metrics.generation_duration >= 0
    
    def test_agent_run_with_function_call(self):
        engine = AutoregressiveEngine(use_stub=True)
        registry = FunctionRegistry()
        
        registry.register(
            name="execute_code",
            handler=lambda code: {"output": "executed"},
            description="Execute code",
            parameters={"code": {"type": "string"}},
        )
        
        agent = CodeGenAgent(engine=engine, function_registry=registry)
        result = agent.run("Write a fibonacci function")
        
        assert len(result.function_calls) > 0
        assert result.function_calls[0].name == "execute_code"
        
        assert len(result.function_results) > 0
    
    def test_agent_metrics(self):
        engine = AutoregressiveEngine(use_stub=True)
        registry = FunctionRegistry()
        
        agent = CodeGenAgent(engine=engine, function_registry=registry)
        result = agent.run("test task")
        
        metrics = result.metrics
        
        assert metrics.generation_start_time is not None
        assert metrics.generation_end_time is not None
        assert metrics.generation_end_time >= metrics.generation_start_time
        assert metrics.num_iterations >= 1
        assert metrics.iterations[0].generation_duration >= 0

    def test_agent_metrics_with_function_call(self):
        engine = AutoregressiveEngine(use_stub=True)
        registry = FunctionRegistry()
        registry.register(
            name="execute_code",
            handler=lambda code: {"stdout": "ok", "stderr": "", "exception": None, "success": True},
            description="Execute code",
            parameters={"code": {"type": "string"}},
        )

        agent = CodeGenAgent(engine=engine, function_registry=registry)
        result = agent.run("Write a fibonacci function")

        metrics = result.metrics

        # function_detection_time should be AFTER generation_end_time
        assert metrics.function_detection_time is not None
        assert metrics.function_detection_time >= metrics.generation_end_time
        assert metrics.time_to_function_detection > 0
        assert len(metrics.function_execution_times) > 0
    
    def test_extract_code(self):
        engine = AutoregressiveEngine(use_stub=True)
        registry = FunctionRegistry()
        
        agent = CodeGenAgent(engine=engine, function_registry=registry)
        
        text_with_code = '''Some text
```python
def hello():
    print("hello")
```
More text'''
        
        code = agent._extract_code(text_with_code)
        
        assert "def hello():" in code
        assert "print" in code
