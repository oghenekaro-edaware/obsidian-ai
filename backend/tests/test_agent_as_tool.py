"""
Tests for 'Agents as Tools' template and execution flow.
Exercises python tool execution with invoke_agent(agent_id, prompt).
"""
import json
from unittest.mock import MagicMock, patch, AsyncMock
import pytest

from routers.chat_router import _execute_python_tool, _execute_tool
from services.tool_executor import tool_executor, ToolRuntimeContext


def test_invoke_agent_target_not_found(monkeypatch):
    """When invoke_agent is called with a non-existent agent_id, return an error string/json."""
    monkeypatch.setattr("config.DATABASE_TYPE", "sqlite")
    monkeypatch.setattr("routers.chat_router.DATABASE_TYPE", "sqlite")

    mock_db = MagicMock()
    mock_db.query().filter().first.return_value = None

    with patch("database.SessionLocal", return_value=mock_db):
        code = """
def handler(params):
    agent_id = "999999"
    prompt = params.get("prompt", "")
    return invoke_agent(agent_id, prompt)
"""
        result = _execute_python_tool(code, {"prompt": "Hello"}, db=mock_db)
        data = json.loads(result)
        assert "error" in data
        assert "not found" in data["error"].lower()


@patch("services.agent_runner.run_agent_headless", new_callable=AsyncMock)
def test_invoke_agent_success(mock_run_headless, monkeypatch):
    """When invoke_agent is called with a valid agent_id, run_agent_headless is invoked and response returned."""
    monkeypatch.setattr("config.DATABASE_TYPE", "sqlite")
    monkeypatch.setattr("routers.chat_router.DATABASE_TYPE", "sqlite")
    mock_run_headless.return_value = "This is the response from the sub-agent."

    class DummyAgent:
        id = 123
        user_id = 1
        name = "ResearchAgent"

    mock_db = MagicMock()
    mock_db.query().filter().first.return_value = DummyAgent()

    with patch("database.SessionLocal", return_value=mock_db):
        code = """
def handler(params):
    agent_id = "123"
    prompt = params.get("prompt", "")
    return invoke_agent(agent_id, prompt)
"""
        result = _execute_python_tool(code, {"prompt": "Find population of Tokyo"}, db=mock_db)
        assert result == "This is the response from the sub-agent."
        mock_run_headless.assert_called_once()


@patch("services.agent_runner.run_agent_headless", new_callable=AsyncMock)
def test_execute_agent_tool_definition(mock_run_headless, monkeypatch):
    """Test executing a ToolDefinition that uses invoke_agent via _execute_tool."""
    monkeypatch.setattr("config.DATABASE_TYPE", "sqlite")
    monkeypatch.setattr("routers.chat_router.DATABASE_TYPE", "sqlite")
    mock_run_headless.return_value = "Sub-agent completed the task."

    class DummyAgent:
        id = 456
        user_id = 1
        name = "CodeReviewer"

    class DummyToolDef:
        handler_type = "python"
        handler_config = json.dumps({
            "code": "def handler(params):\n    return invoke_agent('456', params.get('prompt', ''))"
        })

    mock_db_main = MagicMock()
    mock_db_main.query().filter().first.return_value = DummyToolDef()

    mock_db_worker = MagicMock()
    mock_db_worker.query().filter().first.return_value = DummyAgent()

    with patch("database.SessionLocal", return_value=mock_db_worker):
        output = _execute_tool("ask_code_reviewer", json.dumps({"prompt": "Review this function"}), mock_db_main)
        assert output == "Sub-agent completed the task."


@pytest.mark.asyncio
@patch("services.agent_runner.run_agent_headless", new_callable=AsyncMock)
async def test_tool_executor_agent_as_tool(mock_run_headless, monkeypatch):
    """Test ToolExecutor running an Agent as Tool definition."""
    monkeypatch.setattr("config.DATABASE_TYPE", "sqlite")
    monkeypatch.setattr("routers.chat_router.DATABASE_TYPE", "sqlite")
    mock_run_headless.return_value = "Executor delegated result."

    class DummyAgent:
        id = 789
        user_id = 1
        name = "Translator"

    class DummyToolDef:
        handler_type = "python"
        handler_config = json.dumps({
            "code": "def handler(params):\n    return invoke_agent('789', params.get('prompt', ''))"
        })

    mock_db_main = MagicMock()
    mock_db_main.query().filter().first.return_value = DummyToolDef()

    mock_db_worker = MagicMock()
    mock_db_worker.query().filter().first.return_value = DummyAgent()

    with patch("database.SessionLocal", return_value=mock_db_worker):
        runtime = ToolRuntimeContext(db=mock_db_main)
        res = await tool_executor.execute(
            tool_name="ask_translator",
            arguments={"prompt": "Translate 'Hello' to Spanish"},
            runtime=runtime,
        )
        assert res.output == "Executor delegated result."
        assert res.error is None
