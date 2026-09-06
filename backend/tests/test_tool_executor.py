import pytest
import json
from unittest.mock import AsyncMock, MagicMock

from services.tool_executor import ToolExecutor, ToolRuntimeContext, ToolExecutionResult
from agent_framework import FunctionInvocationContext


@pytest.mark.asyncio
async def test_tool_executor_builtin():
    executor = ToolExecutor()
    context = ToolRuntimeContext(session_id="test_session")

    # Builtin tool: time
    result = await executor.execute(
        tool_name="time",
        arguments={"timezone": "UTC"},
        runtime=context,
    )
    assert isinstance(result, ToolExecutionResult)
    assert result.error is None
    assert "UTC" in result.output or ":" in result.output


@pytest.mark.asyncio
async def test_tool_executor_mcp():
    executor = ToolExecutor()
    mock_mcp_conn = AsyncMock()
    mock_mcp_conn.call_tool.return_value = json.dumps({"result": "mcp_ok"})

    context = ToolRuntimeContext(
        session_id="test_session",
        mcp_connections={"github": mock_mcp_conn},
    )

    result = await executor.execute(
        tool_name="mcp__github__create_issue",
        arguments={"title": "bug"},
        runtime=context,
    )

    mock_mcp_conn.call_tool.assert_called_once_with("create_issue", {"title": "bug"})
    assert result.output == json.dumps({"result": "mcp_ok"})


@pytest.mark.asyncio
async def test_tool_executor_sandbox():
    executor = ToolExecutor()

    # If sandbox tool is invoked without running sandbox container
    no_sandbox_context = ToolRuntimeContext(session_id="test_session")
    res_err = await executor.execute(
        tool_name="execute_python_code",
        arguments={"code": "print(1)"},
        runtime=no_sandbox_context,
    )
    assert "Sandbox is not running" in res_err.output or "error" in res_err.output.lower()
