import pytest
import json
from unittest.mock import AsyncMock, MagicMock
from contextlib import AsyncExitStack

from mcp_client import MCPManager, MCPConnection


@pytest.mark.asyncio
async def test_mcp_manager_execute():
    manager = MCPManager()
    mock_conn = AsyncMock()
    mock_conn.server_name = "github"
    mock_conn.tools = [{
        "type": "function",
        "function": {"name": "mcp__github__create_issue", "description": "Create issue", "parameters": {}},
    }]
    mock_conn.call_tool.return_value = json.dumps({"status": "created"})

    manager.connections["github"] = mock_conn

    # List tools
    tools = manager.list_tools()
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "mcp__github__create_issue"

    # Execute tool
    result = await manager.execute("mcp__github__create_issue", {"title": "Test Issue"})
    mock_conn.call_tool.assert_called_once_with("create_issue", {"title": "Test Issue"})
    assert result == json.dumps({"status": "created"})


@pytest.mark.asyncio
async def test_mcp_manager_unconnected():
    manager = MCPManager()
    result = await manager.execute("mcp__unknown__tool", {})
    assert "not connected" in result
