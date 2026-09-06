import json
import logging
from dataclasses import dataclass, field
from typing import Any

from builtin_tools import is_builtin_tool, execute_builtin_tool
from sandbox_tools import is_sandbox_tool, execute_sandbox_tool
from mcp_client import parse_mcp_tool_name

logger = logging.getLogger(__name__)


@dataclass
class ToolRuntimeContext:
    session_id: str = ""
    agent_id: str | None = None
    db: Any = None
    mongo_db: Any = None
    sandbox_container_id: str | None = None
    mcp_connections: dict[str, Any] = field(default_factory=dict)
    event_queue: Any | None = None
    agent: Any | None = None
    llm: Any | None = None
    tool_call_id: str | None = None

    @classmethod
    def from_dict_or_context(cls, data: dict[str, Any] | None) -> "ToolRuntimeContext":
        if not data:
            return cls()
        return cls(
            session_id=str(data.get("session_id", "")),
            agent_id=data.get("agent_id"),
            db=data.get("db"),
            mongo_db=data.get("mongo_db"),
            sandbox_container_id=data.get("sandbox_container_id"),
            mcp_connections=data.get("mcp_connections") or {},
            event_queue=data.get("event_queue"),
            agent=data.get("agent"),
            llm=data.get("llm"),
            tool_call_id=data.get("tool_call_id"),
        )


@dataclass
class ToolExecutionResult:
    output: str
    error: str | None = None


class ToolExecutor:
    """Unified application-level execution router for all tool types."""

    async def execute(
        self,
        *,
        tool_name: str,
        arguments: dict | str,
        runtime: ToolRuntimeContext | dict[str, Any] | None = None,
    ) -> ToolExecutionResult:
        if isinstance(runtime, dict):
            ctx = ToolRuntimeContext.from_dict_or_context(runtime)
        elif isinstance(runtime, ToolRuntimeContext):
            ctx = runtime
        else:
            ctx = ToolRuntimeContext()

        if isinstance(arguments, dict):
            tc_arguments_str = json.dumps(arguments)
            tc_arguments_dict = arguments
        else:
            tc_arguments_str = arguments or "{}"
            try:
                tc_arguments_dict = json.loads(tc_arguments_str) if tc_arguments_str else {}
            except json.JSONDecodeError:
                tc_arguments_dict = {}

        # 1. Builtin tools
        if is_builtin_tool(tool_name):
            try:
                out = await execute_builtin_tool(tool_name, tc_arguments_str)
                return ToolExecutionResult(output=out)
            except Exception as e:
                logger.exception("Error executing builtin tool %s", tool_name)
                return ToolExecutionResult(output=json.dumps({"error": str(e)}), error=str(e))

        # 2. Sandbox tools
        if is_sandbox_tool(tool_name):
            if not ctx.sandbox_container_id:
                out = json.dumps({"error": "Sandbox is not running for this agent"})
                return ToolExecutionResult(output=out, error="Sandbox not running")
            try:
                out = await execute_sandbox_tool(tool_name, tc_arguments_str, ctx.sandbox_container_id)
                return ToolExecutionResult(output=out)
            except Exception as e:
                logger.exception("Error executing sandbox tool %s", tool_name)
                return ToolExecutionResult(output=json.dumps({"error": str(e)}), error=str(e))

        # 3. MCP tools
        parsed = parse_mcp_tool_name(tool_name)
        if parsed:
            server_name, original_tool_name = parsed
            conn = ctx.mcp_connections.get(server_name)
            if conn:
                try:
                    out = await conn.call_tool(original_tool_name, tc_arguments_dict)
                    return ToolExecutionResult(output=out)
                except Exception as e:
                    logger.exception("Error calling MCP tool %s on %s", original_tool_name, server_name)
                    return ToolExecutionResult(output=json.dumps({"error": str(e)}), error=str(e))
            else:
                out = json.dumps({"error": f"MCP server '{server_name}' not connected"})
                return ToolExecutionResult(output=out, error=f"MCP server '{server_name}' not connected")

        # 4. Native / DB tools
        try:
            from routers.chat_router import _execute_tool, _execute_tool_mongo
            if ctx.mongo_db:
                out = await _execute_tool_mongo(tool_name, tc_arguments_str, ctx.mongo_db)
            else:
                out = _execute_tool(tool_name, tc_arguments_str, ctx.db)
            return ToolExecutionResult(output=out)
        except Exception as e:
            logger.exception("Error executing native tool %s", tool_name)
            return ToolExecutionResult(output=json.dumps({"error": str(e)}), error=str(e))


# Global default instance
tool_executor = ToolExecutor()
