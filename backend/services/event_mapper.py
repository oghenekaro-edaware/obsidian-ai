"""
Standardized EventMapper service for building Server-Sent Events (SSE) data packages.
"""

import json
from typing import Any, Dict


class EventMapper:
    """Helper class for mapping runtime agent and tool events into SSE dictionary payloads."""

    @staticmethod
    def text_delta(content: str) -> Dict[str, Any]:
        return {"event": "content_delta", "data": json.dumps({"content": content})}

    @staticmethod
    def reasoning_delta(content: str) -> Dict[str, Any]:
        return {"event": "reasoning_delta", "data": json.dumps({"content": content})}

    @staticmethod
    def tool_call(
        call_id: str,
        name: str,
        arguments: Any,
        result: Any | None = None,
        status: str = "running",
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "id": call_id,
            "name": name,
            "arguments": arguments,
            "status": status,
        }
        if result is not None:
            data["result"] = str(result)
        return {"event": "tool_call", "data": json.dumps(data)}

    @staticmethod
    def hitl_required(
        approval_id: str,
        session_id: str,
        tool_call_id: str,
        tool_name: str,
        tool_arguments: Any,
    ) -> Dict[str, Any]:
        return {
            "event": "hitl_approval_required",
            "data": json.dumps({
                "approval_id": approval_id,
                "session_id": session_id,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
                "tool_arguments": tool_arguments,
            }),
        }

    @staticmethod
    def message_complete(msg_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"event": "message_complete", "data": json.dumps(msg_data)}

    @staticmethod
    def token_usage(
        input_tokens: int,
        output_tokens: int,
        session_total_input: int,
        session_total_output: int,
    ) -> Dict[str, Any]:
        return {
            "event": "token_usage",
            "data": json.dumps({
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "session_total_input": session_total_input,
                "session_total_output": session_total_output,
            }),
        }

    @staticmethod
    def error(error_msg: str) -> Dict[str, Any]:
        return {"event": "error", "data": json.dumps({"error": error_msg})}

    @staticmethod
    def done() -> Dict[str, Any]:
        return {"event": "done", "data": "{}"}
