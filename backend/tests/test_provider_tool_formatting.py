"""Tests for provider message building for tool calls and tool results."""

import json
from llm.base import LLMMessage, LLMToolCall
from llm.google_provider import GoogleProvider
from llm.anthropic_provider import AnthropicProvider


def test_google_provider_build_contents_tool_call_and_response():
    provider = GoogleProvider(api_key="fake", model_id="gemini-2.5-flash")

    messages = [
        LLMMessage(role="user", content="What is the current time?"),
        LLMMessage(
            role="assistant",
            content="",
            tool_calls=[LLMToolCall(id="call_1", name="get_datetime", arguments="{}")],
        ),
        LLMMessage(
            role="tool",
            content='{"result": "2026-03-31 12:00:00"}',
            tool_call_id="call_1",
        ),
    ]

    contents = provider._build_contents(messages)

    assert len(contents) == 3

    # Turn 0: user text
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"] == [{"text": "What is the current time?"}]

    # Turn 1: model functionCall
    assert contents[1]["role"] == "model"
    assert len(contents[1]["parts"]) == 1
    assert contents[1]["parts"][0]["functionCall"]["name"] == "get_datetime"
    assert contents[1]["parts"][0]["functionCall"]["args"] == {}

    # Turn 2: user functionResponse
    assert contents[2]["role"] == "user"
    assert contents[2]["parts"][0]["functionResponse"]["name"] == "get_datetime"
    assert contents[2]["parts"][0]["functionResponse"]["response"] == {"result": "2026-03-31 12:00:00"}


def test_anthropic_provider_build_messages_tool_call_and_response():
    provider = AnthropicProvider(api_key="fake", model_id="claude-sonnet-5")

    messages = [
        LLMMessage(role="user", content="Fetch webpage"),
        LLMMessage(
            role="assistant",
            content="I will fetch it.",
            tool_calls=[LLMToolCall(id="call_abc", name="http_request", arguments='{"url": "https://example.com"}')],
        ),
        LLMMessage(
            role="tool",
            content="<html>Hello</html>",
            tool_call_id="call_abc",
        ),
    ]

    built = provider._build_messages(messages)

    assert len(built) == 3
    # Msg 0: user text block
    assert built[0]["role"] == "user"
    assert built[0]["content"] == [{"type": "text", "text": "Fetch webpage"}]

    # Msg 1: assistant text block + tool_use block
    assert built[1]["role"] == "assistant"
    assert built[1]["content"] == [
        {"type": "text", "text": "I will fetch it."},
        {"type": "tool_use", "id": "call_abc", "name": "http_request", "input": {"url": "https://example.com"}},
    ]

    # Msg 2: user tool_result block
    assert built[2]["role"] == "user"
    assert built[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "call_abc", "content": "<html>Hello</html>"},
    ]
