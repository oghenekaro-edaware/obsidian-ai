"""Tests for Gemini JSON Schema adaptation and formatting."""

import json
import pytest
from llm.google_provider import GoogleProvider
from llm.schema_utils import (
    _strip_unsupported_gemini_keys,
    format_schema_dict_for_gemini,
    format_schema_for_gemini,
    normalize_tool_schema_for_gemini,
    normalize_output_schema_for_gemini,
)


def test_strip_unsupported_gemini_keys():
    raw_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "https://example.com/schema.json",
        "title": "TestSchema",
        "type": "object",
        "properties": {
            "field_a": {
                "type": "string",
                "additionalProperties": False,
            },
            "field_b": {
                "type": "object",
                "properties": {
                    "nested": {"type": "integer"}
                },
                "additional_properties": False,
            },
        },
        "additionalProperties": False,
        "additional_properties": False,
    }

    cleaned = _strip_unsupported_gemini_keys(raw_schema)

    assert "$schema" not in cleaned
    assert "$id" not in cleaned
    assert "additionalProperties" not in cleaned
    assert "additional_properties" not in cleaned
    assert "additionalProperties" not in cleaned["properties"]["field_a"]
    assert "additional_properties" not in cleaned["properties"]["field_b"]


def test_normalize_schema_helpers():
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {"foo": {"type": "string", "additionalProperties": False}},
        "additionalProperties": False,
    }
    normalized_out = normalize_output_schema_for_gemini(schema)
    assert "$schema" not in normalized_out
    assert "additionalProperties" not in normalized_out

    normalized_tool = normalize_tool_schema_for_gemini(schema)
    assert "$schema" not in normalized_tool
    assert "additionalProperties" not in normalized_tool


def test_google_provider_convert_tools():
    provider = GoogleProvider(api_key="test_key")
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "web search",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "additionalProperties": False,
                },
            },
        }
    ]
    converted = provider._convert_tools(tools)
    assert len(converted) == 1
    assert "function_declarations" in converted[0]
    decl = converted[0]["function_declarations"][0]
    assert decl["name"] == "search"
    assert "additionalProperties" not in decl["parameters"]
