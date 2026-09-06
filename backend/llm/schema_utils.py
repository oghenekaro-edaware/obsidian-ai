"""Utilities for adapting and formatting JSON Schemas across LLM providers."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

_UNSUPPORTED_GEMINI_SCHEMA_KEYS = {
    "additionalProperties",
    "additional_properties",
    "$schema",
    "$id",
}


def _strip_unsupported_gemini_keys(obj: Any) -> Any:
    """Recursively strip keywords not supported by Google Gemini's responseSchema REST API."""
    if isinstance(obj, dict):
        return {
            k: _strip_unsupported_gemini_keys(v)
            for k, v in obj.items()
            if k not in _UNSUPPORTED_GEMINI_SCHEMA_KEYS
        }
    elif isinstance(obj, list):
        return [_strip_unsupported_gemini_keys(item) for item in obj]
    return obj


def format_schema_dict_for_gemini(schema: Any) -> dict | Any:
    """Recursively strip unsupported JSON Schema keys for Gemini, returning a clean dict."""
    if not isinstance(schema, dict):
        return schema
    return _strip_unsupported_gemini_keys(schema)


def normalize_tool_schema_for_gemini(schema: Any) -> Any:
    """Normalize tool parameter schemas for Gemini REST API."""
    if not isinstance(schema, dict):
        return schema
    return _strip_unsupported_gemini_keys(schema)


def normalize_output_schema_for_gemini(schema: Any) -> Any:
    """Normalize output responseJsonSchema for Gemini REST API."""
    if not isinstance(schema, dict):
        return schema
    return _strip_unsupported_gemini_keys(schema)


def format_schema_for_gemini(schema: Any) -> Any:
    """Adapt a JSON schema dict into a Gemini-compatible Schema object or clean dict.

    Uses google.genai.types.Schema.from_json_schema when available to build a native
    Schema instance, or falls back to a recursively cleaned schema dictionary.
    """
    if not isinstance(schema, dict):
        return schema

    cleaned_dict = _strip_unsupported_gemini_keys(schema)

    try:
        from google.genai import types

        json_schema_obj = types.JSONSchema(**cleaned_dict)
        gemini_schema = types.Schema.from_json_schema(json_schema=json_schema_obj)
        return gemini_schema
    except Exception as e:
        logger.debug("Could not convert schema dict to google.genai.types.Schema: %s", e)
        return cleaned_dict
