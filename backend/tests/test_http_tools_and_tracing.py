import json
import pytest
from unittest.mock import MagicMock, patch

from routers.chat_router import (
    _resolve_http_tool_request_params,
    _extract_usage_tokens,
    _estimate_cost_usd,
    _execute_tool,
)
from services.key_resolution_service import resolve_embedding_credentials
from team_delegation_tools import execute_call_teammate


def test_resolve_http_tool_request_params():
    # Dynamic URL in arguments overrides static handler config
    config = {"url": "https://api.example.com/default", "method": "POST", "headers": {"Authorization": "Bearer 123"}}
    arguments = {"url": "https://www.google.com/", "method": "GET", "q": "test"}

    url, method, headers, params, body = _resolve_http_tool_request_params(config, arguments, "http_request_2")

    assert url == "https://www.google.com/"
    assert method == "GET"
    assert headers == {"Authorization": "Bearer 123"}
    assert params == {"q": "test"}
    assert body is None


def test_resolve_http_tool_request_params_template():
    # URL template formatting
    config = {"url": "https://api.github.com/repos/{owner}/{repo}", "method": "GET"}
    arguments = {"owner": "octocat", "repo": "hello-world"}

    url, method, headers, params, body = _resolve_http_tool_request_params(config, arguments, "github_get")

    assert url == "https://api.github.com/repos/octocat/hello-world"
    assert method == "GET"
    assert params is None


def test_resolve_http_tool_request_params_post_body():
    config = {"url": "https://api.example.com/data", "method": "POST"}
    arguments = {"name": "jules", "action": "run"}

    url, method, headers, params, body = _resolve_http_tool_request_params(config, arguments)

    assert url == "https://api.example.com/data"
    assert method == "POST"
    assert params is None
    assert body == {"name": "jules", "action": "run"}


def test_extract_usage_tokens():
    # Dictionary
    assert _extract_usage_tokens({"input_tokens": 100, "output_tokens": 50}) == (100, 50)
    assert _extract_usage_tokens({"prompt_tokens": 80, "completion_tokens": 20}) == (80, 20)
    assert _extract_usage_tokens({"input_token_count": 60, "output_token_count": 30}) == (60, 30)

    # Object with attributes
    class UsageObj:
        input_token_count = 200
        output_token_count = 100

    assert _extract_usage_tokens(UsageObj()) == (200, 100)

    # None or empty
    assert _extract_usage_tokens(None) == (0, 0)
    assert _extract_usage_tokens({}) == (0, 0)


def test_estimate_cost_usd():
    # Known model (gpt-4o)
    cost = _estimate_cost_usd("gpt-4o", 1_000_000, 1_000_000)
    assert cost == 12.5  # 2.5 + 10.0

    # Unknown model
    assert _estimate_cost_usd("unknown-custom-model", 1000, 500) is None


@pytest.mark.asyncio
async def test_resolve_embedding_credentials_null_provider_safety():
    # provider is None or missing
    kb_config = {"embedding_provider": None, "secret_id": None}
    prov, api_key, model = await resolve_embedding_credentials("1", kb_config, db=None)
    assert prov == "google"
    assert api_key == "dummy_embedding_key"


@pytest.mark.asyncio
async def test_execute_call_teammate_null_safety():
    # teammate_name is None
    args_str = json.dumps({"teammate_name": None, "message": "hello"})
    res = await execute_call_teammate(args_str, [], 1, "sqlite", db=None)
    data = json.loads(res)
    assert "error" in data


def test_execute_tool_http_redirect_and_dynamic_url():
    class DummyToolDef:
        handler_type = "http"
        handler_config = json.dumps({"url": "https://api.example.com/fallback", "method": "POST"})

    mock_db = MagicMock()
    mock_db.query().filter().first.return_value = DummyToolDef()

    with patch("socket.getaddrinfo") as mock_dns:
        mock_dns.return_value = [(2, 1, 6, '', ('93.184.216.34', 443))]
        with patch("httpx.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_resp = MagicMock()
            mock_resp.text = "<html>Google</html>"
            mock_client.get.return_value = mock_resp

            args = json.dumps({"url": "https://www.google.com/", "method": "GET"})
            res = _execute_tool("http_request_2", args, mock_db)

            assert res == "<html>Google</html>"
            mock_client_cls.assert_called_once_with(timeout=30.0, follow_redirects=True)
            mock_client.get.assert_called_once_with("https://www.google.com/", params=None, headers={})


def test_ssrf_blocking_private_ips():
    class DummyToolDef:
        handler_type = "http"
        handler_config = json.dumps({"url": "http://127.0.0.1/admin", "method": "GET"})

    mock_db = MagicMock()
    mock_db.query().filter().first.return_value = DummyToolDef()

    res = _execute_tool("ssrf_test", "", mock_db)
    data = json.loads(res)
    assert "error" in data
    assert "SSRF blocked" in data["error"] or "blocked for security reasons" in data["error"]


def test_ssrf_allow_private_networks_override(monkeypatch):
    monkeypatch.setenv("ALLOW_PRIVATE_NETWORKS", "true")

    class DummyToolDef:
        handler_type = "http"
        handler_config = json.dumps({"url": "http://127.0.0.1/admin", "method": "GET"})

    mock_db = MagicMock()
    mock_db.query().filter().first.return_value = DummyToolDef()

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.text = "ok"
        mock_client.get.return_value = mock_resp

        res = _execute_tool("ssrf_test", "", mock_db)
        assert res == "ok"


def test_dns_resolution_failure_handling():
    class DummyToolDef:
        handler_type = "http"
        handler_config = json.dumps({"url": "https://nonexistent-domain-xyz-1234.com/api", "method": "GET"})

    mock_db = MagicMock()
    mock_db.query().filter().first.return_value = DummyToolDef()

    import socket
    with patch("socket.getaddrinfo", side_effect=socket.gaierror(-5, "No address associated with hostname")):
        res = _execute_tool("dns_test", "", mock_db)
        data = json.loads(res)
        assert "error" in data
        assert "DNS resolution failed" in data["error"] or "No address associated with hostname" in data["error"]
