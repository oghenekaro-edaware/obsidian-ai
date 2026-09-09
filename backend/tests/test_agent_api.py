import os
os.environ["DATABASE_TYPE"] = "sqlite"

import pytest
import config
config.DATABASE_TYPE = "sqlite"

from fastapi.testclient import TestClient
from main import app
from database import get_db, Base, engine
from models import User, Application, APIKey, Agent, Schema, SchemaVersion, AgentAPIConfig
from auth import create_access_token
from services.api_key_service import generate_api_key

client = TestClient(app)

@pytest.fixture
def setup_data():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = next(get_db())
    try:
        # Create test user
        user = User(username="testuser", email="test@example.com", hashed_password="pw", role="admin")
        db.add(user)
        db.flush()

        # Create user token
        token = create_access_token({"user_id": str(user.id), "username": user.username, "role": user.role, "token_type": "user"})

        # Create application
        app_obj = Application(user_id=user.id, name="Test App", status="active", default_scopes_json='["agent:invoke", "agent:read"]')
        db.add(app_obj)
        db.flush()

        # Create API key
        import json
        from services.api_key_service import hash_api_key
        prefix, secret, full_key = generate_api_key()
        key_record = APIKey(
            application_id=app_obj.id,
            name="Test Key",
            key_prefix=prefix,
            secret_hash=hash_api_key(secret),
            scopes_json=json.dumps(["agent:invoke", "agent:read"])
        )
        db.add(key_record)
        db.flush()

        # Create Schemas
        input_schema = Schema(user_id=user.id, name="Input Schema", direction="input")
        db.add(input_schema)
        db.flush()
        in_ver = SchemaVersion(
            schema_id=input_schema.id,
            version_number=1,
            canonical_schema_json='{"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}',
            source_format="json_schema"
        )
        db.add(in_ver)

        output_schema = Schema(user_id=user.id, name="Output Schema", direction="output")
        db.add(output_schema)
        db.flush()
        out_ver = SchemaVersion(
            schema_id=output_schema.id,
            version_number=1,
            canonical_schema_json='{"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}',
            source_format="json_schema"
        )
        db.add(out_ver)
        db.flush()

        # Create Agent
        agent = Agent(user_id=user.id, name="Test Agent", is_active=True)
        db.add(agent)
        db.flush()

        # Expose and publish agent
        api_config = AgentAPIConfig(
            agent_id=agent.id,
            owner_application_id=app_obj.id,
            publication_state="published",
            input_schema_version_id=in_ver.id,
            output_schema_version_id=out_ver.id,
            required_scopes_json='["agent:invoke"]'
        )
        db.add(api_config)
        db.commit()

        yield {
            "user": user,
            "user_token": token,
            "app": app_obj,
            "api_key": full_key,
            "agent": agent,
            "input_schema": input_schema,
            "output_schema": output_schema,
            "in_ver": in_ver,
            "out_ver": out_ver
        }
    finally:
        db.close()

def test_get_agent_api_config(setup_data):
    token = setup_data["user_token"]
    agent_id = setup_data["agent"].id

    # GET via agent-api-configs route
    res = client.get(f"/api/v1/agent-api-configs/{agent_id}", headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200
    data = res.json()
    assert data["agent_id"] == str(agent_id)
    assert data["publication_state"] == "published"

    # GET via agents/{id}/api-config route
    res2 = client.get(f"/api/v1/agents/{agent_id}/api-config", headers={"Authorization": f"Bearer {token}"})
    assert res2.status_code == 200
    assert res2.json()["agent_id"] == str(agent_id)

def test_auth_header_flexibility(setup_data):
    agent_id = setup_data["agent"].id
    key = setup_data["api_key"]

    # Test Bearer header
    payload = {"input": {"query": "hello"}, "output": {"answer": "world"}}
    res = client.post(
        f"/api/v1/agent-invocations/{agent_id}",
        headers={"Authorization": f"Bearer {key}"},
        json=payload
    )
    assert res.status_code == 200
    assert res.json()["status"] == "completed"
    assert res.json()["output"] == {"answer": "world"}

    # Test X-API-Key header
    res2 = client.post(
        f"/api/v1/agent-invocations/{agent_id}",
        headers={"X-API-Key": key},
        json=payload
    )
    assert res2.status_code == 200
    assert res2.json()["status"] == "completed"

def test_application_key_preferred_over_user_bearer(setup_data):
    agent_id = setup_data["agent"].id
    payload = {"input": {"query": "hello"}, "output": {"answer": "world"}}
    res = client.post(
        f"/api/v1/agent-invocations/{agent_id}",
        headers={
            "Authorization": f"Bearer {setup_data['user_token']}",
            "X-API-Key": setup_data["api_key"],
        },
        json=payload,
    )
    assert res.status_code == 200
    assert res.json()["status"] == "completed"

def test_application_key_allows_copied_header_whitespace(setup_data):
    agent_id = setup_data["agent"].id
    payload = {"input": {"query": "hello"}, "output": {"answer": "world"}}
    res = client.post(
        f"/api/v1/agent-invocations/{agent_id}",
        headers={"X-API-Key": f"  {setup_data['api_key']}  "},
        json=payload,
    )
    assert res.status_code == 200

def test_presupplied_output_validation_failure(setup_data):
    agent_id = setup_data["agent"].id
    key = setup_data["api_key"]

    # Invalid output supplied (missing required 'answer' property)
    payload = {"input": {"query": "hello"}, "output": {"invalid_field": 123}}
    res = client.post(
        f"/api/v1/agent-invocations/{agent_id}",
        headers={"Authorization": f"Bearer {key}"},
        json=payload
    )
    assert res.status_code == 502
    data = res.json()
    assert data["detail"]["error"]["code"] == "OUTPUT_SCHEMA_VALIDATION_FAILED"

def test_external_session_is_application_bound(setup_data):
    from models import Session, Message, Application, APIKey
    from services.api_key_service import hash_api_key
    import json
    db = next(get_db())
    try:
        session = Session(
            user_id=setup_data["user"].id,
            application_id=setup_data["app"].id,
            title="Bug report",
            entity_type="agent",
            entity_id=setup_data["agent"].id,
        )
        db.add(session)
        db.flush()
        db.add(Message(session_id=session.id, role="user", content="checkout screenshot"))
        db.commit()
        response = client.get(
            f"/api/v1/agent-sessions/{session.id}/messages",
            headers={"Authorization": f"Bearer {setup_data['api_key']}"},
        )
        assert response.status_code == 200
        assert response.json()["messages"][0]["content"] == "checkout screenshot"

        other_app = Application(user_id=setup_data["user"].id, name="Other", status="active")
        db.add(other_app)
        db.flush()
        other_prefix, other_secret, other_full_key = generate_api_key()
        db.add(APIKey(
            application_id=other_app.id,
            name="Other Key",
            key_prefix=other_prefix,
            secret_hash=hash_api_key(other_secret),
            scopes_json=json.dumps(["agent:read"]),
        ))
        db.commit()
        response = client.get(
            f"/api/v1/agent-sessions/{session.id}/messages",
            headers={"Authorization": f"Bearer {other_full_key}"},
        )
        assert response.status_code == 404
    finally:
        db.close()

def test_meta_schema_validation(setup_data):
    token = setup_data["user_token"]

    # Invalid JSON schema
    invalid_schema = {
        "name": "Bad Schema",
        "direction": "input",
        "canonical_schema": {"type": "unsupported_type_xyz"}
    }
    res = client.post(
        "/api/v1/schemas",
        headers={"Authorization": f"Bearer {token}"},
        json=invalid_schema
    )
    assert res.status_code == 422
    assert "INVALID_JSON_SCHEMA" in res.text


def test_api_config_rejects_schema_from_wrong_direction(setup_data):
    token = setup_data["user_token"]
    agent_id = setup_data["agent"].id
    res = client.put(
        f"/api/v1/agents/{agent_id}/api-config",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "input_schema_version_id": str(setup_data["out_ver"].id),
            "output_schema_version_id": str(setup_data["out_ver"].id),
        },
    )
    assert res.status_code == 422
    assert "Input schema version" in res.text


def test_api_config_rejects_schema_owned_by_another_user(setup_data):
    from models import User

    db = next(get_db())
    try:
        other_user = User(username="other", email="other@example.com", hashed_password="pw", role="user")
        db.add(other_user)
        db.flush()
        other_schema = Schema(user_id=other_user.id, name="Other input", direction="input")
        db.add(other_schema)
        db.flush()
        other_version = SchemaVersion(
            schema_id=other_schema.id,
            version_number=1,
            canonical_schema_json='{"type":"object"}',
            source_format="json_schema",
        )
        db.add(other_version)
        db.commit()
        other_version_id = other_version.id
    finally:
        db.close()

    res = client.put(
        f"/api/v1/agents/{setup_data['agent'].id}/api-config",
        headers={"Authorization": f"Bearer {setup_data['user_token']}"},
        json={
            "input_schema_version_id": str(other_version_id),
            "output_schema_version_id": str(setup_data["out_ver"].id),
        },
    )
    assert res.status_code == 422
    assert "Input schema version" in res.text

def test_external_attachment_requires_data_or_url():
    from pydantic import ValidationError
    from schemas import ExternalInvokeRequest

    with pytest.raises(ValidationError):
        ExternalInvokeRequest(input={"query": "hello"}, attachments=[{
            "filename": "empty.png",
            "media_type": "image/png",
            "file_type": "image",
        }])


def test_external_invoke_supports_session_context():
    from schemas import ExternalInvokeRequest

    request = ExternalInvokeRequest(
        input={"query": "hello"},
        system_instruction="Answer in bullets.",
        knowledge_base_ids=["1", "2", "3"],
    )

    assert request.system_instruction == "Answer in bullets."
    assert request.knowledge_base_ids == ["1", "2", "3"]


def test_agent_invocation_executes_configured_tools_mcps_and_trace_spans(setup_data):
    import json
    from unittest.mock import patch, MagicMock
    from models import ToolDefinition, MCPServer, LLMProvider, TraceSpan, Message, Session

    db = next(get_db())
    try:
        user = setup_data["user"]
        agent = setup_data["agent"]

        # Create LLMProvider
        provider = LLMProvider(
            user_id=user.id,
            name="Mock Provider",
            provider_type="openai",
            model_id="gpt-4o",
            api_key="enc_key",
        )
        db.add(provider)
        db.flush()

        agent = db.get(Agent, setup_data["agent"].id)
        agent.provider_id = provider.id
        db.commit()

        # 1. Create a custom tool definition
        custom_tool = ToolDefinition(
            user_id=user.id,
            name="get_weather_info",
            description="Get weather for location",
            handler_type="python",
            handler_config=json.dumps({"code": "def handler(location):\n    return f'Sunny in {location}'"}),
            parameters_json=json.dumps({
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            }),
            is_active=True,
        )
        db.add(custom_tool)
        db.flush()

        agent.tools_json = json.dumps([custom_tool.id])

        # 2. Create an MCP server record
        mcp_server = MCPServer(
            user_id=user.id,
            name="echo_mcp",
            transport_type="stdio",
            command="echo",
            is_active=True,
        )
        db.add(mcp_server)
        db.flush()

        agent.mcp_servers_json = json.dumps([mcp_server.id])
        db.commit()

        # Mock MCP connection & tool discovery
        mock_mcp_conn = MagicMock()
        async def mock_call_tool(name, args):
            return f"MCP Echo: {args.get('msg')}"
        mock_mcp_conn.call_tool = mock_call_tool

        mcp_tools = [{
            "type": "function",
            "function": {
                "name": "mcp__echo_mcp__echo_tool",
                "description": "Echo message via MCP",
                "parameters": {"type": "object", "properties": {"msg": {"type": "string"}}},
            }
        }]

        async def mock_connect_mcp_servers(stack, configs):
            return {"echo_mcp": mock_mcp_conn}, mcp_tools

        call_count = 0
        async def mock_chat_stream(messages, system_prompt=None, tools=None, response_schema=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                tool_names = [t["function"]["name"] for t in tools] if tools else []
                assert "get_weather_info" in tool_names
                assert "mcp__echo_mcp__echo_tool" in tool_names

                class ToolChunk1:
                    type = "tool_call"
                    class ToolCall1:
                        id = "call_1"
                        name = "get_weather_info"
                        arguments = {"location": "London"}
                    tool_call = ToolCall1()

                class ToolChunk2:
                    type = "tool_call"
                    class ToolCall2:
                        id = "call_2"
                        name = "mcp__echo_mcp__echo_tool"
                        arguments = {"msg": "hello"}
                    tool_call = ToolCall2()

                class DoneChunk:
                    type = "done"
                    usage = {"input_tokens": 10, "output_tokens": 10}
                    finish_reason = "tool_calls"
                    tool_call = None

                yield ToolChunk1()
                yield ToolChunk2()
                yield DoneChunk()
            else:
                class ContentChunk:
                    type = "content"
                    content = json.dumps({"answer": "Weather is sunny and MCP responded hello"})
                    tool_call = None
                class DoneChunk:
                    type = "done"
                    usage = {"input_tokens": 15, "output_tokens": 15}
                    finish_reason = "stop"
                    tool_call = None
                yield ContentChunk()
                yield DoneChunk()

        mock_llm = MagicMock()
        mock_llm.chat_stream = mock_chat_stream

        with patch("llm.provider_factory.create_provider_from_config", return_value=mock_llm), \
             patch("encryption.decrypt_api_key", return_value="fake_key"), \
             patch("routers.chat_router._connect_mcp_servers", side_effect=mock_connect_mcp_servers), \
             patch("rag_service.VectorStoreContextProvider.before_run"), \
             patch("routers.memory_router.MemoryContextProvider.before_run"):

            payload = {"input": {"query": "What is the weather?"}}
            res = client.post(
                f"/api/v1/agent-invocations/{agent.id}",
                headers={"Authorization": f"Bearer {setup_data['api_key']}"},
                json=payload,
            )

            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "completed"
            assert "Weather is sunny" in data["output"]["answer"]

            session_id = int(data["session_id"])
            spans = db.query(TraceSpan).filter(TraceSpan.session_id == session_id).all()
            assert len(spans) >= 2

            assistant_msg = db.query(Message).filter(Message.session_id == session_id, Message.role == "assistant").first()
            assert assistant_msg is not None

            for span in spans:
                assert span.message_id == assistant_msg.id

            mcp_spans = [s for s in spans if s.span_type == "mcp_call"]
            tool_spans = [s for s in spans if s.span_type == "tool_call"]
            assert len(mcp_spans) >= 1
            assert len(tool_spans) >= 1
    finally:
        db.close()


def test_agent_invocation_uses_published_version_snapshot_tools(setup_data):
    import json
    from unittest.mock import patch, MagicMock
    from models import ToolDefinition, LLMProvider, Agent, AgentAPIConfig

    db = next(get_db())
    try:
        user = setup_data["user"]

        provider = LLMProvider(
            user_id=user.id,
            name="Mock Provider 2",
            provider_type="openai",
            model_id="gpt-4o",
            api_key="enc_key",
        )
        db.add(provider)
        db.flush()

        agent = db.get(Agent, setup_data["agent"].id)
        agent.provider_id = provider.id

        published_tool = ToolDefinition(
            user_id=user.id,
            name="published_tool_fn",
            description="Tool available in published snapshot",
            handler_type="python",
            handler_config=json.dumps({"code": "def handler(x):\n    return f'Result {x}'"}),
            parameters_json=json.dumps({
                "type": "object",
                "properties": {"x": {"type": "string"}},
            }),
            is_active=True,
        )
        db.add(published_tool)
        db.flush()
        tool_id = published_tool.id

        agent.tools_json = json.dumps([tool_id])
        db.commit()

        # Verify tool_id in db
        tool_in_db = db.query(ToolDefinition).filter(ToolDefinition.id == tool_id).first()
        assert tool_in_db is not None

        # Publish agent
        token = setup_data["user_token"]
        pub_res = client.post(
            f"/api/v1/agents/{agent.id}/publish",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert pub_res.status_code == 200

        # Now edit the draft agent directly in DB to clear its tools
        agent = db.get(Agent, setup_data["agent"].id)
        agent.tools_json = None
        db.commit()

        received_tools = []
        async def mock_chat_stream(messages, system_prompt=None, tools=None, response_schema=None):
            nonlocal received_tools
            received_tools = ["WAS_CALLED"] + [t["function"]["name"] for t in (tools or [])]
            class ContentChunk:
                type = "content"
                content = json.dumps({"answer": "snapshot tool verified"})
                tool_call = None
            class DoneChunk:
                type = "done"
                usage = {"input_tokens": 5, "output_tokens": 5}
                finish_reason = "stop"
                tool_call = None
            yield ContentChunk()
            yield DoneChunk()

        mock_llm = MagicMock()
        mock_llm.chat_stream.side_effect = mock_chat_stream

        with patch("llm.provider_factory.create_provider_from_config", return_value=mock_llm), \
             patch("encryption.decrypt_api_key", return_value="fake_key"), \
             patch("rag_service.VectorStoreContextProvider.before_run"), \
             patch("routers.memory_router.MemoryContextProvider.before_run"):

            payload = {"input": {"query": "test published tool"}}
            res = client.post(
                f"/api/v1/agent-invocations/{agent.id}",
                headers={"Authorization": f"Bearer {setup_data['api_key']}"},
                json=payload,
            )

            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "completed"
            assert "published_tool_fn" in received_tools
    finally:
        db.close()


def test_agent_invocation_and_session_with_kb_external_id(setup_data):
    import json
    from unittest.mock import patch, MagicMock
    from models import KnowledgeBase, LLMProvider, Agent

    db = next(get_db())
    try:
        user = setup_data["user"]
        agent = db.get(Agent, setup_data["agent"].id)

        # Create provider for agent
        provider = LLMProvider(
            user_id=user.id,
            name="KB Provider",
            provider_type="openai",
            model_id="gpt-4o",
            api_key="enc_key",
        )
        db.add(provider)
        db.flush()
        agent.provider_id = provider.id

        # Create KnowledgeBase with external_id
        kb = KnowledgeBase(
            user_id=user.id,
            owner_id=str(user.id),
            app_id="my-app",
            external_id="proj-kb-ext-100",
            name="Project KB 100",
            description="Knowledge base description",
            scope_type="workspace",
            embedding_provider="openai",
            is_active=True,
        )
        db.add(kb)
        db.commit()
        db.refresh(kb)
        kb_id_int = kb.id

        # Test 1: create_external_session with knowledge_base_ids containing external_id
        sess_res = client.post(
            f"/api/v1/agent-sessions/{agent.id}",
            headers={"Authorization": f"Bearer {setup_data['api_key']}"},
            json={
                "title": "Session with KB Ext ID",
                "knowledge_base_ids": ["proj-kb-ext-100"],
            },
        )
        assert sess_res.status_code == 200
        sess_data = sess_res.json()
        assert sess_data["title"] == "Session with KB Ext ID"

        # Test 2: agent invocation passing knowledge_base_ids with external_id
        captured_search_kb_ids = []
        async def mock_search_kb_async(target_kb_id, query, top_k=5, **kwargs):
            captured_search_kb_ids.append(target_kb_id)
            return [{"text": "Found grounded info from KB 100", "score": 0.9, "metadata": {"doc_name": "Doc 100"}}]

        async def mock_chat_stream(messages, system_prompt=None, tools=None, response_schema=None):
            assert "Grounded Knowledge Base Context" in system_prompt or "Found grounded info" in system_prompt or "Knowledge Base" in system_prompt
            class ContentChunk:
                type = "content"
                content = json.dumps({"answer": "KB context retrieved successfully"})
                tool_call = None
            class DoneChunk:
                type = "done"
                usage = {"input_tokens": 10, "output_tokens": 10}
                finish_reason = "stop"
                tool_call = None
            yield ContentChunk()
            yield DoneChunk()

        mock_llm = MagicMock()
        mock_llm.chat_stream.side_effect = mock_chat_stream

        with patch("llm.provider_factory.create_provider_from_config", return_value=mock_llm), \
             patch("encryption.decrypt_api_key", return_value="fake_key"), \
             patch("rag_service.RAGService.search_kb_async", side_effect=mock_search_kb_async), \
             patch("services.key_resolution_service.resolve_embedding_credentials", return_value=("openai", "sk-fake", "text-embedding-3-small")):

            payload = {
                "input": {"query": "Tell me about KB 100"},
                "knowledge_base_ids": ["proj-kb-ext-100"],
            }
            res = client.post(
                f"/api/v1/agent-invocations/{agent.id}",
                headers={"Authorization": f"Bearer {setup_data['api_key']}"},
                json=payload,
            )

            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "completed"
            assert data["output"] == {"answer": "KB context retrieved successfully"}

            # Verify search_kb_async was called with resolved internal KB ID integer string
            assert str(kb_id_int) in captured_search_kb_ids
    finally:
        db.close()
