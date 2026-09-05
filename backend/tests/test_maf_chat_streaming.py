"""
Pytest suite for Phase 2: Real-Time Streaming, SSE Protocol & HITL Middleware.
Covers:
1. Streaming Output & SSE Formatting Test
2. HITL Approval Middleware Interception Test (Approve & Deny)
3. Dynamic Tool Proposal Approval Test
4. Client Disconnection & Cancellation Test
"""

import asyncio
import json
import pytest
from unittest.mock import AsyncMock, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from models import Base, User, LLMProvider, Agent, Session as SessionModel, HITLApproval, ToolProposal
from schemas import HITLRespondRequest
from agent_framework import FunctionInvocationContext
from routers.chat_router import (
    _stream_response,
    hitl_and_proposal_middleware,
    respond_hitl,
    approve_tool_proposal,
)


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    user = User(username="testuser", email="test@example.com", hashed_password="pw", role="user")
    session.add(user)
    session.commit()

    provider = LLMProvider(
        user_id=user.id,
        name="Mock Provider",
        provider_type="openai",
        api_key="sk-test",
        model_id="gpt-4o",
    )
    session.add(provider)
    session.commit()

    agent = Agent(
        user_id=user.id,
        name="Test Agent",
        provider_id=provider.id,
        model_id="gpt-4o",
        hitl_confirmation_tools_json=json.dumps(["sandbox_bash"]),
        allow_tool_creation=True,
    )
    session.add(agent)
    session.commit()

    chat_session = SessionModel(
        user_id=user.id,
        entity_type="agent",
        entity_id=agent.id,
        title="Test Stream Session",
    )
    session.add(chat_session)
    session.commit()

    yield session
    session.close()


class MockContentText:
    type = "text"
    text = "Hello token"


class MockContentReasoning:
    type = "text_reasoning"
    text = "Thinking token"


class MockContentFunctionCall:
    type = "function_call"
    call_id = "call_123"
    name = "sandbox_bash"
    arguments = {"command": "echo hi"}


class MockContentFunctionResult:
    type = "function_result"
    call_id = "call_123"
    name = "sandbox_bash"
    result = "hi\n"


class MockUpdate:
    def __init__(self, contents):
        self.contents = contents


# 0. Test LLMMessage to MAF Message conversion and fallback properties
def test_llm_message_maf_conversion_and_properties():
    from llm.base import LLMMessage, to_maf_messages
    from agent_framework import Message

    # Test properties
    msg = LLMMessage(role="user", content="hello", tool_calls=["tc1"], tool_call_id="id1")
    assert msg.contents == ["hello"]
    assert msg.additional_properties == {"tool_calls": ["tc1"], "tool_call_id": "id1"}

    # Test contents setter
    msg.contents = ["new content"]
    assert msg.content == "new content"

    # Test conversion to MAF message
    maf_msg = msg.to_maf_message()
    assert isinstance(maf_msg, Message)
    assert maf_msg.role == "user"
    assert maf_msg.additional_properties == {"tool_calls": ["tc1"], "tool_call_id": "id1"}

    # Test batch conversion
    converted_list = to_maf_messages([msg, Message("assistant", ["hi"])])
    assert len(converted_list) == 2
    assert all(isinstance(m, Message) for m in converted_list)


# 1. Streaming Output & SSE Formatting Test
@pytest.mark.asyncio
async def test_streaming_output_and_sse_formatting(db_session):
    """Assert incremental text, reasoning deltas, and tool invocation markers match expected SSE format."""
    mock_llm = MagicMock()

    async def mock_stream():
        yield MockUpdate([MockContentReasoning()])
        yield MockUpdate([MockContentText()])
        yield MockUpdate([MockContentFunctionCall()])
        yield MockUpdate([MockContentFunctionResult()])

    class MockRunStream:
        def __aiter__(self):
            return mock_stream()

    mock_llm.run.return_value = MockRunStream()

    from llm.base import LLMMessage
    messages = [LLMMessage(role="user", content="Run bash echo hi")]
    provider_record = db_session.query(LLMProvider).first()
    agent = db_session.query(Agent).first()
    chat_session = db_session.query(SessionModel).first()

    events = []
    async for ev in _stream_response(
        llm=mock_llm,
        messages=messages,
        system_prompt="Test system prompt",
        db=db_session,
        session_id=chat_session.id,
        agent_id=agent.id,
        provider_record=provider_record,
        start_time=1000.0,
        agent=agent,
    ):
        events.append(ev)

    event_types = [e["event"] for e in events]
    assert mock_llm.run.call_args.kwargs["options"] == {"instructions": "Test system prompt"}
    assert mock_llm.run.call_args.kwargs.get("middleware") is None
    assert "reasoning_delta" in event_types
    assert "content_delta" in event_types
    assert "tool_call" in event_types
    assert "message_complete" in event_types
    assert "done" in event_types

    reasoning_ev = next(e for e in events if e["event"] == "reasoning_delta")
    assert json.loads(reasoning_ev["data"])["content"] == "Thinking token"

    content_ev = next(e for e in events if e["event"] == "content_delta")
    assert json.loads(content_ev["data"])["content"] == "Hello token"

    tool_evs = [e for e in events if e["event"] == "tool_call"]
    assert len(tool_evs) >= 2
    assert json.loads(tool_evs[0]["data"])["status"] == "running"
    assert json.loads(tool_evs[1]["data"])["status"] == "completed"


# 2. HITL Approval Middleware Interception Test (Approve & Deny)
@pytest.mark.asyncio
async def test_hitl_approval_middleware_interception_approve_and_deny(db_session):
    """
    Configure agent with hitl_confirmation_tools_json=['sandbox_bash'].
    Trigger tool call to sandbox_bash, assert middleware emits hitl_required and records pending in DB.
    Submit approval -> assert stream resumes.
    Submit rejection -> assert graceful denial message returned without execution.
    """
    agent = db_session.query(Agent).first()
    chat_session = db_session.query(SessionModel).first()
    user = db_session.query(User).first()

    event_queue = asyncio.Queue()

    mock_func = MagicMock()
    mock_func.name = "sandbox_bash"

    ctx = MagicMock(spec=FunctionInvocationContext)
    ctx.function = mock_func
    ctx.arguments = {"command": "rm -rf /"}
    ctx.kwargs = {
        "session_id": str(chat_session.id),
        "db": db_session,
        "agent": agent,
        "event_queue": event_queue,
        "tool_call_id": "call_hitl_123",
    }

    async def mock_call_next():
        return "Executed sandbox_bash successfully"

    # --- Test Approval Path ---
    async def approve_task():
        await asyncio.sleep(0.01)

        approval = db_session.query(HITLApproval).filter(
            HITLApproval.session_id == chat_session.id,
            HITLApproval.tool_call_id == "call_hitl_123",
        ).first()
        assert approval is not None
        assert approval.status == "pending"

        token_data = MagicMock(user_id=str(user.id))
        res = await respond_hitl(
            approval_id=str(approval.id),
            request=HITLRespondRequest(status="approved"),
            current_user=token_data,
            db=db_session,
        )
        assert res["status"] == "approved"

    task = asyncio.create_task(approve_task())
    result = await hitl_and_proposal_middleware(ctx, mock_call_next)
    await task

    assert result == "Executed sandbox_bash successfully"

    # --- Test Rejection / Denial Path ---
    ctx.kwargs["tool_call_id"] = "call_hitl_456"
    ctx.kwargs["event_queue"] = asyncio.Queue()

    async def deny_task():
        await asyncio.sleep(0.01)

        approval = db_session.query(HITLApproval).filter(
            HITLApproval.session_id == chat_session.id,
            HITLApproval.tool_call_id == "call_hitl_456",
        ).first()
        assert approval is not None
        assert approval.status == "pending"

        token_data = MagicMock(user_id=str(user.id))
        res = await respond_hitl(
            approval_id=str(approval.id),
            request=HITLRespondRequest(status="denied"),
            current_user=token_data,
            db=db_session,
        )
        assert res["status"] == "denied"

    deny_t = asyncio.create_task(deny_task())
    deny_result = await hitl_and_proposal_middleware(ctx, mock_call_next)
    await deny_t

    assert "User denied execution" in deny_result


# 3. Dynamic Tool Proposal Approval Test
@pytest.mark.asyncio
async def test_dynamic_tool_proposal_approval(db_session):
    """
    Trigger create_tool call; assert MAF middleware intercepts, yields tool_proposal_required,
    and dynamically registers new FunctionTool upon approval.
    """
    agent = db_session.query(Agent).first()
    chat_session = db_session.query(SessionModel).first()
    user = db_session.query(User).first()

    event_queue = asyncio.Queue()

    mock_func = MagicMock()
    mock_func.name = "create_tool"

    ctx = MagicMock(spec=FunctionInvocationContext)
    ctx.function = mock_func
    ctx.arguments = {
        "name": "reverse_string",
        "description": "Reverses input string",
        "handler_type": "python",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
        "handler_config": {"code": "def handler(params):\n    return params.get('text', '')[::-1]"},
    }
    ctx.kwargs = {
        "session_id": str(chat_session.id),
        "db": db_session,
        "agent": agent,
        "event_queue": event_queue,
        "tool_call_id": "call_prop_123",
    }
    added_tools = []
    ctx.add_tools = lambda ft: added_tools.append(ft)

    async def mock_call_next():
        return "Not called for create_tool"

    async def approve_proposal_task():
        await asyncio.sleep(0.01)

        proposal = db_session.query(ToolProposal).filter(
            ToolProposal.session_id == chat_session.id,
            ToolProposal.tool_call_id == "call_prop_123",
        ).first()
        assert proposal is not None
        assert proposal.status == "pending"

        token_data = MagicMock(user_id=str(user.id))
        res = await approve_tool_proposal(
            session_id=str(chat_session.id),
            proposal_id=str(proposal.id),
            current_user=token_data,
            db=db_session,
        )
        assert res["status"] == "approved"

    task = asyncio.create_task(approve_proposal_task())
    res_msg = await hitl_and_proposal_middleware(ctx, mock_call_next)
    await task

    assert "was approved and saved to the toolkit" in res_msg
    assert len(added_tools) == 1
    assert added_tools[0].name == "reverse_string"


# 4. Client Disconnection & Cancellation Test
@pytest.mark.asyncio
async def test_client_disconnection_and_cancellation(db_session):
    """
    Assert that when raw_request.is_disconnected() returns True mid-stream,
    the execution task raises asyncio.CancelledError cleanly.
    """
    mock_llm = MagicMock()

    async def infinite_stream():
        yield MockUpdate([MockContentText()])
        await asyncio.sleep(10.0)

    class MockRunStream:
        def __aiter__(self):
            return infinite_stream()

    mock_llm.run.return_value = MockRunStream()

    mock_raw_request = MagicMock()
    mock_raw_request.is_disconnected = AsyncMock(return_value=True)

    provider_record = db_session.query(LLMProvider).first()
    agent = db_session.query(Agent).first()
    chat_session = db_session.query(SessionModel).first()

    from llm.base import LLMMessage
    stream_gen = _stream_response(
        llm=mock_llm,
        messages=[LLMMessage(role="user", content="hello")],
        system_prompt="System",
        db=db_session,
        session_id=chat_session.id,
        agent_id=agent.id,
        provider_record=provider_record,
        start_time=1000.0,
        agent=agent,
        raw_request=mock_raw_request,
    )

    with pytest.raises(asyncio.CancelledError):
        async for _ in stream_gen:
            pass


class MockContentFunctionResultNoName:
    type = "function_result"
    call_id = "call_none_name_123"
    name = None
    result = "ok\n"


class MockContentFunctionCallWithName:
    type = "function_call"
    call_id = "call_none_name_123"
    name = "http_request_2"
    arguments = {}


@pytest.mark.asyncio
async def test_maf_streaming_function_result_with_none_name(db_session):
    """
    Test that when MAF streams a FunctionResultContent with name=None,
    the tool_name_map maps call_id back to the function_call name ('http_request_2'),
    and exports trace_spans without a NULL constraint failure.
    """
    from tracing_service import set_trace_provider, DatabaseTraceProvider
    set_trace_provider(DatabaseTraceProvider(db=db_session))

    mock_llm = MagicMock()

    async def mock_stream():
        yield MockUpdate([MockContentFunctionCallWithName()])
        yield MockUpdate([MockContentFunctionResultNoName()])

    class MockRunStream:
        def __aiter__(self):
            return mock_stream()

    mock_llm.run.return_value = MockRunStream()

    from llm.base import LLMMessage
    messages = [LLMMessage(role="user", content="Call http_request_2")]
    provider_record = db_session.query(LLMProvider).first()
    agent = db_session.query(Agent).first()
    chat_session = db_session.query(SessionModel).first()

    events = []
    async for ev in _stream_response(
        llm=mock_llm,
        messages=messages,
        system_prompt="Test system prompt",
        db=db_session,
        session_id=chat_session.id,
        agent_id=agent.id,
        provider_record=provider_record,
        start_time=1000.0,
        agent=agent,
    ):
        events.append(ev)

    # Wait briefly for background trace export asyncio tasks to finish
    await asyncio.sleep(0.05)

    from models import TraceSpan
    spans = db_session.query(TraceSpan).filter(TraceSpan.session_id == chat_session.id).all()
    tool_spans = [s for s in spans if s.span_type == "tool_call"]
    assert len(tool_spans) >= 2
    for s in tool_spans:
        assert s.name == "http_request_2"
