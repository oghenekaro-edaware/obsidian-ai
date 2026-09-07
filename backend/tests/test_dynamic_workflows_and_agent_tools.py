import os
os.environ["DATABASE_TYPE"] = "sqlite"

import config
config.DATABASE_TYPE = "sqlite"

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from database import Base, engine, SessionLocal
from models import User, LLMProvider, Agent, Workflow, WorkflowRun, ToolDefinition, Session as SessionModel
from services.tool_executor import ToolExecutor, ToolRuntimeContext
from builtin_tools import execute_builtin_tool, is_builtin_tool
from routers.workflow_runs_router import _run_workflow_sqlite
from schemas import WorkflowRunRequest


@pytest.fixture
def db_session():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_data(db_session):
    # Setup test user and provider
    user = db_session.query(User).filter(User.username == "test_wf_user").first()
    if not user:
        user = User(username="test_wf_user", email="test_wf@example.com", hashed_password="pw", role="user")
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

    provider = db_session.query(LLMProvider).filter(LLMProvider.user_id == user.id).first()
    if not provider:
        provider = LLMProvider(user_id=user.id, provider_type="openai", name="OpenAI", api_key="sk-test", model_id="gpt-4o")
        db_session.add(provider)
        db_session.commit()
        db_session.refresh(provider)

    agent = db_session.query(Agent).filter(Agent.user_id == user.id, Agent.name == "Worker Agent").first()
    if not agent:
        agent = Agent(user_id=user.id, name="Worker Agent", description="Worker", system_prompt="You are a worker.", model_id="gpt-4o", provider_id=provider.id)
        db_session.add(agent)
        db_session.commit()
        db_session.refresh(agent)

    return {"user": user, "provider": provider, "agent": agent}


@pytest.mark.asyncio
async def test_run_workflow_sqlite_non_agent_nodes(db_session, test_data):
    """Test that _run_workflow_sqlite handles start/end/non-agent steps without TypeError on agent_id=None."""
    user = test_data["user"]
    agent = test_data["agent"]

    steps = [
        {"id": "node_1", "order": 1, "node_type": "start", "task": "Start workflow", "agent_id": None},
        {"id": "node_2", "order": 2, "node_type": "agent", "task": "Process data", "agent_id": str(agent.id), "depends_on": ["node_1"]},
        {"id": "node_3", "order": 3, "node_type": "end", "task": "End workflow", "agent_id": None, "depends_on": ["node_2"]},
    ]

    workflow = Workflow(
        user_id=user.id,
        name="Mixed Workflow",
        steps_json=json.dumps(steps),
        is_active=True,
    )
    db_session.add(workflow)
    db_session.commit()
    db_session.refresh(workflow)

    class MockUser:
        user_id = str(user.id)

    req = WorkflowRunRequest(input="Hello World")

    # Call _run_workflow_sqlite which previously raised TypeError: int() argument must be...
    res = await _run_workflow_sqlite(str(workflow.id), req, MockUser(), db_session)
    assert res is not None


@pytest.mark.asyncio
async def test_builtin_tool_create_dynamic_tool(db_session, test_data):
    user = test_data["user"]
    agent = test_data["agent"]
    executor = ToolExecutor()
    context = ToolRuntimeContext(session_id="test_session", agent=agent, db=db_session)

    assert is_builtin_tool("create_dynamic_tool") is True

    tool_args = {
        "name": "custom_python_calc",
        "description": "Custom calculator tool",
        "handler_type": "python",
        "parameters": {
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
        },
        "handler_config": {"code": "def handler(args):\n    return args.get('x', 0) + args.get('y', 0)"},
    }

    result = await executor.execute(
        tool_name="create_dynamic_tool",
        arguments=tool_args,
        runtime=context,
    )

    assert result.error is None
    res_data = json.loads(result.output)
    assert res_data["status"] == "success"
    assert res_data["name"] == "custom_python_calc"

    # Verify tool definition created in DB
    tool_rec = db_session.query(ToolDefinition).filter(
        ToolDefinition.name == "custom_python_calc", ToolDefinition.user_id == user.id
    ).first()
    assert tool_rec is not None
    assert tool_rec.handler_type == "python"


@pytest.mark.asyncio
async def test_builtin_tool_create_agent(db_session, test_data):
    user = test_data["user"]
    agent = test_data["agent"]
    executor = ToolExecutor()
    context = ToolRuntimeContext(session_id="test_session", agent=agent, db=db_session)

    assert is_builtin_tool("create_agent") is True

    agent_args = {
        "name": "Dynamic Analyst",
        "role": "Data Analyst",
        "system_prompt": "Analyze incoming dataset.",
        "model_id": "gpt-4o",
    }

    result = await executor.execute(
        tool_name="create_agent",
        arguments=agent_args,
        runtime=context,
    )

    assert result.error is None
    res_data = json.loads(result.output)
    assert res_data["status"] == "success"
    assert res_data["name"] == "Dynamic Analyst"

    created_ag = db_session.query(Agent).filter(
        Agent.name == "Dynamic Analyst", Agent.user_id == user.id
    ).first()
    assert created_ag is not None
    assert created_ag.description == "Data Analyst"


@pytest.mark.asyncio
async def test_builtin_tool_create_and_execute_workflow(db_session, test_data):
    user = test_data["user"]
    agent = test_data["agent"]
    executor = ToolExecutor()
    context = ToolRuntimeContext(session_id="test_session", agent=agent, db=db_session)

    assert is_builtin_tool("create_workflow") is True
    assert is_builtin_tool("execute_workflow") is True

    wf_args = {
        "name": "Dynamic Pipeline",
        "description": "Pipeline spawned by AI agent",
        "steps": [
            {"node_type": "start", "task": "Start node"},
            {"node_type": "agent", "agent_name": "Worker Agent", "task": "Summarize input"},
            {"node_type": "end", "task": "End node"},
        ],
    }

    create_res = await executor.execute(
        tool_name="create_workflow",
        arguments=wf_args,
        runtime=context,
    )

    assert create_res.error is None
    c_data = json.loads(create_res.output)
    assert c_data["status"] == "success"
    wf_id = c_data["workflow_id"]

    # Mock execute_dag to test execute_workflow tool
    mock_events = [
        {"event": "workflow_start", "run_id": "1", "workflow_name": "Dynamic Pipeline", "total_steps": 3},
        {"event": "node_complete", "node_id": "step_1", "agent_name": "Start", "output": "Hello"},
        {"event": "node_complete", "node_id": "step_2", "agent_name": "Worker Agent", "output": "Summary of Hello"},
        {"event": "node_complete", "node_id": "step_3", "agent_name": "End", "output": "Summary of Hello"},
        {
            "event": "workflow_done",
            "status": "completed",
            "outputs": {"step_3": "Summary of Hello"},
            "step_results": [],
        },
    ]

    async def mock_dag_gen(*args, **kwargs):
        for ev in mock_events:
            yield ev

    with patch("dag_executor.execute_dag", side_effect=mock_dag_gen):
        exec_res = await executor.execute(
            tool_name="execute_workflow",
            arguments={"workflow_id_or_name": "Dynamic Pipeline", "input_text": "Hello World"},
            runtime=context,
        )

        assert exec_res.error is None
        e_data = json.loads(exec_res.output)
        assert e_data["status"] == "completed"
        assert e_data["final_output"] == "Summary of Hello"
