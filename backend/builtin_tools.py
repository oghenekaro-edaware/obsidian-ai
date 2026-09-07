"""
Built-in tool implementations decorated with MAF @ai_function / @tool.

Provides:
  - web_search   : Tavily Search API
  - calculator   : Safe mathematical expression evaluation
  - weather      : Weather lookup
  - time         : Current time in specified timezone
  - fetch_url    : HTTP GET with response text extraction
"""

import asyncio
import json
import re
from datetime import datetime
from html.parser import HTMLParser
import zoneinfo

from agent_framework import tool, FunctionTool

# Alias for directive compliance
ai_function = tool


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """Strip HTML tags and extract visible text."""

    SKIP_TAGS = {"script", "style", "noscript", "head", "meta", "link"}

    def __init__(self):
        super().__init__()
        self._skip = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() in self.SKIP_TAGS:
            self._skip += 1

    def handle_endtag(self, tag):
        if tag.lower() in self.SKIP_TAGS:
            self._skip = max(0, self._skip - 1)

    def handle_data(self, data):
        if not self._skip:
            text = data.strip()
            if text:
                self.parts.append(text)


def _strip_html(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception:
        pass
    text = " ".join(parser.parts)
    text = re.sub(r"\s{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Tool Definitions with MAF @ai_function
# ---------------------------------------------------------------------------

@ai_function
async def web_search(query: str, max_results: int = 8) -> str:
    """Search the web for current information. Returns a list of results with titles, snippets, and URLs."""
    import httpx
    import os

    max_results = min(max(1, max_results), 20)
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        return json.dumps({
            "query": query,
            "results": [],
            "note": "TAVILY_API_KEY not set. Add it to backend/.env to enable web search.",
        })

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                    "include_answer": False,
                    "include_raw_content": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return json.dumps({"query": query, "results": [], "note": f"Search failed: {e}"})

    results = [
        {
            "title": r.get("title", ""),
            "snippet": r.get("content", ""),
            "url": r.get("url", ""),
        }
        for r in data.get("results", [])
    ]

    if not results:
        return json.dumps({"query": query, "results": [], "note": "No results found."})

    return json.dumps({"query": query, "results": results})


import ast
import math
import operator

_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": pow,
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
}

_ALLOWED_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval_ast(node):
    if isinstance(node, ast.Expression):
        return _safe_eval_ast(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")
    elif isinstance(node, ast.Name):
        if node.id in _ALLOWED_CONSTANTS:
            return _ALLOWED_CONSTANTS[node.id]
        raise ValueError(f"Name '{node.id}' is not allowed")
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type in _ALLOWED_OPERATORS:
            return _ALLOWED_OPERATORS[op_type](_safe_eval_ast(node.operand))
        raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type in _ALLOWED_OPERATORS:
            left = _safe_eval_ast(node.left)
            right = _safe_eval_ast(node.right)
            return _ALLOWED_OPERATORS[op_type](left, right)
        raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _ALLOWED_FUNCTIONS:
            args = [_safe_eval_ast(arg) for arg in node.args]
            return _ALLOWED_FUNCTIONS[node.func.id](*args)
        raise ValueError("Unsupported function call")
    else:
        raise ValueError(f"Unsupported expression syntax: {type(node).__name__}")


@ai_function
def calculator(expression: str) -> str:
    """Safely evaluate a mathematical expression using an AST parser and return the result."""
    try:
        clean_expr = expression.strip()
        parsed = ast.parse(clean_expr, mode="eval")
        result = _safe_eval_ast(parsed)
        return json.dumps({"expression": expression, "result": str(result)})
    except Exception as e:
        return json.dumps({"expression": expression, "error": f"Calculation error: {e}"})


@ai_function
def weather(location: str) -> str:
    """Get current weather details for a given location."""
    loc_clean = location.strip()
    if not loc_clean:
        return json.dumps({"error": "Location parameter required"})
    # Mock / standard structured weather response
    return json.dumps({
        "location": loc_clean,
        "temperature": "22°C",
        "condition": "Partly Cloudy",
        "humidity": "55%",
        "wind": "12 km/h",
    })


@ai_function
def time(timezone: str = "UTC") -> str:
    """Get current time for a given IANA timezone string (default 'UTC')."""
    tz_str = timezone.strip() or "UTC"
    try:
        tz = zoneinfo.ZoneInfo(tz_str)
        now = datetime.now(tz)
        return json.dumps({"timezone": tz_str, "current_time": now.isoformat()})
    except Exception:
        now = datetime.now(zoneinfo.ZoneInfo("UTC"))
        return json.dumps({"timezone": "UTC", "current_time": now.isoformat(), "note": f"Unknown timezone '{tz_str}', fell back to UTC"})


def _rewrite_github_url(url: str) -> tuple[str, str | None]:
    m = re.match(r"https?://(?:www\.)?github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if m:
        owner, repo = m.group(1), m.group(2)
        return f"https://api.github.com/repos/{owner}/{repo}", f"https://api.github.com/repos/{owner}/{repo}/readme"

    m = re.match(r"https?://(?:www\.)?github\.com/([^/]+)/([^/]+)/blob/(.+)", url)
    if m:
        owner, repo, path = m.group(1), m.group(2), m.group(3)
        return f"https://raw.githubusercontent.com/{owner}/{repo}/{path}", None

    return url, None


@ai_function
async def fetch_url(url: str, max_chars: int = 8000) -> str:
    """Fetch the text content of a URL."""
    max_chars = min(max(500, max_chars), 50000)

    import httpx
    import base64

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/json,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    rewritten_url, readme_url = _rewrite_github_url(url)
    is_github_repo = readme_url is not None

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=20.0, headers=headers) as client:
            resp = await client.get(rewritten_url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            raw = resp.text

            readme_text = ""
            if is_github_repo and readme_url:
                try:
                    readme_resp = await client.get(readme_url)
                    if readme_resp.status_code == 200:
                        readme_data = readme_resp.json()
                        encoded = readme_data.get("content", "")
                        readme_text = base64.b64decode(encoded).decode("utf-8", errors="replace")
                except Exception:
                    pass
    except Exception as e:
        return json.dumps({"error": f"Fetch failed: {e}", "url": url})

    if "html" in content_type:
        text = _strip_html(raw)
    elif "json" in content_type:
        try:
            text = json.dumps(json.loads(raw), indent=2)
        except Exception:
            text = raw
    else:
        text = raw

    combined = text + "\n\n--- README ---\n\n" + readme_text if readme_text else text
    if len(combined) > max_chars:
        combined = combined[:max_chars] + f"\n\n[truncated — {len(combined) - max_chars} more characters]"

    return json.dumps({"url": url, "content": combined})


def _get_context_db(runtime):
    db = getattr(runtime, "db", None) if runtime else None
    mongo_db = getattr(runtime, "mongo_db", None) if runtime else None
    if not db and not mongo_db:
        from config import DATABASE_TYPE
        if DATABASE_TYPE == "mongo":
            from database_mongo import get_database
            mongo_db = get_database()
        else:
            from database import SessionLocal
            db = SessionLocal()
    return db, mongo_db


@ai_function
async def create_dynamic_tool(
    name: str,
    description: str,
    handler_type: str,
    parameters: dict = None,
    handler_config: dict = None,
    **kwargs,
) -> str:
    """Create or update a custom tool definition (Python code or HTTP endpoint) that can be used by AI agents and workflows."""
    runtime = kwargs.get("runtime")
    db, mongo_db = _get_context_db(runtime)

    params_json = json.dumps(parameters) if isinstance(parameters, dict) else (parameters or "{}")
    config_json = json.dumps(handler_config) if isinstance(handler_config, dict) else (handler_config or "{}")

    if mongo_db is not None:
        from models_mongo import ToolDefinitionCollection
        user_id = str(runtime.agent.get("user_id")) if runtime and hasattr(runtime, "agent") and isinstance(runtime.agent, dict) and runtime.agent.get("user_id") else "1"
        existing = await mongo_db[ToolDefinitionCollection.collection_name].find_one({"name": name, "user_id": user_id, "is_active": True})
        if existing:
            await ToolDefinitionCollection.update(mongo_db, str(existing["_id"]), user_id, {
                "description": description,
                "handler_type": handler_type,
                "parameters_json": params_json,
                "handler_config": config_json,
                "is_model_created": True,
            })
            tool_id = str(existing["_id"])
        else:
            created = await ToolDefinitionCollection.create(mongo_db, {
                "user_id": user_id,
                "name": name,
                "description": description,
                "handler_type": handler_type,
                "parameters_json": params_json,
                "handler_config": config_json,
                "is_model_created": True,
            })
            tool_id = str(created["_id"])
        return json.dumps({"status": "success", "tool_id": tool_id, "name": name, "message": f"Tool '{name}' created/updated successfully."})

    from models import ToolDefinition, User
    user_id = 1
    if runtime and getattr(runtime, "agent", None):
        user_id = getattr(runtime.agent, "user_id", 1) or 1
    else:
        first_user = db.query(User).first()
        if first_user:
            user_id = first_user.id

    existing = db.query(ToolDefinition).filter(
        ToolDefinition.name == name,
        ToolDefinition.user_id == user_id,
        ToolDefinition.is_active == True,
    ).first()

    if existing:
        existing.description = description
        existing.handler_type = handler_type
        existing.parameters_json = params_json
        existing.handler_config = config_json
        existing.is_model_created = True
        db.commit()
        db.refresh(existing)
        tool_id = str(existing.id)
    else:
        tool = ToolDefinition(
            user_id=user_id,
            name=name,
            description=description,
            handler_type=handler_type,
            parameters_json=params_json,
            handler_config=config_json,
            is_active=True,
            is_model_created=True,
        )
        db.add(tool)
        db.commit()
        db.refresh(tool)
        tool_id = str(tool.id)

    return json.dumps({"status": "success", "tool_id": tool_id, "name": name, "message": f"Tool '{name}' created/updated successfully."})


@ai_function
async def create_agent(
    name: str,
    description: str = "",
    system_prompt: str = "",
    model_id: str = "gpt-4o",
    tool_names: list[str] = None,
    role: str = None,
    **kwargs,
) -> str:
    """Create a new AI agent with a specific role/description, system prompt, model, and assigned tools."""
    runtime = kwargs.get("runtime")
    db, mongo_db = _get_context_db(runtime)

    agent_desc = description or role or ""

    if mongo_db is not None:
        from models_mongo import AgentCollection, ToolDefinitionCollection
        user_id = str(runtime.agent.get("user_id")) if runtime and hasattr(runtime, "agent") and isinstance(runtime.agent, dict) and runtime.agent.get("user_id") else "1"
        provider = await mongo_db["llm_providers"].find_one({"is_active": True}) or await mongo_db["llm_providers"].find_one({})
        provider_id = str(provider["_id"]) if provider else None

        tool_ids = []
        if tool_names:
            for tn in tool_names:
                td = await mongo_db[ToolDefinitionCollection.collection_name].find_one({"name": tn, "is_active": True})
                if td:
                    tool_ids.append(str(td["_id"]))

        created = await AgentCollection.create(mongo_db, {
            "user_id": user_id,
            "name": name,
            "description": agent_desc,
            "system_prompt": system_prompt,
            "model_id": model_id or "gpt-4o",
            "provider_id": provider_id,
            "tools_json": json.dumps(tool_ids),
        })
        return json.dumps({"status": "success", "agent_id": str(created["_id"]), "name": name, "description": agent_desc, "message": f"Agent '{name}' created successfully."})

    from models import Agent, LLMProvider, ToolDefinition, User
    user_id = 1
    if runtime and getattr(runtime, "agent", None):
        user_id = getattr(runtime.agent, "user_id", 1) or 1
    else:
        first_user = db.query(User).first()
        if first_user:
            user_id = first_user.id

    provider = db.query(LLMProvider).filter(LLMProvider.user_id == user_id).first() or db.query(LLMProvider).first()
    provider_id = provider.id if provider else None

    tool_ids = []
    if tool_names:
        tools_defs = db.query(ToolDefinition).filter(ToolDefinition.name.in_(tool_names), ToolDefinition.is_active == True).all()
        tool_ids = [td.id for td in tools_defs]

    agent = Agent(
        user_id=user_id,
        name=name,
        description=agent_desc,
        system_prompt=system_prompt,
        model_id=model_id or "gpt-4o",
        provider_id=provider_id,
        tools_json=json.dumps(tool_ids),
        is_active=True,
    )
    db.add(agent)
    db.commit()
    db.refresh(agent)

    return json.dumps({"status": "success", "agent_id": str(agent.id), "name": name, "description": agent_desc, "message": f"Agent '{name}' created successfully."})


@ai_function
async def create_workflow(
    name: str,
    description: str = "",
    steps: list[dict] = None,
    **kwargs,
) -> str:
    """Create or spawn a dynamic workflow (linear or DAG graph with steps/nodes) connecting agents, tools, and tasks."""
    runtime = kwargs.get("runtime")
    db, mongo_db = _get_context_db(runtime)

    steps = steps or []
    if not steps:
        return json.dumps({"error": "steps parameter cannot be empty"})

    processed_steps = []
    for idx, s in enumerate(steps):
        s_copy = dict(s)
        s_copy["order"] = s_copy.get("order", idx + 1)
        s_copy["node_type"] = s_copy.get("node_type", "agent")
        s_copy["task"] = s_copy.get("task", "")
        s_copy["id"] = s_copy.get("id") or f"step_{s_copy['order']}"

        agent_name = s_copy.get("agent_name")
        if agent_name and not s_copy.get("agent_id"):
            if mongo_db is not None:
                ag = await mongo_db["agents"].find_one({"name": agent_name, "is_active": True})
                if ag:
                    s_copy["agent_id"] = str(ag["_id"])
            elif db is not None:
                from models import Agent
                ag = db.query(Agent).filter(Agent.name == agent_name, Agent.is_active == True).first()
                if ag:
                    s_copy["agent_id"] = str(ag.id)
        processed_steps.append(s_copy)

    steps_str = json.dumps(processed_steps)

    if mongo_db is not None:
        from models_mongo import WorkflowCollection
        user_id = str(runtime.agent.get("user_id")) if runtime and hasattr(runtime, "agent") and isinstance(runtime.agent, dict) and runtime.agent.get("user_id") else "1"
        created = await WorkflowCollection.create(mongo_db, {
            "user_id": user_id,
            "name": name,
            "description": description or "",
            "steps_json": steps_str,
        })
        return json.dumps({"status": "success", "workflow_id": str(created["_id"]), "name": name, "steps_count": len(processed_steps), "message": f"Workflow '{name}' created successfully."})

    from models import Workflow, User
    user_id = 1
    if runtime and getattr(runtime, "agent", None):
        user_id = getattr(runtime.agent, "user_id", 1) or 1
    else:
        first_user = db.query(User).first()
        if first_user:
            user_id = first_user.id

    workflow = Workflow(
        user_id=user_id,
        name=name,
        description=description or "",
        steps_json=steps_str,
        is_active=True,
    )
    db.add(workflow)
    db.commit()
    db.refresh(workflow)

    return json.dumps({"status": "success", "workflow_id": str(workflow.id), "name": name, "steps_count": len(processed_steps), "message": f"Workflow '{name}' created successfully."})


@ai_function
async def execute_workflow(
    workflow_id_or_name: str,
    input_text: str,
    **kwargs,
) -> str:
    """Execute an existing workflow by ID or name with input text, and return the final execution result."""
    runtime = kwargs.get("runtime")
    db, mongo_db = _get_context_db(runtime)

    workflow_obj = None
    if mongo_db is not None:
        from models_mongo import WorkflowCollection
        if len(workflow_id_or_name) == 24:
            workflow_obj = await WorkflowCollection.find_by_id(mongo_db, workflow_id_or_name)
        if not workflow_obj:
            workflow_obj = await mongo_db[WorkflowCollection.collection_name].find_one({"name": workflow_id_or_name, "is_active": True})
    else:
        from models import Workflow
        if workflow_id_or_name.isdigit():
            workflow_obj = db.query(Workflow).filter(Workflow.id == int(workflow_id_or_name), Workflow.is_active == True).first()
        if not workflow_obj:
            workflow_obj = db.query(Workflow).filter(Workflow.name == workflow_id_or_name, Workflow.is_active == True).first()

    if not workflow_obj:
        return json.dumps({"error": f"Workflow '{workflow_id_or_name}' not found"})

    wf_name = workflow_obj.get("name") if isinstance(workflow_obj, dict) else workflow_obj.name
    wf_id = str(workflow_obj.get("_id") if isinstance(workflow_obj, dict) else workflow_obj.id)
    steps_raw = workflow_obj.get("steps_json") if isinstance(workflow_obj, dict) else workflow_obj.steps_json
    steps = json.loads(steps_raw) if isinstance(steps_raw, str) else (steps_raw or [])

    if not steps:
        return json.dumps({"error": f"Workflow '{wf_name}' has no steps"})

    # Ensure every step has id and depends_on
    for idx, s in enumerate(steps):
        s.setdefault("id", f"node_{idx+1}")
        if idx > 0 and not s.get("depends_on"):
            s["depends_on"] = [steps[idx-1]["id"]]

    from dag_executor import DagContext, execute_dag

    if mongo_db is not None:
        from models_mongo import AgentCollection, LLMProviderCollection
        from routers.workflow_runs_router import _build_tools_mongo, _load_mcp_configs_mongo, _execute_tool_mongo, _evaluate_condition_mongo, _create_llm_mongo

        async def _get_agent(agent_id: str):
            return await AgentCollection.find_by_id(mongo_db, agent_id)

        async def _get_provider(agent):
            if not agent.get("provider_id"):
                return None
            return await LLMProviderCollection.find_by_id(mongo_db, str(agent["provider_id"]))

        async def _build_tools_a(agent):
            return await _build_tools_mongo(agent, mongo_db)

        async def _load_mcp_a(agent):
            return await _load_mcp_configs_mongo(agent, mongo_db)

        async def _execute_native_tool(name, args):
            return await _execute_tool_mongo(name, args, mongo_db)

        async def _evaluate_condition_a(upstream, uinput, branches, prompt):
            return await _evaluate_condition_mongo(upstream, uinput, branches, prompt, mongo_db)

        async def _update(updates):
            pass

        ctx = DagContext(
            get_agent=_get_agent,
            get_provider=_get_provider,
            create_llm=_create_llm_mongo,
            build_tools=_build_tools_a,
            load_mcp_configs=_load_mcp_a,
            execute_native_tool=_execute_native_tool,
            evaluate_condition=_evaluate_condition_a,
            update_run=_update,
        )
    else:
        from models import Agent, LLMProvider
        from routers.workflow_runs_router import _build_tools, _load_mcp_configs, _execute_tool, _evaluate_condition, _create_llm

        async def _get_agent(agent_id: str):
            return db.query(Agent).filter(Agent.id == int(agent_id)).first()

        async def _get_provider(agent):
            if not agent.provider_id:
                return None
            return db.query(LLMProvider).filter(LLMProvider.id == agent.provider_id).first()

        async def _build_tools_a(agent):
            return _build_tools(agent, db)

        async def _load_mcp_a(agent):
            return _load_mcp_configs(agent, db)

        async def _execute_native_tool(name, args):
            return _execute_tool(name, args, db)

        async def _evaluate_condition_a(upstream, uinput, branches, prompt):
            return await _evaluate_condition(upstream, uinput, branches, prompt, db)

        async def _update(updates):
            pass

        ctx = DagContext(
            get_agent=_get_agent,
            get_provider=_get_provider,
            create_llm=_create_llm,
            build_tools=_build_tools_a,
            load_mcp_configs=_load_mcp_a,
            execute_native_tool=_execute_native_tool,
            evaluate_condition=_evaluate_condition_a,
            update_run=_update,
        )

    step_results = []
    final_output = ""
    try:
        async for ev in execute_dag(steps, wf_name, input_text, ctx):
            if ev["event"] == "workflow_done":
                step_results = ev.get("step_results", [])
                outputs = ev.get("outputs", {})
                sink_ids = [s["id"] for s in steps if not any(s["id"] in (st.get("depends_on") or []) for st in steps)]
                final_output = "\n\n".join(outputs.get(nid, "") for nid in sink_ids if outputs.get(nid))
                if not final_output and outputs:
                    final_output = "\n\n".join(v for v in outputs.values() if v)
    except Exception as e:
        return json.dumps({"error": f"Workflow execution failed: {e}"})

    return json.dumps({
        "status": "completed",
        "workflow_id": wf_id,
        "workflow_name": wf_name,
        "final_output": final_output,
        "steps": step_results,
    })


BUILTIN_TOOLS = {
    "web_search": web_search,
    "calculator": calculator,
    "weather": weather,
    "time": time,
    "fetch_url": fetch_url,
    "create_dynamic_tool": create_dynamic_tool,
    "create_agent": create_agent,
    "create_workflow": create_workflow,
    "execute_workflow": execute_workflow,
}

BUILTIN_TOOL_NAMES = set(BUILTIN_TOOLS.keys())


def is_builtin_tool(tool_name: str) -> bool:
    return tool_name in BUILTIN_TOOL_NAMES


async def execute_builtin_tool(tool_name: str, arguments_str: str, runtime=None) -> str:
    """Legacy/unified dispatcher executing builtin FunctionTool by name."""
    try:
        args = json.loads(arguments_str) if arguments_str else {}
    except json.JSONDecodeError:
        args = {}

    tool_obj = BUILTIN_TOOLS.get(tool_name)
    if not tool_obj:
        return json.dumps({"error": f"Unknown builtin tool: {tool_name}"})

    if tool_name in ("create_dynamic_tool", "create_agent", "create_workflow", "execute_workflow"):
        return await tool_obj.func(**args, runtime=runtime)

    res = await tool_obj.invoke(arguments=args)
    if isinstance(res, list) and len(res) > 0 and hasattr(res[0], "text"):
        return res[0].text
    return str(res)
