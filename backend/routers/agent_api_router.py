import json
import time
import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from config import DATABASE_TYPE
from database import get_db
from models import Agent, AgentVersion, AgentAPIConfig, Application, ApplicationAgentAccess, Schema, SchemaVersion, APIRequest, Message, KnowledgeBase, TraceSpan
from auth import get_current_user, get_application_api_key, TokenData, ApplicationKeyData
from schemas import AgentAPIConfigCreate, ExternalInvokeRequest, ExternalSessionCreate
from sqlalchemy import or_
from services.schema_validation_service import validate_json_schema

router = APIRouter(prefix="/api/v1", tags=["agent-api"])
STATES = {"draft", "testing", "published", "deprecated", "retired"}

def require_owner_agent(db, agent_id, user_id):
    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.user_id == int(user_id)).first()
    if not agent: raise HTTPException(404, "Agent not found")
    return agent

def owned_schema_version(db, version_id, user_id, direction):
    version = (
        db.query(SchemaVersion)
        .join(Schema, Schema.id == SchemaVersion.schema_id)
        .filter(
            SchemaVersion.id == int(version_id),
            Schema.user_id == int(user_id),
            Schema.direction == direction,
        )
        .first()
    )
    if not version:
        raise HTTPException(
            status_code=422,
            detail=f"{direction.capitalize()} schema version not found for this user",
        )
    return version

def config_for(db, agent_id):
    config = db.query(AgentAPIConfig).filter(AgentAPIConfig.agent_id == agent_id).first()
    if not config: raise HTTPException(404, "Agent is not exposed through the API")
    return config

def parse(value, fallback):
    try: return json.loads(value) if value else fallback
    except json.JSONDecodeError: return fallback

def assert_scope(key, required):
    if not set(required).issubset(set(key.scopes)):
        raise HTTPException(403, detail={"code": "INSUFFICIENT_SCOPE", "message": "API key lacks a required scope"})

def authorize_agent(db, key, agent_id, permission):
    config = config_for(db, agent_id)
    assert_scope(key, parse(config.required_scopes_json, [permission]))
    # Owning applications and explicit, non-revoked allowlist entries can use
    # the agent. Knowing an agent ID alone is never sufficient.
    if config.owner_application_id == int(key.application_id): return config
    grant = db.query(ApplicationAgentAccess).filter(ApplicationAgentAccess.application_id == int(key.application_id), ApplicationAgentAccess.agent_id == agent_id, ApplicationAgentAccess.revoked_at.is_(None)).first()
    if not grant or permission not in parse(grant.permissions_json, []):
        raise HTTPException(403, detail={"code": "AGENT_ACCESS_DENIED", "message": "Application is not allowed to access this agent"})
    return config

def application_session(db, key, agent_id, session_id):
    from models import Session as ChatSession
    try:
        query = db.query(ChatSession).filter(
            ChatSession.id == int(session_id),
            ChatSession.application_id == int(key.application_id),
            ChatSession.entity_type == "agent",
        )
        if agent_id:
            query = query.filter(ChatSession.entity_id == agent_id)
        session = query.first()
    except (TypeError, ValueError):
        session = None
    if not session:
        raise HTTPException(404, detail={"code": "SESSION_NOT_FOUND"})
    return session

@router.post("/agent-sessions/{agent_id}")
def create_external_session(
    agent_id: int,
    body: Optional[ExternalSessionCreate] = None,
    db: Session = Depends(get_db),
    key: ApplicationKeyData = Depends(get_application_api_key),
):
    authorize_agent(db, key, agent_id, "agent:invoke")
    from models import Session as ChatSession
    agent = db.get(Agent, agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    title = body.title if body and body.title else "API chat"

    if body and body.knowledge_base_ids is not None:
        requested_ids = {str(kb_id) for kb_id in body.knowledge_base_ids}
        matched_kbs = db.query(KnowledgeBase).filter(
            KnowledgeBase.is_active == True,
            or_(
                KnowledgeBase.id.in_([int(kb_id) for kb_id in requested_ids if kb_id.isdigit()]),
                KnowledgeBase.external_id.in_(list(requested_ids)),
            ),
        ).all()

        allowed_ids = set()
        for kb in matched_kbs:
            if str(kb.owner_id or kb.user_id) == str(agent.user_id) or kb.is_shared:
                allowed_ids.add(str(kb.id))
                if kb.external_id:
                    allowed_ids.add(kb.external_id)

        if not requested_ids.issubset(allowed_ids):
            raise HTTPException(status_code=403, detail={"code": "KNOWLEDGE_BASE_ACCESS_DENIED"})

    session = ChatSession(
        user_id=agent.user_id,
        application_id=int(key.application_id),
        title=title,
        entity_type="agent",
        entity_id=agent_id,
        knowledge_base_ids_json=json.dumps(body.knowledge_base_ids) if (body and body.knowledge_base_ids is not None) else None,
        system_instruction=body.system_instruction if (body and body.system_instruction is not None) else None,
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return {
        "id": str(session.id), "title": session.title, "entity_type": session.entity_type,
        "entity_id": str(session.entity_id), "is_active": session.is_active,
        "created_at": session.created_at, "updated_at": session.updated_at,
    }

@router.get("/agent-sessions/{session_id}")
def get_external_session(session_id: str, db: Session = Depends(get_db), key: ApplicationKeyData = Depends(get_application_api_key)):
    assert_scope(key, ["agent:read"])
    session = application_session(db, key, 0, session_id)
    return {
        "id": str(session.id), "title": session.title, "entity_type": session.entity_type,
        "entity_id": str(session.entity_id), "is_active": session.is_active,
        "created_at": session.created_at, "updated_at": session.updated_at,
    }

@router.get("/agent-sessions/{session_id}/messages")
def get_external_session_messages(session_id: str, limit: int = 100, offset: int = 0, db: Session = Depends(get_db), key: ApplicationKeyData = Depends(get_application_api_key)):
    assert_scope(key, ["agent:read"])
    session = application_session(db, key, 0, session_id)
    messages = db.query(Message).filter(Message.session_id == session.id).order_by(Message.created_at.asc()).offset(offset).limit(min(limit, 500)).all()
    return {"messages": [{
        "id": str(message.id), "session_id": str(message.session_id), "role": message.role,
        "content": message.content, "attachments": parse(message.attachments_json, None),
        "created_at": message.created_at,
    } for message in messages]}

@router.get("/agent-api-configs/{agent_id}")
@router.get("/agents/{agent_id}/api-config")
def get_agent_api_config(agent_id: int, db: Session = Depends(get_db), user: TokenData = Depends(get_current_user)):
    require_owner_agent(db, agent_id, user.user_id)
    config = db.query(AgentAPIConfig).filter(AgentAPIConfig.agent_id == agent_id).first()
    if not config:
        raise HTTPException(status_code=404, detail="Agent API configuration not found")
    published = db.get(AgentVersion, config.published_version_id) if config.published_version_id else None
    return {
        "agent_id": str(agent_id),
        "owner_application_id": str(config.owner_application_id) if config.owner_application_id else None,
        "publication_state": config.publication_state,
        "agent_version": published.version_number if published else None,
        "input_schema_version_id": str(config.input_schema_version_id) if config.input_schema_version_id else None,
        "output_schema_version_id": str(config.output_schema_version_id) if config.output_schema_version_id else None,
        "required_scopes": parse(config.required_scopes_json, []),
        "rate_limit": config.rate_limit,
    }

@router.put("/agents/{agent_id}/api-config")
def configure_agent_api(agent_id: int, body: AgentAPIConfigCreate, db: Session = Depends(get_db), user: TokenData = Depends(get_current_user)):
    require_owner_agent(db, agent_id, user.user_id)
    if body.input_schema_version_id:
        owned_schema_version(db, body.input_schema_version_id, user.user_id, "input")
    if body.output_schema_version_id:
        owned_schema_version(db, body.output_schema_version_id, user.user_id, "output")
    if body.owner_application_id:
        app = db.query(Application).filter(Application.id == int(body.owner_application_id), Application.user_id == int(user.user_id)).first()
        if not app: raise HTTPException(422, "Application must belong to the agent owner")
    config = db.query(AgentAPIConfig).filter(AgentAPIConfig.agent_id == agent_id).first()
    fields = dict(owner_application_id=int(body.owner_application_id) if body.owner_application_id else None, input_schema_version_id=int(body.input_schema_version_id) if body.input_schema_version_id else None, output_schema_version_id=int(body.output_schema_version_id) if body.output_schema_version_id else None, required_scopes_json=json.dumps(body.required_scopes), rate_limit=body.rate_limit)
    if config:
        for k, v in fields.items(): setattr(config, k, v)
    else: config = AgentAPIConfig(agent_id=agent_id, **fields); db.add(config)
    db.commit(); db.refresh(config)
    return {"agent_id": str(agent_id), "publication_state": config.publication_state, "input_schema_version_id": str(config.input_schema_version_id) if config.input_schema_version_id else None, "output_schema_version_id": str(config.output_schema_version_id) if config.output_schema_version_id else None}

@router.post("/agents/{agent_id}/publish")
def publish_agent(agent_id: int, db: Session = Depends(get_db), user: TokenData = Depends(get_current_user)):
    agent = require_owner_agent(db, agent_id, user.user_id); config = config_for(db, agent_id)
    if not config.input_schema_version_id or not config.output_schema_version_id: raise HTTPException(422, "Published API agents require input and output schemas")
    owned_schema_version(db, config.input_schema_version_id, user.user_id, "input")
    owned_schema_version(db, config.output_schema_version_id, user.user_id, "output")
    from routers.agents_router import _snapshot_agent_sqlite
    new_snapshot = _snapshot_agent_sqlite(db, agent, "Published API version")
    db.flush()
    latest = new_snapshot or db.query(AgentVersion).filter(AgentVersion.agent_id == agent_id).order_by(AgentVersion.version_number.desc()).first()
    # Pin contract versions onto the immutable agent-version record; later schema
    # edits create new schema versions and cannot silently alter this deployment.
    latest.input_schema_version_id = config.input_schema_version_id
    latest.output_schema_version_id = config.output_schema_version_id
    config.publication_state = "published"
    config.published_version_id = latest.id
    db.commit()
    return {"agent_id": str(agent_id), "agent_version": latest.version_number, "publication_state": "published"}

@router.post("/agents/{agent_id}/{action}")
def transition_agent(agent_id: int, action: str, db: Session = Depends(get_db), user: TokenData = Depends(get_current_user)):
    if action not in {"deprecate", "retire", "testing"}: raise HTTPException(404, "Unknown lifecycle action")
    require_owner_agent(db, agent_id, user.user_id); config = config_for(db, agent_id)
    config.publication_state = {"deprecate": "deprecated", "retire": "retired", "testing": "testing"}[action]; db.commit()
    return {"agent_id": str(agent_id), "publication_state": config.publication_state}

@router.post("/agent-invocations/{agent_id}")
async def invoke_agent(agent_id: int, body: ExternalInvokeRequest, db: Session = Depends(get_db), key: ApplicationKeyData = Depends(get_application_api_key)):
    """Invoke through the existing runner and enforce JSON contracts at both boundaries."""
    started = time.monotonic(); request_id = str(uuid.uuid4()); config = authorize_agent(db, key, agent_id, "agent:invoke")
    if config.publication_state not in {"published", "testing"}: raise HTTPException(409, detail={"code": "AGENT_NOT_AVAILABLE", "request_id": request_id})
    if config.publication_state == "published" and body.version is not None:
        published = db.query(AgentVersion).filter(AgentVersion.id == config.published_version_id).first()
        if not published or published.version_number != body.version: raise HTTPException(404, detail={"code": "PUBLISHED_VERSION_NOT_FOUND", "request_id": request_id})
    input_schema = db.get(SchemaVersion, config.input_schema_version_id); output_schema = db.get(SchemaVersion, config.output_schema_version_id)
    if not input_schema or not output_schema:
        raise HTTPException(422, detail={"code": "SCHEMA_VERSION_NOT_FOUND", "message": "Configured schema version is missing", "request_id": request_id})
    input_errors = validate_json_schema(json.loads(input_schema.canonical_schema_json), body.input)
    if input_errors: raise HTTPException(422, detail={"code": "INPUT_SCHEMA_VALIDATION_FAILED", "request_id": request_id, "details": input_errors})

    output_schema_dict = json.loads(output_schema.canonical_schema_json)

    # Pre-supplied output path for non-LLM/testing integrations
    if body.output is not None:
        output = body.output
        errors = validate_json_schema(output_schema_dict, output)
        if errors:
            db.add(APIRequest(request_id=request_id, application_id=int(key.application_id), api_key_id=int(key.api_key_id), agent_id=agent_id, agent_version_id=config.published_version_id, status="failed", error_code="OUTPUT_SCHEMA_VALIDATION_FAILED", duration_ms=int((time.monotonic()-started)*1000))); db.commit()
            raise HTTPException(status_code=502, detail={"error": {"code": "OUTPUT_SCHEMA_VALIDATION_FAILED", "message": "Supplied output failed schema validation", "request_id": request_id, "details": errors}})
        db.add(APIRequest(request_id=request_id, application_id=int(key.application_id), api_key_id=int(key.api_key_id), agent_id=agent_id, agent_version_id=config.published_version_id, status="completed", duration_ms=int((time.monotonic()-started)*1000))); db.commit()
        version = db.get(AgentVersion, config.published_version_id) if config.published_version_id else None
        return {"request_id": request_id, "agent_id": str(agent_id), "agent_version": version.version_number if version else None, "input_schema_version": input_schema.version_number, "output_schema_version": output_schema.version_number, "status": "completed", "output": output}

    # Output validation & single bounded repair attempt
    from models import Session as ChatSession
    from services.agent_runner import run_agent_headless
    agent = db.get(Agent, agent_id)
    session = None
    if body.session_id is not None:
        try:
            session = application_session(db, key, agent_id, body.session_id)
        except (TypeError, ValueError):
            session = None
        if not session:
            raise HTTPException(404, detail={"code": "SESSION_NOT_FOUND", "request_id": request_id})
    else:
        session = ChatSession(
            user_id=agent.user_id,
            application_id=int(key.application_id),
            title="API invocation",
            entity_type="agent",
            entity_id=agent_id,
        )
        db.add(session); db.flush()

    if body.system_instruction is not None:
        session.system_instruction = body.system_instruction
    if body.knowledge_base_ids is not None:
        session.knowledge_base_ids_json = json.dumps(body.knowledge_base_ids)
    db.flush()
    session_system_instruction = session.system_instruction
    session_knowledge_base_ids = (
        json.loads(session.knowledge_base_ids_json)
        if session.knowledge_base_ids_json else None
    )
    image_parts = []
    attachment_records = []
    if body.attachments:
        from routers.chat_router import _process_attachments_sqlite
        local_attachments = [attachment for attachment in body.attachments if attachment.data]
        image_parts, attachment_records = _process_attachments_sqlite(local_attachments, session.id, agent.user_id, db)
        for attachment in body.attachments:
            if attachment.url:
                if attachment.file_type != "image" and not attachment.media_type.startswith("image/"):
                    raise HTTPException(422, detail={"code": "URL_ATTACHMENT_MUST_BE_IMAGE", "message": "URL attachments currently support images only"})
                image_parts.append({
                    "type": "image_url",
                    "image_url": {"url": attachment.url},
                })
                attachment_records.append({
                    "filename": attachment.filename,
                    "media_type": attachment.media_type,
                    "file_type": "image",
                    "url": attachment.url,
                })
    user_content = json.dumps(body.input)
    if image_parts:
        user_content = json.dumps([{"type": "text", "text": user_content}, *image_parts])
    user_message = Message(session_id=session.id, role="user", content=user_content, attachments_json=json.dumps(attachment_records) if attachment_records else None)
    db.add(user_message); db.commit(); db.refresh(user_message)

    output = None
    raw = ""
    target_version_id = config.published_version_id if config.publication_state == "published" else None

    try:
        raw = await run_agent_headless(
            session.id,
            agent_id,
            db,
            response_schema=output_schema_dict,
            system_instruction=session_system_instruction,
            knowledge_base_ids=session_knowledge_base_ids,
            agent_version_id=target_version_id,
        )
        output = json.loads(raw or "")
        errors = validate_json_schema(output_schema_dict, output)
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        errors = [f"Output is not valid JSON: {str(e)}"]

    # If validation failed on initial attempt, execute bounded repair attempt (exactly 1 retry)
    if errors:
        try:
            repair_prompt = f"The previous output failed schema validation with errors: {errors}. Please return a corrected valid JSON object matching the required schema."
            db.add(Message(session_id=session.id, role="assistant", content=raw or ""))
            db.add(Message(session_id=session.id, role="user", content=repair_prompt))
            db.commit()
            raw_repair = await run_agent_headless(
                session.id,
                agent_id,
                db,
                response_schema=output_schema_dict,
                system_instruction=session_system_instruction,
                knowledge_base_ids=session_knowledge_base_ids,
                agent_version_id=target_version_id,
            )
            output = json.loads(raw_repair or "")
            errors = validate_json_schema(output_schema_dict, output)
        except Exception as e:
            errors = [f"Repair attempt failed: {str(e)}"]

    if errors:
        db.add(APIRequest(request_id=request_id, application_id=int(key.application_id), api_key_id=int(key.api_key_id), agent_id=agent_id, agent_version_id=config.published_version_id, status="failed", error_code="OUTPUT_SCHEMA_VALIDATION_FAILED", duration_ms=int((time.monotonic()-started)*1000))); db.commit()
        raise HTTPException(status_code=502, detail={"error": {"code": "OUTPUT_SCHEMA_VALIDATION_FAILED", "message": "Agent output failed schema validation", "request_id": request_id, "details": errors}})
    assistant_msg = Message(session_id=session.id, role="assistant", content=json.dumps(output))
    db.add(assistant_msg)
    db.commit()
    db.refresh(assistant_msg)

    # Allow background trace export tasks to complete before backfilling message_id
    import asyncio
    await asyncio.sleep(0.05)

    # Back-fill message_id on all trace spans recorded during this invocation response
    if DATABASE_TYPE == "mongo":
        try:
            from database_mongo import get_database
            mongo_db = get_database()
            await mongo_db["trace_spans"].update_many(
                {"session_id": str(session.id), "message_id": None},
                {"$set": {"message_id": str(assistant_msg.id)}}
            )
        except Exception:
            pass
    else:
        try:
            db.query(TraceSpan).filter(
                TraceSpan.session_id == session.id,
                TraceSpan.message_id.is_(None),
            ).update({"message_id": assistant_msg.id})
            db.commit()
        except Exception:
            pass

    db.add(APIRequest(request_id=request_id, application_id=int(key.application_id), api_key_id=int(key.api_key_id), agent_id=agent_id, agent_version_id=config.published_version_id, status="completed", duration_ms=int((time.monotonic()-started)*1000))); db.commit()
    version = db.get(AgentVersion, config.published_version_id) if config.published_version_id else None
    return {"request_id": request_id, "session_id": str(session.id), "agent_id": str(agent_id), "agent_version": version.version_number if version else None, "input_schema_version": input_schema.version_number, "output_schema_version": output_schema.version_number, "status": "completed", "output": output}
