import os
os.environ["DATABASE_TYPE"] = "sqlite"

import pytest
import config
config.DATABASE_TYPE = "sqlite"

from fastapi.testclient import TestClient
from main import app
from database import get_db, Base, engine
from models import User, APIClient, KnowledgeBase, KnowledgeBaseDocument
from auth import create_access_token, hash_client_secret

client = TestClient(app)

import rag_service
from unittest.mock import patch, AsyncMock

@pytest.fixture(autouse=True)
def mock_embedding_client():
    mock_client = AsyncMock()
    mock_client.embed = AsyncMock(return_value=[0.1] * 768)
    qc = rag_service.get_qdrant_client()
    try:
        qc.delete_collection(rag_service.QDRANT_COLLECTION_NAME)
    except Exception:
        pass
    with patch("rag_service.get_embedding_client", return_value=mock_client) as p:
        yield p

@pytest.fixture
def setup_kb_data():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = next(get_db())
    try:
        user = User(
            username="testuser",
            email="testuser@example.com",
            hashed_password="hashed_pw",
            role="user"
        )
        db.add(user)
        db.flush()

        jwt_token = create_access_token({
            "user_id": str(user.id),
            "username": user.username,
            "role": user.role,
            "token_type": "user"
        })

        client_secret_plain = "secret123456789012345678901234567890"
        api_client = APIClient(
            name="Test App Client",
            client_id="cli_test_client_id_123",
            hashed_secret=hash_client_secret(client_secret_plain),
            created_by=user.id,
            is_active=True
        )
        db.add(api_client)
        db.commit()

        yield {
            "user": user,
            "jwt_token": jwt_token,
            "api_key": "cli_test_client_id_123",
            "api_secret": client_secret_plain,
        }
    finally:
        db.close()


def test_kb_upsert_create_flow(setup_kb_data):
    token = setup_kb_data["jwt_token"]
    payload = {
        "app_id": "issue-tracker",
        "external_id": "proj-99",
        "name": "Project Alpha",
        "description": "Bug tracking knowledge for Project Alpha"
    }

    res = client.put(
        "/knowledge/apps/upsert",
        headers={"Authorization": f"Bearer {token}"},
        json=payload
    )

    assert res.status_code in (200, 201)
    data = res.json()
    assert "kb_id" in data
    assert data["external_id"] == "proj-99"
    assert data["app_id"] == "issue-tracker"
    assert data["name"] == "Project Alpha"
    assert data["description"] == "Bug tracking knowledge for Project Alpha"

    # Verify that searching returns content from the root project description
    search_res = client.post(
        f"/knowledge-bases/{data['kb_id']}/search",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "Bug tracking knowledge"}
    )
    assert search_res.status_code == 200
    search_data = search_res.json()
    assert "results" in search_data
    assert len(search_data["results"]) > 0
    assert "Bug tracking knowledge for Project Alpha" in search_data["results"][0]["text"]


def test_kb_upsert_update_idempotency(setup_kb_data):
    token = setup_kb_data["jwt_token"]
    initial_payload = {
        "app_id": "issue-tracker",
        "external_id": "proj-99",
        "name": "Project Alpha",
        "description": "Bug tracking knowledge for Project Alpha"
    }

    res1 = client.put(
        "/knowledge/apps/upsert",
        headers={"Authorization": f"Bearer {token}"},
        json=initial_payload
    )
    assert res1.status_code in (200, 201)
    kb_id_1 = res1.json()["kb_id"]

    updated_payload = {
        "app_id": "issue-tracker",
        "external_id": "proj-99",
        "name": "Project Alpha Updated",
        "description": "Updated project documentation"
    }

    res2 = client.put(
        "/knowledge/apps/upsert",
        headers={"Authorization": f"Bearer {token}"},
        json=updated_payload
    )
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["kb_id"] == kb_id_1
    assert data2["description"] == "Updated project documentation"

    # Verify root description document was updated/replaced rather than duplicated
    db = next(get_db())
    try:
        docs = db.query(KnowledgeBaseDocument).filter(
            KnowledgeBaseDocument.kb_id == int(kb_id_1),
            KnowledgeBaseDocument.external_id == "proj-99_root"
        ).all()
        assert len(docs) == 1
        assert docs[0].content_text == "Updated project documentation"
    finally:
        db.close()


def test_kb_external_document_ingestion(setup_kb_data):
    token = setup_kb_data["jwt_token"]

    # First ensure target KB exists
    upsert_res = client.put(
        "/knowledge/apps/upsert",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "app_id": "issue-tracker",
            "external_id": "proj-99",
            "name": "Project Alpha",
            "description": "Project Alpha KB"
        }
    )
    kb_id = upsert_res.json()["kb_id"]

    ingest_payload = {
        "app_id": "issue-tracker",
        "external_id": "proj-99",
        "doc_type": "bug_report",
        "title": "Bug #101: Memory Leak",
        "content": "Crash occurred on worker boot due to unhandled memory leak in pool manager.",
        "metadata": {"severity": "high"}
    }

    res = client.post(
        "/knowledge/apps/ingest",
        headers={"Authorization": f"Bearer {token}"},
        json=ingest_payload
    )
    assert res.status_code in (200, 201)
    doc_data = res.json()
    assert doc_data["kb_id"] == str(kb_id)
    assert doc_data["indexed"] is True

    # Assert searching returns content from Bug #101
    search_res = client.post(
        f"/knowledge-bases/{kb_id}/search",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "unhandled memory leak"}
    )
    assert search_res.status_code == 200
    results = search_res.json()["results"]
    assert len(results) > 0
    assert any("Crash occurred on worker boot" in r["text"] for r in results)


def test_kb_api_key_auth_support(setup_kb_data):
    api_key = setup_kb_data["api_key"]
    api_secret = setup_kb_data["api_secret"]
    headers = {
        "X-API-Key": api_key,
        "X-API-Secret": api_secret,
    }

    res = client.put(
        "/knowledge/apps/upsert",
        headers=headers,
        json={
            "app_id": "bug-tracker",
            "external_id": "proj-100",
            "name": "Project Beta",
            "description": "Beta project tracking"
        }
    )
    assert res.status_code in (200, 201)
    assert res.json()["external_id"] == "proj-100"


def test_kb_ingest_404_when_kb_not_found(setup_kb_data):
    token = setup_kb_data["jwt_token"]
    res = client.post(
        "/knowledge/apps/ingest",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "app_id": "non-existent-app",
            "external_id": "non-existent-proj",
            "doc_type": "text",
            "title": "Doc 1",
            "content": "Some content"
        }
    )
    assert res.status_code == 404


def test_kb_ingest_empty_content_validation(setup_kb_data):
    token = setup_kb_data["jwt_token"]
    client.put(
        "/knowledge/apps/upsert",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "app_id": "issue-tracker",
            "external_id": "proj-99",
            "name": "Project Alpha",
            "description": "Alpha KB"
        }
    )

    res = client.post(
        "/knowledge/apps/ingest",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "app_id": "issue-tracker",
            "external_id": "proj-99",
            "doc_type": "text",
            "title": "Empty doc",
            "content": "   "
        }
    )
    assert res.status_code == 422


def test_kb_document_idempotency_via_document_external_id(setup_kb_data):
    token = setup_kb_data["jwt_token"]
    client.put(
        "/knowledge/apps/upsert",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "app_id": "issue-tracker",
            "external_id": "proj-99",
            "name": "Project Alpha",
            "description": "Alpha KB"
        }
    )

    doc_payload_1 = {
        "app_id": "issue-tracker",
        "external_id": "proj-99",
        "document_external_id": "bug-101",
        "doc_type": "bug_report",
        "title": "Bug #101",
        "content": "Initial bug report description."
    }
    res1 = client.post(
        "/knowledge/apps/ingest",
        headers={"Authorization": f"Bearer {token}"},
        json=doc_payload_1
    )
    assert res1.status_code in (200, 201)

    doc_payload_2 = {
        "app_id": "issue-tracker",
        "external_id": "proj-99",
        "document_external_id": "bug-101",
        "doc_type": "bug_report",
        "title": "Bug #101 Updated",
        "content": "Updated bug report description with fix."
    }
    res2 = client.post(
        "/knowledge/apps/ingest",
        headers={"Authorization": f"Bearer {token}"},
        json=doc_payload_2
    )
    assert res2.status_code == 200

    db = next(get_db())
    try:
        docs = db.query(KnowledgeBaseDocument).filter(
            KnowledgeBaseDocument.external_id == "bug-101"
        ).all()
        assert len(docs) == 1
        assert docs[0].name == "Bug #101 Updated"
        assert docs[0].content_text == "Updated bug report description with fix."
    finally:
        db.close()


def test_kb_application_api_key_auth(setup_kb_data):
    import json
    from models import Application, APIKey
    from services.api_key_service import generate_api_key, hash_api_key

    db = next(get_db())
    try:
        user = setup_kb_data["user"]
        app_obj = Application(user_id=user.id, name="KB App", status="active")
        db.add(app_obj)
        db.flush()

        prefix, secret, oba_key = generate_api_key()
        key_record = APIKey(
            application_id=app_obj.id,
            name="KB Key",
            key_prefix=prefix,
            secret_hash=hash_api_key(secret),
            scopes_json=json.dumps(["kb:read", "kb:write"])
        )
        db.add(key_record)
        db.commit()
    finally:
        db.close()

    # 1. Test upsert via Authorization: Bearer oba_...
    upsert_res = client.put(
        "/knowledge/apps/upsert",
        headers={"Authorization": f"Bearer {oba_key}"},
        json={
            "app_id": "app-key-service",
            "external_id": "service-kb-1",
            "name": "Service KB",
            "description": "Knowledge base created via Application API Key"
        }
    )
    assert upsert_res.status_code in (200, 201)
    kb_id = upsert_res.json()["kb_id"]

    # 2. Test ingest via X-API-Key: oba_...
    ingest_res = client.post(
        "/knowledge/apps/ingest",
        headers={"X-API-Key": oba_key},
        json={
            "app_id": "app-key-service",
            "external_id": "service-kb-1",
            "document_external_id": "doc-1",
            "doc_type": "text",
            "title": "Service Doc",
            "content": "Service document content for testing"
        }
    )
    assert ingest_res.status_code in (200, 201)

    # 3. Test list_app_knowledge_bases via Bearer oba_...
    list_res = client.get(
        "/knowledge/apps/app-key-service",
        headers={"Authorization": f"Bearer {oba_key}"}
    )
    assert list_res.status_code == 200
    assert len(list_res.json()["knowledge_bases"]) == 1

    # 4. Test search_knowledge_base_content via X-API-Key oba_...
    search_res = client.post(
        f"/knowledge-bases/{kb_id}/search",
        headers={"X-API-Key": oba_key},
        json={"query": "Service document content"}
    )
    assert search_res.status_code == 200
    assert len(search_res.json()["results"]) > 0


def test_get_knowledge_bases_by_app_id(setup_kb_data):
    token = setup_kb_data["jwt_token"]
    client.put(
        "/knowledge/apps/upsert",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "app_id": "jira-tracker",
            "external_id": "proj-1",
            "name": "Jira Proj 1",
            "description": "Desc 1"
        }
    )
    client.put(
        "/knowledge/apps/upsert",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "app_id": "jira-tracker",
            "external_id": "proj-2",
            "name": "Jira Proj 2",
            "description": "Desc 2"
        }
    )

    res = client.get(
        "/knowledge/apps/jira-tracker",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert res.status_code == 200
    data = res.json()
    assert "knowledge_bases" in data
    assert len(data["knowledge_bases"]) == 2
    ext_ids = [kb["external_id"] for kb in data["knowledge_bases"]]
    assert "proj-1" in ext_ids
    assert "proj-2" in ext_ids
