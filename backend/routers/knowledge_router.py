"""Knowledge Base CRUD + scoped application REST ingestion endpoints."""

import asyncio
from datetime import datetime, timezone
from typing import Union
from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from sqlalchemy.orm import Session

from config import DATABASE_TYPE
from database import get_db
from models import KnowledgeBase, KnowledgeBaseDocument
from schemas import (
    KnowledgeBaseCreate, KnowledgeBaseUpdate, KnowledgeBaseResponse,
    KnowledgeBaseListResponse, KBDocumentCreate, KBDocumentResponse,
    KBDocumentListResponse,
    KnowledgeBaseAppUpsertRequest, KnowledgeBaseAppUpsertResponse,
    KnowledgeAppIngestRequest, KnowledgeAppIngestResponse,
    KBSearchRequest, KBSearchResponse, KBSearchResultItem,
)
from auth import (
    get_current_user, get_current_user_or_api_client, TokenData, APIClientData, ApplicationKeyData, require_permission
)
from file_storage import FileStorageService
from rate_limiter import limiter
from rag_service import RAGService

if DATABASE_TYPE == "mongo":
    from database_mongo import get_database
    from models_mongo import KnowledgeBaseCollection, KBDocumentCollection

router = APIRouter(tags=["knowledge-bases"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_auth_user_id(auth: Union[TokenData, APIClientData, ApplicationKeyData]) -> str:
    if isinstance(auth, TokenData):
        return auth.user_id
    elif isinstance(auth, APIClientData):
        if auth.user_id:
            return auth.user_id
        return auth.client_id
    elif isinstance(auth, ApplicationKeyData):
        if auth.user_id:
            return auth.user_id
        return auth.application_id
    raise HTTPException(status_code=401, detail="Authentication failed")


def _kb_to_response(kb, doc_count: int = 0, is_mongo: bool = False) -> KnowledgeBaseResponse:
    if is_mongo:
        return KnowledgeBaseResponse(
            id=str(kb["_id"]),
            name=kb["name"],
            description=kb.get("description"),
            owner_id=kb.get("owner_id") or kb.get("user_id"),
            app_id=kb.get("app_id"),
            external_id=kb.get("external_id"),
            scope_type=kb.get("scope_type", "workspace"),
            embedding_provider=kb.get("embedding_provider", "google"),
            embedding_model=kb.get("embedding_model", "gemini-embedding-2"),
            secret_id=kb.get("secret_id"),
            is_shared=kb.get("is_shared", False),
            is_active=kb.get("is_active", True),
            document_count=doc_count,
            created_at=kb["created_at"],
        )
    return KnowledgeBaseResponse(
        id=str(kb.id),
        name=kb.name,
        description=kb.description,
        owner_id=kb.owner_id or (str(kb.user_id) if kb.user_id is not None else None),
        app_id=kb.app_id,
        external_id=kb.external_id,
        scope_type=kb.scope_type or "workspace",
        embedding_provider=kb.embedding_provider or "google",
        embedding_model=kb.embedding_model or "gemini-embedding-2",
        secret_id=kb.secret_id,
        is_shared=kb.is_shared,
        is_active=kb.is_active,
        document_count=doc_count,
        created_at=kb.created_at,
    )


def _doc_to_response(doc, is_mongo: bool = False) -> KBDocumentResponse:
    if is_mongo:
        return KBDocumentResponse(
            id=str(doc["_id"]),
            kb_id=str(doc["kb_id"]),
            doc_type=doc["doc_type"],
            name=doc["name"],
            filename=doc.get("filename"),
            media_type=doc.get("media_type"),
            indexed=doc.get("indexed", False),
            created_at=doc["created_at"],
        )
    return KBDocumentResponse(
        id=str(doc.id),
        kb_id=str(doc.kb_id),
        doc_type=doc.doc_type,
        name=doc.name,
        filename=doc.filename,
        media_type=doc.media_type,
        indexed=doc.indexed,
        created_at=doc.created_at,
    )


def _can_access_kb(kb, current_user: TokenData, is_mongo: bool = False) -> bool:
    """Return True if user owns the KB or it's shared."""
    if is_mongo:
        return kb.get("user_id") == current_user.user_id or kb.get("is_shared", False)
    return kb.user_id == int(current_user.user_id) or kb.is_shared


def _owns_kb(kb, current_user: TokenData, is_mongo: bool = False) -> bool:
    if is_mongo:
        return kb.get("user_id") == current_user.user_id
    return kb.user_id == int(current_user.user_id)


# ---------------------------------------------------------------------------
# Scoped Application REST Endpoints
# ---------------------------------------------------------------------------

@router.put("/knowledge/apps/upsert", response_model=KnowledgeBaseAppUpsertResponse)
@limiter.limit("60/minute")
async def upsert_app_knowledge_base(
    request: Request,
    data: KnowledgeBaseAppUpsertRequest,
    response: Response,
    auth: Union[TokenData, APIClientData, ApplicationKeyData] = Depends(get_current_user_or_api_client),
    db: Session = Depends(get_db),
):
    """
    Upsert (Create or Update) an application-scoped Knowledge Base.

    Idempotency Contract:
      - Unique Key Pair: (owner_id, app_id, external_id)
      - Status Codes:
        * 201 Created: Returned when no active KB matches (owner_id, app_id, external_id) and a new KB is created.
        * 200 OK: Returned when an existing KB matching (owner_id, app_id, external_id) is updated in-place.
      - Root Document Behavior: Upserting automatically manages a root document identified by external_id `{external_id}_root`
        containing the KB description, replaces any previous vector embeddings for that root document, and re-indexes the new content.
    """
    owner_id = _get_auth_user_id(auth)
    user_id_int = int(owner_id) if str(owner_id).isdigit() else None

    from services.key_resolution_service import resolve_embedding_credentials

    prov, api_key, model = await resolve_embedding_credentials(
        owner_id,
        {
            "secret_id": data.secret_id,
            "embedding_provider": data.embedding_provider,
        },
        db=db,
    )

    kb_id = None
    created_at = None
    is_new = False

    if DATABASE_TYPE == "mongo":
        mongo_db = get_database()
        existing = await mongo_db[KnowledgeBaseCollection.collection_name].find_one({
            "owner_id": owner_id,
            "app_id": data.app_id,
            "external_id": data.external_id,
            "is_active": True,
        })
        if existing:
            kb_id = str(existing["_id"])
            created_at = existing["created_at"]
            await mongo_db[KnowledgeBaseCollection.collection_name].update_one(
                {"_id": existing["_id"]},
                {"$set": {
                    "name": data.name,
                    "description": data.description,
                    "embedding_provider": prov,
                    "secret_id": data.secret_id,
                    "updated_at": datetime.now(timezone.utc),
                }}
            )
        else:
            is_new = True
            doc = {
                "user_id": owner_id,
                "owner_id": owner_id,
                "app_id": data.app_id,
                "external_id": data.external_id,
                "name": data.name,
                "description": data.description,
                "scope_type": "workspace",
                "embedding_provider": prov,
                "embedding_model": model,
                "secret_id": data.secret_id,
                "is_shared": False,
                "is_active": True,
            }
            created = await KnowledgeBaseCollection.create(mongo_db, doc)
            kb_id = str(created["_id"])
            created_at = created["created_at"]
    else:
        existing = db.query(KnowledgeBase).filter(
            KnowledgeBase.owner_id == owner_id,
            KnowledgeBase.app_id == data.app_id,
            KnowledgeBase.external_id == data.external_id,
            KnowledgeBase.is_active == True,
        ).first()

        if existing:
            kb_id = str(existing.id)
            created_at = existing.created_at
            existing.name = data.name
            existing.description = data.description
            existing.embedding_provider = prov
            existing.secret_id = data.secret_id
            db.commit()
            db.refresh(existing)
        else:
            is_new = True
            kb = KnowledgeBase(
                user_id=user_id_int,
                owner_id=owner_id,
                app_id=data.app_id,
                external_id=data.external_id,
                name=data.name,
                description=data.description,
                scope_type="workspace",
                embedding_provider=prov,
                embedding_model=model,
                secret_id=data.secret_id,
                is_shared=False,
                is_active=True,
            )
            db.add(kb)
            db.commit()
            db.refresh(kb)
            kb_id = str(kb.id)
            created_at = kb.created_at

    root_ext_id = f"{data.external_id}_root"
    root_text = data.description

    if DATABASE_TYPE == "mongo":
        mongo_db = get_database()
        doc_rec = await mongo_db[KBDocumentCollection.collection_name].find_one({
            "kb_id": kb_id,
            "external_id": root_ext_id,
        })
        if doc_rec:
            await mongo_db[KBDocumentCollection.collection_name].update_one(
                {"_id": doc_rec["_id"]},
                {"$set": {
                    "name": data.name,
                    "description": data.description,
                    "content_text": root_text,
                    "indexed": True,
                }}
            )
        else:
            await KBDocumentCollection.create(mongo_db, {
                "kb_id": kb_id,
                "doc_type": "text",
                "name": data.name,
                "description": data.description,
                "app_id": data.app_id,
                "external_id": root_ext_id,
                "content_text": root_text,
                "indexed": True,
            })
    else:
        doc_rec = db.query(KnowledgeBaseDocument).filter(
            KnowledgeBaseDocument.kb_id == int(kb_id),
            KnowledgeBaseDocument.external_id == root_ext_id,
        ).first()
        if doc_rec:
            doc_rec.name = data.name
            doc_rec.description = data.description
            doc_rec.content_text = root_text
            doc_rec.indexed = True
            db.commit()
        else:
            doc_obj = KnowledgeBaseDocument(
                kb_id=int(kb_id),
                doc_type="text",
                name=data.name,
                description=data.description,
                app_id=data.app_id,
                external_id=root_ext_id,
                content_text=root_text,
                indexed=True,
            )
            db.add(doc_obj)
            db.commit()

    if is_new:
        response.status_code = status.HTTP_201_CREATED
    else:
        response.status_code = status.HTTP_200_OK

    if root_text.strip():
        await RAGService.delete_document_vectors_async(kb_id, root_ext_id)
        await RAGService.index_kb_document_async(
            kb_id=kb_id,
            text=root_text,
            metadata={
                "doc_name": data.name,
                "app_id": data.app_id,
                "external_id": root_ext_id,
            },
            embedding_provider=prov,
            api_key=api_key,
            model=model,
        )

    return KnowledgeBaseAppUpsertResponse(
        kb_id=kb_id,
        app_id=data.app_id,
        external_id=data.external_id,
        name=data.name,
        description=data.description,
        created_at=created_at,
    )


@router.post("/knowledge/apps/ingest", response_model=KnowledgeAppIngestResponse)
@limiter.limit("100/minute")
async def ingest_app_knowledge_document(
    request: Request,
    data: KnowledgeAppIngestRequest,
    response: Response,
    auth: Union[TokenData, APIClientData, ApplicationKeyData] = Depends(get_current_user_or_api_client),
    db: Session = Depends(get_db),
):
    """
    Ingest (Create or Update) a document into an existing application-scoped Knowledge Base.

    Idempotency Contract:
      - Parent KB Scoping: Requires an active Knowledge Base matching (owner_id, app_id, external_id). If not found, returns 404 NOT FOUND.
      - Document Scope Key: (kb_id, document_external_id) when `document_external_id` is supplied.
      - Status Codes:
        * 201 Created: Returned when ingesting a new document or when no document matches `document_external_id`.
        * 200 OK: Returned when updating an existing document matching `document_external_id`.
      - Vector Re-indexing: When `document_external_id` is provided, existing vector embeddings for that document external ID
        are cleared before new vector chunks are indexed.
    """
    owner_id = _get_auth_user_id(auth)

    kb = None
    if DATABASE_TYPE == "mongo":
        mongo_db = get_database()
        kb = await mongo_db[KnowledgeBaseCollection.collection_name].find_one({
            "owner_id": owner_id,
            "app_id": data.app_id,
            "external_id": data.external_id,
            "is_active": True,
        })
    else:
        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.owner_id == owner_id,
            KnowledgeBase.app_id == data.app_id,
            KnowledgeBase.external_id == data.external_id,
            KnowledgeBase.is_active == True,
        ).first()

    if not kb:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Knowledge base for app_id '{data.app_id}' and external_id '{data.external_id}' not found",
        )

    kb_id = str(kb["_id"]) if DATABASE_TYPE == "mongo" else str(kb.id)
    secret_id = kb.get("secret_id") if DATABASE_TYPE == "mongo" else kb.secret_id
    embedding_provider = kb.get("embedding_provider", "google") if DATABASE_TYPE == "mongo" else kb.embedding_provider
    embedding_model = kb.get("embedding_model", "gemini-embedding-2") if DATABASE_TYPE == "mongo" else kb.embedding_model

    from services.key_resolution_service import resolve_embedding_credentials

    prov, api_key, model = await resolve_embedding_credentials(
        owner_id,
        {
            "secret_id": secret_id,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
        },
        db=db,
    )

    doc_ext_id = data.document_external_id
    doc_id = None
    created_at = None
    is_update = False

    if DATABASE_TYPE == "mongo":
        mongo_db = get_database()
        existing_doc = None
        if doc_ext_id:
            existing_doc = await mongo_db[KBDocumentCollection.collection_name].find_one({
                "kb_id": kb_id,
                "external_id": doc_ext_id,
            })

        if existing_doc:
            is_update = True
            doc_id = str(existing_doc["_id"])
            created_at = existing_doc["created_at"]
            await mongo_db[KBDocumentCollection.collection_name].update_one(
                {"_id": existing_doc["_id"]},
                {"$set": {
                    "name": data.title,
                    "doc_type": data.doc_type,
                    "content_text": data.content,
                    "indexed": True,
                }}
            )
        else:
            doc_rec = {
                "kb_id": kb_id,
                "doc_type": data.doc_type,
                "name": data.title,
                "app_id": data.app_id,
                "external_id": doc_ext_id,
                "content_text": data.content,
                "indexed": True,
            }
            created = await KBDocumentCollection.create(mongo_db, doc_rec)
            doc_id = str(created["_id"])
            created_at = created["created_at"]
    else:
        existing_doc = None
        if doc_ext_id:
            existing_doc = db.query(KnowledgeBaseDocument).filter(
                KnowledgeBaseDocument.kb_id == int(kb_id),
                KnowledgeBaseDocument.external_id == doc_ext_id,
            ).first()

        if existing_doc:
            is_update = True
            doc_id = str(existing_doc.id)
            created_at = existing_doc.created_at
            existing_doc.name = data.title
            existing_doc.doc_type = data.doc_type
            existing_doc.content_text = data.content
            existing_doc.indexed = True
            db.commit()
        else:
            doc_obj = KnowledgeBaseDocument(
                kb_id=int(kb_id),
                doc_type=data.doc_type,
                name=data.title,
                app_id=data.app_id,
                external_id=doc_ext_id,
                content_text=data.content,
                indexed=True,
            )
            db.add(doc_obj)
            db.commit()
            db.refresh(doc_obj)
            doc_id = str(doc_obj.id)
            created_at = doc_obj.created_at

    if is_update:
        response.status_code = status.HTTP_200_OK
    else:
        response.status_code = status.HTTP_201_CREATED

    if doc_ext_id:
        await RAGService.delete_document_vectors_async(kb_id, doc_ext_id)

    meta = {
        "doc_name": data.title,
        "doc_type": data.doc_type,
        "app_id": data.app_id,
        "external_id": doc_ext_id,
    }
    if data.metadata:
        meta.update(data.metadata)

    await RAGService.index_kb_document_async(
        kb_id=kb_id,
        text=data.content,
        metadata=meta,
        embedding_provider=prov,
        api_key=api_key,
        model=model,
    )

    return KnowledgeAppIngestResponse(
        doc_id=doc_id,
        kb_id=kb_id,
        app_id=data.app_id,
        external_id=data.external_id,
        doc_type=data.doc_type,
        title=data.title,
        indexed=True,
        created_at=created_at,
    )


@router.get("/knowledge/apps/{app_id}", response_model=KnowledgeBaseListResponse)
@limiter.limit("60/minute")
async def list_app_knowledge_bases(
    request: Request,
    app_id: str,
    auth: Union[TokenData, APIClientData, ApplicationKeyData] = Depends(get_current_user_or_api_client),
    db: Session = Depends(get_db),
):
    """
    List all active Knowledge Bases scoped to a specific `app_id` owned by the authenticated user/API client.
    """
    owner_id = _get_auth_user_id(auth)

    if DATABASE_TYPE == "mongo":
        mongo_db = get_database()
        cursor = mongo_db[KnowledgeBaseCollection.collection_name].find({
            "owner_id": owner_id,
            "app_id": app_id,
            "is_active": True,
        })
        kbs = await cursor.to_list(length=100)
        result = []
        for kb in kbs:
            count = await KBDocumentCollection.count_for_kb(mongo_db, str(kb["_id"]))
            result.append(_kb_to_response(kb, doc_count=count, is_mongo=True))
        return KnowledgeBaseListResponse(knowledge_bases=result)

    kbs = db.query(KnowledgeBase).filter(
        KnowledgeBase.owner_id == owner_id,
        KnowledgeBase.app_id == app_id,
        KnowledgeBase.is_active == True,
    ).all()

    result = []
    for kb in kbs:
        count = db.query(KnowledgeBaseDocument).filter(
            KnowledgeBaseDocument.kb_id == kb.id,
        ).count()
        result.append(_kb_to_response(kb, doc_count=count))
    return KnowledgeBaseListResponse(knowledge_bases=result)


@router.post("/knowledge-bases/{kb_id}/search", response_model=KBSearchResponse)
async def search_knowledge_base_content(
    kb_id: str,
    data: KBSearchRequest,
    auth: Union[TokenData, APIClientData, ApplicationKeyData] = Depends(get_current_user_or_api_client),
    db: Session = Depends(get_db),
):
    owner_id = _get_auth_user_id(auth)

    kb = None
    if DATABASE_TYPE == "mongo":
        mongo_db = get_database()
        kb = await KnowledgeBaseCollection.find_by_id(mongo_db, kb_id)
        if not kb or (kb.get("owner_id") != owner_id and kb.get("user_id") != owner_id and not kb.get("is_shared")):
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        secret_id = kb.get("secret_id")
        provider = kb.get("embedding_provider", "google")
        model = kb.get("embedding_model", "gemini-embedding-2")
    else:
        kb = db.query(KnowledgeBase).filter(
            KnowledgeBase.id == int(kb_id) if kb_id.isdigit() else KnowledgeBase.id == kb_id,
            KnowledgeBase.is_active == True,
        ).first()
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        if kb.owner_id != owner_id and (kb.user_id is not None and str(kb.user_id) != owner_id) and not kb.is_shared:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        secret_id = kb.secret_id
        provider = kb.embedding_provider or "google"
        model = kb.embedding_model or "gemini-embedding-2"

    from services.key_resolution_service import resolve_embedding_credentials

    prov, api_key, model_resolved = await resolve_embedding_credentials(
        owner_id,
        {
            "secret_id": secret_id,
            "embedding_provider": provider,
            "embedding_model": model,
        },
        db=db,
    )

    results = await RAGService.query_kb_async(
        kb_id=kb_id,
        query=data.query,
        top_k=data.top_k,
        embedding_provider=prov,
        api_key=api_key,
        model=model_resolved,
    )

    items = [
        KBSearchResultItem(
            text=r.get("text", ""),
            score=r.get("score", 0.0),
            metadata=r.get("metadata"),
        )
        for r in results
    ]
    return KBSearchResponse(results=items)


# ---------------------------------------------------------------------------
# Knowledge Base CRUD (standard UI routes)
# ---------------------------------------------------------------------------

@router.post("/knowledge-bases", response_model=KnowledgeBaseResponse)
async def create_knowledge_base(
    data: KnowledgeBaseCreate,
    current_user: TokenData = Depends(get_current_user),
    _perm=Depends(require_permission("create_knowledge_bases")),
    db: Session = Depends(get_db),
):
    # Only admins may create shared knowledge bases
    if data.is_shared and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can create shared knowledge bases")

    owner_id_val = data.owner_id or current_user.user_id
    if DATABASE_TYPE == "mongo":
        mongo_db = get_database()
        doc = {
            "user_id": current_user.user_id,
            "owner_id": owner_id_val,
            "app_id": data.app_id,
            "external_id": data.external_id,
            "name": data.name,
            "description": data.description,
            "scope_type": data.scope_type,
            "embedding_provider": data.embedding_provider,
            "embedding_model": data.embedding_model,
            "secret_id": data.secret_id,
            "is_shared": data.is_shared,
        }
        created = await KnowledgeBaseCollection.create(mongo_db, doc)
        return _kb_to_response(created, is_mongo=True)

    kb = KnowledgeBase(
        user_id=int(current_user.user_id) if current_user.user_id.isdigit() else None,
        owner_id=owner_id_val,
        app_id=data.app_id,
        external_id=data.external_id,
        name=data.name,
        description=data.description,
        scope_type=data.scope_type,
        embedding_provider=data.embedding_provider,
        embedding_model=data.embedding_model,
        secret_id=data.secret_id,
        is_shared=data.is_shared,
    )
    db.add(kb)
    db.commit()
    db.refresh(kb)
    return _kb_to_response(kb)


@router.get("/knowledge-bases", response_model=KnowledgeBaseListResponse)
async def list_knowledge_bases(
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if DATABASE_TYPE == "mongo":
        mongo_db = get_database()
        kbs = await KnowledgeBaseCollection.find_accessible(mongo_db, current_user.user_id)
        result = []
        for kb in kbs:
            count = await KBDocumentCollection.count_for_kb(mongo_db, str(kb["_id"]))
            result.append(_kb_to_response(kb, doc_count=count, is_mongo=True))
        return KnowledgeBaseListResponse(knowledge_bases=result)

    from sqlalchemy import or_
    kbs = db.query(KnowledgeBase).filter(
        KnowledgeBase.is_active == True,
        or_(
            KnowledgeBase.user_id == int(current_user.user_id),
            KnowledgeBase.is_shared == True,
        ),
    ).all()

    result = []
    for kb in kbs:
        count = db.query(KnowledgeBaseDocument).filter(
            KnowledgeBaseDocument.kb_id == kb.id,
        ).count()
        result.append(_kb_to_response(kb, doc_count=count))
    return KnowledgeBaseListResponse(knowledge_bases=result)


@router.get("/knowledge-bases/{kb_id}", response_model=KnowledgeBaseResponse)
async def get_knowledge_base(
    kb_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if DATABASE_TYPE == "mongo":
        mongo_db = get_database()
        kb = await KnowledgeBaseCollection.find_by_id(mongo_db, kb_id)
        if not kb or not _can_access_kb(kb, current_user, is_mongo=True):
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        count = await KBDocumentCollection.count_for_kb(mongo_db, kb_id)
        return _kb_to_response(kb, doc_count=count, is_mongo=True)

    from sqlalchemy import or_
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == int(kb_id),
        KnowledgeBase.is_active == True,
        or_(
            KnowledgeBase.user_id == int(current_user.user_id),
            KnowledgeBase.is_shared == True,
        ),
    ).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    count = db.query(KnowledgeBaseDocument).filter(KnowledgeBaseDocument.kb_id == kb.id).count()
    return _kb_to_response(kb, doc_count=count)


@router.put("/knowledge-bases/{kb_id}", response_model=KnowledgeBaseResponse)
async def update_knowledge_base(
    kb_id: str,
    data: KnowledgeBaseUpdate,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if data.is_shared is True and current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Only admins can make knowledge bases shared")

    if DATABASE_TYPE == "mongo":
        mongo_db = get_database()
        kb = await KnowledgeBaseCollection.find_by_id(mongo_db, kb_id)
        if not kb or not _owns_kb(kb, current_user, is_mongo=True):
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        updates = data.model_dump(exclude_unset=True)
        updated = await KnowledgeBaseCollection.update(mongo_db, kb_id, current_user.user_id, updates)
        count = await KBDocumentCollection.count_for_kb(mongo_db, kb_id)
        return _kb_to_response(updated, doc_count=count, is_mongo=True)

    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == int(kb_id),
        KnowledgeBase.user_id == int(current_user.user_id),
        KnowledgeBase.is_active == True,
    ).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(kb, key, value)
    db.commit()
    db.refresh(kb)
    count = db.query(KnowledgeBaseDocument).filter(KnowledgeBaseDocument.kb_id == kb.id).count()
    return _kb_to_response(kb, doc_count=count)


@router.delete("/knowledge-bases/{kb_id}")
async def delete_knowledge_base(
    kb_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    import json as _json
    if DATABASE_TYPE == "mongo":
        mongo_db = get_database()
        kb = await KnowledgeBaseCollection.find_by_id(mongo_db, kb_id)
        if not kb or not _owns_kb(kb, current_user, is_mongo=True):
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        await KnowledgeBaseCollection.delete(mongo_db, kb_id, current_user.user_id)
        await mongo_db[KBDocumentCollection.collection_name].delete_many({"kb_id": kb_id})
        RAGService.delete_kb_index(kb_id)
        # Remove this KB from any agent that references it
        agents_col = mongo_db["agents"]
        async for agent in agents_col.find({"knowledge_base_ids_json": {"$exists": True}}):
            raw = agent.get("knowledge_base_ids_json")
            if not raw:
                continue
            ids = _json.loads(raw) if isinstance(raw, str) else raw
            if kb_id in [str(i) for i in ids]:
                new_ids = [i for i in ids if str(i) != kb_id]
                await agents_col.update_one(
                    {"_id": agent["_id"]},
                    {"$set": {"knowledge_base_ids_json": _json.dumps(new_ids)}},
                )
        return {"message": "Knowledge base deleted"}

    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == int(kb_id),
        KnowledgeBase.user_id == int(current_user.user_id),
        KnowledgeBase.is_active == True,
    ).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    db.query(KnowledgeBaseDocument).filter(KnowledgeBaseDocument.kb_id == int(kb_id)).delete(synchronize_session=False)
    db.delete(kb)
    db.commit()
    RAGService.delete_kb_index(kb_id)
    # Remove this KB from any agent that references it
    from models import Agent as _Agent
    agents_with_kb = db.query(_Agent).filter(_Agent.knowledge_base_ids_json.isnot(None)).all()
    for agent in agents_with_kb:
        try:
            ids = _json.loads(agent.knowledge_base_ids_json)
        except (_json.JSONDecodeError, TypeError):
            continue
        if kb_id in [str(i) for i in ids]:
            new_ids = [i for i in ids if str(i) != kb_id]
            agent.knowledge_base_ids_json = _json.dumps(new_ids) if new_ids else None
    db.commit()
    return {"message": "Knowledge base deleted"}


# ---------------------------------------------------------------------------
# Document endpoints
# ---------------------------------------------------------------------------

@router.get("/knowledge-bases/{kb_id}/documents", response_model=KBDocumentListResponse)
async def list_documents(
    kb_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if DATABASE_TYPE == "mongo":
        mongo_db = get_database()
        kb = await KnowledgeBaseCollection.find_by_id(mongo_db, kb_id)
        if not kb or not _can_access_kb(kb, current_user, is_mongo=True):
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        docs = await KBDocumentCollection.find_by_kb(mongo_db, kb_id)
        return KBDocumentListResponse(documents=[_doc_to_response(d, is_mongo=True) for d in docs])

    from sqlalchemy import or_
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == int(kb_id),
        KnowledgeBase.is_active == True,
        or_(
            KnowledgeBase.user_id == int(current_user.user_id),
            KnowledgeBase.is_shared == True,
        ),
    ).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    docs = db.query(KnowledgeBaseDocument).filter(
        KnowledgeBaseDocument.kb_id == kb.id,
    ).all()
    return KBDocumentListResponse(documents=[_doc_to_response(d) for d in docs])


@router.post("/knowledge-bases/{kb_id}/documents", response_model=KBDocumentResponse)
async def add_document(
    kb_id: str,
    data: KBDocumentCreate,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if DATABASE_TYPE == "mongo":
        mongo_db = get_database()
        kb = await KnowledgeBaseCollection.find_by_id(mongo_db, kb_id)
        if not kb or not _can_access_kb(kb, current_user, is_mongo=True):
            raise HTTPException(status_code=404, detail="Knowledge base not found")

        file_id = None
        text_to_index = ""

        if data.doc_type == "text":
            if not data.content_text:
                raise HTTPException(status_code=400, detail="content_text required for text documents")
            text_to_index = data.content_text
        elif data.doc_type == "file":
            if not data.file_data or not data.filename:
                raise HTTPException(status_code=400, detail="file_data and filename required for file documents")
            file_bytes, _ = FileStorageService.decode_data_uri(data.file_data)
            file_id = await FileStorageService.save_file_gridfs(
                mongo_db, data.filename, file_bytes,
                {"kb_id": kb_id, "doc_name": data.name},
            )
            loop = asyncio.get_event_loop()
            text_to_index = await loop.run_in_executor(
                None, RAGService.extract_text, file_bytes, data.filename, data.media_type or ""
            )
        else:
            raise HTTPException(status_code=400, detail="doc_type must be 'text' or 'file'")

        owner_id = kb.get("owner_id") or kb.get("user_id") or current_user.user_id
        secret_id = kb.get("secret_id")
        embedding_provider = kb.get("embedding_provider", "google")
        embedding_model = kb.get("embedding_model", "gemini-embedding-2")

        from services.key_resolution_service import resolve_embedding_credentials

        prov, api_key, model = await resolve_embedding_credentials(
            str(owner_id),
            {
                "secret_id": secret_id,
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
            },
            db=db,
        )

        indexed = False
        if text_to_index.strip():
            await RAGService.index_kb_document_async(
                kb_id=kb_id,
                text=text_to_index,
                metadata={"doc_name": data.name, "filename": data.filename},
                embedding_provider=prov,
                api_key=api_key,
                model=model,
            )
            indexed = True

        doc_rec = {
            "kb_id": kb_id,
            "doc_type": data.doc_type,
            "name": data.name,
            "content_text": data.content_text if data.doc_type == "text" else None,
            "file_id": file_id,
            "filename": data.filename,
            "media_type": data.media_type,
            "indexed": indexed,
        }
        created = await KBDocumentCollection.create(mongo_db, doc_rec)
        return _doc_to_response(created, is_mongo=True)

    # SQLite path
    from sqlalchemy import or_
    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == int(kb_id),
        KnowledgeBase.is_active == True,
        or_(
            KnowledgeBase.user_id == int(current_user.user_id),
            KnowledgeBase.is_shared == True,
        ),
    ).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    file_id = None
    text_to_index = ""

    if data.doc_type == "text":
        if not data.content_text:
            raise HTTPException(status_code=400, detail="content_text required for text documents")
        text_to_index = data.content_text
    elif data.doc_type == "file":
        if not data.file_data or not data.filename:
            raise HTTPException(status_code=400, detail="file_data and filename required for file documents")
        file_bytes, _ = FileStorageService.decode_data_uri(data.file_data)
        file_id = FileStorageService.save_file_sqlite(f"kb_{kb_id}", data.filename, file_bytes)
        loop = asyncio.get_event_loop()
        text_to_index = await loop.run_in_executor(
            None, RAGService.extract_text, file_bytes, data.filename, data.media_type or ""
        )
    else:
        raise HTTPException(status_code=400, detail="doc_type must be 'text' or 'file'")

    owner_id = kb.owner_id or (str(kb.user_id) if kb.user_id is not None else None) or current_user.user_id
    secret_id = kb.secret_id
    embedding_provider = kb.embedding_provider or "google"
    embedding_model = kb.embedding_model or "gemini-embedding-2"

    from services.key_resolution_service import resolve_embedding_credentials

    prov, api_key, model = await resolve_embedding_credentials(
        str(owner_id),
        {
            "secret_id": secret_id,
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
        },
        db=db,
    )

    indexed = False
    if text_to_index.strip():
        await RAGService.index_kb_document_async(
            kb_id=kb_id,
            text=text_to_index,
            metadata={"doc_name": data.name, "filename": data.filename},
            embedding_provider=prov,
            api_key=api_key,
            model=model,
        )
        indexed = True

    doc = KnowledgeBaseDocument(
        kb_id=int(kb_id),
        doc_type=data.doc_type,
        name=data.name,
        content_text=data.content_text if data.doc_type == "text" else None,
        file_id=file_id,
        filename=data.filename,
        media_type=data.media_type,
        indexed=indexed,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return _doc_to_response(doc)


@router.delete("/knowledge-bases/{kb_id}/documents/{doc_id}")
async def delete_document(
    kb_id: str,
    doc_id: str,
    current_user: TokenData = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if DATABASE_TYPE == "mongo":
        mongo_db = get_database()
        kb = await KnowledgeBaseCollection.find_by_id(mongo_db, kb_id)
        if not kb or not _owns_kb(kb, current_user, is_mongo=True):
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        await KBDocumentCollection.delete(mongo_db, doc_id)
        return {"message": "Document deleted"}

    kb = db.query(KnowledgeBase).filter(
        KnowledgeBase.id == int(kb_id),
        KnowledgeBase.user_id == int(current_user.user_id),
        KnowledgeBase.is_active == True,
    ).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    doc = db.query(KnowledgeBaseDocument).filter(
        KnowledgeBaseDocument.id == int(doc_id),
        KnowledgeBaseDocument.kb_id == int(kb_id),
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    db.delete(doc)
    db.commit()
    return {"message": "Document deleted"}
