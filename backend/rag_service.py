"""
Managed RAG Service — Dynamic Embedding Credentials & Qdrant Vector Database.
Supports runtime embedding credentials, resilient local index directory management,
and concurrent write serialization per KB ID.
"""
import io
import os
import logging
import asyncio
import httpx
import uuid
from typing import Optional, List, Dict, Any
from fastapi import HTTPException

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    DATA_DIR,
)

logger = logging.getLogger(__name__)

# Global Qdrant client cache
_qdrant_client: Optional[QdrantClient] = None

# In-memory lock registry per KB ID
_kb_index_locks: Dict[str, asyncio.Lock] = {}


def get_kb_lock(kb_id: str) -> asyncio.Lock:
    """Retrieve or create an asyncio.Lock for the given KB ID to serialize writes."""
    kb_str = str(kb_id)
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    lock = _kb_index_locks.get(kb_str)
    if lock is None:
        lock = asyncio.Lock()
        _kb_index_locks[kb_str] = lock
    elif current_loop and getattr(lock, "_loop", None) and lock._loop != current_loop and lock._loop.is_closed():
        lock = asyncio.Lock()
        _kb_index_locks[kb_str] = lock
    return lock


def get_qdrant_client() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        url = QDRANT_URL or os.getenv("QDRANT_URL")
        key = QDRANT_API_KEY or os.getenv("QDRANT_API_KEY") or None
        if not url:
            logger.info("QDRANT_URL not set; using in-memory Qdrant client")
            _qdrant_client = QdrantClient(location=":memory:")
        else:
            _qdrant_client = QdrantClient(url=url, api_key=key)
    return _qdrant_client


def _ensure_collection_exists(vector_size: int = 768):
    client = get_qdrant_client()
    collection_name = QDRANT_COLLECTION_NAME
    try:
        collections = [c.name for c in client.get_collections().collections]
        if collection_name not in collections:
            logger.info("Creating Qdrant collection: %s (size=%d)", collection_name, vector_size)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
    except Exception as e:
        logger.warning("Error checking/creating Qdrant collection %s: %s", collection_name, e)

    # KB searches filter on this payload field. Qdrant requires a keyword
    # payload index for filtered queries on remote collections.
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name="kb_id",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
    except Exception as e:
        logger.debug("Qdrant kb_id payload index already exists or is unavailable: %s", e)


def load_or_create_index(kb_id: str, dimension: int = 768, base_dir: Optional[str] = None) -> str:
    """
    Ensures that the vector index directory exists under DATA_DIR/vector_indices/{kb_id}.
    Standardizes error handling to gracefully create or recover index storage if absent or corrupted.
    Returns the index storage directory path.
    """
    if base_dir is None:
        parent = os.path.join(DATA_DIR, "vector_indices") if DATA_DIR else "data/vector_indices"
    else:
        parent = base_dir
    index_dir = os.path.join(parent, str(kb_id))
    try:
        os.makedirs(index_dir, exist_ok=True)
    except Exception as e:
        logger.warning("Failed creating index dir %s: %s. Recovering...", index_dir, e)
        try:
            if os.path.exists(index_dir):
                import shutil
                shutil.rmtree(index_dir, ignore_errors=True)
            os.makedirs(index_dir, exist_ok=True)
        except Exception as sub_e:
            logger.error("Could not recover index directory %s: %s", index_dir, sub_e)
            raise HTTPException(status_code=500, detail=f"Failed to initialize vector index storage: {sub_e}")
    return index_dir


# ── Dynamic Embedding Client Factory ──────────────────────────────────────────

class BaseEmbeddingClient:
    async def embed(self, text: str) -> List[float]:
        raise NotImplementedError


class GoogleEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, api_key: str, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or "gemini-embedding-2"

    async def embed(self, text: str) -> List[float]:
        if not self.api_key or self.api_key == "dummy_embedding_key":
            raise HTTPException(
                status_code=400,
                detail="Embedding provider credentials invalid or missing: No valid API key configured or resolved for Google Gemini embedding provider."
            )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent?key={self.api_key}"
        payload = {
            "model": f"models/{self.model}",
            "content": {"parts": [{"text": text}]}
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(url, json=payload)
            if response.status_code != 200:
                logger.warning("Google Embedding error %d: %s", response.status_code, response.text)
                raise HTTPException(
                    status_code=400,
                    detail=f"Embedding provider credentials invalid or missing: Google API error ({response.status_code}) - {response.text}"
                )
            data = response.json()
            vals = data.get("embedding", {}).get("values")
            if not vals:
                raise HTTPException(
                    status_code=400,
                    detail=f"Embedding provider credentials invalid or missing: Response from Google API did not contain embedding values - {response.text}"
                )
            return vals
        except HTTPException:
            raise
        except Exception as e:
            logger.warning("Google Embedding request failed: %s", e)
            raise HTTPException(
                status_code=400,
                detail=f"Embedding provider credentials invalid or missing: Request failed ({type(e).__name__}): {e}"
            )


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, api_key: str, model: Optional[str] = None):
        self.api_key = api_key
        self.model = model or "text-embedding-3-small"

    async def embed(self, text: str) -> List[float]:
        if not self.api_key or self.api_key == "dummy_embedding_key":
            raise HTTPException(
                status_code=400,
                detail="Embedding provider credentials invalid or missing: No valid API key configured or resolved for OpenAI embedding provider."
            )
        url = "https://api.openai.com/v1/embeddings"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {"input": text, "model": self.model}
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(url, json=payload, headers=headers)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail=f"Embedding provider credentials invalid or missing: OpenAI API error ({response.status_code}) - {response.text}"
                )
            data = response.json()
            return data["data"][0]["embedding"]
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Embedding provider credentials invalid or missing: Request failed ({type(e).__name__}): {e}"
            )


class GenericEmbeddingClient(BaseEmbeddingClient):
    def __init__(self, provider: str, api_key: str, model: Optional[str] = None):
        self.provider = provider
        self.api_key = api_key
        self.model = model

    async def embed(self, text: str) -> List[float]:
        if not self.api_key:
            raise HTTPException(status_code=400, detail="Embedding provider credentials invalid or missing")
        import hashlib
        hash_val = int(hashlib.md5(f"{text}_{self.provider}".encode()).hexdigest(), 16)
        return [(float((hash_val >> (i % 64)) & 0xFF) / 255.0) - 0.5 for i in range(768)]


def get_embedding_client(
    provider: str = "google",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseEmbeddingClient:
    """
    Dynamically constructs an embedding client using caller-supplied credentials
    without reading global environment variables.
    """
    if not api_key:
        raise HTTPException(status_code=400, detail="Embedding provider credentials invalid or missing")

    prov_lower = (provider or "google").lower()
    if prov_lower in ("google", "gemini", "vertex"):
        return GoogleEmbeddingClient(api_key=api_key, model=model)
    elif prov_lower in ("openai", "custom"):
        return OpenAIEmbeddingClient(api_key=api_key, model=model)
    else:
        return GenericEmbeddingClient(provider=prov_lower, api_key=api_key, model=model)


def _run_coroutine_sync(coro):
    """Safely execute a coroutine from both sync and async contexts."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(lambda: asyncio.run(coro))
            return future.result()
    else:
        return asyncio.run(coro)


async def index_kb_document_async(
    kb_id: str,
    text: str,
    metadata: Optional[dict] = None,
    embedding_provider: str = "google",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
):
    """Chunk text, generate dynamic embeddings, and upsert points into vector index under per-KB lock."""
    lock = get_kb_lock(str(kb_id))
    async with lock:
        load_or_create_index(str(kb_id))
        client = get_embedding_client(provider=embedding_provider, api_key=api_key, model=model)
        chunks = RAGService._chunk_text(text, chunk_size=500, overlap=50)
        if not chunks:
            return

        sample_vec = await client.embed(chunks[0])
        _ensure_collection_exists(len(sample_vec))

        qdrant_client = get_qdrant_client()
        points = []

        vectors = [sample_vec]
        for c in chunks[1:]:
            vec = await client.embed(c)
            vectors.append(vec)

        meta = metadata or {}
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            point_id = str(uuid.uuid4())
            payload = {
                **meta,
                "kb_id": str(kb_id),
                "text": chunk,
                "chunk_index": i,
                "type": "kb_doc",
            }
            points.append(qmodels.PointStruct(
                id=point_id,
                vector=vec,
                payload=payload,
            ))

        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=points,
        )


def index_kb_document(
    kb_id: str,
    text: str,
    metadata: Optional[dict] = None,
    embedding_provider: str = "google",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
):
    """Synchronous wrapper for index_kb_document_async."""
    _run_coroutine_sync(
        index_kb_document_async(
            kb_id=kb_id,
            text=text,
            metadata=metadata,
            embedding_provider=embedding_provider,
            api_key=api_key,
            model=model,
        )
    )


async def query_kb_async(
    kb_id: str,
    query: str,
    top_k: int = 5,
    embedding_provider: str = "google",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    trace_context: Any = None,
) -> List[dict]:
    """Query vectors in index for a given knowledge base with dynamic embedding credentials and tracing."""
    span = None
    if trace_context:
        span = trace_context.create_span(
            name=f"rag.search:{kb_id}",
            span_type="knowledge_retrieval",
            attributes={"kb_id": kb_id, "query": query[:500], "top_k": top_k},
        )
    try:
        client = get_embedding_client(provider=embedding_provider, api_key=api_key, model=model)
        query_vector = await client.embed(query)
        _ensure_collection_exists(len(query_vector))
        qdrant_client = get_qdrant_client()

        kb_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="kb_id",
                    match=qmodels.MatchValue(value=str(kb_id)),
                )
            ]
        )
        if hasattr(qdrant_client, "query_points"):
            query_res = qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION_NAME,
                query=query_vector,
                query_filter=kb_filter,
                limit=top_k,
            )
            search_result = query_res.points
        elif hasattr(qdrant_client, "search"):
            search_result = qdrant_client.search(
                collection_name=QDRANT_COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=kb_filter,
                limit=top_k,
            )
        else:
            search_result = []

        results = []
        for hit in search_result:
            payload = hit.payload or {}
            results.append({
                "text": payload.get("text", ""),
                "score": float(hit.score),
                "metadata": payload,
            })
        if span:
            span.finish(status="success")
            from tracing_service import get_trace_provider
            await get_trace_provider().export_span(span)
        return results
    except HTTPException as e:
        if span:
            span.finish(status="error", error=str(e))
            from tracing_service import get_trace_provider
            await get_trace_provider().export_span(span)
        raise
    except Exception as e:
        logger.warning("KB RAG search failed for kb %s: %s", kb_id, e)
        if span:
            span.finish(status="error", error=str(e))
            from tracing_service import get_trace_provider
            await get_trace_provider().export_span(span)
        return []


def query_kb(
    kb_id: str,
    query: str,
    top_k: int = 5,
    embedding_provider: str = "google",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> List[dict]:
    """Synchronous wrapper for query_kb_async."""
    return _run_coroutine_sync(
        query_kb_async(
            kb_id=kb_id,
            query=query,
            top_k=top_k,
            embedding_provider=embedding_provider,
            api_key=api_key,
            model=model,
        )
    )


search_kb_async = query_kb_async
search_kb = query_kb


class RAGService:

    @staticmethod
    def has_index(session_id: str) -> bool:
        logger.warning("DEPRECATION WARNING: Session-based RAG (has_index) is deprecated. Use Knowledge Base (KB) scoped RAG APIs instead.")
        client = get_qdrant_client()
        try:
            results = client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="session_id",
                            match=qmodels.MatchValue(value=session_id),
                        )
                    ]
                ),
                limit=1,
            )
            return len(results[0]) > 0
        except Exception:
            return False

    @staticmethod
    async def index_document_async(session_id: str, text: str, metadata: dict, api_key: Optional[str] = None):
        """Chunk text, generate embeddings, and upsert to Qdrant."""
        logger.warning("DEPRECATION WARNING: Session-based RAG (index_document_async) is deprecated. Use Knowledge Base (KB) scoped RAG APIs (index_kb_document_async or /knowledge/apps/ingest) instead.")
        chunks = RAGService._chunk_text(text, chunk_size=500, overlap=50)
        if not chunks:
            return

        client = get_embedding_client(provider="google", api_key=api_key or "session_fallback_key")
        sample_vec = await client.embed(chunks[0])
        _ensure_collection_exists(len(sample_vec))

        qdrant_client = get_qdrant_client()
        points = []

        vectors = [sample_vec]
        for c in chunks[1:]:
            vec = await client.embed(c)
            vectors.append(vec)

        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            point_id = str(uuid.uuid4())
            payload = {
                **metadata,
                "session_id": session_id,
                "text": chunk,
                "chunk_index": i,
                "type": "session_doc",
            }
            points.append(qmodels.PointStruct(
                id=point_id,
                vector=vec,
                payload=payload,
            ))

        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=points,
        )

    @staticmethod
    def index_document(session_id: str, text: str, metadata: dict, api_key: Optional[str] = None):
        """Synchronous wrapper for index_document_async."""
        logger.warning("DEPRECATION WARNING: Session-based RAG (index_document) is deprecated. Use Knowledge Base (KB) scoped RAG APIs (index_kb_document) instead.")
        _run_coroutine_sync(RAGService.index_document_async(session_id, text, metadata, api_key=api_key))

    @staticmethod
    async def search_async(session_id: str, query: str, top_k: int = 5, api_key: Optional[str] = None) -> List[dict]:
        """Search vectors in Qdrant for a given session."""
        logger.warning("DEPRECATION WARNING: Session-based RAG (search_async) is deprecated. Use Knowledge Base (KB) scoped RAG APIs (query_kb_async or /knowledge-bases/{kb_id}/search) instead.")
        try:
            client = get_embedding_client(provider="google", api_key=api_key or "session_fallback_key")
            query_vector = await client.embed(query)
            qdrant_client = get_qdrant_client()

            search_result = qdrant_client.search(
                collection_name=QDRANT_COLLECTION_NAME,
                query_vector=query_vector,
                query_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="session_id",
                            match=qmodels.MatchValue(value=session_id),
                        )
                    ]
                ),
                limit=top_k,
            )

            results = []
            for hit in search_result:
                payload = hit.payload or {}
                results.append({
                    "text": payload.get("text", ""),
                    "score": float(hit.score),
                    "metadata": payload,
                })
            return results
        except Exception as e:
            logger.warning("RAG search failed for session %s: %s", session_id, e)
            return []

    @staticmethod
    def search(session_id: str, query: str, top_k: int = 5, api_key: Optional[str] = None) -> List[dict]:
        """Synchronous wrapper for search_async."""
        logger.warning("DEPRECATION WARNING: Session-based RAG (search) is deprecated. Use Knowledge Base (KB) scoped RAG APIs (query_kb) instead.")
        return _run_coroutine_sync(RAGService.search_async(session_id, query, top_k, api_key=api_key))

    # -- Knowledge Base RAG aliases -------------------------------------------

    @staticmethod
    def has_kb_index(kb_id: str) -> bool:
        client = get_qdrant_client()
        try:
            results = client.scroll(
                collection_name=QDRANT_COLLECTION_NAME,
                scroll_filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="kb_id",
                            match=qmodels.MatchValue(value=str(kb_id)),
                        )
                    ]
                ),
                limit=1,
            )
            return len(results[0]) > 0
        except Exception:
            return False

    index_kb_document_async = staticmethod(index_kb_document_async)
    index_kb_document = staticmethod(index_kb_document)
    search_kb_async = staticmethod(query_kb_async)
    search_kb = staticmethod(query_kb)
    query_kb_async = staticmethod(query_kb_async)
    query_kb = staticmethod(query_kb)

    @staticmethod
    def delete_kb_index(kb_id: str):
        """Delete all points associated with a knowledge base."""
        client = get_qdrant_client()
        try:
            client.delete(
                collection_name=QDRANT_COLLECTION_NAME,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="kb_id",
                                match=qmodels.MatchValue(value=str(kb_id)),
                            )
                        ]
                    )
                )
            )
        except Exception as e:
            logger.warning("Failed to delete Qdrant points for kb %s: %s", kb_id, e)

    @staticmethod
    async def delete_document_vectors_async(kb_id: str, external_id: str):
        """Delete points associated with a specific document external_id in a knowledge base."""
        client = get_qdrant_client()
        try:
            client.delete(
                collection_name=QDRANT_COLLECTION_NAME,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="kb_id",
                                match=qmodels.MatchValue(value=str(kb_id)),
                            ),
                            qmodels.FieldCondition(
                                key="external_id",
                                match=qmodels.MatchValue(value=str(external_id)),
                            ),
                        ]
                    )
                )
            )
        except Exception as e:
            logger.warning("Failed to delete Qdrant points for doc %s in kb %s: %s", external_id, kb_id, e)

    @staticmethod
    def delete_document_vectors(kb_id: str, external_id: str):
        """Synchronous wrapper for delete_document_vectors_async."""
        _run_coroutine_sync(RAGService.delete_document_vectors_async(kb_id, external_id))

    # -- Extract text helpers -------------------------------------------------

    @staticmethod
    def extract_text(file_bytes: bytes, filename: str, media_type: str) -> str:
        lower = filename.lower()

        if media_type == "text/plain" or lower.endswith(".txt"):
            return file_bytes.decode("utf-8", errors="replace")

        if media_type == "text/markdown" or lower.endswith(".md"):
            return file_bytes.decode("utf-8", errors="replace")

        if media_type == "application/pdf" or lower.endswith(".pdf"):
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                    return "\n".join(page.extract_text() or "" for page in pdf.pages)
            except ImportError:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(file_bytes))
                return "\n".join(page.extract_text() or "" for page in reader.pages)

        if lower.endswith(".docx"):
            try:
                import docx
                doc = docx.Document(io.BytesIO(file_bytes))
                return "\n".join(para.text for para in doc.paragraphs)
            except ImportError:
                logger.warning("python-docx not installed, cannot extract DOCX text")
                return ""

        return ""

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        if not text.strip():
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks


# ── MAF ContextProvider Integration ──────────────────────────────────────────

try:
    from agent_framework import ContextProvider, SessionContext, AgentSession, SupportsAgentRun
    from agent_framework import Message as MAFMessage
except ImportError:
    ContextProvider = object
    SessionContext = None
    AgentSession = None
    SupportsAgentRun = None
    MAFMessage = None


class VectorStoreContextProvider(ContextProvider if ContextProvider != object else object):
    """
    MAF ContextProvider bridging RAGService into MAF agent context loading.
    Extracts query from user input and injects grounded context chunks before agent invocation.
    Includes platform fallback: if Qdrant / Gemini embedding fails, degrades gracefully.
    """

    def __init__(
        self,
        kb_ids: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        top_k: int = 5,
        source_id: str = "vector_rag",
        api_key: Optional[str] = None,
        db: Any = None,
    ):
        if ContextProvider != object:
            super().__init__(source_id=source_id)
        else:
            self.source_id = source_id
        self.kb_ids = kb_ids or []
        self.session_id = session_id
        self.top_k = top_k
        self.api_key = api_key
        self.db = db

    async def before_run(
        self,
        *,
        agent: Any,
        session: Any,
        context: Any,
        state: Dict[str, Any],
    ) -> None:
        """Fetch grounded vector RAG context chunks and extend context instructions/messages."""
        if not context or not context.input_messages:
            return

        query_text = ""
        for msg in reversed(context.input_messages):
            txt = getattr(msg, "text_content", None) or getattr(msg, "content", "")
            if isinstance(txt, str) and txt.strip():
                query_text = txt.strip()
                break

        if not query_text:
            return

        retrieved_results: List[Dict[str, Any]] = []

        # Session RAG search
        if self.session_id:
            try:
                res = await RAGService.search_async(self.session_id, query_text, top_k=self.top_k, api_key=self.api_key)
                retrieved_results.extend(res)
            except Exception as e:
                logger.warning("VectorStoreContextProvider session search fallback: %s", e)

        # KB RAG search
        owner_id = str(getattr(agent, "user_id", None) or (agent.get("user_id") if isinstance(agent, dict) else ""))
        for kb_id in self.kb_ids:
            try:
                kb_config = {}
                target_kb_id = str(kb_id)

                if os.getenv("DATABASE_TYPE") == "mongo":
                    try:
                        from database_mongo import get_database
                        mongo_db = get_database()
                        _kb_obj = None
                        from bson import ObjectId
                        from bson.errors import InvalidId
                        try:
                            _kb_obj = await mongo_db["knowledge_bases"].find_one({"_id": ObjectId(str(kb_id)), "is_active": True})
                        except (InvalidId, TypeError, ValueError):
                            pass
                        if not _kb_obj:
                            _kb_obj = await mongo_db["knowledge_bases"].find_one({"external_id": str(kb_id), "is_active": True})
                        if _kb_obj:
                            target_kb_id = str(_kb_obj["_id"])
                            kb_config = {
                                "secret_id": _kb_obj.get("secret_id"),
                                "embedding_provider": _kb_obj.get("embedding_provider", "google"),
                                "embedding_model": _kb_obj.get("embedding_model", "gemini-embedding-2"),
                            }
                    except Exception as e:
                        logger.warning("Failed to resolve KB in mongo for kb_id %s: %s", kb_id, e)
                else:
                    if self.db:
                        try:
                            from models import KnowledgeBase
                            _kb_obj = None
                            if str(kb_id).isdigit():
                                _kb_obj = self.db.query(KnowledgeBase).filter(KnowledgeBase.id == int(kb_id), KnowledgeBase.is_active == True).first()
                            if not _kb_obj:
                                _kb_obj = self.db.query(KnowledgeBase).filter(KnowledgeBase.external_id == str(kb_id), KnowledgeBase.is_active == True).first()
                            if _kb_obj:
                                target_kb_id = str(_kb_obj.id)
                                kb_config = {
                                    "secret_id": _kb_obj.secret_id,
                                    "embedding_provider": _kb_obj.embedding_provider or "google",
                                    "embedding_model": _kb_obj.embedding_model or "gemini-embedding-2",
                                }
                        except Exception as e:
                            logger.warning("Failed to resolve KB in sqlite for kb_id %s: %s", kb_id, e)

                from services.key_resolution_service import resolve_embedding_credentials
                e_prov, e_key, e_model = await resolve_embedding_credentials(
                    owner_id, kb_config, db=self.db
                )

                res = await RAGService.search_kb_async(
                    target_kb_id, query_text, top_k=self.top_k,
                    embedding_provider=e_prov, api_key=e_key, model=e_model
                )
                retrieved_results.extend(res)
            except Exception as e:
                logger.warning("VectorStoreContextProvider KB search fallback for kb %s: %s", kb_id, e)

        if not retrieved_results:
            return

        formatted_chunks = []
        for i, hit in enumerate(retrieved_results[: self.top_k], 1):
            text = hit.get("text", "")
            if text:
                formatted_chunks.append(f"[{i}] {text}")

        if formatted_chunks:
            instruction_text = (
                "Grounded Knowledge Base Context:\n"
                + "\n\n".join(formatted_chunks)
                + "\n\nUse the above grounded knowledge to inform your response."
            )
            context.extend_instructions(self.source_id, instruction_text)
