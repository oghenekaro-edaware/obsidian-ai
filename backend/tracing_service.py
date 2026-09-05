"""
Tracing Service — Provider-Neutral Execution Tracing & OTLP Telemetry.

Provides hierarchical trace contexts (trace_id, span_id, parent_span_id),
GenAI OpenTelemetry semantic attribute mapping, resource metric tracking (psutil),
sanitization, and backend provider abstractions (Otel, Database, Noop).
"""

import time
import uuid
import json
import logging
import psutil
from dataclasses import dataclass, field
from typing import Any, Protocol, Optional, Dict, List
from crypto_utils import sanitize_trace_data
from config import DATABASE_TYPE

logger = logging.getLogger(__name__)


def generate_trace_id() -> str:
    return uuid.uuid4().hex


def generate_span_id() -> str:
    return uuid.uuid4().hex[:16]


def get_process_metrics() -> Dict[str, Any]:
    """Retrieve process memory (RSS) and CPU time metrics."""
    try:
        proc = psutil.Process()
        mem = proc.memory_info()
        cpu = proc.cpu_times()
        return {
            "process_rss_bytes": mem.rss,
            "process_cpu_user_sec": cpu.user,
            "process_cpu_system_sec": cpu.system,
        }
    except Exception as e:
        logger.debug("Failed to sample process metrics: %s", e)
        return {}


@dataclass
class SpanData:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    session_id: Optional[str] = None
    workflow_run_id: Optional[str] = None
    message_id: Optional[str] = None
    span_type: str = "custom"
    name: str = "span"
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    cost_usd: Optional[float] = None
    duration_ms: int = 0
    status: str = "success"
    stop_reason: Optional[str] = None
    input_data: Optional[str] = None
    output_data: Optional[str] = None
    sequence: int = 0
    round_number: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def finish(self, status: str = "success", error: Optional[str] = None):
        self.end_time = time.time()
        self.duration_ms = int((self.end_time - self.start_time) * 1000)
        self.status = status
        if error:
            self.attributes["error.message"] = str(error)
        proc_metrics = get_process_metrics()
        self.attributes.update(proc_metrics)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "session_id": self.session_id,
            "workflow_run_id": self.workflow_run_id,
            "message_id": self.message_id,
            "span_type": self.span_type,
            "name": self.name or self.span_type or "span",
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cost_usd": self.cost_usd,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "input_data": sanitize_trace_data(self.input_data) if self.input_data else None,
            "output_data": sanitize_trace_data(self.output_data) if self.output_data else None,
            "sequence": self.sequence,
            "round_number": self.round_number,
            "attributes": {k: sanitize_trace_data(v) if isinstance(v, (str, dict)) else v for k, v in self.attributes.items()},
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class TraceContext:
    """Hierarchical trace context tracking active trace_id, current span_id stack, and sequence index."""

    def __init__(self, trace_id: Optional[str] = None, session_id: Optional[str] = None, workflow_run_id: Optional[str] = None):
        self.trace_id = trace_id or generate_trace_id()
        self.session_id = session_id
        self.workflow_run_id = workflow_run_id
        self.span_stack: List[str] = []
        self.sequence = 0

    @property
    def current_span_id(self) -> Optional[str]:
        return self.span_stack[-1] if self.span_stack else None

    def next_sequence(self) -> int:
        seq = self.sequence
        self.sequence += 1
        return seq

    def create_span(
        self,
        name: str,
        span_type: str = "custom",
        attributes: Optional[Dict[str, Any]] = None,
        parent_span_id: Optional[str] = None,
    ) -> SpanData:
        span_id = generate_span_id()
        parent_id = parent_span_id or self.current_span_id
        span = SpanData(
            trace_id=self.trace_id,
            span_id=span_id,
            parent_span_id=parent_id,
            session_id=str(self.session_id) if self.session_id is not None else None,
            workflow_run_id=str(self.workflow_run_id) if self.workflow_run_id is not None else None,
            span_type=span_type,
            name=name or span_type or "span",
            sequence=self.next_sequence(),
            attributes=attributes or {},
        )
        return span


class TraceProvider(Protocol):
    """Protocol for pluggable trace exporting backends."""

    async def export_span(self, span: SpanData) -> None:
        ...

    async def flush(self) -> None:
        ...


class NoopTraceProvider:
    """Zero-overhead provider when tracing is disabled."""

    async def export_span(self, span: SpanData) -> None:
        pass

    async def flush(self) -> None:
        pass


class DatabaseTraceProvider:
    """Asynchronous database writer for SQL and MongoDB trace span persistence."""

    def __init__(self, db=None, mongo_db=None):
        self.db = db
        self.mongo_db = mongo_db

    async def export_span(self, span: SpanData) -> None:
        span_dict = span.to_dict()
        if self.mongo_db is not None or (DATABASE_TYPE == "mongo" and self.db is None):
            try:
                from database_mongo import get_database
                from models_mongo import TraceSpanCollection
                db_inst = self.mongo_db or get_database()
                await TraceSpanCollection.create(db_inst, span_dict)
            except Exception as e:
                logger.warning("Failed to export trace span to Mongo: %s", e)
        else:
            try:
                from models import TraceSpan
                from database import SessionLocal
                session = self.db or SessionLocal()
                try:
                    # Convert session_id and workflow_run_id to int if integer IDs are used
                    sid = int(span.session_id) if (span.session_id and span.session_id.isdigit()) else None
                    wrid = int(span.workflow_run_id) if (span.workflow_run_id and span.workflow_run_id.isdigit()) else None
                    mid = int(span.message_id) if (span.message_id and span.message_id.isdigit()) else None

                    record = TraceSpan(
                        trace_id=span.trace_id,
                        span_id=span.span_id,
                        parent_span_id=span.parent_span_id,
                        session_id=sid,
                        workflow_run_id=wrid,
                        message_id=mid,
                        span_type=span.span_type,
                        name=span.name or span.span_type or "span",
                        input_tokens=span.input_tokens,
                        output_tokens=span.output_tokens,
                        cache_read_tokens=span.cache_read_tokens,
                        cache_creation_tokens=span.cache_creation_tokens,
                        cost_usd=span.cost_usd,
                        duration_ms=span.duration_ms,
                        status=span.status,
                        stop_reason=span.stop_reason,
                        input_data=span_dict["input_data"],
                        output_data=span_dict["output_data"],
                        sequence=span.sequence,
                        round_number=span.round_number,
                    )
                    session.add(record)
                    session.commit()
                finally:
                    if self.db is None:
                        session.close()
            except Exception as e:
                logger.warning("Failed to export trace span to SQL database: %s", e)

    async def flush(self) -> None:
        pass


class OtelTraceProvider:
    """OpenTelemetry OTLP exporter integration with database fallback."""

    def __init__(self, endpoint: Optional[str] = None, fallback_provider: Optional[TraceProvider] = None):
        from config import OTEL_EXPORTER_OTLP_ENDPOINT
        self.endpoint = endpoint or OTEL_EXPORTER_OTLP_ENDPOINT
        self.fallback_provider = fallback_provider or DatabaseTraceProvider()
        self._tracer = None
        self._setup_otel()

    def _setup_otel(self):
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            provider = TracerProvider()
            if self.endpoint:
                exporter = OTLPSpanExporter(endpoint=self.endpoint, insecure=True)
                provider.add_span_processor(BatchSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer("obsidian-tracing")
        except Exception as e:
            logger.info("OpenTelemetry SDK / exporter not initialized (%s); using fallback provider.", e)
            self._tracer = None

    async def export_span(self, span: SpanData) -> None:
        # Always export to database fallback for product UI/analytics
        await self.fallback_provider.export_span(span)

        if self._tracer:
            try:
                from opentelemetry.trace import Status, StatusCode
                d = span.to_dict()
                start_time_ns = int(span.start_time * 1e9)
                end_time_ns = int((span.end_time or time.time()) * 1e9)
                otel_span = self._tracer.start_span(
                    span.name,
                    start_time=start_time_ns,
                    attributes=d["attributes"],
                )
                if span.status == "error":
                    otel_span.set_status(Status(StatusCode.ERROR))
                otel_span.end(end_time=end_time_ns)
            except Exception as e:
                logger.debug("Failed emitting OTel span: %s", e)

    async def flush(self) -> None:
        await self.fallback_provider.flush()


def _init_provider() -> TraceProvider:
    from config import TRACE_STORAGE_PROVIDER, OTEL_ENABLED
    if TRACE_STORAGE_PROVIDER == "noop":
        return NoopTraceProvider()
    if TRACE_STORAGE_PROVIDER == "otel" or OTEL_ENABLED:
        return OtelTraceProvider()
    return DatabaseTraceProvider()


_current_provider: TraceProvider = _init_provider()


def get_trace_provider() -> TraceProvider:
    return _current_provider


def set_trace_provider(provider: TraceProvider) -> None:
    global _current_provider
    _current_provider = provider
