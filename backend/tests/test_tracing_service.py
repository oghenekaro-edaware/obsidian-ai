import pytest
import asyncio
from tracing_service import TraceContext, SpanData, DatabaseTraceProvider, OtelTraceProvider, NoopTraceProvider

@pytest.mark.asyncio
async def test_trace_context_and_span_creation():
    ctx = TraceContext(session_id="123")
    assert ctx.trace_id is not None
    assert ctx.current_span_id is None

    span1 = ctx.create_span("root_span", span_type="agent_run")
    assert span1.trace_id == ctx.trace_id
    assert span1.parent_span_id is None

    ctx.span_stack.append(span1.span_id)
    assert ctx.current_span_id == span1.span_id

    span2 = ctx.create_span("child_span", span_type="llm_call")
    assert span2.trace_id == ctx.trace_id
    assert span2.parent_span_id == span1.span_id

    span2.finish(status="success")
    assert span2.duration_ms >= 0
    assert "process_rss_bytes" in span2.attributes

@pytest.mark.asyncio
async def test_noop_provider():
    provider = NoopTraceProvider()
    ctx = TraceContext()
    span = ctx.create_span("noop_test")
    await provider.export_span(span)
    await provider.flush()

@pytest.mark.asyncio
async def test_span_creation_with_none_or_empty_name():
    ctx = TraceContext(session_id="123")
    span_none = ctx.create_span(name=None, span_type="tool_call")
    assert span_none.name == "tool_call"
    assert span_none.to_dict()["name"] == "tool_call"

    span_empty = ctx.create_span(name="", span_type="custom")
    assert span_empty.name == "custom"
    assert span_empty.to_dict()["name"] == "custom"

@pytest.mark.asyncio
async def test_database_trace_provider_export_span_null_name_handling():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    from models import Base, TraceSpan

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    try:
        provider = DatabaseTraceProvider(db=db)
        ctx = TraceContext(session_id="100")
        span = ctx.create_span(name=None, span_type="tool_call")
        # Explicitly set span.name to None to test DatabaseTraceProvider's fallback
        span.name = None

        await provider.export_span(span)

        record = db.query(TraceSpan).first()
        assert record is not None
        assert record.name == "tool_call"
    finally:
        db.close()
