# Platform Analysis Report — Obsidian AI

## Executive Summary
Obsidian AI is an open-source AI Agent Management & Orchestration Platform built with a **FastAPI** backend and a **Next.js 16 / React 19** frontend. It enables users to build, deploy, and orchestrate AI agents, multi-agent teams, and visual workflow DAGs across multiple LLM providers (OpenAI, Anthropic, Google Gemini, OpenRouter, and custom OpenAI-compatible endpoints).

This report summarizes the current platform architecture, recent modernization phases (Phases 1–4), key component implementations, security model, and test verification results.

---

## 🏗️ Platform Architecture & Tech Stack

### Tech Stack Overview
- **Backend**: Python 3.12, FastAPI, Microsoft Agent Framework (MAF), SQLAlchemy (SQLite), Motor (MongoDB), Pydantic v2, APScheduler, SlowAPI.
- **Frontend**: Next.js 16 (App Router), React 19, NextAuth v5, Tailwind CSS 4, Radix UI / shadcn/ui, Zustand, `@xyflow/react`.
- **Integrations**: Node.js Baileys WhatsApp Bridge (`wa-bridge`), Model Context Protocol (MCP), Qdrant / FAISS Vector RAG, Qwen3-TTS / Pocket TTS voice synthesis.
- **Dual Database Support**: Automatic runtime branching on `DATABASE_TYPE` (`sqlite` or `mongo`).

### Directory Layout
```
obsidian-ai/
├── backend/                  # FastAPI Application
│   ├── main.py               # Application entrypoint & lifespan
│   ├── models.py / models_mongo.py  # SQLite ORM & Mongo collection models
│   ├── services/             # Core business logic & agent runner
│   ├── routers/              # API REST routers
│   ├── llm/                  # Provider implementations
│   └── tests/                # Pytest test suite (133 tests)
├── frontend/                 # Next.js Application
│   ├── app/                  # App Router pages & API routes
│   ├── components/           # UI components & AI primitives
│   ├── stores/               # Zustand state stores
│   └── lib/                  # Crypto, API client, streaming handlers
├── wa-bridge/                # Node.js WhatsApp Baileys bridge
└── docs/                     # Platform & integration documentation
```

---

## 🔑 Key Subsystems & Implementations

### 1. Agent Runtime & Execution Platform (`services/agent_runner.py`)
- Centralized execution engine powering streaming chat, headless external API calls (`/api/v1/agent-invocations/{agent_id}`), and WhatsApp incoming webhooks.
- Integrates MAF Context Providers (`VectorStoreContextProvider` and `MemoryContextProvider`) to inject grounding knowledge and long-term memories prior to model invocation.
- Features dynamic tool creation capabilities via `builtin_tools.py` (`create_dynamic_tool`, `create_agent`, `create_workflow`, `execute_workflow`).

### 2. Multi-Agent Orchestration & Workflows
- **Teams**: Modernized with MAF `HandoffBuilder` (model-intent directed handoff graphs) and `MagenticBuilder` (supervisor delegation).
- **Workflow DAG Engine**: Supports visual graph editing with async topological execution, cycle detection, parallel step execution, and cron scheduling via APScheduler.

### 3. External Agent API Platform
- Managed applications (`applications_router.py`), scoped API keys (`api_key_service.py`), schema definitions & versions (`schemas_router.py`), and agent publishing/invocations (`agent_api_router.py`).
- Supports input/output JSON schema validation with bounded repair retries and deterministic error structures.

### 4. WhatsApp Channel Integration & Audio Pipeline
- `wa-bridge`: Baileys-based WhatsApp Web socket sidecar providing QR code authentication and headless message relay.
- Voice pipeline: Incoming voice notes transcribed via Groq Whisper / faster-whisper; outgoing replies synthesized via Qwen3-TTS (CUDA GPU) or Pocket TTS (CPU fallback) with voice cloning support.

### 5. Security & Secret Non-Leakage
- JWT auth + TOTP 2FA.
- Fernet encryption at rest for provider credentials and user secrets (`PROVIDER_KEY_SECRET`).
- AES payload encryption in transit (`ENCRYPTION_KEY`).
- Automated trace sanitization (`crypto_utils.py`) redacting Authorization headers, API keys, and Fernet ciphertext across execution traces.

---

## 🏁 Modernization Phase Completion Status

| Phase | Description | Status |
| :--- | :--- | :--- |
| **Phase 1** | Agent Control Plane Core Foundations & Schema Migration | **Completed** |
| **Phase 2** | Dynamic Tool Creation, Agent Provisioning & Execution Routing | **Completed** |
| **Phase 3** | Dual Database Persistence (SQLite + MongoDB) & Trace Sanitization | **Completed** |
| **Phase 4** | MAF Context Providers, Multi-Agent Teams & Channel Interoperability | **Completed** |

---

## 🧪 Test Verification & System Health

The backend test suite was executed using `cd backend && uv run --extra dev pytest`.

- **Total Tests Executed**: 133
- **Passed**: 133
- **Failed**: 0
- **Execution Time**: ~38.89s

All core subsystems—including MAF context providers, team handoffs, headless execution, schema scoping, RAG retrieval, and HITL approval middlewares—are fully operational and verified.
