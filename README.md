<h1 align="center">
  <br>
  Wilo Agent
  <br>
</h1>

<p align="center">
  <b>Local AI automation agent &amp; MCP server for context, RAG, and bounded repo analysis</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-experimental-yellow?style=flat-square" alt="Experimental" />
  <img src="https://img.shields.io/badge/language-Rust-orange?style=flat-square" alt="Rust" />
  <img src="https://img.shields.io/badge/protocol-MCP-blue?style=flat-square" alt="MCP" />
  <img src="https://img.shields.io/badge/GPU-RTX_5080-76b900?style=flat-square&logo=nvidia" alt="NVIDIA RTX 5080" />
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License" />
</p>

---

> **⚠️ Experimental — personal project.** Wilo Agent is an early-stage experiment built and maintained by a single developer. It is not production software and is not accepting contributions at this time. This repo serves as a public build log — GitHub issues, PRs, and commits track design decisions and progress in the open.

**Wilo Agent** (`agentos-core`) is a local Rust-based [MCP](https://modelcontextprotocol.io/) server that exposes canonical repo context, local RAG pipelines, and bounded analysis to AI coding clients — all running on your own hardware.

Built for the **NVIDIA RTX 5080** and open-weight models, Wilo keeps your code, embeddings, and inference fully local and private.

## Development Workflow

Wilo is developed with an AI-assisted, multi-agent workflow:

- **GitHub** — serves as the primary build log and progress tracker; issues, PRs, and commits document every design decision in the open
- **[Codex](https://openai.com/index/openai-codex/)** — cloud agent handling heavy repo work: large refactors, cross-file changes, and deep code generation
- **PR review bots** — automated code audit on every pull request to catch issues before merge
- **[Linear](https://linear.app)** — project planning, issue tracking, and sprint management
- **Local models** — on-device inference via Ollama/vLLM for fast iteration, private code analysis, and RAG queries

## Highlights

- **MCP server over stdio** — plug into Codex, Cursor, VS Code, Windsurf, Trae, Antigravity, or JetBrains
- **Local RAG pipeline** — Ollama embeddings → Qdrant vector store → semantic code search
- **Repo analysis** — public API extraction, module graph, terminology validation, invariant checks
- **Session management** — stateful research sessions scoped to a repo snapshot
- **Multi-provider model routing** — Ollama, vLLM, OpenAI, Anthropic, Google, Azure, NVIDIA NIM, and more
- **HTTP RAG orchestrator** — file/diff ingest, full rebuilds, vector upsert/delete, and cleanup endpoints
- **Observability** — Sentry release markers and New Relic custom events (CLI-only, no embedded SDKs)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   AI Coding Clients                 │
│  Codex · Cursor · VS Code · Windsurf · JetBrains   │
└────────────────────────┬────────────────────────────┘
                         │ stdio / MCP
┌────────────────────────▼────────────────────────────┐
│               agentos-core  (Rust)                  │
│                                                     │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐ │
│  │ Sessions │  │ Repo     │  │ Model Router      │ │
│  │ Manager  │  │ Analysis │  │ role-based routing │ │
│  └──────────┘  └──────────┘  └───────────────────┘ │
│  ┌──────────────────────────────────────────────┐   │
│  │         RAG Orchestrator (HTTP)              │   │
│  │  ingest · rebuild · cleanup · vector ops     │   │
│  └──────────────────────────────────────────────┘   │
└────────┬───────────────────────────────┬────────────┘
         │                               │
┌────────▼──────────┐         ┌──────────▼──────────┐
│  Ollama / vLLM    │         │     Qdrant          │
│  local inference  │         │  vector storage     │
│  + embeddings     │         │                     │
└───────────────────┘         └─────────────────────┘
```

## Model Backends

Wilo Agent separates the **control plane** (agent runtime, MCP, orchestration) from the **model plane** (pluggable inference backends). Current evaluation targets:

| Model | Params | Architecture | Format | Runtime | Est. VRAM |
|-------|--------|-------------|--------|---------|-----------|
| `NVIDIA-Nemotron-3-Nano-4B-BF16` | 4B | Transformer | BF16 | Transformers / GGUF | ~8 GB |
| `NVIDIA-Nemotron-Nano-9B-v2-FP8` | 9B | Transformer | FP8 | vLLM / TensorRT-LLM | ~9 GB |
| `Mamba-Codestral-7B-v0.1-GGUF` | 7B | SSM (Mamba) | GGUF | Ollama / llama.cpp | ~8 GB |
| `AI21-Jamba-Reasoning-3B-GGUF` | 3B | Hybrid SSM+Transformer (Jamba) | GGUF | Ollama / llama.cpp | ~3–4 GB |
| `Falcon-H1R-7B-GGUF` | 7B | Hybrid SSM+Transformer | GGUF | Ollama / llama.cpp | ~8 GB |

### Why Hybrid Architectures

Three of the five evaluation targets — Mamba-Codestral, Jamba, and Falcon-H1R — use **SSM (state-space model)** or **hybrid SSM + Transformer** designs instead of pure Transformers. This is a deliberate choice:

- **Linear-time inference** — SSM layers scale O(n) with sequence length instead of the O(n²) self-attention cost, enabling faster processing of large codebases and long context windows
- **Constant memory per token** — SSM state is fixed-size regardless of sequence length, reducing VRAM pressure during long agent workflows with tool calls
- **Better long-context scaling** — hybrid models keep Transformer attention for tasks that benefit from global reasoning while offloading sequential processing to SSM layers

This makes hybrid models a natural fit for Wilo's core workloads: repo audit (large file scans), RAG-assisted research (high chunk counts), and multi-step agent loops where context accumulates over many tool calls. The Jamba 3B in particular can serve as a fast triage model that pre-filters before routing heavier tasks to the 9B Nemotron.

Role-based routing dispatches requests across providers:

```
DefaultCoding  →  Ollama / OpenCode
LongContext    →  Google AI Studio
LocalPrivate   →  Ollama
RagEmbedding   →  Ollama (nomic-embed-text)
CudaGpu        →  Ollama
Fallback       →  OpenCode → Alibaba → Google → Ollama
```

## Quick Start

### Prerequisites

- Rust toolchain (edition 2024)
- [Ollama](https://ollama.com) running locally with `nomic-embed-text`
- [Qdrant](https://qdrant.tech) running locally (default `localhost:6333`)

### Build

```bash
cargo build --release
```

### Run the MCP Server

```bash
./scripts/launch.sh
```

The launch script auto-builds if no release binary is found, then starts the MCP server over stdio.

### Run the RAG HTTP Orchestrator

```bash
cargo run -- serve-http --bind 127.0.0.1:8765
```

### CLI Commands

```bash
# Health check
cargo run -- doctor

# Print client-specific MCP config
cargo run -- print-config codex

# Install MCP server into AI clients
cargo run -- install --clients codex,cursor,vscode,windsurf,trae,antigravity,jetbrains

# Uninstall
cargo run -- uninstall --clients codex,cursor

# Show model routing table
cargo run -- route

# Index repos for RAG
cargo run -- index
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `create_session` | Start a stateful research session scoped to a repo |
| `list_sessions` | List all active sessions |
| `get_session` | Retrieve session details and snapshot |
| `update_session` | Update session stage, terms, invariants, or notes |
| `reset_session` | Re-scan repo and refresh the session snapshot |
| `drop_session` | Destroy a session |
| `repo_summary` | High-level repo overview with architecture and key modules |
| `search_canon` | Full-text search across the repo snapshot |
| `resolve_term` | Look up a term across the codebase with context |
| `list_public_api` | Extract all public structs, enums, traits, functions, and constants |
| `module_graph` | Build and return the module dependency graph |
| `validate_terminology` | Check pinned terms for consistency across the repo |
| `run_invariants` | Execute invariant checks against the current snapshot |

## MCP Resources

| URI | Description |
|-----|-------------|
| `canon://policy` | Canon policy and invariants |
| `canon://summary` | Repo summary |
| `canon://readme` | Repository README |
| `canon://public-api` | Public API listing |
| `canon://module-graph` | Module dependency graph |
| `canon://terminology` | Pinned terminology definitions |
| `canon://file/<path>` | Read any file from the repo snapshot |

## RAG HTTP Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest/file` | Single-file ingest or editor save hooks |
| `POST` | `/ingest/diff` | Changed file lists from commits or pushes |
| `POST` | `/rebuild` | Full rebuild (supports `dry_run`) |
| `POST` | `/cleanup` | Full vector cleanup (`mode: "all"`) |
| `POST` | `/vectors/upsert` | Local vector MCP facade — upsert |
| `POST` | `/vectors/delete` | Local vector MCP facade — delete |

### Environment Variables

```bash
# RAG orchestrator
export RAG_COLLECTION=repos
export EMBEDDING_MODEL=nomic-embed-text:latest
export BATCH_SIZE=128
export CHUNK_TOKENS=800
export CHUNK_OVERLAP=0.25
export RAG_REPO_ROOTS=/path/to/repo1:/path/to/repo2
export AGENTOS_RAG_INDEX_MANIFEST=/etc/agentos/configs/rag_index_manifest.json
export AGENTOS_RAG_JWT=<optional-bearer-token>

# Observability (CLI-only, no embedded SDKs)
export SENTRY_ORG="<sentry-org>"
export SENTRY_PROJECT_AGENTOS_CORE="<sentry-project>"
export NEW_RELIC_ACCOUNT_ID="<account-id>"
```

## Canon Policy

The server enforces a strict repo-centric canon:

- Treat the current repo as authoritative for implementation details
- Use shared canon only for terminology resolution
- Do not import assumptions from sibling repos unless context explicitly names them
- Surface ambiguity instead of normalizing it across repos

The default canon root is set by the deployment environment via `AGENTOS_CORE_CANON_ROOT` or `config/server.toml`.

## Supported Clients

| Client | Install Method |
|--------|---------------|
| Codex | Auto-config via CLI |
| Cursor | Auto-config via CLI |
| VS Code | Auto-config via CLI |
| Windsurf | Auto-config via CLI |
| Trae | Auto-config via CLI |
| Antigravity | Auto-config via CLI |
| JetBrains | `~/.junie/mcp/mcp.json` + AI Assistant import snippet |

## Project Structure

```
├── Cargo.toml
├── scripts/
│   ├── launch.sh                   # MCP server launcher
│   └── observability/
│       ├── sentry_release.sh       # Sentry release/deploy marker
│       └── newrelic_event.sh       # New Relic custom event poster
└── src/
    ├── main.rs                     # CLI entrypoint and subcommands
    ├── config.rs                   # Server config and repo home discovery
    ├── server.rs                   # MCP server — tools and resources
    ├── session.rs                  # Stateful session store
    ├── repo.rs                     # Repo scanning, public API, module graph
    ├── rag.rs                      # RAG config, chunking, search
    ├── orchestrator.rs             # RAG HTTP orchestrator and vector ops
    ├── http.rs                     # Axum HTTP server with JWT auth
    ├── router.rs                   # Multi-provider model routing
    ├── schema.rs                   # JSON Schema helpers
    ├── install.rs                  # Client installer helpers
    └── tools/
        ├── ollama.rs               # Ollama embed + generate client
        └── qdrant.rs               # Qdrant upsert, batch upsert, delete
```

## License

[MIT](LICENSE)
