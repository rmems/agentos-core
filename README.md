# AgentOS Core

`agentos-core` is a local Rust MCP server that exposes AgentOS context, local
RAG, and bounded repo analysis to local AI clients.

The repository and client-facing MCP server name are `agentos-core`. The
Rust crate, release binary, and launcher all use the `agentos-core` name.

## Scope

This server is intentionally `Context + Compute`:

- canonical repo context and terminology
- read-heavy search and retrieval
- public API and module graph extraction
- terminology and invariant validation
- session handles for explicit research state

It does **not** provide arbitrary shell execution, unrestricted writes, or cross-repo mutation.

## Canon Policy

The default canon root is configured locally by the deployment environment.

The server bakes in the following repo policy:

- treat the current repo as authoritative for implementation details
- use shared canon only for terminology resolution
- do not import assumptions from sibling repos unless context explicitly names them
- if a term is ambiguous across repos, surface the ambiguity instead of normalizing it

## Build

```bash
cargo build --release
```

## Run

Run the MCP server over stdio:

```bash
./scripts/launch.sh
```

Or run management helpers:

```bash
cargo run -- doctor
cargo run -- print-config codex
cargo run -- install --clients codex,cursor,vscode,windsurf,trae,antigravity,jetbrains
```

## Observability CLI Setup

`agentos-core` keeps observability credentials out of the repository. Sentry and
New Relic integration is CLI-only for now: the Rust server does not embed
Sentry, New Relic, or OpenTelemetry SDK keys.

Required environment:

```bash
export SENTRY_ORG="<sentry-org>"
export SENTRY_PROJECT_AGENTOS_CORE="<sentry-project>"
export SENTRY_ENVIRONMENT="local"

export NEW_RELIC_ACCOUNT_ID="<account-id>"
export NEW_RELIC_ENTITY_SEARCH_AGENTOS_CORE="name = 'agentos-core'"
export NEW_RELIC_USER="agentos"
```

Create or update the Sentry release/deploy marker:

```bash
./scripts/observability/sentry_release.sh
```

Post a New Relic custom event:

```bash
./scripts/observability/newrelic_event.sh doctor 0 true none
```

The helper scripts derive the release from `AGENTOS_GIT_SHA` when set, otherwise
from the current git commit. They never store tokens, DSNs, account IDs, or
machine-specific paths in this repository.

Local MCP clients should register the server as `agentos-core` and point to:

```text
<repo-root>/scripts/launch.sh
```


## RAG HTTP Orchestrator

`agentos-core` can also run a local HTTP RAG/vector orchestrator:

```bash
cargo run -- serve-http --bind 127.0.0.1:8765
```

Environment knobs:

```bash
export RAG_COLLECTION=repos
export EMBEDDING_MODEL=nomic-embed-text:latest
export BATCH_SIZE=128
export CHUNK_TOKENS=800
export CHUNK_OVERLAP=0.25
export RAG_REPO_ROOTS=/path/to/repo1:/path/to/repo2
export AGENTOS_RAG_INDEX_MANIFEST=/etc/agentos/configs/rag_index_manifest.json
export AGENTOS_RAG_JWT=<optional-bearer-token>
```

If `AGENTOS_RAG_INDEX_MANIFEST` is unset, the server uses the AgentOS system config path when writable and otherwise falls back to the local user data directory.

Endpoints:

- `POST /ingest/file` for single-file ingest or editor save hooks
- `POST /ingest/diff` for changed file lists from commits or pushes
- `POST /rebuild` for full rebuilds, including `dry_run`
- `POST /cleanup` for full vector cleanup with `mode: "all"`; `stale` and `missing` are reserved and return explicit errors until implemented
- `POST /vectors/upsert` and `POST /vectors/delete` as the local vector MCP facade

Vector metadata includes deterministic SHA256 chunk IDs, source, commit, detected
language, chunk index, embedding model, checksum, timestamp, and tags. Qdrant
point IDs are derived from the SHA256 IDs in UUID form so upserts/deletes remain
stable while preserving the canonical ID in metadata and API responses.

## Exposed MCP Features

### Tools

- `create_session`
- `list_sessions`
- `get_session`
- `update_session`
- `reset_session`
- `drop_session`
- `repo_summary`
- `search_canon`
- `resolve_term`
- `list_public_api`
- `module_graph`
- `validate_terminology`
- `run_invariants`

### Resources

- `canon://policy`
- `canon://summary`
- `canon://readme`
- `canon://public-api`
- `canon://module-graph`
- `canon://terminology`
- `canon://file/<relative-path>`

## Client Helpers

The CLI includes installer helpers for the local clients named in the request:

- Codex
- Cursor
- VS Code
- Windsurf
- Trae
- Antigravity
- JetBrains

JetBrains is handled in two ways:

- update `~/.junie/mcp/mcp.json` for the JetBrains agent surface already present locally
- emit an importable JSON snippet for the AI Assistant MCP UI

## AgentOS Integration

OpenCode and Cursor should use `agentos-core` as the MCP server key. AgentOS
keeps the matching local tool definition outside this repository.

## OpenCode Provider Plan

OpenCode is the local model front end for AgentOS. It is intended to route
across:

- ChatGPT Pro / OpenAI for primary cloud reasoning and general coding
- Alibaba DashScope for Qwen cloud fallback and high-throughput coding
- Google AI Studio for Gemini long-context and alternate reasoning
- Ollama for local models and embeddings

AgentOS keeps provider credentials, model routing, RAG settings, vector database
settings, and MCP server definitions in local configuration files outside this
repository.

The intended routing shape is:

```json
{
  "local": ["ollama"],
  "cloud": ["opencode", "alibaba", "google", "github_copilot"],
  "fallback": "opencode",
  "fallback_order": ["opencode", "alibaba", "google", "ollama"]
}
```

Local RAG uses Ollama embeddings with Qdrant. The active embedding model is
`nomic-embed-text`, and the vector collection is `repos`.
