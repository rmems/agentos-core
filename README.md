# AgentOS Core

`agentos-core` is a local Rust MCP server that exposes authoritative
`corinth-canal` context and bounded repo analysis to local AI clients.

The repository and client-facing MCP server name are `agentos-core`. The
current Rust crate and release binary are still named `saaq-discovery`, and the
launcher script handles that internal binary path.

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

Local MCP clients should register the server as `agentos-core` and point to:

```text
<repo-root>/scripts/launch.sh
```

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
