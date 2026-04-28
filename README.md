# SAAQ Discovery

`saaq-discovery` is a local Rust MCP server that exposes authoritative `corinth-canal`
context and bounded repo analysis to local AI clients.

## Scope

This server is intentionally `Context + Compute`:

- canonical repo context and terminology
- read-heavy search and retrieval
- public API and module graph extraction
- terminology and invariant validation
- session handles for explicit research state

It does **not** provide arbitrary shell execution, unrestricted writes, or cross-repo mutation.

## Canon Policy

The default canon root is `/home/raulmc/corinth-canal`.

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
