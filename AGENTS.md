# Wilo Agent contributor instructions

## Project identity

This repository is `rmems/wilo-agent`.

Wilo Agent is a Rust-first local automation-agent runtime for Linux workstations and repo-aware AI workflows. It is not a generic chatbot.

Wilo Agent is the runtime/control plane. Do not rename it to Metis. Metis is reserved for a future model architecture.

## Architecture boundaries

Keep Wilo Agent as a small control plane.

Core layers:

- Agent control plane: planning, task state, structured logs, workflow runner
- Tool plane: permissioned local tools, MCP tools, shell/git/cargo/julia/nvidia-smi wrappers
- Context plane: RAG, vector DB, repo indexing, memory, manifests
- Model plane: Ollama first, optional model providers later
- Safety/permissions: allowlists, read-only mode, safe-build mode, no silent destructive actions

## Current engineering rules

Prefer small, focused diffs.

Do not introduce dependency soup.

Do not add model backends unless the issue explicitly asks.

Do not split the repo into workspace crates unless the issue explicitly asks.

Do not perform broad AgentOS-to-Wilo renames unless the issue explicitly asks.

Do not add destructive shell execution.

Do not silently mask routing, provider, API-key, or config errors.

## Rust workflow

Use stable Rust.

Before opening a PR, run:

```bash
cargo fmt --check
cargo test
