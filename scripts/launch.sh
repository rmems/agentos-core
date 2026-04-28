#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AGENTOS_CORE_HOME="$repo_root"

bin="$repo_root/target/release/agentos-core"
if [[ ! -x "$bin" ]]; then
  cargo build --manifest-path "$repo_root/Cargo.toml" --release
fi

exec "$bin" "$@"
