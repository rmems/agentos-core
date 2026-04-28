#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SAAQ_DISCOVERY_HOME="$repo_root"

bin="$repo_root/target/release/saaq-discovery"
if [[ ! -x "$bin" ]]; then
  cargo build --manifest-path "$repo_root/Cargo.toml" --release
fi

exec "$bin" "$@"
