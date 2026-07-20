#!/usr/bin/env bash
#
# The P2a conformance gate.
#
# Builds the release binary, then runs every row of
# `semantic-sidecar-protocol-v1.md` § Conformance (groups A, B, C) against that binary
# spawned as a real child process over stdio.
#
# Both pinned models must already be in the shared cache:
#   cargo run --release -- prepare --model qwen3-0.6b-f16
#   cargo run --release -- prepare --model bge-small-en-v1.5-f32
#
# Environment:
#   FIXTURES_DIR                  frozen fixture set (corpus + goldens)
#   JULIE_SIDECAR_FORCE_BACKEND   forced to cpu; goldens are CPU-generated
#
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export FIXTURES_DIR="${FIXTURES_DIR:-/Users/murphy/source/miller/eval/sidecar-conformance}"
export JULIE_SIDECAR_FORCE_BACKEND=cpu

if [[ ! -f "$FIXTURES_DIR/corpus.jsonl" ]]; then
  echo "conformance: FIXTURES_DIR does not hold corpus.jsonl: $FIXTURES_DIR" >&2
  exit 1
fi

echo "conformance: fixtures  $FIXTURES_DIR"
echo "conformance: backend   $JULIE_SIDECAR_FORCE_BACKEND"
echo "conformance: building release binary"
cargo build --release

echo "conformance: running groups A, B, and C against the release binary"
cargo test --release --test conformance -- --ignored --test-threads=1 --nocapture
