#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

binary=""
backend=""
fixtures_dir="${FIXTURES_DIR:-/Users/murphy/source/miller/eval/sidecar-conformance}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary) binary="${2:?--binary needs a value}"; shift 2 ;;
    --backend) backend="${2:?--backend needs a value}"; shift 2 ;;
    --fixtures) fixtures_dir="${2:?--fixtures needs a value}"; shift 2 ;;
    *) echo "conformance: unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$binary" || -z "$backend" ]]; then
  echo "usage: scripts/conformance.sh --binary <unpacked-sidecar> --backend <cpu|metal|vulkan|cuda> [--fixtures <dir>]" >&2
  exit 2
fi
case "$backend" in
  cpu|metal|vulkan|cuda) ;;
  *) echo "conformance: unsupported backend: $backend" >&2; exit 2 ;;
esac
if [[ ! -x "$binary" ]]; then
  echo "conformance: binary is not executable: $binary" >&2
  exit 1
fi
if [[ ! -f "$fixtures_dir/corpus.jsonl" ]]; then
  echo "conformance: fixtures do not hold corpus.jsonl: $fixtures_dir" >&2
  exit 1
fi

export FIXTURES_DIR="$fixtures_dir"
export JULIE_CONFORMANCE_BIN="$binary"
export JULIE_SIDECAR_FORCE_BACKEND="$backend"

echo "conformance: binary   $JULIE_CONFORMANCE_BIN"
echo "conformance: fixtures $FIXTURES_DIR"
echo "conformance: backend  $JULIE_SIDECAR_FORCE_BACKEND"
cargo test --release --test conformance -- --ignored --test-threads=1 --nocapture
