#!/usr/bin/env bash
#
# Release layout builder for julie-semantic-sidecar.
#
# FROZEN LAYOUT RULE — do not change without changing src/backend_select.rs:
#
#   Everything the binary loads at runtime lives in the SAME directory as the
#   executable. `backend_select::plugin_dir()` is `current_exe().parent()`, so any
#   ggml backend plugin module (libggml-vulkan.so / ggml-vulkan.dll / libggml-metal.dylib)
#   and any non-system shared library MUST be copied next to the executable — never
#   into a lib/ or bin/ subdirectory, and never left to an rpath outside the archive.
#
#   Archive root layout (one flat directory per target triple):
#     julie-semantic-sidecar[.exe]      the executable
#     <backend plugin modules>          TODO: accelerated builds only — see below
#     <bundled shared libraries>        TODO: accelerated builds only — see below
#     LICENSE
#     README.md
#
# CURRENT BUILD = CPU-ONLY. `Cargo.toml` pins llama-cpp-2/llama-cpp-sys-2 =0.1.151 with
# `default-features = false`, which statically links ggml-cpu. That build ships exactly one
# file plus the docs, so the flat layout is trivially satisfied today.
#
# TODO — accelerated builds (exact flags from the plan's Global Constraints; NOT enabled
# here because they cannot be built or tested on this machine/leg):
#   macOS arm64   : --features metal, embedded Metal shaders, CMake -DGGML_NATIVE=OFF
#   Linux/Windows : --features vulkan with backend-DL (GGML_BACKEND_DL), CMake -DGGML_NATIVE=OFF
#                   → copy the produced ggml backend plugin module next to the executable
#                     and add it to the per-platform file list below.
#   macOS x64     : stays CPU-only.
# `-DGGML_NATIVE=OFF` is required on every leg for cross-machine determinism once the
# build stops being a plain `cargo build`.
#
# Usage: scripts/package.sh [--smoke] [--target <triple>]
#   --smoke   run the packaged smoke (--version + offline not-ready health) before archiving
#
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

run_smoke=0
target=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) run_smoke=1; shift ;;
    --target) target="${2:?--target needs a triple}"; shift 2 ;;
    *) echo "package: unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$target" ]]; then
  target="$(rustc -vV | awk '/^host: /{print $2}')"
fi
version="$(awk -F'"' '/^version = /{print $2; exit}' Cargo.toml)"

exe="julie-semantic-sidecar"
archive_kind="tar.gz"
case "$target" in
  *windows*) exe="julie-semantic-sidecar.exe"; archive_kind="zip" ;;
esac

echo "package: version   $version"
echo "package: target    $target"
echo "package: archive   $archive_kind"

echo "package: building release binary (CPU-only feature set)"
cargo build --release

stage_root="$repo_root/dist"
stage="$stage_root/$target"
rm -rf "$stage"
mkdir -p "$stage"

cp "target/release/$exe" "$stage/$exe"
cp LICENSE "$stage/LICENSE"
cp README.md "$stage/README.md"
chmod +x "$stage/$exe"

echo "package: staged layout"
ls -l "$stage"

if [[ "$run_smoke" == "1" ]]; then
  echo "package: smoke — packaged --version"
  "$stage/$exe" --version

  echo "package: smoke — offline not-ready health from an empty cache dir"
  smoke_cache="$(mktemp -d)"
  smoke_out="$(
    printf '%s\n%s\n' \
      '{"schema":"julie.embedding.sidecar","version":1,"request_id":"smoke-health","method":"health","params":{}}' \
      '{"schema":"julie.embedding.sidecar","version":1,"request_id":"smoke-stop","method":"shutdown","params":{}}' \
      | JULIE_EMBEDDING_CACHE_DIR="$smoke_cache" "$stage/$exe" serve
  )"
  rm -rf "$smoke_cache"
  echo "$smoke_out"
  case "$smoke_out" in
    *'"ready":false'*) ;;
    *) echo "package: smoke FAILED — health did not report ready:false" >&2; exit 1 ;;
  esac
  case "$smoke_out" in
    *'"degraded_reason":"model_not_prepared"'*) ;;
    *) echo "package: smoke FAILED — health did not report model_not_prepared" >&2; exit 1 ;;
  esac
  case "$smoke_out" in
    *'"stopping":true'*) ;;
    *) echo "package: smoke FAILED — shutdown did not answer" >&2; exit 1 ;;
  esac
  echo "package: smoke OK"
fi

archive_base="julie-semantic-sidecar-${version}-${target}"
cd "$stage_root"
rm -f "$archive_base.$archive_kind" "$archive_base.$archive_kind.sha256"

if [[ "$archive_kind" == "zip" ]]; then
  if command -v 7z >/dev/null 2>&1; then
    7z a -tzip "$archive_base.zip" "./$target/*" >/dev/null
  else
    (cd "$target" && zip -q -r "../$archive_base.zip" .)
  fi
else
  tar -czf "$archive_base.tar.gz" -C "$target" .
fi

archive="$archive_base.$archive_kind"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$archive" > "$archive.sha256"
else
  shasum -a 256 "$archive" > "$archive.sha256"
fi

echo "package: archive   $stage_root/$archive"
echo "package: sha256    $(cat "$archive.sha256")"
