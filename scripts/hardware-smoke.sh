#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

archive=""
expected_sha256=""
backend=""
lane=""
cache_dir=""
fixtures_dir="${FIXTURES_DIR:-/Users/murphy/source/miller/eval/sidecar-conformance}"
evidence_dir=""
artifact_validation=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive) archive="${2:?--archive needs a value}"; shift 2 ;;
    --sha256) expected_sha256="${2:?--sha256 needs a value}"; shift 2 ;;
    --backend) backend="${2:?--backend needs a value}"; shift 2 ;;
    --lane) lane="${2:?--lane needs a value}"; shift 2 ;;
    --cache-dir) cache_dir="${2:?--cache-dir needs a value}"; shift 2 ;;
    --fixtures) fixtures_dir="${2:?--fixtures needs a value}"; shift 2 ;;
    --evidence-dir) evidence_dir="${2:?--evidence-dir needs a value}"; shift 2 ;;
    --artifact-validation) artifact_validation=1; shift ;;
    *) echo "hardware-smoke: unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$archive" || -z "$expected_sha256" || -z "$backend" || -z "$lane" ]]; then
  echo "usage: scripts/hardware-smoke.sh --archive <path> --sha256 <hex> --backend <metal|vulkan|cuda> --lane <name> [--cache-dir <dir>] [--fixtures <dir>] [--evidence-dir <dir>] [--artifact-validation]" >&2
  exit 2
fi
case "$backend" in
  metal|vulkan|cuda) ;;
  *) echo "hardware-smoke: unsupported backend: $backend" >&2; exit 2 ;;
esac
expected_sha256="$(printf '%s' "$expected_sha256" | tr '[:upper:]' '[:lower:]')"
if [[ ! "$expected_sha256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "hardware-smoke: --sha256 must be exactly 64 hexadecimal characters" >&2
  exit 2
fi
if [[ ! -f "$archive" ]]; then
  echo "hardware-smoke: archive does not exist: $archive" >&2
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  actual_sha256="$(sha256sum "$archive" | awk '{print $1}')"
else
  actual_sha256="$(shasum -a 256 "$archive" | awk '{print $1}')"
fi
if [[ "$actual_sha256" != "$expected_sha256" ]]; then
  echo "hardware-smoke: archive checksum $actual_sha256 does not match $expected_sha256" >&2
  exit 1
fi

unpack_dir="$(mktemp -d)"
empty_cache="$(mktemp -d)"
cleanup() {
  rm -rf "$unpack_dir" "$empty_cache"
}
trap cleanup EXIT

python3 - "$archive" "$unpack_dir" <<'PY'
import os, pathlib, stat, sys, tarfile, zipfile

archive = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2])

def safe_name(name):
    path = pathlib.PurePosixPath(name)
    if "\\" in name or path.is_absolute() or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        raise SystemExit(f"archive member is not flat and safe: {name!r}")
    return path.name

if archive.name.endswith(".zip"):
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            name = safe_name(member.filename)
            if member.is_dir():
                raise SystemExit(f"archive contains a directory: {member.filename!r}")
            destination = root / name
            destination.write_bytes(source.read(member))
            mode = (member.external_attr >> 16) & 0o777
            destination.chmod(mode or 0o644)
else:
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            name = safe_name(member.name)
            if not member.isfile():
                raise SystemExit(f"archive member is not a regular file: {member.name!r}")
            stream = source.extractfile(member)
            if stream is None:
                raise SystemExit(f"archive member cannot be read: {member.name!r}")
            destination = root / name
            destination.write_bytes(stream.read())
            destination.chmod(member.mode & 0o777)
PY

binary="$unpack_dir/julie-semantic-sidecar"
if [[ -f "$unpack_dir/julie-semantic-sidecar.exe" ]]; then
  binary="$unpack_dir/julie-semantic-sidecar.exe"
fi
if [[ ! -x "$binary" ]]; then
  echo "hardware-smoke: unpacked archive has no executable sidecar" >&2
  exit 1
fi
manifest="$unpack_dir/package-manifest.json"
if [[ ! -f "$manifest" ]]; then
  echo "hardware-smoke: unpacked archive has no package-manifest.json" >&2
  exit 1
fi

python3 - "$unpack_dir" "$backend" <<'PY'
import hashlib, json, pathlib, sys

root = pathlib.Path(sys.argv[1])
backend = sys.argv[2]
manifest = json.loads((root / "package-manifest.json").read_text(encoding="utf-8"))
if manifest.get("advertised_backend") != backend:
    raise SystemExit(f"manifest backend {manifest.get('advertised_backend')!r} does not match {backend!r}")
declared = {"package-manifest.json"}
for item in manifest.get("files", []):
    name = item.get("path", "")
    if pathlib.PurePath(name).name != name:
        raise SystemExit(f"manifest path is not flat: {name!r}")
    path = root / name
    if not path.is_file():
        raise SystemExit(f"manifest payload is missing: {name!r}")
    data = path.read_bytes()
    if len(data) != item.get("size"):
        raise SystemExit(f"manifest size mismatch: {name!r}")
    if hashlib.sha256(data).hexdigest() != item.get("sha256"):
        raise SystemExit(f"manifest checksum mismatch: {name!r}")
    if item.get("role") == "model_weight":
        raise SystemExit(f"model weight declared in archive: {name!r}")
    declared.add(name)
actual = {path.name for path in root.iterdir() if path.is_file()}
if actual != declared:
    raise SystemExit(f"archive inventory differs from manifest: actual={sorted(actual)} declared={sorted(declared)}")
PY

if [[ -z "$evidence_dir" ]]; then
  evidence_dir="$repo_root/hardware-evidence/${lane}-${actual_sha256:0:12}"
fi
mkdir -p "$evidence_dir/raw-logs"
cp "$manifest" "$evidence_dir/package-manifest.json"
printf '%s  %s\n' "$actual_sha256" "$(basename "$archive")" >"$evidence_dir/archive.sha256"
{
  printf 'hardware_lane=%s\n' "$lane"
  printf 'advertised_backend=%s\n' "$backend"
  printf 'archive=%s\n' "$(basename "$archive")"
  printf 'archive_sha256=%s\n' "$actual_sha256"
  printf 'host=%s\n' "$(uname -a)"
  printf 'recorded_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >"$evidence_dir/identity.txt"
"$binary" --version >"$evidence_dir/raw-logs/version.log" 2>&1

protocol_smoke() {
  local label="$1"
  local smoke_cache="$2"
  local forced_backend="$3"
  local expectation="$4"
  local -a smoke_env=(env)
  if [[ -z "$forced_backend" ]]; then
    smoke_env+=(-u JULIE_SIDECAR_FORCE_BACKEND)
  else
    smoke_env+=("JULIE_SIDECAR_FORCE_BACKEND=$forced_backend")
  fi
  smoke_env+=("JULIE_EMBEDDING_CACHE_DIR=$smoke_cache")
  "${smoke_env[@]}" python3 - "$binary" "$expectation" "$backend" \
    >"$evidence_dir/raw-logs/$label.stdout.jsonl" \
    2>"$evidence_dir/raw-logs/$label.stderr.log" <<'PY'
import json, os, subprocess, sys

binary, expectation, advertised = sys.argv[1:]
requests = [
    {"schema":"julie.embedding.sidecar","version":1,"request_id":"health","method":"health","params":{}},
]
if expectation != "absent":
    requests += [
        {"schema":"julie.embedding.sidecar","version":1,"request_id":"query","method":"embed_query","params":{"text":"archive query smoke"}},
        {"schema":"julie.embedding.sidecar","version":1,"request_id":"batch","method":"embed_batch","params":{"texts":["archive batch one","archive batch two"]}},
    ]
requests.append({"schema":"julie.embedding.sidecar","version":1,"request_id":"shutdown","method":"shutdown","params":{}})
process = subprocess.Popen([binary, "serve"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=os.environ.copy())
payload = "\n".join(json.dumps(request, separators=(",", ":")) for request in requests) + "\n"
stdout, stderr = process.communicate(payload, timeout=900)
sys.stderr.write(stderr)
if process.returncode != 0:
    raise SystemExit(f"sidecar exited {process.returncode}")
lines = [line for line in stdout.splitlines() if line]
if len(lines) != len(requests):
    raise SystemExit(f"stdout purity/count failure: expected {len(requests)} protocol lines, got {len(lines)}: {stdout!r}")
responses = [json.loads(line) for line in lines]
health = responses[0].get("result", {})
if expectation == "absent":
    if health.get("ready") is not False or health.get("degraded_reason") != "model_not_prepared":
        raise SystemExit(f"absent-model health mismatch: {responses[0]}")
else:
    if health.get("ready") is not True:
        raise SystemExit(f"prepared health mismatch: {responses[0]}")
    if expectation == "accelerated" and (health.get("resolved_backend") != advertised or health.get("accelerated") is not True):
        raise SystemExit(f"accelerator health mismatch: {responses[0]}")
    if expectation == "accelerated" and any(marker in str(health.get("device", "")).lower() for marker in ("llvmpipe", "lavapipe", "swiftshader", "software rasterizer", "microsoft basic render")):
        raise SystemExit(f"software device was selected: {responses[0]}")
    if expectation == "cpu" and (health.get("resolved_backend") != "cpu" or health.get("accelerated") is not False):
        raise SystemExit(f"forced CPU health mismatch: {responses[0]}")
    if expectation == "fallback" and (health.get("resolved_backend") != "cpu" or health.get("accelerated") is not False or not health.get("degraded_reason")):
        raise SystemExit(f"fallback health mismatch: {responses[0]}")
    dims = health.get("dims")
    if len(responses[1].get("result", {}).get("vector", [])) != dims:
        raise SystemExit(f"query dimensions mismatch: {responses[1]}")
    vectors = responses[2].get("result", {}).get("vectors", [])
    if len(vectors) != 2 or any(len(vector) != dims for vector in vectors):
        raise SystemExit(f"batch shape mismatch: {responses[2]}")
if responses[-1].get("result") != {"stopping": True}:
    raise SystemExit(f"shutdown mismatch: {responses[-1]}")
sys.stdout.write(stdout)
PY
}

protocol_smoke "absent-model" "$empty_cache" "cpu" "absent"

if [[ "$artifact_validation" == "1" ]]; then
  printf 'artifact_validation=passed\nsupport_evidence=false\n' >>"$evidence_dir/identity.txt"
  echo "hardware-smoke: artifact validation passed; this is not support evidence"
  echo "hardware-smoke: evidence $evidence_dir"
  exit 0
fi

if [[ -z "$cache_dir" ]]; then
  echo "hardware-smoke: --cache-dir is required for real-device proof" >&2
  exit 2
fi
mkdir -p "$cache_dir"
if [[ ! -f "$fixtures_dir/corpus.jsonl" ]]; then
  echo "hardware-smoke: fixtures do not hold corpus.jsonl: $fixtures_dir" >&2
  exit 1
fi

case "$backend" in
  metal)
    [[ "$(uname -s)" == "Darwin" ]] || { echo "hardware-smoke: Metal requires Darwin" >&2; exit 1; }
    system_profiler SPDisplaysDataType >"$evidence_dir/raw-logs/device.txt"
    sw_vers >"$evidence_dir/raw-logs/runtime.txt"
    fallback_backend="vulkan"
    ;;
  vulkan)
    command -v vulkaninfo >/dev/null || { echo "hardware-smoke: vulkaninfo is required" >&2; exit 1; }
    vulkaninfo --summary >"$evidence_dir/raw-logs/device.txt" 2>&1
    { uname -a; vulkaninfo --summary; } >"$evidence_dir/raw-logs/runtime.txt" 2>&1
    fallback_backend="metal"
    ;;
  cuda)
    command -v nvidia-smi >/dev/null || { echo "hardware-smoke: nvidia-smi is required" >&2; exit 1; }
    nvidia-smi -q >"$evidence_dir/raw-logs/device.txt" 2>&1
    { uname -a; nvidia-smi; } >"$evidence_dir/raw-logs/runtime.txt" 2>&1
    fallback_backend="metal"
    ;;
esac
device_report="$evidence_dir/raw-logs/device.txt"
if grep -Eiq 'llvmpipe|lavapipe|swiftshader|software rasterizer|microsoft basic render' "$device_report" \
  && { [[ "$backend" != "vulkan" ]] || ! grep -Eq 'deviceType.*PHYSICAL_DEVICE_TYPE_(INTEGRATED|DISCRETE)_GPU' "$device_report"; }; then
  echo "hardware-smoke: software renderer is not real-device evidence" >&2
  exit 1
fi

prepare_model() {
  local model="$1"
  local log="$evidence_dir/raw-logs/prepare-$model.log"
  : >"$log"
  for attempt in 1 2 3; do
    if JULIE_EMBEDDING_CACHE_DIR="$cache_dir" "$binary" prepare --model "$model" >>"$log" 2>&1; then
      return 0
    fi
    printf 'hardware-smoke: prepare attempt %s failed for %s\n' "$attempt" "$model" >>"$log"
    [[ "$attempt" == "3" ]] || sleep $((attempt * 30))
  done
  echo "hardware-smoke: prepare failed after 3 attempts: $model" >&2
  return 1
}

for model in bge-small-en-v1.5-f32 qwen3-0.6b-f16; do
  prepare_model "$model"
done

selection_cache="$cache_dir/backend-selection.json"
rm -f "$selection_cache"
protocol_smoke "selection-rebuild" "$cache_dir" "" "accelerated"
if grep -Eiq 'using device .*\b(llvmpipe|lavapipe|swiftshader|software rasterizer|microsoft basic render)\b' \
  "$evidence_dir/raw-logs/selection-rebuild.stderr.log"; then
  echo "hardware-smoke: software device was selected" >&2
  exit 1
fi
if [[ ! -f "$selection_cache" ]]; then
  echo "hardware-smoke: selection cache was not rebuilt" >&2
  exit 1
fi
if command -v sha256sum >/dev/null 2>&1; then
  selection_before="$(sha256sum "$selection_cache" | awk '{print $1}')"
else
  selection_before="$(shasum -a 256 "$selection_cache" | awk '{print $1}')"
fi
protocol_smoke "selection-reuse" "$cache_dir" "" "accelerated"
if command -v sha256sum >/dev/null 2>&1; then
  selection_after="$(sha256sum "$selection_cache" | awk '{print $1}')"
else
  selection_after="$(shasum -a 256 "$selection_cache" | awk '{print $1}')"
fi
if [[ "$selection_before" != "$selection_after" ]]; then
  echo "hardware-smoke: cached selection changed during reuse" >&2
  exit 1
fi

protocol_smoke "forced-cpu" "$cache_dir" "cpu" "cpu"
protocol_smoke "fallback" "$cache_dir" "$fallback_backend" "fallback"

JULIE_EMBEDDING_CACHE_DIR="$cache_dir" JULIE_CONFORMANCE_UNAVAILABLE_BACKEND="$fallback_backend" bash scripts/conformance.sh --binary "$binary" --backend cpu --fixtures "$fixtures_dir" \
  >"$evidence_dir/raw-logs/conformance-cpu.log" 2>&1
JULIE_EMBEDDING_CACHE_DIR="$cache_dir" JULIE_CONFORMANCE_UNAVAILABLE_BACKEND="$fallback_backend" bash scripts/conformance.sh --binary "$binary" --backend "$backend" --fixtures "$fixtures_dir" \
  >"$evidence_dir/raw-logs/conformance-$backend.log" 2>&1

for measured_backend in cpu "$backend"; do
  JULIE_EMBEDDING_CACHE_DIR="$cache_dir" JULIE_SIDECAR_FORCE_BACKEND="$measured_backend" \
    python3 scripts/bench-throughput.py --binary "$binary" --batch 1 --rounds 4 --floor 0 \
    --expect-backend "$measured_backend" --json \
    >"$evidence_dir/raw-logs/bench-$measured_backend-batch-1.json"
  JULIE_EMBEDDING_CACHE_DIR="$cache_dir" JULIE_SIDECAR_FORCE_BACKEND="$measured_backend" \
    python3 scripts/bench-throughput.py --binary "$binary" --batch 16 --rounds 4 --floor 0 \
    --expect-backend "$measured_backend" --json \
    >"$evidence_dir/raw-logs/bench-$measured_backend-batch-16.json"
done

cp "$selection_cache" "$evidence_dir/backend-selection.json"
printf 'artifact_validation=passed\nsupport_evidence=real-device-pending-review\n' >>"$evidence_dir/identity.txt"
echo "hardware-smoke: real-device evidence captured for manual review"
echo "hardware-smoke: evidence $evidence_dir"
