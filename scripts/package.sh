#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

profile=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) profile="${2:?--profile needs a value}"; shift 2 ;;
    *) echo "package: unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$profile" ]]; then
  echo "package: --profile is required" >&2
  exit 2
fi

case "$profile" in
  apple-arm64-metal-portable)
    target="aarch64-apple-darwin"; backend="metal"; tier="portable"; features="metal" ;;
  apple-x64-metal-portable)
    target="x86_64-apple-darwin"; backend="metal"; tier="portable"; features="metal" ;;
  linux-x64-vulkan-portable)
    target="x86_64-unknown-linux-gnu"; backend="vulkan"; tier="portable"; features="vulkan,dynamic-backends" ;;
  windows-x64-vulkan-portable)
    target="x86_64-pc-windows-msvc"; backend="vulkan"; tier="portable"; features="vulkan,dynamic-backends" ;;
  linux-x64-cuda-vendor)
    target="x86_64-unknown-linux-gnu"; backend="cuda"; tier="vendor"; features="cuda,dynamic-backends" ;;
  windows-x64-cuda-vendor)
    target="x86_64-pc-windows-msvc"; backend="cuda"; tier="vendor"; features="cuda,dynamic-backends" ;;
  *) echo "package: unknown profile: $profile" >&2; exit 2 ;;
esac

host_triple="$(rustc -vV | awk '/^host: /{print $2}')"
if [[ "$host_triple" != "$target" ]]; then
  echo "package: profile $profile must run on $target, current host is $host_triple" >&2
  exit 1
fi

effective_rustflags="${RUSTFLAGS:-} ${CARGO_ENCODED_RUSTFLAGS:-}"
compact_rustflags="${effective_rustflags//[[:space:]]/}"
if [[ "$compact_rustflags" == *"target-cpu=native"* ]]; then
  echo "package: effective Rust flags must not contain -Ctarget-cpu=native" >&2
  exit 1
fi

version="$(awk -F'"' '/^version = /{print $2; exit}' Cargo.toml)"
exe="julie-semantic-sidecar"
helper="julie-package-manifest"
archive_kind="tar.gz"
if [[ "$target" == *windows* ]]; then
  exe="$exe.exe"
  helper="$helper.exe"
  archive_kind="zip"
fi

build_messages="$(mktemp)"
helper_run_dir=""
cargo_target_dir="${CARGO_TARGET_DIR:-$repo_root/target}"
if [[ "$cargo_target_dir" != /* ]]; then
  cargo_target_dir="$repo_root/$cargo_target_dir"
fi
vendor_parent="$cargo_target_dir/package-vendor/$profile"
vendor_root="$vendor_parent/vendor"
vendor_config="$vendor_parent/config.toml"
rm -rf "$vendor_parent"
mkdir -p "$vendor_root"
cleanup() {
  rm -f "$build_messages"
  rm -rf "$vendor_parent"
  if [[ -n "$helper_run_dir" ]]; then
    rm -rf "$helper_run_dir"
  fi
}
trap cleanup EXIT
cargo vendor --locked --versioned-dirs "$vendor_root" >"$vendor_config"
export JULIE_NATIVE_PATCH_IDENTITY
JULIE_NATIVE_PATCH_IDENTITY="$(
  python3 scripts/patch-native-source.py --vendor-root "$vendor_root"
)"
if [[ "$JULIE_NATIVE_PATCH_IDENTITY" == *$'\n'* ]] ||
  [[ ! "$JULIE_NATIVE_PATCH_IDENTITY" =~ ^llama-cpp-sys-2-0\.1\.151:vulkan-infinity-v2:[0-9a-f]{64}$ ]]; then
  echo "package: native source patch returned an invalid identity" >&2
  exit 1
fi
cargo --config "$vendor_config" build --release --target "$target" --features "$features" --bins --message-format=json-render-diagnostics >"$build_messages"
native_out="$(python3 - "$build_messages" <<'PY'
import json, sys
found = []
with open(sys.argv[1], encoding="utf-8") as stream:
    for line in stream:
        message = json.loads(line)
        if message.get("reason") == "build-script-executed" and "llama-cpp-sys-2" in message.get("package_id", ""):
            found.append(message["out_dir"])
if len(found) != 1:
    raise SystemExit(f"expected one llama-cpp-sys out_dir, found {len(found)}")
print(found[0])
PY
)"

build_dir="$cargo_target_dir/$target/release"
stage_root="$repo_root/dist"
stage="$stage_root/$profile"
rm -rf "$stage"
mkdir -p "$stage"
cp "$build_dir/$exe" "$stage/$exe"
cp LICENSE "$stage/LICENSE"
cp README.md "$stage/README.md"
chmod +x "$stage/$exe"

copy_native_file() {
  local source="$1"
  local name
  name="$(basename "$source")"
  case "$name" in
    *.dll|*.so|*.so.*|*.dylib|*.dylib.*) cp -L "$source" "$stage/$name" ;;
  esac
}

if [[ "$features" == *dynamic-backends* ]]; then
  for native_lib_dir in "$native_out/lib" "$native_out/lib64"; do
    for source in "$native_lib_dir/"*; do
      [[ -e "$source" ]] && copy_native_file "$source"
    done
  done
  for source in "$native_out/backends/"*; do
    [[ -e "$source" ]] || continue
    name="$(basename "$source")"
    case "$name" in
      libggml-cpu*.so|ggml-cpu*.dll|libggml-cpu*.dylib) copy_native_file "$source" ;;
      "libggml-$backend.so"|"ggml-$backend.dll"|"libggml-$backend.dylib") copy_native_file "$source" ;;
    esac
  done
fi

helper_path="$build_dir/$helper"
if [[ "$features" == *dynamic-backends* ]]; then
  helper_run_dir="$(mktemp -d)"
  cp "$helper_path" "$helper_run_dir/$helper"
  for source in "$stage/"*; do
    [[ -e "$source" ]] && copy_name="$(basename "$source")"
    case "${copy_name:-}" in
      *.dll|*.so|*.so.*|*.dylib|*.dylib.*) cp -L "$source" "$helper_run_dir/$copy_name" ;;
    esac
  done
  helper_path="$helper_run_dir/$helper"
fi
"$helper_path" create --root "$stage" --target "$target" --tier "$tier" --backend "$backend"
"$helper_path" verify-patched --root "$stage"

if [[ "$target" == *linux* && "$features" == *dynamic-backends* ]]; then
  if ! readelf -d "$stage/$exe" | grep -E '(RPATH|RUNPATH).*[\$]ORIGIN' >/dev/null; then
    echo "package: Linux dynamic executable lacks an \$ORIGIN runpath" >&2
    exit 1
  fi
fi

archive_base="julie-semantic-sidecar-${version}-${target}-${backend}-${tier}"
archive_name="$archive_base.$archive_kind"
archive="$stage_root/$archive_name"
rm -f "$archive" "$archive.sha256"
python3 - "$stage" "$archive" "$archive_kind" <<'PY'
import gzip, pathlib, stat, sys, tarfile, zipfile
root, output, kind = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
files = sorted(path for path in root.iterdir() if path.is_file())
if kind == "zip":
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            info = zipfile.ZipInfo(path.name, (1980, 1, 1, 0, 0, 0))
            mode = 0o755 if path.name.endswith(".exe") else 0o644
            info.external_attr = (stat.S_IFREG | mode) << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
else:
    with output.open("wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for path in files:
                data = path.read_bytes()
                info = tarfile.TarInfo(path.name)
                info.size = len(data); info.mtime = 0; info.uid = 0; info.gid = 0
                info.uname = ""; info.gname = ""
                info.mode = 0o755 if path.name == "julie-semantic-sidecar" else 0o644
                import io
                archive.addfile(info, io.BytesIO(data))
PY

if command -v sha256sum >/dev/null 2>&1; then
  (cd "$stage_root" && sha256sum "$archive_name" >"$archive_name.sha256")
else
  (cd "$stage_root" && shasum -a 256 "$archive_name" >"$archive_name.sha256")
fi
echo "package: manifest $stage/package-manifest.json"
echo "package: archive  $archive"
echo "package: sha256   $(cat "$archive.sha256")"
