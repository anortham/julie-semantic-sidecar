#!/usr/bin/env python3

import argparse
import hashlib
import json
import sys
from pathlib import Path


CRATE = "llama-cpp-sys-2-0.1.151"
SHADER_ROOT = Path("llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders")
PATCHES = {
    SHADER_ROOT / "topk_moe.comp": (
        (
            b"const float INFINITY = 1.0 / 0.0;",
            b"#define NEGATIVE_INFINITY uintBitsToFloat(0xFF800000u)",
            1,
        ),
        (b"-INFINITY", b"NEGATIVE_INFINITY", 6),
    ),
    SHADER_ROOT / "copy_to_quant.comp": (
        (
            b"float vmin = 1.0/0.0;",
            b"float vmin = uintBitsToFloat(0x7F800000);",
            1,
        ),
    ),
    SHADER_ROOT / "flash_attn_split_k_reduce.comp": (
        (
            b"float m_max = -1.0/0.0;",
            b"float m_max = uintBitsToFloat(0xFF800000);",
            1,
        ),
    ),
}
PATCH_IDENTITY_PREFIX = "llama-cpp-sys-2-0.1.151:vulkan-infinity-v2"


def patch(vendor_root: Path) -> str:
    crate_root = vendor_root / CRATE
    checksum_path = crate_root / ".cargo-checksum.json"
    try:
        checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(
            f"cannot read vendored checksum manifest: {checksum_path}: {error}"
        ) from error
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid vendored checksum manifest: {checksum_path}: {error}") from error
    if not isinstance(checksums, dict) or not isinstance(checksums.get("files"), dict):
        raise ValueError(f"invalid vendored checksum manifest: {checksum_path}")

    patched_sources = {}
    for source_path, replacements in PATCHES.items():
        shader = crate_root / source_path
        try:
            source = shader.read_bytes()
        except OSError as error:
            raise ValueError(
                f"cannot read pinned native shader source: {shader}: {error}"
            ) from error
        for original, replacement, expected_count in replacements:
            if source.count(original) != expected_count or replacement in source:
                raise ValueError(
                    f"expected pinned unpatched infinity expressions in {shader}"
                )
        source_key = source_path.as_posix()
        expected_source_checksum = hashlib.sha256(source).hexdigest()
        if checksums.get("files", {}).get(source_key) != expected_source_checksum:
            raise ValueError(f"vendored checksum does not match {shader}")
        patched_source = source
        for original, replacement, _ in replacements:
            patched_source = patched_source.replace(original, replacement)
        patched_sources[source_path] = patched_source

    identity_digest = hashlib.sha256()
    for source_path, patched_source in sorted(patched_sources.items()):
        identity_digest.update(source_path.as_posix().encode())
        identity_digest.update(b"\0")
        identity_digest.update(patched_source)
        (crate_root / source_path).write_bytes(patched_source)
        checksums["files"][source_path.as_posix()] = hashlib.sha256(
            patched_source
        ).hexdigest()
    checksum_path.write_text(
        json.dumps(checksums, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    return f"{PATCH_IDENTITY_PREFIX}:{identity_digest.hexdigest()}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor-root", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        print(patch(arguments.vendor_root))
    except ValueError as error:
        print(f"patch-native-source: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
