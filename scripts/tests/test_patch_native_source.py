from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "patch-native-source.py"
CRATE = "llama-cpp-sys-2-0.1.151"
SHADER_ROOT = Path("llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders")
PATCHES = {
    SHADER_ROOT / "topk_moe.comp": (
        b"const float INFINITY = 1.0 / 0.0;\n",
        b"const float INFINITY = uintBitsToFloat(0x7F800000);\n",
    ),
    SHADER_ROOT / "copy_to_quant.comp": (
        b"float vmin = 1.0/0.0;\n",
        b"float vmin = uintBitsToFloat(0x7F800000);\n",
    ),
    SHADER_ROOT / "flash_attn_split_k_reduce.comp": (
        b"float m_max = -1.0/0.0;\n",
        b"float m_max = uintBitsToFloat(0xFF800000);\n",
    ),
}
PATCH_IDENTITY_PREFIX = "llama-cpp-sys-2-0.1.151:vulkan-infinity-v1"


def expected_patch_identity() -> str:
    digest = hashlib.sha256()
    for path, (_, patched) in sorted(PATCHES.items()):
        digest.update(path.as_posix().encode())
        digest.update(b"\0")
        digest.update(patched)
    return f"{PATCH_IDENTITY_PREFIX}:{digest.hexdigest()}"


class NativeSourcePatchTests(unittest.TestCase):
    def run_patch(
        self,
        replacements: dict[Path, bytes] | None = None,
        checksum_replacements: dict[Path, str] | None = None,
        missing_checksum: bool = False,
        checksum_document: object | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        vendor_root = Path(temporary.name)
        crate_root = vendor_root / CRATE
        sources = {path: original for path, (original, _) in PATCHES.items()}
        sources.update(replacements or {})
        for path, source in sources.items():
            shader = crate_root / path
            shader.parent.mkdir(parents=True, exist_ok=True)
            shader.write_bytes(source)
        checksum = crate_root / ".cargo-checksum.json"
        checksums = {
            path.as_posix(): hashlib.sha256(source).hexdigest()
            for path, source in sources.items()
        }
        checksums.update(
            {
                path.as_posix(): checksum
                for path, checksum in (checksum_replacements or {}).items()
            }
        )
        if not missing_checksum:
            checksum.write_text(
                json.dumps(
                    checksum_document
                    if checksum_document is not None
                    else {
                        "files": checksums,
                        "package": "pinned-package-checksum",
                    },
                ),
                encoding="utf-8",
            )
        result = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--vendor-root",
                str(vendor_root),
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        return result, crate_root, checksum

    def test_replaces_every_undefined_infinity_expression(self) -> None:
        result, crate_root, checksum = self.run_patch()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), expected_patch_identity())
        checksums = json.loads(checksum.read_text(encoding="utf-8"))
        for path, (_, patched) in PATCHES.items():
            self.assertEqual((crate_root / path).read_bytes(), patched)
            self.assertEqual(
                checksums["files"][path.as_posix()],
                hashlib.sha256(patched).hexdigest(),
            )
        self.assertEqual(checksums["package"], "pinned-package-checksum")

    def test_rejects_an_unexpected_upstream_expression(self) -> None:
        path = SHADER_ROOT / "topk_moe.comp"
        source = b"const float INFINITY = 42.0;\n"
        result, crate_root, checksum = self.run_patch({path: source})

        self.assertEqual(result.returncode, 2)
        self.assertIn("expected exactly one unpatched infinity expression", result.stderr)
        self.assertEqual((crate_root / path).read_bytes(), source)
        self.assertEqual(
            json.loads(checksum.read_text(encoding="utf-8"))["files"][path.as_posix()],
            hashlib.sha256(source).hexdigest(),
        )

    def test_rejects_an_already_patched_source(self) -> None:
        path = SHADER_ROOT / "topk_moe.comp"
        patched = PATCHES[path][1]
        result, crate_root, checksum = self.run_patch({path: patched})

        self.assertEqual(result.returncode, 2)
        self.assertIn("expected exactly one unpatched infinity expression", result.stderr)
        self.assertEqual((crate_root / path).read_bytes(), patched)
        self.assertEqual(
            json.loads(checksum.read_text(encoding="utf-8"))["files"][path.as_posix()],
            hashlib.sha256(patched).hexdigest(),
        )

    def test_rejects_a_mismatched_vendor_checksum_without_mutation(self) -> None:
        path = SHADER_ROOT / "topk_moe.comp"
        original = PATCHES[path][0]
        result, crate_root, _ = self.run_patch(
            checksum_replacements={path: "0" * 64}
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("vendored checksum does not match", result.stderr)
        self.assertEqual((crate_root / path).read_bytes(), original)

    def test_rejects_a_missing_vendor_checksum_manifest(self) -> None:
        path = SHADER_ROOT / "topk_moe.comp"
        original = PATCHES[path][0]
        result, crate_root, _ = self.run_patch(missing_checksum=True)

        self.assertEqual(result.returncode, 2)
        self.assertIn("cannot read vendored checksum manifest", result.stderr)
        self.assertEqual((crate_root / path).read_bytes(), original)

    def test_rejects_a_non_object_vendor_checksum_manifest(self) -> None:
        path = SHADER_ROOT / "topk_moe.comp"
        original = PATCHES[path][0]
        result, crate_root, _ = self.run_patch(checksum_document=[])

        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid vendored checksum manifest", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertEqual((crate_root / path).read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
