from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "verify-vulkan-shader-constants.py"
SPIRV_MAGIC = 0x07230203
OP_TYPE_FLOAT = 22
OP_CONSTANT = 43
OP_FCONVERT = 115
OP_STORE = 62


def instruction(opcode: int, *operands: int) -> list[int]:
    return [((len(operands) + 1) << 16) | opcode, *operands]


def shader(constant: int) -> bytes:
    words = [SPIRV_MAGIC, 0x00010500, 0, 8, 0]
    words += instruction(OP_TYPE_FLOAT, 1, 32)
    words += instruction(OP_TYPE_FLOAT, 2, 16)
    words += instruction(OP_CONSTANT, 1, 3, constant)
    words += instruction(OP_FCONVERT, 2, 4, 3)
    words += instruction(OP_STORE, 5, 4)
    return struct.pack(f"<{len(words)}I", *words)


def shader_with_direct_half_zero() -> bytes:
    words = [SPIRV_MAGIC, 0x00010500, 0, 8, 0]
    words += instruction(OP_TYPE_FLOAT, 2, 16)
    words += instruction(OP_CONSTANT, 2, 3, 0)
    words += instruction(OP_STORE, 5, 3)
    return struct.pack(f"<{len(words)}I", *words)


class VulkanShaderConstantTests(unittest.TestCase):
    def run_verifier(
        self,
        diag: int = 0,
        tri: int = 0,
        missing: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        native_out = Path(temporary.name)
        for name, value in {"diag_f16": diag, "tri_f16": tri}.items():
            if name != missing:
                path = native_out / "build" / "vulkan-shaders.spv" / f"{name}.spv"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(shader(value))
        return subprocess.run(
            [sys.executable, str(SCRIPT), "--native-out", str(native_out)],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_accepts_zero_constants_for_both_half_float_shaders(self) -> None:
        result = self.run_verifier()

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout.strip(),
            "verified Vulkan half-float zero stores: diag_f16.spv, tri_f16.spv",
        )

    def test_accepts_a_direct_half_float_zero_store(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        native_out = Path(temporary.name)
        root = native_out / "vulkan-shaders.spv"
        root.mkdir()
        for name in ("diag_f16.spv", "tri_f16.spv"):
            (root / name).write_bytes(shader_with_direct_half_zero())

        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--native-out", str(native_out)],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rejects_infinity_before_half_float_conversion(self) -> None:
        result = self.run_verifier(diag=0x7F800000)

        self.assertEqual(result.returncode, 2)
        self.assertIn("diag_f16.spv", result.stderr)
        self.assertIn("0x7f800000", result.stderr)

    def test_rejects_nonzero_garbage_before_half_float_conversion(self) -> None:
        result = self.run_verifier(tri=0x7B2B93A8)

        self.assertEqual(result.returncode, 2)
        self.assertIn("tri_f16.spv", result.stderr)
        self.assertIn("0x7b2b93a8", result.stderr)

    def test_rejects_a_missing_shader_intermediate(self) -> None:
        result = self.run_verifier(missing="diag_f16")

        self.assertEqual(result.returncode, 2)
        self.assertIn("expected one diag_f16.spv, found 0", result.stderr)

    def test_rejects_duplicate_shader_intermediates(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        native_out = Path(temporary.name)
        for parent in ("first", "second"):
            root = native_out / parent / "vulkan-shaders.spv"
            root.mkdir(parents=True)
            (root / "diag_f16.spv").write_bytes(shader(0))
        root = native_out / "first" / "vulkan-shaders.spv"
        (root / "tri_f16.spv").write_bytes(shader(0))

        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--native-out", str(native_out)],
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("expected one diag_f16.spv, found 2", result.stderr)


if __name__ == "__main__":
    unittest.main()
