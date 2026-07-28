from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path


SPIRV_MAGIC = 0x07230203
OP_TYPE_FLOAT = 22
OP_CONSTANT = 43
OP_CONSTANT_NULL = 46
OP_STORE = 62
OP_FCONVERT = 115
SHADERS = ("diag_f16.spv", "tri_f16.spv")


def instructions(data: bytes) -> list[list[int]]:
    if len(data) < 20 or len(data) % 4 != 0:
        raise ValueError("invalid SPIR-V byte length")
    words = struct.unpack(f"<{len(data) // 4}I", data)
    if words[0] != SPIRV_MAGIC:
        raise ValueError("invalid SPIR-V magic")
    parsed = []
    offset = 5
    while offset < len(words):
        word_count = words[offset] >> 16
        if word_count == 0 or offset + word_count > len(words):
            raise ValueError(f"invalid SPIR-V instruction at word {offset}")
        parsed.append(list(words[offset : offset + word_count]))
        offset += word_count
    return parsed


def constant_half_store(shader: Path) -> int:
    parsed = instructions(shader.read_bytes())
    float_widths: dict[int, int] = {}
    constants: dict[int, tuple[int, int]] = {}
    null_constants: dict[int, int] = {}
    half_conversions: dict[int, int] = {}

    for item in parsed:
        opcode = item[0] & 0xFFFF
        if opcode == OP_TYPE_FLOAT and len(item) == 3:
            float_widths[item[1]] = item[2]
        elif opcode == OP_CONSTANT and len(item) >= 4:
            constants[item[2]] = (item[1], item[3])
        elif opcode == OP_CONSTANT_NULL and len(item) == 3:
            null_constants[item[2]] = item[1]
        elif opcode == OP_FCONVERT and len(item) == 4:
            if float_widths.get(item[1]) == 16:
                half_conversions[item[2]] = item[3]

    stored_values = []
    for item in parsed:
        opcode = item[0] & 0xFFFF
        if opcode != OP_STORE or len(item) < 3:
            continue
        value_id = item[2]
        if value_id in null_constants and float_widths.get(null_constants[value_id]) == 16:
            stored_values.append(0)
            continue
        direct = constants.get(value_id)
        if direct is not None and float_widths.get(direct[0]) == 16:
            stored_values.append(direct[1] & 0xFFFF)
            continue
        source_id = half_conversions.get(value_id)
        source = constants.get(source_id) if source_id is not None else None
        if source is not None and float_widths.get(source[0]) == 32:
            stored_values.append(source[1])
        elif source_id in null_constants and float_widths.get(null_constants[source_id]) == 32:
            stored_values.append(0)

    if len(stored_values) != 1:
        raise ValueError(
            f"{shader.name}: expected one constant-derived half-float store, "
            f"found {len(stored_values)}"
        )
    return stored_values[0]


def verify(native_out: Path) -> None:
    for name in SHADERS:
        matches = sorted(native_out.rglob(name))
        if len(matches) != 1:
            raise ValueError(f"expected one {name}, found {len(matches)}")
        value = constant_half_store(matches[0])
        if value != 0:
            raise ValueError(
                f"{name}: expected zero before the half-float store, found 0x{value:08x}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-out", type=Path, required=True)
    args = parser.parse_args()
    try:
        verify(args.native_out)
    except (OSError, ValueError) as error:
        print(f"verify Vulkan shader constants: {error}", file=sys.stderr)
        return 2
    print(
        "verified Vulkan half-float zero stores: "
        + ", ".join(SHADERS)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
