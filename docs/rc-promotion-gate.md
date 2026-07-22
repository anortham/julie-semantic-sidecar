# RC -> v0.1.0 promotion gate

The checklist a release candidate must pass before it is promoted from a prerelease
(`0.1.0-rc.N`) to a non-prerelease (`0.1.0`). **Promotion itself is a user decision** — this
gate produces the evidence; it does not authorize the release. Run every applicable item on real
target hardware against the exact archive checksum under consideration before proposing promotion.

## Gate items

1. **Protocol conformance suite** — `scripts/conformance.sh` passes (groups A, B, C of
   `semantic-sidecar-protocol-v1.md` § Conformance). No changes to its pass rule.
2. **Unit tests, both feature sets** — `cargo test` and `cargo test --features metal` both green.
   Every added supported build profile also runs its profile-specific tests; the existing assertions
   are not weakened.
3. **Package manifest and archive integrity** — the unpacked archive matches its machine-readable
   manifest: target, portability tier, advertised backend, sidecar version, native build identity,
   file inventory, per-file checksums, and archive SHA-256. The archive contains no model weights,
   development paths, or undeclared native libraries.
4. **Packaged smoke** — the release archive built by `scripts/package.sh --smoke` unpacks, its
   bundled binary answers `--version`, and an offline `health` probe against an empty cache dir
   reports `ready: false` / `degraded_reason: "model_not_prepared"` (the archive ships no model,
   so the smoke proves the fail-loud path — a `ready: false` here is the expected pass).
5. **Real-device archive proof** — the exact archive SHA-256 passes ready `health`, `embed_query`,
   `embed_batch`, shutdown, and every applicable golden vector on a real compatible GPU. Record the
   GPU model, driver, runtime, selected backend, raw results, and package manifest. Compilation,
   archive creation, and software rendering do not satisfy this item.
6. **CPU fallback from the same archive** — the exact archive SHA-256 remains ready and reports CPU
   with a non-null degradation reason when its accelerator is forced unavailable or fails to load.
   Run this for every supported portable and vendor-specific archive.
7. **Throughput floor (this document)** — the packaged binary sustains
   **≥ 40 units/s steady-state on the M2 Ultra reference machine (64-text batches, warm model)**,
   measured by `scripts/bench-throughput.py`.

Items defined by scripts retain their script-level pass rules. This document adds the checksum-bound
package, hardware, fallback, and throughput promotion rules.

## Portable and vendor-specific lanes

Portable archives are the default consumer artifacts and must prove both their named accelerator and
CPU fallback: Metal on macOS arm64, Vulkan on Windows x64, and Vulkan on Linux x64. Evidence for one
platform or archive checksum never promotes another.

CUDA, HIP/ROCm, and SYCL archives are vendor-specific lanes. They do not substitute for portable-lane
proof and become supported only after their own exact-checksum real-device, golden-vector, fallback,
and performance evidence exists. A vendor lane must also show a material benefit over the matching
portable archive. HIP/ROCm or SYCL health fields additionally require the Miller-owned wire-contract
amendment before those lanes can report support truthfully.

## Throughput floor — the check

One command, run on the target machine against the binary unpacked from the checksum being promoted:

```
scripts/bench-throughput.py --binary /path/to/unpacked/julie-semantic-sidecar
```

It probes `health` and **fails the bench** unless the sidecar reports `ready: true` — a
`model_not_prepared` binary can never pass, so the gate cannot be satisfied by measuring zeros.
It then times `embed_batch` rounds after a discarded warm-up round and prints steady-state
units/s with a PASS/FAIL verdict against the floor (default `40`, overridable with `--floor`).
Exit code: `0` PASS, `1` below floor, `2` not-ready / bad arguments / protocol error. Use
`--json` for machine-readable output; `--batch`/`--rounds` to vary the shape (batch is capped at
the protocol maximum of 250).

Record the archive SHA-256 and measured steady-state number in the promotion evidence, not just the
PASS. The 40 units/s value is the pre-change Qwen-derived minimum; a default-model change requires
fresh exact-archive measurement and may tighten the floor, but cannot inherit an older model's result.

## The floor: 40 units/s

| Measurement | units/s | Machine |
|---|---|---|
| rc.2 steady-state, 64-text batches | 78.9 | M2 Ultra |
| rc.2 steady-state, 250-text batches | 77.4 | M2 Ultra |
| P0 llama-server reference floor | 52.3 | M2 Ultra |
| **Gate floor** | **40** | M2 Ultra |
| CPU-only backend regression | ~6.6 | M2 Ultra |

**40 is roughly half of rc.2's observed rate.** It sits well below the healthy Metal-backed
range (77–89 units/s in repeat runs) and the P0 llama-server reference (52.3), so machine noise
and normal run-to-run variance never trip it. It sits far **above** a CPU-only regression (~6.6,
about 12× under the floor), so a backend that silently falls back to CPU — the exact failure this
gate exists to catch — fails loudly.

## Why this gate exists (the rc.1 lesson)

**Harness numbers are not engine numbers.** A CPU-only RC shipped at roughly **12× under the
design throughput floor** because the throughput that had been validated was a benchmark
harness's, not the shipping engine's on the real artifact. The full record is in Miller's
`docs/findings/2026-07-20-first-real-shadow-converge-benchmark.md`.

The correction is this gate: the floor is measured on the **target machine**, against the
**checksum-identified packaged binary**, over the **real embedding path** (`health` + `embed_batch`
over stdio) — not a harness, not a synthetic microbenchmark, not a different backend. A binary that
cannot clear 40 units/s on the reference machine does not get promoted, regardless of what any other
benchmark reported.
