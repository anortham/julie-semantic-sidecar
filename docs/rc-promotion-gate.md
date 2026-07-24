# RC -> v0.1.0 promotion gate

The checklist a release candidate must pass before it is promoted from a prerelease
(`0.1.0-rc.N`) to a non-prerelease (`0.1.0`). **Promotion itself is a user decision** — this
gate produces the evidence; it does not authorize the release. Run every applicable item on real
target hardware against the exact archive checksum under consideration before proposing promotion.

## Gate items

1. **Protocol conformance suite** — `scripts/conformance.sh --binary <unpacked-binary> --backend
   <cpu|metal|vulkan|cuda>` passes groups A, B, and C of `semantic-sidecar-protocol-v1.md` for CPU
   and the advertised accelerator. Both runs use the same Rust harness and BGE/Qwen golden vectors.
2. **Unit tests, both feature sets** — `cargo test` and `cargo test --features metal` both green.
   Every added supported build profile also runs its profile-specific tests; the existing assertions
   are not weakened.
3. **Package manifest and archive integrity** — the unpacked archive matches its machine-readable
   manifest: target, portability tier, advertised backend, sidecar version, native build identity,
   file inventory, per-file checksums, and archive SHA-256. The archive contains no model weights,
   development paths, or undeclared native libraries.
4. **Unpacked artifact validation** — `scripts/hardware-smoke.sh --artifact-validation` or
   `scripts/hardware-smoke.ps1 -ArtifactValidation` verifies the supplied SHA-256, extracts into a
   new temporary directory, validates the flat manifest inventory, runs that extracted binary's
   `--version`, and proves absent-model health plus clean shutdown and stdout purity. This compile
   and package proof is explicitly not support evidence.
5. **Real-device archive proof** — the same command without artifact-validation prepares BGE and
   Qwen, passes ready `health`, `embed_query`, `embed_batch`, shutdown, stdout purity, and both models'
   complete conformance vectors on CPU and the advertised accelerator. It rejects software renderers
   and records the GPU, driver, runtime, exact archive SHA-256, manifest, and raw logs.
6. **CPU fallback from the same archive** — the exact archive SHA-256 remains ready and reports CPU
   with a non-null degradation reason when its accelerator is forced unavailable or fails to load.
   Run this for every supported portable and vendor-specific archive.
7. **Selection and measurements** — remove and rebuild `backend-selection.json`, prove a second start
   reuses the same selection, and record fixed batch-1 and 16-text indexing-batch measurements for CPU
   and accelerator. The accelerator may resolve by default only when the first-start benchmark wins.
8. **Throughput floor (this document)** — the packaged binary sustains
   **≥ 40 units/s steady-state on the M2 Ultra reference machine (64-text batches, warm model)**,
   measured by `scripts/bench-throughput.py`.
9. **Concurrent process determinism** — `scripts/probe-concurrency.py` runs at least three independent
   processes with eight pipelined queries each against both forced CPU and the advertised accelerator. The
   records must bind the unpacked binary SHA-256, cache and forcing environment, expected backend, model, and UTC
   timestamp; require the expected backend rather than accepting silent CPU fallback; return bit-exact vectors
   across processes; and shut down every process cleanly.

Items defined by scripts retain their script-level pass rules. This document adds the checksum-bound
package, hardware, fallback, and throughput promotion rules.

## Archive proof commands

The scripts accept only an archive plus its exact expected checksum. They never execute a binary from
`target/` or the package staging directory.

```bash
scripts/hardware-smoke.sh \
  --archive /path/to/julie-semantic-sidecar-archive.tar.gz \
  --sha256 <64-lowercase-hex> \
  --backend metal \
  --lane apple-arm64-metal-portable \
  --cache-dir /path/to/proof-cache \
  --fixtures /path/to/miller/eval/sidecar-conformance \
  --evidence-dir /path/to/evidence
```

```powershell
scripts/hardware-smoke.ps1 `
  -Archive C:\path\to\julie-semantic-sidecar-archive.zip `
  -Sha256 <64-lowercase-hex> `
  -Backend vulkan `
  -Lane windows-x64-vulkan-portable `
  -CacheDir C:\path\to\proof-cache `
  -FixturesDir C:\path\to\miller\eval\sidecar-conformance `
  -EvidenceDir C:\path\to\evidence
```

The evidence directory contains the package manifest, checksum, device/runtime identities, selection
cache, protocol smoke output, CPU and accelerator conformance logs, and batch-1/indexing-batch results.
Review this evidence before changing support status; script success does not promote a backend.

Run the concurrency probe separately for the forced CPU and accelerator lanes:

```bash
JULIE_SIDECAR_FORCE_BACKEND=cpu \
JULIE_EMBEDDING_CACHE_DIR=/path/to/proof-cache \
python3 scripts/probe-concurrency.py \
  --binary /path/to/unpacked/julie-semantic-sidecar \
  --clients 3 --requests 8 --expect-backend cpu --json

env -u JULIE_SIDECAR_FORCE_BACKEND \
JULIE_EMBEDDING_CACHE_DIR=/path/to/proof-cache \
python3 scripts/probe-concurrency.py \
  --binary /path/to/unpacked/julie-semantic-sidecar \
  --clients 3 --requests 8 --expect-backend metal --json
```

## Artifact workflow boundary

The manual artifact workflow requires the protected `artifact-release-approval` environment plus an
exact `hardware_lane` and `expected_archive_sha256`. It always builds the four portable candidates
and optionally builds CUDA candidates. Outputs remain workflow artifacts containing archives,
checksums, manifests, and raw validation logs. The workflow does not publish, tag, promote a backend,
change the Miller pin, or create public assets.

## Portable and vendor-specific lanes

Portable archives are the default consumer artifacts and must prove both their named accelerator and
CPU fallback: Metal on macOS arm64 (`apple-arm64-metal-portable`), Metal on macOS x64
(`apple-x64-metal-portable`), Vulkan on Windows x64 (`windows-x64-vulkan-portable`), and Vulkan on
Linux x64 (`linux-x64-vulkan-portable`). Evidence for one platform or archive checksum never
promotes another.

The Apple x64 lane is built and artifact-validated on GitHub's `macos-15-intel` runner. That proves
the exact target can compile, package, unpack, and satisfy its deterministic manifest; it is not
physical Intel-Mac support evidence. Promotion requires the checksum-selected archive to pass the
real-device, golden-vector, Metal-selection, CPU-fallback, and performance gates on a physical Intel
Mac. It cannot inherit Apple arm64 evidence or the M2 Ultra throughput result; record an approved
Intel reference machine and lane-specific floor before marking the x64 archive supported.

CUDA, HIP/ROCm, and SYCL archives are vendor-specific lanes. They do not substitute for portable-lane
proof and become supported only after their own exact-checksum real-device, golden-vector, fallback,
and performance evidence exists. A vendor lane must also show a material benefit over the matching
portable archive. HIP/ROCm or SYCL health fields additionally require the Miller-owned wire-contract
amendment before those lanes can report support truthfully.

## Throughput floor — the check

One command, run on the target machine against the binary unpacked from the checksum being promoted:

```
scripts/bench-throughput.py \
  --binary /path/to/unpacked/julie-semantic-sidecar \
  --expect-backend metal
```

It probes `health` and **fails the bench** unless the sidecar reports `ready: true` and the named backend with
truthful acceleration — a `model_not_prepared` binary or silent CPU fallback can never pass.
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
| Historical Qwen CPU-only regression | ~6.6 | M2 Ultra |
| RC3 BGE Metal exact archive | 775.7 | M2 Ultra |

**40 remains the minimum useful-throughput floor, not the backend-identity test.** It sits well below the
healthy Qwen Metal range (77–89 units/s), the P0 llama-server reference (52.3), and the RC3 BGE Metal result,
so machine noise does not trip it. `--expect-backend` is the load-bearing guard against silent CPU fallback;
BGE CPU throughput can exceed the historical Qwen-derived floor.

## Why this gate exists (the rc.1 lesson)

**Harness numbers are not engine numbers.** A CPU-only RC shipped at roughly **12× under the
design throughput floor** because the throughput that had been validated was a benchmark
harness's, not the shipping engine's on the real artifact. The full record is in Miller's
`docs/findings/2026-07-20-first-real-shadow-converge-benchmark.md`.

The correction is this gate: the floor is measured on the **target machine**, against the
**checksum-identified packaged binary**, over the **real embedding path** (`health` + `embed_batch`
over stdio) — not a harness, not a synthetic microbenchmark, not a different backend. A binary that
does not select the expected accelerated backend or cannot clear 40 units/s on the reference machine does not
get promoted, regardless of what any other benchmark reported.
