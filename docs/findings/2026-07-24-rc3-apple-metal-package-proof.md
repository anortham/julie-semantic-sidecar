# RC3 Apple Silicon Metal package proof — 2026-07-24

## Verdict

The exact released `v0.1.0-rc.3` Apple arm64 archive passes the complete CPU and
Metal protocol/golden conformance suite on physical Apple Silicon. Automatic
Metal selection, cache reuse, forced CPU, unavailable-backend fallback, package
inventory, stdout purity, the 40 texts/s promotion floor, and RSS checks passed.

This closes only the Apple arm64 RC3 lane. It does not prove the released
Windows/Linux archives, macOS x64 packaging, CUDA, ROCm, Intel Arc, multi-client
behavior on another platform, or the final BGE-small versus CodeRankEmbed
decision.

## Bound identity

- Release: `v0.1.0-rc.3`
- Release commit: `24ce6257bee7f41865b10daf1457ed9b4fd71a8a`
- Archive: `julie-semantic-sidecar-0.1.0-rc.3-aarch64-apple-darwin-metal-portable.tar.gz`
- Archive SHA-256: `92a873438635f843d46e166c105bb122a30e86199ba1428cef13bb00f9ebf6e0`
- Executable SHA-256: `af9f686fe8118b013d86b0d6d0ebb3ebfffdc48b2b220c9372f0647f2c283d68`
- Sidecar version: `0.1.0-rc.3`
- Target/profile: `aarch64-apple-darwin`, portable Metal
- Host: Apple M2 Ultra, 60 GPU cores, Metal 4, macOS 26.5.2
- Default model: `bge-small-en-v1.5-f32`

The downloaded archive matched both GitHub's published asset digest and its
release `.sha256` sidecar. The captured
`raw-logs/github-release-assets.json` binds that API digest to the asset name,
size, and release URL. The unpacked flat inventory and every manifest size and
digest matched `package-manifest.json`; no model weight was packaged.

## Protocol and fallback

- CPU conformance: 9 passed, 0 failed.
- Metal conformance: 9 passed, 0 failed.
- Both BGE-small and Qwen Group C golden rows passed, including the 250-position
  batch probe.
- Automatic selection resolved to `metal` on `Apple M2 Ultra`.
- A second launch reused an unchanged selection-cache identity.
- Forced CPU stayed ready with `accelerated=false`.
- Forced unavailable Vulkan fell back to ready CPU with
  `degraded_reason="requested backend is unavailable"`.
- An empty cache returned `ready=false` and
  `degraded_reason="model_not_prepared"`.

## BGE-small throughput and RSS

| Backend | Batch | Texts/s | RSS MiB |
|---|---:|---:|---:|
| CPU | 1 | 33.15 | 180.20 |
| CPU | 16 | 276.18 | 180.66 |
| Metal | 1 | 161.95 | 190.56 |
| Metal | 16 | 739.92 | 193.72 |

Metal was 4.89 times faster at batch 1 and 2.68 times faster at batch 16 while
adding about 10–13 MiB RSS, so accepting Metal as the cached winner is supported
by the measured package.

The promotion-floor run used the default 64-text batch, four measured rounds,
the real 40 texts/s floor, and an explicit `--expect-backend metal` assertion.
It sustained 775.69 texts/s on Metal and passed.
The earlier batch-1 and batch-16 records are measurements rather than floor
gates.

## Concurrency and multi-process determinism

`scripts/probe-concurrency.py` started three independent released sidecar
processes simultaneously. Each process received one health request followed by
eight pipelined query requests without request/response pauses.

| Backend | Processes | Queries | Total wall ms | Per-process elapsed ms |
|---|---:|---:|---:|---|
| CPU | 3 | 24 | 9,480.66 | 573.93–9,436.40 |
| Metal | 3 | 24 | 642.61 | 576.57–577.02 |

Both lanes returned bit-exact vectors across all three processes, identical
health objects, ordered request IDs, correct dimensions, and clean zero exits.
The hardened harness also required `resolved_backend=cpu` for the forced CPU
lane and `resolved_backend=metal` with `accelerated=true` for the Metal lane.
Each JSON record carries the unpacked binary path and SHA-256, cache path,
forcing environment, expected backend, model, timeout, and UTC timestamp. This
proves concurrent independent-process behavior and per-process pipelining for
the RC3 Apple archive. The CPU run also records a 9.4-second process outlier
under simultaneous model load; it passed the 60-second response timeout but is
retained as resource-contention evidence rather than hidden. Miller's in-process
broker serialization remains covered by Miller's own tests.

The first probe attempt exposed a harness-only stderr deadlock. Fresh Claude
review then found that the initial harness did not bind its binary or
environment, could accept silent CPU fallback, had unbounded reads and unsafe
cleanup, and allowed an arbitrarily large pipelined request count. The versioned
probe now records its own SHA-256 plus the binary identity, enforces the expected
backend, times out reads, kills every child on any failure or partial spawn,
bounds the pipeline at 32 requests, and has five executable regression tests.

## Reproduction

Raw evidence is outside tracked source at:

`/Users/murphy/source/julie-semantic-sidecar-evidence/2026-07-24-rc3-metal`

The released-archive hardware smoke is reproduced from release commit
`24ce625`:

```bash
scripts/hardware-smoke.sh \
  --archive /Users/murphy/source/julie-semantic-sidecar-evidence/2026-07-24-rc3-metal/archive/julie-semantic-sidecar-0.1.0-rc.3-aarch64-apple-darwin-metal-portable.tar.gz \
  --sha256 92a873438635f843d46e166c105bb122a30e86199ba1428cef13bb00f9ebf6e0 \
  --backend metal \
  --lane apple-arm64-metal-portable-rc3 \
  --cache-dir /Users/murphy/source/julie-semantic-sidecar-evidence/2026-07-24-rc3-metal/cache \
  --fixtures /Users/murphy/source/miller/eval/sidecar-conformance \
  --evidence-dir /Users/murphy/source/julie-semantic-sidecar-evidence/2026-07-24-rc3-metal
```

The evidence tree contains the archive and manifest, device/runtime identity,
selection cache, prepare logs, protocol transcripts, complete CPU/Metal
conformance logs, four batch-1/batch-16 benchmark records, the batch-64
promotion-floor record, GitHub release asset metadata, review binding, and the
CPU/Metal concurrency probe records.

The concurrency records are reproduced separately from the hardware-smoke
command using the versioned proof-branch harness whose SHA-256 is
`83e1678a7d1f14933d72f367ba193503335912d7d5d56abbaf146f079b92a947`:

```bash
JULIE_EMBEDDING_CACHE_DIR=/Users/murphy/source/julie-semantic-sidecar-evidence/2026-07-24-rc3-metal/cache \
JULIE_SIDECAR_FORCE_BACKEND=cpu \
python3 -B scripts/probe-concurrency.py \
  --binary /Users/murphy/source/julie-semantic-sidecar-evidence/2026-07-24-rc3-metal/unpacked/julie-semantic-sidecar \
  --clients 3 --requests 8 --expect-backend cpu --json

env -u JULIE_SIDECAR_FORCE_BACKEND \
JULIE_EMBEDDING_CACHE_DIR=/Users/murphy/source/julie-semantic-sidecar-evidence/2026-07-24-rc3-metal/cache \
python3 -B scripts/probe-concurrency.py \
  --binary /Users/murphy/source/julie-semantic-sidecar-evidence/2026-07-24-rc3-metal/unpacked/julie-semantic-sidecar \
  --clients 3 --requests 8 --expect-backend metal --json
```

The promotion-floor record is reproduced with:

```bash
env -u JULIE_SIDECAR_FORCE_BACKEND \
JULIE_EMBEDDING_CACHE_DIR=/Users/murphy/source/julie-semantic-sidecar-evidence/2026-07-24-rc3-metal/cache \
python3 -B scripts/bench-throughput.py \
  --binary /Users/murphy/source/julie-semantic-sidecar-evidence/2026-07-24-rc3-metal/unpacked/julie-semantic-sidecar \
  --batch 64 --floor 40 --expect-backend metal --json
```

`raw-logs/review-binding.txt` records release/harness commit
`24ce6257bee7f41865b10daf1457ed9b4fd71a8a`, confirms the conformance and
hardware-smoke owners were unchanged, binds the Miller fixture tree to SHA-256
`7a4b4c2f1f28fefcbe450aa95c2786cc3515f625cf644bddab7feb427a484156`,
and closes the original `real-device-pending-review` marker for Apple arm64
only. Both concurrency JSON records independently carry the same harness
SHA-256.
