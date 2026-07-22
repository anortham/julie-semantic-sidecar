# Accelerated backend baseline — 2026-07-22

This record freezes the evidence before the accelerated packaging and standalone-default changes.
It does not advertise a backend, authorize a release, or replace the frozen wire contract.

## Frozen coordinates

| Input | Frozen value | Evidence |
|---|---|---|
| Sidecar source | `1c9ea3a3110ca56c47f298e53f400548ce958e90` | This worktree's pre-change `HEAD` |
| Miller evidence source | `d3fa7ae5027bd383943201bf311846d38d861b91` | Read-only Miller checkout |
| Wire contract | `julie.embedding.sidecar` v1, status `frozen`; Miller commit contains `8edfa14` | [Protocol contract](/Users/murphy/source/miller/docs/contracts/semantic-sidecar-protocol-v1.md) |
| Approved product direction | BGE is the standalone default; Miller continues explicit model selection | [Accelerated-backends design](../plans/2026-07-22-accelerated-backends-and-model-strategy-design.md) |

The Julie checkout was not read, refreshed, or modified for this baseline. Its historical CodeRankEmbed
score remains context named by the design, not evidence for the current sidecar or Miller task.

## Model identity before the default change

The pre-change sidecar manifest contains exactly two GGUF pins. The SHA-256 values below identify model
files; Miller's vector fingerprint is a separate end-to-end encoder identity.

| Manifest id | Pre-change sidecar tier | GGUF SHA-256 | Native dims | Served dims |
|---|---|---|---:|---:|
| `qwen3-0.6b-f16` | default | `421a27e58d165478cc7acb984a688c2aa41404968b0203e7cd743ece44c54340` | 1024 | 512 |
| `bge-small-en-v1.5-f32` | fallback | `bf40c42ad7d89382e9ba7376d5c4b73f6b556cb541fab37aaa1da9c320149b65` | 384 | 384 |

Source: [sidecar manifest](../../src/manifest.rs) and the contract's
[model-knob table](/Users/murphy/source/miller/docs/contracts/semantic-sidecar-protocol-v1.md#model-knob-table).
The approved direction changes omission behavior to BGE without removing exact Qwen selection.

## Miller quality and cost evidence

- The comparable fused-arm result is retrieval evidence, not a task-completion score: BGE scored
  `0.5727` nDCG@10 and `0.6163` recall@10; Qwen scored `0.5794` and `0.6094`. BGE retained 98.8% of
  Qwen's fused nDCG and passed the registered encoder-selection rule. Source:
  [fused-arm benchmark](/Users/murphy/source/miller/docs/findings/2026-07-21-fused-arm-encoder-benchmark.md#results--task-4-arm-generation--scoring).
- The corrected production replay uses BGE fingerprint
  `sha256:3e8b7e8a0890dc84f702db1d13c47e312501905ee9d1aafb772bdc803616d7f4` and reports `0.6434`
  nDCG@10, `0.6892` recall@10, all 14 intent clusters, and no zero-result rows. It explicitly says
  this visible replay is not promotion evidence and that no sealed paired task result exists. Source:
  [corrected semantic baseline](/Users/murphy/source/miller/docs/findings/2026-07-22-semantic-decision-baseline.md).
- On the frozen 49k-symbol cost run, Qwen versus BGE was 1.198 GB versus 133.6 MB download,
  312.7 s versus 40.4 s median end-to-end build, 12.34 GiB versus 470 MiB peak sidecar RSS, and
  4048 ms versus 802 ms warm query embedding. Source:
  [real-artifact cost table](/Users/murphy/source/miller/docs/findings/2026-07-21-fused-arm-encoder-benchmark.md#cost-table--medians-of-clean-runs-no-bench-workloads--cpu-idle--60).

## Package and backend state

| Current archive lane | Build in `scripts/package.sh` | What exists | What is missing |
|---|---|---|---|
| macOS arm64 | `metal` feature | Artifact-only workflow lane; local M2 Ultra Metal engine evidence | Exact-archive manifest, checksum-bound Metal load, golden-vector, and CPU-fallback proof |
| macOS x64 | CPU only | Artifact-only workflow lane and offline not-ready smoke | An accelerated portable lane; the approved design does not carry this lane forward |
| Linux x64 | CPU only | Artifact-only workflow lane; CI installs lavapipe for a non-blocking diagnostic | Packaged Vulkan module and real-GPU proof |
| Windows x64 | CPU only | Artifact-only workflow lane and offline not-ready smoke | Packaged Vulkan module and real-GPU proof |

Sources: [packaging script](../../scripts/package.sh), [artifact workflow](../../.github/workflows/release.yml),
and [CI workflow](../../.github/workflows/ci.yml). The workflow is manual, read-only, and uploads artifacts;
it has no publish step.

The current runtime always has CPU and recognizes only the compile-time `metal` feature as accelerated.
Its cache plumbing is keyed by sidecar version, model SHA-256, GPU identity, and driver identity, but the
production callback returns the compile-time backend rather than timing CPU against Metal. The loader calls
`LlamaBackend::init()` directly; executable-relative dynamic backend loading is not implemented. Sources:
[backend selection](../../src/backend_select.rs) and [engine load](../../src/engine.rs).

## Known hardware evidence

The current model-specific accelerated record is a local M2 Ultra Metal engine run: Qwen sustained
82.9 units/s with 1.27 GiB RSS, and BGE sustained 743.7 units/s with 196 MiB RSS. Both reported
`ready: true`, `accelerated: true`, and `resolved_backend: metal`. Source:
[sidecar throughput and RSS record](2026-07-20-model-throughput-rss-bench.md).

That record is not tied to an archive checksum and does not prove packaged CPU fallback. Metal packaging is
therefore not yet proven. Windows and Linux acceleration have no packaged real-GPU proof. Software Vulkan,
successful compilation, and archive creation are diagnostics only.

## Pre-change verification

At sidecar commit `1c9ea3a3110ca56c47f298e53f400548ce958e90`, the lead's pre-parallel baseline run completed
`cargo test`, `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check` with exit code 0.
The full model-backed conformance gate is not claimed by this documentation worker; it remains part of the
integrated branch and exact-archive gates.

## Promotion evidence still required

- Every supported archive needs a machine-readable package manifest and a recorded archive SHA-256.
- That exact checksum needs full protocol and golden-vector proof on a real compatible accelerator.
- The same exact checksum needs explicit CPU-forcing or accelerator-failure fallback proof.
- Portable Metal/Vulkan evidence and optional vendor CUDA/HIP/SYCL evidence stay separate.
- Miller asset pins and release publication remain approval-gated; neither is authorized by this record.

The binding requirements are maintained in the [RC promotion gate](../rc-promotion-gate.md).
