# Accelerated Backends and Model Strategy

**Date:** 2026-07-22
**Status:** Approved design; implementation has not started
**Primary repo:** `/Users/murphy/source/julie-semantic-sidecar`
**Consumers:** Miller now; Julie only if it remains a product and adopts the frozen protocol later

## Decision

Keep `julie-semantic-sidecar` as the Rust implementation of the frozen
`julie.embedding.sidecar` v1 protocol for the next production milestone. Do not rewrite it in .NET
before hardware packaging and model quality are measured through the existing boundary.

The sidecar owns model acquisition, model identity, native runtime loading, backend selection,
fallback, and truthful health. Miller always launches it with an explicit model id and consumes only
the protocol. BGE-small is the default model. Qwen remains an optional comparison lane until its
removal is justified by a sealed model decision. CodeRankEmbed is a candidate evaluated through the
same Miller task corpus, not a replacement selected from incomparable Julie results.

Every supported package contains CPU inference and at least one platform-appropriate accelerated
backend. A usable GPU that fails to initialize or loses the first-start benchmark falls back to CPU
with a visible degradation reason. A model checksum mismatch, model identity mismatch, vector
dimension mismatch, or wire-contract violation remains a hard failure.

## Product Goal

Minimize an agent's **time to sufficient evidence** while preserving correctness. Retrieval quality is
the first gate. Calls, tokens, wall time, irrelevant output, indexing time, download size, and resident
memory are cost measures after correctness passes.

The sidecar succeeds only when accelerated semantic retrieval improves the complete Miller task, not
merely when a model loads or a GPU benchmark is fast.

## Current Evidence

- Miller selects `bge-small-en-v1.5-f32` explicitly and serves a 384-dimensional lane. The sidecar's
  standalone default and its repository guidance still name `qwen3-0.6b-f16`.
- On Miller's fused retrieval benchmark, BGE scored `0.5727` overall nDCG versus Qwen's `0.5794`,
  retaining 98.8% of Qwen's fused quality and passing the pre-registered non-inferiority rule.
- On the measured 49k-symbol workspace, BGE used about 27 times less sidecar RSS, built vectors about
  8 times faster, embedded warm queries about 5 times faster, and downloaded about 9 times less model
  data than Qwen.
- Julie's CodeRankEmbed scorecard reported 93.3% top-5 and 0.776 MRR over 30 cases. That result is not
  comparable to Miller's BGE result because the corpus, search implementation, fusion, and evaluation
  harness differ.
- The Rust sidecar has Metal work on `main`, backend-selection policy, and truthful health shapes, but
  `scripts/package.sh` still records accelerated release packaging as unfinished. Cross-platform
  accelerated artifacts are not yet proven.
- `llama.cpp` supplies Metal, CUDA, HIP, Vulkan, and SYCL backends. The implementation language does
  not remove the native build, packaging, driver, or real-hardware verification work.

### Evidence and current runtime documentation

- Miller fused model and cost evidence:
  `/Users/murphy/source/miller/docs/findings/2026-07-21-fused-arm-encoder-benchmark.md`
- Miller corrected BGE task baseline:
  `/Users/murphy/source/miller/docs/findings/2026-07-22-semantic-decision-baseline.md`
- Julie CodeRankEmbed scorecard:
  `/Users/murphy/source/julie/docs/eval/semantic-value/results/2026-05-23T18-10-58Z.md`
- Sidecar throughput/RSS evidence: `docs/findings/2026-07-20-model-throughput-rss-bench.md`
- `llama.cpp` backend matrix: <https://github.com/ggml-org/llama.cpp>
- ONNX Runtime execution-provider matrix: <https://onnxruntime.ai/docs/execution-providers/>
- ONNX Runtime DirectML status: <https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html>
- LLamaSharp backend packages: <https://github.com/SciSharp/LLamaSharp>

## Scope

### In scope

- Align the sidecar's default model and documentation with Miller's BGE default.
- Produce accelerated sidecar packages for Apple Silicon, Windows x64, and Linux x64.
- Preserve CPU fallback in every package.
- Make backend availability, selection, acceleration, and degradation truthful through `health`.
- Prove each advertised backend on real compatible hardware.
- Compare BGE and CodeRankEmbed through one Miller-owned agent-task evaluation.
- Run a bounded .NET/ONNX Runtime spike only when it can unlock a model or hardware lane the Rust
  implementation cannot deliver competitively.
- Amend the Miller-owned protocol before emitting any new wire fields needed for HIP or SYCL.

### Out of scope

- Eros integration or fleet semantics.
- Rewriting Miller's semantic retrieval pipeline.
- Replacing `julie.embedding.sidecar` v1 with an in-process API.
- A speculative multi-engine abstraction inside the Rust sidecar before a second production engine
  exists.
- Selecting a universal model from results gathered on different corpora.
- Publishing a release without explicit user approval.

## Architecture Quality

**Affected modules:** `manifest`, `engine`, `backend_select`, `health`, `prepare`, the CLI dispatcher,
packaging scripts, release workflows, conformance tests, and Miller's sidecar pin/restore integration.

**Caller-facing interface:** the frozen NDJSON protocol remains the primary interface. The launch
interface remains `serve --model <id>` and `prepare --model <id>`; consumers must pass the model
explicitly. Standalone omission resolves to BGE-small. Health remains the authoritative runtime
outcome.

**Depth/locality check:** model files, native libraries, accelerator choice, driver identity, benchmark
caching, and fallback stay inside the sidecar. Miller knows only the executable, the selected model
identity, protocol health, and vectors. Deleting the process boundary would spread native-runtime and
platform policy into Miller, so the boundary earns its keep.

**Test surface:** protocol conformance, packaged CLI behavior, real-backend health, golden-vector
conformance, and Miller's end-to-end semantic task harness. Tests do not assert private backend-loader
plumbing.

**Seams/adapters:** retain the existing protocol and engine trait. Native `llama.cpp` backends are
internal implementation choices. Do not add a general provider plugin API until a second engine passes
the same task and packaging gates.

**Rejected shortcuts:** a .NET rewrite without comparative evidence; build-only GPU claims; silent CPU
fallback; mapping HIP or SYCL to misleading existing capability keys; shipping Qwen because it remains
the sidecar default; treating CodeRankEmbed's Julie scorecard as a Miller result.

**Architecture risk:** high. The wire boundary is stable, but release correctness spans native backend
builds, driver/runtime availability, platform packaging, real hardware, model identity, and consumer
integration.

## Runtime and Distribution Shape

Each archive contains one executable, CPU inference, and one named accelerated backend. The process
benchmarks the packaged accelerator against CPU on first start and caches the winner by sidecar version,
model SHA-256, backend build identity, GPU identity, and driver identity. A key change invalidates the
choice and reruns the benchmark.

| Platform artifact | Accelerated backend | Purpose | Promotion evidence |
|---|---|---|---|
| macOS arm64 portable | Metal | Default Apple Silicon package | Real M-series load, query, batch, fallback, and golden-vector proof |
| Windows x64 portable | Vulkan | Broad NVIDIA, AMD, and Intel coverage; DirectML-equivalent role | Real Windows GPU proof from at least AMD or Intel plus NVIDIA |
| Windows x64 NVIDIA | CUDA | Optional vendor-optimized package | Real NVIDIA proof and improvement over the portable package |
| Linux x64 portable | Vulkan | Broad default accelerated package | Real Vulkan device proof and CPU fallback proof |
| Linux x64 NVIDIA | CUDA | Vendor-optimized package | Real NVIDIA proof and improvement over the portable package |
| Linux x64 AMD | HIP/ROCm | Vendor-optimized package | Real supported AMD GPU proof and improvement over the portable package |
| Windows/Linux x64 Intel | SYCL | Optional Intel Arc optimization | Real Arc proof and improvement over Vulkan |

On Apple Silicon, Metal is the native `llama.cpp` accelerator and satisfies the hardware-acceleration
requirement; MPS is PyTorch's backend name and is not required when the selected runtime uses Metal
directly. On Windows, Vulkan is the portable DirectML-equivalent lane. A later ONNX Runtime candidate
may use WinML/DirectML only if it wins the conditional runtime spike.

Miller bundles the portable artifact for its target. Vendor-specific packages are release assets with
the same executable name and protocol version; adopting one must not change Miller code or artifact
formats. The existing source override remains the development path until release assets and checksums
are published.

The current v1 capability map permits the required `cpu`, `cuda`, `directml`, and `mps` keys plus the
additive `metal` and `vulkan` keys. HIP and SYCL must not be represented through false aliases. Before
those packages report named capability keys, a separate Miller change must amend
`docs/contracts/semantic-sidecar-protocol-v1.md`, update its consumer tests, and publish the exact
additive shape. The sidecar then implements the amended contract.

## Backend Selection and Failure Behavior

1. Discover only backends actually present in the package and usable on the current machine.
2. Keep CPU unconditionally available.
3. On an uncached identity, run the existing batch-1 and indexing-batch micro-benchmarks for CPU and
   every usable packaged accelerator.
4. Cache the winner. Never report `accelerated: true` merely because a device or library exists.
5. Report requested backend, resolved backend, device, acceleration, capabilities, and degradation
   through `health`.
6. If an accelerator probe, load, or benchmark fails, remain ready on CPU and include a stable,
   actionable degradation reason.
7. A diagnostic forced backend that is unavailable follows the frozen v1 behavior: ready on CPU with
   a non-null degradation reason.
8. Keep fd-level stdout protection active around discovery, native loading, probing, benchmarking, and
   inference so protocol stdout remains pure NDJSON.
9. Refuse model/dimension/contract mismatches. Miller then degrades visibly to lexical retrieval under
   its existing semantic failure policy.

## Model Policy

### Shipping policy

- `bge-small-en-v1.5-f32` is the default for omitted standalone CLI selection and Miller packaging.
- Miller continues to pass `serve --model bge-small-en-v1.5-f32`; it never relies on the omission
  default.
- `qwen3-0.6b-f16` remains preparable and servable during the comparison window, but is not downloaded
  or loaded by default.
- A model change creates a new vector generation and preserves rollback through Miller's existing
  fingerprint and shadow-rebuild rules.

### CodeRankEmbed evaluation

Use Julie's current Python sidecar only as a protocol-compatible reference producer for the first
comparison. Do not modify the active Julie product session. Add CodeRankEmbed to Miller's evaluation
only after its vectors can be imported or served without changing production routing.

The common evaluation freezes the same repositories, commits, queries, relevance judgments, result
surface, fusion policy, and task scorer for every model. It records:

- task correctness and sufficient-evidence completion;
- calls, tokens, wall time, retries, and irrelevant output;
- recall and nDCG as diagnostics;
- cold start, warm query latency, batch throughput, indexing time, download size, and peak sidecar RSS;
- failure, fallback, and multi-session behavior.

CodeRankEmbed may replace BGE only if it passes the correctness floor and improves total agent-task
efficiency enough to justify any added runtime and packaging cost. Results from Julie's historical
30-query scorecard remain context, not promotion evidence.

## Conditional .NET Spike

Do not start with a rewrite. Run this spike only if CodeRankEmbed cannot be served through the existing
Rust/`llama.cpp` implementation or a required accelerator lane remains unshippable.

Build the smallest protocol-compatible candidate that supports `health`, `embed_query`, and
`embed_batch` using ONNX Runtime. Exercise CoreML on Apple Silicon, WinML/DirectML on Windows, CUDA on
NVIDIA, MIGraphX on AMD Linux, and OpenVINO on Intel where official packages and the chosen model permit.
LLamaSharp is not a separate model-runtime strategy; it wraps the same native `llama.cpp` family and is
useful only if a .NET host materially simplifies packaging.

The .NET candidate earns replacement consideration only if it:

- passes the same protocol and vector-conformance surface;
- packages every required platform without a weaker fallback story;
- matches or improves task correctness;
- improves total task cost or materially reduces installation/support burden;
- preserves stdout purity, deterministic model acquisition, and failure semantics.

Otherwise delete the spike and keep Rust. Do not retain a second implementation merely because it was
built.

## Implementation Sequence

### Phase 0 — Freeze the baseline

**Files:** `README.md`, `AGENTS.md`, `docs/rc-promotion-gate.md`, a new findings record, and Miller's
current semantic evaluation artifacts.

- Record the current sidecar commit, Miller commit, model fingerprints, package state, and known real
  hardware evidence.
- Re-run the existing Rust unit, clippy, formatting, and conformance gates before changing behavior.
- Record Miller's current BGE task baseline and the existing Qwen/BGE cost table.
- Do not refresh, modify, or depend on the active Julie checkout; consume only its committed scorecard
  and an isolated CodeRank reference process when the comparison phase begins.

### Phase 1 — Align model identity

**Files:** `src/lib.rs`, `src/manifest.rs`, `src/main.rs`, manifest/CLI tests, `README.md`, and `AGENTS.md`.

- Change the omitted standalone default from Qwen to BGE-small.
- Keep exact `--model` selection for both `serve` and `prepare`.
- Keep both frozen model pins until the model decision closes.
- Add tests proving omitted selection is BGE, explicit Qwen remains possible, and health reports the
  exact selected identity and dimensions.
- Verify Miller's explicit `serve --model` path remains the consumer behavior.

### Phase 2 — Make accelerated artifacts reproducible

**Files:** `Cargo.toml`, `build.rs` if required, `scripts/package.sh`, a PowerShell packaging mirror,
`.github/workflows/ci.yml`, `.github/workflows/release.yml`, and package-layout tests.

- Define backend-specific build profiles with `GGML_NATIVE=OFF` and reproducible target flags.
- Resolve native backend modules relative to the executable; never use a build-directory path.
- Package the CPU runtime and exactly the advertised accelerator module in each artifact.
- Emit a machine-readable package manifest containing target, backend, sidecar version, native build
  identity, files, and checksums.
- Verify archives contain no model weights, development paths, or undeclared native libraries.
- Keep release publication approval-gated.

### Phase 3 — Complete runtime selection and truthful health

**Files:** `src/backend_select.rs`, `src/engine.rs`, `src/health.rs`, backend-selection tests, health tests,
and protocol tests.

- Discover and load executable-relative Metal, Vulkan, CUDA, HIP, or SYCL modules for the matching
  package.
- Preserve the first-start benchmark and identity-keyed cache.
- Expand diagnostic forcing to the backend names the package can contain while preserving CPU forcing.
- Make every capability boolean derive from real package and runtime facts.
- Add failure tests for missing modules, unavailable devices, corrupt selection caches, driver identity
  changes, slow accelerators, native loader chatter, and accelerator initialization failures.
- Land any HIP/SYCL health-key work only after the Miller contract amendment is merged.

### Phase 4 — Prove each platform lane

**Files:** conformance scripts, hardware smoke scripts, CI/release workflows, and findings records.

- Run the full golden-vector corpus through CPU and the real accelerator. Every text must pass the
  contract's vector checks; there is no percentage allowance.
- Prove `health` before embedding, batch behavior, shutdown, timeout recovery, and CPU degradation from
  the packaged archive.
- Record batch-1 and indexing-batch measurements for CPU and accelerator under an idle-system gate.
- Require the selected accelerator to win the existing first-start benchmark. A slower backend may ship
  for diagnostics but must not resolve as accelerated by default.
- Do not mark a backend supported from compilation or software rendering. Record GPU model, driver,
  runtime, package checksum, and raw results.

### Phase 5 — Integrate portable packages with Miller

**Miller-owned files:** `scripts/semantic-pins.json`, restore scripts, package smoke tests, release
workflow, semantic scale tests, and release documentation.

- Publish nothing until explicitly approved.
- Once sidecar assets exist, pin exact versions, filenames, and SHA-256 values per Miller target.
- Bundle Metal for Apple Silicon and Vulkan for Windows/Linux portable releases, each with CPU fallback.
- Prove `miller semantic prepare`, vector convergence, search, health, and lexical fallback from the final
  packaged layout.
- Keep vendor-specific package selection outside Miller's public tool surface unless dogfood proves a
  consumer-facing selector is necessary.

### Phase 6 — Run the BGE versus CodeRankEmbed decision

**Primary owner:** Miller evaluation harness.
**Sidecar owner:** only protocol/model adapter work required to produce comparable vectors.

- Freeze a paired task corpus containing identifier, prose, documentation, configuration, ambiguous
  concept, negative, and multi-language cases.
- Run lexical control, BGE semantic/fused, and CodeRank semantic/fused arms with identical routing and
  fusion.
- Score correctness before cost. Reject any candidate that improves averages while regressing the
  correctness floor or identifier stability.
- Select the default from total task evidence. Record the decision and remove losing default-only
  complexity after the rollback window.

### Phase 7 — Decide whether a .NET runtime earns a place

- Run the conditional ONNX Runtime spike only under the triggers above.
- Compare the Rust and .NET candidates on identical model weights when possible, then on the winning
  model if runtimes require different formats.
- Choose one production implementation. Delete the losing spike and its packaging path.

## Verification Gates

### Every change

```bash
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

### Contract and package gates

- All protocol group A/B tests pass.
- Every applicable group C golden vector passes on CPU and each advertised accelerator.
- Packaged archives pass `prepare`, `serve`, `health`, `embed_query`, `embed_batch`, and `shutdown` smokes.
- Stdout contains protocol NDJSON only for the entire serve lifetime.
- Missing/unusable acceleration yields ready CPU health with a non-null degradation reason.
- Model, checksum, dimensions, backend identity, and package manifest agree end to end.

### Product gates

- Miller's sealed task correctness floor does not regress.
- Exact identifier ranked lists remain unchanged unless a separately approved contract says otherwise.
- Semantic task completion improves over lexical control.
- Calls, tokens, wall time, retries, and irrelevant output are reported for every arm.
- Production promotion still requires Miller's powered decision-canary gate; a local replay alone is not
  sufficient.

### Release gates

- Each advertised accelerator has real-device evidence from the exact archive checksum.
- CPU fallback is proven from every portable archive.
- Sidecar and Miller pins, manifests, checksums, help text, and docs agree.
- No release is published without explicit user approval.

## Acceptance Criteria

- [ ] BGE-small is the aligned sidecar and Miller default; Miller passes it explicitly.
- [ ] Qwen is optional and never downloaded or loaded by default.
- [ ] Apple Silicon packages use and truthfully report Metal, with proven CPU fallback.
- [ ] Windows portable packages use and truthfully report Vulkan, with proven CPU fallback.
- [ ] Linux portable packages use and truthfully report Vulkan, with proven CPU fallback.
- [ ] CUDA packages are proven on real NVIDIA hardware.
- [ ] HIP/ROCm packages are proven on real AMD Linux hardware.
- [ ] Intel Arc has a proven Vulkan lane; SYCL ships only if it beats that lane and its health contract is
      amended truthfully.
- [ ] Every advertised accelerator passes protocol, golden-vector, package, and real-device gates.
- [ ] BGE and CodeRankEmbed are compared on one frozen Miller task corpus.
- [ ] The winning model is selected by correctness-first total agent-task evidence.
- [ ] A .NET runtime is adopted only if its bounded spike beats the Rust implementation on the approved
      gates; otherwise the spike is deleted.
- [ ] The final Miller package passes semantic prepare, convergence, search, health, and lexical fallback
      smokes.
- [ ] Release publication remains explicitly approval-gated.

## New-Session Starting Point

1. Verify the sidecar repo path, branch, commit, dirty state, and all worktrees.
2. Read `AGENTS.md`, this design, the existing P2a plan, `docs/rc-promotion-gate.md`, and Miller's frozen
   sidecar protocol.
3. Confirm no other session is changing this repo or the Miller semantic contract.
4. Use `razorback:writing-plans` to turn Phases 0–5 into an implementation plan. Keep Phases 6–7 as
   separate evidence-gated plans.
5. Start with Phase 0 and Phase 1. Do not publish, change the wire contract locally, or begin the .NET
   spike.
