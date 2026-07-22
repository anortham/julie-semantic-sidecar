# Accelerated Backends and Model Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `razorback:subagent-driven-development` when delegation is available. Fall back to `razorback:executing-plans` for tightly sequential or no-delegation work. Every behavior change uses `razorback:test-driven-development`.

**Goal:** Align the standalone sidecar with Miller's BGE-small model choice, ship reproducible Metal and Vulkan portable artifacts with CPU fallback, make runtime backend selection evidence-based and truthful, prove the packaged lanes, and integrate approved release assets into Miller without changing the frozen v1 wire contract.

**Architecture:** Keep the existing Rust process boundary, protocol module, and engine trait. The sidecar remains the sole owner of model pins, native libraries, device discovery, backend benchmarking, selection caching, and fallback. Dynamic llama.cpp modules and their core shared libraries stay flat beside the executable, preserving the existing executable-relative loader rule and satisfying Windows loader semantics; Linux dynamic builds embed an `$ORIGIN` runpath. A build-only Rust helper creates and verifies the package manifest so Bash and PowerShell packaging share one schema and checksum implementation. Miller continues to pass an explicit model id and learns nothing about native backend mechanics.

**Tech stack:** Rust 2021, exact-pinned `llama-cpp-2 = 0.1.151` and `llama-cpp-sys-2 = 0.1.151`, llama.cpp dynamic backends, serde/serde_json, sha2, Bash, PowerShell, GitHub Actions, and Miller's existing .NET integration tests.

**Source design:** `docs/plans/2026-07-22-accelerated-backends-and-model-strategy-design.md`.

## Grounded Starting State

- Sidecar worktree: `/Users/murphy/source/julie-semantic-sidecar/.worktrees/accelerated-backends-model-strategy`, branch `codex/accelerated-backends-model-strategy`, base `1c9ea3a3110ca56c47f298e53f400548ce958e90`.
- Miller contract checkout: `/Users/murphy/source/miller`, `main` at `d3fa7ae5027bd383943201bf311846d38d861b91`; the frozen protocol is later than required commit `8edfa14`.
- Baseline gates at the sidecar base are green: `cargo test`, `cargo clippy --all-targets -- -D warnings`, and `cargo fmt --check`.
- `DEFAULT_MODEL_ID` and the manifest default tier still select `qwen3-0.6b-f16`; Miller's current `DefaultEncoder` is BGE-small and `ProcessSemanticSidecarLauncher.ForServe` passes `serve --model <id>` explicitly.
- Metal is a compile-time feature. `backend_select::select` currently returns a fixed build choice rather than timing real CPU and accelerator inference.
- The selection cache key includes sidecar version, model SHA-256, GPU identity, and driver identity, but not the native backend build identity required by the design.
- `scripts/package.sh` enables Metal only for Apple Silicon; Windows and Linux are CPU-only and no package manifest or native backend modules are staged.

## Global Constraints

- Read `/Users/murphy/source/miller/docs/contracts/semantic-sidecar-protocol-v1.md` in full before engine, health, or protocol work. Implement it exactly; never amend or work around it in this repo.
- Keep the launch surface exactly `serve [--model <id>]`, `prepare [--model <id>]`, and `--version`. Omission changes to BGE-small; explicit Qwen remains supported.
- Keep environment knobs exactly `JULIE_EMBEDDING_CACHE_DIR` and `JULIE_SIDECAR_FORCE_BACKEND`. Do not add configuration variables for loaders, devices, or packages.
- Preserve fd-level stdout protection around backend discovery, module loading, device enumeration, model loading, probing, benchmarking, and inference.
- CPU is always available. Missing, unusable, failing, or slower acceleration resolves to ready CPU health with a stable non-null degradation reason.
- Model checksum, selected model identity, dimensions, protocol shape, and golden-vector mismatches remain hard failures.
- Health capability booleans come from the loaded package and usable runtime facts. Do not alias HIP or SYCL to `cuda`, `vulkan`, `directml`, or `mps`.
- HIP and SYCL named health support cannot land until a Miller-owned protocol amendment defines the additive keys and consumer tests. Finding that amendment necessary is an expected approval boundary, not permission for a local wire change.
- Do not modify the user's dirty Miller checkout. Phase 5 work uses a dedicated Miller worktree created from the verified commit after sidecar assets exist.
- Do not run a paid GitHub Actions matrix, publish assets, update public pins, push, release, or deploy without explicit user approval.
- Do not start Phase 6 CodeRankEmbed evaluation or Phase 7 .NET work in this plan.

## Verified Native API Surface

- Enable `llama-cpp-2/dynamic-backends` and the matching backend feature for Vulkan/CUDA/ROCm builds; keep both llama crates exact-pinned and commit `Cargo.lock`. In the pinned sys crate, `dynamic-backends` transitively enables `dynamic-link`, which sets `BUILD_SHARED_LIBS=ON`; package tests must assert that feature relationship and the produced shared artifacts instead of relying on assumption.
- Call `llama_cpp_2::llama_backend::load_backends_from_path` with the executable's parent directory before `LlamaBackend::init`. Never call the no-argument loader because it uses the build output path. Check that the path is UTF-8 and NUL-free before calling the crate API so an unsupported install path degrades to CPU instead of panicking.
- Enumerate usable devices with `llama_cpp_2::list_llama_ggml_backend_devices`; record backend, device name, memory, and device type from that runtime result.
- Select the candidate device through `LlamaModelParams::with_devices` and set `n_gpu_layers = 0` for CPU so reported placement and actual placement cannot diverge.
- Release packaging must reject `RUSTFLAGS` or target-specific rustflags containing `-Ctarget-cpu=native`; the pinned sys build sets `GGML_NATIVE=OFF` automatically when that flag is absent. Record the effective target flags in the native build identity. Dynamic backend builds must use the transitive shared-library configuration because llama.cpp's `GGML_BACKEND_DL` requires it.
- The pinned crate exposes Metal, Vulkan, CUDA, ROCm, and dynamic-backend features, but no verified SYCL feature. SYCL stays outside this implementation until both the crate/build path and wire contract are proved.

## Architecture Quality

- **Affected modules:** `manifest`, CLI constants/tests, `backend_select`, `engine`, `health`, a new package-manifest module/helper, packaging scripts, CI/release workflows, conformance/hardware scripts, findings, and later Miller's pins/restore/package smoke surfaces.
- **Caller-facing interface:** the NDJSON protocol and explicit `serve --model` consumer path remain unchanged. Only the standalone omitted model changes.
- **New seam:** `backend_select` owns declared package backends, runtime facts, cache policy, and verdicts; `engine` owns loading a candidate and measuring real embedding work. Runtime enumeration is intersected with the sidecar's declared package feature so Apple Silicon's upstream force-compiled Metal dependency is not advertised by a sidecar build that did not declare Metal. The engine returns measurements to policy rather than teaching policy how to embed.
- **Package seam:** `src/package_manifest.rs` owns manifest schema and validation; `src/bin/julie-package-manifest.rs` is build tooling only and is never placed in release archives.
- **Depth/locality:** native paths, devices, timings, and degradation reasons stay in the sidecar. Miller receives only existing health fields and vectors.
- **Rejected shortcuts:** compile-only GPU claims, fixed build-choice selection, silent CPU fallback, build-tree loader paths, shell-specific manifest schemas, bundled model weights, and fake HIP/SYCL capability aliases.
- **Risk:** high. Native artifact correctness and real-device evidence are release gates, not unit-test claims.

## Verification Strategy

- Every behavior task starts with a focused failing test, proves the failure, implements the smallest complete behavior, then reruns the focused test.
- After each completed task run:

  ```bash
  cargo test
  cargo clippy --all-targets -- -D warnings
  cargo fmt --check
  ```

- Backend feature gates additionally compile and test the supported local feature set. On Apple Silicon, the upstream crate always compiles its sys-level Metal dependency; the plain gate therefore proves sidecar policy excludes undeclared Metal, while the feature gate proves the Metal package path:

  ```bash
  cargo test --features metal
  cargo clippy --all-targets --features metal -- -D warnings
  ```

- Package gates run against the staged archive, not `target/`: manifest verification, file allowlist, no model weights, no development paths, CLI smoke, protocol health, embedding, batch, shutdown, stdout purity, and CPU fallback.
- Contract gates run Miller's Group A/B tests and every applicable Group C vector for BGE and Qwen. Advertised accelerators must reproduce the same golden-vector surface as CPU.
- Real-device evidence records target triple, exact archive SHA-256, package manifest, GPU, driver, backend runtime, batch-1 and indexing-batch timings, health, fallback result, and raw logs.
- A compiled backend, a software renderer, or an unverified CI artifact is not support evidence.
- Worker ceiling is the focused tests plus the full Rust gate. The lead owns feature builds, archive verification, conformance, worktree reconciliation, and final branch verification.
- Verification ledger: `.razorback/sdd/progress.md`.

## Parallel Execution Contract

| Task | Batch | File ownership | Serialization | Reason |
|---|---|---|---|---|
| 1. Freeze baseline | A | Create `docs/findings/2026-07-22-accelerated-backend-baseline.md`; modify `docs/rc-promotion-gate.md` | No | Documentation-only evidence snapshot. |
| 2. Align BGE default | A | Modify `src/lib.rs`, `src/manifest.rs`, `src/main.rs`, `tests/manifest_tests.rs`, `tests/cli_tests.rs`, `tests/prepare_tests.rs`, `tests/serve_tests.rs`, `.github/workflows/ci.yml`, `README.md`, `AGENTS.md` | No | Separate from Task 1 files except docs; behavior is local to model selection. |
| 3. Implement real backend selection | None | Create `build.rs`; modify `Cargo.toml`, `Cargo.lock`, `src/backend_select.rs`, `src/engine.rs`, `src/health.rs`, `tests/engine_tests.rs`, `tests/manifest_tests.rs`, `tests/serve_tests.rs`, `tests/protocol_tests.rs` | Yes | Core runtime policy and model loading must move together. Starts after Task 2 to avoid overlapping manifest tests. |
| 4. Build reproducible packages | None | Create `src/package_manifest.rs`, `src/bin/julie-package-manifest.rs`, `scripts/package.ps1`, `tests/package_manifest_tests.rs`; modify `src/lib.rs`, `Cargo.toml`, `scripts/package.sh`, `README.md` | Yes | Consumes Task 3 feature names, build identity, and archive layout. |
| 5. Encode CI and archive gates | None | Create `scripts/hardware-smoke.sh`, `scripts/hardware-smoke.ps1`; modify `scripts/conformance.sh`, `tests/conformance.rs`, `.github/workflows/ci.yml`, `.github/workflows/release.yml`, `docs/rc-promotion-gate.md` | Yes | Consumes the final package scripts and manifest verifier. |
| 6. Prove the local Apple lane | None | Create `docs/findings/2026-07-22-metal-package-proof.md`; modify only defects found in Tasks 3–5 owned files | Yes | Requires the exact staged archive and local Apple Silicon hardware. |
| 7. Prove Windows/Linux lanes | B after approval | Create one findings record per exact archive; modify only defects found in Tasks 3–5 owned files | Per lane | Hardware legs can run independently, but paid workflow execution requires approval. |
| 8. Integrate Miller portable assets | None after release approval | Miller worktree only: `scripts/semantic-pins.json`, restore scripts, package smoke/scale tests, release workflow/docs | Yes | Exact filenames and checksums do not exist before approved assets are produced. |

## Task 1: Freeze the baseline

**Files:** per ownership table.

**What to build:** Record the exact sidecar/Miller commits, frozen contract edition, model ids/SHA-256/dimensions, current package matrix, current backend selection behavior, known hardware proof, baseline Rust gate results, Miller's corrected BGE task score, and the Qwen/BGE cost table. Update the promotion gate so support requires evidence from the exact archive checksum and so portable versus vendor-specific lanes are distinct.

**Steps:**

1. Create the findings record from committed Miller and sidecar evidence; link the source artifacts instead of copying result bodies.
2. Add explicit package-manifest, real-device, CPU-fallback, and exact-checksum requirements to `docs/rc-promotion-gate.md`.
3. Run `git diff --check` and the three baseline Rust gates.

**Acceptance:**

- [x] The record makes existing, missing, and approval-gated evidence unambiguous.
- [x] It states that Metal packaging and Windows/Linux acceleration are not yet proven.
- [x] No Julie checkout is modified or refreshed.
- [x] Rust gates remain green.

## Task 2: Align the standalone default with BGE-small

**Files:** per ownership table.

**Interfaces:** `DEFAULT_MODEL_ID`, `manifest::default_model`, no-argument `serve`, and omitted `prepare` resolve to `bge-small-en-v1.5-f32`. `serve --model qwen3-0.6b-f16` and `prepare --model qwen3-0.6b-f16` remain exact and supported. Miller's explicit launcher behavior remains unchanged.

**Steps:**

1. Change tests first: omitted CLI selection is BGE, exactly one BGE pin carries `Tier::Default`, Qwen carries `Tier::Fallback`, explicit Qwen still parses/resolves, and health for each selected pin reports its exact id and dimensions.
2. Run the focused manifest/CLI/prepare/serve tests and preserve the expected failures.
3. Change `DEFAULT_MODEL_ID` and the two tier assignments without changing either frozen pin's URL, SHA-256, size, query prefix, pooling, token budget, or dimensions.
4. Change CI model preparation to two explicit commands, one for `qwen3-0.6b-f16` and one for `bge-small-en-v1.5-f32`, so a clean cache still prepares both conformance models after the default flips.
5. Update README and AGENTS default-model text. State that consumers should pass an explicit model id.
6. Verify the focused sidecar tests. Verify Miller's explicit launch behavior read-only from `ForServe_LauncherPassesTheExplicitServeVerbAndSelectedModel` and `ProcessSemanticSidecarLauncher.ForServe`; do not run a build in the user's dirty Miller checkout.
7. Run the full Rust gate.

**Acceptance:**

- [x] Omitted standalone selection is BGE-small everywhere.
- [x] Qwen remains explicitly preparable and servable but is never selected by omission.
- [x] Health reports BGE 384 dimensions and Qwen 512 dimensions for the selected pin.
- [x] Miller still launches `serve --model bge-small-en-v1.5-f32` explicitly.

## Task 3: Implement discovery, real benchmarking, cached selection, and truthful health

**Files:** per ownership table.

**Interfaces:**

- Add crate features for `metal`, `vulkan`, `cuda`, `rocm`, and `dynamic-backends`. `dynamic-backends` maps to the pinned top-level crate feature, whose sys feature transitively enables `dynamic-link`; prove that feature graph with `cargo tree -e features` in the implementation report. Do not add `sycl` until its native and Rust build surfaces are proved.
- Replace compile-time `build_selection` with runtime candidates derived from the intersection of sidecar-declared package backends, loaded modules, and `list_llama_ggml_backend_devices`. This filter is load-bearing on Apple arm64 because the upstream top-level crate force-enables its sys-level Metal dependency even when the sidecar's `metal` feature is absent.
- Extend the selection-cache key to `sidecar version + model SHA-256 + native build identity + packaged backend identity + GPU identity + driver identity`.
- Keep `Selection` in the existing health shape. Stable backend names are `cpu`, `metal`, `vulkan`, `cuda`, and, only after a contract amendment, `hip`/`sycl`.
- Engine benchmarking loads each usable candidate with explicit device placement, embeds the same fixed batch-1 and 16-text indexing-batch probes used by the conformance generator, and returns comparable elapsed measurements. The policy selects acceleration only when it beats CPU; ties and failures select CPU with a reason.

**Steps:**

1. Add failing unit tests for executable-relative module loading, non-UTF-8 loader paths on Unix, missing module fallback, unavailable device fallback, declared-versus-enumerated backend filtering, supported forced values, unknown forced values, corrupt cache recovery, every cache-key component, alternating model identities, driver changes, failed probes, slower acceleration, and accurate capability flags.
2. Add engine-facing tests around candidate placement and timing through injected loader/benchmark seams; do not require a real GPU for policy tests.
3. Enable the verified llama features and compile-time build identity while retaining exact dependency pins and `publish = false`. Add a Linux `$ORIGIN` runpath for dynamic-backend builds so flat sibling core libraries resolve before Rust code starts.
4. Validate the executable parent path, then load sibling dynamic modules before `LlamaBackend::init` and enumerate actual devices. Metal uses its compiled backend but becomes a candidate only when the sidecar's declared package feature includes Metal and a usable device exists.
5. Refactor `LlamaEngine::load` so the stdout guard covers module loading, initialization, enumeration, all candidate loads, probe embeddings, timings, and final winner load.
6. Benchmark real fixed texts at batch 1 and batch 16. Store each verdict atomically in a per-identity cache entry so alternating BGE and Qwen sessions retain independent benchmark results.
7. Derive health capabilities from successfully loaded/usable backends. `accelerated` is true only for the selected non-CPU winner.
8. Preserve CPU forcing as a complete discovery/benchmark short circuit. Other forced backend names probe that backend; unavailable or failing values remain ready on CPU with a non-null reason.
9. Add process tests that inject native chatter during discovery/benchmarking and prove stdout remains NDJSON-only.
10. Run the CPU, Metal, and full Rust gates available on the local machine.

**Acceptance:**

- [x] No production path returns a fixed accelerator choice without timing it against CPU.
- [x] The cached choice invalidates for native build, package backend, GPU, driver, sidecar version, or model SHA changes.
- [x] Alternating BGE and Qwen reuses each model's matching cached verdict instead of overwriting one global slot.
- [x] Missing/failing/slower acceleration is ready on CPU and explains why.
- [x] Reported device, requested/resolved backend, acceleration, degradation, and capabilities match the runtime outcome.
- [x] No HIP/SYCL key or false alias is emitted under the current v1 contract.
- [x] Entire-session stdout purity still passes.

## Task 4: Produce reproducible, self-describing archives

**Files:** per ownership table.

**Archive layout:**

```text
julie-semantic-sidecar[.exe]
required core shared runtime libraries  # dynamic packages only
ggml-cpu.*                               # dynamic packages only
ggml-vulkan.* | ggml-cuda.* | ggml-hip.*
package-manifest.json
LICENSE
README.md
```

Metal remains compiled into the Apple Silicon executable, so its manifest declares a built-in `metal` backend and no fake plugin file.

**Package manifest:** schema version, sidecar version, Rust target triple, portable/vendor tier, advertised backend, native llama build identity, model policy ids and default id, and a sorted list of archive-relative files with SHA-256, size, and role. Paths must be relative, normalized, and free of build/worktree prefixes.

**Steps:**

1. Add failing tests for deterministic manifest ordering, checksums, required executable/runtime/backend roles, backend/file agreement, undeclared files, absolute/development paths, traversal, model-weight extensions, extra accelerator modules, and misplaced native libraries.
2. Implement `package_manifest` creation/validation and the build-only `julie-package-manifest` helper. The helper writes atomically and verifies an already-staged directory.
3. Extend Bash packaging and add the PowerShell mirror. Both choose an explicit profile, reject effective rustflags containing `-Ctarget-cpu=native`, locate the exact native outputs, stage required shared libraries flat beside the executable, invoke the shared helper, and verify before archiving. Linux dynamic profiles must verify the executable contains an `$ORIGIN` runpath; Windows profiles must launch without mutating `PATH`.
4. Define portable profiles: Apple arm64 Metal, Windows x64 Vulkan, and Linux x64 Vulkan. Retire the existing macOS x64 CPU-only lane from the new release matrix because the approved design requires every supported package to carry an accelerator; record this deliberate compatibility change in the baseline and release findings. Define optional CUDA profiles but do not call them supported until Task 7 evidence exists.
5. Keep ROCm out of release output until the Miller health amendment is merged. Keep SYCL out until a verified crate/build path exists.
6. Make archive names encode version, target, and backend tier so portable and vendor artifacts cannot collide.
7. Build the local Apple Metal package, unpack it into a new temporary directory, verify its manifest, and prove forced CPU from the same archive. For each dynamic profile, inspect runtime dependencies with the platform tool (`ldd`/`readelf`, `otool`, or `dumpbin`) and fail on build-tree paths or unresolved libraries.

**Acceptance:**

- [ ] Every archive contains one sidecar executable, CPU inference, exactly one advertised accelerator, and no model weights.
- [ ] Every file is declared with a verified checksum; no undeclared native library or development path survives.
- [ ] Dynamic modules and their dependencies load from the unpacked flat executable-relative layout without `PATH`, `LD_LIBRARY_PATH`, or build-tree help.
- [ ] Bash and PowerShell use one manifest schema/validator.
- [ ] Publication remains absent from both scripts.

## Task 5: Encode package, conformance, and hardware gates in automation

**Files:** per ownership table.

**What to build:** CI compiles every portable profile and validates archive structure without claiming hardware support. Hardware workflows consume an exact archive and emit raw evidence. Release workflow remains `workflow_dispatch`, artifact-only, and approval-gated.

**Steps:**

1. Add archive smokes that run the unpacked binary through `--version`, absent-model health, prepared-model health, query, batch, shutdown, stdout purity, and forced CPU fallback.
2. Add a runtime binary-path override to `tests/conformance.rs`, falling back to `CARGO_BIN_EXE_julie-semantic-sidecar` for ordinary Cargo tests. Make `scripts/conformance.sh` accept an unpacked binary path and a requested backend, rather than hard-coding the Cargo-built binary and forced CPU, so the same Groups A/B/C suite can prove CPU and accelerator runs from an archive.
3. Add hardware scripts that reject software renderers, collect device/driver/runtime identities, run CPU and accelerator conformance, measure fixed batch-1/indexing batches, invalidate/rebuild the selection cache, and prove fallback.
4. Update CI to build/test CPU and compile/package the portable target matrix. Mark compile-only jobs explicitly as artifact validation, not support evidence.
5. Update the release workflow to remove the macOS x64 CPU-only leg, produce portable Apple arm64 Metal and Windows/Linux x64 Vulkan archives plus optional CUDA candidates, upload manifests/checksums/raw logs, and perform no release publication.
6. Add workflow inputs for exact archive checksum and hardware lane; do not automatically promote a backend from a successful build.
7. Run all local workflow-equivalent checks that do not require external hardware or paid runners.

**Acceptance:**

- [ ] Package smokes execute only from an unpacked archive.
- [ ] CPU and advertised accelerators use the same protocol and golden-vector surface.
- [ ] Workflow names and findings distinguish compile proof from real-device proof.
- [ ] No workflow creates a tag, GitHub release, public asset, or Miller pin automatically.

## Task 6: Prove the exact Apple Silicon Metal archive

**Files:** per ownership table.

**Steps:**

1. Produce the Apple arm64 portable archive from the task branch and record its SHA-256 and embedded package manifest.
2. Unpack it outside the repository and run the full archive smoke and BGE/Qwen conformance suites.
3. Run real CPU and Metal batch-1/indexing-batch measurements under the idle-system gate; remove the selection cache between uncached trials and verify cached reuse separately.
4. Prove Metal wins before accepting `resolved_backend: metal`; otherwise keep CPU as the default winner and record the measured result.
5. Force CPU and induce an unavailable-accelerator path to prove ready degradation and non-null reasons.
6. Record raw commands/results, GPU/driver/runtime identity, package checksum, health objects, and any defects fixed.
7. Rerun the final full Rust, Metal, package, and conformance gates after fixes.

**Acceptance:**

- [ ] The exact archive passes all protocol and applicable golden vectors on CPU and Metal.
- [ ] Health truthfully matches the selected winner and fallback result.
- [ ] The findings record is sufficient to reproduce the evidence.

## Task 7: Prove Windows/Linux portable and vendor lanes

**Approval boundary:** Ask before starting paid GitHub Actions or other metered hardware. If user-provided compatible runners are free and already authorized, proceed without changing the acceptance bar.

**Lane order:** Linux Vulkan, Windows Vulkan on NVIDIA, Windows Vulkan on AMD or Intel, Linux/Windows CUDA, Linux ROCm only after the contract amendment, and Intel SYCL only after both the contract amendment and verified build support.

**Steps per lane:**

1. Build the exact archive, record its checksum, and execute the same archive/conformance/hardware script as Task 6.
2. Prove a physical compatible GPU, not lavapipe/software rendering.
3. Prove CPU fallback from that same archive.
4. For vendor packages, compare against the portable package on the same hardware and retain the vendor lane only if it materially improves the fixed benchmark without weakening correctness.
5. File and fix implementation defects in the sidecar branch, regenerate the archive, and discard superseded evidence.

**Acceptance:**

- [ ] Windows and Linux portable Vulkan archives have real-device proof and CPU fallback proof.
- [ ] CUDA is called supported only after real NVIDIA proof.
- [ ] ROCm and SYCL remain unshipped until their separate contract/build prerequisites and real hardware gates pass.

## Task 8: Integrate approved portable assets into Miller

**Approval boundary:** Do not execute until the user approves sidecar publication and the exact portable assets exist. Recheck both repos/worktrees after approval; any changed commit or dirty state invalidates the approval snapshot.

**Miller worktree:** Create or reuse a clean dedicated worktree from the then-current Miller `main`. Never edit `/Users/murphy/source/miller` while its unrelated untracked design file is present.

**Steps:**

1. Publish only the approved, fully proved sidecar assets and checksums. Download each published archive again and reverify its manifest/checksum before changing Miller.
2. Update `scripts/semantic-pins.json` with exact version, target/backend filename, URL, archive SHA-256, and manifest identity for Metal on Apple arm64 and Vulkan on Windows/Linux x64.
3. Update Bash/PowerShell restore scripts to preserve the archive's flat executable-relative native-library layout and verify both outer pin and inner package manifest.
4. Extend package smoke and release workflow tests for the backend directory and manifest while preserving the explicit `serve --model bge-small-en-v1.5-f32` launch.
5. Run Miller semantic prepare, sidecar health, vector convergence, semantic/fused search, scale tests, and lexical fallback from the restored final package.
6. Update release docs with the portable package contract; keep vendor selection outside the public tool surface.
7. Run Miller's focused tests, full affected solution gate, package smoke, and worktree-state checks. Do not push or release without a second explicit approval.

**Acceptance:**

- [ ] Miller pins exact proved portable archives and preserves their backend layout.
- [ ] Miller always passes BGE explicitly and never depends on the sidecar omission default.
- [ ] Prepare, convergence, search, health, and lexical fallback pass from restored packages.
- [ ] Sidecar and Miller versions, filenames, manifests, checksums, help text, and docs agree.

## Completion Boundary

- This plan implements design Phases 0–5 only.
- Phase 6 BGE versus CodeRankEmbed requires a new Miller-owned evaluation plan after portable package evidence exists.
- Phase 7 .NET/ONNX Runtime remains conditional and requires its own approved spike plan.
- The plan is not complete until Tasks 1–6 pass locally, approved real-hardware Tasks 7 pass for promoted lanes, and approval-gated Task 8 passes against exact published assets.
