# P2a julie-semantic-sidecar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use razorback:subagent-driven-development when subagent delegation is available. Fall back to razorback:executing-plans for single-task, tightly-sequential, or no-delegation runs.

**Goal:** Implement `julie-semantic-sidecar` — a thin Rust stdio binary speaking the frozen `julie.embedding.sidecar` v1 protocol, embedding text via a pinned llama.cpp with Metal/Vulkan/CPU backends — and prove it against the full three-group conformance contract (protocol, lifecycle, numeric) on all four platforms.

**Architecture:** P2 lane (a) of the program design (`~/source/miller/docs/plans/2026-07-19-miller-semantic-integration-design.md` §4, §10). The wire contract is FROZEN in Miller's `docs/contracts/semantic-sidecar-protocol-v1.md` **at Miller main commit `8edfa14` or later** (the P1 merge `60b6a96` plus the special-token-overhead truncation amendment) — this plan implements it, never amends it; a contract defect found during implementation is a lead-owned Miller-repo amendment (pre-ship window), reported as a plan mismatch. The engine binding is `llama-cpp-2` (decision memo: exact-pinned crate, vendored llama.cpp, backend-DL plugins for Vulkan, embedded Metal shaders — users install nothing).

**Tech Stack:** Rust (edition 2021+), `llama-cpp-2` and `llama-cpp-sys-2` BOTH exact-pinned (`=` constraints) with committed `Cargo.lock`; serde/serde_json for NDJSON; no async runtime (blocking stdio loop); no Python anywhere.

**Architecture Quality:** Approved shape: one crate with a library target (`src/lib.rs` exposing every module; `src/main.rs` a thin verb dispatcher) so integration tests exercise the same code paths the binary runs; a pure protocol module with the engine behind a trait; an engine module owning all model knobs keyed by model identity; a single embedded model manifest owned by one module; a `prepare` subcommand. Main risks: llama.cpp packaging drift per platform (mitigated: exact pins, packaged-layout tests, executable-relative backend loading) and gate theater (mitigated: Task 7 executes every contract conformance row through the packaged stdio binary, not unit shims).

## Global Constraints

- **The protocol contract is law — pinned edition.** `~/source/miller/docs/contracts/semantic-sidecar-protocol-v1.md` at Miller main `8edfa14`+ (read it in full before implementing anything): schema `"julie.embedding.sidecar"`, `version: 1`, methods `health | embed_query | embed_batch | shutdown`, error codes `invalid_request | invalid_json | unknown_method | internal_error` ONLY, `id` accepted as `request_id` alias **with `request_id` precedence** (the reference checks `request_id` first; an invalid `request_id` is `invalid_request` even when a valid `id` is present — `~/source/julie/python/embeddings_sidecar/sidecar/protocol.py:44-57`), `schema`/`version` optional on inbound, `shutdown` → `{"stopping": true}`, dims echo, batch count match, exactly-one-of result/error, request-id echo (`""` for unparseable lines), blank lines skipped, unknown top-level fields ignored.
- **Launch interface (frozen by this plan; non-wire, so it does not amend the protocol):** `serve [--model <manifest-id>]` — omitted means the default tier (`qwen3-0.6b-f16`), preserving the arg-less drop-in launch Julie's consumer uses; `prepare [--model <manifest-id>]` per the contract. Env `JULIE_SIDECAR_FORCE_BACKEND=cpu` forces the CPU backend, skipping probe and micro-benchmark (conformance/CI use; documented in README as a diagnostic surface, never required for normal operation). `JULIE_EMBEDDING_CACHE_DIR` per the contract. No other knobs.
- **Model pins** (embedded manifest; values from `~/source/miller/eval/model-bench/bench-pins.json`, single source): `qwen3-0.6b-f16` (Qwen3-Embedding-0.6B f16 GGUF, sha256 `421a27e58d165478cc7acb984a688c2aa41404968b0203e7cd743ece44c54340`, size 1197629632, native 1024d MRL, lanes [256,512,1024], serve 512d, pooling `last`, EOS `<|endoftext|>`, query instruction `"Instruct: Given a code search query, retrieve the code or documentation that answers it\nQuery: "`, `max_text_tokens` 32768) and `bge-small-en-v1.5-f32` (sha256 `bf40c42ad7d89382e9ba7376d5c4b73f6b556cb541fab37aaa1da9c320149b65`, size 133609568, 384d, pooling `cls`, no EOS, query instruction `"Represent this sentence for searching relevant passages: "`, `max_text_tokens` 512). Document instructions are empty strings. Source URLs copied from bench-pins.json; `model_revision` for the encoder fingerprint is the HF revision component of those URLs (`main`) with the sha256 as the integrity anchor.
- **Truncation (frozen, amended contract):** per input, cut the tokenized prefixed string to `max_text_tokens − eos_reserve − special_token_overhead`, where `special_token_overhead` is the add-special tokenization delta measured once per model (2 for bge's `[CLS]`/`[SEP]`, 1 for qwen3), then apply the detokenize round-trip stability rule, then append EOS. Fit BEFORE EOS append. The sidecar serves **lane-sliced, re-normalized f32** vectors at the dims `health` declares (slice+renormalize in-shim; wire never carries quantized vectors).
- **Sanitization scope (two layers, do not conflate):** at the WIRE, a non-string `embed_query.text` or non-string batch element is `invalid_request` (contract rows A14/A15) — never sanitized. At the ENGINE, a string that is empty, whitespace-only, or becomes blank after NUL-stripping embeds as the literal `"[empty]"`; NUL bytes are stripped from all strings. Sanitization never errors.
- **Batch isolation:** binary-search a failing batch to the poison text; zero-vector the single bad item; response shape unchanged (N vectors), process stays alive (contract row A18).
- **Stdout purity:** nothing but protocol NDJSON ever reaches fd1 for the whole session (contract row A22). fd-level redirection (dup/dup2 with finally-restore) wraps EVERY native call that may print: model load, backend probe, AND the first-start micro-benchmark.
- **Backends:** CPU always, built `-DGGML_NATIVE=OFF` for cross-machine determinism; macOS arm64 Metal with embedded shaders; macOS x64 CPU-only; Linux/Windows Vulkan via backend-DL plugin layout loaded from an **executable-relative path** (the crate's no-argument loader resolves a compile-time build directory and can no-op in a packaged install — use the path-based loading API against a frozen archive layout). Missing Vulkan loader or no GPU = silent CPU fallback with degraded health reporting, never an error. Users install nothing.
- **Pins and build identity:** `llama-cpp-2` AND `llama-cpp-sys-2` exact-pinned with `=` (verify the exact latest 0.1.x at implementation time — the memo's `=0.1.151` is the floor candidate; record the verified pin); `Cargo.lock` committed. Task 5 must RECORD the vendored llama.cpp commit/build number (from the crate's vendored source) in its report and surface it as `llama_cpp_build` in health/manifest. The conformance goldens were generated on llama.cpp release `b10068`; whether the crate's vendored llama.cpp is older or newer is UNKNOWN until recorded — do not assume drift direction. If CPU conformance fails tolerance, first diagnose against the recorded vendored build; a genuine upstream-drift failure is a STOP (plan-mismatch report): golden re-pinning is a lead-owned Miller-repo decision, not a local workaround.
- **Conformance is the gate — all three groups.** Task 7 must execute every row of the contract's Group A (A1–A23), Group B (B1–B6, including the hard 120,000 ms cold-health and 30,000 ms 250-text-batch budgets — these are HARD gates, not report-only), and Group C (dims exact, |norm−1| ≤ 1e-3 on wire floats, cosine ≥ 0.999 per text, both roles, BOTH pinned models, every corpus text) against Miller's fixture set `~/source/miller/eval/sidecar-conformance/` at the pinned Miller commit, through the REAL packaged stdio binary.
- **Cache/env:** `JULIE_EMBEDDING_CACHE_DIR` → else `~/.cache/julie-semantic` (Linux AND macOS), `%LOCALAPPDATA%`-rooted on Windows — exactly the contract's rule; no `Library/Caches` variant. Shared with Julie by construction.
- License MIT. No network access at embed time; only `prepare` downloads. No publish/release without explicit user approval.
- Rust hygiene: `cargo clippy -- -D warnings` clean; `cargo fmt --check` clean; tests via `cargo test`. Comments only for non-obvious constraints (repo rule mirrors Miller's).

## Verification Strategy

**Project source of truth:** this plan + AGENTS.md (Task 1 writes it); commands are standard cargo.

**Worker red/green scope:** `cargo test <module filter>` for the task's module; protocol/manifest/prepare tasks must not require a model download (fake engine, local HTTP fixtures). Engine-touching tasks may run the real CPU engine iff the model file is already in the local cache (`prepare` from Task 4, or copy `~/source/miller/eval/model-bench/.cache/dist/*.gguf` into the cache layout to avoid re-downloading).

**Worker ceiling:** `cargo test` (whole crate) + `cargo clippy -- -D warnings` + `cargo fmt --check`. Workers do not run the cross-platform CI or release packaging.

**Worker gate invariant:** stated per task.

**Lead affected-change scope:** full `cargo test` + clippy + fmt after each batch.

**Branch gate:** `cargo test` + clippy + fmt + the Task 7 conformance harness (Groups A+B+C, both models, CPU-forced) green on local macOS arm64 against Miller's fixtures at the pinned commit.

**Replay/metric evidence:** Group B budgets (120 s cold health, 30 s 250-text batch) and Group C tolerances are HARD gates. Embed throughput and micro-benchmark timings beyond those budgets are report-only.

**Escalation triggers:** conformance tolerance failure after the vendored-build diagnosis → STOP, lead-owned. Any need to change the protocol contract → STOP, plan mismatch.

**Assigned verification failure:** workers stop and report; never weaken a gate.

**Verification ledger:** `.razorback/sdd/progress.md` in this repo; same rules as Miller's.

## Parallel Execution Contract

| Task | Parallel batch | File ownership | Serialization required | Dependency reason |
|---|---|---|---|---|
| Task 1: Repo bootstrap | None - serial | Create: `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `.gitignore`, `LICENSE`, `AGENTS.md`, `README.md`, `src/main.rs`, `src/lib.rs`, stub modules `src/protocol.rs`, `src/engine_trait.rs`, `src/manifest.rs`, `src/health.rs`, `src/prepare.rs`, `src/sanitize.rs`, `src/truncate.rs` | Yes | Everything depends on the workspace skeleton, the lib target, and the pre-declared dependency set. |
| Task 2: Protocol core | Batch A | Modify: `src/protocol.rs`, `src/engine_trait.rs`; Create: `tests/protocol_tests.rs` | No | None - safe parallel batch. |
| Task 3: Manifest + health assembly | Batch A | Modify: `src/manifest.rs`, `src/health.rs`; Create: `tests/manifest_tests.rs` | No | None - safe parallel batch. |
| Task 4: prepare subcommand | None - serial | Modify: `src/prepare.rs`; Create: `tests/prepare_tests.rs` | Yes | Consumes Task 3's manifest module directly (single manifest ownership — no duplicated pin structs). |
| Task 5: Engine (llama-cpp-2) | None - serial | Create: `src/engine.rs`, `tests/engine_tests.rs`; Modify: `Cargo.toml` (llama deps + features), `src/lib.rs` (engine module), `src/main.rs` (serve wire-up), `src/sanitize.rs`, `src/truncate.rs` | Yes | Integrates Batch A's trait/manifest surfaces; sole owner of Cargo.toml/main.rs changes after Task 1. |
| Task 6: Backend selection, stdout purity, unready state | None - serial | Create: `src/backend_select.rs`, `src/stdio_guard.rs`; Modify: `src/engine.rs`, `src/main.rs`, `src/lib.rs` | Yes | Modifies Task 5's files. |
| Task 7: Conformance harness | None - serial | Create: `tests/conformance.rs`, `scripts/conformance.sh` | Yes | Needs the full engine and lifecycle behavior (Tasks 5–6). |
| Task 8: CI + packaging draft | None - serial | Create: `.github/workflows/ci.yml`, `.github/workflows/release.yml`, `scripts/package.sh` | Yes | Encodes the build matrix and packaged layout proven by Tasks 5–7. |

## Task 1: Repo bootstrap

**Files:** per ownership row.

**Interfaces:** Produces the workspace every task builds in: one crate `julie-semantic-sidecar` with BOTH a library target (`src/lib.rs` declaring `pub mod protocol; pub mod engine_trait; pub mod manifest; pub mod health; pub mod prepare; pub mod sanitize; pub mod truncate;` — integration tests link the lib) and a binary whose `src/main.rs` parses verbs: `serve [--model <id>]` (default verb and default model `qwen3-0.6b-f16`), `prepare [--model <id>]`, `--version` (prints `julie-semantic-sidecar <cargo version>`), unknown verb → exit 2 with usage on stderr. Verb handlers call stub functions in the owned modules (`prepare::run(model_id: Option<&str>) -> std::process::ExitCode`, serve loop stub blocking on stdin EOF) so later tasks fill bodies without touching `main.rs`. `Cargo.toml` pre-declares every non-llama dependency Batch A + Task 4 need: `serde` (derive), `serde_json`, `sha2`, `ureq` (or `minreq`; pick one, record it), a file-lock crate (`fs4` or `fd-lock`), `tempfile`, `dirs`, and dev-dep `tiny_http` (prepare tests' local fixture server). No llama dependency yet (Task 5 adds it). `rust-toolchain.toml` pins current stable.

**Contract inputs:** Global Constraints (launch interface, license, hygiene).

**Serialization:** Yes — see table.

**What to build:** Minimal compiling workspace with the verb skeleton, lib/bin split, pre-declared deps, and repo hygiene files. AGENTS.md: concise working notes — the pinned Miller contract/fixture paths and commit, the frozen-contract rule (defects are Miller-repo amendments, not local edits), verification commands, the no-publish rule, the two-layer sanitization rule. LICENSE: MIT, copyright Alan Northam. `.gitignore`: `/target`, `.razorback/`, local model caches.

**Acceptance criteria:**
- [ ] `cargo build` + `cargo test` + clippy + fmt green on the stub
- [ ] `--version` prints the semver; unknown verb exits 2; `serve` blocks on stdin and exits on EOF; `prepare` verb dispatches to `prepare::run` stub
- [ ] `src/lib.rs` exposes all stub modules; an integration test can `use julie_semantic_sidecar::protocol` (compile check)
- [ ] AGENTS.md carries the contract pointers (with pinned Miller commit) and command reference

## Task 2: Protocol core (pure, engine behind a trait)

**Files:** per ownership row.

**Interfaces:**
- Consumes: Task 1's stub modules and dep set.
- Produces: `trait EmbedEngine` (in `src/engine_trait.rs`: `health_facts()`, `embed(texts: &[String], role: Role) -> Result<EmbedOutput, EngineError>` — refine as the contract requires; document the final shape in the report) and `protocol::run_loop(stdin, stdout, engine)` implementing the FULL wire contract: NDJSON parse, envelope validation, the four error codes with the contract's exact emission conditions, `request_id` precedence over the `id` alias (invalid `request_id` errors even beside a valid `id`), optional inbound schema/version, wrong schema/version → `invalid_request` with request-id echo, request-id echo `""` on unparseable lines, exactly-one-of, blank-line skip, unknown-field tolerance, non-string text/element → `invalid_request`, `shutdown` → `{"stopping": true}` + loop break, dims echo, batch count match, empty batch → empty vectors, process-alive-after-error.
- Compact JSON output (no pretty-print), one line per response.

**Contract inputs:** the contract's § Envelopes/§ Methods/§ Errors plus conformance Group A rows A1–A21, A23 as the test checklist; Julie's consumer tests (`~/source/julie/crates/julie-pipeline/src/embeddings/sidecar_protocol.rs` + `~/source/julie/src/tests/core/embedding_sidecar_provider.rs`, read-only) as behavioral vectors; `~/source/julie/python/embeddings_sidecar/sidecar/protocol.py` as the reference for precedence/validation order.

**Serialization:** No — Batch A.

**What to build:** The protocol module + a `FakeEngine` test double. Tests: one per Group A row that is testable without a real engine (A1–A17, A19–A21, A23 — batch isolation A18 lands with the engine in Task 5), plus the both-keys precedence cases (valid+valid → request_id wins; invalid request_id + valid id → `invalid_request`).

**Acceptance criteria:**
- [ ] Every § Errors emission condition and every fake-testable Group A row has a named test; error vocabulary is exactly the four codes
- [ ] `request_id`-precedence tests pass against the reference's behavior
- [ ] Protocol tests run with no llama dependency and no network

## Task 3: Manifest + health assembly

**Files:** per ownership row.

**Interfaces:**
- Consumes: Task 1's stub modules.
- Produces: `manifest::manifest()` — THE single embedded model manifest (both pins, all knobs from Global Constraints, source URLs, sizes, `model_revision`) — and `health::build(engine_facts, model_state) -> serde_json::Value` assembling the contract's full health response: `ready`, `degraded_reason` (incl. the exact string `model_not_prepared`), `capabilities` (torch-compat four keys `cpu|cuda|directml|mps` each an object with boolean `available`, plus additive `metal`/`vulkan`), `accelerated`, `load_policy` (requested/resolved backend; `load_policy.accelerated` and `load_policy.degraded_reason` mirror the top-level fields; degraded_reason non-null whenever requested≠resolved), model identity fields, `pooling`, `normalization: "l2"`, `instruction_policy_version: 1`, `max_text_tokens`, `max_batch_items`, `max_request_bytes`, `native_dims`, `mrl_lanes`, `dims` (present iff ready), `llama_cpp_build`, `sidecar_version`.

**Contract inputs:** contract § Health metadata + § Model knob table + Group A rows A7–A9; `dims` optional when `ready:false` (contract cites `sidecar_protocol.rs:168-170`).

**Serialization:** No — Batch A.

**Acceptance criteria:**
- [ ] Manifest values byte-match Global Constraints (tests against literal strings, both models)
- [ ] Health assembly tested for: ready model, missing model (`ready:false` + exact `model_not_prepared`, no `dims` required), degraded backend (A9 mirror invariants enforced)
- [ ] All four torch-compat capability keys always present as objects with boolean `available`

## Task 4: `prepare` subcommand

**Files:** per ownership row.

**Interfaces:**
- Consumes: `manifest::manifest()` from Task 3 — single ownership, NO duplicated pin structs.
- Produces: `prepare::run(model_id: Option<&str>) -> ExitCode` with this FROZEN command interface (settling the contract's "format is a decision" clause):
  - stdout emits NDJSON events only: `{"event":"progress","model_id":...,"received_bytes":N,"total_bytes":N}` (at most ~1/s), `{"event":"waiting","model_id":...}` when another invocation holds the lock, `{"event":"done","model_id":...,"path":...,"sha256":...}`, `{"event":"error","model_id":...,"message":...,"expected_path":...,"source_url":...}`.
  - Exit code 0 on success (including already-cached), 1 on any failure, 2 on unknown model id (message lists known ids).
  - Behavior: resolve cache dir per the contract rule; disk preflight against manifest size; download to a temp file in the cache dir; streaming sha256 verify BEFORE atomic rename; mismatch deletes temp + `error` event + exit 1; cache lock file makes concurrent invocations safe (one downloads, waiters block then observe the finished file); network unreachable → `error` event naming model id, expected path, and source URL + exit 1 (fail loud, never half-prepare); already-present verified file → `done` immediately.
- Also produces `prepare::clean_stale_partials(cache_dir)` — removes orphaned temp files; `serve` calls it at startup (wired in Task 6).

**Contract inputs:** contract § prepare subcommand obligations table (verbatim); Global Constraints cache-path rule.

**Serialization:** Yes — after Task 3 (manifest ownership).

**What to build:** The full download path, tested against a local `tiny_http` fixture server (no live network in tests): success, sha256 mismatch, disk-preflight failure (fake a huge manifest size), concurrent-lock (two threads, one server), unknown id, cache-dir env override, stale-partial cleanup.

**Acceptance criteria:**
- [ ] All seven fixture-server scenarios tested; sha256 mismatch provably deletes the temp and exits 1
- [ ] Event stream shape matches the frozen interface above (tests parse stdout lines as JSON)
- [ ] No live-network access in any test

## Task 5: Engine — llama-cpp-2 integration

**Files:** per ownership row.

**Interfaces:**
- Consumes: `EmbedEngine` trait (Task 2), `manifest::manifest()` (Task 3).
- Produces: `LlamaEngine` implementing the trait: model load from cache path, context with embeddings enabled + pooling per manifest, tokenize/detokenize, the frozen truncation algorithm in `src/truncate.rs` (`max_text_tokens − eos_reserve − special_token_overhead`, add-special-delta measurement, tail cut, round-trip stability loop, EOS append last), engine-layer sanitization in `src/sanitize.rs` (NUL strip; empty/whitespace/blank-after-strip → `"[empty]"`), instruction prefixing per role, encode → per-sequence embeddings, L2 normalize, MRL slice→renormalize to the manifest serve-lane dims, count/dims invariants, batch binary-search isolation emitting zero-vectors for poison texts (A18), per-batch memory hygiene.
- Modifies `Cargo.toml`: adds `llama-cpp-2 = "=<verified>"` AND `llama-cpp-sys-2 = "=<verified>"` with the feature set for CPU-default builds; commits `Cargo.lock`.
- REPORTS the vendored llama.cpp commit/build number and surfaces it as `llama_cpp_build`.

**Contract inputs:** decision memo (crate APIs: `LlamaContext::encode`, `embeddings_seq_ith`, pooling via context params); amended contract § Truncation (six steps incl. `special_token_overhead`)/§ Prompt templates/§ Per-item failure isolation; golden evidence: bge truncation rows stabilize at 510 content tokens = 512 with specials (`~/source/miller/eval/sidecar-conformance/README.md`).

**Serialization:** Yes — see table.

**What to build:** The real engine, CPU-only in tests. `truncate.rs`/`sanitize.rs` are pure (token ops injected via a closure/trait so tests run without a model). Engine integration tests (`#[ignore]`-gated unless the model file is present in cache) load each model from the cache and assert: dims/norm/count invariants, a 2-text embed round trip, the bge truncation cut matching the goldens' 510-content-token point, and zero-vector isolation via a forced-failure seam.

**Acceptance criteria:**
- [ ] `truncate.rs` reproduces the amended contract algorithm (tests: below-budget no-op, exact-budget, over-budget cut at `max_text_tokens − eos_reserve − special_token_overhead` for both models' constants, stability-loop shrink case)
- [ ] `sanitize.rs` matches the engine-layer rules only (non-string handling is Task 2's wire rejection, not sanitization)
- [ ] Model-gated integration tests green locally with cached models; vendored llama.cpp build recorded in the report
- [ ] `cargo build` remains green with the CPU-default feature set (no Metal/Vulkan features)

## Task 6: Backend selection, stdout purity, unready state

**Files:** per ownership row.

**Interfaces:** Consumes Task 5's engine, Task 3's health assembly, Task 4's `clean_stale_partials`. Produces:
- `stdio_guard`: fd-level dup/dup2 redirection (fd1→fd2) with finally-restore, wrapped around EVERY native call that may print — model load, backend probe, and the first-start micro-benchmark. Test: a `serve` session that loads a model (model-gated) emits zero non-NDJSON stdout bytes from spawn to exit.
- Unready serve path: `serve` with the model absent from cache starts the protocol loop WITHOUT loading; `health` answers `ready:false, degraded_reason:"model_not_prepared"`; embed methods answer `internal_error` without crashing; startup runs `clean_stale_partials`.
- Backend selection: backend loading from an **executable-relative** plugin path (resolve `current_exe()`'s directory; use the crate's path-based loading API, NOT the no-arg helper that bakes in the build directory); probe encode after load with CPU fallback; first-start micro-benchmark (batch-1 + batch-N timing, accelerated vs CPU) with the choice cached beside the model cache, keyed by `sidecar_version + model_sha256 + GPU identity + driver identity` (define a stable GPU+driver fingerprint per platform; document the source per backend in the report); any key component change re-runs the benchmark; `JULIE_SIDECAR_FORCE_BACKEND=cpu` skips probe+benchmark entirely; "accelerated slower than CPU" → cached CPU choice, `ready:true, accelerated:false`, non-null degraded_reason (B6).

**Contract inputs:** contract § Stdout purity (implementation obligations verbatim) + § Backend selection + Group A row A22, Group B rows B3/B6.

**Serialization:** Yes.

**Acceptance criteria:**
- [ ] Stdout purity proven by a whole-session test (spawn → load → embed → shutdown, zero non-protocol stdout bytes)
- [ ] Unready serve path tested end-to-end without any model file (empty temp cache dir)
- [ ] Selection cache round-trips; invalidation tested for each key component including driver identity; a cached "cpu" choice skips the probe
- [ ] Backend plugin loading uses an executable-relative path (unit-test the path resolution; packaged proof lands in Task 8)

## Task 7: Conformance harness — Groups A, B, and C

**Files:** per ownership row.

**Interfaces:** Consumes the finished binary (Tasks 5–6). Produces the P2a GATE: `tests/conformance.rs` (`#[ignore]`-gated, run explicitly by `scripts/conformance.sh`) that spawns the REAL built binary as a child process over stdio and executes:
- **Group A (A1–A23):** every row as a wire assertion against the spawned process, including A18 poison isolation (a text the engine cannot encode — use the forced-failure seam or an over-limit construction), A22 whole-session stdout purity, A23 post-error liveness.
- **Group B (B1–B6):** B1 EOF → exit; B2 SIGKILL → no orphan/lock/cleanup burden; B3 missing model (empty `JULIE_EMBEDDING_CACHE_DIR`) → exact unready health; B4 cold start with model present → first health within **120,000 ms (HARD)**; B5 250-text `embed_batch` → answer within **30,000 ms (HARD)**; B6 CPU-over-GPU selection reporting (assert via `JULIE_SIDECAR_FORCE_BACKEND=cpu` that degraded/accelerated fields obey the contract shape).
- **Group C:** read `--fixtures <dir>` (Miller checkout at the pinned commit), embed all 39 corpus texts per model — `serve --model` each pinned model in turn, CPU-forced — role-correct (query texts via `embed_query`, document texts via `embed_batch`), including the 250-position expansion of `batch-group-001` the fixture README makes load-bearing, and apply the frozen tolerances (dims exact, |norm−1| ≤ 1e-3, cosine ≥ 0.999 per text — one failing text fails the run).
- Per-row results printed as a table; any failure names the row.

**Contract inputs:** contract § Conformance (all three group tables verbatim — the checklist IS the contract's rows); fixture README pass rule and batch-probe requirement; Global Constraints drift-diagnosis rule.

**Serialization:** Yes.

**Acceptance criteria:**
- [ ] Every A/B/C row from the contract appears as a named assertion; B4/B5 budgets enforced as hard failures
- [ ] Harness green on local macOS arm64 CPU for BOTH models against Miller fixtures at the pinned commit (branch-gate entry)
- [ ] A deliberately perturbed vector and a deliberately shifted truncation point each fail Group C (negative tests)
- [ ] A tolerance failure produces the vendored-build diagnostic (recorded `llama_cpp_build` vs goldens' `b10068`) before any escalation

## Task 8: CI + packaging draft

**Files:** per ownership row.

**Interfaces:** Consumes the proven build knobs and packaged layout from Tasks 5–7. Produces:
- `scripts/package.sh`: builds the release layout and FREEZES it: executable + required shared libraries + backend plugin modules in one directory (exact file list per platform recorded in the script), matching Task 6's executable-relative loader.
- `.github/workflows/ci.yml`: fast job (clippy+fmt+`cargo test`, no models) on ubuntu; conformance matrix on macos-15 (arm64, Metal build, CPU-forced conformance), macos-15-intel (CPU), ubuntu (Vulkan SDK build-time + backend-DL; install `mesa-vulkan-drivers` for a lavapipe ICD and assert the Vulkan backend module actually loads/registers from the packaged layout before CPU-forced conformance), windows-2025 (pinned LunarG SDK + backend-DL; assert the plugin file ships and the load attempt falls back cleanly to CPU — runner has no GPU). Model acquisition: a dedicated cache step keyed by the two manifest sha256s, populated by running the built binary's own `prepare` for both models (also an E2E test of Task 4); Miller checkout pinned to the fixture commit for `--fixtures`. sccache or cargo cache for build times.
- `.github/workflows/release.yml`: workflow_dispatch ONLY, no publish step armed — builds the 4 platform archives + sha256 sidecars via `package.sh`, then a packaged smoke: `--version`, one embed round-trip from the packaged layout, and the plugin-discovery probe (backend list from the packaged directory).

**Contract inputs:** decision memo CI matrix + risks; Task 6's frozen loader path rule; Global Constraints no-publish rule.

**Serialization:** Yes.

**Acceptance criteria:**
- [ ] ci.yml validates (actionlint) and encodes the matrix, the model-cache stage (prepare-driven, sha256-keyed), the pinned Miller fixture checkout, and the Linux packaged Vulkan-load assertion
- [ ] Conformance jobs run the FULL Task 7 harness (A+B+C, both models) on every platform leg
- [ ] release.yml has no armed publish; archives + sha256 + packaged smoke only
- [ ] `package.sh`'s frozen file list matches what Task 6's loader expects (one source of truth, cross-referenced)
