# Autonomous Execution Report - P2a julie-semantic-sidecar Implementation

**Status:** Complete (PR open; merge pending first-ever CI run)
**Plan:** docs/plans/2026-07-19-p2a-sidecar-implementation-plan.md
**Branch:** p2a-implementation
**PR:** pending — filled in after PR creation
**Phases:** 3/7 program phases complete (P0, P1, P2a of P0–P6)
**Tasks:** 8/8 plan tasks + 1 gate-found fix + 7/7 pre-merge review findings fixed

## What shipped
- Complete Rust stdio sidecar speaking frozen `julie.embedding.sidecar` v1: protocol loop (38 row-named tests), embedded manifest + health assembly, `prepare` with atomic verified downloads, llama-cpp-2 =0.1.151 engine (frozen truncation incl. special_token_overhead, MRL slice+renormalize, batch isolation), fd-level stdout guard, honest backend selection with cached choice, unready-serve path.
- Conformance harness = the contract's full Groups A (23 rows), B (6 rows incl. hard 120s/30s budgets: measured 213–225 ms and 7.8 s), C (39 texts + 250-position probe, BOTH models, cosine ≥0.999) against Miller fixtures, spawning the real release binary. GATE GREEN both models.
- CI matrix (4 platforms, model cache keyed by pin sha256s, pinned Miller fixture checkout) + release.yml (workflow_dispatch, artifacts only, NO publish armed) + package.sh frozen layout.
- Two Miller contract amendments (pre-ship window, on worktree-semantic-p2): special_token_overhead in truncation (8edfa14), not-ready health owes only ready+degraded_reason (b5858a1).

## Judgment calls
- max_request_bytes = 32 MiB (contract names the field, states no value; derivation recorded in task-3 report) — flag if you want a different number.
- llama_cpp_build = "llama-cpp-2 0.1.151 (llama-cpp-rs 7f0a0d95, vendored llama.cpp)" — the published crate strips git metadata, so no upstream b-number is recoverable offline.
- qwen3 32k inputs use decode (causal) not encode: encoder n_ubatch assert makes a 32k encode physically impossible (15 GB buffer, SIGSEGV). Contract-consistent; recorded in task-5 report.
- Serve verifies model digest at every start (~1 s/GB release; within the 120 s budget) after codex F2; memoize later if per-request spawning ever appears.

## External review (codex, adversarial)
- Plan review: 15 findings, 15 verified real, 0 dismissed → plan rewritten before execution (docs/plans/2026-07-19-p2a-plan-review-record.md).
- Pre-merge branch review: 7 findings, 7 verified real, 0 dismissed. Fixed: forced-CPU was metadata-only (29/29 layers were on Metal while health said cpu — placement now enforced, proven by load-log assertion), digest check at serve load, systemic failures no longer zero-vector "successes", lock-guarded partial cleanup, download size ceiling, wire limits enforced (042a56f, 2340eb9, 71560f4). F7: accelerated Metal/Vulkan packaging honestly re-marked as DEFERRED follow-up scope (51fa34e).
- Codex surfaces no per-request token counts.

## Tests
- 160 non-ignored tests green; model-backed ignored set green; full conformance A/B/C green both models post-fixes (89.9 s). clippy --all-targets -D warnings + fmt clean.
- The conformance gate caught a real engine bug live (fixed-8-byte detokenize buffer vs bge 9–11-byte WordPiece pieces) — fixed (0c6cd7f) and re-proven.

## Blockers hit
- None. Windows CI leg is the first-ever compile of the cfg(windows) stdio_guard arm — may fail visibly on PR CI; fix-forward if so.

## Next steps
- Merge PR when the first CI run is green (Windows leg is the watch item).
- Follow-up unit (one lane): accelerated Metal/Vulkan builds + backend-DL packaging + packaged Vulkan-load assertion + real micro-benchmark timing + GPU/driver identity providers.
- P2b–P2e Miller lanes per design §10; Miller worktree branch worktree-semantic-p2 carries the two contract amendments and needs a PR eventually.
- User sanity-checks still open from P0/P1 reports (five frozen canary statistical values; pre-ship amendment policy).
