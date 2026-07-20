# P2a plan — codex adversarial review record

Reviewer: codex (adversarial, read-only, schema output). Verdict: needs-attention, 15 findings.
Lead verification: 15/15 verified real against the frozen contract, fixtures, and reference sources;
zero dismissed. The plan was rewritten in place before any implementation task dispatched.

| # | Sev | Finding | Disposition |
|---|---|---|---|
| 1 | critical | P1 contracts/fixtures not published at cited Miller-main paths | Overtaken by events: Miller PR #7 merged (`60b6a96`) mid-review; plan now pins Miller `8edfa14`+ for all paths and CI checkouts. Design-doc six-code residue was already corrected in P1. |
| 2 | critical | Truncation formula omitted tokenizer special-token overhead | Lead-owned P1 contract amendment `8edfa14` (pre-ship window): frozen algorithm now includes `special_token_overhead`; plan Task 5 carries the full formula plus both models' constants. |
| 3 | critical | Branch gate covered only numeric conformance, not Groups A/B | Task 7 rewritten: every A1–A23/B1–B6 row as a named assertion through the spawned packaged binary; B4/B5 budgets are hard gates; 250-position batch probe required. |
| 4 | high | No model-selection or CPU-forcing launch interface | Global Constraints freeze `serve [--model <id>]` (default tier arg-less for Julie drop-in) + `JULIE_SIDECAR_FORCE_BACKEND=cpu` (non-wire). |
| 5 | high | Batch A could not build independently (Cargo.toml/main.rs ownership, duplicated manifest structs, no lib target) | Task 1 now creates `src/lib.rs` + all stub modules + pre-declared non-llama deps; Task 4 serialized after Task 3 with single manifest ownership; verb wiring stubbed in Task 1. |
| 6 | high | Missing-model health only unit-assembled, never process-integrated | Task 6 adds the unready serve path (loop without load, exact `model_not_prepared`, startup stale-partial cleanup); Task 7 B3 tests it end-to-end. |
| 7 | critical | macOS cache path contradicted the contract | Fixed: `~/.cache/julie-semantic` on Linux AND macOS, `%LOCALAPPDATA%`-rooted Windows — contract verbatim. |
| 8 | high | Dynamic backend loading tied to build tree, not packaged archive | Tasks 6+8: executable-relative path-based loading, frozen archive layout in `package.sh`, Linux packaged Vulkan-load assertion with a lavapipe ICD, Windows clean-fallback assertion. |
| 9 | high | Stdout guard covered model load only | Guard now wraps load + backend probe + micro-benchmark; A22 whole-session purity test. |
| 10 | high | CI had no way to obtain model weights | Task 8 adds a sha256-keyed cache stage populated by the binary's own `prepare` (doubling as an E2E test). |
| 11 | high | llama.cpp drift premise unverifiable; crates not exact-pinned | Both crates `=`-pinned; vendored llama.cpp build recorded at implementation and surfaced as `llama_cpp_build`; drift direction treated as unknown until recorded; diagnosis precedes any escalation. |
| 12 | medium | Non-string sanitization contradicted wire behavior | Two-layer rule frozen: wire non-strings → `invalid_request` (A14/A15); `[empty]` only for empty/whitespace/NUL-blank strings at the engine. |
| 13 | medium | `id` alias lacked precedence rule | `request_id` precedence frozen from `protocol.py:44-57`; both-keys test cases required in Task 2. |
| 14 | medium | Backend cache key omitted driver identity | Key now `sidecar_version + model_sha256 + GPU identity + driver identity`; per-component invalidation tests required. |
| 15 | medium | `prepare` machine interface undecided | Frozen in Task 4: NDJSON event vocabulary (progress/waiting/done/error), exit codes 0/1/2, lock-waiter behavior, fail-loud offline semantics. |

Raw reviewer output: session scratchpad `p2a-plan-review.json` (not committed; findings reproduced above).
