# julie-semantic-sidecar — agent working notes

A thin Rust stdio binary that embeds text via a pinned llama.cpp and speaks the frozen
`julie.embedding.sidecar` v1 protocol. Consumers are Miller and Julie; neither one owns
model acquisition, model paths, or download URLs — this binary does.

## The contract is law

- **Wire contract:** `/Users/murphy/source/miller/docs/contracts/semantic-sidecar-protocol-v1.md`
  at Miller `main` commit `8edfa14` or later (the P1 merge `60b6a96` plus the
  special-token-overhead truncation amendment). Read it in full before implementing anything.
- **Conformance fixtures:** `/Users/murphy/source/miller/eval/sidecar-conformance/` at the same
  pinned commit (`corpus.jsonl`, `golden-qwen3-0.6b-f16.jsonl`, `golden-bge-small-f32.jsonl`).
- **Frozen-contract rule:** this repo implements the contract, it never amends it. A contract
  defect found during implementation is a **lead-owned amendment in the Miller repo** — never a
  local edit, never a local workaround, never a weakened test. Report it as a plan mismatch and
  stop.
- **Implementation plan:** `docs/plans/2026-07-19-p2a-sidecar-implementation-plan.md`. Task file
  ownership and per-task gates live there.

## Launch interface (frozen by the plan; non-wire)

```
julie-semantic-sidecar [serve [--model <id>]]   # default verb; default model qwen3-0.6b-f16
julie-semantic-sidecar prepare [--model <id>]
julie-semantic-sidecar --version
```

Unknown verb → exit 2 with usage on stderr. Env knobs are exactly two:
`JULIE_EMBEDDING_CACHE_DIR` (per the contract) and `JULIE_SIDECAR_FORCE_BACKEND=cpu`
(diagnostic/CI only). No others.

## Load-bearing rules

- **Two-layer sanitization — do not conflate.** At the WIRE, a non-string `embed_query.text` or a
  non-string `embed_batch` element is `invalid_request` (contract rows A14/A15); it is never
  sanitized. At the ENGINE, a string that is empty, whitespace-only, or blank after NUL-stripping
  embeds as the literal `"[empty]"`, and NUL bytes are stripped from all strings. Engine
  sanitization never errors.
- **Stdout purity.** Nothing but protocol NDJSON ever reaches fd1 for the life of a `serve`
  session (row A22) — no banners, no progress, no native library chatter.
- **No publish, no release** without explicit user approval. `publish = false` is set in
  `Cargo.toml`; keep it.
- **Comment discipline.** Comments only for a non-obvious constraint the code cannot express (an
  external quirk, a deliberate workaround, a safety invariant) — say why, never what. API doc
  comments on public items are welcome. Tests carry no comments; the test name states the
  behavior.

## Verification

```
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

All three must be green before a task is reported complete. Never weaken a gate to make work look
finished; stop and report instead.

## Layout

`src/lib.rs` is the library target (integration tests link it); `src/main.rs` is only a verb
dispatcher whose handlers call into library modules, so later tasks fill bodies without touching
`main.rs`.

| Module | Owns |
|---|---|
| `protocol` | NDJSON envelope parsing, method dispatch, the four error codes, serve loop |
| `engine_trait` | The engine abstraction that keeps `protocol` pure and model-free in tests |
| `manifest` | Embedded model manifest (id → sha256, size, source URL, serving knobs) + cache paths |
| `health` | `health` result assembly: readiness, dims, capabilities, load policy |
| `prepare` | The `prepare` subcommand: atomic download, sha256 verify, cache lock, disk preflight |
| `sanitize` | Engine-layer input sanitization (see the two-layer rule above) |
| `truncate` | Token-budget truncation of the prefixed string before EOS append |

## Dependency notes

- HTTP: `ureq` (blocking, rustls-backed, no async runtime — the sidecar has none by design).
- File locking: `fs4`. Its `FileExt::lock`/`unlock` are shadowed by `std::fs::File`'s inherent
  methods (stabilized in Rust 1.89), so call them fully qualified: `fs4::FileExt::lock(&file)`.
- `llama-cpp-2` is NOT a dependency yet; Task 5 adds it with exact `=` pins.
