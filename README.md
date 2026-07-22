# julie-semantic-sidecar

A thin Rust stdio binary that turns text into embedding vectors. It speaks the frozen
`julie.embedding.sidecar` v1 protocol — newline-delimited JSON on stdin/stdout, methods
`health`, `embed_query`, `embed_batch`, `shutdown` — and embeds via a pinned llama.cpp with
Metal / Vulkan / CPU backends. Consumers (Miller, Julie) spawn it and talk NDJSON; they never
download models or compute model paths.

**Status: under construction.** The verb skeleton and module layout are in place; the protocol
loop, model manifest, `prepare` subcommand, and engine land in later tasks.

## Launch interface

```
julie-semantic-sidecar [serve [--model <id>]]   # default verb; default model bge-small-en-v1.5-f32
julie-semantic-sidecar prepare [--model <id>]   # download + verify a model into the shared cache
julie-semantic-sidecar --version
```

`serve` reads requests from stdin and writes one response line per request to stdout until EOF.
An unknown verb exits 2 with usage on stderr.
Consumers should always pass an explicit model id so their selected encoder never follows a
standalone-default change.

## Environment

| Variable | Effect |
|---|---|
| `JULIE_EMBEDDING_CACHE_DIR` | Model cache root. Defaults to `~/.cache/julie-semantic` (macOS and Linux), `%LOCALAPPDATA%`-rooted on Windows. Shared with Julie by construction. |
| `JULIE_SIDECAR_FORCE_BACKEND=cpu` | Forces the CPU backend, skipping the device probe and micro-benchmark. A diagnostic and CI surface — never required for normal operation. |

## Build and test

```
cargo build --release
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

The toolchain is pinned in `rust-toolchain.toml`.

## License

MIT — see [LICENSE](LICENSE).
