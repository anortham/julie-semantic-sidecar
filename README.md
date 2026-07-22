# julie-semantic-sidecar

A thin Rust stdio binary that turns text into embedding vectors. It speaks the frozen
`julie.embedding.sidecar` v1 protocol — newline-delimited JSON on stdin/stdout, methods
`health`, `embed_query`, `embed_batch`, `shutdown` — and embeds via a pinned llama.cpp with
Metal / Vulkan / CPU backends. Consumers (Miller, Julie) spawn it and talk NDJSON; they never
download models or compute model paths.

**Status: release candidate under active validation.** CPU and Metal run locally; Vulkan and CUDA
packages remain compile candidates until the platform and real-hardware gates pass.

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
| `JULIE_SIDECAR_FORCE_BACKEND=cpu\|metal\|vulkan\|cuda` | CPU skips discovery and benchmarking. Accelerator values probe only that backend, but still resolve to ready CPU when unavailable, failing, tied, or slower. A diagnostic and CI surface — never required for normal operation. |

## Build and test

```
cargo build --release
cargo test
cargo clippy --all-targets -- -D warnings
cargo fmt --check
```

The toolchain is pinned in `rust-toolchain.toml`.

## Package profiles

Packages use an explicit profile and never bundle model weights:

```text
apple-arm64-metal-portable
linux-x64-vulkan-portable
windows-x64-vulkan-portable
linux-x64-cuda-vendor
windows-x64-cuda-vendor
```

Metal is built into the Apple arm64 executable. Windows and Linux profiles place llama.cpp core
libraries, CPU modules, and the advertised Vulkan or CUDA module flat beside the executable.
CUDA archives are candidates, not supported releases, until real NVIDIA hardware validation passes.

Build with `scripts/package.sh --profile <name>` or
`scripts/package.ps1 -Profile <name>`. Archive names include sidecar version, Rust target, backend,
and portable/vendor tier. Both scripts reject `-Ctarget-cpu=native`, create and verify the same
`package-manifest.json`, and contain no publication step.

The manifest records every payload file with its SHA-256, size, and role. The manifest itself is
the sole metadata exception because a file cannot truthfully contain its own SHA-256. Verification
rejects every other undeclared file, nested path, model weight, or backend/profile mismatch.

## License

MIT — see [LICENSE](LICENSE).
