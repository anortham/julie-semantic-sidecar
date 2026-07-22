# Apple Silicon Metal package proof — 2026-07-22

This record proves the exact Apple arm64 portable archive on physical Apple Silicon. It does not
authorize publication, change the wire contract, or claim support for another hardware lane.

## Bound artifact

| Input | Proven value |
|---|---|
| Source commit | `69abd2d367ee56ac6beb6cebeddf966943c97a29` |
| Package profile | `apple-arm64-metal-portable` |
| Archive | `julie-semantic-sidecar-0.1.0-rc.2-aarch64-apple-darwin-metal-portable.tar.gz` |
| Archive SHA-256 | `8079cb64eb950cc53fcf8138512f9ac616877b05b8ee6e9a6d0b9e67e82de0dc` |
| Sidecar version | `0.1.0-rc.2` |
| Rust target | `aarch64-apple-darwin` |
| Advertised backend | `metal` |
| Default model | `bge-small-en-v1.5-f32` |

The embedded schema-1 manifest declares only `LICENSE`, `README.md`, and the executable, with a
per-file SHA-256 and size. It declares both supported model ids, no model weights, and the exact
native build identity `package_features=metal`. The hardware harness verified the archive checksum,
safe flat extraction, and every manifest digest before executing only the binary unpacked under
macOS's temporary directory.

## Host and idle gate

| Fact | Recorded value |
|---|---|
| Machine | `Mac14,14`, Apple M2 Ultra, 24 logical CPUs, 64 GiB unified memory |
| GPU | Apple M2 Ultra, 60 cores, Metal 4 |
| OS and driver identity | macOS `26.5.2`, build `25F84`; Darwin `25.5.0` |
| Idle preflight | not separately recorded; no support decision depends on absolute throughput |
| Evidence start | `2026-07-22T19:30:53Z` |

The llama.cpp runtime independently enumerated `MTL0 (Apple M2 Ultra)`, used the embedded Metal
library, and assigned all BGE and Qwen layers to that device during forced-Metal execution. No
software-renderer signature was present.

## Protocol and golden conformance

The full ignored conformance target was run once with forced CPU and once with forced Metal. Each run
passed all 9 selected tests: all 23 Group A protocol rows, all 6 Group B lifecycle rows, and both
Group C model tests.

| Backend | Group A/B result | BGE Group C | Qwen Group C | Test result |
|---|---|---:|---:|---|
| CPU | all rows passed | 39 texts + all 250 batch positions | 39 texts + all 250 batch positions | 9 passed, 0 failed in `22.54 s` |
| Metal | all rows passed | 39 texts + all 250 batch positions | 39 texts + all 250 batch positions | 9 passed, 0 failed in `14.41 s` |

CPU cold health arrived in `475 ms` and its 250-text request completed in `1977 ms`. Metal cold health
arrived in `647 ms` and its 250-text request completed in `165 ms`. All protocol stdout was valid NDJSON;
native loader and inference output remained on stderr.

## Selection, fallback, and measurements

The proof cache started without `backend-selection.json`. The first omitted-backend BGE serve rebuilt
the cache with `requested: metal`, `resolved: metal`, `metal: true`, and no degradation reason. A second
serve returned identical ready Metal health and left the cache byte-for-byte unchanged at SHA-256
`02646f09c7f9479d637b61aa5c746bb4fefaa1a69eba385bbafecbf1179dd728`.

| Path | Ready | Requested | Resolved | Accelerated | Degradation |
|---|---:|---|---|---:|---|
| Auto selection and cached reuse | true | Metal | Metal | true | none |
| Forced CPU | true | CPU | CPU | false | none |
| Forced unavailable Vulkan | true | Vulkan | CPU | false | `requested backend is unavailable` |
| Empty cache | false | n/a | n/a | n/a | `model_not_prepared` |

The fixed BGE throughput measurements used one warmup, four measured rounds, the exact unpacked binary,
and explicit CPU or Metal forcing.

| Backend | Batch 1 | Batch 16 | RSS, batch 1 | RSS, batch 16 |
|---|---:|---:|---:|---:|
| CPU | 26.94 texts/s | 233.17 texts/s | 191,692,800 bytes | 194,592,768 bytes |
| Metal | 144.44 texts/s | 685.31 texts/s | 201,654,272 bytes | 204,537,856 bytes |

Metal was 5.36 times faster at batch 1 and 2.94 times faster at batch 16, so resolving the Apple
portable package to Metal is consistent with the observed winner. BGE remains the package default;
Qwen was prepared and proven only as the explicit comparison model.

## Reproduction and raw evidence

The complete evidence tree is outside tracked source at
`/Users/murphy/source/julie-semantic-sidecar-evidence/2026-07-22-69abd2d-metal-review-fixes`. It contains the exact
archive and manifest, checksum, device/runtime identity, selection cache, health transcripts,
prepare records, full CPU/Metal conformance logs, and all four benchmark JSON files.

Build the archive from the source commit, then run:

```bash
scripts/package.sh --profile apple-arm64-metal-portable
scripts/hardware-smoke.sh \
  --archive /Users/murphy/source/julie-semantic-sidecar-evidence/2026-07-22-69abd2d-metal-review-fixes/archive/julie-semantic-sidecar-0.1.0-rc.2-aarch64-apple-darwin-metal-portable.tar.gz \
  --sha256 8079cb64eb950cc53fcf8138512f9ac616877b05b8ee6e9a6d0b9e67e82de0dc \
  --backend metal \
  --lane apple-arm64-metal-portable \
  --cache-dir /Users/murphy/source/julie-semantic-sidecar-evidence/2026-07-22-69abd2d-metal-review-fixes/cache \
  --fixtures /Users/murphy/source/miller/eval/sidecar-conformance \
  --evidence-dir /Users/murphy/source/julie-semantic-sidecar-evidence/2026-07-22-69abd2d-metal-review-fixes
```

Both model files were already present under their exact manifest checksums, so `prepare` validated and
reused them. This proof covers the Claude review fixes for warmup, cache identity and invalidation,
multi-device selection, loader diagnostics, and driver-identity caching.
