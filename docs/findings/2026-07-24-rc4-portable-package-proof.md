# RC4 portable package proof — 2026-07-24

## Verdict

The exact public `v0.1.0-rc.4` portable archives are reproducible and internally valid. Two clean
GitHub Actions runs built byte-identical Apple arm64 Metal, Apple x64 Metal, Linux x64 Vulkan, and
Windows x64 Vulkan archives from commit `7cab71190b5d5b4424747ffff6eafcb49a94aec0`.

The public Apple arm64 archive additionally passes physical Apple Silicon CPU and Metal proof.
Hosted artifact validation for the other three archives proves build, package, manifest, and
unpacked behavior only; it is not physical-hardware support evidence.

## Reproducibility

- First retained run: `30104080099`
- Checksum-bound green run: `30104924982`
- Native patch identity:
  `llama-cpp-sys-2-0.1.151:vulkan-infinity-v3:65f336fc251be7b4d5d929eef0c1824cd252a992896309e45cbf24c272d83a73`

| Archive | SHA-256 |
|---|---|
| `julie-semantic-sidecar-0.1.0-rc.4-aarch64-apple-darwin-metal-portable.tar.gz` | `8b6169894c4f72f78c64dcc1eb6a0ec98f022b413bbe16d05bd2a30781620d54` |
| `julie-semantic-sidecar-0.1.0-rc.4-x86_64-apple-darwin-metal-portable.tar.gz` | `d48294dde655f1878caa7a716db2a4aefdeb0df7816be8525c20073d65e88552` |
| `julie-semantic-sidecar-0.1.0-rc.4-x86_64-unknown-linux-gnu-vulkan-portable.tar.gz` | `10dfd99d0a0847813482145b4f78fc7601a77a4a136a6cb4e72dd729945e38d1` |
| `julie-semantic-sidecar-0.1.0-rc.4-x86_64-pc-windows-msvc-vulkan-portable.zip` | `3ddcbb981293eef8f3b4446c74cc94ab611113256d6654dd65397962d0f0dce3` |

Each archive matched its generated checksum sidecar, matched the same-named archive from the other
run with `cmp`, and matched the corresponding public GitHub asset after a fresh release download.
All four manifests report sidecar version `0.1.0-rc.4`, the expected target/backend, and the same
content-derived native patch digest.

## Reproducibility incident

The initial Windows package alternated one SPIR-V `OpConstant` between positive infinity and zero.
The pinned llama.cpp shaders used three division-by-zero infinity expressions. RC4 packages from a
deterministic Cargo-vendored dependency tree, replaces those expressions with exact IEEE-754 bit
values, and records the patched-source digest in both binaries and the package manifest.

The first cross-platform attempt also exposed two clean-build defects before release:

- GLSL forbids `uintBitsToFloat` in a global constant initializer. The top-k shader now expands
  exact negative infinity at its six function-local use sites.
- The pinned native build script followed dangling SONAME symlinks when testing hard-link
  destinations. Its three guards now use no-follow symlink metadata.

Both corrections are strict pinned-source patches. Unexpected upstream bytes or occurrence counts
fail before Cargo builds.

## Apple arm64 physical proof

- Public archive SHA-256:
  `8b6169894c4f72f78c64dcc1eb6a0ec98f022b413bbe16d05bd2a30781620d54`
- Executable SHA-256:
  `b026a25f7693ff9b17097c358dcf085c37c25bc2c17196cd9aece76982670345`
- Host: Apple M2 Ultra, macOS 26.5.2
- CPU conformance: 9 passed, 0 failed
- Metal conformance: 9 passed, 0 failed
- Automatic selection: Metal on Apple M2 Ultra
- Forced CPU: ready with `accelerated=false`
- Forced unavailable Vulkan: ready CPU fallback with a degradation reason
- Batch-64 Metal promotion floor: 668.25 texts/s against a 40 texts/s floor

| Backend | Batch | Texts/s | RSS MiB |
|---|---:|---:|---:|
| CPU | 1 | 29.87 | 182.89 |
| CPU | 16 | 246.37 | 183.84 |
| Metal | 1 | 146.23 | 190.33 |
| Metal | 16 | 669.06 | 190.47 |

## Concurrency

The public binary passed separate forced-CPU and automatic-Metal probes. Each probe started three
independent processes and sent eight pipelined embedding requests to each process.

| Backend | Processes | Requests | Wall ms | Common overlap ms |
|---|---:|---:|---:|---:|
| CPU | 3 | 24 | 10167.13 | 10085.88 |
| Metal | 3 | 24 | 654.11 | 592.07 |

Both probes returned bit-exact vectors and identical health across processes, selected the expected
backend, overlapped all three live pipelines, and exited cleanly.

## Evidence boundary

Raw evidence is outside tracked source at:

`/Users/murphy/source/julie-semantic-sidecar-evidence/2026-07-24-rc4-public-metal`

The evidence binds the public archive and executable hashes, GitHub release assets, package
manifest, device/runtime identity, model pins, selection cache, protocol transcripts, CPU/Metal
conformance, throughput records, concurrency records, harness hashes, and Miller fixture tree
`e29e0c1fae78758545334c9857efdbb2b0ace714`.

Apple x64, Linux Vulkan, and Windows Vulkan remain package candidates until these exact public
archives pass the corresponding physical-hardware gates. CUDA, ROCm, DirectML, MPS, and Intel Arc
are not promoted by this proof.
