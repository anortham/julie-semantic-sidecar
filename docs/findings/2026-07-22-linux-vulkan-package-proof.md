# Linux x64 Vulkan package proof - 2026-07-22

This record proves one exact Linux x64 portable archive on physical Vulkan hardware. It does not
authorize publication, change the wire contract, or claim support for another platform or checksum.

## Bound artifact

| Input | Proven value |
|---|---|
| Source commit | `56cced9d977a3a76ae31c85f7be1358605380978` |
| Package profile | `linux-x64-vulkan-portable` |
| Archive | `julie-semantic-sidecar-0.1.0-rc.2-x86_64-unknown-linux-gnu-vulkan-portable.tar.gz` |
| Archive SHA-256 | `65dbb3ac82f2bac579762b18c6995c93143bb60072543e6440563a4324b1d6d1` |
| Sidecar version | `0.1.0-rc.2` |
| Rust target | `x86_64-unknown-linux-gnu` |
| Advertised backend | `vulkan` |
| Portability tier | `portable` |

The schema-1 manifest declares 30 files: the executable, license, README, core llama/ggml shared
libraries, 14 CPU variants, and the Vulkan backend. It contains no model weights. The native build
identity is `package_features=vulkan,dynamic-backends` with baseline x64 Rust target features and no
native CPU flag. The archive validator checked the flat inventory, every size and digest, the Linux
`$ORIGIN` runpath, absent-model behavior, stdout purity, and clean shutdown from a fresh extraction.

## Host and device gate

| Fact | Recorded value |
|---|---|
| OS | Fedora 44, kernel `7.1.4-202.fc44.x86_64` |
| CPU architecture | x86_64 |
| Discrete GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB |
| NVIDIA driver | `610.43.03` |
| Integrated GPU | Intel Iris Xe Graphics, Mesa `26.1.4` |
| Vulkan instance | `1.4.341` |
| Evidence start | `2026-07-22T21:02:31Z` |

The host also exposed llvmpipe. The hardware gate required at least one physical integrated or
discrete Vulkan device and separately inspected llama.cpp's selected-device diagnostics, so the
software ICD could coexist without being mistaken for proof. Runtime selection chose `Vulkan1`
(the NVIDIA GPU), assigned all 13 BGE layers to it, and did not select llvmpipe.

## Protocol and golden conformance

The full ignored conformance target ran against the extracted archive once with forced CPU and once
with forced Vulkan. Both runs passed all 9 selected tests: every Group A protocol row, every Group B
lifecycle row, and both Group C model tests.

| Backend | Group A/B | BGE Group C | Qwen Group C | Result |
|---|---|---:|---:|---|
| CPU | all rows passed | 39 texts + all 250 positions | 39 texts + all 250 positions | 9 passed, 0 failed in `92.23 s` |
| Vulkan | all rows passed | 39 texts + all 250 positions | 39 texts + all 250 positions | 9 passed, 0 failed in `25.76 s` |

CPU cold health arrived in `123 ms` and its 250-text BGE request completed in `2292 ms`. Vulkan
cold health arrived in `1105 ms` and its request completed in `290 ms`. Qwen's Vulkan 250-position
probe completed in `7244 ms`. The same Qwen test also passed three consecutive isolated archive runs
after the causal group limit was fixed, with probes between `7139 ms` and `7276 ms`.

## Selection, fallback, and measurements

The first omitted-backend serve rebuilt the selection cache with `requested: vulkan`,
`resolved: vulkan`, device index 2, and no degradation. A second serve reused the same selection and
left the cache byte-for-byte unchanged at SHA-256
`d9b74d8eea9d6604611fde01e9fb2b0b877429658d9c8d3e5ce2bcb7c55eec2a`.

| Path | Ready | Resolved | Accelerated | Degradation |
|---|---:|---|---:|---|
| Auto selection and reuse | true | Vulkan | true | none |
| Forced CPU | true | CPU | false | none |
| Forced unavailable Metal | true | CPU | false | `requested backend is unavailable` |
| Empty cache | false | n/a | n/a | `model_not_prepared` |

| Backend | Batch 1 | Batch 16 | RSS, batch 1 | RSS, batch 16 |
|---|---:|---:|---:|---:|
| CPU | 44.63 texts/s | 55.11 texts/s | 162,357,248 bytes | 165,543,936 bytes |
| Vulkan | 92.06 texts/s | 303.18 texts/s | 222,220,288 bytes | 222,633,984 bytes |

Vulkan was 2.06 times faster at batch 1 and 5.50 times faster at batch 16. This is consistent with
the selected portable backend and clears the Linux archive's correctness and fallback gates.

## Reproduction and raw evidence

The ignored local evidence tree is `hardware-evidence/linux-x64-vulkan-fedora44-final-65dbb3ac82f2`.
It contains the checksum, manifest, device and runtime identity, selection cache, protocol
transcripts, prepare logs, CPU/Vulkan conformance logs, and benchmark JSON files.

```bash
scripts/package.sh --profile linux-x64-vulkan-portable
scripts/hardware-smoke.sh \
  --archive dist/julie-semantic-sidecar-0.1.0-rc.2-x86_64-unknown-linux-gnu-vulkan-portable.tar.gz \
  --sha256 65dbb3ac82f2bac579762b18c6995c93143bb60072543e6440563a4324b1d6d1 \
  --backend vulkan \
  --lane linux-x64-vulkan-fedora44-final \
  --cache-dir /tmp/julie-semantic-linux-cache \
  --fixtures /tmp/julie-miller-8edfa14/eval/sidecar-conformance \
  --evidence-dir hardware-evidence/linux-x64-vulkan-fedora44-final-65dbb3ac82f2
```

This run found and fixed Fedora `lib64` packaging, mixed-ICD hardware validation, transient prepare
handling, and Qwen Vulkan context allocation. The final mandatory Rust gates and the
`vulkan,dynamic-backends` feature gates all passed.
