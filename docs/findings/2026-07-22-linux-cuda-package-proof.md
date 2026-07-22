# Linux x64 CUDA package proof - 2026-07-22

This record proves one exact host-specific CUDA candidate on physical NVIDIA hardware. It does not
authorize publication or establish a universal CUDA vendor artifact.

## Bound artifact

| Input | Proven value |
|---|---|
| Source commit | `56cced9d977a3a76ae31c85f7be1358605380978` |
| Package profile | `linux-x64-cuda-vendor` |
| Archive | `julie-semantic-sidecar-0.1.0-rc.2-x86_64-unknown-linux-gnu-cuda-vendor.tar.gz` |
| Archive SHA-256 | `0804aed95d86e73ecbf3377491c1fce88e20e922e6db76c15efb4dffa5739522` |
| Sidecar version | `0.1.0-rc.2` |
| Rust target | `x86_64-unknown-linux-gnu` |
| Advertised backend | `cuda` |
| Portability tier | `vendor` |
| CUDA architecture | `sm_86` only |

The schema-1 manifest declares 30 files: the executable, license, README, core llama/ggml shared
libraries, 14 CPU variants, and the CUDA backend. It contains no model weights. Its native build
identity is `package_features=cuda,dynamic-backends` with baseline x64 Rust target features and no
native CPU flag. The hardware-specific architecture was supplied through `CUDAARCHS=86`; the current
manifest schema does not encode that restriction, so this checksum must not be treated as a
universal CUDA release candidate.

## Host and toolchain

| Fact | Recorded value |
|---|---|
| OS | Fedora 44, kernel `7.1.4-202.fc44.x86_64` |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB, compute capability 8.6 |
| NVIDIA driver | `610.43.03`, CUDA UMD `13.3` |
| CUDA toolkit | `13.2.51` |
| CUDA host compiler | GCC `15.2.1` compatibility compiler |
| Evidence start | `2026-07-22T21:32:44Z` |

CUDA 13.2 supports GCC through major version 15, while Fedora 44 ships GCC 16. The toolkit was
installed without its bundled driver under `~/.local/cuda-13.2`, and nvcc used Fedora's GCC 15
compatibility compiler. A standalone CUDA kernel compiled and ran before the sidecar build. The
existing 610 driver remained installed.

## Protocol and golden conformance

The full ignored conformance target ran against the extracted archive once with forced CPU and once
with forced CUDA. Both runs passed all 9 selected tests.

| Backend | Group A/B | BGE Group C | Qwen Group C | Result |
|---|---|---:|---:|---|
| CPU | all rows passed | 39 texts + all 250 positions | 39 texts + all 250 positions | 9 passed, 0 failed in `93.79 s` |
| CUDA | all rows passed | 39 texts + all 250 positions | 39 texts + all 250 positions | 9 passed, 0 failed in `17.95 s` |

CPU cold health arrived in `123 ms` and its 250-text BGE request completed in `2104 ms`. CUDA cold
health arrived in `662 ms` and its request completed in `622 ms`. Qwen's CUDA 250-position probe
completed in `5315 ms`. CUDA selected `CUDA0` on the RTX 3060 and offloaded all 13 model layers.

## Selection, fallback, and measurements

The first omitted-backend serve rebuilt the cache with `requested: cuda`, `resolved: cuda`, device
index 1, and no degradation. Reuse left it unchanged at SHA-256
`8c4512d50c486fecf1541b6eac00def9eac0b6ac094ed03c79685b1586f95139`.

| Path | Ready | Resolved | Accelerated | Degradation |
|---|---:|---|---:|---|
| Auto selection and reuse | true | CUDA | true | none |
| Forced CPU | true | CPU | false | none |
| Forced unavailable Metal | true | CPU | false | `requested backend is unavailable` |
| Empty cache | false | n/a | n/a | `model_not_prepared` |

| Backend | Batch 1 | Batch 16 | RSS, batch 1 | RSS, batch 16 |
|---|---:|---:|---:|---:|
| CPU | 50.70 texts/s | 56.73 texts/s | 162,177,024 bytes | 165,625,856 bytes |
| CUDA | 115.63 texts/s | 173.24 texts/s | 527,400,960 bytes | 528,863,232 bytes |

CUDA was 2.28 times faster than CPU at batch 1 and 3.05 times faster at batch 16. Against the same
host's portable Vulkan archive, CUDA was 1.26 times faster at batch 1 but only 0.57 times as fast at
batch 16 (`173.24` versus `303.18` texts/s). This mixed result does not establish the material,
workload-wide benefit required to promote an optional vendor lane over the portable archive.

## Reproduction and disposition

The ignored local evidence tree is
`hardware-evidence/linux-x64-cuda-vendor-sm86-fedora44-0804aed95d86`. It contains the checksum,
manifest, GPU and driver identity, selection cache, protocol transcripts, prepare logs, CPU/CUDA
conformance logs, and benchmark JSON files.

```bash
export CUDAToolkit_ROOT="$HOME/.local/cuda-13.2"
export CUDA_PATH="$CUDAToolkit_ROOT"
export NVCC_CCBIN="$HOME/.local/gcc15/usr/bin/g++-15"
export PATH="$CUDAToolkit_ROOT/bin:$PATH"
export LD_LIBRARY_PATH="$CUDAToolkit_ROOT/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDAARCHS=86

scripts/package.sh --profile linux-x64-cuda-vendor
scripts/hardware-smoke.sh \
  --archive dist/julie-semantic-sidecar-0.1.0-rc.2-x86_64-unknown-linux-gnu-cuda-vendor.tar.gz \
  --sha256 0804aed95d86e73ecbf3377491c1fce88e20e922e6db76c15efb4dffa5739522 \
  --backend cuda \
  --lane linux-x64-cuda-vendor-sm86-fedora44 \
  --cache-dir /tmp/julie-semantic-linux-cache \
  --fixtures /tmp/julie-miller-8edfa14/eval/sidecar-conformance \
  --evidence-dir hardware-evidence/linux-x64-cuda-vendor-sm86-fedora44-0804aed95d86
```

The `cuda,dynamic-backends` test and clippy gates passed. This checksum is valid real-device evidence
for `sm_86`, but CUDA remains a candidate pending a universal package proof and a clear advantage
over the Linux Vulkan portable lane.
