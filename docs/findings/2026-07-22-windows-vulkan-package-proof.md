# Windows x64 Vulkan package proof - 2026-07-22

This record proves one exact Windows x64 portable archive on physical Vulkan hardware. It does not
authorize publication, change the wire contract, or claim support for another platform or checksum.

## Bound artifact

| Input | Proven value |
|---|---|
| Binary source commit | `73c79e2b1952e059c64f7c435192e840fc09ecbe` |
| Package profile | `windows-x64-vulkan-portable` |
| Archive | `julie-semantic-sidecar-0.1.0-rc.2-x86_64-pc-windows-msvc-vulkan-portable.zip` |
| Archive SHA-256 | `ab19495575cf0e58cd80af3233a1847b0d811a1662f1df0e35fded470923c654` |
| Sidecar version | `0.1.0-rc.2` |
| Rust target | `x86_64-pc-windows-msvc` |
| Advertised backend | `vulkan` |
| Portability tier | `portable` |

The schema-1 manifest declares 17 files: the executable, license, README, four core llama/ggml
runtime DLLs, nine CPU variants, and the Vulkan backend. It contains no model weights. The native
build identity is `package_features=vulkan,dynamic-backends` with baseline x64 Rust target features
and no native CPU flag. The archive validator checked the flat inventory, every size and digest,
absent-model behavior, stdout purity, and clean shutdown from a fresh extraction without changing
`PATH` for the extracted executable.

## Host and device gate

| Fact | Recorded value |
|---|---|
| OS | Microsoft Windows NT `10.0.26200.0`, x64 |
| CPU | 12th Gen Intel Core i7-12700H, 14 cores / 20 logical processors |
| Discrete GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB |
| NVIDIA driver | `576.83` |
| Integrated GPU | Intel Iris Xe Graphics, driver `101.6881` |
| Vulkan instance | `1.4.304` |
| Evidence start | `2026-07-22T22:33:15Z` |

`vulkaninfo --summary` exposed only the physical Intel and NVIDIA devices and no software renderer.
The first-start benchmark exercised CPU, `Vulkan0` on the Intel GPU, and `Vulkan1` on the NVIDIA
GPU. Runtime selection chose `Vulkan1`, recorded device index 2, and assigned all 13 BGE layers to
the RTX 3060.

## Protocol and golden conformance

The full ignored conformance target ran against the extracted archive once with forced CPU and once
with forced Vulkan. Both runs passed all 9 selected tests: every Group A protocol row, every Group B
lifecycle row, and both Group C model tests.

| Backend | Group A/B | BGE Group C | Qwen Group C | Result |
|---|---|---:|---:|---|
| CPU | all rows passed | 39 texts + all 250 positions | 39 texts + all 250 positions | 9 passed, 0 failed in `99.82 s` |
| Vulkan | all rows passed | 39 texts + all 250 positions | 39 texts + all 250 positions | 9 passed, 0 failed in `84.51 s` |

CPU cold health arrived in `178 ms` and its Group B 250-text request completed in `2623 ms`.
Vulkan cold health arrived in `1513 ms` and its request completed in `3727 ms`. Both remain well
inside the frozen `120000 ms` and `30000 ms` budgets. The complete Qwen Group C probe fell from
`88072 ms` on CPU to `29387 ms` on Vulkan.

## Selection, fallback, and measurements

The first omitted-backend serve rebuilt the selection cache with `requested: vulkan`,
`resolved: vulkan`, device index 2, and no degradation. A second serve reused the same selection and
left the cache byte-for-byte unchanged at SHA-256
`7031ab0f6b15e186f3dff658f369e1fbaec6916b8e889f8b1fb5c59b661dbff1`.

| Path | Ready | Resolved | Accelerated | Degradation |
|---|---:|---|---:|---|
| Auto selection and reuse | true | Vulkan | true | none |
| Forced CPU | true | CPU | false | none |
| Forced unavailable Metal | true | CPU | false | `requested backend is unavailable` |
| Empty cache | false | n/a | n/a | `model_not_prepared` |

| Backend | Batch 1 | Batch 16 | RSS, batch 1 | RSS, batch 16 |
|---|---:|---:|---:|---:|
| CPU | 47.38 texts/s | 45.63 texts/s | unavailable | unavailable |
| Vulkan | 63.54 texts/s | 384.35 texts/s | unavailable | unavailable |

Vulkan was 1.34 times faster at batch 1 and 8.42 times faster at batch 16. Windows process RSS was
not available to the cross-platform benchmark sampler and was recorded as `null`, not zero. The
selected portable backend clears the archive's correctness, fallback, and fixed-throughput gates.

## Reproduction and raw evidence

The ignored local evidence tree is
`hardware-evidence/windows-x64-vulkan-rtx3060-ab19495575cf`. It contains the checksum, manifest,
device and runtime identity, selection cache, protocol transcripts, prepare logs, CPU/Vulkan
conformance logs, and benchmark JSON files. Miller fixtures came from exact commit
`8edfa14ffb4f0f696d6c4c4aaa572d5967495961`.

```powershell
$env:CARGO_TARGET_DIR = 'C:\n'
$env:CMAKE_GENERATOR = 'Ninja'
scripts/package.ps1 -Profile windows-x64-vulkan-portable
scripts/hardware-smoke.ps1 `
  -Archive dist\julie-semantic-sidecar-0.1.0-rc.2-x86_64-pc-windows-msvc-vulkan-portable.zip `
  -Sha256 ab19495575cf0e58cd80af3233a1847b0d811a1662f1df0e35fded470923c654 `
  -Backend vulkan `
  -Lane windows-x64-vulkan-rtx3060 `
  -CacheDir hardware-evidence\cache-windows-vulkan-ab19495575cf `
  -FixturesDir $env:TEMP\julie-miller-8edfa14\eval\sidecar-conformance `
  -EvidenceDir hardware-evidence\windows-x64-vulkan-rtx3060-ab19495575cf
```

This run found and fixed PowerShell's reserved `$IsWindows` collision, Cargo target-directory
resolution for short Windows native-build paths, and the missing `bin` scan for installed core
runtime DLLs. The package was built with MSVC through Ninja because the Visual Studio generator's
nested Vulkan shader project exceeded the host's 260-character path limit. The final mandatory
Rust gates and the `vulkan,dynamic-backends` feature test gate all passed.
