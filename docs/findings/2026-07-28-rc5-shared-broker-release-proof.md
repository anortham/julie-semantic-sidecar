# RC5 shared-broker release proof — 2026-07-28

## Verdict

`v0.1.0-rc.5` was published from commit
`13fff87bcaa9cc93feac465141756f4fc36183f5` as a prerelease with four portable archives and four
matching SHA-256 sidecars.

The archives are reproducible across two independent hosted builds. Both Vulkan lanes passed the
generated-SPIR-V zero-store gate in both runs. Fresh public downloads match the retained candidate
bytes, checksum sidecars, and package manifests.

## Reproducibility

- First retained run: `30360072357`
- Checksum-bound green run: `30360964021`
- Native patch identity:
  `llama-cpp-sys-2-0.1.151:vulkan-repro-v4:4189b1a37fb3e7eece8d3b3785e62c0c7221b3132b9d9cf995e4320805fb2aca`

| Archive | SHA-256 |
|---|---|
| `julie-semantic-sidecar-0.1.0-rc.5-aarch64-apple-darwin-metal-portable.tar.gz` | `4c62e729124ba30640a0b3a8c0f8a4d9f5b8cc4e02a6de640b5baa9039ff2ddc` |
| `julie-semantic-sidecar-0.1.0-rc.5-x86_64-apple-darwin-metal-portable.tar.gz` | `959ab0e1869f0eeb68f237f1ca1266f0440f33a1b42ebbfe370d2e3fb8be8a6e` |
| `julie-semantic-sidecar-0.1.0-rc.5-x86_64-unknown-linux-gnu-vulkan-portable.tar.gz` | `a2f0bcd0135cc056465d12353572462f877e9ae7ca5a988a0012de1038a4a36f` |
| `julie-semantic-sidecar-0.1.0-rc.5-x86_64-pc-windows-msvc-vulkan-portable.zip` | `47f9b1bcc149c781d6d95d74e3e0207142d3f587210872758e5b208fef3b091a` |

All four archive pairs compare byte-for-byte. The first run's Apple arm64 package job built and
uploaded successfully but its selected-checksum step used a local-SDK hash and failed against the
hosted-SDK bytes. The second run selected the exact hosted hash and completed fully green. This
calibration mismatch did not affect the retained archive pair.

Both runs' Linux and Windows package logs contain:

`verified Vulkan half-float zero stores: diag_f16.spv, tri_f16.spv`

## Public verification

The GitHub release is a non-draft prerelease targeting the exact source commit. A fresh release
download returned exactly eight assets.

- Every archive matches its public checksum sidecar.
- Every public archive is byte-identical to its retained run-2 candidate.
- Every unpacked `package-manifest.json` verifies, reports version `0.1.0-rc.5`, and declares the
  expected target, backend, portability tier, and v4 native patch identity.
- The public Apple arm64 archive passes artifact validation.
- Its freshly unpacked broker binds a user-local Unix socket, answers a frozen-protocol health
  request, exits on owner stdin EOF, writes nothing to stdout, and removes the endpoint.

## Support boundary

This is package and lifecycle evidence, not real-device accelerator support evidence. Every RC5
archive has a new checksum and inherits no RC4 hardware proof. Apple arm64, Apple x64, Linux Vulkan,
and Windows Vulkan remain package candidates until these exact public archives pass the applicable
physical-hardware gates. CUDA is not included in RC5.
