# Vulkan zero-constant reproducibility incident — 2026-07-28

## Verdict

RC5 candidate run `30354513975` and retry `30355501461` built from commit
`312e261457e6f2eafcd67ad1949766bb2853bb5d` with identical runner, Rust, MSVC, and Vulkan SDK
versions. Three platform archive pairs were byte-identical. The Windows archives differed because
two generated SPIR-V modules embedded unstable source constants in `ggml-vulkan.dll`.

The RC5 candidate is blocked until a strict pinned-source correction and a package-time bytecode
guard pass two fresh four-platform runs with byte-identical archives.

## Evidence

| Artifact | First run | Second run |
|---|---|---|
| Windows archive SHA-256 | `fb15029827a98928f3cf70bb9f012f9c3f1d984c543bc8eb776db34fddc1376b` | `f910c84c87d794d52b489cbf4559eda4b482b15a3d210e71588e32853b376697` |
| `ggml-vulkan.dll` SHA-256 | `5272e9dd4cd982660a22afde6f3e44d132b7681c2fd76dd2680b7fdd18bd6824` | `d05b048a5bbb9fdbf822fb4e39d799e38b4f04dada4c14abb3061b8ac388167f` |
| `diag_f16` source word | `0x00000000` | `0x7f800000` |
| `tri_f16` source word | `0x00000000` | `0x7b2b93a8` |

The DLLs differed by 54 bytes: the two SPIR-V `OpConstant` words plus PE fields derived by
`/Brepro` from the changed payload. Mapping the embedded module bytes to the Linux ELF symbols
identified the shaders as `diag_f16_data` and `tri_f16_data`. Each unstable constant is consumed by
`OpFConvert` to a 16-bit float and then by `OpStore`.

The pinned shader sources use `D_TYPE=float16_t` and spell the intended else-path value as
`D_TYPE(0)`. Replacing it with `D_TYPE(0.0f)` makes the intermediate floating-point zero explicit
before conversion. The existing v3 patch corrected three division-by-zero infinity expressions but
did not cover these zero expressions.

## RC4 erratum

The public RC4 `ggml-vulkan.dll` has SHA-256
`5272e9dd4cd982660a22afde6f3e44d132b7681c2fd76dd2680b7fdd18bd6824`, byte-identical to the first
RC5 candidate. RC4's two retained runs therefore sampled the same zero variant. Their matching
Windows archives prove identity for those two samples, not deterministic generation. RC4's
four-platform reproducibility claim is withdrawn; its three non-Windows archive pairs remain valid.

## Correction and permanent gate

- The strict native patch replaces exactly one `D_TYPE(0)` expression in each pinned `diag.comp`
  and `tri.comp` source with `D_TYPE(0.0f)`.
- Unexpected bytes, already-patched source, missing expressions, or duplicate expressions fail
  before Cargo builds.
- The content-derived patch identity advances from `vulkan-repro-v3` to `vulkan-repro-v4`.
- Every Vulkan package build parses the generated `diag_f16.spv` and `tri_f16.spv` modules.
- Each module must contain exactly one constant-derived half-float store and its source value must
  be zero. Missing, duplicate, malformed, infinity, or other nonzero values fail before archiving.
- Publication requires two fresh green four-platform runs from one commit with byte-identical
  archives in all four lanes.

Synthetic tests cover direct half zero, 32-bit zero converted to half, infinity, arbitrary nonzero
words, missing modules, and duplicate modules. The verifier was also exercised against the
extracted divergent CI modules: it accepted the first run and rejected the second at
`diag_f16.spv` with `0x7f800000`.

## Closure

Commit `13fff87bcaa9cc93feac465141756f4fc36183f5` implemented the correction and guard. Fresh runs
`30360072357` and `30360964021` produced byte-identical Apple arm64, Apple x64, Linux Vulkan, and
Windows Vulkan archives. Both Vulkan package logs recorded successful zero-store verification, and
the checksum-bound second run was fully green.

The exact public `v0.1.0-rc.5` assets match the retained second-run bytes and their checksum
sidecars. RC5 therefore closes this reproducibility defect; RC4's withdrawn Windows claim remains
withdrawn.
