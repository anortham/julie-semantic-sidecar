# Miller Takeover Phase 9: macOS x64 Package Report

Date: 2026-07-24

## Outcome

The sidecar now defines `apple-x64-metal-portable` as the fourth portable package profile. It
targets `x86_64-apple-darwin`, compiles the built-in Metal backend, and retains the existing runtime
selection and truthful CPU fallback behavior. The frozen sidecar protocol and backend-selection
implementation did not change.

The new profile is a package candidate, not a supported platform claim. CI and the manual candidate
workflow build it on `macos-15-intel`, but artifact validation remains explicitly separate from
physical Intel-Mac support evidence.

## Worktree

- Path: `/Users/murphy/source/julie-semantic-sidecar/.worktrees/miller-takeover-macos-x64`
- Branch: `codex/miller-takeover-macos-x64`
- Base commit: `24ce6257bee7f41865b10daf1457ed9b4fd71a8a`
- Commit created: none
- Other sidecar worktrees modified: none

## Changes

- Added the Apple x64 Metal portable profile to the manifest validator and both package adapters.
- Preserved all three existing portable profiles and both CUDA vendor profiles.
- Added the profile to artifact-validation CI on `macos-15-intel`.
- Added it to the manual approval-gated candidate choices and both release-candidate matrices.
- Kept the existing exact-checksum binding for the selected hardware lane.
- Documented built-in Metal, truthful CPU fallback, the four-lane portable matrix, and the
  physical-Intel promotion boundary.
- Added tests that bind the profile across package scripts, manifest validation, CI, the candidate
  workflow, README, and promotion documentation.

## TDD Evidence

The tests were added before implementation and failed for the intended missing behavior:

- `apple_metal_is_built_in_and_rejects_a_fake_plugin`: x86_64 Apple Metal was an unsupported
  package profile.
- `packaging_scripts_define_only_the_explicit_portable_and_cuda_candidate_profiles`: both adapters
  lacked `apple-x64-metal-portable`.
- `public_docs_and_promotion_gate_name_every_portable_profile`: README lacked the fourth profile.
- `ci_names_portable_packages_as_artifact_validation_not_support_evidence`: CI had zero x64 profile
  entries.
- `release_is_checksum_bound_approval_gated_and_artifact_only`: the manual candidate workflow had
  zero x64 entries.

All five contracts passed after the implementation.

## Verification

- Baseline `cargo test`: passed before changes.
- `cargo test`: 213 passed, 25 model/hardware-gated tests ignored, 0 failed.
- `cargo test --features metal`: 213 passed, 25 model/hardware-gated tests ignored, 0 failed.
- `cargo clippy --all-targets -- -D warnings`: passed.
- `cargo fmt --check`: passed.
- `cargo test --test package_manifest_tests`: 15 passed.
- Focused CI and release workflow contract tests: passed.
- `actionlint .github/workflows/ci.yml .github/workflows/release.yml`: passed.
- `bash -n scripts/package.sh`: passed.
- `python3 -m py_compile scripts/bench-throughput.py`: passed.
- PowerShell parser/runtime check: not available because `pwsh` is not installed on this host; the
  shared Rust contract tests cover both package adapter profile inventories.
- `apple-arm64-metal-portable` release package built twice with the identical archive SHA-256
  `14eb9886ecaae1b567751dbcfb39ef59e06f373b1b244fe36bc81862a71a8cca`.
- Checksum-bound `hardware-smoke.sh --artifact-validation` passed against that newly unpacked arm64
  archive and reported that it is not support evidence.
- An attempted `apple-x64-metal-portable` build on this `aarch64-apple-darwin` host was rejected
  before compilation with the required-host mismatch. No cross-built result was represented as an
  Intel artifact.

## Claude Review Follow-up

A fresh Claude review found two low-severity weaknesses in the first package contract tests. Both
were accepted and fixed:

- The package adapters now have their declared profiles parsed and compared to the exact six-profile
  candidate inventory, with the expected declaration/mapping count for each adapter. A mutation
  probe injects the obsolete ambiguous `macos-x64` CPU profile and proves both Bash and PowerShell
  inventories reject unexpected additions.
- The CI contract now extracts the named lane from the `artifact-validation` job and compares its
  contiguous matrix entry to the exact x64/Intel/Metal/Bash mapping. A mutation probe changes only
  that lane's runner and proves the structural assertion fails.
- The release contract now requires two exact x64/Intel/Metal/Bash JSON objects, one in each manual
  candidate matrix branch. Its mutation probe changes one branch's runner while preserving the old
  total profile count and proves the paired assertion detects the drift.
- The artifact-validation lane extractor stops at either the next matrix lane or the next top-level
  job; a synthetic future-job fixture guards that boundary.

The focused tests first failed because the new structural extractors were absent, then passed after
the extractors were implemented. The post-review full `cargo test` gate passed 213 tests with
25 hardware/model tests ignored; clippy, formatting, and actionlint also remained green.

## Remaining Promotion Evidence

This source change is newer than `v0.1.0-rc.3`. Adoption requires an RC4 candidate and a public
`x86_64-apple-darwin` archive/checksum produced from the reviewed commit. No workflow was dispatched
and nothing was published, tagged, pushed, or released here.

Before the Apple x64 lane can be called supported, the exact public archive checksum must pass on a
physical Intel Mac:

1. deterministic manifest and unpacked artifact validation;
2. BGE and Qwen protocol/golden-vector conformance;
3. live Metal discovery, selection, and accelerated execution;
4. forced-unavailable Metal with ready CPU fallback and a non-null degradation reason;
5. cold/warm latency, memory, concurrency, determinism, and multi-client checks;
6. a recorded Intel reference machine and approved lane-specific performance floor.

Apple arm64 evidence, a hosted Intel compile runner, or an arm64 cross-build cannot satisfy those
requirements.
