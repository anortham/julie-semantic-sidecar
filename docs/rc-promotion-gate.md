# RC -> v0.1.0 promotion gate

The checklist a release candidate must pass before it is promoted from a prerelease
(`0.1.0-rc.N`) to a non-prerelease (`0.1.0`). **Promotion itself is a user decision** — this
gate produces the evidence; it does not authorize the release. Run every item **on the target
reference machine** (M2 Ultra) against the exact binary under consideration before proposing
promotion.

## Gate items

1. **Protocol conformance suite** — `scripts/conformance.sh` passes (groups A, B, C of
   `semantic-sidecar-protocol-v1.md` § Conformance). No changes to its pass rule.
2. **Unit tests, both feature sets** — `cargo test` and `cargo test --features metal` both green.
   No changes to what they assert.
3. **Packaged smoke** — the release archive built by `scripts/package.sh` unpacks and its
   bundled binary answers `--version` and a `health` probe with `ready: true`. No changes to its
   checks.
4. **Throughput floor (this document)** — the packaged binary sustains
   **≥ 40 units/s steady-state on the M2 Ultra reference machine (64-text batches, warm model)**,
   measured by `scripts/bench-throughput.py`.

Items 1–3 are defined by their own scripts and are referenced, not restated, here. Item 4 is the
addition.

## Throughput floor — the check

One command, run on the target machine against the binary being promoted:

```
scripts/bench-throughput.py --binary target/release/julie-semantic-sidecar
```

It probes `health` and **fails the bench** unless the sidecar reports `ready: true` — a
`model_not_prepared` binary can never pass, so the gate cannot be satisfied by measuring zeros.
It then times `embed_batch` rounds after a discarded warm-up round and prints steady-state
units/s with a PASS/FAIL verdict against the floor (default `40`, overridable with `--floor`).
Exit code: `0` PASS, `1` below floor, `2` not-ready / bad arguments / protocol error. Use
`--json` for machine-readable output; `--batch`/`--rounds` to vary the shape (batch is capped at
the protocol maximum of 250).

Record the measured steady-state number in the promotion evidence, not just the PASS.

## The floor: 40 units/s

| Measurement | units/s | Machine |
|---|---|---|
| rc.2 steady-state, 64-text batches | 78.9 | M2 Ultra |
| rc.2 steady-state, 250-text batches | 77.4 | M2 Ultra |
| P0 llama-server reference floor | 52.3 | M2 Ultra |
| **Gate floor** | **40** | M2 Ultra |
| CPU-only backend regression | ~6.6 | M2 Ultra |

**40 is roughly half of rc.2's observed rate.** It sits well below the healthy Metal-backed
range (77–89 units/s in repeat runs) and the P0 llama-server reference (52.3), so machine noise
and normal run-to-run variance never trip it. It sits far **above** a CPU-only regression (~6.6,
about 12× under the floor), so a backend that silently falls back to CPU — the exact failure this
gate exists to catch — fails loudly.

## Why this gate exists (the rc.1 lesson)

**Harness numbers are not engine numbers.** A CPU-only RC shipped at roughly **12× under the
design throughput floor** because the throughput that had been validated was a benchmark
harness's, not the shipping engine's on the real artifact. The full record is in Miller's
`docs/findings/2026-07-20-first-real-shadow-converge-benchmark.md`.

The correction is this gate: the floor is measured on the **target machine**, against the
**packaged binary**, over the **real embedding path** (`health` + `embed_batch` over stdio) — not
a harness, not a synthetic microbenchmark, not a different backend. A binary that cannot clear 40
units/s on the reference machine does not get promoted, regardless of what any other benchmark
reported.
