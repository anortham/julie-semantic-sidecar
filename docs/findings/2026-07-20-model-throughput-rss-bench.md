# Model throughput + RSS bench — v0.1.0-rc.2, M2 Ultra (2026-07-20)

Measured with `scripts/bench-throughput.py` (64-text batches, 4 rounds after one discarded
warm-up, warm model, Metal backend), immediately after adding the `--model` passthrough and the
post-round RSS sample. Evidence for Miller's P4 model-footprint decision — recorded here, decided
there (`docs/findings/2026-07-20-q8-footprint-benchmark.md` in Miller).

| Model (manifest id) | Dims served | Steady units/s | Sidecar RSS after rounds | Weights on disk |
|---|---|---|---|---|
| `qwen3-0.6b-f16` (default tier) | 512 | **82.9** (80.5–86.1) | **1.27 GiB** | 1.12 GiB |
| `bge-small-en-v1.5-f32` (fallback tier) | 384 | **743.7** (738.0–747.9) | **196 MiB** | 127 MiB |

Both runs: `ready:true`, `accelerated:true`, `resolved_backend=metal`, sidecar 0.1.0-rc.2,
PASS at the 40 units/s gate floor (the floor is defined against the default model; bge's number is
informational).

## Notes

- The manifest has **no Q8_0 pin** for Qwen3 — f16 is the only Qwen3 lane. Adding one means: a new
  `ModelPin` (id/file/url/sha256/size), fresh conformance goldens for that lane, and a re-run of
  the P0 eval gate on the Q8_0 weights (the f16 pin is what P0 scored). Until then Q8_0 cannot be
  benchmarked through this sidecar.
- RSS is a single post-round `ps -o rss=` sample of the live process — steady-state serving
  memory, not a peak trace.
- bge-small at 9.0× the throughput and 0.15× the RSS of f16 Qwen3 quantifies the cost side of the
  footprint question; the quality side stays with the P0 eval evidence and P4 shadow evidence.
