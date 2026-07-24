#!/usr/bin/env python3
"""Steady-state embedding throughput benchmark for julie-semantic-sidecar.

Speaks the frozen `julie.embedding.sidecar` v1 protocol (newline-delimited JSON on
stdin/stdout) against a sidecar binary spawned as a child process. It probes `health`
first and refuses to measure a binary that is not `ready:true` — a `model_not_prepared`
sidecar FAILS the bench rather than reporting zeros. It then times `embed_batch` rounds
after a discarded warm-up round and reports steady-state units/s with a PASS/FAIL verdict
against a floor.

This makes the RC->v0.1.0 promotion gate's throughput floor checkable in one command:

    scripts/bench-throughput.py --binary target/release/julie-semantic-sidecar

See docs/rc-promotion-gate.md for the floor and its rationale, and the protocol contract
mirrored in Miller at docs/contracts/semantic-sidecar-protocol-v1.md.

Input texts are generated deterministically from their indices — no randomness, no
timestamps — so two runs embed byte-identical payloads.
"""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time

SCHEMA = "julie.embedding.sidecar"
VERSION = 1
PROTOCOL_MAX_BATCH = 250
DEFAULT_BATCH = 64
DEFAULT_ROUNDS = 4
DEFAULT_FLOOR = 40.0
HEALTH_TIMEOUT_S = 120.0
SUPPORTED_BACKENDS = ("cpu", "cuda", "directml", "mps", "metal", "vulkan")


class BenchError(Exception):
    pass


def deterministic_text(round_index, item_index):
    return (
        f"method Miller.Indexing.Semantic.ProbeType{round_index}.DoWork{item_index} "
        f"public VectorConvergeOutcome DoWork{item_index}(WorkspaceContext workspace, int revision) "
        f"Converges the {item_index}th probe cursor and records the outcome for round {round_index}."
    )


def rpc(request_id, method, params):
    return json.dumps(
        {
            "schema": SCHEMA,
            "version": VERSION,
            "request_id": request_id,
            "method": method,
            "params": params,
        },
        separators=(",", ":"),
    ) + "\n"


class Sidecar:
    def __init__(self, binary, stderr_file, model=None):
        command = [binary, "serve"]
        if model:
            command += ["--model", model]
        self._proc = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            text=True,
            bufsize=1,
        )

    def call(self, request_id, method, params):
        proc = self._proc
        if proc.stdin is None or proc.stdout is None:
            raise BenchError("sidecar stdio pipes are not open")
        start = time.monotonic()
        proc.stdin.write(rpc(request_id, method, params))
        proc.stdin.flush()
        line = proc.stdout.readline()
        elapsed = time.monotonic() - start
        if line == "":
            raise BenchError(
                "sidecar closed stdout before answering "
                f"'{method}' (exit code {proc.poll()})"
            )
        try:
            reply = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchError(f"sidecar emitted non-JSON on stdout: {line!r} ({exc})")
        if reply.get("error"):
            err = reply["error"]
            raise BenchError(
                f"sidecar error for '{method}': "
                f"[{err.get('code')}] {err.get('message')}"
            )
        return elapsed, reply.get("result", {})

    @property
    def pid(self):
        return self._proc.pid

    def close(self):
        proc = self._proc
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait(timeout=10)
        except Exception:
            proc.kill()


def sidecar_rss_bytes(pid):
    """Resident set of the live sidecar in bytes via ps (macOS/Linux); None where unavailable."""
    try:
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return int(out.stdout.strip()) * 1024
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_expected_backend(health, expected_backend):
    if expected_backend is None:
        return
    expected_accelerated = expected_backend != "cpu"
    resolved_backend = health.get("resolved_backend")
    accelerated = health.get("accelerated")
    if resolved_backend != expected_backend or accelerated is not expected_accelerated:
        raise BenchError(
            f"expected backend {expected_backend} with accelerated="
            f"{str(expected_accelerated).lower()}, got backend={resolved_backend} "
            f"accelerated={accelerated}"
        )


def run_bench(binary, batch, rounds, floor, model=None, expected_backend=None):
    resolved_binary = str(pathlib.Path(binary).resolve())
    resolved_harness = str(pathlib.Path(__file__).resolve())
    with tempfile.NamedTemporaryFile(
        mode="w+", suffix=".sidecar-bench-stderr.log", delete=False
    ) as stderr_file:
        stderr_path = stderr_file.name
        sidecar = Sidecar(resolved_binary, stderr_file, model)
        try:
            _, health = sidecar.call("bench-health", "health", {})
            if not health.get("ready", False):
                reason = health.get("degraded_reason") or "unknown"
                raise BenchError(
                    "sidecar health is not ready "
                    f"(degraded_reason={reason}); a not-prepared sidecar cannot be "
                    "benched — run `prepare` first. stderr: " + stderr_path
                )
            validate_expected_backend(health, expected_backend)

            warmup_texts = [deterministic_text(0, i) for i in range(batch)]
            sidecar.call("bench-warmup", "embed_batch", {"texts": warmup_texts})

            rates = []
            for r in range(1, rounds + 1):
                texts = [deterministic_text(r, i) for i in range(batch)]
                elapsed, result = sidecar.call(
                    f"bench-round-{r}", "embed_batch", {"texts": texts}
                )
                n = len(result.get("vectors", []))
                if n != batch:
                    raise BenchError(
                        f"round {r}: expected {batch} vectors, got {n}"
                    )
                if elapsed <= 0:
                    raise BenchError(f"round {r}: non-positive elapsed time")
                rates.append(n / elapsed)

            rss_bytes = sidecar_rss_bytes(sidecar.pid)
        finally:
            sidecar.close()

    steady = sum(rates) / len(rates)
    return {
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "binary": resolved_binary,
        "binary_sha256": sha256_file(resolved_binary),
        "harness": resolved_harness,
        "harness_sha256": sha256_file(resolved_harness),
        "forced_backend": os.environ.get("JULIE_SIDECAR_FORCE_BACKEND"),
        "cache_dir": os.environ.get("JULIE_EMBEDDING_CACHE_DIR"),
        "sidecar_rss_bytes": rss_bytes,
        "batch": batch,
        "rounds": rounds,
        "warmup_rounds": 1,
        "floor_units_per_s": floor,
        "expected_backend": expected_backend,
        "expected_backend_selected": None if expected_backend is None else True,
        "steady_state_units_per_s": steady,
        "per_round_units_per_s": rates,
        "pass": steady >= floor,
        "health": {
            "ready": health.get("ready"),
            "dims": health.get("dims"),
            "model_id": health.get("model_id"),
            "resolved_backend": health.get("resolved_backend"),
            "device": health.get("device"),
            "accelerated": health.get("accelerated"),
            "sidecar_version": health.get("sidecar_version"),
        },
    }


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Steady-state embedding throughput benchmark for julie-semantic-sidecar."
    )
    parser.add_argument("--binary", required=True, help="path to the sidecar binary")
    parser.add_argument(
        "--batch",
        type=int,
        default=DEFAULT_BATCH,
        help=f"texts per embed_batch (default {DEFAULT_BATCH}, protocol max {PROTOCOL_MAX_BATCH})",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DEFAULT_ROUNDS,
        help=f"measured rounds after one warm-up round (default {DEFAULT_ROUNDS})",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=DEFAULT_FLOOR,
        help=f"PASS/FAIL floor in units/s (default {DEFAULT_FLOOR})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="model id passed through to `serve --model` (default: the sidecar's default model)",
    )
    parser.add_argument(
        "--expect-backend",
        choices=SUPPORTED_BACKENDS,
        help="fail before measuring unless health reports this backend truthfully",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the result as a single JSON object"
    )
    args = parser.parse_args(argv)
    if args.batch < 1:
        parser.error("--batch must be >= 1")
    if args.batch > PROTOCOL_MAX_BATCH:
        parser.error(
            f"--batch {args.batch} exceeds the protocol maximum of {PROTOCOL_MAX_BATCH}"
        )
    if args.rounds < 1:
        parser.error("--rounds must be >= 1")
    if args.floor < 0:
        parser.error("--floor must be >= 0")
    return args


def main(argv):
    args = parse_args(argv)
    try:
        report = run_bench(
            args.binary,
            args.batch,
            args.rounds,
            args.floor,
            args.model,
            args.expect_backend,
        )
    except BenchError as exc:
        if args.json:
            print(json.dumps({"error": str(exc), "pass": False}))
        else:
            print(f"bench: FAIL — {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report))
    else:
        h = report["health"]
        print(
            f"health: ready backend={h['resolved_backend']} device={h['device']} "
            f"accelerated={h['accelerated']} dims={h['dims']} model={h['model_id']}"
        )
        for i, rate in enumerate(report["per_round_units_per_s"], start=1):
            print(f"round {i}: {rate:.1f} units/s")
        verdict = "PASS" if report["pass"] else "FAIL"
        print(
            f"steady-state: {report['steady_state_units_per_s']:.1f} units/s "
            f"(batch={report['batch']}, {report['rounds']} rounds, warm model) "
            f"vs floor {report['floor_units_per_s']:.1f} -> {verdict}"
        )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
