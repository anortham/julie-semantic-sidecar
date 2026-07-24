#!/usr/bin/env python3

import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import pathlib
import queue
import subprocess
import sys
import tempfile
import threading
import time


SCHEMA = "julie.embedding.sidecar"
VERSION = 1
MAX_PIPELINED_REQUESTS = 32


class ProbeError(Exception):
    pass


class Sidecar:
    def __init__(self, binary, model, response_timeout):
        command = [binary, "serve"]
        if model:
            command += ["--model", model]
        self.response_timeout = response_timeout
        self.stderr_file = tempfile.TemporaryFile(mode="w+")
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.stderr_file,
                text=True,
                env=os.environ.copy(),
            )
        except BaseException:
            self.stderr_file.close()
            raise
        self.replies = queue.Queue()
        self.reader = threading.Thread(target=self._read_stdout, daemon=True)
        self.reader.start()
        self.closed = False

    def _read_stdout(self):
        for line in self.process.stdout:
            self.replies.put(line)
        self.replies.put(None)

    def _read_reply(self):
        try:
            line = self.replies.get(timeout=self.response_timeout)
        except queue.Empty as error:
            self.abort()
            raise ProbeError(
                f"sidecar response timed out after {self.response_timeout:g}s"
            ) from error
        if line is None:
            self.stderr_file.seek(0)
            stderr = self.stderr_file.read()
            raise ProbeError(f"sidecar exited before replying: {stderr.strip()}")
        return json.loads(line)

    def pipeline(self, client_index, texts):
        requests = [
            {
                "schema": SCHEMA,
                "version": VERSION,
                "request_id": f"client-{client_index}-health",
                "method": "health",
                "params": {},
            }
        ]
        requests.extend(
            {
                "schema": SCHEMA,
                "version": VERSION,
                "request_id": f"client-{client_index}-query-{query_index}",
                "method": "embed_query",
                "params": {"text": text},
            }
            for query_index, text in enumerate(texts)
        )
        started = time.monotonic()
        for request in requests:
            self.process.stdin.write(json.dumps(request, separators=(",", ":")) + "\n")
        self.process.stdin.flush()

        replies = []
        for request in requests:
            reply = self._read_reply()
            if reply.get("request_id") != request["request_id"]:
                raise ProbeError(
                    f"response order mismatch: expected {request['request_id']}, "
                    f"got {reply.get('request_id')}"
                )
            if "error" in reply:
                raise ProbeError(f"{request['request_id']}: {reply['error']}")
            replies.append(reply["result"])
        elapsed_ms = (time.monotonic() - started) * 1000

        health = replies[0]
        if health.get("ready") is not True:
            raise ProbeError(f"client {client_index} not ready: {health}")
        vectors = [result["vector"] for result in replies[1:]]
        if any(len(vector) != health["dims"] for vector in vectors):
            raise ProbeError(f"client {client_index} returned a wrong vector dimension")
        return {
            "client": client_index,
            "elapsed_ms": elapsed_ms,
            "health": health,
            "vectors": vectors,
        }

    def close(self):
        if self.closed:
            return self.process.returncode
        try:
            if self.process.poll() is not None:
                return self.process.returncode
            request = {
                "schema": SCHEMA,
                "version": VERSION,
                "request_id": "shutdown",
                "method": "shutdown",
                "params": {},
            }
            self.process.stdin.write(
                json.dumps(request, separators=(",", ":")) + "\n"
            )
            self.process.stdin.flush()
            reply = self._read_reply()
            if reply.get("result") != {"stopping": True}:
                raise ProbeError(f"shutdown mismatch: {reply}")
            self.process.stdin.close()
            return self.process.wait(timeout=self.response_timeout)
        except (
            BrokenPipeError,
            OSError,
            subprocess.SubprocessError,
            ProbeError,
            ValueError,
        ):
            self.abort()
            raise
        finally:
            self._close_resources()

    def abort(self):
        if self.process.poll() is None:
            self.process.kill()
        try:
            self.process.wait(timeout=self.response_timeout)
        except subprocess.TimeoutExpired:
            pass
        self._close_resources()

    def _close_resources(self):
        if self.closed:
            return
        self.closed = True
        for stream in (self.process.stdin, self.process.stdout, self.stderr_file):
            if stream:
                try:
                    stream.close()
                except OSError:
                    pass


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_probe(binary, clients, requests, model, expected_backend, response_timeout):
    texts = [
        f"deterministic concurrent semantic probe {request_index}"
        for request_index in range(requests)
    ]
    resolved_binary = str(pathlib.Path(binary).resolve())
    resolved_harness = str(pathlib.Path(__file__).resolve())
    sidecars = []
    try:
        for _ in range(clients):
            sidecars.append(Sidecar(resolved_binary, model, response_timeout))
    except BaseException:
        for sidecar in sidecars:
            sidecar.abort()
        raise

    started = time.monotonic()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=clients) as executor:
            futures = [
                executor.submit(sidecar.pipeline, index, texts)
                for index, sidecar in enumerate(sidecars)
            ]
            try:
                results = [future.result() for future in futures]
            except BaseException:
                for sidecar in sidecars:
                    sidecar.abort()
                for future in futures:
                    future.cancel()
                raise

        exit_codes = []
        for sidecar in sidecars:
            try:
                exit_codes.append(sidecar.close())
            except BaseException:
                for remaining in sidecars:
                    remaining.abort()
                raise
    finally:
        for sidecar in sidecars:
            if sidecar.process.poll() is None:
                sidecar.abort()
    total_ms = (time.monotonic() - started) * 1000

    reference = results[0]["vectors"]
    deterministic = all(result["vectors"] == reference for result in results[1:])
    consistent_health = all(
        result["health"] == results[0]["health"] for result in results[1:]
    )
    expected_accelerated = expected_backend != "cpu"
    expected_health = all(
        result["health"].get("resolved_backend") == expected_backend
        and result["health"].get("accelerated") is expected_accelerated
        for result in results
    )
    clean_exit = all(code == 0 for code in exit_codes)
    return {
        "recorded_utc": datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "binary": resolved_binary,
        "binary_sha256": sha256_file(resolved_binary),
        "harness": resolved_harness,
        "harness_sha256": sha256_file(resolved_harness),
        "model": model,
        "expected_backend": expected_backend,
        "forced_backend": os.environ.get("JULIE_SIDECAR_FORCE_BACKEND"),
        "cache_dir": os.environ.get("JULIE_EMBEDDING_CACHE_DIR"),
        "clients": clients,
        "requests_per_client": requests,
        "response_timeout_seconds": response_timeout,
        "total_requests": clients * requests,
        "total_wall_ms": total_ms,
        "client_elapsed_ms": [result["elapsed_ms"] for result in results],
        "health": results[0]["health"],
        "vectors_bit_exact_across_processes": deterministic,
        "health_equal_across_processes": consistent_health,
        "expected_backend_selected": expected_health,
        "clean_exit": clean_exit,
        "pass": deterministic and consistent_health and expected_health and clean_exit,
    }


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Probe pipelined requests and concurrent sidecar processes."
    )
    parser.add_argument("--binary", required=True)
    parser.add_argument("--clients", type=int, default=3)
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--model")
    parser.add_argument(
        "--expect-backend",
        required=True,
        choices=("cpu", "metal", "vulkan", "cuda"),
    )
    parser.add_argument("--response-timeout", type=float, default=60)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.clients < 2:
        parser.error("--clients must be at least 2")
    if args.requests < 2:
        parser.error("--requests must be at least 2")
    if args.requests > MAX_PIPELINED_REQUESTS:
        parser.error(f"--requests must be at most {MAX_PIPELINED_REQUESTS}")
    if args.response_timeout <= 0:
        parser.error("--response-timeout must be greater than 0")
    return args


def main(argv):
    args = parse_args(argv)
    report = run_probe(
        args.binary,
        args.clients,
        args.requests,
        args.model,
        args.expect_backend,
        args.response_timeout,
    )
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"{'PASS' if report['pass'] else 'FAIL'} "
            f"clients={report['clients']} requests={report['total_requests']} "
            f"wall_ms={report['total_wall_ms']:.1f} "
            f"backend={report['health'].get('resolved_backend')}"
        )
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except (ProbeError, OSError, subprocess.SubprocessError, ValueError) as error:
        print(f"probe-concurrency: {error}", file=sys.stderr)
        raise SystemExit(1)
