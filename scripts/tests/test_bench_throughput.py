import contextlib
import datetime
import hashlib
import importlib.util
import io
import os
import pathlib
import tempfile
import unittest
from unittest import mock


SCRIPT_PATH = pathlib.Path(__file__).parents[1] / "bench-throughput.py"
SPEC = importlib.util.spec_from_file_location("bench_throughput", SCRIPT_PATH)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class BenchThroughputTests(unittest.TestCase):
    def test_report_binds_binary_harness_environment_and_backend_truth(self):
        class FakeSidecar:
            def __init__(
                self,
                binary,
                stderr_file,
                model=None,
                response_timeout=BENCH.DEFAULT_RESPONSE_TIMEOUT_S,
            ):
                self.pid = 42

            def call(self, request_id, method, params):
                if method == "health":
                    return 0.01, {
                        "ready": True,
                        "dims": 384,
                        "model_id": "bge-small-en-v1.5-f32",
                        "resolved_backend": "metal",
                        "device": "Apple M2 Ultra",
                        "accelerated": True,
                        "sidecar_version": "0.1.0-rc.3",
                    }
                return 0.01, {"vectors": [[1.0]] * len(params["texts"])}

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp:
            binary = pathlib.Path(temp) / "julie-semantic-sidecar"
            binary.write_bytes(b"released-sidecar")
            cache = pathlib.Path(temp) / "cache"
            with (
                mock.patch.object(BENCH, "Sidecar", FakeSidecar),
                mock.patch.object(BENCH, "sidecar_rss_bytes", return_value=1024),
                mock.patch.object(BENCH.os, "unlink", wraps=os.unlink) as unlink,
                mock.patch.dict(
                    os.environ,
                    {
                        "JULIE_EMBEDDING_CACHE_DIR": str(cache),
                        "JULIE_SIDECAR_FORCE_BACKEND": "metal",
                    },
                    clear=False,
                ),
            ):
                report = BENCH.run_bench(
                    str(binary),
                    batch=2,
                    rounds=1,
                    floor=40,
                    expected_backend="metal",
                )

        unlink.assert_called_once()
        datetime.datetime.fromisoformat(report["recorded_utc"].replace("Z", "+00:00"))
        self.assertEqual(str(binary.resolve()), report["binary"])
        self.assertEqual(
            hashlib.sha256(b"released-sidecar").hexdigest(),
            report["binary_sha256"],
        )
        self.assertEqual(str(SCRIPT_PATH.resolve()), report["harness"])
        self.assertEqual(
            hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
            report["harness_sha256"],
        )
        self.assertEqual(str(cache), report["cache_dir"])
        self.assertEqual("metal", report["forced_backend"])
        self.assertEqual("metal", report["expected_backend"])
        self.assertEqual("metal", report["health"]["resolved_backend"])
        self.assertTrue(report["health"]["accelerated"])

    def test_expected_accelerator_requires_the_named_accelerated_backend(self):
        for backend in ("cuda", "directml", "mps", "metal", "vulkan"):
            with self.subTest(backend=backend):
                BENCH.validate_expected_backend(
                    {"resolved_backend": backend, "accelerated": True},
                    backend,
                )

        with self.assertRaisesRegex(BENCH.BenchError, "expected backend metal"):
            BENCH.validate_expected_backend(
                {"resolved_backend": "cpu", "accelerated": False},
                "metal",
            )

    def test_expected_cpu_requires_nonaccelerated_cpu(self):
        BENCH.validate_expected_backend(
            {"resolved_backend": "cpu", "accelerated": False},
            "cpu",
        )

        with self.assertRaisesRegex(BENCH.BenchError, "expected backend cpu"):
            BENCH.validate_expected_backend(
                {"resolved_backend": "cpu", "accelerated": True},
                "cpu",
            )

    def test_report_derives_expected_backend_selection_from_health(self):
        class FakeSidecar:
            def __init__(
                self,
                binary,
                stderr_file,
                model=None,
                response_timeout=BENCH.DEFAULT_RESPONSE_TIMEOUT_S,
            ):
                self.pid = 42

            def call(self, request_id, method, params):
                if method == "health":
                    return 0.01, {
                        "ready": True,
                        "dims": 384,
                        "model_id": "bge-small-en-v1.5-f32",
                        "resolved_backend": "cpu",
                        "device": None,
                        "accelerated": False,
                        "sidecar_version": "fixture",
                    }
                return 0.01, {"vectors": [[1.0]] * len(params["texts"])}

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temp:
            binary = pathlib.Path(temp) / "julie-semantic-sidecar"
            binary.write_bytes(b"released-sidecar")
            with (
                mock.patch.object(BENCH, "Sidecar", FakeSidecar),
                mock.patch.object(BENCH, "sidecar_rss_bytes", return_value=1024),
                mock.patch.object(BENCH, "validate_expected_backend"),
            ):
                report = BENCH.run_bench(
                    str(binary),
                    batch=2,
                    rounds=1,
                    floor=40,
                    expected_backend="metal",
                )

        self.assertFalse(report["expected_backend_selected"])

    def test_parser_accepts_every_advertised_backend_name(self):
        for backend in ("cpu", "cuda", "directml", "mps", "metal", "vulkan"):
            with self.subTest(backend=backend):
                args = BENCH.parse_args(
                    ["--binary", "/tmp/sidecar", "--expect-backend", backend]
                )
                self.assertEqual(backend, args.expect_backend)

    def test_positive_floor_requires_expected_backend(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                BENCH.parse_args(["--binary", "/tmp/sidecar", "--floor", "40"])

        args = BENCH.parse_args(["--binary", "/tmp/sidecar", "--floor", "0"])
        self.assertIsNone(args.expect_backend)

    def test_programmatic_positive_floor_requires_expected_backend(self):
        with tempfile.TemporaryDirectory() as temp:
            binary = pathlib.Path(temp) / "julie-semantic-sidecar"
            binary.write_bytes(b"released-sidecar")
            with self.assertRaisesRegex(
                BENCH.BenchError,
                "expected backend is required",
            ):
                BENCH.run_bench(
                    str(binary),
                    batch=2,
                    rounds=1,
                    floor=40,
                    expected_backend=None,
                )

    def test_protocol_reply_must_be_an_ordered_object_result(self):
        valid = {
            "request_id": "bench-health",
            "result": {"ready": True},
        }
        self.assertEqual(
            {"ready": True},
            BENCH.validate_reply(valid, "bench-health", "health"),
        )

        invalid = [
            (None, "response envelope is not an object"),
            (
                {"request_id": "other", "result": {}},
                "response order mismatch",
            ),
            (
                {"request_id": "bench-health", "result": []},
                "result is not an object",
            ),
            (
                {"request_id": "bench-health", "error": "bad"},
                "error is not an object",
            ),
        ]
        for reply, message in invalid:
            with self.subTest(reply=reply):
                with self.assertRaisesRegex(BENCH.BenchError, message):
                    BENCH.validate_reply(reply, "bench-health", "health")

    def test_read_timeout_aborts_the_sidecar(self):
        sidecar = object.__new__(BENCH.Sidecar)
        sidecar.replies = BENCH.queue.Queue()
        sidecar.response_timeout = 0.01
        sidecar.abort = mock.Mock()

        with self.assertRaisesRegex(BENCH.BenchError, "timed out"):
            sidecar._read_reply()

        sidecar.abort.assert_called_once_with()

    def test_close_reaps_the_sidecar_after_a_wait_timeout(self):
        process = mock.Mock()
        process.poll.return_value = None
        process.wait.side_effect = [
            BENCH.subprocess.TimeoutExpired("sidecar", 10),
            0,
        ]
        sidecar = object.__new__(BENCH.Sidecar)
        sidecar._proc = process

        sidecar.close()

        process.kill.assert_called_once_with()
        self.assertEqual(2, process.wait.call_count)

    def test_json_os_error_is_a_machine_readable_exit_two(self):
        output = io.StringIO()
        with (
            mock.patch.object(
                BENCH,
                "run_bench",
                side_effect=OSError("binary missing"),
            ),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            try:
                exit_code = BENCH.main(
                    [
                        "--binary",
                        "/tmp/missing-sidecar",
                        "--floor",
                        "0",
                        "--json",
                    ]
                )
            except OSError:
                self.fail("main must render operating-system failures instead of raising")

        self.assertEqual(2, exit_code)
        self.assertEqual(
            {"error": "binary missing", "pass": False},
            BENCH.json.loads(output.getvalue()),
        )


if __name__ == "__main__":
    unittest.main()
