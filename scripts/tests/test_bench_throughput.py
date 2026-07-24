import datetime
import hashlib
import importlib.util
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
            def __init__(self, binary, stderr_file, model=None):
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

    def test_parser_accepts_every_advertised_backend_name(self):
        for backend in ("cpu", "cuda", "directml", "mps", "metal", "vulkan"):
            with self.subTest(backend=backend):
                args = BENCH.parse_args(
                    ["--binary", "/tmp/sidecar", "--expect-backend", backend]
                )
                self.assertEqual(backend, args.expect_backend)


if __name__ == "__main__":
    unittest.main()
