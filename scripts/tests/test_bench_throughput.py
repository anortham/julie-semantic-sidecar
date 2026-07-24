import importlib.util
import pathlib
import unittest


SCRIPT_PATH = pathlib.Path(__file__).parents[1] / "bench-throughput.py"
SPEC = importlib.util.spec_from_file_location("bench_throughput", SCRIPT_PATH)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class BenchThroughputTests(unittest.TestCase):
    def test_expected_accelerator_requires_the_named_accelerated_backend(self):
        BENCH.validate_expected_backend(
            {"resolved_backend": "metal", "accelerated": True},
            "metal",
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


if __name__ == "__main__":
    unittest.main()
