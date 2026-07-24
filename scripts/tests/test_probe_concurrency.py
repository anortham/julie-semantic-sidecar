import contextlib
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import tempfile
import textwrap
import unittest
from unittest import mock


SCRIPT_PATH = pathlib.Path(__file__).parents[1] / "probe-concurrency.py"
SPEC = importlib.util.spec_from_file_location("probe_concurrency", SCRIPT_PATH)
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


class ProbeConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp.name)
        self.binary = self.root / "fake-sidecar"
        self.binary.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                import json
                import os
                import pathlib
                import sys
                import time

                pid_dir = os.environ.get("FAKE_PID_DIR")
                if pid_dir:
                    pathlib.Path(pid_dir, str(os.getpid())).write_text("")
                backend = os.environ.get("FAKE_BACKEND", "cpu")
                for line in sys.stdin:
                    request = json.loads(line)
                    if os.environ.get("FAKE_MODE") == "hang":
                        time.sleep(300)
                    method = request["method"]
                    if method == "embed_query":
                        time.sleep(float(os.environ.get("FAKE_QUERY_DELAY", "0")))
                    if method == "health":
                        result = {
                            "ready": True,
                            "dims": 2,
                            "resolved_backend": backend,
                            "accelerated": backend != "cpu",
                        }
                    elif method == "embed_query":
                        result = {"vector": [0.6, 0.8]}
                        if os.environ.get("FAKE_MODE") == "missing-vector":
                            result = {}
                    elif method == "shutdown":
                        if os.environ.get("FAKE_MODE") == "malformed-shutdown":
                            print("null", flush=True)
                            break
                        result = {"stopping": True}
                    else:
                        raise RuntimeError(method)
                    print(json.dumps({
                        "schema": "julie.embedding.sidecar",
                        "version": 1,
                        "request_id": request["request_id"],
                        "result": result,
                    }), flush=True)
                    if method == "shutdown":
                        break
                """
            ),
            encoding="utf-8",
        )
        self.binary.chmod(0o755)

    def tearDown(self):
        self.temp.cleanup()

    def test_report_binds_binary_environment_and_expected_backend(self):
        cache = self.root / "cache"
        with mock.patch.dict(
            os.environ,
            {
                "FAKE_BACKEND": "cpu",
                "FAKE_QUERY_DELAY": "0.05",
                "JULIE_SIDECAR_FORCE_BACKEND": "cpu",
                "JULIE_EMBEDDING_CACHE_DIR": str(cache),
            },
            clear=False,
        ):
            report = PROBE.run_probe(
                str(self.binary),
                clients=3,
                requests=8,
                model=None,
                expected_backend="cpu",
                response_timeout=5,
            )

        expected_sha = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        expected_harness_sha = hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest()
        self.assertTrue(report["pass"])
        self.assertEqual(expected_sha, report["binary_sha256"])
        self.assertEqual(expected_harness_sha, report["harness_sha256"])
        self.assertEqual(str(SCRIPT_PATH.resolve()), report["harness"])
        self.assertEqual(str(self.binary.resolve()), report["binary"])
        self.assertEqual("cpu", report["forced_backend"])
        self.assertEqual(str(cache), report["cache_dir"])
        self.assertTrue(report["expected_backend_selected"])
        self.assertTrue(report["gate_minimums_satisfied"])
        self.assertTrue(report["pipelines_overlapped"])
        self.assertGreater(report["pipeline_overlap_ms"], 0)

    def test_programmatic_below_gate_shape_cannot_pass(self):
        with mock.patch.dict(
            os.environ,
            {"FAKE_BACKEND": "cpu", "FAKE_QUERY_DELAY": "0.05"},
            clear=False,
        ):
            report = PROBE.run_probe(
                str(self.binary),
                clients=2,
                requests=2,
                model=None,
                expected_backend="cpu",
                response_timeout=5,
            )

        self.assertIn(
            "gate_minimums_satisfied",
            report,
            "programmatic evidence must disclose its gate shape",
        )
        self.assertFalse(report.get("gate_minimums_satisfied"))
        self.assertFalse(report["pass"])

    def test_backend_mismatch_fails_the_probe(self):
        with mock.patch.dict(os.environ, {"FAKE_BACKEND": "cpu"}, clear=False):
            report = PROBE.run_probe(
                str(self.binary),
                clients=2,
                requests=2,
                model=None,
                expected_backend="metal",
                response_timeout=5,
            )

        self.assertFalse(report["pass"])
        self.assertFalse(report["expected_backend_selected"])

    def test_response_timeout_kills_every_sidecar(self):
        pid_dir = self.root / "pids"
        pid_dir.mkdir()
        with mock.patch.dict(
            os.environ,
            {"FAKE_MODE": "hang", "FAKE_PID_DIR": str(pid_dir)},
            clear=False,
        ):
            with self.assertRaises(PROBE.ProbeError):
                PROBE.run_probe(
                    str(self.binary),
                    clients=2,
                    requests=2,
                    model=None,
                    expected_backend="cpu",
                    response_timeout=1,
                )

        pids = [int(path.name) for path in pid_dir.iterdir()]
        self.assertEqual(2, len(pids))
        for pid in pids:
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)

    def test_partial_spawn_failure_aborts_processes_already_started(self):
        first = mock.Mock()
        with mock.patch.object(
            PROBE,
            "Sidecar",
            side_effect=[first, OSError("spawn failed")],
        ):
            with self.assertRaisesRegex(OSError, "spawn failed"):
                PROBE.run_probe(
                    str(self.binary),
                    clients=2,
                    requests=2,
                    model=None,
                    expected_backend="cpu",
                    response_timeout=5,
                )

        first.abort.assert_called_once_with()

    def test_parse_rejects_a_pipeline_that_can_deadlock(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                PROBE.parse_args(
                    [
                        "--binary",
                        str(self.binary),
                        "--clients",
                        "2",
                        "--requests",
                        str(PROBE.MAX_PIPELINED_REQUESTS + 1),
                        "--expect-backend",
                        "cpu",
                    ]
                )

    def test_parse_rejects_below_gate_process_and_request_counts(self):
        for flag, value in (("--clients", "2"), ("--requests", "7")):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    PROBE.parse_args(
                        [
                            "--binary",
                            str(self.binary),
                            "--clients",
                            "3",
                            "--requests",
                            "8",
                            flag,
                            value,
                            "--expect-backend",
                            "cpu",
                        ]
                    )

    def test_json_harness_error_is_a_machine_readable_exit_two(self):
        output = io.StringIO()
        with (
            mock.patch.object(
                PROBE,
                "run_probe",
                side_effect=PROBE.ProbeError("protocol failed"),
            ),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            try:
                exit_code = PROBE.main(
                    [
                        "--binary",
                        str(self.binary),
                        "--clients",
                        "3",
                        "--requests",
                        "8",
                        "--expect-backend",
                        "cpu",
                        "--json",
                    ]
                )
            except PROBE.ProbeError:
                self.fail("main must render harness failures instead of raising")

        self.assertEqual(2, exit_code)
        self.assertEqual(
            {"error": "protocol failed", "pass": False},
            json.loads(output.getvalue()),
        )

    def test_malformed_reply_is_a_machine_readable_exit_two(self):
        output = io.StringIO()
        with (
            mock.patch.dict(os.environ, {"FAKE_MODE": "missing-vector"}, clear=False),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            try:
                exit_code = PROBE.main(
                    [
                        "--binary",
                        str(self.binary),
                        "--clients",
                        "3",
                        "--requests",
                        "8",
                        "--expect-backend",
                        "cpu",
                        "--json",
                    ]
                )
            except KeyError:
                self.fail("malformed replies must become probe errors")

        self.assertEqual(2, exit_code)
        report = json.loads(output.getvalue())
        self.assertFalse(report["pass"])
        self.assertIn("vector", report["error"])

    def test_malformed_shutdown_is_a_machine_readable_exit_two(self):
        output = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "FAKE_BACKEND": "cpu",
                    "FAKE_QUERY_DELAY": "0.05",
                    "FAKE_MODE": "malformed-shutdown",
                },
                clear=False,
            ),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            try:
                exit_code = PROBE.main(
                    [
                        "--binary",
                        str(self.binary),
                        "--clients",
                        "3",
                        "--requests",
                        "8",
                        "--expect-backend",
                        "cpu",
                        "--json",
                    ]
                )
            except AttributeError:
                self.fail("malformed shutdown replies must become probe errors")

        self.assertEqual(2, exit_code)
        report = json.loads(output.getvalue())
        self.assertFalse(report["pass"])
        self.assertIn("shutdown", report["error"])

    def test_parser_accepts_every_advertised_backend_name(self):
        for backend in ("cpu", "cuda", "directml", "mps", "metal", "vulkan"):
            with self.subTest(backend=backend):
                args = PROBE.parse_args(
                    [
                        "--binary",
                        str(self.binary),
                        "--clients",
                        "3",
                        "--requests",
                        "8",
                        "--expect-backend",
                        backend,
                    ]
                )
                self.assertEqual(backend, args.expect_backend)

    def test_common_overlap_requires_every_pipeline_to_be_live(self):
        self.assertEqual(
            60,
            PROBE.common_overlap_ms(
                [
                    {"started_monotonic": 1.00, "finished_monotonic": 1.10},
                    {"started_monotonic": 1.04, "finished_monotonic": 1.12},
                ]
            ),
        )
        self.assertEqual(
            0,
            PROBE.common_overlap_ms(
                [
                    {"started_monotonic": 1.00, "finished_monotonic": 1.04},
                    {"started_monotonic": 1.04, "finished_monotonic": 1.12},
                ]
            ),
        )

    def test_pass_fails_when_pipeline_windows_do_not_overlap(self):
        with (
            mock.patch.dict(
                os.environ,
                {"FAKE_BACKEND": "cpu", "FAKE_QUERY_DELAY": "0.05"},
                clear=False,
            ),
            mock.patch.object(PROBE, "common_overlap_ms", return_value=0),
        ):
            report = PROBE.run_probe(
                str(self.binary),
                clients=2,
                requests=2,
                model=None,
                expected_backend="cpu",
                response_timeout=5,
            )

        self.assertFalse(report["pipelines_overlapped"])
        self.assertFalse(report["pass"])


if __name__ == "__main__":
    unittest.main()
