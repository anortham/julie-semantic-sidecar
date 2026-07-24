import contextlib
import hashlib
import importlib.util
import io
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
                    if method == "health":
                        result = {
                            "ready": True,
                            "dims": 2,
                            "resolved_backend": backend,
                            "accelerated": backend != "cpu",
                        }
                    elif method == "embed_query":
                        result = {"vector": [0.6, 0.8]}
                    elif method == "shutdown":
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
                "JULIE_SIDECAR_FORCE_BACKEND": "cpu",
                "JULIE_EMBEDDING_CACHE_DIR": str(cache),
            },
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


if __name__ == "__main__":
    unittest.main()
