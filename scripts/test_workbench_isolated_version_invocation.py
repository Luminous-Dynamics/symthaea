#!/usr/bin/env python3
from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import verify_workbench_isolated_version_invocation as verifier
import workbench_isolated_version_invocation_capture as producer

FAKE_ROOT = "/nix/store/0123456789abcdfghijklmnpqrsvwxyz-connectome-workbench-2.1.0"
FAKE_SHA = "sha256:" + "a" * 64
FAKE_CLOSURE_CAPTURE = "sha256:" + "b" * 64


def isolation_profile() -> dict:
    fixed = {
        "LANG": "C", "LC_ALL": "C", "TZ": "UTC0",
        "OMP_NUM_THREADS": "1", "OMP_DYNAMIC": "FALSE", "PATH": "",
    }
    dynamic = {
        "PWD": "invocation.cwd", "HOME": "invocation.home",
        "XDG_CONFIG_HOME": "invocation.xdg_config_home",
        "XDG_CACHE_HOME": "invocation.xdg_cache_home",
        "XDG_DATA_HOME": "invocation.xdg_data_home",
        "XDG_STATE_HOME": "invocation.xdg_state_home",
        "XDG_RUNTIME_DIR": "invocation.xdg_runtime_dir",
        "XDG_CONFIG_DIRS": "invocation.xdg_config_dirs",
        "XDG_DATA_DIRS": "invocation.xdg_data_dirs",
        "TMPDIR": "invocation.tmpdir", "TMP": "invocation.tmpdir", "TEMP": "invocation.tmpdir",
    }
    return {"process_environment": {"stage": "root-main-program-execve", "fixed": fixed, "dynamic_bindings": dynamic}}


def parent_profile() -> dict:
    return {"nixpkgs_package": {"version": "2.1.0"}}


class VersionParserTests(unittest.TestCase):
    def test_exact_version_line(self):
        self.assertEqual(producer.parse_version(b"Version: 2.1.0\n", b"", "2.1.0"), "2.1.0")
        self.assertEqual(verifier.independently_parse_version(b"", b"Version: 2.1.0\r\n", "2.1.0"), "2.1.0")

    def test_wrong_version_rejected(self):
        with self.assertRaises(producer.CaptureError):
            producer.parse_version(b"Version: 2.2.0\n", b"", "2.1.0")

    def test_duplicate_version_lines_rejected(self):
        with self.assertRaises(verifier.VerificationError):
            verifier.independently_parse_version(b"Version: 2.1.0\nVersion: 2.1.0\n", b"", "2.1.0")

    def test_non_utf8_rejected(self):
        with self.assertRaises(verifier.VerificationError):
            verifier.independently_parse_version(b"\xff", b"", "2.1.0")


class ProducerPrimitiveTests(unittest.TestCase):
    def test_entry_environment_is_exact_and_host_free(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            base, roots, _ = producer.create_isolation_roots(parent, "case")
            try:
                env = producer.entry_environment(isolation_profile(), roots)
                self.assertEqual(env["PATH"], "")
                self.assertEqual(env["PWD"], str(roots["cwd"]))
                self.assertEqual(env["TMPDIR"], env["TMP"])
                self.assertEqual(env["TMP"], env["TEMP"])
                self.assertNotIn("USER", env)
            finally:
                import shutil
                if base.exists():
                    shutil.rmtree(base)

    def test_isolation_roots_are_private_empty_distinct(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            base, roots, lifecycle = producer.create_isolation_roots(parent, "case")
            try:
                self.assertEqual(len(set(roots.values())), len(producer.ROOT_ROLES))
                for role, path in roots.items():
                    self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700)
                    self.assertEqual(list(path.iterdir()), [])
                    self.assertEqual(lifecycle[role]["directory_mode_octal"], "0700")
            finally:
                import shutil
                shutil.rmtree(base)

    def test_isolation_root_construction_failure_cleans_partial_base(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            original_mkdir = Path.mkdir

            def fail_home(path, *args, **kwargs):
                if path.name == "home":
                    raise OSError("injected root-construction failure")
                return original_mkdir(path, *args, **kwargs)

            with mock.patch.object(Path, "mkdir", new=fail_home):
                with self.assertRaises(OSError):
                    producer.create_isolation_roots(parent, "case")
            self.assertEqual(list(parent.iterdir()), [])

    def test_publish_capture_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            staging = parent / "staging"
            staging.mkdir()
            (staging / "receipt.json").write_text("new")
            final = parent / "final"
            final.mkdir()
            marker = final / "marker"
            marker.write_text("existing")
            with self.assertRaises(producer.CaptureError):
                producer.publish_capture(staging, final)
            self.assertEqual(marker.read_text(), "existing")
            self.assertTrue(staging.exists())

    def test_run_version_uses_exact_environment(self):
        with tempfile.TemporaryDirectory() as td:
            parent = Path(td)
            executable = parent / "wb_command"
            executable.write_text("#!/bin/sh\nprintf 'Version: 2.1.0\\n'\n")
            executable.chmod(0o700)
            seen = {}

            def fake_run(argv, **kwargs):
                seen["argv"] = argv
                seen["kwargs"] = kwargs
                class P:
                    returncode = 0
                    stdout = b"Version: 2.1.0\n"
                    stderr = b""
                return P()

            with mock.patch.object(producer.subprocess, "run", side_effect=fake_run):
                result = producer.run_version(executable, isolation_profile(), parent, "case")
            try:
                self.assertEqual(seen["argv"], [str(executable), "-version"])
                self.assertEqual(seen["kwargs"]["env"]["PATH"], "")
                self.assertFalse(seen["kwargs"]["shell"])
                self.assertTrue(seen["kwargs"]["close_fds"])
                self.assertEqual(seen["kwargs"]["pass_fds"], ())
                self.assertEqual(result["pre_executable_sha256"], result["post_executable_sha256"])
            finally:
                import shutil
                shutil.rmtree(result["_base"])


class ReceiptVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.capture = self.root / "capture"
        (self.capture / "raw").mkdir(parents=True)
        self.stdout = b"Version: 2.1.0\nCommit Date: test\nOperating System: Linux\n"
        self.stderr = b""
        (self.capture / "raw/wb-version.stdout").write_bytes(self.stdout)
        (self.capture / "raw/wb-version.stderr").write_bytes(self.stderr)
        self.execution_profile = self.root / "execution.json"
        self.execution_profile.write_text(json.dumps(parent_profile()))
        self.isolation_profile_path = self.root / "isolation.json"
        self.isolation_profile_value = isolation_profile()
        self.isolation_profile_path.write_text(json.dumps(self.isolation_profile_value))
        self.isolation_verifier_path = self.root / "isolation_verifier.py"
        self.isolation_verifier_path.write_text("x=1\n")
        self.producer_path = self.root / "producer.py"
        self.producer_path.write_text("x=2\n")
        self.parent = {
            "root": FAKE_ROOT,
            "closure_capture_digest": FAKE_CLOSURE_CAPTURE,
            "target": {"content_sha256": FAKE_SHA},
        }
        base = "/tmp/symthaea-test-isolation"
        roots = {role: f"{base}/{role}" for role in verifier.ROOT_ROLES}
        dynamic_map = {
            "PWD": "cwd", "HOME": "home", "XDG_CONFIG_HOME": "xdg_config_home",
            "XDG_CACHE_HOME": "xdg_cache_home", "XDG_DATA_HOME": "xdg_data_home",
            "XDG_STATE_HOME": "xdg_state_home", "XDG_RUNTIME_DIR": "xdg_runtime_dir",
            "XDG_CONFIG_DIRS": "xdg_config_dirs", "XDG_DATA_DIRS": "xdg_data_dirs",
            "TMPDIR": "tmpdir", "TMP": "tmpdir", "TEMP": "tmpdir",
        }
        entries = dict(self.isolation_profile_value["process_environment"]["fixed"])
        entries.update({name: roots[role] for name, role in dynamic_map.items()})
        lifecycle = {
            role: {"path": path, "directory_mode_octal": "0700", "empty_before": True,
                   "reuse_allowed": False, "cleanup_confirmed": True}
            for role, path in roots.items()
        }
        combined = self.stdout + self.stderr
        self.receipt = {
            "schema": verifier.SCHEMA,
            "status": verifier.STATUS_SUCCESS,
            "closure_capture_digest": FAKE_CLOSURE_CAPTURE,
            "root": FAKE_ROOT,
            "executable": {"path": FAKE_ROOT + "/bin/wb_command", "pre_sha256": FAKE_SHA, "post_sha256": FAKE_SHA},
            "implementations": {
                "producer_sha256": verifier.digest_file(self.producer_path),
                "isolation_verifier_sha256": verifier.digest_file(self.isolation_verifier_path),
            },
            "profiles": {
                "execution_capsule_sha256": verifier.digest_file(self.execution_profile),
                "invocation_isolation_sha256": verifier.digest_file(self.isolation_profile_path),
            },
            "command": {
                "argv": [FAKE_ROOT + "/bin/wb_command", "-version"], "exit_code": 0,
                "cwd": roots["cwd"], "stdin": "devnull", "close_fds": True,
                "pass_fds": [], "umask_octal": "0077",
            },
            "environment": {
                "stage": "root-main-program-execve", "inherit_host_environment": False,
                "entries": entries,
                "entry_environment_sha256": verifier.digest_bytes(verifier.canonical_json_bytes(entries)),
            },
            "isolation": {"roots": lifecycle, "cleanup_confirmed": True},
            "diagnostics": {
                "cpu_vendor": "GenuineIntel", "cpu_family": "6", "cpu_model": "1",
                "cpu_stepping": "1", "cpu_flags_digest": "sha256:" + "c" * 64,
                "kernel_release": "test-kernel",
            },
            "stdout": {"path": "raw/wb-version.stdout", "byte_length": len(self.stdout), "sha256": verifier.digest_bytes(self.stdout)},
            "stderr": {"path": "raw/wb-version.stderr", "byte_length": 0, "sha256": verifier.digest_bytes(b"")},
            "version_output": {
                "aggregation": "stdout-then-stderr-v1", "byte_length": len(combined),
                "sha256": verifier.digest_bytes(combined), "parsed_version": "2.1.0",
            },
            "authority": {key: False for key in verifier.RECEIPT_AUTHORITY_KEYS},
            "capture_digest": "",
        }
        self.write_receipt()

    def tearDown(self):
        self.temp.cleanup()

    def write_receipt(self, recompute=True):
        if recompute:
            self.receipt["capture_digest"] = verifier.digest_bytes(
                verifier.canonical_json_bytes({k: v for k, v in self.receipt.items() if k != "capture_digest"})
            )
        (self.capture / "receipt.json").write_bytes(verifier.canonical_json_bytes(self.receipt) + b"\n")

    def verify(self):
        with mock.patch.object(verifier.isolation_verifier, "load") as load, \
             mock.patch.object(verifier.isolation_verifier, "verify_profile") as verify_profile:
            load.side_effect = [parent_profile(), self.isolation_profile_value]
            verify_profile.return_value = self.isolation_profile_value
            return verifier.verify_receipt(
                self.capture,
                expected_parent=self.parent,
                execution_profile_path=self.execution_profile,
                isolation_profile_path=self.isolation_profile_path,
                isolation_verifier_path=self.isolation_verifier_path,
                producer_path=self.producer_path,
            )

    def semantic_tamper(self, mutator):
        mutator(self.receipt)
        self.write_receipt(recompute=True)
        with self.assertRaises(verifier.VerificationError):
            self.verify()

    def test_valid_receipt(self): self.assertEqual(self.verify()["version_output"]["parsed_version"], "2.1.0")
    def test_root_substitution_rejected(self): self.semantic_tamper(lambda r: r.__setitem__("root", FAKE_ROOT.replace("connectome", "other")))
    def test_executable_path_substitution_rejected(self): self.semantic_tamper(lambda r: r["executable"].__setitem__("path", FAKE_ROOT + "/bin/other"))
    def test_pre_hash_substitution_rejected(self): self.semantic_tamper(lambda r: r["executable"].__setitem__("pre_sha256", "sha256:" + "d"*64))
    def test_post_hash_substitution_rejected(self): self.semantic_tamper(lambda r: r["executable"].__setitem__("post_sha256", "sha256:" + "d"*64))
    def test_argv_substitution_rejected(self): self.semantic_tamper(lambda r: r["command"].__setitem__("argv", [FAKE_ROOT + "/bin/wb_command", "--version"]))
    def test_nonzero_exit_rejected(self): self.semantic_tamper(lambda r: r["command"].__setitem__("exit_code", 2))
    def test_host_path_reintroduction_rejected(self): self.semantic_tamper(lambda r: r["environment"]["entries"].__setitem__("PATH", "/usr/bin"))
    def test_pwd_cwd_mismatch_rejected(self): self.semantic_tamper(lambda r: r["environment"]["entries"].__setitem__("PWD", "/tmp/other"))
    def test_temp_alias_mismatch_rejected(self): self.semantic_tamper(lambda r: r["environment"]["entries"].__setitem__("TEMP", "/tmp/other"))
    def test_root_reuse_rejected(self): self.semantic_tamper(lambda r: r["isolation"]["roots"]["home"].__setitem__("reuse_allowed", True))
    def test_uncleaned_root_rejected(self): self.semantic_tamper(lambda r: r["isolation"]["roots"]["cwd"].__setitem__("cleanup_confirmed", False))
    def test_duplicate_root_paths_rejected(self): self.semantic_tamper(lambda r: r["isolation"]["roots"]["home"].__setitem__("path", r["isolation"]["roots"]["cwd"]["path"]))

    def test_stdout_tamper_rejected(self):
        (self.capture / "raw/wb-version.stdout").write_bytes(b"Version: 9.9.9\n")
        with self.assertRaises(verifier.VerificationError): self.verify()

    def test_version_digest_tamper_rejected(self): self.semantic_tamper(lambda r: r["version_output"].__setitem__("sha256", "sha256:" + "e"*64))
    def test_parsed_version_tamper_rejected(self): self.semantic_tamper(lambda r: r["version_output"].__setitem__("parsed_version", "2.2.0"))
    def test_authority_escalation_rejected(self): self.semantic_tamper(lambda r: r["authority"].__setitem__("workbench_execution_qualified", True))

    def test_capture_digest_tamper_rejected(self):
        self.receipt["capture_digest"] = "sha256:" + "f"*64
        self.write_receipt(recompute=False)
        with self.assertRaises(verifier.VerificationError): self.verify()

    def test_extra_inventory_file_rejected(self):
        (self.capture / "extra").write_text("x")
        with self.assertRaises(verifier.VerificationError): self.verify()

    def test_profile_digest_substitution_rejected(self): self.semantic_tamper(lambda r: r["profiles"].__setitem__("invocation_isolation_sha256", "sha256:" + "1"*64))
    def test_environment_digest_substitution_rejected(self): self.semantic_tamper(lambda r: r["environment"].__setitem__("entry_environment_sha256", "sha256:" + "2"*64))
    def test_diagnostic_flags_digest_shape_rejected(self): self.semantic_tamper(lambda r: r["diagnostics"].__setitem__("cpu_flags_digest", "bad"))

    def test_unknown_top_level_field_rejected(self):
        self.receipt["qualified"] = True
        self.write_receipt(recompute=True)
        with self.assertRaises(verifier.VerificationError): self.verify()


if __name__ == "__main__":
    unittest.main()
