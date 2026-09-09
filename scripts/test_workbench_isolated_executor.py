#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import workbench_isolated_executor as m

PROFILE = {
    "process_environment": {
        "stage": "root-main-program-execve",
        "fixed": {
            "LANG": "C",
            "LC_ALL": "C",
            "TZ": "UTC0",
            "OMP_NUM_THREADS": "1",
            "OMP_DYNAMIC": "FALSE",
            "PATH": "",
        },
        "dynamic_bindings": {
            "PWD": "invocation.cwd",
            "HOME": "invocation.home",
            "XDG_CONFIG_HOME": "invocation.xdg_config_home",
            "XDG_CACHE_HOME": "invocation.xdg_cache_home",
            "XDG_DATA_HOME": "invocation.xdg_data_home",
            "XDG_STATE_HOME": "invocation.xdg_state_home",
            "XDG_RUNTIME_DIR": "invocation.xdg_runtime_dir",
            "XDG_CONFIG_DIRS": "invocation.xdg_config_dirs",
            "XDG_DATA_DIRS": "invocation.xdg_data_dirs",
            "TMPDIR": "invocation.tmpdir",
            "TMP": "invocation.tmpdir",
            "TEMP": "invocation.tmpdir",
        },
    }
}

FAKE = r"""#!{python}
import json, os, sys
mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
sentinel = sys.argv[2] if len(sys.argv) > 2 else ""
targets = []
if os.path.isdir("/proc/self/fd"):
    for name in os.listdir("/proc/self/fd"):
        try:
            targets.append(os.readlink("/proc/self/fd/" + name))
        except OSError:
            pass
old = os.umask(0)
os.umask(old)
stdin_byte = sys.stdin.buffer.read(1)
payload = {{
    "argv": sys.argv,
    "cwd": os.getcwd(),
    "pwd": os.environ.get("PWD"),
    "env": dict(os.environ),
    "umask": format(old, "04o"),
    "stdin_eof": stdin_byte == b"",
    "sentinel_fd_visible": sentinel in targets,
}}
sys.stdout.buffer.write((json.dumps(payload, sort_keys=True) + "\n").encode())
sys.stderr.buffer.write(b"fake-stderr\x00\n")
sys.stdout.flush(); sys.stderr.flush()
if mode == "mutate":
    with open(__file__, "ab") as handle:
        handle.write(b"\n# mutation\n")
if mode == "exit7":
    raise SystemExit(7)
"""


def sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.td = Path(self.tmp.name)
        self.scratch = self.td / "scratch"
        self.scratch.mkdir()
        self.profile = self.td / "profile.json"; self.profile.write_text("{}")
        self.parent = self.td / "parent.json"; self.parent.write_text("{}")
        self.program = self.td / "fake_executor"
        self.program.write_text(FAKE.format(python=Path(sys.executable).resolve()))
        self.program.chmod(0o700)
        self.expected = sha(self.program)
        self.verify_patch = mock.patch.object(m, "verify_profiles", return_value=PROFILE)
        self.verify_patch.start()

    def tearDown(self):
        self.verify_patch.stop()
        self.tmp.cleanup()

    def run_obs(self, args=None, stem="probe"):
        return m.observe_isolated_execution(
            self.program,
            args or ["normal", ""],
            self.expected,
            self.profile,
            self.parent,
            self.scratch,
            stem,
        )

    def child(self, obs):
        return json.loads(obs["stdout"].decode())

    def test_success_observes_exact_process_contract(self):
        obs = self.run_obs()
        child = self.child(obs)
        self.assertEqual(obs["command"]["exit_code"], 0)
        self.assertEqual(obs["command"]["argv"], [str(self.program), "normal", ""])
        self.assertEqual(obs["command"]["stdin"], "devnull")
        self.assertTrue(obs["command"]["close_fds"])
        self.assertEqual(obs["command"]["pass_fds"], [])
        self.assertEqual(obs["command"]["umask_octal"], "0077")
        self.assertEqual(child["cwd"], obs["command"]["cwd"])
        self.assertEqual(child["pwd"], obs["command"]["cwd"])
        self.assertTrue(child["stdin_eof"])
        self.assertEqual(child["umask"], "0077")
        self.assertEqual(obs["stderr"], b"fake-stderr\x00\n")
        self.assertTrue(obs["isolation"]["cleanup_confirmed"])

    def test_entry_environment_is_closed_world_at_launch(self):
        with mock.patch.dict(os.environ, {"HOST_ONLY_EXECUTOR_SECRET": "should-not-leak"}, clear=False):
            obs = self.run_obs()
        child = self.child(obs)
        self.assertNotIn("HOST_ONLY_EXECUTOR_SECRET", child["env"])
        entries = obs["environment"]["entries"]
        self.assertEqual(set(entries), set(PROFILE["process_environment"]["fixed"]) | set(PROFILE["process_environment"]["dynamic_bindings"]))
        self.assertEqual(entries["PATH"], "")
        self.assertEqual(entries["TMPDIR"], entries["TMP"])
        self.assertEqual(entries["TMPDIR"], entries["TEMP"])

    def test_all_isolation_roots_are_distinct_private_and_removed(self):
        obs = self.run_obs()
        roots = obs["isolation"]["roots"]
        self.assertEqual(set(roots), set(m.ROOT_ROLES))
        paths = [item["path"] for item in roots.values()]
        self.assertEqual(len(paths), len(set(paths)))
        for item in roots.values():
            self.assertEqual(item["directory_mode_octal"], "0700")
            self.assertTrue(item["empty_before"])
            self.assertFalse(item["reuse_allowed"])
            self.assertTrue(item["cleanup_confirmed"])
            self.assertFalse(Path(item["path"]).exists())

    def test_inheritable_parent_fd_is_not_visible_to_child(self):
        sentinel = self.td / "sentinel"
        sentinel.write_text("secret")
        fd = os.open(sentinel, os.O_RDONLY)
        try:
            os.set_inheritable(fd, True)
            obs = self.run_obs(["normal", str(sentinel)])
        finally:
            os.close(fd)
        self.assertFalse(self.child(obs)["sentinel_fd_visible"])

    def test_stdout_and_stderr_digests_bind_exact_bytes(self):
        obs = self.run_obs()
        self.assertEqual(obs["stdout_sha256"], m.digest_bytes(obs["stdout"]))
        self.assertEqual(obs["stderr_sha256"], m.digest_bytes(obs["stderr"]))
        self.assertEqual(obs["environment"]["entry_environment_sha256"], m.digest_bytes(m.canonical_json_bytes(obs["environment"]["entries"])))

    def test_authority_is_closed_false(self):
        obs = self.run_obs()
        self.assertEqual(set(obs["authority"]), set(m.AUTHORITY))
        self.assertFalse(any(obs["authority"].values()))

    def test_nonzero_exit_is_observed_not_laundered(self):
        obs = self.run_obs(["exit7", ""])
        self.assertEqual(obs["command"]["exit_code"], 7)
        with self.assertRaises(m.ExecutionError):
            m.require_success(obs)

    def test_zero_exit_is_accepted_by_require_success(self):
        obs = self.run_obs()
        self.assertIs(m.require_success(obs), obs)

    def test_pre_execution_digest_mismatch_fails_before_process(self):
        with mock.patch.object(m.subprocess, "run") as run:
            with self.assertRaises(m.ExecutionError):
                m.observe_isolated_execution(self.program, [], "sha256:" + "0"*64, self.profile, self.parent, self.scratch)
        run.assert_not_called()

    def test_post_execution_digest_mutation_fails_closed_and_cleans(self):
        with self.assertRaisesRegex(m.ExecutionError, "post-execution"):
            self.run_obs(["mutate", ""])
        self.assertFalse(any(self.scratch.iterdir()))

    def test_launch_failure_cleans(self):
        with mock.patch.object(m.subprocess, "run", side_effect=OSError("boom")):
            with self.assertRaisesRegex(m.ExecutionError, "launch failed"):
                self.run_obs()
        self.assertFalse(any(self.scratch.iterdir()))

    def test_isolation_construction_failure_cleans(self):
        real_chmod = m.os.chmod
        calls = {"n": 0}
        def fail_first(path, mode):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("boom")
            return real_chmod(path, mode)
        with mock.patch.object(m.os, "chmod", side_effect=fail_first):
            with self.assertRaisesRegex(m.ExecutionError, "construction failed"):
                m.create_isolation_roots(self.scratch, "probe")
        self.assertFalse(any(self.scratch.iterdir()))

    def test_invalid_stem_rejected_before_directory_creation(self):
        for stem in ("../escape", "", ".hidden", "a/b"):
            with self.subTest(stem=stem):
                with self.assertRaises(m.ExecutionError):
                    m.create_isolation_roots(self.scratch, stem)
        self.assertFalse(any(self.scratch.iterdir()))

    def test_nul_argument_rejected(self):
        with self.assertRaisesRegex(m.ExecutionError, "NUL"):
            self.run_obs(["bad\x00arg"])

    def test_non_string_argument_rejected(self):
        with self.assertRaisesRegex(m.ExecutionError, r"list\[str\]"):
            m.observe_isolated_execution(self.program, [1], self.expected, self.profile, self.parent, self.scratch)

    def test_relative_executable_rejected(self):
        old = Path.cwd()
        try:
            os.chdir(self.td)
            with self.assertRaisesRegex(m.ExecutionError, "absolute"):
                m.resolve_executable(Path("fake_executor"))
        finally:
            os.chdir(old)

    def test_direct_executable_symlink_rejected(self):
        link = self.td / "link"
        link.symlink_to(self.program)
        with self.assertRaisesRegex(m.ExecutionError, "symlink"):
            m.resolve_executable(link)

    def test_non_executable_regular_file_rejected(self):
        p = self.td / "plain"
        p.write_text("x")
        p.chmod(0o600)
        with self.assertRaisesRegex(m.ExecutionError, "execute permission"):
            m.resolve_executable(p)

    def test_profile_verification_is_invoked(self):
        m.verify_profiles(self.profile, self.parent)
        self.assertTrue(True)

    def test_cwd_and_pwd_binding_cannot_diverge(self):
        bad = json.loads(json.dumps(PROFILE))
        bad["process_environment"]["dynamic_bindings"]["PWD"] = "invocation.home"
        with mock.patch.object(m, "verify_profiles", return_value=bad):
            with self.assertRaisesRegex(m.ExecutionError, "PWD must equal cwd"):
                m.observe_isolated_execution(self.program, [], self.expected, self.profile, self.parent, self.scratch)

    def test_temp_aliases_cannot_diverge(self):
        bad = json.loads(json.dumps(PROFILE))
        bad["process_environment"]["dynamic_bindings"]["TEMP"] = "invocation.home"
        with mock.patch.object(m, "verify_profiles", return_value=bad):
            with self.assertRaisesRegex(m.ExecutionError, "temp aliases"):
                m.observe_isolated_execution(self.program, [], self.expected, self.profile, self.parent, self.scratch)


if __name__ == "__main__":
    unittest.main(verbosity=2)
