#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import workbench_minimal_isolated_executor as m


def sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


class MinimalIsolatedExecutorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.td = Path(self.temp.name)
        self.isolation_parent = self.td / "isolation-parent"
        self.isolation_parent.mkdir(mode=0o700)
        self.output_dir = self.td / "output"
        self.output_dir.mkdir(mode=0o700)

        self.child = self.td / "child.py"
        self.child.write_text(
            """from __future__ import annotations
import json, os, stat, sys
from pathlib import Path

out = Path(sys.argv[1])
fd_number = int(sys.argv[2])
exit_code = int(sys.argv[3])
expected_dynamic = {
    'PWD': 'cwd', 'HOME': 'home', 'XDG_CONFIG_HOME': 'xdg_config_home',
    'XDG_CACHE_HOME': 'xdg_cache_home', 'XDG_DATA_HOME': 'xdg_data_home',
    'XDG_STATE_HOME': 'xdg_state_home', 'XDG_RUNTIME_DIR': 'xdg_runtime_dir',
    'XDG_CONFIG_DIRS': 'xdg_config_dirs', 'XDG_DATA_DIRS': 'xdg_data_dirs',
    'TMPDIR': 'tmpdir', 'TMP': 'tmpdir', 'TEMP': 'tmpdir',
}
root_state = {}
for variable, role in expected_dynamic.items():
    path = Path(os.environ[variable])
    root_state[variable] = {
        'role': role,
        'absolute': path.is_absolute(),
        'exists': path.is_dir(),
        'mode': oct(stat.S_IMODE(path.stat().st_mode)),
        'entries': sorted(p.name for p in path.iterdir()),
    }
try:
    os.fstat(fd_number)
    inherited_fd_open = True
except OSError:
    inherited_fd_open = False
stdin_bytes = sys.stdin.buffer.read()
umask_file = out / 'umask-created.txt'
with umask_file.open('w') as handle:
    handle.write('x')
record = {
    'argv_tail': sys.argv[4:],
    'cwd': os.getcwd(),
    'environment': dict(os.environ),
    'root_state': root_state,
    'inherited_fd_open': inherited_fd_open,
    'stdin_length': len(stdin_bytes),
    'umask_file_mode': oct(stat.S_IMODE(umask_file.stat().st_mode)),
}
print(json.dumps(record, sort_keys=True, separators=(',', ':')))
print('child-stderr-marker', file=sys.stderr)
raise SystemExit(exit_code)
""",
            encoding="utf-8",
        )

        self.program = self.td / "fake-wb-command"
        self.program.write_text(
            "#!/bin/sh\n"
            f"exec {sys.executable} {self.child} \"$@\"\n",
            encoding="utf-8",
        )
        self.program.chmod(0o700)
        self.program_sha = sha(self.program)

    def tearDown(self):
        self.temp.cleanup()

    def invoke(self, *, exit_code: int = 0, extra: list[str] | None = None):
        inherited = self.td / "inherited-fd"
        inherited.write_text("secret", encoding="utf-8")
        fd = os.open(inherited, os.O_RDONLY)
        high_fd = 200
        os.dup2(fd, high_fd, inheritable=True)
        os.close(fd)
        old_secret = os.environ.get("SYMTHEA_HOST_SECRET")
        os.environ["SYMTHEA_HOST_SECRET"] = "must-not-leak"
        try:
            return m.invoke_isolated(
                self.program,
                [str(self.output_dir), str(high_fd), str(exit_code), *(extra or [])],
                expected_executable_sha256=self.program_sha,
                isolation_parent=self.isolation_parent,
                stem="test-wb",
            )
        finally:
            try:
                os.close(high_fd)
            except OSError:
                pass
            if old_secret is None:
                os.environ.pop("SYMTHEA_HOST_SECRET", None)
            else:
                os.environ["SYMTHEA_HOST_SECRET"] = old_secret

    def test_full_entry_boundary_and_cleanup(self):
        result = self.invoke(extra=["alpha", "beta"])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(result.cleanup_confirmed)
        self.assertEqual(result.pre_executable_sha256, self.program_sha)
        self.assertEqual(result.post_executable_sha256, self.program_sha)
        self.assertEqual(result.stderr, b"child-stderr-marker\n")
        record = json.loads(result.stdout)

        expected_env_keys = set(m.FIXED_ENTRY_ENVIRONMENT) | set(m.DYNAMIC_ENTRY_BINDINGS)
        self.assertEqual(set(record["environment"]), expected_env_keys)
        for key, value in m.FIXED_ENTRY_ENVIRONMENT.items():
            self.assertEqual(record["environment"][key], value)
        self.assertNotIn("SYMTHEA_HOST_SECRET", record["environment"])
        self.assertEqual(record["argv_tail"], ["alpha", "beta"])
        self.assertEqual(record["cwd"], record["environment"]["PWD"])
        self.assertFalse(record["inherited_fd_open"])
        self.assertEqual(record["stdin_length"], 0)
        self.assertEqual(record["umask_file_mode"], "0o600")

        for variable, state in record["root_state"].items():
            self.assertTrue(state["absolute"], variable)
            self.assertTrue(state["exists"], variable)
            self.assertEqual(state["mode"], "0o700", variable)
            # PWD is the cwd root and all roots are empty at process entry. The
            # child itself creates no files inside them before this census.
            self.assertEqual(state["entries"], [], variable)

        self.assertEqual(result.stdin_policy, "devnull")
        self.assertTrue(result.close_fds)
        self.assertEqual(result.pass_fds, ())
        self.assertEqual(result.umask_octal, "0077")
        self.assertEqual(
            result.entry_environment_sha256,
            m.digest_bytes(m.canonical_json_bytes(dict(result.entry_environment))),
        )
        for path in result.isolation_roots.values():
            self.assertFalse(os.path.lexists(path))

    def test_nonzero_child_status_is_retained_not_laundered(self):
        result = self.invoke(exit_code=23)
        self.assertEqual(result.exit_code, 23)
        self.assertTrue(result.cleanup_confirmed)
        self.assertIn(b"child-stderr-marker", result.stderr)

    def test_each_invocation_uses_distinct_nonreused_roots(self):
        first = self.invoke()
        second = self.invoke()
        self.assertNotEqual(
            {str(path) for path in first.isolation_roots.values()},
            {str(path) for path in second.isolation_roots.values()},
        )
        for result in (first, second):
            self.assertTrue(result.cleanup_confirmed)
            self.assertTrue(all(not os.path.lexists(path) for path in result.isolation_roots.values()))

    def test_pre_execution_program_mismatch_fails_before_launch(self):
        marker = self.output_dir / "umask-created.txt"
        with self.assertRaises(m.ExecutionContractError):
            m.invoke_isolated(
                self.program,
                [str(self.output_dir), "200", "0"],
                expected_executable_sha256="sha256:" + "0" * 64,
                isolation_parent=self.isolation_parent,
            )
        self.assertFalse(marker.exists())

    def test_program_mutation_during_execution_rejected(self):
        mutating = self.td / "mutating-wb"
        mutating.write_text(
            "#!/bin/sh\n"
            "printf '# changed\\n' >> \"$0\"\n"
            f"exec {sys.executable} {self.child} \"$@\"\n",
            encoding="utf-8",
        )
        mutating.chmod(0o700)
        expected = sha(mutating)
        with self.assertRaisesRegex(m.ExecutionContractError, "changed during execution"):
            m.invoke_isolated(
                mutating,
                [str(self.output_dir), "200", "0"],
                expected_executable_sha256=expected,
                isolation_parent=self.isolation_parent,
            )
        self.assertNotEqual(sha(mutating), expected)
        self.assertEqual(list(self.isolation_parent.iterdir()), [])

    def test_relative_executable_rejected(self):
        with self.assertRaisesRegex(m.ExecutionContractError, "absolute"):
            m.invoke_isolated(
                Path("fake-wb-command"),
                [],
                expected_executable_sha256=self.program_sha,
                isolation_parent=self.isolation_parent,
            )

    def test_direct_executable_symlink_rejected(self):
        link = self.td / "wb-link"
        link.symlink_to(self.program)
        with self.assertRaisesRegex(m.ExecutionContractError, "symlink"):
            m.invoke_isolated(
                link,
                [],
                expected_executable_sha256=self.program_sha,
                isolation_parent=self.isolation_parent,
            )

    def test_non_regular_executable_rejected(self):
        directory = self.td / "not-a-program"
        directory.mkdir()
        with self.assertRaises(m.ExecutionContractError):
            m.invoke_isolated(
                directory,
                [],
                expected_executable_sha256="sha256:" + "1" * 64,
                isolation_parent=self.isolation_parent,
            )

    def test_invalid_digest_rejected(self):
        with self.assertRaisesRegex(m.ExecutionContractError, "64 lowercase hex"):
            m.invoke_isolated(
                self.program,
                [],
                expected_executable_sha256="SHA256:bad",
                isolation_parent=self.isolation_parent,
            )

    def test_nul_argument_rejected(self):
        with self.assertRaisesRegex(m.ExecutionContractError, "NUL"):
            m.invoke_isolated(
                self.program,
                ["bad\x00arg"],
                expected_executable_sha256=self.program_sha,
                isolation_parent=self.isolation_parent,
            )

    def test_direct_symlink_isolation_parent_rejected(self):
        link = self.td / "parent-link"
        link.symlink_to(self.isolation_parent, target_is_directory=True)
        with self.assertRaisesRegex(m.ExecutionContractError, "symlink"):
            m.invoke_isolated(
                self.program,
                [],
                expected_executable_sha256=self.program_sha,
                isolation_parent=link,
            )

    def test_bad_stem_rejected_and_leaves_no_roots(self):
        with self.assertRaises(m.ExecutionContractError):
            m.invoke_isolated(
                self.program,
                [],
                expected_executable_sha256=self.program_sha,
                isolation_parent=self.isolation_parent,
                stem="bad/stem",
            )
        self.assertEqual(list(self.isolation_parent.iterdir()), [])

    def test_launch_failure_cleans_roots(self):
        not_executable = self.td / "not-executable"
        not_executable.write_text("plain text", encoding="utf-8")
        not_executable.chmod(0o600)
        with self.assertRaisesRegex(m.ExecutionContractError, "launch failed"):
            m.invoke_isolated(
                not_executable,
                [],
                expected_executable_sha256=sha(not_executable),
                isolation_parent=self.isolation_parent,
            )
        self.assertEqual(list(self.isolation_parent.iterdir()), [])

    def test_cleanup_failure_fails_closed_instead_of_returning_result(self):
        with mock.patch.object(
            m,
            "_cleanup_isolation_base",
            side_effect=m.ExecutionContractError("cleanup not confirmed"),
        ):
            with self.assertRaisesRegex(m.ExecutionContractError, "cleanup not confirmed"):
                self.invoke()
        # The patch intentionally prevented cleanup; test teardown owns the temp tree.

    def test_build_entry_environment_requires_exact_roles(self):
        roots = {role: self.td / role for role in m.ROOT_ROLES}
        roots.pop("cwd")
        with self.assertRaisesRegex(m.ExecutionContractError, "exact isolation root roles"):
            m.build_entry_environment(roots)

    def test_environment_contract_has_exact_profile_key_set(self):
        self.assertEqual(
            set(m.FIXED_ENTRY_ENVIRONMENT),
            {"LANG", "LC_ALL", "TZ", "OMP_NUM_THREADS", "OMP_DYNAMIC", "PATH"},
        )
        self.assertEqual(
            set(m.DYNAMIC_ENTRY_BINDINGS),
            {
                "PWD", "HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME",
                "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_RUNTIME_DIR",
                "XDG_CONFIG_DIRS", "XDG_DATA_DIRS", "TMPDIR", "TMP", "TEMP",
            },
        )

    def test_module_has_no_nix_version_cpu_or_scientific_authority_surface(self):
        source = Path(m.__file__).read_text(encoding="utf-8")
        for forbidden in (
            "nix-store", "nix path-info", "-version", "/proc/cpuinfo",
            "fmq010", "neural_alignment", "consciousness_evidence",
            "atlas_correctness", "scientific_execution_qualified",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
