#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Self-tests for the RSK qualification capsule using only Python and Git."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT_UNDER_TEST = Path(__file__).with_name("rsk_qualification.py")


class QualificationHarnessTests(unittest.TestCase):
    def make_repo(self, rust_version: str = "1.96.0") -> tuple[Path, dict[str, str]]:
        root = Path(tempfile.mkdtemp(prefix="rsk-qualification-test-"))
        self.addCleanup(shutil.rmtree, root, True)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "rsk-test@example.invalid"],
            cwd=root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "RSK Qualification Test"],
            cwd=root,
            check=True,
        )

        for rel in (
            "scripts",
            ".github/workflows",
            "docs/architecture/replicator-safety",
            "fakebin",
        ):
            (root / rel).mkdir(parents=True, exist_ok=True)

        shutil.copy2(SCRIPT_UNDER_TEST, root / "scripts/rsk_qualification.py")
        (root / "Cargo.lock").write_text("")
        (root / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.96.0"\n')
        (root / ".github/workflows/rsk-safety.yml").write_text("name: test\n")
        (root / "scripts/check-class-a-changes.sh").write_text("#!/bin/sh\n")
        (
            root
            / "docs/architecture/replicator-safety/RSK_PRODUCTION_ADMISSION_GATES_V0_1.md"
        ).write_text("Production admission status: DENIED / NOT YET ELIGIBLE.\n")

        fakebin = root / "fakebin"
        (fakebin / "rustc").write_text(
            f'#!/bin/sh\necho "rustc {rust_version} (fake)"\n'
        )
        (fakebin / "rustfmt").write_text(
            f'#!/bin/sh\necho "rustfmt {rust_version}"\n'
        )
        (fakebin / "cargo").write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then echo "cargo fake"; exit 0; fi\n'
            'if [ "$1" = "clippy" ] && [ "$2" = "--version" ]; then '
            'echo "clippy fake"; exit 0; fi\n'
            "exit 0\n"
        )
        for tool in fakebin.iterdir():
            tool.chmod(0o755)

        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)

        env = os.environ.copy()
        env["PATH"] = str(fakebin) + os.pathsep + env.get("PATH", "")
        return root, env

    def run_capsule(
        self, root: Path, env: dict[str, str], *args: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["python3", "scripts/rsk_qualification.py", *args],
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    @staticmethod
    def read_receipt(root: Path) -> dict:
        return json.loads((root / "out/receipt.json").read_text())

    def test_clean_matching_toolchain_is_admissible_for_fake_successful_gate(self) -> None:
        root, env = self.make_repo("1.96.0")
        proc = self.run_capsule(
            root, env, "--phase", "format", "--output-dir", "out"
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        receipt = self.read_receipt(root)
        self.assertEqual(receipt["qualification_status"], "pass-clean")
        self.assertTrue(receipt["admissible_evidence"])
        self.assertTrue(receipt["toolchain"]["pin_match"])
        self.assertEqual(
            receipt["production_admission"], "DENIED / NOT YET ELIGIBLE"
        )

    def test_wrong_rust_toolchain_fails_even_when_fake_commands_succeed(self) -> None:
        root, env = self.make_repo("1.95.0")
        proc = self.run_capsule(
            root, env, "--phase", "format", "--output-dir", "out"
        )
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        receipt = self.read_receipt(root)
        self.assertEqual(receipt["qualification_status"], "fail-toolchain-mismatch")
        self.assertFalse(receipt["admissible_evidence"])
        self.assertFalse(receipt["toolchain"]["pin_match"])

    def test_dirty_worktree_is_blocked_before_qualification(self) -> None:
        root, env = self.make_repo("1.96.0")
        (root / "dirty.txt").write_text("untracked\n")
        proc = self.run_capsule(
            root, env, "--phase", "format", "--output-dir", "out"
        )
        self.assertEqual(proc.returncode, 2, proc.stdout)
        receipt = self.read_receipt(root)
        self.assertEqual(receipt["qualification_status"], "blocked-dirty-worktree")
        self.assertFalse(receipt["admissible_evidence"])
        self.assertEqual(receipt["commands"], [])

    def test_receipt_digest_matches_canonical_payload_without_digest_field(self) -> None:
        root, env = self.make_repo("1.96.0")
        proc = self.run_capsule(
            root, env, "--phase", "format", "--output-dir", "out"
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        receipt = self.read_receipt(root)
        claimed = receipt.pop("receipt_sha256")
        canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), claimed)
        self.assertEqual(
            (root / "out/receipt.sha256").read_text().strip(), claimed
        )


if __name__ == "__main__":
    unittest.main()
