#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adversarial self-tests for the independent RSK qualification verifier."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

GENERATOR = Path(__file__).with_name("rsk_qualification.py")
VERIFIER = Path(__file__).with_name("verify_rsk_qualification.py")


class ReceiptVerifierTests(unittest.TestCase):
    def make_repo(self) -> tuple[Path, dict[str, str]]:
        root = Path(tempfile.mkdtemp(prefix="rsk-receipt-verifier-test-"))
        self.addCleanup(shutil.rmtree, root, True)
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.email", "rsk-test@example.invalid"], cwd=root, check=True)
        subprocess.run(["git", "config", "user.name", "RSK Receipt Test"], cwd=root, check=True)

        for rel in ("scripts", ".github/workflows", "docs/architecture/replicator-safety", "fakebin"):
            (root / rel).mkdir(parents=True, exist_ok=True)
        shutil.copy2(GENERATOR, root / "scripts/rsk_qualification.py")
        shutil.copy2(VERIFIER, root / "scripts/verify_rsk_qualification.py")
        (root / "Cargo.lock").write_text("")
        (root / ".gitignore").write_text("/target/\n")
        (root / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.96.0"\n')
        (root / ".github/workflows/rsk-safety.yml").write_text("name: test\n")
        (root / "scripts/check-class-a-changes.sh").write_text("#!/bin/sh\n")
        (root / "scripts/test_rsk_qualification.py").write_text("# fixture\n")
        (
            root
            / "docs/architecture/replicator-safety/RSK_PRODUCTION_ADMISSION_GATES_V0_1.md"
        ).write_text("Production admission status: DENIED / NOT YET ELIGIBLE.\n")

        fakebin = root / "fakebin"
        (fakebin / "rustc").write_text('#!/bin/sh\necho "rustc 1.96.0 (fake)"\n')
        (fakebin / "rustfmt").write_text('#!/bin/sh\necho "rustfmt 1.96.0"\n')
        (fakebin / "cargo").write_text(
            "#!/bin/sh\n"
            'if [ "$1" = "--version" ]; then echo "cargo fake"; exit 0; fi\n'
            'if [ "$1" = "clippy" ] && [ "$2" = "--version" ]; then echo "clippy fake"; exit 0; fi\n'
            "echo fake-success\n"
            "exit 0\n"
        )
        for tool in fakebin.iterdir():
            tool.chmod(0o755)

        subprocess.run(["git", "add", "."], cwd=root, check=True)
        subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
        env = os.environ.copy()
        env["PATH"] = str(fakebin) + os.pathsep + env.get("PATH", "")
        return root, env

    def generate(self, root: Path, env: dict[str, str], phase: str = "core") -> Path:
        proc = subprocess.run(
            ["python3", "scripts/rsk_qualification.py", "--phase", phase, "--output-dir", "target/out"],
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)
        return root / "target/out/receipt.json"

    def verify(self, root: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "python3",
                "scripts/verify_rsk_qualification.py",
                "target/out/receipt.json",
                "--repo",
                ".",
                "--require-current-subject",
            ],
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )

    @staticmethod
    def receipt(root: Path) -> dict:
        return json.loads((root / "target/out/receipt.json").read_text())

    @staticmethod
    def rewrite_receipt(root: Path, receipt: dict) -> None:
        receipt.pop("receipt_sha256", None)
        canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(canonical).hexdigest()
        receipt["receipt_sha256"] = digest
        (root / "target/out/receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        (root / "target/out/receipt.sha256").write_text(digest + "\n")

    def test_valid_receipt_and_logs_verify_against_exact_checkout(self) -> None:
        root, env = self.make_repo()
        self.generate(root, env)
        proc = self.verify(root, env)
        self.assertEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("verification: PASS", proc.stdout)

    def test_receipt_tamper_without_digest_update_is_rejected(self) -> None:
        root, env = self.make_repo()
        self.generate(root, env)
        receipt = self.receipt(root)
        receipt["production_admission"] = "ADMITTED"
        (root / "target/out/receipt.json").write_text(json.dumps(receipt))
        proc = self.verify(root, env)
        self.assertNotEqual(proc.returncode, 0, proc.stdout)

    def test_log_tamper_is_rejected(self) -> None:
        root, env = self.make_repo()
        self.generate(root, env)
        (root / "target/out/logs/check-authority.log").write_text("tampered\n")
        proc = self.verify(root, env)
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("log digest mismatch", proc.stdout)

    def test_rehashed_receipt_cannot_omit_required_command(self) -> None:
        root, env = self.make_repo()
        self.generate(root, env)
        receipt = self.receipt(root)
        receipt["commands"] = receipt["commands"][:-1]
        self.rewrite_receipt(root, receipt)
        proc = self.verify(root, env)
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("command count mismatch", proc.stdout)

    def test_rehashed_receipt_cannot_lie_about_pin_match(self) -> None:
        root, env = self.make_repo()
        self.generate(root, env)
        receipt = self.receipt(root)
        receipt["toolchain"]["pin_match"] = False
        self.rewrite_receipt(root, receipt)
        proc = self.verify(root, env)
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("pin_match disagrees", proc.stdout)

    def test_rehashed_receipt_cannot_lie_about_input_unchanged(self) -> None:
        root, env = self.make_repo()
        self.generate(root, env)
        receipt = self.receipt(root)
        receipt["inputs"]["unchanged"] = False
        self.rewrite_receipt(root, receipt)
        proc = self.verify(root, env)
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("inputs.unchanged disagrees", proc.stdout)

    def test_log_path_traversal_is_rejected_even_after_rehash(self) -> None:
        root, env = self.make_repo()
        self.generate(root, env)
        receipt = self.receipt(root)
        receipt["commands"][0]["log_path"] = "../outside.log"
        self.rewrite_receipt(root, receipt)
        proc = self.verify(root, env)
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("parent traversal forbidden", proc.stdout)

    def test_current_input_substitution_is_rejected(self) -> None:
        root, env = self.make_repo()
        self.generate(root, env)
        (root / "rust-toolchain.toml").write_text('[toolchain]\nchannel = "1.97.0"\n')
        proc = self.verify(root, env)
        self.assertNotEqual(proc.returncode, 0, proc.stdout)
        self.assertIn("current input hash mismatch", proc.stdout)


if __name__ == "__main__":
    unittest.main()
