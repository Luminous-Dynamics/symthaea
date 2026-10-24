"""Tests for the fail-closed casting scenario pre-run source lock."""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from validate_scenario_prerun_lock import git_blob_sha1, validate_lock

REQUIRED_PATHS = [
    "tools/casting-benchmark/multifidelity_stress.py",
    "tools/casting-benchmark/multifidelity_scenario_suite.py",
    "tools/casting-benchmark/multifidelity_scenario_policy_bakeoff.py",
    "tools/casting-benchmark/score_scenario_policy_bakeoff.py",
    "docs/research/CASTING_MULTIFIDELITY_SCENARIO_SUITE_V1.json",
    "docs/research/CASTING_MULTIFIDELITY_POLICY_SCENARIOS_V1.json",
    "tools/casting-benchmark/test_score_scenario_policy_bakeoff.py",
    "tools/casting-benchmark/test_multifidelity_scenario_policy_bakeoff.py",
]


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


class PrerunLockTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        subprocess.check_call(["git", "init", "-q", str(self.root)])
        git(self.root, "config", "user.email", "casting-lock-tests@example.invalid")
        git(self.root, "config", "user.name", "Casting Lock Tests")
        self.expected: list[dict[str, str]] = []
        for index, relative in enumerate(REQUIRED_PATHS):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"frozen fixture {index}\n", encoding="utf-8")
            self.expected.append({
                "path": relative,
                "git_blob_sha1": git_blob_sha1(path.read_bytes()),
            })
        git(self.root, "add", ".")
        git(self.root, "commit", "-qm", "freeze synthetic fixture sources")
        self.source_commit = git(self.root, "rev-parse", "HEAD")
        self.lock_path = self.root / "docs/research/CASTING_MULTIFIDELITY_POLICY_PRERUN_LOCK_V1.json"
        manifest = {
            "schema_version": "1.0.0",
            "lock_id": "CASTING-MF-POLICY-SCENARIOS-PRERUN-001",
            "title": "test lock",
            "status": "PRE_RUN_SOURCE_FREEZE_NOT_EXECUTION_EVIDENCE",
            "source_commit": self.source_commit,
            "hash_convention": "Git blob SHA-1",
            "locked_files": self.expected,
            "preregistration": {"claim_ceiling": "SYNTHETIC_TRACE_REPLAYED_NOT_SOTA_EVIDENCE"},
        }
        self.lock_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_valid_lock_reports_exact_source_set(self) -> None:
        report = validate_lock(self.root, self.lock_path)
        self.assertEqual(report["status"], "PRE_RUN_LOCK_VERIFIED_NOT_EXECUTION_EVIDENCE")
        self.assertEqual(report["locked_file_count"], 8)
        self.assertEqual(report["locked_source_commit"], self.source_commit)
        self.assertEqual(report["lock_sha256"], hashlib.sha256(self.lock_path.read_bytes()).hexdigest())

    def test_modified_worktree_source_is_refused(self) -> None:
        (self.root / REQUIRED_PATHS[0]).write_text("silently changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "dirty"):
            validate_lock(self.root, self.lock_path)

    def test_lock_entry_with_wrong_blob_is_refused(self) -> None:
        data = json.loads(self.lock_path.read_text(encoding="utf-8"))
        data["locked_files"][0]["git_blob_sha1"] = "0" * 40
        self.lock_path.write_text(json.dumps(data) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "pinned source commit"):
            validate_lock(self.root, self.lock_path)

    def test_duplicate_path_is_refused(self) -> None:
        data = json.loads(self.lock_path.read_text(encoding="utf-8"))
        data["locked_files"].append(dict(data["locked_files"][0]))
        self.lock_path.write_text(json.dumps(data) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate locked path"):
            validate_lock(self.root, self.lock_path)

    def test_unknown_top_level_key_is_refused(self) -> None:
        data = json.loads(self.lock_path.read_text(encoding="utf-8"))
        data["allow_mutable_latest"] = True
        self.lock_path.write_text(json.dumps(data) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "schema mismatch"):
            validate_lock(self.root, self.lock_path)

    def test_wrong_source_commit_is_refused(self) -> None:
        data = json.loads(self.lock_path.read_text(encoding="utf-8"))
        data["source_commit"] = "0" * 40
        self.lock_path.write_text(json.dumps(data) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "git cat-file"):
            validate_lock(self.root, self.lock_path)


if __name__ == "__main__":
    unittest.main()
