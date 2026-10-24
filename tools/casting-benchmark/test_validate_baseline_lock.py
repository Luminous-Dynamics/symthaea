"""Fail-closed tests for the external casting baseline lock validator."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from validate_baseline_lock import loads_no_duplicate_keys, validate_lock

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "docs/research/CASTING_EXTERNAL_BASELINE_LOCK_V1.json"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def make_execution_ready_fixture(manifest: dict) -> dict:
    ready = copy.deepcopy(manifest)
    resolved_ids = {pin["baseline_id"] for pin in ready["method_pins"]}
    for pin in ready["method_pins"]:
        pin["pin_status"] = "locked_for_execution"
        pin["execution_eligible"] = True
    for item in ready["unresolved_baselines"]:
        baseline_id = item["baseline_id"]
        if baseline_id in resolved_ids:
            continue
        ready["method_pins"].append({
            "baseline_id": baseline_id,
            "track": "test_fixture",
            "package": "fixture",
            "package_version": "1.0.0",
            "source_tag": "fixture-v1",
            "source_commit": "a" * 40,
            "source_path": "tests/fixture.py",
            "source_blob_sha1": "b" * 40,
            "symbol": "fixture_entrypoint",
            "pin_status": "locked_for_execution",
            "execution_eligible": True,
        })
    ready["unresolved_baselines"] = []
    ready["environment_lock"] = {
        "status": "locked",
        "lockfile_path": "tests/fixtures/environment.lock",
        "lockfile_sha256": "c" * 64,
        "python_version": "3.13.12",
        "framework_versions": {
            "torch": "2.4.1",
            "gpytorch": "1.15.2",
            "linear_operator": "0.6.1",
        },
        "platform": "linux-x86_64",
        "device": "cpu",
        "dependency_artifact_hashes": {"fixture-package": "d" * 64},
        "execution_smoke_test": "passed",
        "note": "Test fixture only; not an actual environment.",
    }
    ready["execution_blockers"] = []
    ready["status"] = "locked_for_execution"
    ready["execution_eligible"] = True
    return ready


class BaselineLockValidatorTests(unittest.TestCase):
    def test_current_manifest_is_structurally_valid_but_not_execution_ready(self) -> None:
        manifest = load_manifest()
        self.assertEqual(validate_lock(manifest), [])
        self.assertFalse(manifest["execution_eligible"])
        self.assertEqual(manifest["environment_lock"]["status"], "not_locked")
        self.assertEqual(len(manifest["method_pins"]), 5)
        self.assertEqual(len(manifest["unresolved_baselines"]), 7)

    def test_execution_mode_fails_closed_until_all_gates_are_met(self) -> None:
        errors = validate_lock(load_manifest(), for_execution=True)
        self.assertTrue(any("execution gate blocked" in error for error in errors))

    def test_duplicate_json_keys_are_rejected_at_any_depth(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            loads_no_duplicate_keys('{"outer":{"x":1,"x":2}}')

    def test_release_artifact_requires_sha256(self) -> None:
        manifest = load_manifest()
        manifest["candidate_release"]["release_artifacts"]["wheel"]["sha256"] = "bad"
        self.assertTrue(any("release_artifacts.wheel.sha256" in error for error in validate_lock(manifest)))

    def test_source_file_pin_requires_git_blob_sha1(self) -> None:
        manifest = load_manifest()
        manifest["method_pins"][0]["source_blob_sha1"] = "bad"
        self.assertTrue(any("source_blob_sha1" in error for error in validate_lock(manifest)))

    def test_method_pin_must_match_candidate_release(self) -> None:
        manifest = load_manifest()
        manifest["method_pins"][0]["source_commit"] = "e" * 40
        self.assertTrue(any("source_commit must match candidate_release.source_commit" in error
                            for error in validate_lock(manifest)))

    def test_unlocked_source_pin_cannot_claim_execution_eligibility(self) -> None:
        manifest = load_manifest()
        manifest["method_pins"][0]["execution_eligible"] = True
        self.assertTrue(any("cannot be execution-eligible while its environment is unlocked" in error
                            for error in validate_lock(manifest)))

    def test_unknown_keys_fail_closed(self) -> None:
        manifest = load_manifest()
        manifest["trust_me"] = True
        self.assertTrue(any("unknown keys: trust_me" in error for error in validate_lock(manifest)))

    def test_missing_baseline_coverage_is_rejected(self) -> None:
        manifest = load_manifest()
        manifest["unresolved_baselines"].pop()
        self.assertTrue(any("baseline coverage mismatch" in error for error in validate_lock(manifest)))

    def test_baseline_cannot_be_both_pinned_and_unresolved(self) -> None:
        manifest = load_manifest()
        manifest["unresolved_baselines"].append({
            "baseline_id": "botorch_mf_hvkg",
            "status": "not_pinned",
            "blocker": "test duplicate",
        })
        self.assertTrue(any("both pinned and unresolved" in error for error in validate_lock(manifest)))

    def test_execution_eligible_true_is_rejected_while_blockers_remain(self) -> None:
        manifest = load_manifest()
        manifest["execution_eligible"] = True
        errors = validate_lock(manifest)
        self.assertTrue(any("contradicts unresolved pins" in error for error in errors))

    def test_fully_locked_fixture_satisfies_execution_gate(self) -> None:
        ready = make_execution_ready_fixture(load_manifest())
        self.assertEqual(validate_lock(ready, for_execution=True), [])

    def test_wrong_protocol_is_rejected(self) -> None:
        manifest = load_manifest()
        manifest["protocol_id"] = "WRONG"
        self.assertTrue(any("protocol_id must" in error for error in validate_lock(manifest)))

    def test_method_ids_are_covered_exactly_once(self) -> None:
        manifest = load_manifest()
        manifest["required_method_ids"].append("not_a_real_baseline")
        self.assertTrue(any("baseline coverage mismatch" in error for error in validate_lock(manifest)))


if __name__ == "__main__":
    unittest.main()
