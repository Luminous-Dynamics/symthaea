#!/usr/bin/env python3
"""Integration and adversarial tests for VLEB two-sided oracle-control replay."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import Any

MODULE_PATH = Path(__file__).with_name("verify_oracle_controls.py")
SPEC = importlib.util.spec_from_file_location("vleb_verify_oracle_controls", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import oracle-control verifier")
verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def jbytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def zip_bytes(entries: list[tuple[str, bytes]]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in entries:
            archive.writestr(name, data)
    return output.getvalue()


class OracleControlReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ambient_env_before = os.environ.get("VLEB_AMBIENT_SENTINEL")
        os.environ["VLEB_AMBIENT_SENTINEL"] = "must-not-leak-into-test-commands"
        self.temporary = tempfile.TemporaryDirectory(prefix="vleb-oracle-replay-test-")
        self.root = Path(self.temporary.name)
        self.repo = self.root / "source"
        self.repo.mkdir()
        self.run_git(self.repo, "init", "-q")
        self.run_git(self.repo, "config", "user.name", "VLEB Test")
        self.run_git(self.repo, "config", "user.email", "vleb-tests@example.invalid")
        (self.repo / "app.py").write_text("VALUE = 0\n", encoding="utf-8")
        self.run_git(self.repo, "add", "app.py")
        self.run_git(self.repo, "commit", "-qm", "baseline source")
        self.revision = self.run_git(self.repo, "rev-parse", "HEAD").stdout.strip()

        # Produce a genuine Git patch from the committed source, then restore a clean checkout.
        (self.repo / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
        self.patch_raw = self.run_git(self.repo, "diff", "--", "app.py").stdout.encode("utf-8")
        self.run_git(self.repo, "checkout", "--", "app.py")

        self.bundle_raw = zip_bytes([
            ("test_contract.py", b"""import os
import unittest
import app

class ContractTest(unittest.TestCase):
    def test_expected_behavior(self):
        self.assertEqual(os.environ.get("VLEB_ONLY_DECLARED"), "yes")
        self.assertIsNone(os.environ.get("VLEB_AMBIENT_SENTINEL"))
        self.assertEqual(app.VALUE, 1)

if __name__ == "__main__":
    unittest.main()
""")
        ])
        self.manifest = {
            "schema_version": "vleb-oracle-control-v1",
            "commands": [{
                "command_id": "python-contract",
                "argv": [sys.executable, "-m", "unittest", "test_contract"],
                "environment": {
                    "PATH": os.environ.get("PATH", ""),
                    "HOME": os.environ.get("HOME", str(Path.home())),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "VLEB_ONLY_DECLARED": "yes",
                },
                "working_directory": ".",
                "timeout_seconds": 10,
                "test_start_pattern": r"Ran [1-9]\d* tests?",
                "pass_pattern": r"^OK$",
                "fail_pattern": r"^FAILED(?: \(|$)",
            }],
        }
        self.manifest_raw = jbytes(self.manifest)
        self.evidence = self.root / "evidence"

    def tearDown(self) -> None:
        self.temporary.cleanup()
        if self.ambient_env_before is None:
            os.environ.pop("VLEB_AMBIENT_SENTINEL", None)
        else:
            os.environ["VLEB_AMBIENT_SENTINEL"] = self.ambient_env_before

    @staticmethod
    def run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            ["git", "-C", str(root), "-c", "core.hooksPath=/dev/null", *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        if result.returncode:
            raise AssertionError(f"git {args!r} failed: {result.stdout}")
        return result

    def run_controls(
        self,
        *,
        manifest_raw: bytes | None = None,
        patch_raw: bytes | None = None,
        bundle_raw: bytes | None = None,
        expected_manifest: str | None = None,
        expected_patch: str | None = None,
        expected_bundle: str | None = None,
        expected_verifier: str | None = None,
        source_revision: str | None = None,
        evidence_root: Path | None = None,
    ) -> dict[str, Any]:
        manifest = self.manifest_raw if manifest_raw is None else manifest_raw
        patch = self.patch_raw if patch_raw is None else patch_raw
        bundle = self.bundle_raw if bundle_raw is None else bundle_raw
        return verifier.run_two_sided_controls(
            source_repo=self.repo,
            expected_source_revision=self.revision if source_revision is None else source_revision,
            manifest_raw=manifest,
            expected_manifest_sha256=sha(manifest) if expected_manifest is None else expected_manifest,
            reference_patch_raw=patch,
            expected_reference_patch_sha256=sha(patch) if expected_patch is None else expected_patch,
            test_bundle_raw=bundle,
            expected_test_bundle_sha256=sha(bundle) if expected_bundle is None else expected_bundle,
            expected_verifier_sha256=sha(MODULE_PATH.read_bytes()) if expected_verifier is None else expected_verifier,
            evidence_root=self.evidence if evidence_root is None else evidence_root,
        )

    def test_actual_empty_patch_fails_and_reference_patch_passes(self) -> None:
        result = self.run_controls()
        self.assertTrue(result["valid"], result["issues"])
        self.assertTrue(result["two_sided_gate_passed"])
        self.assertEqual(result["baseline_test_failure_count"], 1)
        self.assertEqual(result["reference_passed_command_count"], 1)
        self.assertTrue(result["analysis_only"])
        self.assertFalse(result["score_eligible"])
        self.assertTrue(result["runner_attestation_required"])
        self.assertFalse(result["network_isolation_independently_verified"])
        self.assertIn("VLEB_ONLY_DECLARED", result["baseline_results"][0]["environment_names"])
        self.assertNotIn("VLEB_AMBIENT_SENTINEL", result["baseline_results"][0]["environment_names"])
        for arm in ("baseline", "reference"):
            checkout = result["isolated_checkout_metadata"][arm]
            self.assertEqual(checkout["head_sha"], self.revision)
            self.assertEqual(checkout["ref_names"], [])
            self.assertEqual(checkout["reachable_commit_count"], 1)
            self.assertFalse(checkout["uses_shared_object_alternates"])
        self.assertEqual(
            self.run_git(self.repo, "status", "--porcelain=v1", "--untracked-files=all").stdout,
            "",
        )

    def test_raw_test_evidence_is_written_and_hashed(self) -> None:
        result = self.run_controls()
        for row in result["baseline_results"] + result["reference_results"]:
            path = self.evidence / row["combined_output_artifact"]
            self.assertTrue(path.is_file())
            self.assertEqual(sha(path.read_bytes()), row["combined_output_sha256"])
        self.assertIn("FAILED", (self.evidence / result["baseline_results"][0]["combined_output_artifact"]).read_text())
        self.assertIn("OK", (self.evidence / result["reference_results"][0]["combined_output_artifact"]).read_text())


    def test_tracked_source_mutation_during_reference_tests_invalidates_gate(self) -> None:
        bundle = zip_bytes([
            ("test_contract.py", b"""import pathlib
import unittest
import app

class ContractTest(unittest.TestCase):
    def test_expected_behavior(self):
        self.assertEqual(app.VALUE, 1)
        pathlib.Path("app.py").write_text("VALUE = 2" + chr(10), encoding="utf-8")
""")
        ])
        result = self.run_controls(bundle_raw=bundle, evidence_root=self.root / "mutated-source-evidence")
        self.assertFalse(result["two_sided_gate_passed"])
        self.assertFalse(result["reference_tree_integrity"]["valid"])
        self.assertTrue(any("tracked source tree changed" in issue for issue in result["issues"]))

    def test_test_bundle_mutation_during_execution_invalidates_gate(self) -> None:
        bundle = zip_bytes([
            ("test_contract.py", b"""import pathlib
import unittest
import app

class ContractTest(unittest.TestCase):
    def test_expected_behavior(self):
        self.assertEqual(app.VALUE, 1)
        pathlib.Path("test_contract.py").write_text("BROKEN = True" + chr(10), encoding="utf-8")
""")
        ])
        result = self.run_controls(bundle_raw=bundle, evidence_root=self.root / "mutated-tests-evidence")
        self.assertFalse(result["two_sided_gate_passed"])
        self.assertFalse(result["reference_tree_integrity"]["valid"])
        self.assertTrue(any("test bundle file changed" in issue for issue in result["issues"]))

    def test_undeclared_untracked_file_during_test_invalidates_gate(self) -> None:
        bundle = zip_bytes([
            ("test_contract.py", b"""import pathlib
import unittest
import app

class ContractTest(unittest.TestCase):
    def test_expected_behavior(self):
        self.assertEqual(app.VALUE, 1)
        pathlib.Path("rogue-output.py").write_text("SHOULD_NOT_EXIST" + chr(10), encoding="utf-8")
""")
        ])
        result = self.run_controls(bundle_raw=bundle, evidence_root=self.root / "untracked-evidence")
        self.assertFalse(result["two_sided_gate_passed"])
        self.assertFalse(result["reference_tree_integrity"]["valid"])
        self.assertTrue(any("untracked-file inventory changed" in issue for issue in result["issues"]))

    def test_reference_that_does_not_fix_behavior_fails_gate(self) -> None:
        (self.repo / "app.py").write_text("VALUE = 2\\n", encoding="utf-8")
        wrong_patch = self.run_git(self.repo, "diff", "--", "app.py").stdout.encode("utf-8")
        self.run_git(self.repo, "checkout", "--", "app.py")
        self.assertEqual(
            self.run_git(self.repo, "status", "--porcelain=v1", "--untracked-files=all").stdout,
            "",
        )
        result = self.run_controls(
            patch_raw=wrong_patch,
            evidence_root=self.root / "wrong-patch-evidence",
        )
        self.assertFalse(result["two_sided_gate_passed"])
        self.assertTrue(any("reference patch did not pass" in issue for issue in result["issues"]))
        self.assertFalse(result["score_eligible"])

    def test_oracle_test_that_passes_empty_patch_is_rejected(self) -> None:
        bundle = zip_bytes([
            ("test_contract.py", b"""import unittest
import app

class ContractTest(unittest.TestCase):
    def test_current_baseline(self):
        self.assertEqual(app.VALUE, 0)
""")
        ])
        result = self.run_controls(bundle_raw=bundle, evidence_root=self.root / "inverted-evidence")
        self.assertFalse(result["two_sided_gate_passed"])
        self.assertTrue(any("empty-patch baseline" in issue for issue in result["issues"]))

    def test_wrong_manifest_pin_is_rejected(self) -> None:
        with self.assertRaisesRegex(verifier.ControlError, "manifest differs"):
            self.run_controls(expected_manifest="f" * 64)

    def test_wrong_reference_patch_pin_is_rejected(self) -> None:
        with self.assertRaisesRegex(verifier.ControlError, "reference patch differs"):
            self.run_controls(expected_patch="f" * 64)

    def test_wrong_test_bundle_pin_is_rejected(self) -> None:
        with self.assertRaisesRegex(verifier.ControlError, "test bundle differs"):
            self.run_controls(expected_bundle="f" * 64)

    def test_wrong_verifier_source_pin_is_rejected(self) -> None:
        with self.assertRaisesRegex(verifier.ControlError, "verifier source differs"):
            self.run_controls(expected_verifier="f" * 64)

    def test_source_revision_must_match_external_pin(self) -> None:
        with self.assertRaisesRegex(verifier.ControlError, "differs from externally expected revision"):
            self.run_controls(source_revision="a" * 40)

    def test_dirty_source_worktree_is_rejected(self) -> None:
        marker = self.repo / "untracked"
        marker.write_text("dirty\n", encoding="utf-8")
        with self.assertRaisesRegex(verifier.ControlError, "worktree must be clean"):
            self.run_controls()

    def test_nonempty_evidence_directory_is_rejected(self) -> None:
        self.evidence.mkdir()
        (self.evidence / "stale.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(verifier.ControlError, "must be new or empty"):
            self.run_controls()

    def test_evidence_directory_inside_source_repo_is_rejected(self) -> None:
        with self.assertRaisesRegex(verifier.ControlError, "outside the source repository"):
            self.run_controls(evidence_root=self.repo / "evidence")

    def test_duplicate_json_keys_rejected(self) -> None:
        with self.assertRaisesRegex(verifier.ControlError, "duplicate JSON object key"):
            verifier.strict_json_loads(b'{"schema_version":"one","schema_version":"two"}')

    def test_manifest_rejects_duplicate_command_ids(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["commands"].append(copy.deepcopy(manifest["commands"][0]))
        with self.assertRaisesRegex(verifier.ControlError, "duplicate command_id"):
            verifier.validate_manifest(manifest)

    def test_manifest_rejects_unsafe_working_directory(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["commands"][0]["working_directory"] = "../outside"
        with self.assertRaisesRegex(verifier.ControlError, "unsafe or non-canonical"):
            verifier.validate_manifest(manifest)

    def test_manifest_rejects_invalid_regex(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["commands"][0]["pass_pattern"] = "["
        with self.assertRaisesRegex(verifier.ControlError, "invalid regex"):
            verifier.validate_manifest(manifest)

    def test_manifest_rejects_secret_environment_variables(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["commands"][0]["environment"]["API_KEY"] = "must-not-be-present"
        with self.assertRaisesRegex(verifier.ControlError, "secret-bearing"):
            verifier.validate_manifest(manifest)

    def test_command_environment_digest_is_recorded_without_environment_values(self) -> None:
        result = self.run_controls()
        row = result["baseline_results"][0]
        self.assertEqual(row["environment_sha256"], sha(json.dumps(
            self.manifest["commands"][0]["environment"],
            sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        ).encode("utf-8")))
        self.assertEqual(row["environment_names"], sorted(self.manifest["commands"][0]["environment"]))
        self.assertNotIn("environment", row)

    def test_zip_rejects_path_traversal(self) -> None:
        with self.assertRaisesRegex(verifier.ControlError, "unsafe test bundle path"):
            verifier.extract_test_bundle(zip_bytes([("../escape.py", b"")]), self.root / "unsafe-root")

    def test_zip_rejects_source_overwrite(self) -> None:
        with self.assertRaisesRegex(verifier.ControlError, "must not overwrite source files"):
            verifier.extract_test_bundle(zip_bytes([("app.py", b"VALUE = 999\n")]), self.repo)

    def test_invalid_test_command_markers_do_not_count_as_failure(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["commands"][0]["test_start_pattern"] = r"NEVER_STARTS"
        result = self.run_controls(
            manifest_raw=jbytes(manifest),
            evidence_root=self.root / "bad-marker-evidence",
        )
        self.assertFalse(result["two_sided_gate_passed"])
        self.assertTrue(any("unresolved" in issue for issue in result["issues"]))

    def test_timeout_is_unresolved_not_a_killed_baseline(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["commands"][0]["argv"] = [sys.executable, "-c", "import time; time.sleep(2)"]
        manifest["commands"][0]["timeout_seconds"] = 1
        result = self.run_controls(
            manifest_raw=jbytes(manifest),
            evidence_root=self.root / "timeout-evidence",
        )
        self.assertFalse(result["two_sided_gate_passed"])
        self.assertTrue(any("unresolved" in issue for issue in result["issues"]))

    def test_zip_rejects_symlink_entry(self) -> None:
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            info = zipfile.ZipInfo("tests/link.py")
            info.create_system = 3
            info.external_attr = (0o120777 << 16)
            archive.writestr(info, "target.py")
        with self.assertRaisesRegex(verifier.ControlError, "symlink entry"):
            verifier.extract_test_bundle(output.getvalue(), self.root / "safe-root")

    def test_zip_directory_entry_cannot_follow_symlink_parent(self) -> None:
        workspace = self.root / "safe-workspace"
        outside = self.root / "outside"
        workspace.mkdir()
        outside.mkdir()
        (workspace / "linked").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(verifier.ControlError, "symlink directory in test bundle path"):
            verifier.extract_test_bundle(
                zip_bytes([("linked/escape-dir/", b"")]),
                workspace,
            )
        self.assertFalse((outside / "escape-dir").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
