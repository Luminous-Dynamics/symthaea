"""Adversarial tests for the fail-closed protein dynamics run-record validator."""
from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from validate_protein_dynamics_run import validate_record


class RunRecordValidatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.record_path = self.root / "record.json"
        self.targets_path = self.root / "expected-targets.json"
        production_schema = Path(__file__).resolve().parents[2] / "docs/research/PROTEIN_DYNAMICS_RUN_RECORD_SCHEMA_v1.json"
        self.schema_path = production_schema if production_schema.is_file() else self.root / "schema.json"
        self.artifact_root = self.root / "artifacts"
        self.artifact_root.mkdir()
        # In the repository, exercise against the real production schema. Outside the repo
        # (for example, this isolated test run), use a minimal schema to test semantic guards.
        if not production_schema.is_file():
            self.schema_path.write_text(json.dumps({
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "type": "object",
            }), encoding="utf-8")
        self.targets = {"target_ids": ["A", "B"]}
        self.write_targets()
        (self.artifact_root / "result.csv").write_text("target,score\nA,0.5\n", encoding="utf-8")
        self.record = self.base_record()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_targets(self) -> None:
        # Freeze exact bytes; record references raw-byte SHA-256, not reserialized JSON.
        self.targets_path.write_text(
            json.dumps(self.targets, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    def digest(self, path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def target(self, target_id: str, score: float = 0.5) -> dict:
        return {
            "target_id": target_id,
            "target_group_id": f"group-{target_id}",
            "evidence_class": "experimental",
            "outcome": "PASS_WITH_VALID_METRICS",
            "candidate_conformers": 3,
            "valid_conformers": 2,
            "invalid_conformers": 1,
            "metrics": [{
                "metric_name": "observable_rmse",
                "value": score,
                "unit": "normalized units",
                "observable_id": "obs-primary",
                "condition_id": "cond-1",
                "evaluation_role": "pre_reweight_compatibility",
                "observable_overlap": "used_for_selection",
            }],
            "diagnostics": {"optimizer_retry_count": 0, "warnings": [], "error_codes": []},
        }

    def base_record(self) -> dict:
        artifact = self.artifact_root / "result.csv"
        return {
            "schema_version": "1.0.0",
            "run_id": "protein-dynamics-test-run",
            "run_status": "incomplete",
            "created_at": "2026-10-11T00:00:00Z",
            "preregistration": {
                "protocol_sha256": "a" * 64,
                "target_manifest_sha256": self.digest(self.targets_path),
                "metric_spec_sha256": "b" * 64,
                "stopping_rule_sha256": "c" * 64,
                "frozen_at": "2026-10-10T23:00:00Z",
                "target_set_frozen": True,
                "metrics_frozen": True,
            },
            "software": {
                "symthaea_commit": "a" * 40,
                "symthaea_worktree_clean": True,
                "adapter_commit": "b" * 40,
                "peptonebench_revision": "f" * 40,
                "evaluator_revision": "repro-test-evaluator-v1",
            },
            "environment": {
                "os_image_or_platform": "test-linux-image@sha256:" + "a" * 64,
                "architecture": "x86_64",
                "python_version": "3.13.0",
                "rust_toolchain": None,
                "dependency_inventory_sha256": "d" * 64,
                "forward_model_images": [],
                "resource_policy": {
                    "cpu_limit": 2,
                    "memory_limit_bytes": 1024,
                    "gpu_policy": "none",
                    "wall_time_limit_seconds": 30,
                },
            },
            "source_inputs": [{
                "input_id": "synthetic-test-input",
                "source_uri": "https://example.invalid/input",
                "source_revision": "test-revision",
                "sha256": "e" * 64,
                "evidence_class": "benchmark_code",
                "license_status": "not_applicable",
                "experiment_group_id": None,
                "condition_id": None,
                "exposure_notes": "synthetic unit test only",
            }],
            "targets": [self.target("A")],
            "resource_accounting": {
                "wall_time_seconds": 1.0,
                "cpu_time_seconds": 1.0,
                "peak_rss_bytes": 1024,
                "gpu_time_seconds": None,
                "specialist_calls": 1,
                "candidate_conformers_generated": 3,
                "estimated_cost_amount": None,
                "estimated_cost_currency": None,
            },
            "output_artifacts": [{
                "artifact_id": "primary-score-table",
                "uri_or_relative_path": "result.csv",
                "sha256": self.digest(artifact),
                "kind": "metrics",
            }],
            "attestations": {
                "all_expected_targets_accounted": False,
                "failures_retained": True,
                "all_claimed_output_hashes_verified": True,
                "held_out_observables_sealed_until_scoring": True,
                "independent_reviewer_or_reproduction": False,
            },
        }

    def run_validator(self) -> tuple[list[str], list[str], str | None]:
        self.record_path.write_text(json.dumps(self.record), encoding="utf-8")
        return validate_record(self.record_path, self.schema_path, self.targets_path, self.artifact_root)

    def test_partial_record_is_valid_but_warns_about_unaccounted_targets(self) -> None:
        errors, warnings, status = self.run_validator()
        self.assertEqual(errors, [])
        self.assertEqual(status, "incomplete")
        self.assertTrue(any("expected targets absent" in x for x in warnings))

    def test_complete_record_requires_exact_target_census(self) -> None:
        self.record["run_status"] = "complete"
        self.record["targets"].append(self.target("B", 0.4))
        self.record["resource_accounting"]["candidate_conformers_generated"] = 6
        self.record["attestations"]["all_expected_targets_accounted"] = True
        errors, _, _ = self.run_validator()
        self.assertEqual(errors, [])

        self.record["targets"].pop()
        errors, _, _ = self.run_validator()
        self.assertTrue(any("expected targets absent" in x for x in errors))

    def test_target_manifest_byte_hash_is_verified(self) -> None:
        self.record["preregistration"]["target_manifest_sha256"] = "0" * 64
        errors, _, _ = self.run_validator()
        self.assertTrue(any("target manifest SHA-256" in x for x in errors))

    def test_artifact_hash_mismatch_fails_closed(self) -> None:
        self.record["output_artifacts"][0]["sha256"] = "0" * 64
        errors, _, _ = self.run_validator()
        self.assertTrue(any("SHA-256 mismatch" in x for x in errors))

    def test_nan_score_is_never_valid(self) -> None:
        self.record["targets"][0]["metrics"][0]["value"] = math.nan
        errors, _, _ = self.run_validator()
        self.assertTrue(any("metric value must be finite" in x for x in errors))

    def test_failure_must_keep_an_error_code(self) -> None:
        row = self.record["targets"][0]
        row["outcome"] = "FAIL_EXECUTION"
        row["metrics"] = []
        errors, _, _ = self.run_validator()
        self.assertTrue(any("failure outcome must retain" in x for x in errors))

    def test_same_observable_reweighting_cannot_be_labeled_held_out(self) -> None:
        metric = self.record["targets"][0]["metrics"][0]
        metric["evaluation_role"] = "independent_held_out_observable"
        metric["observable_overlap"] = "used_for_reweighting"
        errors, _, _ = self.run_validator()
        self.assertTrue(any("held-out metric" in x for x in errors))

    def test_duplicate_target_rows_are_rejected(self) -> None:
        self.record["targets"].append(self.target("A", 0.3))
        errors, _, _ = self.run_validator()
        self.assertTrue(any("duplicate target_id" in x for x in errors))

    def test_path_escape_is_rejected(self) -> None:
        self.record["output_artifacts"][0]["uri_or_relative_path"] = "../outside.csv"
        errors, _, _ = self.run_validator()
        self.assertTrue(any("escapes artifact root" in x for x in errors))

    def test_resource_total_must_match_target_ledger(self) -> None:
        self.record["resource_accounting"]["candidate_conformers_generated"] = 99
        errors, _, _ = self.run_validator()
        self.assertTrue(any("resource_accounting.candidate_conformers_generated" in x for x in errors))

    def test_wrong_target_id_type_is_rejected(self) -> None:
        self.record["targets"][0]["target_id"] = ["A"]
        errors, _, _ = self.run_validator()
        self.assertTrue(any("missing or non-string target_id" in x for x in errors))

    def test_conformer_counts_must_reconcile(self) -> None:
        self.record["targets"][0]["candidate_conformers"] = 10
        errors, _, _ = self.run_validator()
        self.assertTrue(any("candidate_conformers must equal" in x for x in errors))

    def test_malformed_preregistration_fails_closed_without_crashing(self) -> None:
        self.record["preregistration"] = None
        errors, _, _ = self.run_validator()
        self.assertTrue(errors)

    def test_preregistration_must_precede_record_creation(self) -> None:
        self.record["preregistration"]["frozen_at"] = "2026-10-12T00:00:00Z"
        errors, _, _ = self.run_validator()
        self.assertTrue(any("must not be later than created_at" in x for x in errors))

    def test_nonlocal_artifact_is_not_claimed_verified(self) -> None:
        self.record["output_artifacts"][0]["uri_or_relative_path"] = "https://example.invalid/score.csv"
        errors, _, _ = self.run_validator()
        self.assertTrue(any("local relative paths only" in x for x in errors))


    def test_complete_run_cannot_contain_not_run_target(self) -> None:
        self.record["run_status"] = "complete"
        row = self.record["targets"][0]
        row["outcome"] = "NOT_RUN"
        row["metrics"] = []
        self.record["targets"].append(self.target("B"))
        self.record["attestations"]["all_expected_targets_accounted"] = True
        errors, _, _ = self.run_validator()
        self.assertTrue(any("cannot contain a NOT_RUN" in x for x in errors))


if __name__ == "__main__":
    unittest.main()
