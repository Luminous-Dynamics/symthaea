#!/usr/bin/env python3
"""Adversarial tests for matched clean/shifted VLEB continuations."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

MODULE_PATH = Path(__file__).with_name("analyze_shifted_runs.py")
SPEC = importlib.util.spec_from_file_location("vleb_analyze_shifted_runs", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import shifted-run analyzer")
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def jbytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


class ShiftFixture:
    PROTOCOL = "a" * 64
    REVISION = "b" * 40
    IMAGE = "sha256:" + "c" * 64

    def __init__(self) -> None:
        self.seeds = [11, 22]
        self.scenarios = []
        self.runs = []
        for index in range(6):
            scenario_id = f"scenario-{index + 1:02d}"
            scenario = {
                "scenario_id": scenario_id,
                "task_family": "legacy" if index < 3 else "runtime",
                "change_family": ("requirement_update" if index % 2 == 0 else "tool_session"),
                "intervention_position": index + 1,
                "task_contract_before_sha256": sha(f"before:{scenario_id}".encode()),
                "task_contract_after_sha256": sha(f"after:{scenario_id}".encode()),
                "shift_event_sha256": sha(f"shift:{scenario_id}".encode()),
                "initial_state_sha256": sha(f"state:{scenario_id}".encode()),
                "initial_checkpoint_sha256": sha(f"checkpoint:{scenario_id}".encode()),
                "initial_revision": self.REVISION,
                "initial_image_id": self.IMAGE,
            }
            self.scenarios.append(scenario)
        # Explicit scenario transitions exercise success, regressions, recovery,
        # unresolved outcomes and paired accounting. Repeated seeds remain paired.
        transitions = [
            ("pass", "pass"),
            ("pass", "fail"),
            ("fail", "pass"),
            ("pass", "unresolved"),
            ("fail", "fail"),
            ("unresolved", "fail"),
        ]
        self.manifest = {
            "schema_version": "vleb-shift-comparison-v1",
            "protocol_sha256": self.PROTOCOL,
            "cohort_id": "vleb-shift-pilot-001",
            "scenarios": self.scenarios,
            "seeds": self.seeds,
            "bootstrap": {"replicates": 1000, "seed": 20261029, "confidence_level": 0.95},
            "runs": [],
        }
        for index, scenario in enumerate(self.scenarios):
            scenario_sha = analyzer.scenario_digest(scenario)
            clean_outcome, shifted_outcome = transitions[index]
            for seed in self.seeds:
                for condition, outcome in (("clean", clean_outcome), ("shifted", shifted_outcome)):
                    run_id = f"run-{index + 1:02d}-{seed}-{condition}"
                    self.manifest["runs"].append({
                        "scenario_id": scenario["scenario_id"],
                        "seed": seed,
                        "condition": condition,
                        "outcome": outcome,
                        "run_id": run_id,
                        "attempt_number": 1,
                        "scenario_sha256": scenario_sha,
                        "protocol_sha256": self.PROTOCOL,
                        "initial_state_sha256": scenario["initial_state_sha256"],
                        "initial_checkpoint_sha256": scenario["initial_checkpoint_sha256"],
                        "initial_revision": scenario["initial_revision"],
                        "initial_image_id": scenario["initial_image_id"],
                        "model_id": "baseline-agent",
                        "model_version": "v1",
                        "scaffold_sha256": sha(b"scaffold"),
                        "initial_execution_profile_sha256": sha(b"initial-profile"),
                        "post_shift_execution_profile_sha256": None if condition == "clean" else sha(f"post-profile:{scenario['scenario_id']}".encode()),
                        "initial_network_mode": "none",
                        "post_shift_network_mode": None if condition == "clean" else "none",
                        "final_state_sha256": sha(f"final:{run_id}".encode()),
                        "receipt_sha256": sha(f"receipt:{run_id}".encode()),
                        "audit_report_sha256": sha(f"audit:{run_id}".encode()),
                        "cost_usd": 0.08 if condition == "clean" else 0.09,
                        "duration_ms": 1000 + index * 20 + seed + (100 if condition == "shifted" else 0),
                    })

    def analyze(self, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        value = self.manifest if manifest is None else manifest
        return analyzer.analyze_manifest(
            value,
            input_sha256=sha(jbytes(value)),
            expected_protocol_sha256=self.PROTOCOL,
            expected_analyzer_sha256=sha(MODULE_PATH.read_bytes()),
        )


class ShiftedRunsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ShiftFixture()

    def test_valid_shift_comparison_reports_paired_transitions_and_stays_analysis_only(self) -> None:
        result = self.fixture.analyze()
        self.assertTrue(result["valid"])
        self.assertEqual(result["scenario_count"], 6)
        self.assertEqual(result["seed_count_per_scenario"], 2)
        self.assertEqual(result["expected_paired_run_count"], 12)
        self.assertEqual(result["observed_run_count"], 24)
        overall = result["overall"]
        self.assertEqual(overall["scheduled_pair_count"], 12)
        self.assertEqual(sum(overall["transition_matrix"].values()), 12)
        self.assertEqual(overall["transition_matrix"]["clean_pass__shifted_pass"], 2)
        self.assertEqual(overall["transition_matrix"]["clean_pass__shifted_fail"], 2)
        self.assertEqual(overall["transition_matrix"]["clean_pass__shifted_unresolved"], 2)
        self.assertEqual(overall["transition_matrix"]["clean_fail__shifted_pass"], 2)
        self.assertEqual(overall["transition_matrix"]["clean_fail__shifted_fail"], 2)
        self.assertEqual(overall["transition_matrix"]["clean_unresolved__shifted_fail"], 2)
        self.assertEqual(overall["clean_pass_shifted_fail_count"], 2)
        self.assertEqual(overall["clean_pass_shifted_unresolved_count"], 2)
        self.assertEqual(overall["unresolved_pair_count"], 4)
        self.assertAlmostEqual(overall["clean_success_rate_pct"], 6 * 100 / 12)
        self.assertAlmostEqual(overall["shifted_success_rate_pct"], 4 * 100 / 12)
        self.assertAlmostEqual(overall["shift_degradation_pp"], 100 / 6)
        self.assertAlmostEqual(overall["robustness_given_clean_success_pct"], 2 * 100 / 6)
        self.assertEqual(result["by_task_family"]["legacy"]["scenario_count"], 3)
        self.assertEqual(result["by_task_family"]["runtime"]["scenario_count"], 3)
        self.assertEqual(set(result["by_change_family"]), {"requirement_update", "tool_session"})
        self.assertIn("1", result["by_intervention_position"])
        self.assertTrue(result["analysis_only"])
        self.assertFalse(result["score_eligible"])
        self.assertTrue(result["warnings"])

    def test_missing_condition_cell_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["runs"].pop()
        with self.assertRaisesRegex(ValueError, "complete locked scenario"):
            self.fixture.analyze(manifest)

    def test_initial_checkpoint_must_match_locked_scenario(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["runs"][0]["initial_checkpoint_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "initial checkpoint differs"):
            self.fixture.analyze(manifest)

    def test_clean_and_shifted_must_share_initial_execution_identity(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        shifted = next(row for row in manifest["runs"] if row["condition"] == "shifted")
        shifted["initial_execution_profile_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "do not share the same pinned starting identity"):
            self.fixture.analyze(manifest)

    def test_model_configuration_must_not_drift_across_seeds(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        for row in manifest["runs"]:
            if row["scenario_id"] == "scenario-01" and row["seed"] == 22:
                row["initial_execution_profile_sha256"] = sha(b"drifted-profile")
        with self.assertRaisesRegex(ValueError, "drifts in model/scaffold/initial profile across seeds"):
            self.fixture.analyze(manifest)

    def test_post_shift_configuration_must_not_drift_across_seeds(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        for row in manifest["runs"]:
            if row["scenario_id"] == "scenario-01" and row["condition"] == "shifted" and row["seed"] == 22:
                row["post_shift_execution_profile_sha256"] = sha(b"other-post-profile")
        with self.assertRaisesRegex(ValueError, "drifts in post-shift profile/network mode across seeds"):
            self.fixture.analyze(manifest)

    def test_malformed_network_mode_fails_closed(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        row = next(item for item in manifest["runs"] if item["condition"] == "clean")
        row["initial_network_mode"] = ["none"]
        with self.assertRaisesRegex(ValueError, "initial_network_mode must be a valid network-mode string"):
            self.fixture.analyze(manifest)
        manifest = copy.deepcopy(self.fixture.manifest)
        row = next(item for item in manifest["runs"] if item["condition"] == "shifted")
        row["post_shift_network_mode"] = ["none"]
        with self.assertRaisesRegex(ValueError, "post_shift_network_mode must be a valid network-mode string"):
            self.fixture.analyze(manifest)

    def test_clean_condition_cannot_claim_post_shift_configuration(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        clean = next(row for row in manifest["runs"] if row["condition"] == "clean")
        clean["post_shift_network_mode"] = "none"
        with self.assertRaisesRegex(ValueError, "clean condition must not claim"):
            self.fixture.analyze(manifest)

    def test_shifted_condition_requires_post_shift_profile(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        shifted = next(row for row in manifest["runs"] if row["condition"] == "shifted")
        shifted["post_shift_execution_profile_sha256"] = None
        with self.assertRaisesRegex(ValueError, "post_shift_execution_profile_sha256"):
            self.fixture.analyze(manifest)

    def test_replayed_receipt_or_audit_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["runs"][1]["receipt_sha256"] = manifest["runs"][0]["receipt_sha256"]
        with self.assertRaisesRegex(ValueError, "receipt digest replay"):
            self.fixture.analyze(manifest)
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["runs"][1]["audit_report_sha256"] = manifest["runs"][0]["audit_report_sha256"]
        with self.assertRaisesRegex(ValueError, "audit-report digest replay"):
            self.fixture.analyze(manifest)

    def test_scenario_hash_mismatch_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["runs"][0]["scenario_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "canonical locked scenario"):
            self.fixture.analyze(manifest)

    def test_task_family_and_change_family_labels_are_validated(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["scenarios"][0]["change_family"] = "../tool"
        with self.assertRaisesRegex(ValueError, "change_family"):
            self.fixture.analyze(manifest)

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            analyzer.strict_json_loads(b'{"cohort_id":"one","cohort_id":"two"}')

    def test_nonfinite_json_numbers_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
            analyzer.strict_json_loads(b'{"cost_usd":NaN}')

    def test_deterministic_bootstrap_repeats_exactly(self) -> None:
        first = self.fixture.analyze()
        second = self.fixture.analyze()
        self.assertEqual(first["overall"], second["overall"])
        self.assertEqual(first["by_task_family"], second["by_task_family"])

    def test_zero_clean_success_makes_conditional_robustness_undefined(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        for row in manifest["runs"]:
            if row["condition"] == "clean":
                row["outcome"] = "fail"
            if row["condition"] == "shifted":
                row["outcome"] = "fail"
        result = self.fixture.analyze(manifest)
        self.assertIsNone(result["overall"]["robustness_given_clean_success_pct"])

    def test_cli_pins_input_protocol_and_analyzer(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vleb-shift-cli-") as temporary:
            root = Path(temporary)
            source = root / "manifest.json"
            report = root / "report.json"
            source.write_bytes(jbytes(self.fixture.manifest))
            completed = subprocess.run(
                [sys.executable, str(MODULE_PATH), "--input", str(source),
                 "--expected-input-sha256", sha(source.read_bytes()),
                 "--expected-protocol-sha256", self.fixture.PROTOCOL,
                 "--expected-analyzer-sha256", sha(MODULE_PATH.read_bytes()),
                 "--output", str(report)],
                capture_output=True, text=True, check=False, timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            output = json.loads(completed.stdout)
            self.assertTrue(output["valid"])
            self.assertTrue(output["analysis_only"])
            self.assertFalse(output["score_eligible"])
            self.assertEqual(output, json.loads(report.read_text(encoding="utf-8")))

    def test_cli_wrong_manifest_pin_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vleb-shift-cli-reject-") as temporary:
            source = Path(temporary) / "manifest.json"
            source.write_bytes(jbytes(self.fixture.manifest))
            completed = subprocess.run(
                [sys.executable, str(MODULE_PATH), "--input", str(source),
                 "--expected-input-sha256", "f" * 64,
                 "--expected-protocol-sha256", self.fixture.PROTOCOL,
                 "--expected-analyzer-sha256", sha(MODULE_PATH.read_bytes())],
                capture_output=True, text=True, check=False, timeout=30,
            )
            self.assertEqual(completed.returncode, 1)
            output = json.loads(completed.stdout)
            self.assertFalse(output["valid"])
            self.assertTrue(output["analysis_only"])
            self.assertFalse(output["score_eligible"])
            self.assertEqual(output["issues"][0]["code"], "shift_comparison_rejected")


if __name__ == "__main__":
    unittest.main(verbosity=2)
