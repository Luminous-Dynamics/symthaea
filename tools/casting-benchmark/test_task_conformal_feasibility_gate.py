#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from task_conformal_feasibility_gate import (
    RECEIPT_LABEL,
    evaluate,
    main,
    validate_config,
)

SOURCE = "a" * 64
POLICY = "b" * 64
ENV = "c" * 64


def pool_digest(count=2):
    roster = [f"candidate-{index:03d}" for index in range(count)]
    return hashlib.sha256(json.dumps(roster, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def config(**overrides):
    value = {
        "schema_version": "1.0.0",
        "config_id": "task-conformal-test-v1",
        "evidence_class": "synthetic",
        "policy_source_sha256": SOURCE,
        "policy_config_sha256": POLICY,
        "candidate_pool_sha256": pool_digest(),
        "alpha": 0.05,
        "minimum_calibration_tasks_per_stratum": 19,
        "maximum_descriptor_distance": 0.20,
        "frozen_before_outcome_unseal": True,
    }
    value.update(overrides)
    return value


def calibration(count=20, *, rows_per_task=1, stratum="known_process"):
    tasks = []
    for i in range(count):
        rows = []
        for j in range(rows_per_task):
            # The task-level score is the maximum overprediction among all
            # candidates in this task's frozen finite pool.
            score = i / 100.0
            rows.append({
                "candidate_id": f"candidate-{j:03d}",
                "predicted_margin": score,
                "observed_margin": 0.0,
            })
        tasks.append({
            "task_id": f"cal-task-{i:03d}",
            "task_family": "calibration_family",
            "stratum": stratum,
            "descriptor": [0.50 + (i % 5) / 100.0, 0.50],
            "rows": rows,
        })
    return {
        "schema_version": "1.0.0",
        "calibration_id": "calibration-v1",
        "evidence_class": "synthetic",
        "policy_source_sha256": SOURCE,
        "policy_config_sha256": POLICY,
        "split": "independent_calibration",
        "policy_frozen_before_outcomes": True,
        "candidate_pool_frozen_before_outcomes": True,
        "outcomes_unsealed_after_freeze": True,
        "independence_unit": "task_id",
        "independence_note": "Each calibration task is separately generated; points are nested within task.",
        "tasks": tasks,
    }


def target(*, stratum="known_process", descriptor=None, task_id="target-task-001",
           predictions=None):
    if descriptor is None:
        descriptor = [0.50, 0.50]
    if predictions is None:
        predictions = [
            {"candidate_id": "candidate-000", "predicted_margin": 0.25},
            {"candidate_id": "candidate-001", "predicted_margin": 0.18},
        ]
    return {
        "schema_version": "1.0.0",
        "target_manifest_id": "target-predictions-v1",
        "evidence_class": "synthetic",
        "policy_source_sha256": SOURCE,
        "policy_config_sha256": POLICY,
        "split": "independent_held_out_predictions",
        "candidate_pool_frozen_before_outcomes": True,
        "tasks": [{
            "task_id": task_id,
            "task_family": "heldout_task_family",
            "stratum": stratum,
            "descriptor": descriptor,
            "predictions": predictions,
        }],
    }


class TaskConformalFeasibilityGateTests(unittest.TestCase):
    def test_task_level_quantile_and_conservative_feasibility_screen(self):
        result = evaluate(config(), calibration(), target())
        self.assertEqual(result["receipt_label"], RECEIPT_LABEL)
        self.assertEqual(result["decision"], "evaluated_with_abstention")
        self.assertEqual(result["calibration_task_count"], 20)
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(result["recommended_feasible_count"], 1)
        self.assertEqual(result["abstained_candidate_count"], 1)
        self.assertAlmostEqual(
            result["strata"]["known_process"]["overprediction_quantile"], 0.19
        )
        rows = result["target_task_results"][0]["candidates"]
        self.assertEqual(rows[0]["action"], "recommend_feasible")
        self.assertAlmostEqual(rows[0]["calibrated_lower_margin"], 0.06)
        self.assertEqual(rows[1]["action"], "abstain")
        self.assertAlmostEqual(rows[1]["calibrated_lower_margin"], -0.01)

    def test_candidate_points_do_not_inflate_independent_calibration_n(self):
        large_pool = calibration(count=20, rows_per_task=100)
        cfg = config(candidate_pool_sha256=pool_digest(100))
        predictions = [
            {"candidate_id": f"candidate-{index:03d}",
             "predicted_margin": 0.25 if index == 0 else 0.18}
            for index in range(100)
        ]
        result = evaluate(cfg, large_pool, target(predictions=predictions))
        self.assertEqual(result["calibration_task_count"], 20)
        self.assertEqual(result["strata"]["known_process"]["calibration_task_count"], 20)
        self.assertEqual(result["strata"]["known_process"]["conformal_rank_1based"], 20)
        self.assertAlmostEqual(
            result["strata"]["known_process"]["overprediction_quantile"], 0.19
        )
        self.assertEqual(result["candidate_count"], 100)

    def test_insufficient_calibration_tasks_force_abstention(self):
        result = evaluate(config(), calibration(count=18), target())
        self.assertEqual(result["decision"], "inconclusive")
        self.assertEqual(result["underpowered_target_strata"], ["known_process"])
        self.assertEqual(result["recommended_feasible_count"], 0)
        self.assertIsNone(result["strata"]["known_process"]["overprediction_quantile"])

    def test_unseen_stratum_forces_abstention_and_is_explicitly_reported(self):
        result = evaluate(config(), calibration(), target(stratum="unseen_process"))
        self.assertEqual(result["decision"], "inconclusive")
        self.assertEqual(result["unsupported_target_strata"], ["unseen_process"])
        self.assertEqual(result["recommended_feasible_count"], 0)
        self.assertIsNone(
            result["target_task_results"][0]["nearest_calibration_descriptor_distance"]
        )

    def test_descriptor_distant_target_is_not_recommended(self):
        result = evaluate(config(maximum_descriptor_distance=0.15), calibration(),
                          target(descriptor=[1.0, 1.0]))
        task = result["target_task_results"][0]
        self.assertFalse(task["descriptor_support_ok"])
        self.assertGreater(task["nearest_calibration_descriptor_distance"], 0.15)
        self.assertEqual(task["recommended_feasible_count"], 0)
        self.assertTrue(all(row["action"] == "abstain" for row in task["candidates"]))

    def test_missing_frozen_candidate_pool_flag_is_rejected(self):
        cal = calibration()
        cal["candidate_pool_frozen_before_outcomes"] = False
        with self.assertRaisesRegex(ValueError, "candidate pool must be frozen"):
            evaluate(config(), cal, target())
        tgt = target()
        tgt["candidate_pool_frozen_before_outcomes"] = False
        with self.assertRaisesRegex(ValueError, "candidate pool must be frozen"):
            evaluate(config(), calibration(), tgt)

    def test_candidate_pool_roster_digest_and_order_must_match(self):
        cal = calibration()
        cal["tasks"][1]["rows"][0]["candidate_id"] = "other-id"
        with self.assertRaisesRegex(ValueError, "same ordered frozen candidate pool"):
            evaluate(config(), cal, target())
        tgt = target()
        tgt["tasks"][0]["predictions"][0]["candidate_id"] = "other-id"
        with self.assertRaisesRegex(ValueError, "exact frozen calibration candidate-pool roster"):
            evaluate(config(), calibration(), tgt)
        cfg = config(candidate_pool_sha256="d" * 64)
        with self.assertRaisesRegex(ValueError, "roster digest"):
            evaluate(cfg, calibration(), target())

    def test_target_manifest_cannot_smuggle_outcome_labels(self):
        tgt = target()
        tgt["tasks"][0]["predictions"][0]["observed_margin"] = 0.5
        with self.assertRaisesRegex(ValueError, "keys mismatch"):
            evaluate(config(), calibration(), tgt)

    def test_policy_digests_and_evidence_classes_must_match(self):
        cal = calibration()
        cal["policy_source_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "policy_source_sha256 does not match"):
            evaluate(config(), cal, target())
        tgt = target()
        tgt["policy_config_sha256"] = "d" * 64
        with self.assertRaisesRegex(ValueError, "policy_config_sha256 does not match"):
            evaluate(config(), calibration(), tgt)
        tgt = target()
        tgt["evidence_class"] = "physical"
        with self.assertRaisesRegex(ValueError, "evidence class does not match"):
            evaluate(config(), calibration(), tgt)

    def test_task_ids_and_candidate_ids_are_unique_and_splits_are_disjoint(self):
        cal = calibration()
        cal["tasks"][1]["task_id"] = cal["tasks"][0]["task_id"]
        with self.assertRaisesRegex(ValueError, "duplicate calibration task_id"):
            evaluate(config(), cal, target())
        cal = calibration()
        tgt = target(task_id=cal["tasks"][0]["task_id"])
        with self.assertRaisesRegex(ValueError, "overlaps calibration task"):
            evaluate(config(), cal, tgt)
        tgt = target(predictions=[
            {"candidate_id": "dup", "predicted_margin": 0.25},
            {"candidate_id": "dup", "predicted_margin": 0.30},
        ])
        with self.assertRaisesRegex(ValueError, "duplicate target candidate_id"):
            evaluate(config(), calibration(), tgt)

    def test_bad_config_cannot_claim_a_finite_quantile_with_too_few_tasks(self):
        with self.assertRaisesRegex(ValueError, "too small for a finite"):
            validate_config(config(minimum_calibration_tasks_per_stratum=18))
        with self.assertRaisesRegex(ValueError, "alpha must be"):
            validate_config(config(alpha=0.5))
        with self.assertRaisesRegex(ValueError, "maximum_descriptor_distance"):
            validate_config(config(maximum_descriptor_distance=-0.1))

    def test_strict_schema_rejects_extra_fields_and_malformed_descriptors(self):
        bad = config()
        bad["quiet_override"] = True
        with self.assertRaisesRegex(ValueError, "keys mismatch"):
            evaluate(bad, calibration(), target())
        cal = calibration()
        cal["tasks"][0]["descriptor"] = [0.2, 0.3, 0.4]
        with self.assertRaisesRegex(ValueError, "same dimension"):
            evaluate(config(), cal, target())
        tgt = target(descriptor=[0.2, 0.3, 1.1])
        with self.assertRaisesRegex(ValueError, "normalized to"):
            evaluate(config(), calibration(), tgt)

    def test_task_score_is_one_sided_overprediction_not_absolute_error(self):
        cal = calibration(count=20)
        for task in cal["tasks"]:
            task["rows"] = [{
                "candidate_id": "c",
                "predicted_margin": -0.5,
                "observed_margin": 0.5,
            }]
        result = evaluate(config(), cal, target())
        self.assertEqual(result["strata"]["known_process"]["overprediction_quantile"], 0.0)
        self.assertEqual(result["recommended_feasible_count"], 2)

    def test_cli_emits_digest_bound_receipt_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg_path, cal_path, target_path, out_path = [
                root / name for name in ("config.json", "calibration.json", "target.json", "receipt.json")
            ]
            cfg_bytes = json.dumps(config(), separators=(",", ":"), sort_keys=True).encode()
            cal_bytes = json.dumps(calibration(), separators=(",", ":"), sort_keys=True).encode()
            target_bytes = json.dumps(target(), separators=(",", ":"), sort_keys=True).encode()
            cfg_path.write_bytes(cfg_bytes)
            cal_path.write_bytes(cal_bytes)
            target_path.write_bytes(target_bytes)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main([str(cfg_path), str(cal_path), str(target_path), "--output", str(out_path)])
            self.assertEqual(code, 0)
            receipt = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["decision"], "evaluated_with_abstention")
            self.assertEqual(receipt["source_artifacts"]["config_sha256"], hashlib.sha256(cfg_bytes).hexdigest())
            self.assertEqual(receipt["source_artifacts"]["calibration_manifest_sha256"], hashlib.sha256(cal_bytes).hexdigest())
            self.assertEqual(receipt["source_artifacts"]["target_prediction_manifest_sha256"], hashlib.sha256(target_bytes).hexdigest())
            self.assertEqual(len(receipt["source_artifacts"]["evaluator_sha256"]), 64)
            self.assertIn("recommended=1 abstained=1", stdout.getvalue())
            original = out_path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                code = main([str(cfg_path), str(cal_path), str(target_path), "--output", str(out_path)])
            self.assertEqual(code, 2)
            self.assertEqual(out_path.read_bytes(), original)

    def test_cli_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "config.json"
            cal = root / "calibration.json"
            tgt = root / "target.json"
            out = root / "receipt.json"
            cfg.write_text('{"schema_version":"1.0.0","schema_version":"1.0.0"}', encoding="utf-8")
            cal.write_text(json.dumps(calibration()), encoding="utf-8")
            tgt.write_text(json.dumps(target()), encoding="utf-8")
            error = io.StringIO()
            with contextlib.redirect_stderr(error):
                code = main([str(cfg), str(cal), str(tgt), "--output", str(out)])
            self.assertEqual(code, 2)
            self.assertIn("duplicate JSON key", error.getvalue())


if __name__ == "__main__":
    unittest.main()
