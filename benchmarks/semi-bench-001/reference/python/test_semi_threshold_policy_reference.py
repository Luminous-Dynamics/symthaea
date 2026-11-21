"""Known-answer and adversarial tests for validation threshold policy."""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import semi_threshold_policy_reference as policy  # noqa: E402
from semi_binary_metrics_reference import ContractError  # noqa: E402

FIXTURE = HERE.parents[1] / "fixtures" / "validation-threshold-known-answer-v1.json"


class ValidationThresholdPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_known_answer_selects_validation_precision_floor_then_scores_test(self) -> None:
        result = policy.evaluate_with_validation_threshold(
            self.case["validation_samples"], self.case["test_subject"],
            self.case["minimum_validation_precision"], self.case["fallback_threshold"],
        )
        expected_path = HERE.parents[1] / "fixtures" / "validation-threshold-known-answer-v1.expected.json"
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        self.assertEqual(result, expected)
        self.assertEqual(result["selection"]["threshold"], 0.91)
        self.assertEqual(result["selection"]["validation_precision"], 1.0)
        self.assertAlmostEqual(result["test_metrics"]["threshold_metrics"]["sensitivity"], 0.4)

    def test_rejects_validation_test_identity_overlap(self) -> None:
        case = copy.deepcopy(self.case)
        case["test_subject"]["samples"][0]["sample_id"] = case["validation_samples"][0]["sample_id"]
        with self.assertRaisesRegex(ContractError, "identity overlap"):
            policy.evaluate_with_validation_threshold(
                case["validation_samples"], case["test_subject"],
                case["minimum_validation_precision"], case["fallback_threshold"],
            )

    def test_records_fallback_when_precision_floor_is_unattainable(self) -> None:
        case = copy.deepcopy(self.case)
        for row in case["validation_samples"]:
            row["score"] = 0.9 if not row["positive"] else 0.1
        result = policy.evaluate_with_validation_threshold(
            case["validation_samples"], case["test_subject"], 1.0, 0.5,
        )
        self.assertTrue(result["fallback_used"])
        self.assertIsNone(result["selection"])
        self.assertEqual(result["test_metrics"]["threshold_metrics"]["threshold"], 0.5)


if __name__ == "__main__":
    unittest.main()
