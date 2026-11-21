"""Known-answer and adversarial tests for the independent SEMI-BENCH scorer."""
from __future__ import annotations

import copy
import json
import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import semi_binary_metrics_reference as scorer  # noqa: E402

FIXTURE = HERE.parents[1] / "fixtures" / "binary-metrics-known-answer-v1.json"


class BinaryMetricsReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.subject = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_known_answer_confusion_and_threshold_metrics(self) -> None:
        result = scorer.score_subject(self.subject)
        metrics = result["threshold_metrics"]
        self.assertEqual((metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]), (2, 1, 3, 2))
        self.assertAlmostEqual(metrics["sensitivity"], 0.5)
        self.assertAlmostEqual(metrics["specificity"], 0.75)
        self.assertAlmostEqual(metrics["precision"], 2 / 3)
        self.assertAlmostEqual(metrics["accuracy"], 0.625)
        self.assertAlmostEqual(metrics["balanced_accuracy"], 0.625)

    def test_known_answer_ranking_probability_and_recall_metrics(self) -> None:
        result = scorer.score_subject(self.subject)
        ranking = result["ranking_metrics"]
        probability = result["probability_metrics"]
        self.assertAlmostEqual(ranking["average_precision"], 0.830357142857)
        self.assertAlmostEqual(ranking["auroc"], 0.78125)
        self.assertEqual(ranking["fpr_at_recall"], {
            "target_recall": 0.75,
            "threshold": 0.4,
            "achieved_recall": 0.75,
            "false_positive_rate": 0.25,
        })
        self.assertAlmostEqual(probability["brier_score"], 0.21)
        self.assertAlmostEqual(probability["expected_calibration_error"], 0.225)

    def test_equal_score_and_equal_confidence_ties_are_grouped(self) -> None:
        result = scorer.score_subject(self.subject)
        # Equal probability scores are admitted together in AP; equal confidence
        # scores are admitted together in the risk-coverage curve.
        ap_points = result["ranking_metrics"]["average_precision"]
        self.assertAlmostEqual(ap_points, 0.830357142857)
        curve = result["risk_coverage"]
        self.assertEqual([point["accepted"] for point in curve], [2, 5, 7, 8])
        self.assertEqual([point["errors"] for point in curve], [0, 1, 2, 3])

    def test_checked_in_expected_report_is_reproduced_exactly(self) -> None:
        result = scorer.score_subject(self.subject)
        expected_path = HERE.parents[1] / "fixtures" / "binary-metrics-known-answer-v1.expected.json"
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        self.assertEqual(result, expected)

    def test_input_order_does_not_change_metrics(self) -> None:
        reordered = copy.deepcopy(self.subject)
        reordered["samples"] = list(reversed(reordered["samples"]))
        self.assertEqual(scorer.score_subject(self.subject), scorer.score_subject(reordered))

    def test_rejects_empty_subject(self) -> None:
        subject = copy.deepcopy(self.subject)
        subject["samples"] = []
        with self.assertRaisesRegex(scorer.ContractError, "non-empty"):
            scorer.score_subject(subject)

    def test_rejects_duplicate_sample_identity(self) -> None:
        subject = copy.deepcopy(self.subject)
        subject["samples"][1]["sample_id"] = subject["samples"][0]["sample_id"]
        with self.assertRaisesRegex(scorer.ContractError, "duplicate"):
            scorer.score_subject(subject)

    def test_rejects_nan_infinity_and_out_of_range_scores(self) -> None:
        for bad in (math.nan, math.inf, -math.inf, -0.001, 1.001):
            subject = copy.deepcopy(self.subject)
            subject["samples"][0]["score"] = bad
            with self.subTest(score=bad), self.assertRaises(scorer.ContractError):
                scorer.score_subject(subject)

    def test_rejects_non_boolean_label(self) -> None:
        subject = copy.deepcopy(self.subject)
        subject["samples"][0]["positive"] = 1
        with self.assertRaisesRegex(scorer.ContractError, "boolean"):
            scorer.score_subject(subject)

    def test_rejects_one_class_only(self) -> None:
        subject = copy.deepcopy(self.subject)
        for row in subject["samples"]:
            row["positive"] = True
        with self.assertRaisesRegex(scorer.ContractError, "both positive and negative"):
            scorer.score_subject(subject)

    def test_rejects_invalid_threshold_and_calibration_bins(self) -> None:
        for field, value in (("decision_threshold", 1.1), ("calibration_bins", 1), ("recall_target", 0.0)):
            subject = copy.deepcopy(self.subject)
            subject[field] = value
            with self.subTest(field=field), self.assertRaises(scorer.ContractError):
                scorer.score_subject(subject)


if __name__ == "__main__":
    unittest.main()
