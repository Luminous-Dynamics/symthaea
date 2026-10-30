"""Synthetic controls for matched judge/human score geometry."""
from __future__ import annotations

import copy
import json
import unittest

from judge_score_geometry import GeometryInputError, analyze_score_geometry, validate_input


def _clip(value: int) -> int:
    return max(0, min(1_000_000, value))


def make_input(
    count: int = 30,
    *,
    aligned: bool = False,
    constant_judges: bool = False,
    bootstrap_resamples: int = 100,
) -> dict:
    items = []
    for index in range(count):
        x = 200_000 + ((index * 7919) % 600_000)
        y = 400_000 + ((index * 317) % 60_000)
        z = 450_000 + ((index * 521) % 40_000)
        humans = []
        judges = []
        for rater in range(3):
            offset = (rater - 1) * 2500
            human_scores = [
                _clip(x + offset),
                _clip(y + offset * 2),
                _clip(z - offset),
            ]
            humans.append({
                "rater_id_sha256": f"{rater + 1:064x}",
                "scores_millionths": human_scores,
            })
            if aligned:
                judge_scores = human_scores[:]
            elif constant_judges:
                judge_scores = [500_000, 500_000, 500_000]
            else:
                # Humans vary primarily on criterion 0; judges vary primarily
                # on criterion 1 with a narrower range.
                judge_scores = [
                    _clip(500_000 + offset // 10),
                    _clip(500_000 + ((x - 500_000) // 3) + offset),
                    _clip(500_000 - offset // 10),
                ]
            judges.append({
                "rater_id_sha256": f"{rater + 11:064x}",
                "scores_millionths": judge_scores,
            })
        items.append({
            "item_id": f"item-{index:03d}",
            "task_family": "objective" if index % 2 == 0 else "subjective",
            "judge_ratings": judges,
            "human_ratings": humans,
        })
    return {
        "schema": "symthaea.science_bench_refresh_judge_score_geometry_input.v1",
        "purpose": "example_only",
        "analysis_id": "score-geometry-synthetic-001",
        "judge_source_revision": "NOT_IMPLEMENTED_YET",
        "human_protocol_sha256": "a" * 64,
        "rubric_sha256": "b" * 64,
        "criterion_ids": ["accuracy", "evidence", "clarity"],
        "analysis_config": {
            "bootstrap_resamples": bootstrap_resamples,
            "min_items_for_bootstrap": 20,
            "min_items_per_task_family": 10,
            "jacobi_tolerance": 1e-8,
        },
        "items": items,
    }


def encode(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


class JudgeScoreGeometryTests(unittest.TestCase):
    def test_orthogonal_compressed_judges_are_detected(self) -> None:
        report = analyze_score_geometry(encode(make_input()))
        self.assertTrue(report["valid"])
        self.assertEqual(report["item_count"], 30)
        self.assertEqual(report["criterion_count"], 3)
        self.assertEqual(report["pairwise_agreement"]["judge_judge"]["pair_count"], 3)
        self.assertEqual(report["pairwise_agreement"]["judge_human"]["pair_count"], 9)
        self.assertEqual(report["pairwise_agreement"]["human_human"]["pair_count"], 3)
        self.assertEqual(report["matched_item_count"], 30)
        self.assertEqual(report["judge_rater_count"], 3)
        self.assertEqual(report["human_rater_count"], 3)
        self.assertEqual(report["pairwise_agreement"]["judge_human"]["criterion_comparison_count"], 27)
        self.assertLess(report["judge_panel_mean_score_spread_ratio"], 0.6)
        self.assertGreater(report["leading_axis_angle_degrees"], 75.0)
        self.assertTrue(report["judge_leading_axis_identifiable"])
        self.assertTrue(report["human_leading_axis_identifiable"])
        self.assertFalse(report["generalized_accuracy_estimated"])
        self.assertFalse(report["judge_quality_established"])
        self.assertFalse(report["scoring_eligible_declared"])
        self.assertTrue(report["external_anchor_required"])
        self.assertEqual(len(report["task_family_summaries"]), 2)
        self.assertTrue(all(row["evaluable"] for row in report["task_family_summaries"]))

    def test_aligned_panels_have_zero_axis_angle_and_unit_spread(self) -> None:
        report = analyze_score_geometry(encode(make_input(aligned=True)))
        self.assertAlmostEqual(report["judge_panel_mean_score_spread_ratio"], 1.0, places=12)
        self.assertAlmostEqual(report["leading_axis_angle_degrees"], 0.0, places=6)

    def test_constant_judges_have_null_correlations_and_unidentifiable_axis(self) -> None:
        report = analyze_score_geometry(encode(make_input(constant_judges=True)))
        judge_judge = report["pairwise_agreement"]["judge_judge"]
        self.assertEqual(judge_judge["pair_count"], 3)
        self.assertEqual(judge_judge["defined_pair_count"], 0)
        self.assertEqual(judge_judge["undefined_pair_count"], 3)
        self.assertIsNone(judge_judge["mean_pairwise_pearson"])
        self.assertIsNone(judge_judge["mean_pairwise_spearman"])
        self.assertFalse(report["judge_leading_axis_identifiable"])
        self.assertIsNone(report["leading_axis_angle_degrees"])
        self.assertIsNone(report["judge_effective_rank"])

    def test_spearman_is_reported_separately_from_pearson(self) -> None:
        report = analyze_score_geometry(encode(make_input(aligned=True)))
        pairwise = report["pairwise_agreement"]["judge_human"]
        self.assertEqual(pairwise["pair_count"], 9)
        self.assertEqual(pairwise["defined_pair_count"], 9)
        self.assertEqual(pairwise["spearman_defined_pair_count"], 9)
        self.assertAlmostEqual(pairwise["mean_pairwise_spearman"], 1.0, places=12)

    def test_item_cluster_bootstrap_is_deterministic(self) -> None:
        payload = encode(make_input(bootstrap_resamples=120))
        first = analyze_score_geometry(payload)
        second = analyze_score_geometry(payload)
        self.assertEqual(first, second)
        self.assertEqual(first["bootstrap"]["planned_resamples"], 120)
        self.assertEqual(first["bootstrap"]["valid_spread_resamples"], 120)
        self.assertEqual(first["bootstrap"]["resampling_unit"], "item_cluster")
        self.assertEqual(len(first["bootstrap"]["score_spread_ratio_ci_95"]), 2)
        self.assertEqual(len(first["bootstrap"]["leading_axis_angle_ci_95"]), 2)

    def test_bootstrap_is_null_below_preregistered_minimum(self) -> None:
        data = make_input(count=12, bootstrap_resamples=50)
        report = analyze_score_geometry(encode(data))
        self.assertIsNone(report["bootstrap"]["score_spread_ratio_ci_95"])
        self.assertIsNone(report["bootstrap"]["leading_axis_angle_ci_95"])
        self.assertIn("below min_items_for_bootstrap", report["bootstrap"]["not_evaluable_reason"])

    def test_small_task_family_stratum_is_not_given_fake_estimate(self) -> None:
        data = make_input(count=30)
        data["analysis_config"]["min_items_per_task_family"] = 20
        report = analyze_score_geometry(encode(data))
        self.assertTrue(all(not row["evaluable"] for row in report["task_family_summaries"]))
        self.assertTrue(all(row["score_spread_ratio"] is None for row in report["task_family_summaries"]))

    def test_duplicate_item_id_is_rejected(self) -> None:
        data = make_input()
        data["items"][1]["item_id"] = data["items"][0]["item_id"]
        with self.assertRaises(GeometryInputError):
            validate_input(data)

    def test_duplicate_rater_id_within_panel_is_rejected(self) -> None:
        data = make_input()
        data["items"][0]["judge_ratings"][1]["rater_id_sha256"] = data["items"][0]["judge_ratings"][0]["rater_id_sha256"]
        with self.assertRaises(GeometryInputError):
            validate_input(data)

    def test_rater_id_cannot_be_shared_between_judge_and_human_panels(self) -> None:
        data = make_input()
        data["items"][0]["human_ratings"][0]["rater_id_sha256"] = data["items"][0]["judge_ratings"][0]["rater_id_sha256"]
        with self.assertRaises(GeometryInputError):
            validate_input(data)

    def test_invalid_criterion_vector_dimension_is_rejected(self) -> None:
        data = make_input()
        data["items"][0]["human_ratings"][0]["scores_millionths"].pop()
        with self.assertRaises(GeometryInputError):
            validate_input(data)

    def test_out_of_range_and_boolean_scores_are_rejected(self) -> None:
        for score in (1_000_001, True, -1):
            data = make_input()
            data["items"][0]["judge_ratings"][0]["scores_millionths"][0] = score
            with self.subTest(score=score), self.assertRaises(GeometryInputError):
                validate_input(data)

    def test_unhashable_purpose_is_rejected_not_raised(self) -> None:
        data = make_input()
        data["purpose"] = ["example_only"]
        with self.assertRaises(GeometryInputError):
            validate_input(data)

    def test_unknown_fields_are_rejected(self) -> None:
        data = make_input()
        data["hidden_oracle"] = True
        with self.assertRaises(GeometryInputError):
            validate_input(data)

    def test_report_is_fixed_matrix_only_and_not_scoring_eligible(self) -> None:
        report = analyze_score_geometry(encode(make_input()))
        self.assertTrue(report["fixed_score_matrix_only"])
        self.assertFalse(report["item_population_sampling_claimed"])
        self.assertFalse(report["irt_model_fitted"])
        self.assertFalse(report["construct_validity_proven"])
        self.assertFalse(report["judge_quality_established"])
        self.assertFalse(report["scoring_eligible_declared"])


if __name__ == "__main__":
    unittest.main()
