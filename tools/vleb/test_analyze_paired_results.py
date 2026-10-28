from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator
from analyze_paired_results import analyze, _exact_mcnemar_p, _wilson


def sha(char: str) -> str:
    return char * 64


def outcome(result: str, cost: float = 1.0, duration: int = 100) -> dict:
    return {"outcome": result, "cost_usd": cost, "duration_ms": duration}


def valid_input() -> dict:
    return {
        "schema_version": "vleb-paired-results-v1",
        "cohort_id": "pilot-locked-001",
        "protocol_sha256": sha("a"),
        "analysis_plan": {
            "primary_metric": "assurance_adjusted_semantic_success_rate",
            "pairwise_test": "exact_mcnemar_two_sided",
            "confidence_level": 0.95,
            "attempt_aggregation": "One task-level outcome from exactly three preregistered attempts (pass@3); attempts are not treated as independent tasks.",
            "cost_currency": "USD",
        },
        "candidates": {
            "A": {"name": "baseline-agent", "configuration_sha256": sha("b")},
            "B": {"name": "candidate-agent", "configuration_sha256": sha("c")},
        },
        "tasks": [
            {"task_id": "T-001", "lane": "repair", "candidate_A": outcome("assured_pass", 0.5), "candidate_B": outcome("fail", 0.2), "exclusion": None},
            {"task_id": "T-002", "lane": "repair", "candidate_A": outcome("raw_pass_unverified", 0.3), "candidate_B": outcome("assured_pass", 0.4), "exclusion": None},
            {"task_id": "T-003", "lane": "migration", "candidate_A": outcome("unresolved", 0.7), "candidate_B": outcome("abstained", 0.1), "exclusion": None},
            {"task_id": "T-004", "lane": "migration", "candidate_A": outcome("fail", 0.2), "candidate_B": outcome("fail", 0.3), "exclusion": None},
        ],
    }


SCHEMA_PATH = Path(__file__).with_name("paired-results.schema.json")


class PairedResultsTests(unittest.TestCase):
    def test_json_schema_is_valid_and_accepts_positive_control(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(schema)
        self.assertEqual(list(validator.iter_errors(valid_input())), [])
        unknown = valid_input()
        unknown["unexpected"] = True
        self.assertNotEqual(list(validator.iter_errors(unknown)), [])

    def test_analysis_output_never_grants_score_eligibility(self) -> None:
        result = analyze(valid_input())
        self.assertTrue(result["analysis_only"])
        self.assertFalse(result["score_eligible"])

    def test_aassr_excludes_raw_unverified_passes(self) -> None:
        result = analyze(valid_input())
        self.assertEqual(result["candidate_A"]["aassr"], 0.25)
        self.assertEqual(result["candidate_A"]["raw_resolution_rate"], 0.5)

    def test_unresolved_and_abstained_stay_in_denominator(self) -> None:
        result = analyze(valid_input())
        self.assertEqual(result["candidate_A"]["n"], 4)
        self.assertEqual(result["candidate_A"]["outcome_counts"]["unresolved"], 1)
        self.assertEqual(result["candidate_B"]["outcome_counts"]["abstained"], 1)

    def test_exact_mcnemar_is_paired_and_two_sided(self) -> None:
        result = analyze(valid_input())
        paired = result["paired_aassr_comparison"]
        self.assertEqual(paired["candidate_A_only_assured_pass"], 1)
        self.assertEqual(paired["candidate_B_only_assured_pass"], 1)
        self.assertEqual(paired["exact_mcnemar_p_value_two_sided"], 1.0)
        self.assertEqual(_exact_mcnemar_p(6, 0), 0.03125)

    def test_identical_candidates_have_no_measured_difference(self) -> None:
        data = valid_input()
        for task in data["tasks"]:
            task["candidate_B"] = copy.deepcopy(task["candidate_A"])
        result = analyze(data)
        self.assertEqual(result["paired_aassr_comparison"]["aassr_risk_difference_A_minus_B"], 0)
        self.assertEqual(result["paired_aassr_comparison"]["exact_mcnemar_p_value_two_sided"], 1.0)

    def test_pre_unblinding_exclusion_removes_both_arms_symmetrically(self) -> None:
        data = valid_input()
        data["tasks"][0]["exclusion"] = {
            "reason": "pre-registered dependency image unavailable",
            "adjudicated_before_unblinding": True,
            "adjudication_sha256": sha("d"),
        }
        result = analyze(data)
        self.assertEqual(result["assignment_accounting"]["assigned_tasks"], 4)
        self.assertEqual(result["assignment_accounting"]["scored_tasks"], 3)
        self.assertEqual(result["assignment_accounting"]["pre_unblinding_exclusions"], 1)
        self.assertEqual(result["candidate_A"]["n"], result["candidate_B"]["n"])

    def test_exclusion_after_unblinding_is_rejected(self) -> None:
        data = valid_input()
        data["tasks"][0]["exclusion"] = {
            "reason": "bad result",
            "adjudicated_before_unblinding": False,
            "adjudication_sha256": sha("d"),
        }
        with self.assertRaisesRegex(ValueError, "exclusion_not_adjudicated_blind"):
            analyze(data)

    def test_duplicate_task_ids_are_rejected(self) -> None:
        data = valid_input()
        data["tasks"][1]["task_id"] = data["tasks"][0]["task_id"]
        with self.assertRaisesRegex(ValueError, "duplicate_task_id"):
            analyze(data)

    def test_missing_candidate_outcome_is_rejected(self) -> None:
        data = valid_input()
        del data["tasks"][0]["candidate_B"]
        with self.assertRaisesRegex(ValueError, "expected_object"):
            analyze(data)

    def test_unknown_fields_fail_closed(self) -> None:
        data = valid_input()
        data["unexpected"] = "silently ignored must not happen"
        with self.assertRaisesRegex(ValueError, "unknown_field"):
            analyze(data)

    def test_excluded_task_cost_still_counts_toward_total_economics(self) -> None:
        data = valid_input()
        data["tasks"][0]["exclusion"] = {
            "reason": "pre-registered dependency image unavailable",
            "adjudicated_before_unblinding": True,
            "adjudication_sha256": sha("d"),
        }
        result = analyze(data)
        # A's costs across all four assigned episodes: 0.5 + 0.3 + 0.7 + 0.2.
        self.assertAlmostEqual(result["candidate_A"]["total_cost_usd"], 1.7)

    def test_invalid_configuration_digest_is_rejected(self) -> None:
        data = valid_input()
        data["candidates"]["A"]["configuration_sha256"] = "NOT-A-DIGEST"
        with self.assertRaisesRegex(ValueError, "invalid_identity_or_configuration_pin"):
            analyze(data)

    def test_negative_and_non_finite_costs_are_rejected(self) -> None:
        data = valid_input()
        data["tasks"][0]["candidate_A"]["cost_usd"] = -0.01
        with self.assertRaisesRegex(ValueError, "invalid_cost"):
            analyze(data)
        data = valid_input()
        data["tasks"][0]["candidate_A"]["cost_usd"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid_cost"):
            analyze(data)

    def test_boolean_duration_is_not_an_integer_duration(self) -> None:
        data = valid_input()
        data["tasks"][0]["candidate_A"]["duration_ms"] = True
        with self.assertRaisesRegex(ValueError, "invalid_duration"):
            analyze(data)

    def test_cost_per_verified_pass_is_undefined_with_zero_passes(self) -> None:
        data = valid_input()
        for task in data["tasks"]:
            task["candidate_A"]["outcome"] = "fail"
        result = analyze(data)
        self.assertIsNone(result["candidate_A"]["cost_per_independently_verified_pass_usd"])

    def test_all_excluded_cohort_is_rejected(self) -> None:
        data = valid_input()
        for task in data["tasks"]:
            task["exclusion"] = {"reason": "pre-run", "adjudicated_before_unblinding": True, "adjudication_sha256": sha("e")}
        with self.assertRaisesRegex(ValueError, "no tasks remain"):
            analyze(data)

    def test_wilson_interval_handles_all_pass_and_zero_sample(self) -> None:
        interval = _wilson(10, 10, 0.95)
        self.assertIsNotNone(interval)
        self.assertLess(interval[0], 1.0)
        self.assertEqual(interval[1], 1.0)
        self.assertIsNone(_wilson(0, 0, 0.95))

    def test_per_lane_results_are_reported(self) -> None:
        result = analyze(valid_input())
        self.assertEqual(set(result["by_lane"]), {"repair", "migration"})
        self.assertEqual(result["by_lane"]["repair"]["assigned_tasks"], 2)

    def test_incompatible_analysis_plan_is_rejected(self) -> None:
        data = valid_input()
        data["analysis_plan"]["primary_metric"] = "average_weighted_score"
        with self.assertRaisesRegex(ValueError, "primary_metric_must_be_aassr"):
            analyze(data)

    def test_unregistered_attempt_aggregation_is_rejected(self) -> None:
        data = valid_input()
        data["analysis_plan"]["attempt_aggregation"] = " "
        with self.assertRaisesRegex(ValueError, "attempt_aggregation_must_be_preregistered"):
            analyze(data)


if __name__ == "__main__":
    unittest.main(verbosity=2)
