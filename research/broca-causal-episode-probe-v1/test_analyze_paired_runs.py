"""Tests for grouped paired-run analysis and fail-closed pairing."""
from __future__ import annotations

import copy
import unittest

from analyze_paired_runs import analyze_paired_runs, validate_run


def make_row(
    question_id: str,
    group_id: str,
    correct: bool,
    *,
    domain: str = "web",
    status: str = "success",
    latency: float = 100.0,
    context: int = 100,
    cost: float = 1.0,
) -> dict[str, object]:
    return {
        "domain": domain,
        "question_id": question_id,
        "group_id": group_id,
        "status": status,
        "correct": correct,
        "query_latency_ms": latency,
        "context_tokens": context,
        "lifecycle_cost_units": cost,
    }


class PairedRunAnalysisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.baseline = [
            make_row("q1", "session-a", True),
            make_row("q2", "session-a", False),
            make_row("q3", "session-b", True),
            make_row("q4", "session-b", False),
            make_row("q5", "session-c", False),
            make_row("q6", "session-c", True),
        ]
        self.candidate = copy.deepcopy(self.baseline)
        self.candidate[1]["correct"] = True
        self.candidate[2]["correct"] = False
        self.candidate[3]["correct"] = True
        self.candidate[4]["correct"] = True

    def test_paired_delta_and_wins_losses_are_correct(self) -> None:
        report = analyze_paired_runs(
            self.baseline, self.candidate, bootstrap_replicates=500, seed=17
        )
        self.assertEqual(report["question_count"], 6)
        self.assertEqual(report["group_count"], 3)
        self.assertEqual(report["accuracy"]["baseline_percent"], 50.0)
        self.assertEqual(report["accuracy"]["candidate_percent"], 83.3333)
        self.assertEqual(report["accuracy"]["delta_percentage_points"], 33.3333)
        self.assertEqual(report["accuracy"]["paired_wins"], 3)
        self.assertEqual(report["accuracy"]["paired_losses"], 1)
        self.assertEqual(report["accuracy"]["paired_ties"], 2)

    def test_group_bootstrap_is_deterministic_for_fixed_seed(self) -> None:
        a = analyze_paired_runs(self.baseline, self.candidate, bootstrap_replicates=500, seed=99)
        b = analyze_paired_runs(self.baseline, self.candidate, bootstrap_replicates=500, seed=99)
        self.assertEqual(a["accuracy"]["delta_ci95_percentage_points"], b["accuracy"]["delta_ci95_percentage_points"])
        self.assertEqual(a["lifecycle_cost"]["reduction_ci95_percent"], b["lifecycle_cost"]["reduction_ci95_percent"])

    def test_exact_question_pairing_rejects_missing_and_extra_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "paired question keys differ"):
            analyze_paired_runs(self.baseline, self.candidate[:-1], bootstrap_replicates=100)

    def test_pairing_rejects_group_mismatch(self) -> None:
        wrong = copy.deepcopy(self.candidate)
        wrong[0]["group_id"] = "different-session"
        with self.assertRaisesRegex(ValueError, "group_id mismatch"):
            analyze_paired_runs(self.baseline, wrong, bootstrap_replicates=100)

    def test_failed_rows_are_never_silently_dropped(self) -> None:
        bad = copy.deepcopy(self.baseline)
        bad[0]["status"] = "timeout"
        with self.assertRaisesRegex(ValueError, "failed rows cannot be silently dropped"):
            validate_run(bad, "baseline")

    def test_duplicate_domain_question_keys_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate domain/question key"):
            validate_run(self.baseline + [copy.deepcopy(self.baseline[0])], "baseline")

    def test_correct_must_be_boolean_not_integer(self) -> None:
        bad = copy.deepcopy(self.baseline)
        bad[0]["correct"] = 1
        with self.assertRaisesRegex(ValueError, "correct must be a JSON boolean"):
            validate_run(bad, "baseline")

    def test_nonfinite_latency_fails_closed(self) -> None:
        bad = copy.deepcopy(self.baseline)
        bad[0]["query_latency_ms"] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            validate_run(bad, "baseline")

    def test_efficiency_gate_requires_noninferior_quality_and_cost_reduction(self) -> None:
        baseline = [make_row(f"q{i}", f"s{i // 2}", i % 2 == 0, cost=10.0) for i in range(12)]
        candidate = [make_row(f"q{i}", f"s{i // 2}", i % 2 == 0, cost=5.0, latency=70.0) for i in range(12)]
        report = analyze_paired_runs(
            baseline,
            candidate,
            bootstrap_replicates=500,
            seed=3,
            cost_reduction_percent=25.0,
            quality_noninferiority_pp=1.0,
        )
        self.assertEqual(report["accuracy"]["delta_percentage_points"], 0.0)
        self.assertEqual(report["lifecycle_cost"]["reduction_percent"], 50.0)
        self.assertTrue(report["decision_gates"]["efficiency"]["pass"])
        self.assertFalse(report["decision_gates"]["quality"]["pass"])

    def test_zero_baseline_lifecycle_cost_cannot_define_reduction(self) -> None:
        baseline = [make_row("q1", "s1", True, cost=0.0)]
        candidate = [make_row("q1", "s1", True, cost=0.0)]
        with self.assertRaisesRegex(ValueError, "baseline total lifecycle cost must be positive"):
            analyze_paired_runs(baseline, candidate, bootstrap_replicates=100)

    def test_too_few_bootstrap_replicates_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "bootstrap_replicates"):
            analyze_paired_runs(self.baseline, self.candidate, bootstrap_replicates=99)


if __name__ == "__main__":
    unittest.main()
