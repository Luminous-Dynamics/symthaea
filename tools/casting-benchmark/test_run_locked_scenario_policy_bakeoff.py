"""Contract tests for the locked casting scenario runner harness."""
from __future__ import annotations

import unittest
from pathlib import Path

from run_locked_scenario_policy_bakeoff import output_outside_repo, validate_output_contract


def preregistration() -> dict:
    return {
        "benchmark_id": "CASTING-MF-POLICY-SCENARIOS-001",
        "design": {
            "scenario_ids": ["smooth_stationary", "local_bias_island", "warped_coordinates", "constraint_boundary", "discontinuous_step", "heteroscedastic_noise"],
            "dimensions": [2, 5, 10],
            "severities": [0.5, 1, 2],
            "paired_seeds": [17, 23, 29],
            "design_count": 128,
            "cost_budget": 200,
            "fidelity_costs": {"low": 1, "medium": 5, "high": 25},
            "policy_ids": ["high_only", "low_screen", "fixed_top_k", "discrepancy_aware", "adaptive_budget"],
            "expected_run_count": 810,
            "expected_cell_summary_count": 270,
            "expected_paired_comparison_count": 216,
        },
    }


def valid_result() -> dict:
    p = preregistration()
    d = p["design"]
    return {
        "benchmark_id": p["benchmark_id"],
        "synthetic_fixture_only": True,
        "scenario_ids": d["scenario_ids"],
        "dimensions": d["dimensions"],
        "severities": d["severities"],
        "seeds": d["paired_seeds"],
        "design_count": d["design_count"],
        "cost_budget": d["cost_budget"],
        "fidelity_costs": d["fidelity_costs"],
        "policies": d["policy_ids"],
        "run_count": d["expected_run_count"],
        "summary_count": d["expected_cell_summary_count"],
        "paired_comparison_count": d["expected_paired_comparison_count"],
        "paired_runs": [{} for _ in range(d["expected_run_count"])],
    }


class LockedRunnerContractTests(unittest.TestCase):
    def test_exact_preregistered_output_contract_passes(self) -> None:
        validate_output_contract(valid_result(), preregistration())

    def test_wrong_run_count_fails_closed(self) -> None:
        result = valid_result()
        result["run_count"] = 809
        with self.assertRaisesRegex(ValueError, "run_count"):
            validate_output_contract(result, preregistration())

    def test_missing_raw_run_ledger_fails_closed(self) -> None:
        result = valid_result()
        result.pop("paired_runs")
        with self.assertRaisesRegex(ValueError, "paired_runs"):
            validate_output_contract(result, preregistration())

    def test_changed_fidelity_costs_fail_closed(self) -> None:
        result = valid_result()
        result["fidelity_costs"] = {"low": 1, "medium": 10, "high": 50}
        with self.assertRaisesRegex(ValueError, "fidelity_costs"):
            validate_output_contract(result, preregistration())

    def test_repository_output_path_is_refused(self) -> None:
        root = Path("/tmp/casting-lock-test-repo").resolve()
        with self.assertRaisesRegex(ValueError, "outside the repository"):
            output_outside_repo(root / "results.json", root, "output")


if __name__ == "__main__":
    unittest.main()
