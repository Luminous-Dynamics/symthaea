#!/usr/bin/env python3
"""Guard VLEB scorecard semantics so targets/comparators cannot become fake results."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from typing import Any

SCORECARD = Path(__file__).with_name("scorecard.json")


def validate_scorecard(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if data.get("schema_version") != "vleb-scorecard-v1":
        errors.append("schema_version")
    if data.get("project_status") != "development_prototype":
        errors.append("project_status")

    legacy = data.get("industry_comparators", {}).get("legacy_bench", {})
    if legacy.get("metric") != "avg@3" or not legacy.get("source_url"):
        errors.append("legacy_bench_protocol")
    entries = legacy.get("entries", [])
    if not entries or any(not isinstance(row.get("score_pct"), (int, float)) or not 0 <= row["score_pct"] <= 100 for row in entries):
        errors.append("legacy_bench_scores")
    if any("pass@1" == row.get("metric") for row in entries):
        errors.append("legacy_bench_metric_mislabeled")

    targets = data.get("proposed_targets", {})
    for target_name, target in targets.items():
        if target.get("status") != "target_not_measured":
            errors.append("target_status:" + target_name)
        if "target" not in target:
            errors.append("target_missing_value:" + target_name)

    current = data.get("current_measurements", {})
    for measure_name, measurement in current.items():
        if "value" not in measurement:
            continue
        value = measurement.get("value")
        if value is None and measurement.get("eligible") is not False:
            errors.append("null_measurement_eligible:" + measure_name)
        if value is not None and measurement.get("eligible") is True and not measurement.get("evidence_artifact_url"):
            errors.append("eligible_measurement_missing_artifact:" + measure_name)

    smoke = current.get("active_synthetic_mutation_smoke_test", {})
    active = smoke.get("active_mutants")
    counts = [smoke.get("compilable_mutants"), smoke.get("killed"), smoke.get("survived"), smoke.get("unresolved")]
    if not isinstance(active, int) or any(not isinstance(n, int) or n < 0 for n in counts):
        errors.append("mutation_counts_invalid")
    elif sum(counts[1:]) != active or counts[0] != active:
        errors.append("mutation_counts_inconsistent")
    if smoke.get("scope") != "single public synthetic ledger fixture" or smoke.get("eligible_as_broad_mutation_score") is not False:
        errors.append("synthetic_mutation_result_overclaimed")
    if smoke.get("killed") == smoke.get("active_mutants") and smoke.get("eligible_as_broad_mutation_score") is True:
        errors.append("all_mutants_killed_overclaimed")

    v2 = data.get("industry_comparators", {}).get("swe_bench_pro_v2", {})
    if v2.get("task_count") != 642 or v2.get("removed_invalid_tasks") != 89:
        errors.append("swe_bench_pro_v2_reference_drift")
    security = data.get("industry_comparators", {}).get("agent_security_league", {})
    if security.get("top_listed_functional_pct") != 87.2 or security.get("top_listed_security_correctness_pct") != 37.4:
        errors.append("security_comparator_reference_drift")
    dual_gauge = data.get("industry_comparators", {}).get("dual_gauge", {})
    if dual_gauge.get("task_count") != 307 or dual_gauge.get("reported_best_joint_functional_security_success_pct_upper_bound") != 15.0:
        errors.append("dual_gauge_reference_drift")
    last_secure = data.get("industry_comparators", {}).get("last_secure_code_benchmark", {})
    if last_secure.get("task_count") != 450 or last_secure.get("reported_best_joint_functional_security_success_pct") != 25.1:
        errors.append("last_secure_benchmark_reference_drift")
    return errors


class ScorecardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scorecard = json.loads(SCORECARD.read_text(encoding="utf-8"))

    def test_current_scorecard_is_valid(self) -> None:
        self.assertEqual(validate_scorecard(self.scorecard), [])

    def test_cost_frontier_target_is_lower_cost_than_listed_leader(self) -> None:
        target = self.scorecard["proposed_targets"]["legacy_bench_cost_frontier"]
        leader = self.scorecard["industry_comparators"]["legacy_bench"]["entries"][0]
        self.assertEqual(target["status"], "target_not_measured")
        self.assertEqual(target["metric"], "avg@3")
        self.assertGreaterEqual(target["target"], 60.0)
        self.assertLessEqual(target["max_cost_usd_per_task"], leader["cost_usd_per_task"])

    def test_industry_leader_does_not_become_symthaea_measurement(self) -> None:
        self.assertIsNone(self.scorecard["current_measurements"]["legacy_bench_avg_at_3_pct"]["value"])
        self.assertFalse(self.scorecard["current_measurements"]["legacy_bench_avg_at_3_pct"]["eligible"])

    def test_proposed_target_cannot_be_marked_as_measured(self) -> None:
        data = copy.deepcopy(self.scorecard)
        data["proposed_targets"]["legacy_bench_avg_at_3_pct"]["status"] = "measured"
        self.assertIn("target_status:legacy_bench_avg_at_3_pct", validate_scorecard(data))

    def test_four_mutants_are_scoped_to_smoke_test_not_broad_claim(self) -> None:
        smoke = self.scorecard["current_measurements"]["active_synthetic_mutation_smoke_test"]
        self.assertEqual((smoke["active_mutants"], smoke["killed"], smoke["survived"], smoke["unresolved"]), (4, 4, 0, 0))
        self.assertFalse(smoke["eligible_as_broad_mutation_score"])

    def test_inconsistent_mutation_counts_are_rejected(self) -> None:
        data = copy.deepcopy(self.scorecard)
        data["current_measurements"]["active_synthetic_mutation_smoke_test"]["killed"] = 3
        self.assertIn("mutation_counts_inconsistent", validate_scorecard(data))

    def test_comparator_cannot_be_mislabeled_as_pass_at_one(self) -> None:
        data = copy.deepcopy(self.scorecard)
        data["industry_comparators"]["legacy_bench"]["metric"] = "pass@1"
        self.assertIn("legacy_bench_protocol", validate_scorecard(data))

    def test_unknown_scored_security_result_is_not_eligible(self) -> None:
        security = self.scorecard["current_measurements"]["security_correctness_pct"]
        self.assertIsNone(security["value"])
        self.assertFalse(security["eligible"])

    def test_joint_security_functional_success_remains_unmeasured(self) -> None:
        joint = self.scorecard["current_measurements"]["security_joint_functional_success_pct"]
        self.assertIsNone(joint["value"])
        self.assertFalse(joint["eligible"])

    def test_security_comparators_are_not_vleb_results(self) -> None:
        self.assertIsNone(self.scorecard["current_measurements"]["security_correctness_pct"]["value"])
        self.assertEqual(self.scorecard["industry_comparators"]["dual_gauge"]["task_count"], 307)
        self.assertEqual(self.scorecard["industry_comparators"]["last_secure_code_benchmark"]["task_count"], 450)


if __name__ == "__main__":
    unittest.main(verbosity=2)
