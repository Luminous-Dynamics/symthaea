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
    definitions = data.get("metric_definitions", {})
    aassr = definitions.get("assurance_adjusted_semantic_success_rate", {})
    if aassr.get("formula") != "independently_verified_passes / all tasks assigned to the locked confirmatory cohort":
        errors.append("aassr_formula_drift")
    if aassr.get("composite_weighted_score") is not False:
        errors.append("aassr_must_not_hide_failures_in_weighted_score")
    if not any("failed, unresolved and abstained" in rule for rule in aassr.get("denominator_rules", [])):
        errors.append("aassr_denominator_drops_unresolved")
    cost = definitions.get("cost_per_independently_verified_fix", {})
    if "undefined" not in cost.get("zero_pass_rule", ""):
        errors.append("cost_zero_pass_semantics_missing")
    dual_gauge = data.get("industry_comparators", {}).get("dual_gauge", {})
    if dual_gauge.get("task_count") != 307 or dual_gauge.get("reported_best_joint_functional_security_success_pct_upper_bound") != 15.0:
        errors.append("dual_gauge_reference_drift")
    legacy_code = data.get("industry_comparators", {}).get("legacy_code_bench_2026", {})
    if legacy_code.get("task_count") != 200 or legacy_code.get("complexity_tiers") != 4:
        errors.append("legacy_code_bench_reference_drift")
    if legacy_code.get("report_status") != "benchmark-author-reported":
        errors.append("legacy_code_bench_result_attribution_missing")
    ibm = data.get("industry_comparators", {}).get("ibm_enterprise_cobol_to_java_icse_2026", {})
    if ibm.get("public_programs") != 20 or ibm.get("proprietary_customer_applications") != 10:
        errors.append("ibm_cobol_translation_scope_drift")
    rank_audit = data.get("industry_comparators", {}).get("swe_bench_leaderboard_stability_audit", {})
    if rank_audit.get("audited_submission_count") != 254:
        errors.append("swe_bench_rank_audit_scope_drift")
    if rank_audit.get("findings", {}).get("adjacent_top_30_verified_pairs_significant_at_alpha_0_05") != 0:
        errors.append("rank_audit_result_drift")
    last_secure = data.get("industry_comparators", {}).get("last_secure_code_benchmark", {})
    if last_secure.get("task_count") != 450 or last_secure.get("reported_best_joint_functional_security_success_pct") != 25.1:
        errors.append("last_secure_benchmark_reference_drift")
    trace_target = data.get("proposed_targets", {}).get("requirement_to_test_traceability_coverage_pct", {})
    if trace_target.get("target") != 100 or trace_target.get("status") != "target_not_measured":
        errors.append("requirement_to_test_traceability_target_drift")
    trace_current = data.get("current_measurements", {}).get("requirement_to_test_traceability_coverage_pct", {})
    if trace_current.get("value") is not None or trace_current.get("eligible") is not False:
        errors.append("requirement_to_test_traceability_score_fabricated")
    trace_definition = data.get("metric_definitions", {}).get("requirement_to_test_traceability", {})
    if trace_definition.get("composite_weighted_score") is not False:
        errors.append("requirement_to_test_traceability_must_not_use_weighted_score")
    if not any("never remove a requirement or test" in rule for rule in trace_definition.get("denominator_rules", [])):
        errors.append("requirement_to_test_traceability_denominator_drift")

    for metric_name in ("requirement_source_item_coverage_pct", "required_test_trace_coverage_pct"):
        target = data.get("proposed_targets", {}).get(metric_name, {})
        current = data.get("current_measurements", {}).get(metric_name, {})
        if target.get("target") != 100 or target.get("status") != "target_not_measured":
            errors.append(metric_name + "_target_drift")
        if current.get("value") is not None or current.get("eligible") is not False:
            errors.append(metric_name + "_score_fabricated")

    family_target = data.get("proposed_targets", {}).get("benchmark_auditor_sentinel_family_kill_rate_pct", {})
    family_current = data.get("current_measurements", {}).get("benchmark_auditor_sentinel_family_kill_rate_pct", {})
    family_definition = data.get("metric_definitions", {}).get("benchmark_auditor_sentinel_family_kill_rate", {})
    if family_target.get("target") != 100 or family_target.get("status") != "target_not_measured":
        errors.append("auditor_sentinel_family_target_drift")
    if family_current.get("value") is not None or family_current.get("eligible") is not False:
        errors.append("auditor_sentinel_family_score_fabricated")
    if family_definition.get("composite_weighted_score") is not False:
        errors.append("auditor_sentinel_family_must_not_use_weighted_score")

    auditor_mutation_target = data.get("proposed_targets", {}).get("benchmark_auditor_sentinel_mutation_kill_rate_pct", {})
    if auditor_mutation_target.get("target") != 100 or auditor_mutation_target.get("status") != "target_not_measured":
        errors.append("auditor_sentinel_mutation_target_drift")
    auditor_mutation_current = data.get("current_measurements", {}).get("benchmark_auditor_sentinel_mutation_kill_rate_pct", {})
    if auditor_mutation_current.get("value") is not None or auditor_mutation_current.get("eligible") is not False:
        errors.append("auditor_sentinel_mutation_score_fabricated")
    auditor_mutation_definition = data.get("metric_definitions", {}).get("benchmark_auditor_sentinel_mutation_kill_rate", {})
    if auditor_mutation_definition.get("composite_weighted_score") is not False:
        errors.append("auditor_sentinel_mutation_must_not_use_weighted_score")

    baseline_target = data.get("proposed_targets", {}).get("per_test_baseline_oracle_alignment_pct", {})
    if baseline_target.get("target") != 100 or baseline_target.get("status") != "target_not_measured":
        errors.append("per_test_baseline_oracle_target_drift")
    baseline_current = data.get("current_measurements", {}).get("per_test_baseline_oracle_alignment_pct", {})
    if baseline_current.get("value") is not None or baseline_current.get("eligible") is not False:
        errors.append("per_test_baseline_oracle_score_fabricated")
    baseline_definition = data.get("metric_definitions", {}).get("per_test_baseline_oracle_alignment", {})
    if baseline_definition.get("composite_weighted_score") is not False:
        errors.append("per_test_baseline_oracle_must_not_use_weighted_score")
    if not any("cannot be dropped" in rule for rule in baseline_definition.get("denominator_rules", [])):
        errors.append("per_test_baseline_oracle_denominator_drift")

    claim_metric = data.get("current_measurements", {}).get("source_claim_false_support_rate_pct", {})
    if claim_metric.get("value") is not None or claim_metric.get("eligible") is not False:
        errors.append("source_claim_false_support_rate_fabricated")
    claim_target = data.get("proposed_targets", {}).get("source_claim_false_support_rate_pct", {})
    if claim_target.get("target") != 0 or claim_target.get("status") != "target_not_measured":
        errors.append("source_claim_false_support_target_drift")
    return errors


class ScorecardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.scorecard = json.loads(SCORECARD.read_text(encoding="utf-8"))

    def test_current_scorecard_is_valid(self) -> None:
        self.assertEqual(validate_scorecard(self.scorecard), [])

    def test_aassr_denominator_keeps_unresolved_and_abstentions(self) -> None:
        definition = self.scorecard["metric_definitions"]["assurance_adjusted_semantic_success_rate"]
        self.assertFalse(definition["composite_weighted_score"])
        rules = " ".join(definition["denominator_rules"])
        self.assertIn("remain non-passes", rules)
        self.assertIn("before arm assignment", rules)

    def test_zero_verified_passes_never_means_zero_cost(self) -> None:
        definition = self.scorecard["metric_definitions"]["cost_per_independently_verified_fix"]
        self.assertIn("undefined", definition["zero_pass_rule"])

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

    def test_legacy_comprehension_comparators_are_context_not_vleb_scores(self) -> None:
        legacy = self.scorecard["industry_comparators"]["legacy_code_bench_2026"]
        ibm = self.scorecard["industry_comparators"]["ibm_enterprise_cobol_to_java_icse_2026"]
        self.assertEqual(legacy["task_count"], 200)
        self.assertEqual(ibm["public_programs"], 20)
        self.assertIsNone(self.scorecard["current_measurements"]["assurance_adjusted_semantic_success_rate_pct"]["value"])

    def test_leaderboard_stability_audit_requires_paired_uncertainty(self) -> None:
        audit = self.scorecard["industry_comparators"]["swe_bench_leaderboard_stability_audit"]
        self.assertEqual(audit["audited_submission_count"], 254)
        self.assertEqual(audit["findings"]["adjacent_top_30_verified_pairs_significant_at_alpha_0_05"], 0)
        self.assertIn("observational", audit["report_status"])

    def test_claim_outcome_accounting_and_witness_receipts_are_targets(self) -> None:
        targets = self.scorecard["proposed_targets"]
        for key in (
            "claim_outcome_accounting_completeness_pct",
            "evaluator_reproduced_witness_receipt_coverage_pct",
        ):
            self.assertEqual(targets[key]["target"], 100)
            self.assertEqual(targets[key]["status"], "target_not_measured")
        gate = self.scorecard["current_measurements"]["claim_evidence_gate"]
        self.assertIsNone(gate["value"])
        self.assertFalse(gate["eligible"])
        self.assertEqual(gate["status"], "draft_not_validated")

    def test_confirmatory_aassr_requires_claim_evidence_gate(self) -> None:
        requirements = self.scorecard["metric_definitions"]["assurance_adjusted_semantic_success_rate"]["pass_requirements"]
        self.assertTrue(any("claim inventory" in item and "claim-evidence audit" in item for item in requirements))

    def test_fcv_comparator_is_threat_specific_not_universal(self) -> None:
        comparator = self.scorecard["industry_comparators"]["google_fcv_patch_study"]
        self.assertEqual(comparator["evaluated_agent_model_combinations"], 12)
        self.assertEqual(comparator["example_cwe"], "CWE-538")
        self.assertIn("not a universal vulnerability rate", comparator["interpretation"])

    def test_claim_false_support_is_a_target_not_an_achieved_result(self) -> None:
        measurement = self.scorecard["current_measurements"]["source_claim_false_support_rate_pct"]
        target = self.scorecard["proposed_targets"]["source_claim_false_support_rate_pct"]
        self.assertIsNone(measurement["value"])
        self.assertFalse(measurement["eligible"])
        self.assertEqual(target["target"], 0)
        self.assertEqual(target["status"], "target_not_measured")


if __name__ == "__main__":
    unittest.main(verbosity=2)
