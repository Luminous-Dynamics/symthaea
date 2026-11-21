#!/usr/bin/env python3
"""Independent stdlib-only oracle for orbital structural evidence VOI fixtures."""
import hashlib
import json
import math
import sys

SCHEMA = "luminous.orbital-joint-evidence-voi.scenarios.v0.1"
CLAIM = "synthetic_decision_value_semantics_only"
TOL = 1.0e-9


def regret_for(loss_matrix, decisions, state, decision):
    return loss_matrix[decision][state] - min(loss_matrix[d][state] for d in decisions)


def minimax_regret(scenario):
    decisions = sorted(scenario["loss_matrix"])
    states = scenario["states"]
    return min(max(regret_for(scenario["loss_matrix"], decisions, s, d) for s in states)
               for d in decisions)


def post_action_minimax_regret(scenario, action):
    decisions = sorted(scenario["loss_matrix"])
    groups = {}
    for state in scenario["states"]:
        groups.setdefault(action["observation_by_state"][state], []).append(state)
    group_minima = []
    for group_states in groups.values():
        group_minima.append(min(
            max(regret_for(scenario["loss_matrix"], decisions, state, decision)
                for state in group_states)
            for decision in decisions
        ))
    return max(group_minima)


def expected_loss(scenario, prior):
    decisions = sorted(scenario["loss_matrix"])
    return min(sum(prior[state] * scenario["loss_matrix"][decision][state]
                   for state in scenario["states"]) for decision in decisions)


def expected_post_loss(scenario, prior, likelihoods):
    marginals = {}
    for state in scenario["states"]:
        row = likelihoods[state]
        if abs(sum(row.values()) - 1.0) > TOL:
            raise ValueError("observation likelihoods do not sum to one")
        for outcome, likelihood in row.items():
            marginals[outcome] = marginals.get(outcome, 0.0) + prior[state] * likelihood
    total = 0.0
    for outcome, marginal in marginals.items():
        if marginal <= 0:
            continue
        posterior = {
            state: prior[state] * likelihoods[state].get(outcome, 0.0) / marginal
            for state in scenario["states"]
        }
        total += marginal * expected_loss(scenario, posterior)
    return total


def resource_fits(action, budget):
    cost = action["resources"]
    return all(cost[key] <= budget[key] + TOL for key in (
        "engineering_hours", "compute_core_hours", "facility_hours",
        "specimen_count", "review_hours", "external_cost_units"))


def evaluate_action(scenario, action, baseline):
    state = action["admissibility"]["state"]
    budget_ok = resource_fits(action, scenario["budget"])
    claim_ok = action["target_claim_id"] == scenario["target_claim_id"]
    dependencies = set(action["dependencies"])
    common_cause = False
    if scenario["requires_independent_evidence"]:
        for group in scenario["current_evidence_dependency_groups"].values():
            if dependencies.intersection(group):
                common_cause = True
    covered = set(action.get("obligations_covered", []))
    obligations_ok = set(scenario["required_obligations"]).issubset(covered)
    terminal = action.get("observed_execution_state") in {
        "FAILED", "ABORTED", "INVALID", "CENSORED", "OUT_OF_DOMAIN"
    }
    calibrated = (scenario["decision_regime"] != "calibrated_probability" or
                  (action.get("probability_model_ref") and
                   action.get("calibration_status") == "calibrated_for_fixture_only"))
    eligible = (state == "ADMISSIBLE" and budget_ok and claim_ok and
                not common_cause and obligations_ok and not terminal and calibrated)
    value = None
    post = None
    if eligible and scenario["decision_regime"] == "distribution_free":
        post = post_action_minimax_regret(scenario, action)
        value = max(0.0, baseline - post)
    elif eligible and scenario["decision_regime"] == "calibrated_probability":
        likelihoods = action.get("observation_likelihoods")
        if likelihoods is not None:
            post = expected_post_loss(scenario, scenario["prior"], likelihoods)
            value = max(0.0, baseline - post)
    return {
        "eligible": eligible,
        "budget_fits": budget_ok,
        "target_matches": claim_ok,
        "common_cause": common_cause,
        "obligations_covered": obligations_ok,
        "terminal": terminal,
        "value": value,
        "post": post,
    }


def audit_scenario(scenario):
    if scenario["evidence_class"] != "synthetic" or scenario["claim_ceiling"] != CLAIM:
        raise ValueError("unsupported evidence/claim ceiling")
    states = scenario["states"]
    if len(states) < 2 or len(set(states)) != len(states):
        raise ValueError("states must be unique and non-empty")
    for decision, row in scenario["loss_matrix"].items():
        if set(row) != set(states) or any(not math.isfinite(v) or v < 0 for v in row.values()):
            raise ValueError("loss matrix has incomplete or invalid row")
    regime = scenario["decision_regime"]
    if regime == "distribution_free":
        if scenario["prior"] is not None:
            raise ValueError("set-valued case must not contain a prior")
        baseline = minimax_regret(scenario)
        expected_baseline = scenario["expected"]["baseline_minimax_regret"]
    elif regime == "calibrated_probability":
        prior = scenario["prior"]
        if set(prior) != set(states) or any(not math.isfinite(v) or v <= 0 for v in prior.values()):
            raise ValueError("invalid prior")
        if abs(sum(prior.values()) - 1.0) > TOL:
            raise ValueError("prior probabilities do not sum to one")
        baseline = expected_loss(scenario, prior)
        expected_baseline = scenario["expected"]["baseline_expected_loss"]
    else:
        raise ValueError("unknown decision regime")
    if abs(baseline - expected_baseline) > TOL:
        raise ValueError("baseline score differs from known-answer oracle")

    by_id = {}
    actual_totals = {
        "engineering_hours": 0.0,
        "compute_core_hours": 0.0,
        "facility_hours": 0.0,
        "specimen_count": 0,
        "review_hours": 0.0,
        "external_cost_units": 0.0,
    }
    actual_over_plan_count = 0
    actual_record_count = 0
    for action in scenario["actions"]:
        if action["action_id"] in by_id:
            raise ValueError("duplicate action id")
        by_id[action["action_id"]] = action
        observed_state = action.get("observed_execution_state")
        actual = action.get("actual_resources")
        if observed_state is not None and actual is None:
            raise ValueError("execution state has no actual resource record")
        if actual is not None and observed_state is None:
            raise ValueError("actual resources have no execution state")
        if actual is not None:
            numeric = ("engineering_hours", "compute_core_hours", "facility_hours",
                       "review_hours", "external_cost_units")
            if any(not math.isfinite(actual[key]) or actual[key] < 0 for key in numeric):
                raise ValueError("actual resource record is invalid")
            if not isinstance(actual["specimen_count"], int) or actual["specimen_count"] < 0:
                raise ValueError("actual specimen count is invalid")
            actual_record_count += 1
            if not resource_fits(action, actual):
                actual_over_plan_count += 1
            for key in actual_totals:
                actual_totals[key] += actual[key]
    actual_over_budget = not all(
        actual_totals[key] <= scenario["budget"][key] + TOL for key in actual_totals
    )
    audits = {action_id: evaluate_action(scenario, action, baseline)
              for action_id, action in by_id.items()}

    for action_id in scenario["expected"]["must_not_select"]:
        if action_id not in audits:
            raise ValueError("must_not_select references unknown action")
        result = audits[action_id]
        if result["eligible"] and result["value"] is not None and result["value"] > TOL:
            raise ValueError("must_not_select contains positive-value eligible action")

    best_id = scenario["expected"]["best_action_id"]
    sentinel = best_id == "ABSTAIN" or best_id.startswith("REPLAN-")
    if sentinel:
        if not scenario["expected"]["abstain"]:
            raise ValueError("abstention sentinel must set abstain")
        if any(r["eligible"] and r["value"] is not None and r["value"] > TOL
               for r in audits.values()):
            raise ValueError("abstention hides positive-value eligible action")
        expected_post = (scenario["expected"].get("post_action_worst_case_regret")
                         if regime == "distribution_free"
                         else scenario["expected"].get("post_action_expected_loss"))
        if expected_post is not None and abs(expected_post - baseline) > TOL:
            raise ValueError("abstention post-score differs from baseline")
        chosen = None
    else:
        if scenario["expected"]["abstain"] or best_id not in audits:
            raise ValueError("selected action missing or abstain flag inconsistent")
        selected = audits[best_id]
        if not selected["eligible"] or selected["value"] is None or selected["value"] <= TOL:
            raise ValueError("expected action is not eligible or has no positive decision value")
        expected_post = (scenario["expected"].get("post_action_worst_case_regret")
                         if regime == "distribution_free"
                         else scenario["expected"].get("post_action_expected_loss"))
        if expected_post is not None and abs(expected_post - selected["post"]) > TOL:
            raise ValueError("post-action score differs from oracle")
        if regime == "calibrated_probability":
            if abs(selected["value"] - scenario["expected"]["evsi"]) > TOL:
                raise ValueError("EVSI differs from known-answer oracle")
        chosen = selected["value"]

    return {
        "scenario_id": scenario["scenario_id"],
        "baseline_regret_or_loss": baseline,
        "expected_action_id": best_id,
        "expected_action_value": chosen,
        "eligible_action_count": sum(1 for result in audits.values() if result["eligible"]),
        "must_not_select_count": len(scenario["expected"]["must_not_select"]),
        "actual_resources_consumed": actual_totals,
        "actual_resources_over_scenario_budget": actual_over_budget,
        "actual_over_plan_action_count": actual_over_plan_count,
        "actions_with_actual_resource_records": actual_record_count,
        "oracle_matched": True,
    }


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: reference_oracle.py <scenarios.json> <40-hex-source-commit>")
    raw = open(sys.argv[1], "rb").read()
    source_commit = sys.argv[2]
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise SystemExit("invalid source commit")
    corpus = json.loads(raw)
    if corpus.get("schema") != SCHEMA or corpus.get("benchmark_id") != "ORBITAL-JOINT-EVIDENCE-VOI-001":
        raise SystemExit("unsupported corpus schema or benchmark id")
    if corpus.get("evidence_class") != "synthetic" or corpus.get("claim_ceiling") != CLAIM:
        raise SystemExit("unsupported evidence class or claim ceiling")
    ids = [scenario["scenario_id"] for scenario in corpus["scenarios"]]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate scenario ID")
    results = sorted((audit_scenario(scenario) for scenario in corpus["scenarios"]),
                     key=lambda item: item["scenario_id"])
    matches = sum(1 for result in results if result["oracle_matched"])
    report = {
        "schema": "luminous.orbital-joint-evidence-voi.reference-oracle.v0.1",
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "source_commit": source_commit,
        "scenario_count": len(results),
        "oracle_matches": matches,
        "oracle_mismatches": len(results) - matches,
        "scenarios": results,
    }
    print(json.dumps(report, sort_keys=True, indent=2))
    if len(results) != 13 or matches != 13:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
