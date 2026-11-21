#!/usr/bin/env python3
"""Independent stdlib-only oracle for finite-horizon evidence campaign planning."""
import hashlib
import json
import math
import sys

SCHEMA = "luminous.orbital-joint-evidence-voi.sequential.v0.1"
CLAIM = "finite_horizon_sequential_campaign_semantics_only"
TOL = 1.0e-9
RESOURCE_KEYS = (
    "engineering_hours", "compute_core_hours", "facility_hours",
    "specimen_count", "review_hours", "external_cost_units",
)


def zero_resources():
    return {key: 0.0 for key in RESOURCE_KEYS}


def expected_resources(resources):
    return {key: float(value) for key, value in resources.items()}


def resource_add(left, right):
    return {key: left[key] + right[key] for key in RESOURCE_KEYS}


def resource_scale(value, factor):
    return {key: value[key] * factor for key in RESOURCE_KEYS}


def resource_max(left, right):
    return {key: max(left[key], right[key]) for key in RESOURCE_KEYS}


def resource_fits(cost, budget):
    return all(cost[key] <= budget[key] + TOL for key in RESOURCE_KEYS)


def resource_dominates(left, right):
    return (
        all(left[key] <= right[key] + TOL for key in RESOURCE_KEYS)
        and any(left[key] < right[key] - TOL for key in RESOURCE_KEYS)
    )


def terminal_regret(scenario, belief, covered_obligations):
    decisions = scenario["decisions"]
    best = None
    for decision in decisions:
        required = set(scenario["decision_obligations"][decision])
        if not required.issubset(set(covered_obligations)):
            continue
        worst = max(
            scenario["loss_matrix"][decision][state]
            - min(scenario["loss_matrix"][candidate][state] for candidate in decisions)
            for state in belief
        )
        candidate = (worst, decision)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("no terminal disposition supported by path-complete evidence")
    return best[0], {
        "node_kind": "Terminal",
        "belief_states": list(belief),
        "decision": best[1],
        "worst_case_regret": best[0],
        "covered_obligations": sorted(set(covered_obligations)),
    }


def partitions(belief, action):
    groups = {}
    for state in belief:
        groups.setdefault(action["observation_by_state"][state], []).append(state)
    return dict(sorted(groups.items()))


def branch_probability(scenario, belief, branch):
    for action in scenario["actions"]:
        possible_outcomes = set(action["observation_by_state"].values())
        mapping = action.get("obligations_by_observation")
        if not isinstance(mapping, dict) or set(mapping) != possible_outcomes:
            raise ValueError("action outcome-to-obligation map is incomplete")
        for covered in mapping.values():
            if (not isinstance(covered, list) or len(covered) != len(set(covered))
                    or not set(covered).issubset(declared_obligations)):
                raise ValueError("action contains unknown or duplicate evidence obligations")
    weights = scenario["resource_frequency_model"]["state_weights"]
    total = sum(weights[state] for state in belief)
    part = sum(weights[state] for state in branch)
    if total <= 0 or not math.isfinite(total) or not math.isfinite(part):
        raise ValueError("invalid resource-frequency mass")
    return part / total


def expected_from_action(action):
    return expected_resources(action["resource_plan"])


def choose_plan(candidates):
    best_risk = min(plan["risk"] for plan in candidates)
    top = [plan for plan in candidates if abs(plan["risk"] - best_risk) <= TOL]
    frontier = [
        candidate for candidate in top
        if candidate["expected"] is not None and not any(
            other is not candidate and other["expected"] is not None
            and resource_dominates(other["expected"], candidate["expected"])
            for other in top
        )
    ]
    if len(frontier) == 1:
        return frontier[0]
    first_actions = sorted({
        plan["node"]["action_id"] for plan in frontier
        if plan["node"]["node_kind"] == "Test"
    })
    max_path = zero_resources()
    for plan in frontier:
        max_path = resource_max(max_path, plan["worst_path"])
    return {
        "risk": best_risk,
        "expected": None,
        "worst_path": max_path,
        "node": {
            "node_kind": "HumanReview",
            "candidate_first_actions": first_actions,
            "worst_case_regret": best_risk,
        },
    }


def solve(scenario, belief, remaining_budget, depth, used, covered_obligations):
    stop_risk, stop_node = terminal_regret(scenario, belief, covered_obligations)
    candidates = [{
        "risk": stop_risk,
        "expected": zero_resources(),
        "worst_path": zero_resources(),
        "node": stop_node,
    }]
    if depth <= 0:
        return candidates[0]

    context = scenario["decision_context"]
    for action in sorted(scenario["actions"], key=lambda item: item["action_id"]):
        if action["action_id"] in used:
            continue
        if action["admissibility"] != "ADMISSIBLE":
            continue
        if action["decision_context"] != context:
            continue
        if not resource_fits(action["resource_plan"], remaining_budget):
            continue
        branches = partitions(belief, action)
        if len(branches) < 2:
            continue

        next_used = set(used)
        next_used.add(action["action_id"])
        child_branches = {}
        worst_risk = 0.0
        expected_child = zero_resources()
        worst_child = zero_resources()
        for observation, child_belief in branches.items():
            child_budget = {
                key: remaining_budget[key] - action["resource_plan"][key]
                for key in RESOURCE_KEYS
            }
            next_covered = set(covered_obligations) | set(
                action["obligations_by_observation"][observation]
            )
            child = solve(scenario, child_belief, child_budget, depth - 1, next_used, next_covered)
            worst_risk = max(worst_risk, child["risk"])
            probability = branch_probability(scenario, belief, child_belief)
            if child["expected"] is None:
                expected_child = None
            elif expected_child is not None:
                expected_child = resource_add(
                    expected_child, resource_scale(child["expected"], probability)
                )
            worst_child = resource_max(worst_child, child["worst_path"])
            child_branches[observation] = child["node"]

        action_expected = expected_from_action(action)
        expected_total = (
            resource_add(action_expected, expected_child)
            if expected_child is not None else None
        )
        worst_total = resource_add(action["resource_plan"], worst_child)
        if not resource_fits(worst_total, scenario["budget"]):
            continue
        candidates.append({
            "risk": worst_risk,
            "expected": expected_total,
            "worst_path": worst_total,
            "node": {
                "node_kind": "Test",
                "action_id": action["action_id"],
                "branches": child_branches,
            },
        })
    return choose_plan(candidates)


def evaluate_fixed(scenario):
    by_id = {action["action_id"]: action for action in scenario["actions"]}
    total = zero_resources()
    for action_id in scenario["fixed_campaign_sequence"]:
        action = by_id[action_id]
        if (action["admissibility"] != "ADMISSIBLE"
                or action["decision_context"] != scenario["decision_context"]):
            raise ValueError("fixed action not admissible in exact decision context")
        total = resource_add(total, action["resource_plan"])
    if not resource_fits(total, scenario["budget"]):
        raise ValueError("fixed campaign exceeds budget")
    groups = {}
    for state in scenario["states"]:
        signature = tuple(
            by_id[action_id]["observation_by_state"][state]
            for action_id in scenario["fixed_campaign_sequence"]
        )
        groups.setdefault(signature, []).append(state)
    grouped_coverage = {}
    for state in scenario["states"]:
        signature = tuple(
            by_id[action_id]["observation_by_state"][state]
            for action_id in scenario["fixed_campaign_sequence"]
        )
        covered = set()
        for action_id, observation in zip(scenario["fixed_campaign_sequence"], signature):
            covered.update(by_id[action_id]["obligations_by_observation"][observation])
        grouped_coverage[signature] = covered
    risk = max(
        terminal_regret(scenario, belief, grouped_coverage[signature])[0]
        for signature, belief in groups.items()
    )
    return risk, total


def evaluate_myopic(scenario, baseline):
    scored = []
    for action in scenario["actions"]:
        if (action["admissibility"] != "ADMISSIBLE"
                or action["decision_context"] != scenario["decision_context"]
                or not resource_fits(action["resource_plan"], scenario["budget"])):
            continue
        groups = partitions(scenario["states"], action)
        if len(groups) < 2:
            continue
        risk = max(
            terminal_regret(
                scenario,
                belief,
                set(action["obligations_by_observation"][observation])
            )[0]
            for observation, belief in groups.items()
        )
        scored.append((risk, action["action_id"]))
    scored.sort()
    if scored and scored[0][0] + TOL < baseline:
        return scored[0][1], scored[0][0], False
    return None, baseline, True


def audit_scenario(scenario):
    if (scenario["evidence_class"] != "synthetic"
            or scenario["claim_ceiling"] != CLAIM):
        raise ValueError("unsupported evidence or claim ceiling")
    states = scenario["states"]
    decisions = scenario["decisions"]
    if (set(scenario["decision_obligations"]) != set(decisions)
            or any(not isinstance(items, list) or len(items) != len(set(items))
                   or any(not isinstance(item, str) or not item.strip() for item in items)
                   for items in scenario["decision_obligations"].values())
            or not scenario["decision_obligations"]["ACCEPT"]
            or not scenario["decision_obligations"]["REJECT"]):
        raise ValueError("decision acceptance obligations are incomplete or duplicated")
    declared_obligations = set().union(*(set(items) for items in scenario["decision_obligations"].values()))
    if len(states) < 2 or len(set(states)) != len(states):
        raise ValueError("invalid hidden-state set")
    context = scenario["decision_context"]
    if any(not isinstance(context.get(key), str) or not context[key].strip()
           for key in ("profile_ref", "article_subject_ref", "configuration_id", "icd_revision")):
        raise ValueError("incomplete exact decision context")
    weights = scenario["resource_frequency_model"]
    if (weights["calibration_status"] != "calibrated_for_fixture_only"
            or weights["used_for"] != "EXPECTED_RESOURCE_USE_ONLY"
            or set(weights["state_weights"]) != set(states)
            or any(not math.isfinite(value) or value <= 0
                   for value in weights["state_weights"].values())
            or abs(sum(weights["state_weights"].values()) - 1.0) > TOL):
        raise ValueError("resource-frequency model is invalid or used outside cost accounting")

    baseline, _ = terminal_regret(scenario, states, set())
    myopic_id, myopic_risk, myopic_stops = evaluate_myopic(scenario, baseline)
    fixed_risk, fixed_resources = evaluate_fixed(scenario)
    adaptive = solve(
        scenario, states, scenario["budget"], scenario["max_actions"], set(), set()
    )
    first_action = (
        adaptive["node"]["action_id"]
        if adaptive["node"]["node_kind"] == "Test" else None
    )
    expected = scenario["expected"]
    if abs(baseline - expected["baseline_worst_case_regret"]) > TOL:
        raise ValueError("baseline minimax-regret known answer mismatch")
    if abs(myopic_risk - expected["myopic_worst_case_regret"]) > TOL:
        raise ValueError("myopic known answer mismatch")
    if myopic_stops != expected["myopic_stops_without_action"]:
        raise ValueError("myopic stop disposition mismatch")
    if abs(fixed_risk - expected["fixed_campaign_worst_case_regret"]) > TOL:
        raise ValueError("fixed-campaign risk mismatch")
    if abs(adaptive["risk"] - expected["adaptive_worst_case_regret"]) > TOL:
        raise ValueError("adaptive risk mismatch")
    if first_action != expected["adaptive_first_action"]:
        raise ValueError("adaptive first action mismatch")
    if adaptive["expected"] is None:
        raise ValueError("known-answer adaptive policy unexpectedly requires human review")
    for key in RESOURCE_KEYS:
        if abs(adaptive["expected"][key] - expected["adaptive_expected_resources"][key]) > TOL:
            raise ValueError("adaptive expected resource mismatch: " + key)
        if abs(fixed_resources[key] - expected["fixed_expected_resources"][key]) > TOL:
            raise ValueError("fixed expected resource mismatch: " + key)
    if expected["require_adaptive_resource_pareto_dominates_fixed"]:
        if not resource_dominates(adaptive["expected"], fixed_resources):
            raise ValueError("adaptive resource vector does not dominate fixed campaign")
    if not resource_fits(adaptive["worst_path"], scenario["budget"]):
        raise ValueError("adaptive worst path exceeds budget")
    return {
        "scenario_id": scenario["scenario_id"],
        "baseline_worst_case_regret": baseline,
        "myopic_action_id": myopic_id,
        "myopic_worst_case_regret": myopic_risk,
        "fixed_campaign_worst_case_regret": fixed_risk,
        "adaptive_first_action": first_action,
        "adaptive_worst_case_regret": adaptive["risk"],
        "adaptive_expected_resources": adaptive["expected"],
        "fixed_expected_resources": fixed_resources,
        "expected_resource_savings": {
            key: fixed_resources[key] - adaptive["expected"][key]
            for key in RESOURCE_KEYS
        },
        "adaptive_policy": adaptive["node"],
        "oracle_matched": True,
    }


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: sequential_reference_oracle.py <campaign.json> <40-hex-source-commit>")
    raw = open(sys.argv[1], "rb").read()
    source_commit = sys.argv[2]
    if len(source_commit) != 40 or any(char not in "0123456789abcdef" for char in source_commit):
        raise SystemExit("invalid source commit")
    corpus = json.loads(raw)
    if (corpus.get("schema") != SCHEMA
            or corpus.get("benchmark_id") != "ORBITAL-JOINT-SEQUENTIAL-PLANNING-001"
            or corpus.get("evidence_class") != "synthetic"
            or corpus.get("claim_ceiling") != CLAIM):
        raise SystemExit("unsupported sequential corpus")
    results = sorted((audit_scenario(item) for item in corpus["scenarios"]),
                     key=lambda item: item["scenario_id"])
    report = {
        "schema": "luminous.orbital-joint-evidence-voi.sequential-reference.v0.1",
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "source_commit": source_commit,
        "scenario_count": len(results),
        "oracle_matches": sum(1 for item in results if item["oracle_matched"]),
        "oracle_mismatches": sum(1 for item in results if not item["oracle_matched"]),
        "scenarios": results,
    }
    print(json.dumps(report, sort_keys=True, indent=2))
    if len(results) != 2 or report["oracle_matches"] != 2:
        raise SystemExit(1)
    by_id = {item["scenario_id"]: item for item in results}
    if by_id["SEQ-001"]["adaptive_first_action"] != "A-COMBINED-LOAD-SCREEN":
        raise SystemExit("SEQ-001 first action mismatch")
    if by_id["SEQ-002"]["adaptive_first_action"] != "B-THERMAL-INSPECTION":
        raise SystemExit("SEQ-002 first action mismatch")


if __name__ == "__main__":
    main()
