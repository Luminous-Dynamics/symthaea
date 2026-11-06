#!/usr/bin/env python3
"""Independent exact finite-family mechanism-identifiability oracle.

Uses direct categorical product distributions, not the Rust audit's planned
data structures. This is a bounded synthetic/software oracle, not a biological
mechanism or causal-discovery engine.
"""
from __future__ import annotations

import copy
import itertools
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def valid_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def total_variation(left: list[float], right: list[float]) -> float:
    require(len(left) == len(right), "TV distributions must have the same outcome dimension")
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right))


def product_testing_error(left: list[float], right: list[float], n: int) -> float:
    """Exact equal-prior Bayes error for n IID categorical replicates."""
    require(len(left) == len(right), "product distributions must share an outcome vocabulary")
    require(1 <= n <= 8, "replicate count exceeds exact V1 oracle budget")
    overlap = 0.0
    for sequence in itertools.product(range(len(left)), repeat=n):
        p_left = math.prod(left[index] for index in sequence)
        p_right = math.prod(right[index] for index in sequence)
        overlap += min(p_left, p_right)
    return 0.5 * overlap


def pair_key(action_id: str, mechanism_a: str, mechanism_b: str) -> str:
    left, right = sorted((mechanism_a, mechanism_b))
    return f"{action_id}|{left}|{right}"


def verify(data: dict[str, Any]) -> dict[str, Any]:
    require(data["schema_version"] == 1, "unsupported schema version")
    require(data["claim_scope"] == "synthetic_known_truth", "V1 executable oracle accepts only synthetic known-truth fixtures")
    require(data["query"]["query_id"].strip(), "query ID must be non-empty")
    require(valid_sha(data["query"]["query_spec_sha256"]), "query specification digest malformed")
    require(valid_sha(data["query"]["decision_rule_sha256"]), "query decision-rule digest malformed")
    require(data["replicate_design"]["independence_scope"] == "SYNTHETIC_IID_KNOWN_TRUTH", "synthetic IID contract required")
    require(data["replicate_design"]["independent_unit"] == "synthetic_replicate", "synthetic replicate is the declared independent unit")
    require(valid_sha(data["replicate_design"]["split_manifest_sha256"]), "split manifest digest malformed")

    mechanisms = sorted(data["mechanisms"], key=lambda row: row["mechanism_id"])
    nuisance = sorted(data["nuisance_profiles"], key=lambda row: row["nuisance_profile_id"])
    interventions = sorted(data["interventions"], key=lambda row: row["intervention_id"])
    require(len(mechanisms) >= 2, "at least two mechanism hypotheses required")
    require(len({row["mechanism_id"] for row in mechanisms}) == len(mechanisms), "duplicate mechanism IDs")
    require(len({row["nuisance_profile_id"] for row in nuisance}) == len(nuisance), "duplicate nuisance profile IDs")
    require(len({row["intervention_id"] for row in interventions}) == len(interventions), "duplicate intervention IDs")
    require(abs(sum(row["prior_probability"] for row in mechanisms) - 1.0) <= 1e-9, "mechanism priors must sum to one")
    require(0.0 < data["alpha"] < 0.5, "alpha must be between zero and one half")

    nuisance_ids = [row["nuisance_profile_id"] for row in nuisance]
    intervention_by_id = {row["intervention_id"]: row for row in interventions}
    intervention_ids = [row["intervention_id"] for row in interventions]
    mechanism_ids = [row["mechanism_id"] for row in mechanisms]
    require(nuisance_ids, "at least one nuisance profile required")

    allowed_classes = {
        "DIRECT_SYNAPTIC", "POLYSYNAPTIC", "EXTRASYNAPTIC_LATENT", "SHARED_INPUT",
        "CELL_INTRINSIC_STATE", "OBSERVATION_ARTIFACT", "MIXTURE_UNRESOLVED", "OTHER",
    }
    for row in mechanisms:
        require(valid_sha(row["source_evidence_sha256"]), "mechanism source digest malformed")
        require(valid_sha(row["model_sha256"]), "mechanism model digest malformed")
        require(row["mechanism_class"] in allowed_classes, "unsupported mechanism class")
    for row in nuisance:
        require(valid_sha(row["profile_sha256"]), "nuisance profile digest malformed")
    for action in interventions:
        require(action["cost_units"] > 0 and math.isfinite(action["cost_units"]), "intervention cost must be finite and positive")
        require(action["feasibility_status"] in {"FEASIBLE", "DISALLOWED", "INACCESSIBLE", "NOT_ASSESSED"}, "unknown feasibility status")
        require(valid_sha(action["feasibility_receipt_sha256"]), "feasibility receipt malformed")
        require(valid_sha(action["safety_profile_sha256"]), "safety profile digest malformed")
        require(len(action["outcome_ids"]) >= 2 and len(action["outcome_ids"]) == len(set(action["outcome_ids"])), "outcome vocabulary must be unique and non-trivial")

    expected_keys = {
        (action_id, mechanism_id, nuisance_id)
        for action_id in intervention_ids
        for mechanism_id in mechanism_ids
        for nuisance_id in nuisance_ids
    }
    predictions: dict[tuple[str, str, str], list[float]] = {}
    for row in data["predictions"]:
        key = (row["intervention_id"], row["mechanism_id"], row["nuisance_profile_id"])
        require(key in expected_keys, f"prediction has a dangling reference: {key}")
        require(key not in predictions, f"duplicate prediction cell: {key}")
        require(valid_sha(row["observation_model_sha256"]), "observation model digest malformed")
        action = intervention_by_id[key[0]]
        outcomes = row["outcomes"]
        ids = [item["outcome_id"] for item in outcomes]
        require(len(ids) == len(set(ids)), f"duplicate outcome ID in {key}")
        require(set(ids) == set(action["outcome_ids"]), f"outcome vocabulary mismatch in {key}")
        probability_map = {item["outcome_id"]: item["probability"] for item in outcomes}
        vector = [probability_map[outcome_id] for outcome_id in action["outcome_ids"]]
        require(all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in vector), f"invalid probability in {key}")
        require(abs(sum(vector) - 1.0) <= 1e-9, f"outcome probabilities do not sum to one in {key}")
        predictions[key] = vector
    require(set(predictions) == expected_keys, "prediction matrix is not a complete action × mechanism × nuisance Cartesian product")

    n = data["replicate_design"]["replicate_count"]
    require(1 <= n <= 8, "replicate count outside exact V1 limits")
    alpha = data["alpha"]
    action_reports: dict[str, Any] = {}
    pair_risk_map: dict[str, float] = {}
    for action_id in intervention_ids:
        pair_reports = []
        for index, mechanism_a in enumerate(mechanisms):
            for mechanism_b in mechanisms[index + 1:]:
                tv_by_profile = {}
                error_by_profile = {}
                for nuisance_id in nuisance_ids:
                    left = predictions[(action_id, mechanism_a["mechanism_id"], nuisance_id)]
                    right = predictions[(action_id, mechanism_b["mechanism_id"], nuisance_id)]
                    tv_by_profile[nuisance_id] = total_variation(left, right)
                    error_by_profile[nuisance_id] = product_testing_error(left, right, n)
                robust_tv = min(tv_by_profile.values())
                worst_error = max(error_by_profile.values())
                key = pair_key(action_id, mechanism_a["mechanism_id"], mechanism_b["mechanism_id"])
                pair_risk_map[key] = worst_error
                pair_reports.append({
                    "mechanism_a": mechanism_a["mechanism_id"],
                    "mechanism_b": mechanism_b["mechanism_id"],
                    "different_query_decision": mechanism_a["query_decision"] != mechanism_b["query_decision"],
                    "different_mechanism_class": mechanism_a["mechanism_class"] != mechanism_b["mechanism_class"],
                    "tv_by_nuisance_profile": tv_by_profile,
                    "robust_minimum_tv": robust_tv,
                    "testing_error_by_nuisance_profile": error_by_profile,
                    "worst_case_testing_error": worst_error,
                    "pair_identified_at_budget": worst_error <= alpha,
                })
        query_pairs = [row for row in pair_reports if row["different_query_decision"]]
        mechanism_pairs = [row for row in pair_reports if row["different_mechanism_class"]]
        query_identified = bool(query_pairs) and all(row["pair_identified_at_budget"] for row in query_pairs)
        mechanism_identified = bool(mechanism_pairs) and all(row["pair_identified_at_budget"] for row in mechanism_pairs)
        action_reports[action_id] = {
            "feasibility_status": intervention_by_id[action_id]["feasibility_status"],
            "cost_units": intervention_by_id[action_id]["cost_units"],
            "query_identified": query_identified,
            "mechanism_identified": mechanism_identified,
            "status": (
                "QUERY_AND_MECHANISM_IDENTIFIED" if query_identified and mechanism_identified
                else "QUERY_IDENTIFIED_MECHANISM_UNIDENTIFIED" if query_identified
                else "QUERY_UNIDENTIFIED"
            ),
            "pairwise_results": pair_reports,
        }

    feasible_joint = [
        action_id for action_id, report in action_reports.items()
        if report["feasibility_status"] == "FEASIBLE"
        and report["query_identified"] and report["mechanism_identified"]
    ]
    selected = sorted(feasible_joint, key=lambda action_id: (action_reports[action_id]["cost_units"], action_id))[0] if feasible_joint else None
    expected = data["known_answer"]
    require(selected == expected["expected_selected_intervention_id"], "minimum-cost feasible joint-identification action mismatch")
    for action_id, wanted in expected["expected_action_statuses"].items():
        require(action_id in action_reports, f"known answer references missing action {action_id}")
        actual = action_reports[action_id]
        require(actual["query_identified"] == wanted["query_identified"], f"query-identifiability mismatch for {action_id}")
        require(actual["mechanism_identified"] == wanted["mechanism_identified"], f"mechanism-identifiability mismatch for {action_id}")
    for key, wanted in expected["expected_pairwise_risk"].items():
        require(key in pair_risk_map, f"known answer references missing pair {key}")
        require(abs(pair_risk_map[key] - wanted) <= 1e-8, f"finite-sample pairwise risk mismatch for {key}")

    require(intervention_by_id[selected]["feasibility_status"] == "FEASIBLE", "selected action is not feasible")
    require(selected != "infeasible-perfect", "disallowed perfect discriminator was selected")
    return {
        "schema_version": 1,
        "tool": "connectome-mechanism-identifiability-oracle",
        "run_id": data["run_id"],
        "claim_scope": data["claim_scope"],
        "replicate_count": n,
        "alpha": alpha,
        "mechanism_count": len(mechanisms),
        "nuisance_profile_count": len(nuisance_ids),
        "intervention_count": len(interventions),
        "prediction_cell_count": len(predictions),
        "selected_intervention_id": selected,
        "action_reports": action_reports,
        "pairwise_worst_case_testing_error": pair_risk_map,
        "query_vs_mechanism_semantics": "SEPARATE_GATES",
        "nuisance_scope": "DECLARED_COMMON_PROFILES_ONLY",
        "biological_mechanism": "NOT_ESTABLISHED_BY_SYNTHETIC_ORACLE",
        "status": "SYNTHETIC_MECHANISM_IDENTIFIABILITY_CONTRACT_VALIDATED",
    }


def self_test(data: dict[str, Any]) -> None:
    baseline = verify(copy.deepcopy(data))
    require(baseline["selected_intervention_id"] == "orthogonal-joint", "starter fixture known answer failed")

    missing = copy.deepcopy(data)
    missing["predictions"].pop()
    try:
        verify(missing)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing Cartesian prediction cell was not rejected")

    infeasible = copy.deepcopy(data)
    next(row for row in infeasible["interventions"] if row["intervention_id"] == "infeasible-perfect")["feasibility_status"] = "FEASIBLE"
    try:
        verify(infeasible)
    except AssertionError:
        pass
    else:
        raise AssertionError("known-answer mutation unexpectedly passed after feasibility change")

    underpowered = copy.deepcopy(data)
    underpowered["replicate_design"]["replicate_count"] = 1
    try:
        verify(underpowered)
    except AssertionError:
        pass
    else:
        raise AssertionError("underpowered mutation unexpectedly retained the n=7 known answer")
    print("mechanism-identifiability negative controls: PASS")


def main() -> int:
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[2] != "--self-test"):
        print("usage: verify_mechanism_identifiability_oracle.py FIXTURE.json [--self-test]", file=sys.stderr)
        return 2
    try:
        data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        report = verify(data)
        if len(sys.argv) == 3:
            self_test(data)
        print(json.dumps(report, indent=2, sort_keys=True))
        print("mechanism-identifiability independent oracle: PASS")
        return 0
    except Exception as exc:
        print(f"mechanism-identifiability oracle: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
