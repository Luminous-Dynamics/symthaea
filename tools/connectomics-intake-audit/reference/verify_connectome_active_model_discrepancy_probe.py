#!/usr/bin/env python3
"""Independent exact finite oracle for active model-discrepancy probe design."""
from __future__ import annotations
import json
import math
import sys
from pathlib import Path
from typing import Any


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def valid_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def unique(values: list[str], label: str) -> None:
    require(len(values) == len(set(values)), f"duplicate {label}")


def tv(p: dict[str, float], q: dict[str, float]) -> float:
    return 0.5 * sum(abs(p.get(y, 0.0) - q.get(y, 0.0)) for y in set(p) | set(q))


def baseline_risk(worlds: list[dict[str, Any]], query: dict[str, Any]) -> tuple[float, bool]:
    p_true = sum(w["prior_probability"] for w in worlds if w["query_decision"])
    p_false = 1.0 - p_true
    true_decision_loss = p_false * query["loss_false_true"]
    false_decision_loss = p_true * query["loss_true_false"]
    return min(true_decision_loss, false_decision_loss), true_decision_loss <= false_decision_loss


def model_action_risk(action_id: str, model_id: str, worlds: list[dict[str, Any]],
                      predictions: dict[tuple[str, str, str], dict[str, float]],
                      query: dict[str, Any]) -> float:
    outcomes = set()
    for world in worlds:
        outcomes.update(predictions[(action_id, world["world_id"], model_id)])
    total = 0.0
    for y in outcomes:
        joint_true = sum(w["prior_probability"] * predictions[(action_id, w["world_id"], model_id)].get(y, 0.0)
                         for w in worlds if w["query_decision"])
        joint_false = sum(w["prior_probability"] * predictions[(action_id, w["world_id"], model_id)].get(y, 0.0)
                          for w in worlds if not w["query_decision"])
        total += min(joint_false * query["loss_false_true"], joint_true * query["loss_true_false"])
    return total


def marginal_model_prediction(action_id: str, model_id: str, worlds: list[dict[str, Any]],
                               predictions: dict[tuple[str, str, str], dict[str, float]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for world in worlds:
        for outcome, probability in predictions[(action_id, world["world_id"], model_id)].items():
            result[outcome] = result.get(outcome, 0.0) + world["prior_probability"] * probability
    return result


def model_mi_bits(action_id: str, models: list[dict[str, Any]], worlds: list[dict[str, Any]],
                  predictions: dict[tuple[str, str, str], dict[str, float]]) -> float:
    """Mutual information between the declared model ID and outcome; diagnostic only."""
    model_ids = [m["model_id"] for m in models]
    outcomes = set()
    for model_id in model_ids:
        outcomes.update(marginal_model_prediction(action_id, model_id, worlds, predictions))
    p_y: dict[str, float] = {}
    for model in models:
        pm = model["model_prior_probability"]
        for y, py_given_m in marginal_model_prediction(action_id, model["model_id"], worlds, predictions).items():
            p_y[y] = p_y.get(y, 0.0) + pm * py_given_m
    h_y = -sum(p * math.log2(p) for p in p_y.values() if p > 0)
    h_y_given_m = 0.0
    for model in models:
        dist = marginal_model_prediction(action_id, model["model_id"], worlds, predictions)
        h_y_given_m += model["model_prior_probability"] * (-sum(p * math.log2(p) for p in dist.values() if p > 0))
    return h_y - h_y_given_m


def pareto_frontier(actions: list[dict[str, Any]], q_value: dict[str, float],
                    d_value: dict[str, float]) -> list[str]:
    feasible = [a for a in actions if a["feasibility_status"] == "FEASIBLE"]
    frontier = []
    for candidate in feasible:
        cid = candidate["action_id"]
        dominated = False
        for other in feasible:
            oid = other["action_id"]
            if oid == cid:
                continue
            no_worse = (
                q_value[oid] >= q_value[cid] - 1e-12
                and d_value[oid] >= d_value[cid] - 1e-12
                and other["cost_units"] <= candidate["cost_units"] + 1e-12
            )
            strictly_better = (
                q_value[oid] > q_value[cid] + 1e-12
                or d_value[oid] > d_value[cid] + 1e-12
                or other["cost_units"] < candidate["cost_units"] - 1e-12
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(cid)
    return sorted(frontier)


def verify(data: dict[str, Any]) -> dict[str, Any]:
    require(data["schema_version"] == 1, "unsupported schema")
    require(data["claim_scope"] == "synthetic_known_truth", "oracle only accepts synthetic known-truth scope")
    require(data["dataset_binding"]["evidence_tier"] == "SYNTHETIC", "dataset tier mismatch")
    for f in ("revision_sha256", "source_access_receipt_sha256"):
        require(valid_sha(data["dataset_binding"][f]), f"malformed dataset digest {f}")
    query = data["query"]
    for f in ("query_spec_sha256", "decision_rule_sha256"):
        require(valid_sha(query[f]), f"malformed query digest {f}")
    require(math.isfinite(query["loss_false_true"]) and math.isfinite(query["loss_true_false"])
            and query["loss_false_true"] >= 0 and query["loss_true_false"] >= 0, "invalid query loss")

    worlds, models, actions = data["worlds"], data["models"], data["actions"]
    unique([w["world_id"] for w in worlds], "world IDs")
    unique([m["model_id"] for m in models], "model IDs")
    unique([a["action_id"] for a in actions], "action IDs")
    require(all(w["biological_disposition"] == "ADMISSIBLE" for w in worlds), "non-admissible world")
    for w in worlds:
        require(valid_sha(w["source_evidence_sha256"]), f"bad source digest {w['world_id']}")
        require(math.isfinite(w["prior_probability"]) and 0 <= w["prior_probability"] <= 1, "bad world prior")
    require(abs(sum(w["prior_probability"] for w in worlds) - 1.0) <= 1e-12, "world priors do not sum to one")
    for m in models:
        for f in ("model_sha256", "observation_kernel_sha256", "model_evidence_sha256"):
            require(valid_sha(m[f]), f"bad model digest {m['model_id']}/{f}")
        require(math.isfinite(m["model_prior_probability"]) and 0 <= m["model_prior_probability"] <= 1, "bad model prior")
    require(abs(sum(m["model_prior_probability"] for m in models) - 1.0) <= 1e-12, "model priors do not sum to one")

    challenge = data["challenge_binding"]
    baseline_ids, challenger_ids = challenge["baseline_model_ids"], challenge["challenger_model_ids"]
    require(set(baseline_ids) and set(challenger_ids), "baseline/challenger partition is empty")
    model_by_id = {m["model_id"]: m for m in models}
    require(set(baseline_ids).isdisjoint(challenger_ids), "model role overlap")
    require(set(baseline_ids) | set(challenger_ids) == set(model_by_id), "model role partition does not cover exact model set")
    require(all(model_by_id[x]["role"] == "BASELINE" for x in baseline_ids), "baseline model role mismatch")
    require(all(model_by_id[x]["role"] == "CHALLENGER" for x in challenger_ids), "challenger model role mismatch")
    require(challenge["adjudication_status"] == "ADMITTED_FOR_SENSITIVITY", "challenge not admitted for sensitivity")
    require(valid_sha(challenge["challenge_evidence_sha256"]), "bad challenge evidence digest")
    require(set(challenge["challenger_discovery_unit_ids"]).issubset(discovery), "challenge evidence uses units outside the discovery partition")
    shared_ancestry = set()
    for baseline_id in baseline_ids:
        for challenger_id in challenger_ids:
            shared_ancestry.update(set(model_by_id[baseline_id]["ancestry_group_ids"]) & set(model_by_id[challenger_id]["ancestry_group_ids"]))
    require(shared_ancestry.issubset(set(challenge["ancestry_group_ids"])), "challenge binding omits shared model ancestry")

    split = data["split_manifest"]
    fit, discovery, lockbox = (set(split["model_fit_units"]), set(split["challenger_discovery_units"]),
                               set(split["confirmation_lockbox_units"]))
    require(fit and discovery and lockbox, "empty data split partition")
    require(not (fit & discovery or fit & lockbox or discovery & lockbox), "model-fit/discovery/confirmation leakage")
    require(valid_sha(split["split_sha256"]), "bad split digest")

    expected_keys = {(a["action_id"], w["world_id"], m["model_id"]) for a in actions for w in worlds for m in models}
    pred: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in data["predictions"]:
        key = (row["action_id"], row["world_id"], row["model_id"])
        require(key in expected_keys, f"unknown prediction reference {key}")
        require(key not in pred, f"duplicate prediction record {key}")
        outcomes = {}
        for o in row["outcomes"]:
            require(o["outcome_id"] not in outcomes, "duplicate outcome ID")
            p = o["probability"]
            require(math.isfinite(p) and 0 <= p <= 1, "outcome probability outside [0,1]")
            outcomes[o["outcome_id"]] = p
        require(abs(sum(outcomes.values()) - 1.0) <= 1e-12, f"outcome probabilities do not sum to one {key}")
        pred[key] = outcomes
    require(set(pred) == expected_keys, "prediction Cartesian product incomplete")

    for a in actions:
        require(math.isfinite(a["cost_units"]) and a["cost_units"] > 0, "action cost must be finite and positive")
        require(a["feasibility_status"] in {"FEASIBLE", "DISALLOWED", "INACCESSIBLE", "NOT_ASSESSED"}, "unknown feasibility")
        for f in ("feasibility_receipt_sha256", "safety_profile_sha256"):
            require(valid_sha(a[f]), f"bad action receipt {a['action_id']}/{f}")
    require(any(a["feasibility_status"] == "FEASIBLE" for a in actions), "no feasible actions")

    prior_loss, prior_decision = baseline_risk(worlds, query)
    require(prior_decision == query["baseline_decision"], "declared baseline query decision mismatch")
    query_loss, query_value, discrepancy_tv, discrepancy_value, mi = {}, {}, {}, {}, {}
    for action in actions:
        aid = action["action_id"]
        model_losses = {m["model_id"]: model_action_risk(aid, m["model_id"], worlds, pred, query) for m in models}
        worst_query_loss = max(model_losses.values())
        query_loss[aid] = worst_query_loss
        query_value[aid] = max(0.0, prior_loss - worst_query_loss) / action["cost_units"]
        base_distributions = [marginal_model_prediction(aid, mid, worlds, pred) for mid in baseline_ids]
        challenge_distributions = [marginal_model_prediction(aid, mid, worlds, pred) for mid in challenger_ids]
        pairwise = [tv(p, q) for p in base_distributions for q in challenge_distributions]
        robust_tv = min(pairwise)
        discrepancy_tv[aid] = robust_tv
        discrepancy_value[aid] = robust_tv / action["cost_units"]
        mi[aid] = model_mi_bits(aid, models, worlds, pred)

    feasible = [a for a in actions if a["feasibility_status"] == "FEASIBLE"]
    def best_by(values: dict[str, float]) -> str | None:
        ranked = sorted(feasible, key=lambda a: (-values[a["action_id"]], a["action_id"]))
        return ranked[0]["action_id"] if ranked and values[ranked[0]["action_id"]] > 1e-12 else None
    query_best = best_by(query_value)
    discrepancy_best = best_by(discrepancy_value)
    frontier = pareto_frontier(actions, query_value, discrepancy_value)
    status = "SYNTHETIC_CONTRACT_VALIDATED" if discrepancy_best is not None else "NO_FEASIBLE_DISCREPANCY_PROBE"

    expected = data["oracle_expectations"]
    require(abs(expected["baseline_query_loss"] - prior_loss) <= 1e-12, "baseline query loss mismatch")
    require(expected["query_only_action_id"] == query_best, "query-only action mismatch")
    require(expected["discrepancy_only_action_id"] == discrepancy_best, "discrepancy-only action mismatch")
    require(expected["pareto_frontier_action_ids"] == frontier, "Pareto frontier mismatch")
    for aid in query_value:
        for name, actual in (
            ("query_value_per_cost", query_value[aid]),
            ("discrepancy_tv_per_cost", discrepancy_value[aid]),
            ("discrepancy_tv", discrepancy_tv[aid]),
            ("model_mutual_information_bits_diagnostic", mi[aid]),
        ):
            require(aid in expected[name], f"missing known-answer {name}/{aid}")
            require(abs(expected[name][aid] - actual) <= 1e-12, f"known-answer mismatch {name}/{aid}")
    ancestry = {
        mid: sorted(set.intersection(*(set(model_by_id[x]["ancestry_group_ids"]) for x in baseline_ids))
                    & set(model_by_id[mid]["ancestry_group_ids"]))
        for mid in challenger_ids
    }
    return {
        "status": status,
        "baseline_query_loss": prior_loss,
        "baseline_decision": prior_decision,
        "worst_case_query_loss": query_loss,
        "query_value_per_cost": query_value,
        "robust_discrepancy_tv": discrepancy_tv,
        "discrepancy_value_per_cost": discrepancy_value,
        "model_mutual_information_bits_diagnostic": mi,
        "query_only_action_id": query_best,
        "discrepancy_only_action_id": discrepancy_best,
        "pareto_frontier_action_ids": frontier,
        "model_ancestry_overlap": ancestry,
        "model_adequacy_claim": "NOT_ESTABLISHED_BY_FINITE_CHALLENGE_FAMILY",
        "independence_claim": "NOT_ESTABLISHED_BY_ANCESTRY_LABELS",
    }


def self_test() -> None:
    require(abs(tv({"X": 1.0}, {"X": 0.5, "Y": 0.5}) - 0.5) <= 1e-12, "TV known answer")
    require(abs(tv({"X": 0.5, "Y": 0.5}, {"X": 0.5, "Y": 0.5})) <= 1e-12, "zero TV known answer")
    print("active model-discrepancy probe oracle self-test: PASS")


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--self-test":
        try:
            self_test()
            return 0
        except Exception as exc:
            print(f"probe oracle self-test: FAIL: {exc}", file=sys.stderr)
            return 1
    if len(sys.argv) != 2:
        print("usage: verify_connectome_active_model_discrepancy_probe.py FIXTURE.json | --self-test", file=sys.stderr)
        return 2
    try:
        raw = Path(sys.argv[1]).read_bytes()
        data = json.loads(raw)
        report = verify(data)
        print(json.dumps(report, indent=2, sort_keys=True))
        print("active model-discrepancy probe oracle: PASS")
        return 0
    except Exception as exc:
        print(f"active model-discrepancy probe oracle: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
