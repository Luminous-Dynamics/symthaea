#!/usr/bin/env python3
"""Exact finite oracle for anytime-valid connectome discrepancy monitoring."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = ROOT / "docs/neuroscience/examples/CONNECTOME_ANYTIME_VALID_DISCREPANCY_MONITOR.example.json"
SHA64 = re.compile(r"^[0-9a-f]{64}$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def rat(value: Any, label: str) -> Fraction:
    require(isinstance(value, str), f"{label}: rational must be a string")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"{label}: invalid rational {value!r}") from exc


def rational_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}" if value.denominator != 1 else str(value.numerator)


def canonical_digest(data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_kernels(
    model: dict[str, Any],
    policy: dict[tuple[str, ...], str],
    alphabet: tuple[str, ...],
    challenger: bool,
) -> dict[tuple[tuple[str, ...], str], dict[str, Fraction]]:
    expected_fields = {"model_id", "kernels"} | ({"prior_probability"} if challenger else set())
    require(set(model) == expected_fields, "model has unknown or missing fields")
    model_id = model["model_id"]
    require(isinstance(model_id, str) and model_id, "model has empty ID")
    result: dict[tuple[tuple[str, ...], str], dict[str, Fraction]] = {}
    for row in model["kernels"]:
        require(set(row) == {"history", "action_id", "probabilities"}, f"{model_id}: malformed kernel")
        history = tuple(row["history"])
        require(history in policy, f"{model_id}: undeclared history {history}")
        action_id = row["action_id"]
        require(action_id == policy[history], f"{model_id}: action mismatch at {history}")
        key = (history, action_id)
        require(key not in result, f"{model_id}: duplicate kernel at {history}")
        probabilities = row["probabilities"]
        require(set(probabilities) == set(alphabet), f"{model_id}: incomplete outcome alphabet at {history}")
        distribution = {
            outcome: rat(probabilities[outcome], f"{model_id}/{history}/{outcome}")
            for outcome in alphabet
        }
        require(all(Fraction(0) <= p <= Fraction(1) for p in distribution.values()),
                f"{model_id}: probability outside [0,1]")
        require(sum(distribution.values(), Fraction(0)) == 1,
                f"{model_id}: conditional probabilities do not sum exactly to one")
        result[key] = distribution
    expected_keys = {(history, action_id) for history, action_id in policy.items()}
    require(set(result) == expected_keys, f"{model_id}: missing or extra history/action kernels")
    return result


def prepare(data: dict[str, Any]) -> dict[str, Any]:
    root_fields = {
        "schema_version", "run_id", "claim_scope", "horizon", "alpha", "outcome_alphabet",
        "actions", "policy", "null_models", "challenger_models", "observed_history",
        "oracle_expectations",
    }
    require(set(data) == root_fields, "root has unknown or missing fields")
    require(data["schema_version"] == 1 and data["claim_scope"] == "synthetic_known_truth",
            "V1 accepts synthetic known-truth fixtures only")
    require(data["horizon"] == 2, "V1 horizon must be exactly two")
    alpha = rat(data["alpha"], "alpha")
    require(Fraction(0) < alpha < Fraction(1), "alpha must be in (0,1)")
    threshold = 1 / alpha
    alphabet = tuple(data["outcome_alphabet"])
    require(alphabet == ("H", "L"), "V1 outcome alphabet must be exactly H,L")

    actions: dict[str, dict[str, Any]] = {}
    for action in data["actions"]:
        require(set(action) == {"action_id", "feasibility_status", "feasibility_receipt_sha256"},
                "malformed action record")
        action_id = action["action_id"]
        require(isinstance(action_id, str) and action_id and action_id not in actions,
                "duplicate or empty action ID")
        require(action["feasibility_status"] == "SYNTHETIC_FIXTURE_ELIGIBLE",
                f"{action_id}: action is not eligible")
        digest = action["feasibility_receipt_sha256"]
        require(isinstance(digest, str) and SHA64.fullmatch(digest),
                f"{action_id}: malformed feasibility receipt digest")
        actions[action_id] = action

    policy: dict[tuple[str, ...], str] = {}
    for row in data["policy"]:
        require(set(row) == {"history", "action_id"},
                "policy rows may contain only observed history and action ID")
        history = tuple(row["history"])
        require(len(history) < 2 and all(value in alphabet for value in history),
                "invalid policy history")
        require(history not in policy, f"duplicate policy history {history}")
        action_id = row["action_id"]
        require(action_id in actions, f"policy references unknown action {action_id}")
        policy[history] = action_id
    require(set(policy) == {(), ("H",), ("L",)},
            "policy must cover empty, H and L histories exactly once")

    null_models: dict[str, dict[tuple[tuple[str, ...], str], dict[str, Fraction]]] = {}
    for model in data["null_models"]:
        model_id = model.get("model_id")
        require(model_id not in null_models, "duplicate null model ID")
        null_models[model_id] = parse_kernels(model, policy, alphabet, challenger=False)
    require(len(null_models) >= 1, "empty null model family")

    challenger_models: dict[str, dict[tuple[tuple[str, ...], str], dict[str, Fraction]]] = {}
    challenger_priors: dict[str, Fraction] = {}
    for model in data["challenger_models"]:
        model_id = model.get("model_id")
        require(model_id not in challenger_models and model_id not in null_models,
                "duplicate or cross-family model ID")
        prior = rat(model["prior_probability"], f"{model_id}: prior")
        require(Fraction(0) < prior <= Fraction(1), "challenger prior must be in (0,1]")
        challenger_priors[model_id] = prior
        challenger_models[model_id] = parse_kernels(model, policy, alphabet, challenger=True)
    require(challenger_models, "empty challenger family")
    require(sum(challenger_priors.values(), Fraction(0)) == 1,
            "challenger priors must sum exactly to one")

    observed = data["observed_history"]
    require(len(observed) <= 2, "observed history exceeds the frozen horizon")
    for index, row in enumerate(observed):
        require(set(row) == {"action_id", "outcome_id"}, "malformed observed step")
        history = tuple(step["outcome_id"] for step in observed[:index])
        require(history in policy, f"missing policy action at observed history {history}")
        require(row["action_id"] == policy[history],
                f"observed action is not the frozen policy action at {history}")
        require(isinstance(row["outcome_id"], str) and row["outcome_id"],
                "empty observed outcome ID")

    return {
        "data": data,
        "alpha": alpha,
        "threshold": threshold,
        "alphabet": alphabet,
        "actions": actions,
        "policy": policy,
        "null_models": null_models,
        "challenger_models": challenger_models,
        "challenger_priors": challenger_priors,
        "all_models": {**null_models, **challenger_models},
    }


def history_likelihood(model_id: str, history: tuple[str, ...], prepared: dict[str, Any]) -> Fraction:
    model = prepared["all_models"][model_id]
    likelihood = Fraction(1)
    for index, outcome in enumerate(history):
        prefix = history[:index]
        action_id = prepared["policy"][prefix]
        likelihood *= model[(prefix, action_id)][outcome]
    return likelihood


def challenger_predictive(
    history: tuple[str, ...], prepared: dict[str, Any]
) -> dict[str, Fraction] | None:
    if history not in prepared["policy"]:
        return None
    masses = {
        model_id: prior * history_likelihood(model_id, history, prepared)
        for model_id, prior in prepared["challenger_priors"].items()
    }
    normalizer = sum(masses.values(), Fraction(0))
    if normalizer == 0:
        return None
    action_id = prepared["policy"][history]
    result = {
        outcome: sum(
            (
                masses[model_id]
                * prepared["challenger_models"][model_id][(history, action_id)][outcome]
                for model_id in masses
            ),
            Fraction(0),
        ) / normalizer
        for outcome in prepared["alphabet"]
    }
    require(sum(result.values(), Fraction(0)) == 1,
            f"challenger predictive is not normalized at {history}")
    return result


def null_envelope(history: tuple[str, ...], prepared: dict[str, Any]) -> dict[str, Fraction]:
    action_id = prepared["policy"][history]
    return {
        outcome: max(model[(history, action_id)][outcome]
                     for model in prepared["null_models"].values())
        for outcome in prepared["alphabet"]
    }


def factor_text(q: Fraction, p_star: Fraction) -> str:
    if p_star == 0:
        return "INFINITY" if q > 0 else "OUT_OF_SUPPORT_UNMODELED"
    return rational_text(q / p_star)


def e_factor_table(history: tuple[str, ...], prepared: dict[str, Any]) -> dict[str, str]:
    predictive = challenger_predictive(history, prepared)
    require(predictive is not None, f"challenger mixture exhausted at {history}")
    envelope = null_envelope(history, prepared)
    return {
        outcome: factor_text(predictive[outcome], envelope[outcome])
        for outcome in prepared["alphabet"]
    }


def validate_false_alarm_bounds(false_alarm: dict[str, Fraction], alpha: Fraction) -> None:
    for model_id, probability in false_alarm.items():
        require(probability <= alpha,
                f"false-alarm probability under {model_id} exceeds alpha: "
                f"{rational_text(probability)} > {rational_text(alpha)}")


def crossing_probability(null_model_id: str, prepared: dict[str, Any]) -> Fraction:
    model = prepared["null_models"][null_model_id]
    threshold = prepared["threshold"]

    def walk(history: tuple[str, ...], wealth: Fraction) -> Fraction:
        if len(history) >= 2:
            return Fraction(0)
        predictive = challenger_predictive(history, prepared)
        if predictive is None:
            return Fraction(0)
        action_id = prepared["policy"][history]
        null_distribution = model[(history, action_id)]
        envelope = null_envelope(history, prepared)
        total = Fraction(0)
        for outcome in prepared["alphabet"]:
            probability = null_distribution[outcome]
            if probability == 0:
                continue
            denominator = envelope[outcome]
            require(denominator > 0, "null probability contradicts zero null envelope")
            next_wealth = wealth * predictive[outcome] / denominator
            if next_wealth >= threshold:
                total += probability
            else:
                total += probability * walk(history + (outcome,), next_wealth)
        return total

    return walk((), Fraction(1))


def crossing_paths(prepared: dict[str, Any]) -> list[list[str]]:
    result: list[list[str]] = []

    def walk(history: tuple[str, ...], wealth: Fraction) -> None:
        if len(history) >= 2:
            return
        predictive = challenger_predictive(history, prepared)
        if predictive is None:
            return
        envelope = null_envelope(history, prepared)
        for outcome in prepared["alphabet"]:
            q = predictive[outcome]
            p_star = envelope[outcome]
            if p_star == 0:
                if q > 0:
                    result.append(list(history + (outcome,)))
                continue
            next_wealth = wealth * q / p_star
            if next_wealth >= prepared["threshold"]:
                result.append(list(history + (outcome,)))
            else:
                walk(history + (outcome,), next_wealth)

    walk((), Fraction(1))
    return result


def replay_observed(prepared: dict[str, Any]) -> dict[str, Any]:
    history: tuple[str, ...] = ()
    wealth = Fraction(1)
    path_wealth: list[str] = []
    status: str | None = None
    for row in prepared["data"]["observed_history"]:
        require(history in prepared["policy"], "observed history exceeds policy support")
        require(row["action_id"] == prepared["policy"][history],
                f"observed action mismatch at history {history}")
        outcome = row["outcome_id"]
        if outcome not in prepared["alphabet"]:
            status = "OUT_OF_SUPPORT_UNMODELED"
            break
        predictive = challenger_predictive(history, prepared)
        if predictive is None:
            status = "CHALLENGER_MIXTURE_EXHAUSTED"
            break
        envelope = null_envelope(history, prepared)
        q = predictive[outcome]
        p_star = envelope[outcome]
        if p_star == 0:
            if q > 0:
                status = "MODEL_DISCREPANCY_DETECTED"
                path_wealth.append("INFINITY")
            else:
                status = "OUT_OF_SUPPORT_UNMODELED"
            break
        wealth *= q / p_star
        path_wealth.append(rational_text(wealth))
        history = history + (outcome,)
        if wealth >= prepared["threshold"]:
            status = "MODEL_DISCREPANCY_DETECTED"
            break
    if status is None:
        status = (
            "NO_DISCREPANCY_DETECTED_WITHIN_DECLARED_CHALLENGE_FAMILY"
            if len(path_wealth) == 2
            else "MONITORING_CONTINUES"
        )
    return {
        "status": status,
        "observed_e_value": rational_text(wealth),
        "path_e_values": path_wealth,
    }


def verify_expected(report: dict[str, Any], expected: dict[str, Any]) -> None:
    require(report["challenger_predictive_by_history"] == expected["challenger_predictive_by_history"],
            "challenger predictive known-answer mismatch")
    require(report["e_factors_by_history"] == expected["e_factors_by_history"],
            "e-factor known-answer mismatch")
    require(report["crossing_path"] == expected["crossing_paths"], "crossing-path set mismatch")
    require(report["observed"]["observed_e_value"] == expected["observed_e_value"],
            "observed e-value mismatch")
    require(report["false_alarm_probability_by_null"] == expected["false_alarm_probability_by_null"],
            "false-alarm known-answer mismatch")
    require(report["observed"]["status"] == expected["expected_status"], "observed status mismatch")


def verify(data: dict[str, Any], check_expected: bool = True) -> dict[str, Any]:
    prepared = prepare(data)
    false_alarm = {
        model_id: crossing_probability(model_id, prepared)
        for model_id in prepared["null_models"]
    }
    validate_false_alarm_bounds(false_alarm, prepared["alpha"])

    histories = [(), ("H",), ("L",)]
    predictive_rows = []
    factor_rows = []
    for history in histories:
        predictive = challenger_predictive(history, prepared)
        if predictive is None:
            predictive_rows.append({
                "history": list(history),
                "status": "CHALLENGER_MIXTURE_EXHAUSTED",
            })
            factor_rows.append({
                "history": list(history),
                "status": "CHALLENGER_MIXTURE_EXHAUSTED",
            })
            continue
        predictive_rows.append({
            "history": list(history),
            "probabilities": {
                outcome: rational_text(predictive[outcome])
                for outcome in prepared["alphabet"]
            },
        })
        factor_rows.append({
            "history": list(history),
            "factors": e_factor_table(history, prepared),
        })

    observed = replay_observed(prepared)
    report = {
        "run_id": data["run_id"],
        "campaign_sha256": canonical_digest(data),
        "alpha": rational_text(prepared["alpha"]),
        "threshold": rational_text(prepared["threshold"]),
        "challenger_predictive_by_history": predictive_rows,
        "e_factors_by_history": factor_rows,
        "crossing_path": crossing_paths(prepared),
        "observed": observed,
        "false_alarm_probability_by_null": {
            model_id: rational_text(probability)
            for model_id, probability in false_alarm.items()
        },
        "status": "SYNTHETIC_CONTRACT_VALIDATED",
        "model_adequacy_claim": "NOT_ESTABLISHED_BY_FINITE_CHALLENGE_FAMILY",
    }
    if check_expected and observed["status"] not in {
        "OUT_OF_SUPPORT_UNMODELED", "CHALLENGER_MIXTURE_EXHAUSTED"
    }:
        verify_expected(report, data["oracle_expectations"])
    return report


def expect_rejection(label: str, operation: Any) -> None:
    try:
        operation()
    except (ValueError, KeyError, TypeError):
        return
    raise ValueError(f"negative control failed to reject: {label}")


def self_test() -> None:
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    report = verify(data)
    require(report["observed"]["observed_e_value"] == "325/144", "HH wealth mismatch")
    require(report["false_alarm_probability_by_null"] == {"N0": "1/4", "N1": "9/25"},
            "exact false-alarm probabilities mismatch")

    expect_rejection(
        "precomputed false-alarm probability above declared alpha",
        lambda: validate_false_alarm_bounds({"N1": Fraction(9, 25)}, Fraction(1, 4)),
    )

    bad = copy.deepcopy(data)
    bad["policy"][0]["hidden_model_id"] = "N0"
    expect_rejection("hidden model ID in policy", lambda: verify(bad, check_expected=False))

    bad = copy.deepcopy(data)
    bad["null_models"][0]["kernels"].pop()
    expect_rejection("missing kernel", lambda: verify(bad, check_expected=False))

    bad = copy.deepcopy(data)
    bad["challenger_models"][0]["kernels"][0]["probabilities"]["H"] = "2/3"
    expect_rejection("unnormalized kernel", lambda: verify(bad, check_expected=False))

    bad = copy.deepcopy(data)
    bad["observed_history"][0]["action_id"] = "follow-H"
    expect_rejection("policy/action mismatch", lambda: verify(bad, check_expected=False))

    bad = copy.deepcopy(data)
    bad["observed_history"] = [{"action_id": "screen", "outcome_id": "Z"}]
    out_of_support = verify(bad, check_expected=False)
    require(out_of_support["observed"]["status"] == "OUT_OF_SUPPORT_UNMODELED",
            "out-of-alphabet observation was not kept separate")

    bad = copy.deepcopy(data)
    bad["observed_history"] = [
        {"action_id": "screen", "outcome_id": "L"},
        {"action_id": "follow-L", "outcome_id": "H"},
    ]
    for model in bad["challenger_models"]:
        model["kernels"][0]["probabilities"] = {"H": "1", "L": "0"}
    exhausted = verify(bad, check_expected=False)
    require(exhausted["observed"]["status"] == "CHALLENGER_MIXTURE_EXHAUSTED",
            "challenger support exhaustion was silently refreshed")

    print("anytime-valid discrepancy monitor self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", nargs="?", help="monitor input JSON")
    parser.add_argument("--self-test", action="store_true", help="run known-answer and mutation controls")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        if args.fixture is None:
            return 0
    if args.fixture is None:
        parser.error("provide a fixture or --self-test")
    data = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    print(json.dumps(verify(data), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
