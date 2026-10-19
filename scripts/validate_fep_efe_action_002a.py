#!/usr/bin/env python3
"""Independent standard-library oracle for the frozen finite-categorical EIG corpus.

This is a qualification helper, not a production EIG implementation. It derives
both equivalent EIG formulas from model probabilities; case IDs are used only
for order/integrity and declared metamorphic comparisons, never as an answer map.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any


EXPECTED_SCHEMA = "fep-efe-action-002a-reference-v1"
EXPECTED_STATUS = "frozen_unqualified_reference_source"
EXPECTED_PROFILE = "FiniteCategoricalExpectedInformationGainV1"
EXPECTED_CASE_COUNT = 35
EXPECTED_MUTATION_COUNT = 10
TOL = 1.0e-12


class CorpusError(Exception):
    """A deterministic source/corpus or model-contract failure."""


class InvalidModel(Exception):
    """A model is outside the frozen probability-domain contract."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CorpusError(message)


def entropy(probabilities: list[float]) -> float:
    """Shannon entropy in nats; the 0*ln(0)=0 convention is explicit."""
    return -math.fsum(p * math.log(p) for p in probabilities if p > 0.0)


def parse_probability(value: Any, context: str) -> float:
    # JSON numbers are deliberately forbidden: source data must preserve the
    # declared decimal-string representation and malformed-shape mutant I31.
    if not isinstance(value, str):
        raise InvalidModel("probability_not_decimal_string", context)
    try:
        parsed = float(value)
    except ValueError as error:
        raise InvalidModel("non_finite_probability", context) from error
    if not math.isfinite(parsed):
        raise InvalidModel("non_finite_probability", context)
    return parsed


def model_eig(case: dict[str, Any]) -> dict[str, Any]:
    """Compute expected-KL and entropy-reduction forms independently."""
    states = case.get("state_ids")
    observations = case.get("observation_ids")
    raw_prior = case.get("prior")
    raw_likelihood = case.get("likelihood_by_state")

    if not isinstance(states, list) or len(states) == 0:
        raise InvalidModel("empty_state_support")
    if not isinstance(observations, list) or len(observations) == 0:
        raise InvalidModel("empty_observation_support")
    if any(not isinstance(x, str) or not x for x in states + observations):
        raise InvalidModel("empty_or_invalid_support_id")
    if len(set(states)) != len(states) or len(set(observations)) != len(observations):
        raise InvalidModel("duplicate_support_id")
    if not isinstance(raw_prior, list) or len(raw_prior) != len(states):
        raise InvalidModel("prior_state_dimension_mismatch")
    if not isinstance(raw_likelihood, list) or len(raw_likelihood) != len(states):
        raise InvalidModel("likelihood_state_dimension_mismatch")

    # Check matrix shape before parsing individual values. This makes the
    # expected I31 dimension rejection deterministic even though its entries
    # also intentionally use the wrong JSON scalar type.
    for state_index, row in enumerate(raw_likelihood):
        if not isinstance(row, list) or len(row) != len(observations):
            raise InvalidModel(
                "likelihood_observation_dimension_mismatch",
                f"row {state_index}",
            )

    prior = [
        parse_probability(value, f"prior[{index}]")
        for index, value in enumerate(raw_prior)
    ]
    if any(p < 0.0 for p in prior):
        raise InvalidModel("negative_prior")
    if any(p > 1.0 for p in prior):
        raise InvalidModel("prior_probability_above_one")
    if abs(math.fsum(prior) - 1.0) > TOL:
        raise InvalidModel("prior_not_normalized")

    likelihood: list[list[float]] = []
    for state_index, raw_row in enumerate(raw_likelihood):
        row = [
            parse_probability(value, f"likelihood[{state_index}][{obs_index}]")
            for obs_index, value in enumerate(raw_row)
        ]
        if any(value < 0.0 for value in row):
            raise InvalidModel("negative_likelihood", f"row {state_index}")
        if any(value > 1.0 for value in row):
            raise InvalidModel("likelihood_probability_above_one", f"row {state_index}")
        if abs(math.fsum(row) - 1.0) > TOL:
            raise InvalidModel("likelihood_row_not_normalized", f"row {state_index}")
        likelihood.append(row)

    predictive = [
        math.fsum(prior[s] * likelihood[s][o] for s in range(len(states)))
        for o in range(len(observations))
    ]
    if any(not math.isfinite(p) or p < -TOL for p in predictive):
        raise InvalidModel("invalid_predictive_observation_distribution")

    # Form 1: sum_o q(o|a) KL(q(s|o,a) || q(s)).
    expected_kl_terms: list[float] = []
    posterior_entropy_terms: list[float] = []
    posterior_by_observation: list[list[float] | None] = []
    per_observation_kl_contribution: list[float] = []

    for observation_index, observation_mass in enumerate(predictive):
        if observation_mass <= 0.0:
            # A zero-mass branch contributes zero and has no posterior.
            posterior_by_observation.append(None)
            per_observation_kl_contribution.append(0.0)
            continue

        posterior = [
            prior[s] * likelihood[s][observation_index] / observation_mass
            for s in range(len(states))
        ]
        if any(not math.isfinite(p) or p < -TOL for p in posterior):
            raise InvalidModel("invalid_hypothetical_posterior")
        if abs(math.fsum(posterior) - 1.0) > 10.0 * TOL:
            raise InvalidModel("hypothetical_posterior_not_normalized")

        kl_terms: list[float] = []
        for state_index, posterior_mass in enumerate(posterior):
            if posterior_mass <= 0.0:
                continue
            prior_mass = prior[state_index]
            if prior_mass <= 0.0:
                raise InvalidModel("posterior_support_exceeds_prior_support")
            kl_terms.append(posterior_mass * math.log(posterior_mass / prior_mass))

        branch_kl = math.fsum(kl_terms)
        contribution = observation_mass * branch_kl
        expected_kl_terms.append(contribution)
        per_observation_kl_contribution.append(contribution)
        posterior_entropy_terms.append(observation_mass * entropy(posterior))
        posterior_by_observation.append(posterior)

    eig_expected_kl = math.fsum(expected_kl_terms)

    # Form 2: H(prior) - E_o H(posterior | o). This path uses the prior
    # entropy and posterior entropy sum, not the KL values above.
    eig_entropy_reduction = entropy(prior) - math.fsum(posterior_entropy_terms)
    if not math.isfinite(eig_expected_kl) or not math.isfinite(eig_entropy_reduction):
        raise InvalidModel("non_finite_eig")
    if abs(eig_expected_kl - eig_entropy_reduction) > TOL * max(
        1.0, abs(eig_expected_kl), abs(eig_entropy_reduction)
    ):
        raise CorpusError(
            f"{case.get('id')}: equivalent EIG forms disagree: "
            f"{eig_expected_kl!r} vs {eig_entropy_reduction!r}"
        )

    eig = 0.5 * (eig_expected_kl + eig_entropy_reduction)
    return {
        "eig_expected_kl": eig_expected_kl,
        "eig_entropy_reduction": eig_entropy_reduction,
        "eig": eig,
        "predictive_observation": predictive,
        "posterior_by_observation": posterior_by_observation,
        "per_observation_kl_contribution": per_observation_kl_contribution,
    }


def reverse_kl_mutant(case: dict[str, Any]) -> float:
    """Evaluate the deliberately wrong KL(prior || posterior) mutant."""
    model = model_eig(case)
    prior = [parse_probability(v, "reverse-KL prior") for v in case["prior"]]
    total: list[float] = []
    for observation_mass, posterior in zip(
        model["predictive_observation"], model["posterior_by_observation"]
    ):
        if observation_mass <= 0.0:
            continue
        if posterior is None:
            raise CorpusError("positive-mass observation unexpectedly has no posterior")
        terms: list[float] = []
        for prior_mass, posterior_mass in zip(prior, posterior):
            if prior_mass <= 0.0:
                continue
            if posterior_mass <= 0.0:
                return math.inf
            terms.append(prior_mass * math.log(prior_mass / posterior_mass))
        total.append(observation_mass * math.fsum(terms))
    return math.fsum(total)


def symbolic_expectation(expression: str) -> float:
    """A closed, reviewed expression vocabulary; never evaluates arbitrary code."""
    h = entropy
    allowed = {
        "0": 0.0,
        "ln(2)": math.log(2.0),
        "ln(3)": math.log(3.0),
        "ln(2)-H(1/4)": math.log(2.0) - h([0.25, 0.75]),
        "ln(2)-H(1/10)": math.log(2.0) - h([0.1, 0.9]),
        "H(1/4)": h([0.25, 0.75]),
        "ln(3)-(2/3)*ln(2)": math.log(3.0) - (2.0 / 3.0) * math.log(2.0),
        "ln(2)-H(2/5)": math.log(2.0) - h([0.4, 0.6]),
        "H(1/100)": h([0.01, 0.99]),
        "H((1/5,3/10,1/2))": h([0.2, 0.3, 0.5]),
        "H(0.55)-0.5*H(0.8)-0.5*H(0.3)": (
            h([0.55, 0.45]) - 0.5 * h([0.8, 0.2]) - 0.5 * h([0.3, 0.7])
        ),
    }
    if expression not in allowed:
        raise CorpusError(f"unreviewed symbolic expression: {expression}")
    return allowed[expression]


def validate_corpus(document: dict[str, Any]) -> dict[str, Any]:
    expected_top = {
        "schema_version",
        "status",
        "intended_issue",
        "profile",
        "formulas",
        "identity_contract",
        "numeric_policy",
        "cases",
        "mutation_controls",
        "claim_ceiling",
    }
    require(set(document) == expected_top, "unknown or missing top-level corpus field")
    require(document["schema_version"] == EXPECTED_SCHEMA, "schema version mismatch")
    require(document["status"] == EXPECTED_STATUS, "unexpected source status")
    require(document["profile"] == EXPECTED_PROFILE, "EIG profile identity mismatch")
    require(
        document["numeric_policy"].get("normalization_tolerance") == TOL,
        "normalization tolerance drift",
    )
    require(document["numeric_policy"].get("no_silent_renormalization") is True,
            "silent normalization must remain disabled")
    require(document["formulas"].get("score_orientation") ==
            "E_info(a)=-EIG(a) under the EFE minimization convention",
            "EIG sign convention drift")

    cases = document["cases"]
    require(isinstance(cases, list) and len(cases) == EXPECTED_CASE_COUNT,
            "case count mismatch")
    case_ids = [case.get("id") for case in cases]
    require(len(set(case_ids)) == len(case_ids), "duplicate case IDs")
    for expected_number, case in enumerate(cases, start=1):
        require(
            isinstance(case.get("id"), str)
            and case["id"].startswith(f"I{expected_number:02d}-"),
            f"case ordering mismatch at index {expected_number}",
        )
        expected_case_keys = {
            "id", "kind", "state_ids", "prior", "observation_ids",
            "likelihood_by_state", "expected",
        }
        require(set(case) == expected_case_keys, f"{case['id']}: unknown or missing case field")
        expected = case.get("expected")
        require(isinstance(expected, dict), f"{case['id']}: expected object missing")
        allowed_expected = {
            "valid", "eig_expression", "relation_to_zero", "compare_to",
            "relation_to_compared_case", "zero_mass_branch",
            "asymmetric_branch_contributions", "reject_reason",
        }
        require(set(expected).issubset(allowed_expected),
                f"{case['id']}: unknown expected field")

    valid_cases = [case for case in cases if case["expected"].get("valid") is True]
    invalid_cases = [case for case in cases if case["expected"].get("valid") is False]
    require(len(valid_cases) == 23, "valid case count mismatch")
    require(len(invalid_cases) == 12, "invalid case count mismatch")
    require(all(case["kind"] == "valid_model" for case in valid_cases),
            "valid case kind mismatch")
    require(all(case["kind"] == "invalid_model" for case in invalid_cases),
            "invalid case kind mismatch")

    results: dict[str, dict[str, Any]] = {}
    expression_errors: list[float] = []
    for case in valid_cases:
        result = model_eig(case)
        case_id = case["id"]
        expected = case["expected"]
        relation = expected.get("relation_to_zero")
        require(relation in {"equal", "strictly_positive"},
                f"{case_id}: unsupported relation-to-zero")
        if relation == "equal":
            require(abs(result["eig"]) <= TOL, f"{case_id}: expected exact/tolerance tie")
        else:
            require(result["eig"] > TOL, f"{case_id}: expected strictly positive EIG")
        expression = expected.get("eig_expression")
        require(isinstance(expression, str), f"{case_id}: symbolic expectation missing")
        symbolic = symbolic_expectation(expression)
        expression_error = abs(result["eig"] - symbolic)
        expression_errors.append(expression_error)
        require(expression_error <= 5.0 * TOL, f"{case_id}: symbolic expression mismatch")
        results[case_id] = result

    rejection_reasons: dict[str, str] = {}
    for case in invalid_cases:
        case_id = case["id"]
        try:
            model_eig(case)
        except InvalidModel as error:
            rejection_reasons[case_id] = error.reason
            require(
                error.reason == case["expected"].get("reject_reason"),
                f"{case_id}: expected {case['expected'].get('reject_reason')}, "
                f"got {error.reason}",
            )
        else:
            raise CorpusError(f"{case_id}: malformed model was accepted")

    relation_results: list[dict[str, Any]] = []
    for case in valid_cases:
        expected = case["expected"]
        other_id = expected.get("compare_to")
        if other_id is None:
            require("relation_to_compared_case" not in expected,
                    f"{case['id']}: comparison relation without target")
            continue
        require(other_id in results, f"{case['id']}: comparison target missing")
        relation = expected.get("relation_to_compared_case")
        delta = results[case["id"]]["eig"] - results[other_id]["eig"]
        if relation == "equal":
            ok = abs(delta) <= TOL
        elif relation == "less":
            ok = delta < -TOL
        elif relation == "greater":
            ok = delta > TOL
        else:
            raise CorpusError(f"{case['id']}: invalid metamorphic comparison")
        require(ok, f"{case['id']}: metamorphic relation to {other_id} failed")
        relation_results.append({"id": case["id"], "target": other_id,
                                 "relation": relation, "delta": delta})

    # The asymmetric fixture ensures omitting the expectation over observations
    # is observable: individual positive-mass branches have unequal contributions.
    asymmetric = results["I35-asymmetric-observation-branches"]
    branch_contributions = asymmetric["per_observation_kl_contribution"]
    require(len(branch_contributions) == 2, "I35 must have two observation branches")
    require(abs(branch_contributions[0] - branch_contributions[1]) > TOL,
            "I35 no longer distinguishes asymmetric branch contributions")
    first_branch_only_mutant = branch_contributions[0]
    require(abs(first_branch_only_mutant - asymmetric["eig"]) > TOL,
            "I35 no longer detects dropping the expectation over observations")

    # Mutation 01: flatten state-conditioned likelihoods to the prior-weighted
    # mean observation distribution. This destroys information in I03 and must
    # not pass as the genuine state-conditioned likelihood model.
    informative_case = next(case for case in valid_cases
                            if case["id"] == "I03-binary-symmetric-error-quarter")
    flattened = copy.deepcopy(informative_case)
    prior_for_flatten = [parse_probability(v, "flatten mutant prior")
                         for v in flattened["prior"]]
    raw_rows = [
        [parse_probability(v, "flatten mutant likelihood") for v in row]
        for row in flattened["likelihood_by_state"]
    ]
    mean_observation = [
        math.fsum(prior_for_flatten[s] * raw_rows[s][o]
                  for s in range(len(prior_for_flatten)))
        for o in range(len(flattened["observation_ids"]))
    ]
    flattened["likelihood_by_state"] = [
        [repr(value) for value in mean_observation]
        for _ in flattened["state_ids"]
    ]
    flattened_eig = model_eig(flattened)["eig"]
    require(abs(flattened_eig - results[informative_case["id"]]["eig"]) > TOL,
            "state-independent observation-mean mutant was not detected")

    # Mutation 03: the reversed KL direction is a different objective.
    asymmetric_case = next(case for case in valid_cases
                           if case["id"] == "I35-asymmetric-observation-branches")
    reversed_kl = reverse_kl_mutant(asymmetric_case)
    require(math.isfinite(reversed_kl), "reverse-KL witness should be finite")
    require(abs(reversed_kl - results[asymmetric_case["id"]]["eig"]) > TOL,
            "reversed KL direction mutant was not detected")

    # Mutation 07: zero predictive-mass branches have no fabricated posterior.
    impossible = results["I09-impossible-observation-branch"]
    zero_indices = [i for i, mass in enumerate(impossible["predictive_observation"])
                    if mass == 0.0]
    require(bool(zero_indices), "I09 must contain a zero-mass observation branch")
    require(all(impossible["posterior_by_observation"][i] is None
                for i in zero_indices),
            "zero-mass observation branch must not fabricate a posterior")

    # Mutation 08: the stronger objective identity cannot silently change.
    require(document["profile"] == EXPECTED_PROFILE,
            "predicted-entropy proxy/profile promotion mutant not detected")

    # Mutation 10: EIG is nonnegative; its configured minimization contribution
    # is -EIG and therefore nonpositive (strictly negative for positive EIG).
    revealing_eig = results["I02-perfect-binary-reveal"]["eig"]
    require(revealing_eig > 0.0 and -revealing_eig < 0.0,
            "EIG/score orientation mutation not detected")

    # Input-mutant controls are independently exercised by the malformed cases.
    require(rejection_reasons.get("I24-unnormalized-prior") == "prior_not_normalized",
            "silent-prior-renormalization mutant not detected")
    require(rejection_reasons.get("I26-negative-likelihood") == "negative_likelihood",
            "negative-likelihood mutant not detected")

    controls = document["mutation_controls"]
    require(isinstance(controls, list) and len(controls) == EXPECTED_MUTATION_COUNT,
            "mutation-control count mismatch")
    control_ids = [item.get("id") for item in controls]
    require(len(set(control_ids)) == len(control_ids), "duplicate mutation-control ID")
    for control in controls:
        require(set(control) == {
            "id", "mutation", "witness_case", "expected_detection"
        }, f"{control.get('id')}: mutation-control shape drift")
        require(control.get("witness_case") in results or
                control.get("witness_case") in rejection_reasons,
                f"{control.get('id')}: witness case missing")

    return {
        "result": "PASS",
        "profile": EXPECTED_PROFILE,
        "schema_version": EXPECTED_SCHEMA,
        "case_count": len(cases),
        "valid_cases": len(valid_cases),
        "invalid_cases": len(invalid_cases),
        "mutation_controls": len(controls),
        "max_expected_kl_vs_entropy_form_abs_difference": max(
            abs(result["eig_expected_kl"] - result["eig_entropy_reduction"])
            for result in results.values()
        ),
        "max_symbolic_expectation_abs_difference": max(expression_errors),
        "metamorphic_comparisons": len(relation_results),
        "invalid_cases_rejected": len(rejection_reasons),
        "asymmetric_branch_contributions": branch_contributions,
        "mutation_checks_executed": [
            "state-independent likelihood/mean flattening",
            "dropping expectation over asymmetric observations",
            "reversed KL direction",
            "prior normalization rejection",
            "negative likelihood rejection",
            "state/observation relabeling invariance",
            "zero-mass branch without posterior",
            "proxy/EIG profile identity",
            "EIG score orientation",
        ],
        "invalid_rejection_reasons": rejection_reasons,
        "hypothetical_state_purity": "NOT_TESTED_BY_PROBABILITY_ORACLE",
        "production_correspondence": "NOT_ESTABLISHED",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus",
        default="docs/fep/data/fep_efe_action_002a_reference_v1.json",
    )
    args = parser.parse_args()
    try:
        raw = Path(args.corpus).read_text(encoding="utf-8")
        document = json.loads(raw)
        if not isinstance(document, dict):
            raise CorpusError("corpus top-level value must be an object")
        report = validate_corpus(document)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (OSError, json.JSONDecodeError, CorpusError, InvalidModel) as error:
        print(json.dumps({"result": "FAIL", "error": str(error)}, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
