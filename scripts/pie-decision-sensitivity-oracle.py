#!/usr/bin/env python3
"""PIE-009C independent decision-sensitivity / experiment-priority oracle.

Uses interval corners and Pareto membership only. It does not assign probabilities
or collapse multiple objectives into a hidden weighted score.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Parameter:
    parameter_id: str
    low: float
    nominal: float
    high: float


@dataclass(frozen=True)
class Alternative:
    alternative_id: str
    seed_mass_t: float
    base_output: float
    output_coeffs: dict[str, float]
    base_closure: float
    closure_coeffs: dict[str, float]


@dataclass(frozen=True)
class Evaluation:
    alternative_id: str
    seed_mass_t: float
    useful_output: float
    critical_closure: float


@dataclass(frozen=True)
class SensitivityResult:
    parameter_id: str
    baseline_conditional_count: int
    fixed_conditional_count: int
    ambiguity_reduction: int


def validate_parameters(parameters: tuple[Parameter, ...]) -> None:
    seen = set()
    for parameter in parameters:
        if not parameter.parameter_id or parameter.parameter_id in seen:
            raise ValueError("parameter IDs must be unique and non-empty")
        seen.add(parameter.parameter_id)
        if not all(math.isfinite(x) for x in (parameter.low, parameter.nominal, parameter.high)):
            raise ValueError("parameter values must be finite")
        if parameter.low > parameter.nominal or parameter.nominal > parameter.high:
            raise ValueError("parameter envelope must satisfy low <= nominal <= high")


def validate_alternatives(
    alternatives: tuple[Alternative, ...], parameter_ids: set[str]
) -> None:
    seen = set()
    for alternative in alternatives:
        if not alternative.alternative_id or alternative.alternative_id in seen:
            raise ValueError("alternative IDs must be unique and non-empty")
        seen.add(alternative.alternative_id)
        if not math.isfinite(alternative.seed_mass_t) or alternative.seed_mass_t <= 0:
            raise ValueError("seed mass must be positive finite")
        if not all(math.isfinite(x) for x in (alternative.base_output, alternative.base_closure)):
            raise ValueError("base objectives must be finite")
        if not set(alternative.output_coeffs) <= parameter_ids:
            raise ValueError("alternative references unknown output parameter")
        if not set(alternative.closure_coeffs) <= parameter_ids:
            raise ValueError("alternative references unknown closure parameter")
        if not all(math.isfinite(x) for x in alternative.output_coeffs.values()):
            raise ValueError("output coefficients must be finite")
        if not all(math.isfinite(x) for x in alternative.closure_coeffs.values()):
            raise ValueError("closure coefficients must be finite")


def corners(
    parameters: tuple[Parameter, ...], fixed_nominal: str | None = None
) -> list[dict[str, float]]:
    validate_parameters(parameters)
    axes = []
    for parameter in parameters:
        if parameter.parameter_id == fixed_nominal:
            axes.append([(parameter.parameter_id, parameter.nominal)])
        else:
            axes.append(
                [
                    (parameter.parameter_id, parameter.low),
                    (parameter.parameter_id, parameter.high),
                ]
            )
    return [dict(combo) for combo in itertools.product(*axes)]


def evaluate(alternative: Alternative, values: dict[str, float]) -> Evaluation:
    output = alternative.base_output + sum(
        alternative.output_coeffs.get(key, 0.0) * value
        for key, value in values.items()
    )
    closure = alternative.base_closure + sum(
        alternative.closure_coeffs.get(key, 0.0) * value
        for key, value in values.items()
    )
    return Evaluation(
        alternative.alternative_id,
        alternative.seed_mass_t,
        output,
        closure,
    )


def dominates(a: Evaluation, b: Evaluation) -> bool:
    no_worse = (
        a.seed_mass_t <= b.seed_mass_t
        and a.useful_output >= b.useful_output
        and a.critical_closure >= b.critical_closure
    )
    strict = (
        a.seed_mass_t < b.seed_mass_t
        or a.useful_output > b.useful_output
        or a.critical_closure > b.critical_closure
    )
    return no_worse and strict


def frontier(evaluations: list[Evaluation]) -> set[str]:
    return {
        evaluation.alternative_id
        for evaluation in evaluations
        if not any(
            dominates(other, evaluation)
            for other in evaluations
            if other != evaluation
        )
    }


def membership(
    parameters: tuple[Parameter, ...],
    alternatives: tuple[Alternative, ...],
    fixed_nominal: str | None = None,
) -> dict[str, set[bool]]:
    validate_parameters(parameters)
    parameter_ids = {parameter.parameter_id for parameter in parameters}
    validate_alternatives(alternatives, parameter_ids)
    out = {alternative.alternative_id: set() for alternative in alternatives}
    for values in corners(parameters, fixed_nominal):
        pareto = frontier([evaluate(alternative, values) for alternative in alternatives])
        for alternative in alternatives:
            out[alternative.alternative_id].add(
                alternative.alternative_id in pareto
            )
    return out


def classify(membership_values: set[bool]) -> str:
    if membership_values == {True}:
        return "AlwaysPareto"
    if membership_values == {False}:
        return "NeverPareto"
    return "ConditionalPareto"


def sensitivity(
    parameters: tuple[Parameter, ...], alternatives: tuple[Alternative, ...]
) -> list[SensitivityResult]:
    baseline = membership(parameters, alternatives)
    baseline_conditional = sum(
        classify(values) == "ConditionalPareto" for values in baseline.values()
    )
    results = []
    for parameter in parameters:
        fixed = membership(parameters, alternatives, parameter.parameter_id)
        fixed_conditional = sum(
            classify(values) == "ConditionalPareto" for values in fixed.values()
        )
        results.append(
            SensitivityResult(
                parameter.parameter_id,
                baseline_conditional,
                fixed_conditional,
                baseline_conditional - fixed_conditional,
            )
        )
    return results


def fixture() -> tuple[tuple[Parameter, ...], tuple[Alternative, ...]]:
    parameters = (
        Parameter("recovery", 0.5, 0.7, 0.9),
        Parameter("uptime", 0.7, 0.85, 1.0),
        Parameter("minor_loss", 0.0, 0.05, 0.1),
    )
    alternatives = (
        Alternative(
            "bulk-first",
            20,
            80,
            {"recovery": 100, "minor_loss": -10},
            0.45,
            {"uptime": 0.05},
        ),
        Alternative(
            "closure-first",
            20,
            130,
            {"recovery": 40, "minor_loss": -5},
            0.40,
            {"recovery": 0.20, "uptime": 0.05},
        ),
        Alternative(
            "balanced",
            19,
            120,
            {"recovery": 50, "uptime": 10, "minor_loss": -5},
            0.50,
            {"recovery": 0.05, "uptime": 0.03},
        ),
    )
    return parameters, alternatives


def self_test() -> None:
    parameters, alternatives = fixture()
    baseline_membership = membership(parameters, alternatives)
    classes = {
        key: classify(values) for key, values in baseline_membership.items()
    }
    assert "ConditionalPareto" in classes.values()
    assert len(corners(parameters)) == 8

    by_parameter = {
        result.parameter_id: result
        for result in sensitivity(parameters, alternatives)
    }
    assert by_parameter["recovery"].ambiguity_reduction > 0
    assert (
        by_parameter["recovery"].ambiguity_reduction
        >= by_parameter["minor_loss"].ambiguity_reduction
    )

    assert len(corners(parameters, "recovery")) == 4

    nominal = {parameter.parameter_id: parameter.nominal for parameter in parameters}
    nominal_frontier = frontier(
        [evaluate(alternative, nominal) for alternative in alternatives]
    )
    assert len(nominal_frontier) >= 2

    try:
        validate_parameters((Parameter("bad", 2, 1, 3),))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid parameter envelope must fail")

    try:
        validate_alternatives(
            (
                Alternative(
                    "bad-alt", 1, 0, {"unknown": 1}, 0, {}
                ),
            ),
            {parameter.parameter_id for parameter in parameters},
        )
    except ValueError:
        pass
    else:
        raise AssertionError("unknown parameter reference must fail")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("ok")
        return

    parameters, alternatives = fixture()
    baseline_membership = membership(parameters, alternatives)
    print(
        json.dumps(
            {
                "pareto_status": {
                    key: classify(values)
                    for key, values in baseline_membership.items()
                },
                "sensitivity": [
                    asdict(result)
                    for result in sensitivity(parameters, alternatives)
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
