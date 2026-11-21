"""Reference expected-information-gain calculations for CETB theory design.

Pure standard-library implementation. All outcome categories, including missing,
failed and ambiguous measurements, must be explicit in each theory likelihood.
This is a design utility, not an empirical theory test.
"""
from __future__ import annotations
import math
from typing import Mapping, Sequence


class DesignValidationError(ValueError):
    """Raised when theory priors or outcome likelihoods are invalid."""


def _validate_distribution(values: Mapping[str, float], name: str, tolerance: float) -> None:
    if not values:
        raise DesignValidationError(f"{name} must not be empty")
    if any(not isinstance(key, str) or not key.strip() for key in values):
        raise DesignValidationError(f"{name} has an empty outcome/theory key")
    probabilities = []
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DesignValidationError(f"{name}[{key!r}] must be numeric")
        probability = float(value)
        if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
            raise DesignValidationError(f"{name}[{key!r}] must be finite and in [0, 1]")
        probabilities.append(probability)
    total = math.fsum(probabilities)
    if abs(total - 1.0) > tolerance:
        raise DesignValidationError(f"{name} probability mass sums to {total}, not 1")


def theory_entropy_bits(prior: Mapping[str, float], tolerance: float = 1e-10) -> float:
    """Shannon entropy of the theory prior in bits."""
    _validate_distribution(prior, "prior", tolerance)
    return -math.fsum(
        float(probability) * math.log2(float(probability))
        for probability in prior.values()
        if probability > 0.0
    )


def expected_information_gain_bits(
    prior: Mapping[str, float],
    likelihoods: Mapping[str, Mapping[str, float]],
    tolerance: float = 1e-10,
) -> float:
    """Compute I(T;Y|d) in bits for categorical theory predictions.

    Each theory must provide a normalized likelihood over the same explicit
    outcome space, including missing, ambiguous and failed outcomes.
    """
    _validate_distribution(prior, "prior", tolerance)
    if set(likelihoods) != set(prior):
        raise DesignValidationError("likelihood theory keys must match prior theory keys")
    first = next(iter(likelihoods.values()), None)
    if first is None:
        raise DesignValidationError("likelihoods must contain every theory")
    outcome_keys = set(first)
    if not outcome_keys:
        raise DesignValidationError("the outcome space must not be empty")
    for theory, distribution in likelihoods.items():
        if set(distribution) != outcome_keys:
            raise DesignValidationError(
                f"likelihood outcomes for {theory!r} do not match the common outcome space"
            )
        _validate_distribution(distribution, f"likelihoods[{theory!r}]", tolerance)

    marginal = {
        outcome: math.fsum(
            float(prior[theory]) * float(likelihoods[theory][outcome])
            for theory in prior
        )
        for outcome in outcome_keys
    }
    information = math.fsum(
        float(prior[theory])
        * float(likelihoods[theory][outcome])
        * math.log2(float(likelihoods[theory][outcome]) / marginal[outcome])
        for theory in prior
        for outcome in outcome_keys
        if prior[theory] > 0.0
        and likelihoods[theory][outcome] > 0.0
        and marginal[outcome] > 0.0
    )
    entropy = theory_entropy_bits(prior, tolerance)
    if information < -tolerance or information > entropy + tolerance:
        raise ArithmeticError(f"EIG {information} is outside [0, prior entropy {entropy}]")
    return min(entropy, max(0.0, information))


def weighted_lower_quantile(values_and_weights: Sequence[tuple[float, float]], q: float) -> float:
    """Left-continuous weighted lower quantile without interpolation."""
    if not values_and_weights:
        raise DesignValidationError("at least one nuisance scenario is required")
    if isinstance(q, bool) or not isinstance(q, (int, float)) or not math.isfinite(q) or not 0.0 <= q <= 1.0:
        raise DesignValidationError("q must be finite and in [0, 1]")
    total_weight = math.fsum(weight for _, weight in values_and_weights)
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise DesignValidationError("scenario weights must have positive finite mass")
    normalized = []
    for value, weight in values_and_weights:
        if not math.isfinite(value):
            raise DesignValidationError("scenario EIG must be finite")
        if not math.isfinite(weight) or weight < 0.0:
            raise DesignValidationError("scenario weights must be finite and non-negative")
        normalized.append((value, weight / total_weight))
    normalized.sort(key=lambda item: item[0])
    if q == 0.0:
        return normalized[0][0]
    cumulative = 0.0
    for value, weight in normalized:
        cumulative = math.fsum((cumulative, weight))
        if cumulative + 1e-15 >= q:
            return value
    return normalized[-1][0]


def evaluate_design_scenarios(
    prior: Mapping[str, float],
    scenario_likelihoods: Sequence[tuple[float, Mapping[str, Mapping[str, float]]]],
    cost: float,
    lower_quantile: float = 0.10,
    tolerance: float = 1e-10,
) -> dict[str, float | int]:
    """Summarize EIG across predeclared nuisance scenarios and a declared cost."""
    if not math.isfinite(cost) or cost <= 0.0:
        raise DesignValidationError("design cost must be finite and positive")
    if not scenario_likelihoods:
        raise DesignValidationError("at least one nuisance scenario is required")
    weighted_eigs = []
    for index, (weight, likelihoods) in enumerate(scenario_likelihoods):
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise DesignValidationError(f"scenario {index} weight must be numeric")
        if not math.isfinite(weight) or weight < 0.0:
            raise DesignValidationError(f"scenario {index} weight must be finite and non-negative")
        eig = expected_information_gain_bits(prior, likelihoods, tolerance)
        weighted_eigs.append((eig, float(weight)))
    total_weight = math.fsum(weight for _, weight in weighted_eigs)
    if total_weight <= 0.0:
        raise DesignValidationError("scenario weights must have positive mass")
    expected = math.fsum(value * weight for value, weight in weighted_eigs) / total_weight
    robust = weighted_lower_quantile(weighted_eigs, lower_quantile)
    return {
        "scenario_count": len(weighted_eigs),
        "expected_eig_bits": expected,
        "lower_quantile_eig_bits": robust,
        "expected_eig_bits_per_cost": expected / cost,
        "lower_quantile_eig_bits_per_cost": robust / cost,
        "cost": cost,
        "lower_quantile": lower_quantile,
    }
