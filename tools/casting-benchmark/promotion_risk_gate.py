#!/usr/bin/env python3
"""Finite-sample gate for false-feasible promotion risk.

This is an audit gate, not a claim that optimization decisions are safe. It uses
the one-sided exact Clopper-Pearson upper confidence bound for a binomial failure
rate. A gate decision applies only to the exchangeable validation population
represented by the observations; distribution shift invalidates that assumption.
"""
from __future__ import annotations

import math
from typing import Any

DEFAULT_RISK_TARGET = 0.05
DEFAULT_CONFIDENCE = 0.95


def _log_binomial_cdf_at_most(k: int, n: int, p: float) -> float:
    """Compute log P[X <= k] for X~Binomial(n,p), avoiding direct underflow."""
    if k < 0:
        return -math.inf
    if k >= n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return -math.inf

    log_p = math.log(p)
    log_q = math.log1p(-p)
    terms = [
        math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
        + j * log_p + (n - j) * log_q
        for j in range(k + 1)
    ]
    peak = max(terms)
    return peak + math.log(sum(math.exp(term - peak) for term in terms))


def clopper_pearson_upper_bound(
    failures: int,
    trials: int,
    confidence: float = DEFAULT_CONFIDENCE,
) -> float:
    """One-sided exact binomial upper confidence bound.

    For zero observed failures, the closed form is 1 - (1-confidence)**(1/n).
    For other counts, bisection solves P_p[X <= failures] = 1-confidence.
    """
    if isinstance(failures, bool) or not isinstance(failures, int):
        raise ValueError("failures must be an integer")
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1:
        raise ValueError("trials must be a positive integer")
    if failures < 0 or failures > trials:
        raise ValueError("failures must be between zero and trials")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValueError("confidence must be finite and strictly between zero and one")
    alpha = 1.0 - float(confidence)
    if failures == trials:
        return 1.0
    if failures == 0:
        return -math.expm1(math.log(alpha) / trials)

    log_alpha = math.log(alpha)
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        log_cdf = _log_binomial_cdf_at_most(failures, trials, mid)
        if log_cdf > log_alpha:
            lo = mid
        else:
            hi = mid
    return hi


def assess_promotion_risk(
    false_feasible_count: int,
    predicted_feasible_count: int,
    risk_target: float = DEFAULT_RISK_TARGET,
    confidence: float = DEFAULT_CONFIDENCE,
    independent_held_out: bool = False,
) -> dict[str, Any]:
    """Return a fail-closed decision from predeclared independent validation outcomes.

    Set independent_held_out only when outcomes came from a frozen policy evaluated
    on an independent, representative validation sample. An optimizer's adaptively
    selected promotions must not be used to certify its own risk.
    """
    if (
        isinstance(predicted_feasible_count, bool)
        or not isinstance(predicted_feasible_count, int)
        or predicted_feasible_count < 0
    ):
        raise ValueError("predicted_feasible_count must be a non-negative integer")
    if (
        isinstance(false_feasible_count, bool)
        or not isinstance(false_feasible_count, int)
        or false_feasible_count < 0
        or false_feasible_count > predicted_feasible_count
    ):
        raise ValueError("false_feasible_count must be between zero and predicted_feasible_count")
    if (
        isinstance(risk_target, bool)
        or not isinstance(risk_target, (int, float))
        or not math.isfinite(float(risk_target))
        or not 0.0 < float(risk_target) < 1.0
    ):
        raise ValueError("risk_target must be finite and strictly between zero and one")
    if predicted_feasible_count == 0:
        return {
            "validated_predicted_feasible_count": 0,
            "false_feasible_count": false_feasible_count,
            "observed_false_feasible_rate": None,
            "one_sided_confidence": float(confidence),
            "upper_confidence_bound": None,
            "risk_target": float(risk_target),
            "validation_source": "independent_held_out" if independent_held_out else "optimization_trace",
            "certified": False,
            "decision": "abstain",
            "reason": "no_validated_predicted_feasible_promotions",
        }
    upper = clopper_pearson_upper_bound(
        false_feasible_count, predicted_feasible_count, confidence
    )
    observed = false_feasible_count / predicted_feasible_count
    certified = bool(independent_held_out) and upper <= float(risk_target)
    reason = (
        "upper_bound_within_target" if certified else
        "validation_not_independent_or_exchangeable" if not independent_held_out else
        "observed_rate_exceeds_target" if observed > float(risk_target) else
        "insufficient_evidence_for_target"
    )
    return {
        "validated_predicted_feasible_count": predicted_feasible_count,
        "false_feasible_count": false_feasible_count,
        "observed_false_feasible_rate": observed,
        "one_sided_confidence": float(confidence),
        "upper_confidence_bound": upper,
        "risk_target": float(risk_target),
        "validation_source": "independent_held_out" if independent_held_out else "optimization_trace",
        "certified": certified,
        "decision": "allow" if certified else "abstain",
        "reason": reason,
    }


def main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--false-feasible", type=int, required=True)
    parser.add_argument("--validated-predicted-feasible", type=int, required=True)
    parser.add_argument("--risk-target", type=float, default=DEFAULT_RISK_TARGET)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument(
        "--independent-held-out", action="store_true",
        help="assert that outcomes are from a frozen policy on a predeclared independent, representative validation sample",
    )
    args = parser.parse_args()
    try:
        result = assess_promotion_risk(
            args.false_feasible, args.validated_predicted_feasible,
            args.risk_target, args.confidence, args.independent_held_out,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
