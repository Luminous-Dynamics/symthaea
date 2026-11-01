#!/usr/bin/env python3
"""Finite reference oracle for history-dependent projective intersection betting.

The model is a one-sided shared-control Gaussian experiment. At global update t,
the active vector Z_{I_t,t} has conditional mean mu_{I_t,t} and covariance

    Sigma_m = 0.5 * I_m + 0.5 * 1_m 1_m^T.

For a finite intersection S, its active coordinates are A_{S,t}=S intersect I_t.
A predictable scalar lambda_{S,t} >= 0 defines the factor

    X_{S,t} = exp(lambda * sum_{i in A} Z_{i,t}
                  - 0.5 * lambda^2 * 1^T Sigma_A 1).

Under H_S: mu_i <= 0 for i in S, its conditional expectation is
exp(lambda * sum_i mu_i) <= 1. The lambda policy below uses only the past
observations restricted to A, so dormant coordinates have no effect and the
intersection family is projective.

This is a model-specific test-martingale prototype, not an arbitrary-
dependence merger. It checks algebra, predictable-policy replay, projectivity,
and local closure. It does not authenticate real timestamps or prove that
real data follow the declared Gaussian conditional model.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Iterable, Mapping, Sequence


def validate_ids(values: Iterable[int]) -> tuple[int, ...]:
    try:
        ids = tuple(values)
    except TypeError as error:
        raise TypeError("hypothesis ids must be iterable") from error
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 1 for i in ids):
        raise ValueError("hypothesis ids must be positive integers")
    if len(set(ids)) != len(ids):
        raise ValueError("hypothesis ids must be unique")
    return tuple(sorted(ids))


def nonempty_subsets(ids: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    values = validate_ids(ids)
    result = []
    for mask in range(1, 1 << len(values)):
        result.append(tuple(values[j] for j in range(len(values)) if (mask >> j) & 1))
    return tuple(result)


def shared_control_covariance(size: int) -> tuple[tuple[float, ...], ...]:
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("covariance size must be a nonnegative integer")
    return tuple(
        tuple(1.0 if i == j else 0.5 for j in range(size))
        for i in range(size)
    )


def variance_of_shared_sum(size: int) -> float:
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError("sum size must be a nonnegative integer")
    # 1^T Sigma_m 1 = m(m+1)/2.
    return size * (size + 1) / 2.0


def conditional_log_mgf_bound(
    null_means: Sequence[float],
    predictable_lambda: float,
) -> float:
    """Log conditional mean of the factor under the Gaussian model.

    A nonpositive return value proves E[X | F_{t-1}] <= 1 under the
    declared intersection null. The function receives no current observation.
    """
    if not math.isfinite(predictable_lambda) or predictable_lambda < 0:
        raise ValueError("predictable lambda must be finite and nonnegative")
    means = tuple(float(x) for x in null_means)
    if any(not math.isfinite(x) for x in means):
        raise ValueError("null means must be finite")
    if any(x > 0 for x in means):
        raise ValueError("the supplied mean vector is outside the intersection null")
    return predictable_lambda * math.fsum(means)


def adaptive_lambda(
    active_intersection: Iterable[int],
    past_observations: Sequence[Mapping[int, float]],
    *,
    base_lambda: float = 0.05,
    gain: float = 0.02,
    maximum_lambda: float = 0.5,
) -> float:
    """Predictable nonnegative scale using only pre-update history.

    Uses the positive part of the historical mean over observations belonging
    to currently active coordinates in the fixed intersection. Future dormant
    coordinates are ignored. The same active subset and history therefore
    produce exactly the same lambda under every future horizon extension.
    """
    active = validate_ids(active_intersection)
    for name, value in (
        ("base_lambda", base_lambda),
        ("gain", gain),
        ("maximum_lambda", maximum_lambda),
    ):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if maximum_lambda < base_lambda:
        raise ValueError("maximum_lambda must be at least base_lambda")
    history_values = []
    active_set = set(active)
    for row in past_observations:
        for i in active:
            if i in row:
                value = float(row[i])
                if not math.isfinite(value):
                    raise ValueError("past observations must be finite")
                history_values.append(value)
    historical_mean = (
        math.fsum(history_values) / len(history_values)
        if history_values else 0.0
    )
    scale = base_lambda + gain * max(0.0, historical_mean)
    return min(maximum_lambda, max(0.0, scale))


def _prefix_digest(
    active_intersection: Sequence[int],
    past_observations: Sequence[Mapping[int, float]],
    lambda_value: float,
) -> str:
    payload = {
        "active_intersection": list(active_intersection),
        "past": [
            [[i, float(row[i])] for i in active_intersection if i in row]
            for row in past_observations
        ],
        "lambda": float(lambda_value),
    }
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return sha256(raw).hexdigest()


def _normalize_stream(
    active_by_time: Sequence[Iterable[int]],
    observations_by_time: Sequence[Mapping[int, float]],
) -> tuple[list[tuple[int, ...]], list[dict[int, float]], tuple[int, ...]]:
    if not active_by_time:
        raise ValueError("at least one global update is required")
    if len(active_by_time) != len(observations_by_time):
        raise ValueError("active and observation paths must have equal lengths")
    active = [validate_ids(row) for row in active_by_time]
    for t in range(1, len(active)):
        if not set(active[t - 1]).issubset(active[t]):
            raise ValueError("active sets must be nested")
    normalized_rows: list[dict[int, float]] = []
    universe = set()
    for t, (active_ids, row) in enumerate(zip(active, observations_by_time)):
        if not isinstance(row, Mapping):
            raise TypeError("each observation row must be a mapping")
        if set(row) != set(active_ids):
            raise ValueError(f"observation keys at time {t} must equal the active set")
        values = {}
        for i in active_ids:
            value = float(row[i])
            if not math.isfinite(value):
                raise ValueError("observations must be finite")
            values[i] = value
        normalized_rows.append(values)
        universe.update(active_ids)
    return active, normalized_rows, tuple(sorted(universe))


def build_history_dependent_projective_paths(
    active_by_time: Sequence[Iterable[int]],
    observations_by_time: Sequence[Mapping[int, float]],
    *,
    base_lambda: float = 0.05,
    gain: float = 0.02,
    maximum_lambda: float = 0.5,
) -> dict:
    """Build all finite-intersection log-e-process paths for a small family."""
    active, observations, universe = _normalize_stream(
        active_by_time, observations_by_time
    )
    intersections = ((),) + nonempty_subsets(universe)
    paths: dict[tuple[int, ...], list[float]] = {}
    factor_receipts: dict[tuple[int, ...], list[dict]] = {}
    for intersection in intersections:
        log_wealth = 0.0
        path = []
        receipts = []
        for t, active_ids in enumerate(active):
            active_intersection = tuple(
                i for i in intersection if i in set(active_ids)
            )
            past = observations[:t]
            if not active_intersection:
                lambda_value = 0.0
                log_factor = 0.0
                prefix_sha256 = _prefix_digest(
                    active_intersection, past, lambda_value
                )
            else:
                lambda_value = adaptive_lambda(
                    active_intersection,
                    past,
                    base_lambda=base_lambda,
                    gain=gain,
                    maximum_lambda=maximum_lambda,
                )
                sum_observation = math.fsum(
                    observations[t][i] for i in active_intersection
                )
                variance = variance_of_shared_sum(len(active_intersection))
                log_factor = (
                    lambda_value * sum_observation
                    - 0.5 * lambda_value * lambda_value * variance
                )
                prefix_sha256 = _prefix_digest(
                    active_intersection, past, lambda_value
                )
            log_wealth += log_factor
            path.append(log_wealth)
            receipts.append({
                "time": t,
                "active_intersection": active_intersection,
                "lambda": lambda_value,
                "log_factor": log_factor,
                "pre_update_prefix_sha256": prefix_sha256,
            })
        paths[intersection] = path
        factor_receipts[intersection] = receipts

    projectivity_checks = 0
    projectivity_failures = []
    for intersection in intersections:
        for t, active_ids in enumerate(active):
            current = tuple(i for i in intersection if i in set(active_ids))
            for u in range(t + 1):
                projectivity_checks += 1
                if not math.isclose(
                    paths[intersection][u],
                    paths[current][u],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    projectivity_failures.append(
                        ("projectivity_failure", intersection, current, u, t)
                    )

    return {
        "schema": "history-dependent-projective-shared-control-v1",
        "active_by_time": active,
        "observations_by_time": observations,
        "universe": universe,
        "log_e_process_paths": paths,
        "factor_receipts": factor_receipts,
        "projectivity_checks": projectivity_checks,
        "projectivity_failures": projectivity_failures,
        "projective": not projectivity_failures,
        "policy": {
            "base_lambda": base_lambda,
            "gain": gain,
            "maximum_lambda": maximum_lambda,
            "uses_only_pre_update_history": True,
        },
        "covariance": "Sigma_m = 0.5 I_m + 0.5 1_m 1_m^T",
        "conditional_validity_contract": (
            "under the declared conditional Gaussian model, mean_i<=0 for all "
            "i in the intersection, and predictable lambda>=0"
        ),
        "scope": (
            "finite exact-path reference prototype; stochastic validity depends "
            "on the declared shared-control Gaussian conditional model"
        ),
    }


def build_predictable_arrivals_from_history(
    observation_stream: Sequence[Mapping[int, float]],
    *,
    first_active: int = 1,
    second_active_threshold: float = 0.0,
    third_active_threshold: float = 0.5,
) -> tuple[tuple[int, ...], ...]:
    """Example nested activation policy using only prior-round observations.

    Coordinate 1 is active initially. Coordinate 2 is added at t=1 if the
    previous round's Z_1 exceeded its threshold. Coordinate 3 is added at t=2
    if the previous round's sum over currently active coordinates exceeded
    its threshold. The output at t is determined only by observations before t.
    """
    if isinstance(first_active, bool) or not isinstance(first_active, int) or first_active < 1:
        raise ValueError("first_active must be a positive integer")
    if not math.isfinite(second_active_threshold) or not math.isfinite(third_active_threshold):
        raise ValueError("arrival thresholds must be finite")
    if len(observation_stream) < 1:
        raise ValueError("observation stream must be nonempty")
    active = [(first_active,)]
    for t in range(1, len(observation_stream)):
        previous = observation_stream[t - 1]
        current = set(active[-1])
        if t == 1 and first_active in previous:
            if float(previous[first_active]) > second_active_threshold:
                current.add(first_active + 1)
        elif t >= 2 and sum(float(previous[i]) for i in active[-1] if i in previous) > third_active_threshold:
            current.add(max(current) + 1)
        active.append(tuple(sorted(current)))
    return tuple(active)


def verify_history_dependent_projective_receipt(result: dict) -> dict:
    """Independently replay predictable factors and projective prefixes."""
    active, observations, universe = _normalize_stream(
        result.get("active_by_time", ()),
        result.get("observations_by_time", ()),
    )
    policy = result.get("policy", {})
    base_lambda = float(policy.get("base_lambda"))
    gain = float(policy.get("gain"))
    maximum_lambda = float(policy.get("maximum_lambda"))
    intersections = ((),) + nonempty_subsets(universe)
    failures = []
    paths = result.get("log_e_process_paths", {})
    receipts = result.get("factor_receipts", {})

    for intersection in intersections:
        if intersection not in paths or intersection not in receipts:
            failures.append(("missing_intersection_receipt", intersection))
            continue
        if len(paths[intersection]) != len(active) or len(receipts[intersection]) != len(active):
            failures.append(("intersection_receipt_horizon_mismatch", intersection))
            continue
        accumulated = 0.0
        for t, active_ids in enumerate(active):
            current = tuple(i for i in intersection if i in set(active_ids))
            past = observations[:t]
            receipt = receipts[intersection][t]
            if tuple(receipt.get("active_intersection", ())) != current:
                failures.append(("active_intersection_receipt_mismatch", intersection, t))
            expected_lambda = (
                0.0 if not current else adaptive_lambda(
                    current, past, base_lambda=base_lambda,
                    gain=gain, maximum_lambda=maximum_lambda,
                )
            )
            if not math.isclose(
                float(receipt.get("lambda", math.nan)),
                expected_lambda,
                rel_tol=0.0, abs_tol=1e-12,
            ):
                failures.append(("predictable_lambda_replay_failure", intersection, t))
            expected_digest = _prefix_digest(current, past, expected_lambda)
            if receipt.get("pre_update_prefix_sha256") != expected_digest:
                failures.append(("pre_update_prefix_digest_failure", intersection, t))
            if current:
                sum_z = math.fsum(observations[t][i] for i in current)
                expected_log_factor = (
                    expected_lambda * sum_z
                    - 0.5 * expected_lambda**2 * variance_of_shared_sum(len(current))
                )
            else:
                expected_log_factor = 0.0
            if not math.isclose(
                float(receipt.get("log_factor", math.nan)),
                expected_log_factor,
                rel_tol=0.0, abs_tol=1e-12,
            ):
                failures.append(("conditional_factor_replay_failure", intersection, t))
            accumulated += expected_log_factor
            if not math.isclose(
                float(paths[intersection][t]),
                accumulated,
                rel_tol=0.0, abs_tol=1e-12,
            ):
                failures.append(("log_wealth_replay_failure", intersection, t))

    projectivity_checks = 0
    for intersection in intersections:
        if intersection not in paths:
            continue
        for t, active_ids in enumerate(active):
            current = tuple(i for i in intersection if i in set(active_ids))
            if current not in paths:
                failures.append(("missing_projective_restriction", intersection, current, t))
                continue
            for u in range(t + 1):
                projectivity_checks += 1
                if not math.isclose(
                    float(paths[intersection][u]),
                    float(paths[current][u]),
                    rel_tol=0.0, abs_tol=1e-12,
                ):
                    failures.append(("historical_projectivity_failure", intersection, current, u, t))

    return {
        "valid": not failures,
        "failures": failures,
        "intersections_checked": len(intersections),
        "predictable_factors_replayed": sum(len(receipts.get(s, ())) for s in intersections),
        "historical_projectivity_checks": projectivity_checks,
        "scope": "independent finite path replay; conditional Gaussian model validity remains external",
    }


def verify_local_projective_closure(
    result: dict,
    rejections_by_time: Sequence[Iterable[int]],
    alpha: float,
) -> dict:
    """Check every finite active-intersection constraint in log space."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    active = result["active_by_time"]
    paths = result["log_e_process_paths"]
    if len(rejections_by_time) != len(active):
        raise ValueError("rejection path horizon mismatch")
    failures = []
    checked = 0
    rows = []
    log_alpha = math.log(alpha)
    for t, (active_ids, rejection_row) in enumerate(zip(active, rejections_by_time)):
        rejected = validate_ids(rejection_row)
        if not set(rejected).issubset(active_ids):
            raise ValueError("rejections must be a subset of active hypotheses")
        r = len(rejected)
        constraints = []
        for subset in nonempty_subsets(active_ids):
            fdp = len(set(rejected).intersection(subset)) / r if r else 0.0
            log_e = paths[subset][t]
            certified = fdp == 0.0 or math.log(fdp) <= log_alpha + log_e + 1e-12
            checked += 1
            constraints.append({
                "intersection": subset,
                "fdp": fdp,
                "log_e_value": log_e,
                "certified": certified,
            })
            if not certified:
                failures.append(("local_closure_violation", t, subset, fdp, log_e))
        rows.append({
            "time": t,
            "active": active_ids,
            "rejections": rejected,
            "local_constraints_checked": len(constraints),
            "local_constraints_certified": all(x["certified"] for x in constraints),
            "constraints": constraints,
        })
    return {
        "valid": not failures,
        "failures": failures,
        "constraints_checked": checked,
        "path": rows,
        "scope": "pathwise finite closure replay; Gaussian stochastic validity is external",
    }


if __name__ == "__main__":
    raise SystemExit("Run test_history_dependent_projective_betting.py.")
