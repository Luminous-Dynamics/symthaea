#!/usr/bin/env python3
"""Exact finite-support conformal-risk-control recertification prototype.

The state is a multiset of observations represented by count vectors. All
arithmetic is rational. This is a finite-exchangeability research oracle, not
a continuous-observation or production conformal implementation.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
from itertools import product
import json
import math
from math import comb
from typing import Iterable, Mapping, Sequence

State = tuple[int, ...]
Controller = dict[State, Fraction]


def as_fraction(value: int | float | str | Fraction) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, bool):
        raise TypeError("bool is not a rational input")
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("rational inputs must be finite")
        return Fraction(str(value))
    if isinstance(value, str):
        return Fraction(value)
    raise TypeError(f"unsupported rational input: {type(value).__name__}")


def compositions(total: int, support_size: int) -> Iterable[State]:
    """Yield all nonnegative count vectors of the given total."""
    if not isinstance(total, int) or total < 0:
        raise ValueError("total must be a nonnegative integer")
    if not isinstance(support_size, int) or support_size < 1:
        raise ValueError("support_size must be a positive integer")

    def visit(remaining: int, slots: int, prefix: tuple[int, ...]):
        if slots == 1:
            yield prefix + (remaining,)
            return
        for count in range(remaining + 1):
            yield from visit(remaining - count, slots - 1, prefix + (count,))

    yield from visit(total, support_size, ())


def composition_count(total: int, support_size: int) -> int:
    if total < 0 or support_size < 1:
        raise ValueError("invalid state-space dimensions")
    return comb(total + support_size - 1, support_size - 1)


def _normalise_problem(
    n: int,
    alpha: int | float | str | Fraction,
    base_loss: Sequence[int | float | str | Fraction],
    slope: Sequence[int | float | str | Fraction],
) -> tuple[Fraction, tuple[Fraction, ...], tuple[Fraction, ...]]:
    if not isinstance(n, int) or n < 1:
        raise ValueError("calibration size n must be a positive integer")
    if not base_loss or len(base_loss) != len(slope):
        raise ValueError("base_loss and slope need the same nonzero support size")
    a = as_fraction(alpha)
    b = tuple(as_fraction(x) for x in base_loss)
    d = tuple(as_fraction(x) for x in slope)
    if not 0 < a < 1:
        raise ValueError("alpha must lie in (0,1)")
    if any(x < 0 for x in b) or any(x < 0 for x in d):
        raise ValueError("loss intercepts and slopes must be nonnegative")
    return a, b, d


def _normalise_controller(
    controller: Mapping[State, int | float | str | Fraction],
    n: int,
    support_size: int,
) -> Controller:
    expected = set(compositions(n, support_size))
    if set(controller) != expected:
        missing = expected - set(controller)
        extra = set(controller) - expected
        raise ValueError(
            f"controller state domain mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    result = {state: as_fraction(value) for state, value in controller.items()}
    if any(not 0 <= action <= 1 for action in result.values()):
        raise ValueError("actions must lie in [0,1]")
    return result


def loss(
    observation_type: int,
    action: Fraction,
    base_loss: Sequence[Fraction],
    slope: Sequence[Fraction],
) -> Fraction:
    """Affine, decreasing, convex loss: b_j + d_j * (1 - action)."""
    if not 0 <= observation_type < len(base_loss):
        raise ValueError("observation type is out of range")
    return base_loss[observation_type] + slope[observation_type] * (1 - action)


def add_observation(state: State, observation_type: int) -> State:
    if not 0 <= observation_type < len(state):
        raise ValueError("observation type is out of range")
    result = list(state)
    result[observation_type] += 1
    return tuple(result)


def augmented_slack(
    controller: Mapping[State, Fraction],
    augmented: State,
    alpha: Fraction,
    base_loss: Sequence[Fraction],
    slope: Sequence[Fraction],
) -> Fraction:
    """Unused total risk budget on one hypothetical augmented multiset."""
    n_plus_one = sum(augmented)
    total_loss = Fraction(0)
    for observation_type, multiplicity in enumerate(augmented):
        if multiplicity:
            training = list(augmented)
            training[observation_type] -= 1
            total_loss += multiplicity * loss(
                observation_type,
                controller[tuple(training)],
                base_loss,
                slope,
            )
    return n_plus_one * alpha - total_loss


def controller_digest(controller: Mapping[State, Fraction]) -> str:
    rows = [
        {
            "state": list(state),
            "action": [value.numerator, value.denominator],
        }
        for state, value in sorted(controller.items())
    ]
    raw = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(raw).hexdigest()


def verify_controller(
    controller: Mapping[State, int | float | str | Fraction],
    n: int,
    alpha: int | float | str | Fraction,
    base_loss: Sequence[int | float | str | Fraction],
    slope: Sequence[int | float | str | Fraction],
) -> dict:
    """Exhaustively verify finite-exchangeable risk over every augmented multiset."""
    a, b, d = _normalise_problem(n, alpha, base_loss, slope)
    c = _normalise_controller(controller, n, len(b))
    minimum_slack: Fraction | None = None
    witness: State | None = None
    checked = 0
    for augmented in compositions(n + 1, len(b)):
        slack = augmented_slack(c, augmented, a, b, d)
        checked += 1
        if minimum_slack is None or slack < minimum_slack:
            minimum_slack = slack
            witness = augmented
    assert minimum_slack is not None
    return {
        "valid": minimum_slack >= 0,
        "minimum_slack": minimum_slack,
        "worst_augmented_multiset": witness,
        "augmented_states_checked": checked,
        "controller_sha256": controller_digest(c),
    }


def recertify_once(
    controller: Mapping[State, int | float | str | Fraction],
    n: int,
    alpha: int | float | str | Fraction,
    base_loss: Sequence[int | float | str | Fraction],
    slope: Sequence[int | float | str | Fraction],
) -> Controller:
    """Redistribute each augmented sample's exact slack over held-out positions.

    For each training multiset D, choose the smallest action lambda in [0,1]
    satisfying, for every support point z,

      loss(z, lambda) <= loss(z, A(D)) + Delta_A(D + z) / (n + 1).

    With affine loss b_z + d_z(1-lambda), this has the exact closed form below.
    """
    a, b, d = _normalise_problem(n, alpha, base_loss, slope)
    current = _normalise_controller(controller, n, len(b))
    baseline_check = verify_controller(current, n, a, b, d)
    if not baseline_check["valid"]:
        raise ValueError(
            "baseline controller is not valid under finite exchangeability: "
            f"witness={baseline_check['worst_augmented_multiset']}, "
            f"slack={baseline_check['minimum_slack']}"
        )

    result: Controller = {}
    for training in compositions(n, len(b)):
        lower_bound = Fraction(0)
        for heldout_type in range(len(b)):
            augmented = add_observation(training, heldout_type)
            slack = augmented_slack(current, augmented, a, b, d)
            if slack < 0:
                raise ArithmeticError("negative slack encountered after baseline verification")
            if d[heldout_type] > 0:
                bound = current[training] - slack / ((n + 1) * d[heldout_type])
                lower_bound = max(lower_bound, bound)
        action = max(Fraction(0), lower_bound)
        if action > current[training]:
            raise ArithmeticError("recertification became more conservative")
        result[training] = action
    return result


def verify_recertification_step(
    old_controller: Mapping[State, int | float | str | Fraction],
    new_controller: Mapping[State, int | float | str | Fraction],
    n: int,
    alpha: int | float | str | Fraction,
    base_loss: Sequence[int | float | str | Fraction],
    slope: Sequence[int | float | str | Fraction],
) -> dict:
    """Independently check held-out inequalities and full augmented risk."""
    a, b, d = _normalise_problem(n, alpha, base_loss, slope)
    old = _normalise_controller(old_controller, n, len(b))
    new = _normalise_controller(new_controller, n, len(b))
    failures: list[tuple] = []
    minimum_heldout_margin: Fraction | None = None
    inequalities_checked = 0

    for training in compositions(n, len(b)):
        if new[training] > old[training]:
            failures.append(("pointwise_dominance", training))
        for heldout_type in range(len(b)):
            augmented = add_observation(training, heldout_type)
            slack = augmented_slack(old, augmented, a, b, d)
            lhs = loss(heldout_type, new[training], b, d)
            rhs = loss(heldout_type, old[training], b, d) + slack / (n + 1)
            margin = rhs - lhs
            inequalities_checked += 1
            if minimum_heldout_margin is None or margin < minimum_heldout_margin:
                minimum_heldout_margin = margin
            if margin < 0:
                failures.append(("heldout_budget", training, heldout_type, margin))

    risk = verify_controller(new, n, a, b, d)
    if not risk["valid"]:
        failures.append(
            ("finite_exchangeability_risk", risk["worst_augmented_multiset"],
             risk["minimum_slack"])
        )
    return {
        "valid": not failures,
        "failures": failures,
        "heldout_inequalities_checked": inequalities_checked,
        "minimum_heldout_margin": minimum_heldout_margin,
        "risk": risk,
        "pointwise_no_more_conservative": not any(
            item[0] == "pointwise_dominance" for item in failures
        ),
    }


def iterate_recertification(
    controller: Mapping[State, int | float | str | Fraction],
    n: int,
    alpha: int | float | str | Fraction,
    base_loss: Sequence[int | float | str | Fraction],
    slope: Sequence[int | float | str | Fraction],
    *,
    max_iterations: int = 100,
) -> dict:
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    a, b, d = _normalise_problem(n, alpha, base_loss, slope)
    current = _normalise_controller(controller, n, len(b))
    history = []

    for iteration in range(max_iterations):
        old_validity = verify_controller(current, n, a, b, d)
        if not old_validity["valid"]:
            raise ValueError("iteration input lost its finite-exchangeability guarantee")
        candidate = recertify_once(current, n, a, b, d)
        step_check = verify_recertification_step(current, candidate, n, a, b, d)
        if not step_check["valid"]:
            raise ArithmeticError(f"recertification verification failed: {step_check}")
        step_size = max(abs(current[state] - candidate[state]) for state in current)
        history.append({
            "iteration": iteration + 1,
            "sup_action_change": step_size,
            "minimum_slack": step_check["risk"]["minimum_slack"],
            "controller_sha256": step_check["risk"]["controller_sha256"],
        })
        if candidate == current:
            return {
                "converged": True,
                "controller": candidate,
                "history": history,
                "validity": step_check["risk"],
            }
        current = candidate

    return {
        "converged": False,
        "controller": current,
        "history": history,
        "validity": verify_controller(current, n, a, b, d),
    }
