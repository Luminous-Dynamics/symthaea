#!/usr/bin/env python3
"""Exact finite-support CRC coordinate sweep for monotone finite-grid losses.

This implements the nonconvex finite-observation-space completion in Bai &
Ramdas (2026), Proposition 3.7. Arithmetic is exact via Fraction. The sweep
can handle multiple loss/risk tables sharing one ordered finite action grid.
"""
from __future__ import annotations

from fractions import Fraction
from itertools import product
import math
from typing import Iterable, Mapping, Sequence

from crc_finite_recertification import as_fraction, compositions

State = tuple[int, ...]
GridController = dict[State, int]
LossTables = tuple[tuple[tuple[Fraction, ...], ...], ...]


def _normalise_problem(
    n: int,
    alpha_levels: Sequence[int | float | str | Fraction],
    loss_tables: Sequence[Sequence[Sequence[int | float | str | Fraction]]],
) -> tuple[tuple[Fraction, ...], LossTables, int, int]:
    if not isinstance(n, int) or n < 1:
        raise ValueError("calibration size n must be a positive integer")
    if not alpha_levels or len(alpha_levels) != len(loss_tables):
        raise ValueError("one positive risk level is required for every loss table")
    alphas = tuple(as_fraction(x) for x in alpha_levels)
    if any(x <= 0 for x in alphas):
        raise ValueError("risk levels must be positive")
    if not loss_tables or not loss_tables[0]:
        raise ValueError("at least one observation type is required")

    support_size = len(loss_tables[0])
    action_count = len(loss_tables[0][0]) if loss_tables[0] else 0
    if action_count < 1:
        raise ValueError("the action grid must be nonempty")

    tables: list[tuple[tuple[Fraction, ...], ...]] = []
    for risk_index, raw_table in enumerate(loss_tables):
        if len(raw_table) != support_size:
            raise ValueError(f"risk {risk_index} has inconsistent support size")
        rows = []
        for type_index, raw_row in enumerate(raw_table):
            if len(raw_row) != action_count:
                raise ValueError(
                    f"risk {risk_index}, type {type_index} has inconsistent action grid"
                )
            row = tuple(as_fraction(x) for x in raw_row)
            if any(x < 0 for x in row):
                raise ValueError("losses must be nonnegative")
            if any(row[j + 1] > row[j] for j in range(action_count - 1)):
                raise ValueError("loss must be nonincreasing on the ordered action grid")
            rows.append(row)
        tables.append(tuple(rows))
    return alphas, tuple(tables), support_size, action_count


def _normalise_controller(
    controller: Mapping[State, int],
    n: int,
    support_size: int,
    action_count: int,
) -> GridController:
    expected = set(compositions(n, support_size))
    if set(controller) != expected:
        raise ValueError(
            f"controller state domain mismatch: missing={len(expected-set(controller))}, "
            f"extra={len(set(controller)-expected)}"
        )
    result: GridController = {}
    for state, action in controller.items():
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError("controller actions must be integer grid indices")
        if not 0 <= action < action_count:
            raise ValueError("controller action index is out of range")
        result[state] = action
    return result


def _augmented_totals(
    controller: Mapping[State, int],
    augmented: State,
    tables: LossTables,
) -> tuple[Fraction, ...]:
    totals = [Fraction(0) for _ in tables]
    for type_index, multiplicity in enumerate(augmented):
        if multiplicity == 0:
            continue
        training = list(augmented)
        training[type_index] -= 1
        action = controller[tuple(training)]
        for risk_index, table in enumerate(tables):
            totals[risk_index] += multiplicity * table[type_index][action]
    return tuple(totals)


def verify_grid_controller(
    controller: Mapping[State, int],
    n: int,
    alpha_levels: Sequence[int | float | str | Fraction],
    loss_tables: Sequence[Sequence[Sequence[int | float | str | Fraction]]],
) -> dict:
    """Verify all finite-exchangeable risks on every augmented multiset."""
    alphas, tables, support_size, action_count = _normalise_problem(
        n, alpha_levels, loss_tables
    )
    current = _normalise_controller(controller, n, support_size, action_count)
    minimum_slacks = [None for _ in tables]
    worst_states: list[State | None] = [None for _ in tables]
    checked = 0

    for augmented in compositions(n + 1, support_size):
        totals = _augmented_totals(current, augmented, tables)
        checked += 1
        for risk_index, total in enumerate(totals):
            slack = (n + 1) * alphas[risk_index] - total
            if minimum_slacks[risk_index] is None or slack < minimum_slacks[risk_index]:
                minimum_slacks[risk_index] = slack
                worst_states[risk_index] = augmented

    slacks = tuple(x for x in minimum_slacks)
    return {
        "valid": all(x is not None and x >= 0 for x in slacks),
        "minimum_slacks": slacks,
        "worst_augmented_multisets": tuple(worst_states),
        "augmented_states_checked": checked,
        "risks_checked": len(tables),
    }


def _candidate_is_valid_at_coordinate(
    controller: Mapping[State, int],
    training: State,
    candidate_action: int,
    n: int,
    alphas: tuple[Fraction, ...],
    tables: LossTables,
) -> tuple[bool, tuple | None]:
    old_action = controller[training]
    for heldout_type in range(len(training)):
        augmented = list(training)
        augmented[heldout_type] += 1
        augmented_state = tuple(augmented)
        multiplicity = training[heldout_type] + 1
        totals = _augmented_totals(controller, augmented_state, tables)

        for risk_index, table in enumerate(tables):
            slack = (n + 1) * alphas[risk_index] - totals[risk_index]
            increase = (
                table[heldout_type][candidate_action]
                - table[heldout_type][old_action]
            )
            if multiplicity * increase > slack:
                return False, (
                    augmented_state,
                    risk_index,
                    heldout_type,
                    multiplicity * increase,
                    slack,
                )
    return True, None


def coordinate_sweep_admissible(
    controller: Mapping[State, int],
    n: int,
    alpha_levels: Sequence[int | float | str | Fraction],
    loss_tables: Sequence[Sequence[Sequence[int | float | str | Fraction]]],
) -> dict:
    """Perform one deterministic pass to a pointwise-admissible grid controller.

    Smaller action indices are less conservative and have weakly larger loss.
    The order is reverse lexicographic in the count vector, fixed and
    deterministic. Once a coordinate has no feasible lower action, later
    coordinate decreases can only consume more slack in its augmented states;
    it therefore cannot become reducible later.
    """
    alphas, tables, support_size, action_count = _normalise_problem(
        n, alpha_levels, loss_tables
    )
    current = _normalise_controller(controller, n, support_size, action_count)
    initial = verify_grid_controller(current, n, alphas, tables)
    if not initial["valid"]:
        raise ValueError(f"baseline controller is invalid: {initial}")

    updates = []
    # The reverse order is explicit so reproductions have deterministic output.
    for training in reversed(list(compositions(n, support_size))):
        old_action = current[training]
        selected = old_action
        for candidate in range(old_action + 1):
            feasible, _ = _candidate_is_valid_at_coordinate(
                current, training, candidate, n, alphas, tables
            )
            if feasible:
                selected = candidate
                break
        current[training] = selected
        if selected < old_action:
            updates.append(
                {"training_multiset": training, "from": old_action, "to": selected}
            )

    validity = verify_grid_controller(current, n, alphas, tables)
    admissibility = verify_grid_admissibility(current, n, alphas, tables)
    if not validity["valid"] or not admissibility["admissible"]:
        raise ArithmeticError(
            f"coordinate sweep failed its independent postconditions: "
            f"validity={validity}, admissibility={admissibility}"
        )
    return {
        "controller": current,
        "updates": updates,
        "updated_coordinates": len(updates),
        "validity": validity,
        "admissibility": admissibility,
        "coordinate_order": "reverse_lexicographic_count_vector",
    }


def verify_grid_admissibility(
    controller: Mapping[State, int],
    n: int,
    alpha_levels: Sequence[int | float | str | Fraction],
    loss_tables: Sequence[Sequence[Sequence[int | float | str | Fraction]]],
) -> dict:
    """Check that no single controller coordinate can be lowered validly.

    For monotone losses on a finite observation support, this is the
    coordinatewise criterion for pointwise admissibility of the full controller.
    """
    alphas, tables, support_size, action_count = _normalise_problem(
        n, alpha_levels, loss_tables
    )
    current = _normalise_controller(controller, n, support_size, action_count)
    validity = verify_grid_controller(current, n, alphas, tables)
    if not validity["valid"]:
        return {
            "admissible": False,
            "baseline_valid": False,
            "witness": validity,
        }

    for training in compositions(n, support_size):
        old_action = current[training]
        for candidate in range(old_action):
            feasible, witness = _candidate_is_valid_at_coordinate(
                current, training, candidate, n, alphas, tables
            )
            if feasible:
                return {
                    "admissible": False,
                    "baseline_valid": True,
                    "witness": {
                        "training_multiset": training,
                        "lower_action": candidate,
                        "augmented_constraint": witness,
                    },
                }
    return {
        "admissible": True,
        "baseline_valid": True,
        "witness": None,
        "coordinates_checked": len(current),
    }


if __name__ == "__main__":
    # Module is intentionally library-first; tests provide reproducible examples.
    raise SystemExit("Run test_crc_discrete_admissibility.py for exact fixtures.")
