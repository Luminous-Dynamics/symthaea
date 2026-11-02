"""Prefix-capacity batch selector for exact e-BH credit deficits.

This maximizes the number of candidates funded using their independent
baseline minimum-credit requirements at one frozen score vector. It does
not solve the joint allocation problem with cross-candidate threshold synergy.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import heapq
from typing import Sequence


Rational = Fraction | int | str


def _fraction(value: Rational) -> Fraction:
    if isinstance(value, bool) or isinstance(value, float):
        raise TypeError("exact rational input required; floats are rejected")
    return value if isinstance(value, Fraction) else Fraction(value)


@dataclass(frozen=True)
class AssignmentFragment:
    source_id: int
    token_id: tuple[int, int]
    destination_id: int
    assignment_ordinal: int
    amount: Fraction


def select_max_cardinality(
    deficits: Sequence[Rational | None],
    unassigned_by_source: Sequence[Rational],
) -> tuple[int, ...]:
    """Select a maximum-cardinality fundable set under every prefix capacity.

    Candidate j can only draw from sources i < j. The nested-prefix
    feasibility condition is:

        sum(deficit[k] for selected k <= j) <= sum(bank[i] for i < j).

    Ties in the maximum-deficit heap evict the larger destination ID first.
    """
    if len(deficits) != len(unassigned_by_source):
        raise ValueError("deficit and source-bank vectors must have equal length")

    bank = [_fraction(value) for value in unassigned_by_source]
    if any(value < 0 for value in bank):
        raise ValueError("source balances must be nonnegative")

    normalized: list[Fraction | None] = [
        None if value is None else _fraction(value)
        for value in deficits
    ]
    if any(value is not None and value < 0 for value in normalized):
        raise ValueError("deficits must be nonnegative")

    heap: list[tuple[Fraction, int]] = []
    selected: set[int] = set()
    selected_cost = Fraction(0)
    capacity = Fraction(0)

    for destination, deficit in enumerate(normalized):
        if deficit is not None and deficit > 0:
            heapq.heappush(heap, (-deficit, -destination))
            selected.add(destination)
            selected_cost += deficit

        # Source destination-1 becomes eligible for this destination.
        capacity += bank[destination - 1] if destination > 0 else 0
        while selected_cost > capacity:
            if not heap:
                raise AssertionError("positive selected cost with empty heap")
            negative_cost, negative_destination = heapq.heappop(heap)
            evicted = -negative_destination
            if evicted not in selected:
                continue
            selected.remove(evicted)
            selected_cost -= -negative_cost

    return tuple(sorted(selected))


def assign_selected(
    deficits: Sequence[Rational | None],
    unassigned_tokens: Sequence[Sequence[tuple[tuple[int, int], Rational]]],
    selected: Sequence[int],
) -> tuple[tuple[AssignmentFragment, ...], tuple[tuple[tuple[int, int], Fraction], ...]]:
    """Consume source tokens oldest-first for a selected feasible recipient set.

    Token identity is (source_id, source_local_ordinal). A token can be split
    across recipients; each returned fragment has its own local assignment
    ordinal. The input token state is not mutated.
    """
    n = len(deficits)
    if len(unassigned_tokens) != n:
        raise ValueError("token-bank vector must match the candidate universe")

    normalized_deficits = [
        None if value is None else _fraction(value)
        for value in deficits
    ]
    balances: list[list[list[object]]] = []
    for source, tokens in enumerate(unassigned_tokens):
        source_rows = []
        seen = set()
        for token_id_raw, amount_raw in tokens:
            token_id = tuple(token_id_raw)
            amount = _fraction(amount_raw)
            if len(token_id) != 2 or token_id[0] != source or token_id[1] <= 0:
                raise ValueError("token ID must be (source_id, positive local ordinal)")
            if token_id in seen:
                raise ValueError("duplicate source token ID")
            if amount < 0:
                raise ValueError("token balance must be nonnegative")
            seen.add(token_id)
            source_rows.append([token_id, amount])
        source_rows.sort(key=lambda row: row[0][1])
        balances.append(source_rows)

    selected_set = tuple(sorted(set(selected)))
    if len(selected_set) != len(selected):
        raise ValueError("selected destinations must be unique")
    if any(not 0 <= j < n for j in selected_set):
        raise IndexError("destination out of range")
    if any(normalized_deficits[j] is None or normalized_deficits[j] <= 0
           for j in selected_set):
        raise ValueError("selected destination requires a positive deficit")

    fragments: list[AssignmentFragment] = []
    next_assignment_ordinal: dict[tuple[int, int], int] = {}

    for destination in selected_set:
        remaining = normalized_deficits[destination]
        assert remaining is not None
        for source in range(destination):
            for token_row in balances[source]:
                token_id = token_row[0]
                token_balance = token_row[1]
                take = min(token_balance, remaining)
                if take <= 0:
                    continue
                ordinal_key = token_id
                ordinal = next_assignment_ordinal.get(ordinal_key, 0) + 1
                next_assignment_ordinal[ordinal_key] = ordinal
                fragments.append(AssignmentFragment(
                    source_id=source,
                    token_id=token_id,
                    destination_id=destination,
                    assignment_ordinal=ordinal,
                    amount=take,
                ))
                token_row[1] = token_balance - take
                remaining -= take
                if remaining == 0:
                    break
            if remaining == 0:
                break
        if remaining != 0:
            raise ValueError(
                f"selected destination {destination} is not fundable from earlier sources"
            )

    remaining_tokens = tuple(
        (token_row[0], token_row[1])
        for source_rows in balances
        for token_row in source_rows
    )
    for fragment in fragments:
        assert fragment.source_id < fragment.destination_id
        assert fragment.amount > 0
    return tuple(fragments), remaining_tokens


def verify_prefix_feasibility(
    deficits: Sequence[Rational | None],
    unassigned_by_source: Sequence[Rational],
    selected: Sequence[int],
) -> bool:
    if len(deficits) != len(unassigned_by_source):
        return False
    bank = [_fraction(value) for value in unassigned_by_source]
    selected_set = set(selected)
    cumulative = Fraction(0)
    for destination in range(len(deficits)):
        deficit = deficits[destination]
        if destination in selected_set:
            if deficit is None:
                return False
            cumulative += _fraction(deficit)
        capacity = sum(bank[:destination], Fraction(0))
        if cumulative > capacity:
            return False
    return True
