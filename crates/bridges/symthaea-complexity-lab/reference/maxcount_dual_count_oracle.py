#!/usr/bin/env python3
"""Exact dual-count MaxCount reference: weighted multiplicity + distinct support.

Every original factor-table cell is one local assignment. MaxCount weights
multiplicity; a parallel support channel counts distinct assignments. Support
products are valid because generated factors summarize disjoint eliminated
variable sets. This is a finite reference experiment, not production code.
"""
from __future__ import annotations
from itertools import product
from pathlib import Path
import json
import random

Value = tuple[int, int, int] | None  # (score, multiplicity, distinct support)
ZERO = None
ONE: Value = (0, 1, 1)


def plus(a: Value, b: Value) -> Value:
    if a is None: return b
    if b is None: return a
    if a[0] > b[0]: return a
    if b[0] > a[0]: return b
    return (a[0], a[1] + b[1], a[2] + b[2])


def times(a: Value, b: Value) -> Value:
    if a is None or b is None: return None
    return (a[0] + b[0], a[1] * b[1], a[2] * b[2])


def factor_index(scope, states, cards):
    idx = 0
    for variable in scope:
        idx = idx * cards[variable] + states[variable]
    return idx


def direct(cards, factors):
    result = ZERO
    for states in product(*(range(c) for c in cards)):
        term = ONE
        for factor in factors:
            raw = factor["table"][factor_index(factor["scope"], states, cards)]
            term = times(term, None if raw is None else (raw[0], raw[1], 1))
        result = plus(result, term)
    return result


def eliminate(cards, factors, order):
    active = [{
        "scope": list(f["scope"]),
        "values": [None if x is None else (x[0], x[1], 1) for x in f["table"]],
    } for f in factors]
    for variable in order:
        bucket = [f for f in active if variable in f["scope"]]
        separator = sorted({v for f in bucket for v in f["scope"] if v != variable})
        output = []
        for sep_states in product(*(range(cards[v]) for v in separator)):
            assignment = dict(zip(separator, sep_states))
            candidates = []
            for state in range(cards[variable]):
                assignment[variable] = state
                term = ONE
                for factor in bucket:
                    raw = factor["values"][factor_index(factor["scope"], assignment, cards)]
                    term = times(term, raw)
                    if term is None: break
                if term is not None:
                    candidates.append((state, term))
            aggregate = ZERO
            for _, value in candidates:
                aggregate = plus(aggregate, value)
            output.append(aggregate)
        active = [f for f in active if variable not in f["scope"]]
        active.append({"scope": separator, "values": output})
    result = ONE
    for factor in active:
        if factor["scope"]: raise AssertionError("uneliminated variable")
        result = times(result, factor["values"][0])
    return result


def random_instance(rng, number):
    n = rng.randint(1, 6)
    cards = [rng.choice([2, 2, 3]) for _ in range(n)]
    factors = []
    for fi in range(rng.randint(0, n + 4)):
        scope = sorted(rng.sample(range(n), rng.randint(1, min(3, n))))
        size = 1
        for v in scope: size *= cards[v]
        table = [
            None if rng.random() < 0.12 else [rng.randint(-5, 7), rng.randint(1, 4)]
            for _ in range(size)
        ]
        factors.append({"id": f"f{fi}", "scope": scope, "table": table})
    if rng.random() < 0.4:
        factors.append({
            "id": f"scalar{number}", "scope": [],
            "table": [[rng.randint(-2, 3), rng.randint(1, 5)]],
        })
    return cards, factors


def check_laws(rng, trials=20_000):
    values = [None] + [
        (score, multiplicity, support)
        for score in range(-3, 4)
        for multiplicity in range(1, 4)
        for support in range(1, 4)
    ]
    for _ in range(trials):
        a, b, c = (rng.choice(values) for _ in range(3))
        assert plus(a, ZERO) == a and plus(ZERO, a) == a
        assert times(a, ONE) == a and times(ONE, a) == a
        assert times(a, ZERO) == ZERO and times(ZERO, a) == ZERO
        assert plus(a, b) == plus(b, a)
        assert plus(plus(a, b), c) == plus(a, plus(b, c))
        assert times(a, b) == times(b, a)
        assert times(times(a, b), c) == times(a, times(b, c))
        assert times(a, plus(b, c)) == plus(times(a, b), times(a, c))
        assert times(plus(a, b), c) == plus(times(a, c), times(b, c))
    return trials


def run():
    rng = random.Random(20261103)
    law_trials = check_laws(rng)
    direct_matches = 0
    support_matches = 0
    order_executions = 0
    for i in range(512):
        cards, factors = random_instance(rng, i)
        expected = direct(cards, factors)
        base = list(range(len(cards)))
        orders = [base[:], list(reversed(base))]
        shuffled = base[:]
        rng.shuffle(shuffled)
        orders.append(shuffled)
        actual = None
        for order in orders:
            candidate = eliminate(cards, factors, order)
            assert candidate == expected, (i, cards, factors, order, candidate, expected)
            actual = candidate
            direct_matches += 1
            order_executions += 1
        support = 0 if actual is None else actual[2]
        exact_support = 0
        if actual is not None:
            for states in product(*(range(c) for c in cards)):
                score, feasible = 0, True
                for factor in factors:
                    raw = factor["table"][factor_index(factor["scope"], states, cards)]
                    if raw is None:
                        feasible = False
                        break
                    score += raw[0]
                if feasible and score == actual[0]:
                    exact_support += 1
        assert support == exact_support, (i, support, exact_support)
        support_matches += 1

    weighted_unique = eliminate([2], [
        {"id": "weighted", "scope": [0], "table": [[5,4],[4,1]]}
    ], [0])
    assert weighted_unique == (5,4,1)

    weighted_tie = eliminate([2], [
        {"id": "weighted-tie", "scope": [0], "table": [[5,4],[5,2]]}
    ], [0])
    assert weighted_tie == (5,6,2)

    triangle = [
        {"id": f"e{i}", "scope": [u,v],
         "table": [[0,1],[1,1],[1,1],[0,1]]}
        for i,(u,v) in enumerate([(0,1),(1,2),(0,2)])
    ]
    triangle_result = eliminate([2,2,2], triangle, [0,1,2])
    assert triangle_result == (2,6,6)

    free = eliminate([2,3,4], [], [0,1,2])
    assert free == (0,24,24)

    result = {
        "suite": "Dual-count MaxCount exact semiring reference",
        "seed": 20261103,
        "algebra_law_trials": law_trials,
        "algebra_laws_passed": True,
        "randomized_instances": 512,
        "orders_per_instance": 3,
        "order_algebra_executions": order_executions,
        "direct_maxcount_matches": direct_matches,
        "distinct_support_matches": support_matches,
        "weighted_unique_assignment": {"optimum": 5, "multiplicity": 4, "distinct_assignments": 1},
        "weighted_tie_case": {"optimum": 5, "multiplicity": 6, "distinct_assignments": 2},
        "triangle_maxcut": {"optimum": 2, "multiplicity": 6, "distinct_assignments": 6},
        "unconstrained_three_variable_product": {"optimum": 0, "multiplicity": 24, "distinct_assignments": 24},
        "claim_boundary": "Python arbitrary-precision finite reference only. Assumes each input factor-table cell represents one local assignment; support counts multiply across disjoint eliminated-variable blocks. No Rust/Lean or performance claim.",
    }
    path = Path(__file__).with_name("maxcount_dual_count_results.json")
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
