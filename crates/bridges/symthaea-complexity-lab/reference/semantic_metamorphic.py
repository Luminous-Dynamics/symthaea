#!/usr/bin/env python3
"""Exact metamorphic reference tests for MC/WMC/PMC/PWMC/complex AMC."""
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
import json
import random
import platform
import time

G = tuple[Fraction, Fraction]
ZERO: G = (Fraction(0), Fraction(0))
ONE: G = (Fraction(1), Fraction(0))


def add(a: G, b: G) -> G:
    return a[0] + b[0], a[1] + b[1]


def mul(a: G, b: G) -> G:
    return a[0] * b[0] - a[1] * [1], a[0] * b[1] + a[1] * b[0]


def show(a: G) -> str:
    return str(a[0]) if a[1] == 0 else f"{a[0]}{a[1]:+}"i"


@dataclass
class Instance:
    mode: str
    n: int
    clauses: list[list[int]]
    projection: set[int]
    weights: dict[int, G]


def clone(x: Instance) -> Instance:
    return Instance(x.mode, x.n, [c[:] for c in x.clauses], set(x.projection), dict(x.weights))


def w(x: Instance, lit: int) -> G:
    return x.weights.get(lit, ONE)


def sat_clause(clause: list[int], assignment: dict[int, bool]) -> bool:
    return any(assignment[abs(lit)] == (lit > 0) for lit in clause)


def evaluate(x: Instance) -> G:
    """Direct exact oracle. PMC/PWMC use a set of visible assignments."""
    models = []
    for bits in product((False, True), repeat=x.n):
        a = {i + 1: bits[i] for i in range(x.n)}
        if all(sat_clause(c, a) for c in x.clauses):
            models.append(a)

    if x.mode == "mc":
        return Fraction(len(models)), Fraction(0)
    if x.mode == "pmc":
        projected = {tuple(a[v] for v in sorted(x.projection)) for a in models}
        return Fraction(len(projected)), Fraction(0)

    total = ZERO
    if x.mode in ("wmc", "amc-complex"):
        for a in models:
            term = ONE
            for v in range(1, x.n + 1):
                term = mul(term, w(x, v if a[v] else -v))
            total = add(total, term)
        return total
    if x.mode == "pwmc":
        projected = {tuple(a[v] for v in sorted(x.projection)) for a in models}
        for bits in projected:
            term = ONE
            for v, bit in zip(sorted(x.projection), bits):
                term = mul(term, w(x, v if bit else -v))
            total = add(total, term)
        return total
    raise ValueError(f"unsupported mode: {x.mode}")


def q(rng: random.Random) -> Fraction:
    return Fraction(rng.randint(-3, 3), rng.randint(1, 4))


def qi(rng: random.Random) -> G:
    return q(rng), q(rng)


def generate(rng: random.Random, mode: str) -> Instance:
    n = rng.randint(1, 6)
    clauses = []
    for _ in range(rng.randint(1, 9)):
        width = rng.randint(1, min(3, n))
        variables = rng.sample(range(1, n + 1), width)
        clauses.append([v if rng.choice((True, False)) else -v for v in variables])
    projection = set(rng.sample(range(1, n + 1), rng.randint(1, n))) if mode in ("pmc", "pwmc") else set()
    weights: dict[int, G] = {}
    if mode == "wmc":
        for v in range(1, n + 1):
            weights[v] = (q(rng), Fraction(0))
            weights[-v] = (q(rng), Fraction(0))
    elif mode == "pwmc":
        for v in projection:
            weights[v] = (q(rng), Fraction(0))
            weights[-v] = (q(rng), Fraction(0))
    elif mode == "amc-complex":
        for v in range(1, n + 1):
            weights[v], weights[-v] = qi(rng), qi(rng)
    return Instance(mode, n, clauses, projection, weights)


def syntactic_equivalent(x: Instance, rng: random.Random) -> Instance:
    y = clone(x)
    rng.shuffle(y.clauses)
    for c in y.clauses:
        rng.shuffle(c)
    y.clauses.append(list(rng.choice(y.clauses)))  # duplicate clause
    y.clauses.append([1, -1])                    # tautology
    return y


def complement_variable(x: Instance, v: int) -> Instance:
    y = clone(x)
    y.clauses = [[-lit if abs(lit) == v else lit for lit in c] for c in y.clauses]
    positive, negative = y.weights.pop(v, None), y.weights.pop(-v, None)
    if negative is not None:
        y.weights[v] = negative
    if positive is not None:
        y.weights[-v] = positive
    return y


def permute_variables(x: Instance, rng: random.Random) -> Instance:
    y = clone(x)
    old = list(range(1, x.n + 1))
    new = old[:]
    rng.shuffle(new)
    mapping = dict(zip(old, new))
    y.clauses = [[mapping[abs(lit)] if lit > 0 else -mapping[abs(lit)] for lit in c] for c in x.clauses]
    y.projection = {mapping[v] for v in x.projection}
    y.weights = {(mapping[abs(lit)] if lit > 0 else -mapping[abs(lit)]): val for lit, val in x.weights.items()}
    return y


def add_free_variable(x: Instance, visible: bool, wp: G, wn: G) -> tuple[Instance, G]:
    y = clone(x)
    z = x.n + 1
    y.n += 1
    if visible and x.mode in ("pmc", "pwmc"):
        y.projection.add(z)
    if x.mode in ("wmc", "amc-complex") or (x.mode == "pwmc" and visible):
        y.weights[z], y.weights[-z] = wp, wn
    if x.mode == "mc":
        scale = (Fraction(2), Fraction(0))
    elif x.mode == "wmc" or x.mode == "amc-complex":
        scale = add(wp, wn)
    elif x.mode == "pmc":
        scale = (Fraction(2 if visible else 1), Fraction(0))
    else:
        scale = add(wp, wn) if visible else ONE
    return y, scale


def add_forced_true_variable(x: Instance, visible: bool, wp: G, wn: G) -> tuple[Instance, G]:
    y = clone(x)
    z = x.n + 1
    y.n += 1
    y.clauses.append([z])
    if visible and x.mode in ("pmc", "pwmc"):
        y.projection.add(z)
    if x.mode in ("wmc", "amc-complex") or (x.mode == "pwmc" and visible):
        y.weights[z], y.weights[-z] = wp, wn
    if x.mode in ("mc", "pmc") or (x.mode == "pwmc" and not visible):
        scale = ONE
    else:
        scale = wp
    return y, scale


def add_unsat_extension(x: Instance) -> Instance:
    y = clone(x)
    z = x.n + 1
    y.n += 1
    y.clauses.extend(([z], [-z]))
    if x.mode in ("pmc", "pwmc"):
        y.projection.add(z)
    return y


def targeted_semantic_controls() -> dict:
    clauses = [[1, 2], [1, -2]]
    pmc = Instance("pmc", 2, clauses, {1}, {})
    pwmc = Instance("pwmc", 2, clauses, {1}, {
        1: (Fraction(2, 3), Fraction(0)),
        -1: (Fraction(1, 3), Fraction(0)),
    })
    wmc_zero = Instance("wmc", 1, [], set(), {
        1: (Fraction(1), Fraction(0)),
        -1: (Fraction(-1), Fraction(0)),
    })
    amc_zero = Instance("amc-complex", 1, [], set(), {
        1: (Fraction(1), Fraction(2)),
        -1: (Fraction(-1), Fraction(-2)),
    })

    pmc_value = evaluate(pmc)
    pwmc_value = evaluate(pwmc)
    all_witness_weight = add(
        mul(w(pwmc, 1), ONE), mul(w(pwmc, 1), ONE)
    )
    wmc_value = evaluate(wmc_zero)
    amc_value = evaluate(amc_zero)

    # Deliberately wrong hidden-witness multiplier must be detected.
    hidden, _ = add_free_variable(pmc, False, ONE, ONE)
    wrong_hidden_prediction = mul(pmc_value, (Fraction(2), Fraction(0)))
    assert pmc_value == (Fraction(1), Fraction(0))
    assert evaluate(Instance("mc", 2, clauses, set(), {})) == (Fraction(2), Fraction(0))
    assert pwmc_value == (Fraction(2, 3), Fraction(0))
    assert all_witness_weight == (Fraction(4, 3), Fraction(0))
    assert evaluate(hidden) == pmc_value and evaluate(hidden) != wrong_hidden_prediction
    assert wmc_value == ZERO and amc_value == ZERO
    return {
        "multiple_hidden_witnesses": {
            "ordinary_model_count": show(evaluate(Instance("mc", 2, clauses, set(), {}))),
            "projected_model_count": show(pmc_value),
            "projected_weighted_count": show(pwmc_value),
            "incorrect_witness_multiplication": show(all_witness_weight),
        },
        "wrong_hidden_free_variable_doubling_detected": True,
        "exact_zero_cancellation": {"wmc": show(wmc_value), "amc-complex": show(amc_value)},
        "all_passed": True,
    }


def run(seed: int = 0xE7025, per_mode: int = 100) -> dict:
    started = time.perf_counter()
    rng = random.Random(seed)
    modes = ["mc", "wmc", "pmc", "pwmc", "amc-complex"]
    names = ["syntactic_normalization", "variable_complement", "variable_permutation",
             "free_variable_scaling", "forced_true_variable_scaling", "unsatisfiable_extension_zero"]
    counts = {name: 0 for name in names}
    witness_checks = 0
    total = 0
    failures = []
    exact_zero_bases = 0

    for mode in modes:
        for case in range(per_modei):
            x = generate(rng, mode)
            base = evaluate(x)
            exact_zero_bases += (base == ZERO)
            cases: list[tuple[str, Instance, G]] = [
                ("syntactic_normalization", syntactic_equivalent(x, rng), base),
                ("variable_complement", complement_variable(x, rng.randint(1, x.n)), base),
                ("variable_permutation", permute_variables(x, rng), base),
            ]
            wp = qi(rng) if mode == "amc-complex" else (q(rng), Fraction(0))
            wn = qi(rng) if mode == "amc-complex" else (q(rng), Fraction(0))
            visible = rng.choice((True, False))
            free, scale = add_free_variable(x, visible, wp, wn)
            cases.append(("free_variable_scaling", free, mul(base, scale))
            forced, scale = add_forced_true_variable(x, visible, wp, wn)
            cases.append(("forced_true_variable_scaling", forced, mul(base, scale)))
            cases.append(("unsatisfiable_extension_zero", add_unsat_extension(x), ZERO))

            for name, changed, expected in cases:
                actual = evaluate(changed)
                total += 1
                counts[name] += 1
                if actual != expected:
                    failures.append({"mode": mode, "case": case, "transform": name,
                                     "expected": show(expected), "actual": show(actual)})

            if mode in ("pmc", "pwmc"):
                hidden, _ = add_free_variable(x, False, ONE, ONE)
                actual = evaluate(hidden)
                witness_checks += 1
                if actual != base:
                    failures.append({"mode": mode, "case": case,
                                     "transform": "hidden_witness_invariance",
                                     "expected": show(base), "actual": show(actual)})

    targeted = targeted_semantic_controls()
    return {
        "suite": "ETH-025 exact semantics metamorphic benchmark",
        "python_version": platform.python_version(),
        "seed": hex(seed),
        "instances_per_mode": per_mode,
        "modes": modes,
        "metamorphic_checks": total,
        "checks_by_transform": counts,
        "hidden_witness_invariance_checks": witness_checks,
        "base_instances_with_exact_zero": exact_zero_bases,
        "failure_count": len(failures),
        "failures": failures[:20],
        "targeted_semantic_controls": targeted,
        "all_passed": not failures and targeted["all_passed"],
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "claim_ceiling": "Executed finite exact direct-oracle/metamorphic relations on small instances. Not a differential run of Ganak or Symthaea binaries, an official format-parser qualification, Rust/Lean verification, or a performance result."
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
