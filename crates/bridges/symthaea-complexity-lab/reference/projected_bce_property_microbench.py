#!/usr/bin/env python3
"""Seeded exhaustive property corpus for hidden-variable blocked-clause deletion."""
from __future__ import annotations
import hashlib, json, random, sys
from fractions import Fraction
from itertools import product
from pathlib import Path

SEED = 0xE7028
CASES = 2000


def tautology(lits):
    s = set(lits)
    return any(-x in s for x in s)


def is_blocked(clauses, cid, blocker):
    by_id = {c["id"]: c for c in clauses}
    if cid not in by_id or blocker not in by_id[cid]["lits"]:
        return False, []
    checks = []
    for other in clauses:
        if other["id"] == cid or -blocker not in other["lits"]:
            continue
        resolvent = [x for x in by_id[cid]["lits"] if x != blocker]
        resolvent += [x for x in other["lits"] if x != -blocker]
        if not tautology(resolvent):
            return False, checks + [(other["id"], sorted(set(resolvent)), False)]
        checks.append((other["id"], sorted(set(resolvent)), True))
    return True, checks


def satisfies(clauses, assignment):
    return all(any(assignment[abs(l)] == (l > 0) for l in c["lits"]) for c in clauses)


def oracle(nvars, visible, clauses):
    models = 0
    support = set()
    for bits in product((False, True), repeat=nvars):
        assignment = {i + 1: bits[i] for i in range(nvars)}
        if satisfies(clauses, assignment):
            models += 1
            support.add(tuple(assignment[v] for v in visible))
    return models, support


def weighted_support(support, visible, weights):
    total = Fraction(0, 1)
    for row in support:
        term = Fraction(1, 1)
        for var, bit in zip(visible, row):
            term *= weights[var][1 if bit else 0]
        total += term
    return total


def generate_case(rng, case_id):
    n = rng.randint(3, 5)
    hidden = {1}
    hidden.update(v for v in range(2, n + 1) if rng.random() < 0.2)
    visible = [v for v in range(1, n + 1) if v not in hidden]
    other_vars = rng.sample(range(2, n + 1), rng.randint(1, min(3, n - 1)))
    c_lits = [1] + [v if rng.getrandbits(1) else -v for v in other_vars]
    clauses = [{"id": "c0", "lits": c_lits}]
    opposing_count = rng.randint(1, 4)
    for j in range(opposing_count):
        selected = rng.choice(c_lits[1:])
        forced = -selected
        lits = [-1, forced]
        candidates = [x for x in range(2, n + 1) if abs(x) not in {abs(z) for z in lits}]
        rng.shuffle(candidates)
        for var in candidates[:rng.randint(0, min(2, len(candidates)))]:
            lits.append(var if rng.getrandbits(1) else -var)
        clauses.append({"id": f"o{j}", "lits": lits})
    for j in range(rng.randint(0, 4)):
        candidates = [v if rng.getrandbits(1) else -v for v in range(2, n + 1)]
        rng.shuffle(candidates)
        lits = candidates[:rng.randint(1, min(3, len(candidates)))]
        clauses.append({"id": f"n{j}", "lits": lits})
    weights = {}
    for v in visible:
        a = Fraction(rng.randint(-3, 4), rng.randint(1, 5))
        b = Fraction(rng.randint(-3, 4), rng.randint(1, 5))
        weights[v] = (a, b)
    return {"id": case_id, "nvars": n, "visible": visible, "hidden": sorted(hidden),
            "clauses": clauses, "removed_clause_id": "c0", "blocker": 1, "weights": weights}


def main():
    rng = random.Random(SEED)
    full_changed = 0
    for idx in range(CASES):
        case = generate_case(rng, f"bce-{idx:05d}")
        blocked, resolvents = is_blocked(case["clauses"], case["removed_clause_id"], case["blocker"])
        if not blocked:
            raise AssertionError(f"generator emitted nonblocked case {case['id']}: {resolvents}")
        target = [c for c in case["clauses"] if c["id"] != case["removed_clause_id"]]
        source_models, source_support = oracle(case["nvars"], case["visible"], case["clauses"])
        target_models, target_support = oracle(case["nvars"], case["visible"], target)
        if source_support != target_support:
            raise AssertionError(f"projected support changed in {case['id']}")
        if source_models != target_models:
            full_changed += 1
        source_weight = weighted_support(source_support, case["visible"], case["weights"])
        target_weight = weighted_support(target_support, case["visible"], case["weights"])
        if source_weight != target_weight:
            raise AssertionError(f"PWMC changed in {case['id']}")

    nonblocked = [{"id": "c", "lits": [1]}, {"id": "d", "lits": [-1, 2]}]
    blocked, _ = is_blocked(nonblocked, "c", 1)
    if blocked:
        raise AssertionError("nonblocked negative control incorrectly classified as blocked")
    before_models, before_support = oracle(2, [2], nonblocked)
    after_models, after_support = oracle(2, [2], [nonblocked[1]])
    if before_support == after_support or before_support != {(True,)} or after_support != {(False,), (True,)}:
        raise AssertionError("nonblocked deletion did not expose expected support change")

    result = {
        "format": "symthaea.projected-hidden-bce-property-results",
        "version": 1,
        "claim_ceiling": "seeded_small_formula_exhaustive_property_tests_not_formal_proof_or_large_scale_solver",
        "seed": SEED, "positive_cases": CASES, "positive_cases_passed": CASES,
        "all_positive_cases_preserved_projected_support": True,
        "all_positive_cases_preserved_exact_visible_weighted_value": True,
        "positive_cases_with_changed_full_model_count": full_changed,
        "nonblocked_negative_control": {
            "classified_as_blocked": False, "source_full_models": before_models,
            "target_full_models": after_models, "source_projected_support": ["1"],
            "target_projected_support": ["0", "1"], "support_change_detected": True
        },
        "runtime": {"python": sys.version.split()[0], "platform": sys.platform},
        "script_sha256": "sha256:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "script_git_blob_sha1": hashlib.sha1(b"blob " + str(Path(__file__).stat().st_size).encode() + b"\0" + Path(__file__).read_bytes()).hexdigest(),
        "execution": "uploaded_script_byte_verified"
    }
    out = Path(__file__).with_name("projected_hidden_bce_property_results.v1.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"PASS {CASES}/{CASES} blocked-hidden-variable cases: support and exact PWMC preserved")
    print(f"PASS full-model count changed in {full_changed}/{CASES} cases (not a preserved invariant)")
    print("PASS nonblocked negative control: projected support changes from {1} to {0,1}")


if __name__ == "__main__":
    main()
