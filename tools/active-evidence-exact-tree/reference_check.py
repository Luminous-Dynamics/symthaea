#!/usr/bin/env python3
"""Independent standard-library reference/checker for the Rust exact-tree microbenchmark."""
import csv
import math
import sys

PRIOR = (0.6, 0.3, 0.1)
LOSSES = ((0., 4., 7.), (5., 0., 3.), (8., 2., 0.))
COSTS = (1., 1., 2., 1.)
NOMINAL = ((.88, .34, .25), (.35, .86, .31), (.24, .38, .88), (.76, .70, .18))
Q0_SHIFT = ((.88, .76, .25), (.35, .86, .31), (.24, .38, .88), (.76, .70, .18))
Q1_SHIFT = ((.88, .34, .25), (.83, .22, .31), (.24, .38, .88), (.76, .70, .18))
COMPOUND = ((.88, .74, .25), (.79, .22, .31), (.24, .38, .88), (.76, .83, .18))
OUT_OF_SET = ((.15, .92, .25), (.35, .86, .31), (.24, .38, .88), (.76, .70, .18))
AMBIGUITY = (NOMINAL, Q0_SHIFT, Q1_SHIFT, COMPOUND)
SCENARIOS = (
    ("iid", NOMINAL),
    ("q0_conditional_shift", Q0_SHIFT),
    ("q1_asymmetric_error", Q1_SHIFT),
    ("compound_shift", COMPOUND),
    ("out_of_set_flip", OUT_OF_SET),
)
POLICIES = ("random", "entropy", "plugin_evsi", "robust_evsi", "known_channel_evsi_reference")
EPS = 1e-12
FIELDS = ("scenario", "policy", "query_cap", "expected_loss",
          "expected_decision_regret", "expected_query_cost", "expected_query_count")


def p_observe(matrix, q, h, y):
    p = matrix[q][h]
    return p if y else 1.0 - p


def posterior(b, matrix, q, y):
    values = [b[h] * p_observe(matrix, q, h, y) for h in range(3)]
    total = sum(values)
    return tuple(v / total for v in values) if total > 0 else PRIOR


def action_risk(b, action):
    return sum(b[h] * LOSSES[h][action] for h in range(3))


def best_action(b):
    best = 0
    best_risk = action_risk(b, 0)
    for action in range(1, 3):
        candidate = action_risk(b, action)
        if candidate < best_risk - EPS:
            best = action
            best_risk = candidate
    return best


def bayes_risk(b):
    return action_risk(b, best_action(b))


def entropy(b):
    return -sum(p * math.log2(p) for p in b if p > 0)


def expected_risk_after(b, matrix, q):
    total = 0.0
    for y in (False, True):
        by = posterior(b, matrix, q, y)
        py = sum(b[h] * p_observe(matrix, q, h, y) for h in range(3))
        total += py * bayes_risk(by)
    return total


def robust_gain(b, q):
    action0 = best_action(posterior(b, NOMINAL, q, False))
    action1 = best_action(posterior(b, NOMINAL, q, True))
    worst = -math.inf
    for matrix in AMBIGUITY:
        loss = 0.0
        for h in range(3):
            p = matrix[q][h]
            loss += b[h] * ((1-p) * LOSSES[h][action0] + p * LOSSES[h][action1])
        worst = max(worst, loss)
    return bayes_risk(b) - worst


def select(policy, b, mask, truth):
    available = [q for q in range(4) if mask & (1 << q)]
    if not available:
        return []
    if policy == "random":
        return [(q, 1.0 / len(available)) for q in available]
    matrix = truth if policy == "known_channel_evsi_reference" else NOMINAL
    selected = None
    for q in available:
        if policy == "entropy":
            gain = entropy(b) - sum(
                (sum(b[h] * p_observe(NOMINAL, q, h, y) for h in range(3))
                 * entropy(posterior(b, NOMINAL, q, y)))
                for y in (False, True)
            )
        elif policy == "plugin_evsi":
            gain = bayes_risk(b) - expected_risk_after(b, NOMINAL, q)
        elif policy == "robust_evsi":
            gain = robust_gain(b, q)
        elif policy == "known_channel_evsi_reference":
            gain = bayes_risk(b) - expected_risk_after(b, truth, q)
        else:
            raise ValueError(policy)
        score = gain / COSTS[q]
        if selected is None or score > selected[0] + EPS:
            selected = (score, q, gain)
    _, q, gain = selected
    if policy == "robust_evsi" and gain <= EPS:
        return []
    return [(q, 1.0)]


def recurse(policy, truth, h, b, mask, rounds):
    if rounds == 0 or mask == 0:
        action = best_action(b)
        loss = LOSSES[h][action]
        regret = loss - min(LOSSES[h])
        return (loss, regret, 0.0, 0.0)
    choices = select(policy, b, mask, truth)
    if not choices:
        action = best_action(b)
        loss = LOSSES[h][action]
        regret = loss - min(LOSSES[h])
        return (loss, regret, 0.0, 0.0)
    update = truth if policy == "known_channel_evsi_reference" else NOMINAL
    out = [0.0, 0.0, 0.0, 0.0]
    for q, pq in choices:
        remaining = mask & ~(1 << q)
        for y in (False, True):
            weight = pq * p_observe(truth, q, h, y)
            bnext = posterior(b, update, q, y)
            child = recurse(policy, truth, h, bnext, remaining, rounds-1)
            out[0] += weight * child[0]
            out[1] += weight * child[1]
            out[2] += weight * (COSTS[q] + child[2])
            out[3] += weight * (1.0 + child[3])
    return tuple(out)


def expected(truth, policy, cap):
    accum = [0.0] * 4
    for h, prior_mass in enumerate(PRIOR):
        result = recurse(policy, truth, h, PRIOR, 0b1111, min(cap, 4))
        for i in range(4):
            accum[i] += prior_mass * result[i]
    return accum


def reference_rows():
    rows = []
    for scenario, matrix in SCENARIOS:
        for cap in range(4):
            for policy in POLICIES:
                values = expected(matrix, policy, cap)
                rows.append({
                    "scenario": scenario, "policy": policy, "query_cap": str(cap),
                    "expected_loss": f"{values[0]:.12f}",
                    "expected_decision_regret": f"{values[1]:.12f}",
                    "expected_query_cost": f"{values[2]:.12f}",
                    "expected_query_count": f"{values[3]:.12f}",
                })
    return rows


def write_reference(path):
    with open(path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(reference_rows())


def check_csv(path):
    expected_rows = reference_rows()
    expected_map = {(r["scenario"], r["policy"], r["query_cap"]): r for r in expected_rows}
    try:
        with open(path, newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != FIELDS:
                raise ValueError(f"header mismatch: expected {FIELDS}, got {reader.fieldnames}")
            observed_rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"cannot read candidate CSV: {exc}") from exc

    if len(observed_rows) != len(expected_rows):
        raise ValueError(f"row count mismatch: expected {len(expected_rows)}, got {len(observed_rows)}")
    seen = set()
    numeric_fields = FIELDS[3:]
    for line_no, row in enumerate(observed_rows, 2):
        key = (row["scenario"], row["policy"], row["query_cap"])
        if key in seen:
            raise ValueError(f"duplicate key at line {line_no}: {key}")
        seen.add(key)
        if key not in expected_map:
            raise ValueError(f"unexpected scenario/policy/budget at line {line_no}: {key}")
        reference = expected_map[key]
        for field in numeric_fields:
            try:
                actual = float(row[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid {field} at line {line_no}") from exc
            expected = float(reference[field])
            if not math.isfinite(actual) or abs(actual - expected) > 5e-10:
                raise ValueError(
                    f"{field} mismatch at line {line_no} for {key}: actual={actual}, expected={expected}"
                )
    if seen != set(expected_map):
        raise ValueError(f"missing keys: {set(expected_map) - seen}")
    print(f"PASS: independently recomputed all {len(expected_rows)} rows and 4 numeric fields per row")


def main(argv):
    if len(argv) == 3 and argv[1] == "--write-reference":
        write_reference(argv[2])
        print(f"Wrote reference rows to {argv[2]}")
        return 0
    if len(argv) != 2 or argv[1] in ("-h", "--help"):
        print("Usage: reference_check.py CANDIDATE.csv")
        print("       reference_check.py --write-reference OUTPUT.csv")
        return 2
    try:
        check_csv(argv[1])
        return 0
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
