#!/usr/bin/env python3
"""Independent standard-library reference for calibrated_box.rs."""
import csv
import itertools
import math
import sys

PRIOR = (0.6, 0.3, 0.1)
LOSS = ((0., 4., 7.), (5., 0., 3.), (8., 2., 0.))
COST = (1., 1., 2., 1.)
NOMINAL = ((.88, .34, .25), (.35, .86, .31), (.24, .38, .88), (.76, .70, .18))
SCENARIOS = (
    ("iid", NOMINAL),
    ("q0_conditional_shift", ((.88, .76, .25), (.35, .86, .31), (.24, .38, .88), (.76, .70, .18))),
    ("q1_asymmetric_error", ((.88, .34, .25), (.83, .22, .31), (.24, .38, .88), (.76, .70, .18))),
    ("compound_shift", ((.88, .74, .25), (.79, .22, .31), (.24, .38, .88), (.76, .83, .18))),
    ("out_of_set_flip", ((.15, .92, .25), (.35, .86, .31), (.24, .38, .88), (.76, .70, .18))),
)
SAMPLE_SIZES = (20, 50, 100, 200, 500, 1000)
ALPHA = .05
CELLS = 12
EPS = 1e-12
QUERY_NAMES = ("coarse-shape", "inventory-check", "design-record", "cheap-proxy")
FIELDS = (
    "n_per_cell", "confidence_radius", "scenario", "expected_loss",
    "expected_decision_regret", "expected_query_cost", "expected_query_count",
    "worst_case_box_regret", "worst_case_box_expected_cost",
    "worst_case_box_expected_queries", "planned_first_query",
)


def p_obs(matrix, query, hypothesis, outcome):
    p = matrix[query][hypothesis]
    return p if outcome else 1.0 - p


def normalize(values):
    total = sum(values)
    return tuple(v / total for v in values) if total > 0 else PRIOR


def posterior(belief, query, outcome):
    return normalize([belief[h] * p_obs(NOMINAL, query, h, outcome) for h in range(3)])


def action_risk(belief, action):
    return sum(belief[h] * LOSS[h][action] for h in range(3))


def best_action(belief):
    return min(range(3), key=lambda a: (action_risk(belief, a), a))


def enumerate_trees(belief, mask, rounds):
    trees = [None]
    if rounds == 0 or mask == 0:
        return trees
    for query in range(4):
        if not mask & (1 << query):
            continue
        remainder = mask & ~(1 << query)
        zero = enumerate_trees(posterior(belief, query, False), remainder, rounds - 1)
        one = enumerate_trees(posterior(belief, query, True), remainder, rounds - 1)
        for left in zero:
            for right in one:
                trees.append((query, left, right))
    return trees


def intervals(n):
    radius = math.sqrt(math.log(2 * CELLS / ALPHA) / (2 * n))
    result = []
    for query in range(4):
        row = []
        for hypothesis in range(3):
            successes = round(NOMINAL[query][hypothesis] * n)
            estimate = successes / n
            row.append((max(0., estimate - radius), min(1., estimate + radius)))
        result.append(row)
    return result, radius


def evaluate_conditional(tree, truth, hypothesis, belief):
    if tree is None:
        action = best_action(belief)
        loss = LOSS[hypothesis][action]
        return (loss, loss - min(LOSS[hypothesis]), 0., 0.)
    query, zero, one = tree
    total = [0., 0., 0., 0.]
    for outcome, child in ((False, zero), (True, one)):
        p = p_obs(truth, query, hypothesis, outcome)
        child_metrics = evaluate_conditional(child, truth, hypothesis, posterior(belief, query, outcome))
        total[0] += p * child_metrics[0]
        total[1] += p * child_metrics[1]
        total[2] += p * (COST[query] + child_metrics[2])
        total[3] += p * (1. + child_metrics[3])
    return tuple(total)


def evaluate(tree, truth):
    total = [0., 0., 0., 0.]
    for hypothesis, prior_mass in enumerate(PRIOR):
        values = evaluate_conditional(tree, truth, hypothesis, PRIOR)
        for i in range(4):
            total[i] += prior_mass * values[i]
    return tuple(total)


def worst_box(tree, n):
    boxes, _ = intervals(n)
    worst_loss = worst_cost = worst_queries = 0.
    for hypothesis, prior_mass in enumerate(PRIOR):
        row_loss = row_cost = row_queries = -math.inf
        for bits in itertools.product((0, 1), repeat=4):
            matrix = [list(row) for row in NOMINAL]
            for query, bit in enumerate(bits):
                matrix[query][hypothesis] = boxes[query][hypothesis][bit]
            matrix = tuple(tuple(row) for row in matrix)
            metrics = evaluate_conditional(tree, matrix, hypothesis, PRIOR)
            row_loss = max(row_loss, metrics[0])
            row_cost = max(row_cost, metrics[2])
            row_queries = max(row_queries, metrics[3])
        worst_loss += prior_mass * row_loss
        worst_cost += prior_mass * row_cost
        worst_queries += prior_mass * row_queries
    return (worst_loss, worst_loss, worst_cost, worst_queries)


def choose_tree(trees, n):
    best_tree = trees[0]
    best_metrics = worst_box(best_tree, n)
    for tree in trees[1:]:
        metrics = worst_box(tree, n)
        if metrics[0] < best_metrics[0] - EPS or (
            abs(metrics[0] - best_metrics[0]) <= EPS and metrics[2] < best_metrics[2] - EPS
        ):
            best_tree, best_metrics = tree, metrics
    return best_tree, best_metrics


def reference_rows():
    rows = []
    trees = enumerate_trees(PRIOR, 0b1111, 3)
    for n in SAMPLE_SIZES:
        tree, worst = choose_tree(trees, n)
        first = "none" if tree is None else QUERY_NAMES[tree[0]]
        _, radius = intervals(n)
        for scenario, matrix in SCENARIOS:
            metrics = evaluate(tree, matrix)
            rows.append({
                "n_per_cell": str(n),
                "confidence_radius": f"{radius:.12f}",
                "scenario": scenario,
                "expected_loss": f"{metrics[0]:.12f}",
                "expected_decision_regret": f"{metrics[1]:.12f}",
                "expected_query_cost": f"{metrics[2]:.12f}",
                "expected_query_count": f"{metrics[3]:.12f}",
                "worst_case_box_regret": f"{worst[1]:.12f}",
                "worst_case_box_expected_cost": f"{worst[2]:.12f}",
                "worst_case_box_expected_queries": f"{worst[3]:.12f}",
                "planned_first_query": first,
            })
    return rows


def self_test():
    counts = (1, 5, 65, 3137)
    for depth, expected in enumerate(counts):
        actual = len(enumerate_trees(PRIOR, 0b1111, depth))
        if actual != expected:
            raise ValueError(f"tree inventory mismatch at depth {depth}: {actual} != {expected}")
    previous = math.inf
    for n in SAMPLE_SIZES:
        box, radius = intervals(n)
        if radius >= previous:
            raise ValueError("Hoeffding radius did not shrink with calibration size")
        previous = radius
        for query in range(4):
            for hypothesis in range(3):
                low, high = box[query][hypothesis]
                if not (0 <= low <= NOMINAL[query][hypothesis] <= high <= 1):
                    raise ValueError("nominal parameter not inside frozen calibration interval")
    tree = enumerate_trees(PRIOR, 0b1111, 1)[1]
    factorized = worst_box(tree, 50)
    box, _ = intervals(50)
    full_max = (-math.inf, -math.inf)
    for vertex in range(1 << 12):
        matrix = [list(row) for row in NOMINAL]
        for query in range(4):
            for hypothesis in range(3):
                bit = query * 3 + hypothesis
                matrix[query][hypothesis] = box[query][hypothesis][(vertex >> bit) & 1]
        values = evaluate(tree, tuple(tuple(row) for row in matrix))
        full_max = (max(full_max[0], values[0]), max(full_max[1], values[2]))
    if abs(factorized[0] - full_max[0]) > 1e-9 or abs(factorized[2] - full_max[1]) > 1e-9:
        raise ValueError("factorized 48-row-vertex maximum differs from full 2^12 vertex enumeration")
    print("PASS: tree counts, interval inclusion/width, and exact factorized-vs-full vertex maximum")


def check(path):
    expected = reference_rows()
    with open(path, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"header mismatch: {reader.fieldnames}")
        actual = list(reader)
    if len(actual) != len(expected):
        raise ValueError(f"row count mismatch: expected {len(expected)}, got {len(actual)}")
    numeric = FIELDS[1:2] + FIELDS[3:10]
    categorical = ("n_per_cell", "scenario", "planned_first_query")
    for line, (got, want) in enumerate(zip(actual, expected), 2):
        for field in categorical:
            if got.get(field) != want[field]:
                raise ValueError(f"{field} mismatch on line {line}: {got.get(field)} != {want[field]}")
        for field in numeric:
            try:
                value = float(got[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid {field} on line {line}") from exc
            target = float(want[field])
            if not math.isfinite(value) or abs(value - target) > 5e-9:
                raise ValueError(f"{field} mismatch on line {line}: {value} != {target}")
    print(f"PASS: independently recomputed all {len(expected)} calibrated-box scenario rows")


def main(argv):
    if len(argv) == 2 and argv[1] == "--self-test":
        try:
            self_test()
            return 0
        except (AssertionError, ValueError) as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
    if len(argv) != 2 or argv[1] in ("-h", "--help"):
        print("Usage: reference_check_calibrated_box.py RESULTS.csv")
        print("       reference_check_calibrated_box.py --self-test")
        return 2
    try:
        check(argv[1])
        return 0
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
