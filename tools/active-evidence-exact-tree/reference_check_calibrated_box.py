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
SAMPLE_SIZES = (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
COVERAGE_GRID = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
METHODS = (("fixed_sample", False), ("time_uniform", True))
ALPHA = .05
CELLS = 12
EPS = 1e-12
QUERY_NAMES = ("coarse-shape", "inventory-check", "design-record", "cheap-proxy")
RESULT_FIELDS = (
    "interval_method", "n_per_cell", "confidence_radius", "scenario", "expected_loss",
    "expected_decision_regret", "expected_query_cost", "expected_query_count",
    "worst_case_box_regret", "worst_case_box_expected_cost",
    "worst_case_box_expected_queries", "planned_first_query",
)
COVERAGE_FIELDS = (
    "interval_method", "n_per_cell", "true_probability", "cell_coverage",
    "minimum_grid_cell_coverage", "union_bound_lower_12_cells", "per_cell_target",
)


def p_obs(matrix, query, hypothesis, outcome):
    probability = matrix[query][hypothesis]
    return probability if outcome else 1.0 - probability


def normalize(values):
    total = sum(values)
    return tuple(value / total for value in values) if total > 0.0 else PRIOR


def posterior(belief, query, outcome):
    return normalize([
        belief[hypothesis] * p_obs(NOMINAL, query, hypothesis, outcome)
        for hypothesis in range(3)
    ])


def action_risk(belief, action):
    return sum(belief[hypothesis] * LOSS[hypothesis][action] for hypothesis in range(3))


def best_action(belief):
    best = 0
    best_risk = action_risk(belief, 0)
    for action in range(1, 3):
        candidate = action_risk(belief, action)
        if candidate < best_risk - EPS:
            best = action
            best_risk = candidate
    return best


def enumerate_trees(belief, mask, rounds):
    trees = [None]
    if rounds == 0 or mask == 0:
        return trees
    for query in range(4):
        if not mask & (1 << query):
            continue
        remainder = mask & ~(1 << query)
        zero_trees = enumerate_trees(posterior(belief, query, False), remainder, rounds - 1)
        one_trees = enumerate_trees(posterior(belief, query, True), remainder, rounds - 1)
        for zero in zero_trees:
            for one in one_trees:
                trees.append((query, zero, one))
    return trees


def interval_radius(n, anytime=False):
    if n < 1:
        raise ValueError("calibration sample count must be positive")
    log_term = (
        math.log(2 * CELLS * n * (n + 1) / ALPHA)
        if anytime else math.log(2 * CELLS / ALPHA)
    )
    return math.sqrt(log_term / (2 * n))


def intervals(n, anytime=False):
    radius = interval_radius(n, anytime)
    result = []
    for query in range(4):
        row = []
        for hypothesis in range(3):
            successes = round(NOMINAL[query][hypothesis] * n)  # ties-to-even, matching Rust
            estimate = successes / n
            row.append((max(0.0, estimate - radius), min(1.0, estimate + radius)))
        result.append(row)
    return result, radius


def binomial_pmf(n, probability):
    mass = [0.0] * (n + 1)
    if probability <= 0.0:
        mass[0] = 1.0
        return mass
    if probability >= 1.0:
        mass[n] = 1.0
        return mass
    mode = min(n, math.floor((n + 1) * probability))
    mass[mode] = 1.0
    for k in range(mode, 0, -1):
        mass[k - 1] = mass[k] * (k / (n - k + 1)) * ((1.0 - probability) / probability)
    for k in range(mode, n):
        mass[k + 1] = mass[k] * ((n - k) / (k + 1)) * (probability / (1.0 - probability))
    total = sum(mass)
    return [value / total for value in mass]


def cell_coverage(n, true_probability, anytime=False):
    radius = interval_radius(n, anytime)
    return sum(
        mass
        for successes, mass in enumerate(binomial_pmf(n, true_probability))
        if max(0.0, successes / n - radius) <= true_probability
        <= min(1.0, successes / n + radius)
    )


def evaluate_conditional(tree, truth, hypothesis, belief):
    if tree is None:
        action = best_action(belief)
        loss = LOSS[hypothesis][action]
        return (loss, loss - min(LOSS[hypothesis]), 0.0, 0.0)
    query, zero_tree, one_tree = tree
    total = [0.0, 0.0, 0.0, 0.0]
    for outcome, child in ((False, zero_tree), (True, one_tree)):
        probability = p_obs(truth, query, hypothesis, outcome)
        child_metrics = evaluate_conditional(
            child, truth, hypothesis, posterior(belief, query, outcome)
        )
        total[0] += probability * child_metrics[0]
        total[1] += probability * child_metrics[1]
        total[2] += probability * (COST[query] + child_metrics[2])
        total[3] += probability * (1.0 + child_metrics[3])
    return tuple(total)


def evaluate(tree, truth):
    total = [0.0, 0.0, 0.0, 0.0]
    for hypothesis, prior_mass in enumerate(PRIOR):
        conditional = evaluate_conditional(tree, truth, hypothesis, PRIOR)
        for index in range(4):
            total[index] += prior_mass * conditional[index]
    return tuple(total)


def worst_box(tree, n, anytime=False):
    boxes, _ = intervals(n, anytime)
    worst_loss = worst_cost = worst_queries = 0.0
    for hypothesis, prior_mass in enumerate(PRIOR):
        row_loss = row_cost = row_queries = -math.inf
        for bits in itertools.product((0, 1), repeat=4):
            matrix = [list(row) for row in NOMINAL]
            for query, endpoint in enumerate(bits):
                matrix[query][hypothesis] = boxes[query][hypothesis][endpoint]
            matrix = tuple(tuple(row) for row in matrix)
            metrics = evaluate_conditional(tree, matrix, hypothesis, PRIOR)
            row_loss = max(row_loss, metrics[0])
            row_cost = max(row_cost, metrics[2])
            row_queries = max(row_queries, metrics[3])
        worst_loss += prior_mass * row_loss
        worst_cost += prior_mass * row_cost
        worst_queries += prior_mass * row_queries
    return (worst_loss, worst_loss, worst_cost, worst_queries)


def choose_tree(trees, n, anytime=False):
    best_tree = trees[0]
    best_metrics = worst_box(best_tree, n, anytime)
    for tree in trees[1:]:
        metrics = worst_box(tree, n, anytime)
        if metrics[0] < best_metrics[0] - EPS or (
            abs(metrics[0] - best_metrics[0]) <= EPS and metrics[2] < best_metrics[2] - EPS
        ):
            best_tree, best_metrics = tree, metrics
    return best_tree, best_metrics


def result_rows():
    rows = []
    trees = enumerate_trees(PRIOR, 0b1111, 3)
    for method, anytime in METHODS:
        for n in SAMPLE_SIZES:
            tree, worst = choose_tree(trees, n, anytime)
            first = "none" if tree is None else QUERY_NAMES[tree[0]]
            _, radius = intervals(n, anytime)
            for scenario, matrix in SCENARIOS:
                metrics = evaluate(tree, matrix)
                rows.append({
                    "interval_method": method,
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


def coverage_rows():
    rows = []
    for method, anytime in METHODS:
        for n in SAMPLE_SIZES:
            by_probability = [(p, cell_coverage(n, p, anytime)) for p in COVERAGE_GRID]
            minimum = min(coverage for _, coverage in by_probability)
            union_bound = 1.0 - CELLS * (1.0 - minimum)
            target = 1.0 - ALPHA / CELLS
            for true_probability, coverage in by_probability:
                rows.append({
                    "interval_method": method,
                    "n_per_cell": str(n),
                    "true_probability": f"{true_probability:.12f}",
                    "cell_coverage": f"{coverage:.12f}",
                    "minimum_grid_cell_coverage": f"{minimum:.12f}",
                    "union_bound_lower_12_cells": f"{union_bound:.12f}",
                    "per_cell_target": f"{target:.12f}",
                })
    return rows


def read_csv(path, fields):
    try:
        with open(path, newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            if tuple(reader.fieldnames or ()) != fields:
                raise ValueError(f"{path}: header mismatch: {reader.fieldnames}")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"{path}: unable to read CSV: {exc}") from exc
    return rows


def compare_rows(path, fields, expected, key_fields, numeric_fields, description):
    actual = read_csv(path, fields)
    if len(actual) != len(expected):
        raise ValueError(f"{description}: row count mismatch, expected {len(expected)}, got {len(actual)}")
    expected_keys = [tuple(row[key] for key in key_fields) for row in expected]
    actual_keys = [tuple(row[key] for key in key_fields) for row in actual]
    if actual_keys != expected_keys:
        raise ValueError(f"{description}: missing, duplicate, extra, or reordered key inventory")
    for line, (got, want) in enumerate(zip(actual, expected), 2):
        for field in key_fields:
            if got.get(field) != want[field]:
                raise ValueError(f"{description}: {field} mismatch on line {line}")
        for field in numeric_fields:
            try:
                value = float(got[field])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{description}: invalid {field} on line {line}") from exc
            target = float(want[field])
            if not math.isfinite(value) or abs(value - target) > 5e-9:
                raise ValueError(
                    f"{description}: {field} mismatch on line {line}: {value} != {target}"
                )


def check_results_csv(path):
    compare_rows(
        path,
        RESULT_FIELDS,
        result_rows(),
        ("interval_method", "n_per_cell", "scenario", "planned_first_query"),
        ("confidence_radius", "expected_loss", "expected_decision_regret", "expected_query_cost",
         "expected_query_count", "worst_case_box_regret", "worst_case_box_expected_cost",
         "worst_case_box_expected_queries"),
        "calibrated-box results",
    )
    print("PASS: independently recomputed fixed-sample and time-uniform results")


def check_coverage_grid(path):
    compare_rows(
        path,
        COVERAGE_FIELDS,
        coverage_rows(),
        ("interval_method", "n_per_cell", "true_probability"),
        ("cell_coverage", "minimum_grid_cell_coverage", "union_bound_lower_12_cells",
         "per_cell_target"),
        "calibrated-box coverage grid",
    )
    print("PASS: independently recomputed fixed-sample and time-uniform finite-grid coverage")


def self_test():
    expected_counts = (1, 5, 65, 3137)
    for depth, expected in enumerate(expected_counts):
        actual = len(enumerate_trees(PRIOR, 0b1111, depth))
        if actual != expected:
            raise ValueError(f"tree inventory mismatch at depth {depth}: {actual} != {expected}")

    target = 1.0 - ALPHA / CELLS
    for method, anytime in METHODS:
        previous = math.inf
        for n in SAMPLE_SIZES:
            boxes, radius = intervals(n, anytime)
            if radius >= previous:
                raise ValueError(f"{method}: radius did not shrink with calibration size")
            previous = radius
            for query in range(4):
                for hypothesis in range(3):
                    low, high = boxes[query][hypothesis]
                    if not (0 <= low <= NOMINAL[query][hypothesis] <= high <= 1):
                        raise ValueError(f"{method}: interval excludes nominal fixture")
            minimum_coverage = min(cell_coverage(n, p, anytime) for p in COVERAGE_GRID)
            if minimum_coverage + 1e-10 < target:
                raise ValueError(f"{method}, n={n}: grid coverage {minimum_coverage} below {target}")
            union_bound = 1.0 - CELLS * (1.0 - minimum_coverage)
            if union_bound + 1e-10 < 1.0 - ALPHA:
                raise ValueError(f"{method}, n={n}: grid union-bound below 95%")
    for n in SAMPLE_SIZES:
        if interval_radius(n, True) < interval_radius(n, False):
            raise ValueError("time-uniform interval narrower than fixed-sample interval")

    tree = enumerate_trees(PRIOR, 0b1111, 1)[1]
    factorized = worst_box(tree, 50, False)
    boxes, _ = intervals(50, False)
    full_max_loss = full_max_cost = -math.inf
    for vertex in range(1 << 12):
        matrix = [list(row) for row in NOMINAL]
        for query in range(4):
            for hypothesis in range(3):
                bit = query * 3 + hypothesis
                matrix[query][hypothesis] = boxes[query][hypothesis][(vertex >> bit) & 1]
        values = evaluate(tree, tuple(tuple(row) for row in matrix))
        full_max_loss = max(full_max_loss, values[0])
        full_max_cost = max(full_max_cost, values[2])
    if abs(factorized[0] - full_max_loss) > 1e-9 or abs(factorized[2] - full_max_cost) > 1e-9:
        raise ValueError("factorized 48-row-vertex maximum differs from full 2^12 vertex enumeration")
    print("PASS: tree counts, fixed/anytime interval coverage and monotonicity, and factorization check")


def main(argv):
    if len(argv) == 2 and argv[1] == "--self-test":
        try:
            self_test()
            return 0
        except (AssertionError, ValueError) as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
    if len(argv) == 3 and argv[1] == "--check-coverage-grid":
        try:
            check_coverage_grid(argv[2])
            return 0
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            return 1
    if len(argv) != 2 or argv[1] in ("-h", "--help"):
        print("Usage: reference_check_calibrated_box.py RESULTS.csv")
        print("       reference_check_calibrated_box.py --check-coverage-grid COVERAGE.csv")
        print("       reference_check_calibrated_box.py --self-test")
        return 2
    try:
        check_results_csv(argv[1])
        return 0
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
