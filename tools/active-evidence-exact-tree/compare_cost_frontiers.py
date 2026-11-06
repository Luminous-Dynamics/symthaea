#!/usr/bin/env python3
"""Cost-frontier comparison for the exact-tree benchmark; standard library only."""
import csv
import math
import sys

AMBIGUITY_SCENARIOS = (
    "iid", "q0_conditional_shift", "q1_asymmetric_error", "compound_shift"
)
ALL_SCENARIOS = AMBIGUITY_SCENARIOS + ("out_of_set_flip",)
BASELINE_POLICIES = ("random", "entropy", "plugin_evsi", "robust_evsi")
BUDGETS = ("0", "0.5", "1", "1.5", "2", "2.5", "2.632", "2.75", "3")
GREEDY_FIELDS = (
    "scenario", "policy", "query_cap", "expected_loss", "expected_decision_regret",
    "expected_query_cost", "expected_query_count"
)
TREE_FIELDS = (
    "mode", "limit", "scenario", "expected_loss", "expected_decision_regret",
    "expected_query_cost", "expected_query_count", "worst_case_ambiguity_regret",
    "worst_case_ambiguity_cost", "planned_first_query"
)
OUTPUT_FIELDS = (
    "expected_cost_budget", "best_greedy_policy", "best_greedy_query_cap",
    "greedy_worst_case_regret", "greedy_worst_case_expected_cost",
    "tree_worst_case_regret", "tree_worst_case_expected_cost",
    "absolute_regret_improvement", "relative_regret_improvement"
)
EPS = 1e-8


def read_csv(path, expected_fields):
    try:
        with open(path, newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != expected_fields:
                raise ValueError(f"{path}: header mismatch: {reader.fieldnames}")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise ValueError(f"{path}: cannot read CSV: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: empty CSV")
    return rows


def index_unique(rows, key_fields, source):
    result = {}
    for line_number, row in enumerate(rows, 2):
        key = tuple(row[field] for field in key_fields)
        if key in result:
            raise ValueError(f"{source}: duplicate key at line {line_number}: {key}")
        result[key] = row
    return result


def number(row, field, where):
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{where}: invalid numeric field {field}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{where}: non-finite {field}")
    return value


def load_greedy(path):
    rows = read_csv(path, GREEDY_FIELDS)
    indexed = index_unique(rows, ("scenario", "policy", "query_cap"), path)
    expected_keys = {
        (scenario, policy, str(cap))
        for scenario in ALL_SCENARIOS
        for policy in BASELINE_POLICIES + ("known_channel_evsi_reference",)
        for cap in range(4)
    }
    if set(indexed) != expected_keys:
        raise ValueError(
            f"{path}: required greedy key inventory mismatch; "
            f"missing={sorted(expected_keys - set(indexed))[:5]}, "
            f"extra={sorted(set(indexed) - expected_keys)[:5]}"
        )
    return indexed


def load_tree(path):
    rows = read_csv(path, TREE_FIELDS)
    indexed = index_unique(rows, ("mode", "limit", "scenario"), path)
    expected_keys = {
        ("query_cap", str(cap), scenario)
        for cap in range(4) for scenario in ALL_SCENARIOS
    } | {
        ("expected_cost_budget", budget, scenario)
        for budget in BUDGETS for scenario in ALL_SCENARIOS
    }
    if set(indexed) != expected_keys:
        raise ValueError(
            f"{path}: required tree key inventory mismatch; "
            f"missing={sorted(expected_keys - set(indexed))[:5]}, "
            f"extra={sorted(set(indexed) - expected_keys)[:5]}"
        )
    return indexed


def compare(greedy, tree):
    output = []
    for budget in BUDGETS:
        feasible = []
        for policy_order, policy in enumerate(BASELINE_POLICIES):
            for cap in range(4):
                selected = [greedy[(s, policy, str(cap))] for s in AMBIGUITY_SCENARIOS]
                max_cost = max(number(r, "expected_query_cost", (s, policy, cap))
                               for r, s in zip(selected, AMBIGUITY_SCENARIOS))
                worst_regret = max(number(r, "expected_decision_regret", (s, policy, cap))
                                   for r, s in zip(selected, AMBIGUITY_SCENARIOS))
                if max_cost <= float(budget) + EPS:
                    feasible.append((
                        worst_regret, max_cost, policy_order, cap, policy
                    ))
        if not feasible:
            raise ValueError(f"no greedy baseline feasible at budget {budget}")
        greedy_best = min(feasible)

        tree_rows = [tree[("expected_cost_budget", budget, scenario)]
                     for scenario in AMBIGUITY_SCENARIOS]
        tree_regret = max(number(r, "expected_decision_regret", (budget, r["scenario"]))
                          for r in tree_rows)
        tree_cost = max(number(r, "expected_query_cost", (budget, r["scenario"]))
                        for r in tree_rows)
        declared_risk = {round(number(r, "worst_case_ambiguity_regret", budget), 8)
                         for r in tree_rows}
        declared_cost = {round(number(r, "worst_case_ambiguity_cost", budget), 8)
                         for r in tree_rows}
        if len(declared_risk) != 1 or abs(next(iter(declared_risk)) - tree_regret) > EPS:
            raise ValueError(f"tree risk receipt mismatch for budget {budget}")
        if len(declared_cost) != 1 or abs(next(iter(declared_cost)) - tree_cost) > EPS:
            raise ValueError(f"tree cost receipt mismatch for budget {budget}")
        if tree_cost > float(budget) + EPS:
            raise ValueError(f"tree exceeded expected-cost budget {budget}: {tree_cost}")

        greedy_regret, greedy_cost, _, cap, policy = greedy_best
        improvement = greedy_regret - tree_regret
        relative = improvement / greedy_regret if greedy_regret > 0 else 0.0
        output.append({
            "expected_cost_budget": budget,
            "best_greedy_policy": policy,
            "best_greedy_query_cap": str(cap),
            "greedy_worst_case_regret": f"{greedy_regret:.12f}",
            "greedy_worst_case_expected_cost": f"{greedy_cost:.12f}",
            "tree_worst_case_regret": f"{tree_regret:.12f}",
            "tree_worst_case_expected_cost": f"{tree_cost:.12f}",
            "absolute_regret_improvement": f"{improvement:.12f}",
            "relative_regret_improvement": f"{relative:.12f}",
        })
    return output


def self_check(rows):
    at_15 = next(row for row in rows if row["expected_cost_budget"] == "1.5")
    at_2632 = next(row for row in rows if row["expected_cost_budget"] == "2.632")
    at_3 = next(row for row in rows if row["expected_cost_budget"] == "3")
    checks = (
        (float(at_15["greedy_worst_case_regret"]), 1.778),
        (float(at_15["tree_worst_case_regret"]), 1.735898),
        (float(at_2632["greedy_worst_case_regret"]), 1.778),
        (float(at_2632["tree_worst_case_regret"]), 1.6090232),
        (float(at_2632["tree_worst_case_expected_cost"]), 2.632),
        (float(at_3["greedy_worst_case_regret"]), 1.735898),
        (float(at_3["tree_worst_case_regret"]), 1.6090232),
    )
    for actual, expected in checks:
        if abs(actual - expected) > EPS:
            raise ValueError(f"frozen frontier witness mismatch: {actual} != {expected}")


def main(argv):
    if len(argv) != 3 or argv[1] in ("-h", "--help"):
        print("Usage: compare_cost_frontiers.py GREEDY.csv ROBUST_TREE.csv")
        return 2
    try:
        greedy = load_greedy(argv[1])
        tree = load_tree(argv[2])
        result = compare(greedy, tree)
        self_check(result)
        writer = csv.DictWriter(sys.stdout, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(result)
        print("PASS: matched expected-cost frontier and frozen witness controls", file=sys.stderr)
        return 0
    except (OSError, UnicodeError, csv.Error, ValueError, KeyError, StopIteration) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
