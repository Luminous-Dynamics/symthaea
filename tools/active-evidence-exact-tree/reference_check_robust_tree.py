#!/usr/bin/env python3
"""Independent Python reference for src/bin/robust_tree.rs; no third-party packages."""
import csv
import math
import sys

PRIOR = (0.6, 0.3, 0.1)
LOSS = ((0., 4., 7.), (5., 0., 3.), (8., 2., 0.))
COST = (1., 1., 2., 1.)
NOMINAL = ((.88, .34, .25), (.35, .86, .31), (.24, .38, .88), (.76, .70, .18))
Q0_SHIFT = ((.88, .76, .25), (.35, .86, .31), (.24, .38, .88), (.76, .70, .18))
Q1_SHIFT = ((.88, .34, .25), (.83, .22, .31), (.24, .38, .88), (.76, .70, .18))
COMPOUND = ((.88, .74, .25), (.79, .22, .31), (.24, .38, .88), (.76, .83, .18))
OUT_OF_SET = ((.15, .92, .25), (.35, .86, .31), (.24, .38, .88), (.76, .70, .18))
AMBIGUITY = (NOMINAL, Q0_SHIFT, Q1_SHIFT, COMPOUND)
SCENARIOS = (
    ("iid", NOMINAL), ("q0_conditional_shift", Q0_SHIFT),
    ("q1_asymmetric_error", Q1_SHIFT), ("compound_shift", COMPOUND),
    ("out_of_set_flip", OUT_OF_SET),
)
QUERY_NAMES = ("coarse-shape", "inventory-check", "design-record", "cheap-proxy")
EPS = 1e-12
COST_BUDGETS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.632, 2.75, 3.0)
COST_BUDGET_LABELS = ("0", "0.5", "1", "1.5", "2", "2.5", "2.632", "2.75", "3")
FIELDS = ("mode", "limit", "scenario", "expected_loss", "expected_decision_regret",
          "expected_query_cost", "expected_query_count", "worst_case_ambiguity_regret",
          "worst_case_ambiguity_cost", "planned_first_query")


def p_obs(matrix, query, hypothesis, outcome):
    p = matrix[query][hypothesis]
    return p if outcome else 1.0 - p


def normalize(values):
    total = sum(values)
    return tuple(v / total for v in values) if total > 0 else PRIOR


def posterior(belief, matrix, query, outcome):
    return normalize([belief[h] * p_obs(matrix, query, h, outcome) for h in range(3)])


def action_risk(belief, action):
    return sum(belief[h] * LOSS[h][action] for h in range(3))


def best_action(belief):
    best = 0
    best_risk = action_risk(belief, 0)
    for action in range(1, 3):
        candidate = action_risk(belief, action)
        if candidate < best_risk - EPS:
            best = action
            best_risk = candidate
    return best


def predictive(belief, matrix, query, outcome):
    return sum(belief[h] * p_obs(matrix, query, h, outcome) for h in range(3))


def enumerate_plans(nominal_belief, scenario_beliefs, mask, rounds):
    action = best_action(nominal_belief)
    plans = [{
        "risk": tuple(action_risk(scenario_beliefs[s], action) for s in range(4)),
        "cost": (0., 0., 0., 0.),
        "queries": (0., 0., 0., 0.),
        "tree": None,
    }]
    if rounds == 0 or mask == 0:
        return plans
    available = [q for q in range(4) if mask & (1 << q)]
    for query in available:
        remainder = mask & ~(1 << query)
        nominal_zero = posterior(nominal_belief, NOMINAL, query, False)
        nominal_one = posterior(nominal_belief, NOMINAL, query, True)
        belief_zero, belief_one, prob_zero, prob_one = [], [], [], []
        for s, matrix in enumerate(AMBIGUITY):
            belief_zero.append(posterior(scenario_beliefs[s], matrix, query, False))
            belief_one.append(posterior(scenario_beliefs[s], matrix, query, True))
            prob_zero.append(predictive(scenario_beliefs[s], matrix, query, False))
            prob_one.append(predictive(scenario_beliefs[s], matrix, query, True))
        zero_plans = enumerate_plans(nominal_zero, tuple(belief_zero), remainder, rounds-1)
        one_plans = enumerate_plans(nominal_one, tuple(belief_one), remainder, rounds-1)
        for zero_plan in zero_plans:
            for one_plan in one_plans:
                risk = tuple(prob_zero[s] * zero_plan["risk"][s]
                             + prob_one[s] * one_plan["risk"][s] for s in range(4))
                cost = tuple(COST[query] + prob_zero[s] * zero_plan["cost"][s]
                             + prob_one[s] * one_plan["cost"][s] for s in range(4))
                queries = tuple(1. + prob_zero[s] * zero_plan["queries"][s]
                                + prob_one[s] * one_plan["queries"][s] for s in range(4))
                plans.append({
                    "risk": risk, "cost": cost, "queries": queries,
                    "tree": (query, zero_plan["tree"], one_plan["tree"]),
                })
    return plans


def is_better(candidate, incumbent):
    c_risk, i_risk = max(candidate["risk"]), max(incumbent["risk"])
    c_cost, i_cost = max(candidate["cost"]), max(incumbent["cost"])
    return c_risk < i_risk - EPS or (
        abs(c_risk - i_risk) <= EPS and c_cost < i_cost - EPS
    )


def choose_plan(rounds):
    plans = enumerate_plans(PRIOR, tuple(PRIOR for _ in AMBIGUITY), 0b1111, min(rounds, 3))
    best = plans[0]
    for candidate in plans[1:]:
        if is_better(candidate, best):
            best = candidate
    return best


def choose_plan_cost_bounded(rounds, budget):
    plans = enumerate_plans(PRIOR, tuple(PRIOR for _ in AMBIGUITY), 0b1111, min(rounds, 3))
    feasible = [p for p in plans if max(p["cost"]) <= budget + EPS]
    if not feasible:
        raise ValueError("no policy tree meets expected-cost budget")
    best = feasible[0]
    for candidate in feasible[1:]:
        if is_better(candidate, best):
            best = candidate
    return best


def terminal(h, belief):
    action = best_action(belief)
    loss = LOSS[h][action]
    return (loss, loss - min(LOSS[h]), 0., 0.)


def evaluate_tree(tree, truth, h, belief):
    if tree is None:
        return terminal(h, belief)
    query, zero_tree, one_tree = tree
    total = [0., 0., 0., 0.]
    for outcome, child in ((False, zero_tree), (True, one_tree)):
        probability = p_obs(truth, query, h, outcome)
        metrics = evaluate_tree(child, truth, h, posterior(belief, NOMINAL, query, outcome))
        total[0] += probability * metrics[0]
        total[1] += probability * metrics[1]
        total[2] += probability * (COST[query] + metrics[2])
        total[3] += probability * (1. + metrics[3])
    return tuple(total)


def evaluate_plan(tree, truth):
    total = [0., 0., 0., 0.]
    for h, mass in enumerate(PRIOR):
        metrics = evaluate_tree(tree, truth, h, PRIOR)
        for i in range(4):
            total[i] += mass * metrics[i]
    return total


def rows_reference():
    rows = []

    def append_plan(mode, limit, plan):
        first = "none" if plan["tree"] is None else QUERY_NAMES[plan["tree"][0]]
        worst_regret = max(plan["risk"])
        worst_cost = max(plan["cost"])
        for scenario, matrix in SCENARIOS:
            metrics = evaluate_plan(plan["tree"], matrix)
            rows.append({
                "mode": mode,
                "limit": limit,
                "scenario": scenario,
                "expected_loss": f"{metrics[0]:.12f}",
                "expected_decision_regret": f"{metrics[1]:.12f}",
                "expected_query_cost": f"{metrics[2]:.12f}",
                "expected_query_count": f"{metrics[3]:.12f}",
                "worst_case_ambiguity_regret": f"{worst_regret:.12f}",
                "worst_case_ambiguity_cost": f"{worst_cost:.12f}",
                "planned_first_query": first,
            })

    for cap in range(4):
        append_plan("query_cap", str(cap), choose_plan(cap))
    for budget, label in zip(COST_BUDGETS, COST_BUDGET_LABELS):
        append_plan("expected_cost_budget", label, choose_plan_cost_bounded(3, budget))
    return rows


def check(path):
    expected = rows_reference()
    with open(path, newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != FIELDS:
            raise ValueError(f"header mismatch: got {reader.fieldnames}")
        observed = list(reader)
    if len(observed) != len(expected):
        raise ValueError(f"row count mismatch: expected {len(expected)}, got {len(observed)}")
    numeric = ("expected_loss", "expected_decision_regret", "expected_query_cost",
               "expected_query_count", "worst_case_ambiguity_regret", "worst_case_ambiguity_cost")
    categorical = ("mode", "limit", "scenario", "planned_first_query")
    for line, (actual, reference) in enumerate(zip(observed, expected), 2):
        for field in categorical:
            if actual.get(field) != reference[field]:
                raise ValueError(f"{field} mismatch on line {line}: {actual.get(field)} vs {reference[field]}")
        for field in numeric:
            try:
                value = float(actual[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid {field} on line {line}") from exc
            target = float(reference[field])
            if not math.isfinite(value) or abs(value - target) > 5e-9:
                raise ValueError(f"{field} mismatch on line {line}: {value} vs {target}")
    print(f"PASS: independently recomputed {len(expected)} robust-tree rows and all numeric metrics")


def self_test():
    expected_counts = (1, 5, 65, 3137)
    for cap, count in enumerate(expected_counts):
        plans = enumerate_plans(PRIOR, tuple(PRIOR for _ in AMBIGUITY), 0b1111, cap)
        assert len(plans) == count, (cap, len(plans))
    plan = choose_plan(3)
    assert abs(max(plan["risk"]) - 1.6090232) < 1e-8
    iid = evaluate_plan(plan["tree"], NOMINAL)
    assert abs(iid[1] - 1.0751888) < 1e-8
    assert abs(iid[2] - 2.632) < 1e-8
    low = choose_plan_cost_bounded(3, 1.5)
    assert abs(max(low["risk"]) - 1.735898) < 1e-8
    assert abs(max(low["cost"]) - 1.48318) < 1e-8
    frontier = choose_plan_cost_bounded(3, 2.632)
    assert abs(max(frontier["risk"]) - 1.6090232) < 1e-8
    assert abs(max(frontier["cost"]) - 2.632) < 1e-8
    previous = math.inf
    for budget in COST_BUDGETS:
        candidate = choose_plan_cost_bounded(3, budget)
        assert max(candidate["cost"]) <= budget + 1e-8
        current = max(candidate["risk"])
        assert current <= previous + 1e-8
        previous = current
    out_of_set = evaluate_plan(plan["tree"], OUT_OF_SET)
    assert out_of_set[1] > 2.3
    print("PASS: tree counts, IID improvement, cost-constrained frontier, monotonicity and out-of-set negative control")


def main(argv):
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    if len(argv) != 2 or argv[1] in ("-h", "--help"):
        print("Usage: reference_check_robust_tree.py CANDIDATE.csv")
        print("       reference_check_robust_tree.py --self-test")
        return 2
    try:
        check(argv[1])
        return 0
    except (OSError, UnicodeError, csv.Error, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
