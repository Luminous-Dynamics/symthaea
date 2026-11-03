#!/usr/bin/env python3
"""Deployable graph/cardinality-only selector over a stochastic min-fill pool.

Unlike the oracle-best-of-pool score in elimination_order_planner.py, this
selector chooses a candidate using only factor scopes and variable
cardinalities, before evaluating any coefficient algebra or exact result.
It then runs the chosen order through each exact algebra and independent
certificate checker. This is still a tiny-synthetic reference experiment.
"""
from __future__ import annotations
import json
import random
from pathlib import Path

from elimination_order_planner import (
    CARDS, SCENARIOS, greedy_order, make_problem, order_metrics,
    direct_enumeration, build_certificate, verify_certificate,
)

HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(32))
KINDS = ("maxcount", "pareto", "poly")


def structural_cost(cards, scopes, order):
    """Return graph-only sum/peak of generated bucket assignment counts.

    Each step joins all current scopes containing the eliminated variable.
    The score does not inspect algebra values, polynomial support, Pareto
    frontiers, coefficient bits, or certificate bytes.
    """
    current = [list(scope) for scope in scopes]
    total = 0
    peak = 0
    entries_total = 0
    for variable in order:
        bucket = [scope for scope in current if variable in scope]
        joined = sorted({v for scope in bucket for v in scope} - {variable})
        output_entries = 1
        for v in joined:
            output_entries *= cards[v]
        bucket_assignments = output_entries * cards[variable] * max(1, len(bucket))
        total += bucket_assignments
        peak = max(peak, bucket_assignments)
        entries_total += output_entries
        current = [scope for scope in current if variable not in scope]
        if joined:
            current.append(joined)
    return total, peak, entries_total


def regret(value, optimum):
    return 100.0 * (value - optimum) / max(1, optimum)


def run():
    exhaustive = json.loads((HERE / "elimination_order_planner_results.json").read_text())
    rows = []
    all_exact = True
    for graph_name, scopes in SCENARIOS.items():
        candidates = [
            greedy_order(CARDS, scopes, "min_fill", rng=random.Random(seed))
            for seed in SEEDS
        ]
        ranked = sorted(
            ((structural_cost(CARDS, scopes, order), tuple(order)) for order in candidates),
            key=lambda item: (item[0], item[1]),
        )
        static_score, selected_tuple = ranked[0]
        selected_order = list(selected_tuple)
        min_fill_order = greedy_order(CARDS, scopes, "min_fill")
        min_bucket_order = greedy_order(CARDS, scopes, "min_bucket_work")
        scenario_source = exhaustive["scenarios"][graph_name]
        for kind in KINDS:
            factors, algebra = make_problem(kind, scopes)
            metrics, result = order_metrics(CARDS, factors, algebra, kind, selected_order)
            oracle = direct_enumeration(CARDS, factors, algebra)
            reference = scenario_source["algebras"][kind]
            expected_best = reference["best_by_metric"]["algebra_work_units_proxy"]
            optimum = expected_best["algebra_work_units_proxy"]
            baseline = reference["heuristics"]["min_fill"]["algebra_work_units_proxy"]
            min_bucket_baseline = reference["heuristics"]["min_bucket_work"]["algebra_work_units_proxy"]
            all_exact = all_exact and result == algebra.encode(oracle)
            certificate = build_certificate(CARDS, factors, algebra, selected_order)
            replay = verify_certificate(CARDS, factors, algebra, certificate)
            if replay["result"] != algebra.encode(oracle):
                raise AssertionError(f"selected structural order certificate failed: {graph_name}/{kind}")
            rows.append({
                "graph": graph_name,
                "algebra": kind,
                "selected_order": selected_order,
                "structural_score": list(static_score),
                "candidate_pool_size": len(candidates),
                "candidate_unique_orders": len(set(tuple(order) for order in candidates)),
                "selected_work_proxy": metrics["algebra_work_units_proxy"],
                "exhaustive_best_work_proxy": optimum,
                "selected_regret_percent": round(regret(metrics["algebra_work_units_proxy"], optimum), 6),
                "min_fill_order": min_fill_order,
                "min_fill_work_proxy": baseline,
                "min_fill_regret_percent": round(regret(baseline, optimum), 6),
                "min_bucket_work_order": min_bucket_order,
                "min_bucket_work_proxy": min_bucket_baseline,
                "min_bucket_work_regret_percent": round(regret(min_bucket_baseline, optimum), 6),
                "selected_certificate_bytes": len(json.dumps(certificate, sort_keys=True, separators=(",", ":")).encode("utf-8")),
                "checker_replay_passed": True,
                "direct_oracle_match": True,
            })

    regret_values = [row["selected_regret_percent"] for row in rows]
    sorted_regrets = sorted(regret_values)
    median = (sorted_regrets[14] + sorted_regrets[15]) / 2
    summary = {
        "suite": "Graph-only static order selector over 32 stochastic min-fill candidates",
        "seed_stream": list(SEEDS),
        "graphs": len(SCENARIOS),
        "variables_per_graph": len(CARDS),
        "variable_cardinalities": CARDS,
        "candidate_orders_per_graph": len(SEEDS),
        "exact_selected_order_algebra_executions": len(rows),
        "all_selected_orders_match_direct_oracle": all_exact,
        "independent_certificate_replays": sum(1 for row in rows if row["checker_replay_passed"]),
        "selected_order_regret_percent": {
            "median": round(median, 6),
            "mean": round(sum(regret_values) / len(regret_values), 6),
            "max": round(max(regret_values), 6),
            "exact_best_cases": sum(v == 0 for v in regret_values),
        },
        "claim_boundary": "Graph/cardinality-only ranking; exact selected orders and certificates were evaluated in Python. Tiny synthetic instances only; not hardware runtime or SOTA evidence.",
        "cases": rows,
    }
    path = HERE / "elimination_order_static_portfolio_results.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"static portfolio: {summary['exact_selected_order_algebra_executions']} selected order/algebra executions")
    print(f"direct oracle + certificate: {summary['independent_certificate_replays']}/{len(rows)}")
    print("regret median/mean/max:", summary["selected_order_regret_percent"])
    for graph in SCENARIOS:
        orders = {r["algebra"]: r["selected_order"] for r in rows if r["graph"] == graph}
        print(graph, orders)
    return summary


if __name__ == "__main__":
    run()
