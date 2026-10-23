#!/usr/bin/env python3
"""Prior-shift stress test for objective-matched exact causal-probing oracles.

Uses the 16-hypothesis exact game and its Bellman/policy implementation, but
varies graph beliefs, intervention-kernel beliefs, and graph-kernel dependence.
Every policy is scored against the oracle for the same terminal entropy objective.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
from objective_matched_oracle_benchmark import (
    HYPOTHESES, OBJECTIVES, POLICIES, H0, policy_value, oracle_value,
    objective_entropy,
)

SCHEMA = "symthaea-prior-shift-objective-matched-oracle-v1"
BASE_SOURCE_BLOB = "f101346c2f37cfad4e029ae280e9a002548f309c"
PRIOR_NAMES = (
    "uniform", "chain_biased", "fork_biased", "low_spill_biased",
    "high_spill_biased", "graph_kernel_correlated",
)
HORIZONS = (1, 2, 3)

def build_prior(name):
    weights = []
    for hyp in HYPOTHESES:
        graph, *spill_bits = hyp
        if name == "uniform":
            p = 1.0 / 16
        elif name == "chain_biased":
            p = (0.85 if graph == 0 else 0.15) / 8
        elif name == "fork_biased":
            p = (0.15 if graph == 0 else 0.85) / 8
        elif name == "low_spill_biased":
            p_graph = 0.5
            p = p_graph * math.prod(0.85 if bit == 0 else 0.15 for bit in spill_bits)
        elif name == "high_spill_biased":
            p_graph = 0.5
            p = p_graph * math.prod(0.15 if bit == 0 else 0.85 for bit in spill_bits)
        elif name == "graph_kernel_correlated":
            # Chain hypotheses favor low-spill kernels; fork hypotheses favor high-spill kernels.
            prob_high = 0.2 if graph == 0 else 0.8
            p = 0.5 * math.prod(prob_high if bit == 1 else 1.0 - prob_high for bit in spill_bits)
        else:
            raise ValueError(f"unknown prior {name}")
        weights.append(p)
    total = sum(weights)
    if total <= 0 or any(p <= 0 for p in weights):
        raise AssertionError("prior must have positive support on every hypothesis")
    normalized = tuple(p / total for p in weights)
    if not math.isclose(sum(normalized), 1.0, abs_tol=1e-12):
        raise AssertionError("prior does not normalize")
    return normalized

def run_benchmark(max_horizon=3, prior_names=PRIOR_NAMES):
    if max_horizon not in HORIZONS:
        raise ValueError(f"horizon must be in {HORIZONS}")
    rows = []
    for prior_name in prior_names:
        prior = build_prior(prior_name)
        # Independent caches per prior, as posterior states are prior-specific.
        cache = {"oracle": {}}
        initial = {obj: objective_entropy(prior, obj) for obj in OBJECTIVES}
        for horizon in range(1, max_horizon + 1):
            oracle_by_objective = {
                obj: oracle_value(prior, horizon, obj, cache["oracle"])
                for obj in OBJECTIVES
            }
            for policy in POLICIES:
                for scored_objective in OBJECTIVES:
                    terminal = policy_value(prior, horizon, policy, (), scored_objective, cache)
                    regret = terminal - oracle_by_objective[scored_objective]
                    if regret < -1e-8:
                        raise AssertionError(
                            f"{policy} beats {scored_objective} oracle under {prior_name}: {regret}"
                        )
                    rows.append({
                        "prior": prior_name,
                        "horizon": horizon,
                        "budget": horizon,
                        "policy": policy,
                        "policy_objective": (
                            policy.split("_", 1)[1]
                            if policy.startswith(("greedy_", "coverage_", "oracle_"))
                            else None
                        ),
                        "scored_objective": scored_objective,
                        "initial_entropy_bits": initial[scored_objective],
                        "expected_terminal_entropy_bits": terminal,
                        "expected_information_gain_bits": initial[scored_objective] - terminal,
                        "matched_oracle_terminal_entropy_bits": oracle_by_objective[scored_objective],
                        "regret_bits_vs_matched_oracle": max(0.0, regret),
                    })
    aggregates = {}
    for prior_name in prior_names:
        for horizon in range(1, max_horizon + 1):
            for objective in OBJECTIVES:
                group = [r for r in rows if r["prior"] == prior_name and r["horizon"] == horizon and r["scored_objective"] == objective]
                aggregates[f"{prior_name}/h{horizon}/{objective}"] = {
                    "n_policies": len(group),
                    "best_policy_by_terminal_entropy": min(group, key=lambda r: r["expected_terminal_entropy_bits"])["policy"],
                    "max_regret_bits": max(r["regret_bits_vs_matched_oracle"] for r in group),
                    "mean_regret_bits_across_policies": sum(r["regret_bits_vs_matched_oracle"] for r in group) / len(group),
                }
    # Rank stability at horizon 3: determine best policy by each prior/objective and count wins.
    wins = {}
    if max_horizon >= 3:
        for objective in OBJECTIVES:
            for prior_name in prior_names:
                group = [r for r in rows if r["prior"] == prior_name and r["horizon"] == 3 and r["scored_objective"] == objective]
                best = min(r["expected_terminal_entropy_bits"] for r in group)
                winners = sorted(set(r["policy"] for r in group if abs(r["expected_terminal_entropy_bits"] - best) <= 1e-9))
                for policy in winners:
                    wins[f"{objective}/{policy}"] = wins.get(f"{objective}/{policy}", 0) + 1
    return {
        "schema": SCHEMA,
        "status": "EXACT_PRIOR_SHIFT_ORACLE_STRESS_LOCAL_NOT_EXTERNAL_QUALIFICATION",
        "base_source": {"git_blob": BASE_SOURCE_BLOB, "path": "validation/objective_matched_oracle_benchmark.py"},
        "protocol": {
            "hypotheses": len(HYPOTHESES),
            "priors": list(prior_names),
            "prior_semantics": {
                "uniform": "equal weight on all 16 hypotheses",
                "chain_biased": "P(chain)=0.85, P(fork)=0.15; kernel states conditionally uniform",
                "fork_biased": "P(chain)=0.15, P(fork)=0.85; kernel states conditionally uniform",
                "low_spill_biased": "independent per-action P(low spill)=0.85",
                "high_spill_biased": "independent per-action P(high spill)=0.85",
                "graph_kernel_correlated": "P(high spill per action)=0.2 if chain and 0.8 if fork",
            },
            "objectives": list(OBJECTIVES),
            "policies": list(POLICIES),
            "horizons": list(range(1, max_horizon + 1)),
            "rows": len(rows),
            "oracle": "exact Bellman recursion with full posterior tuple cache keys, recomputed per prior",
            "scoring": "each policy is compared with an oracle minimizing the same terminal entropy objective",
            "limitations": [
                "Reduced 16-hypothesis chain/fork game only",
                "Exact realized-subset telemetry assumed",
                "The priors shift belief over a fixed model family, not the graph/mechanism family itself",
                "No external benchmark or published method has been executed in this lane",
            ],
        },
        "horizon_3_policy_wins_by_objective_and_prior": wins,
        "aggregates": aggregates,
        "rows": rows,
    }

def self_test():
    for name in PRIOR_NAMES:
        p = build_prior(name)
        assert len(p) == 16 and all(x > 0 for x in p)
        assert math.isclose(sum(p), 1.0, abs_tol=1e-12)
    result = run_benchmark(2, prior_names=("uniform", "graph_kernel_correlated"))
    assert len(result["rows"]) == 2 * 2 * len(POLICIES) * len(OBJECTIVES)
    assert all(r["regret_bits_vs_matched_oracle"] >= 0 for r in result["rows"])
    for prior_name in ("uniform", "graph_kernel_correlated"):
        for horizon in (1, 2):
            for objective in OBJECTIVES:
                row = next(r for r in result["rows"] if r["prior"] == prior_name and r["horizon"] == horizon and r["policy"] == f"oracle_{objective}" and r["scored_objective"] == objective)
                assert abs(row["regret_bits_vs_matched_oracle"]) < 1e-9
    return True

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--priors", default=",".join(PRIOR_NAMES))
    parser.add_argument("--source-git-blob", default="UNBOUND")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test(); print("PRIOR_SHIFT_ORACLE_TESTS_PASSED"); return 0
        names = tuple(args.priors.split(","))
        if not names or any(n not in PRIOR_NAMES for n in names):
            raise ValueError("unknown/empty prior list")
        result = run_benchmark(args.horizon, names)
        result["source_git_blob"] = args.source_git_blob
        raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        digest = hashlib.sha256(raw).hexdigest()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_bytes(raw)
            print(f"wrote {args.output} bytes={len(raw)} sha256={digest}")
        else: print(raw.decode(), end="")
        return 0
    except (ValueError, AssertionError, OSError, RuntimeError, RecursionError) as exc:
        print(f"NOT QUALIFIED: {exc}"); return 2
if __name__ == "__main__": raise SystemExit(main())
