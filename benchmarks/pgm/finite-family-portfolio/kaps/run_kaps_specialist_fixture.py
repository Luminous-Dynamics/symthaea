#!/usr/bin/env python3
"""One-step exact-oracle fixture for the published SUMI-lab KAPS implementation.

Run with KAPS_ROOT pointing at a checkout of SUMI-lab/KAPS. This imports only
KAPS's own MDP/family classes and its 'ours' nested branch-and-bound solver;
it does not invoke the optional Gurobi MILP, CREG, or CEMR solvers.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

kaps_root = Path(os.environ["KAPS_ROOT"]).resolve()
if not (kaps_root / "src" / "algorithm" / "multi_policy_solver.py").is_file():
    raise SystemExit(f"KAPS_ROOT does not contain the KAPS source tree: {kaps_root}")
sys.path.insert(0, str(kaps_root))

from src.algorithm.multi_policy_solver import MultiPolicySolver  # noqa: E402
from src.environments.mdp import Family, MarkovDecisionProcess  # noqa: E402


def one_state_model(action_rewards: tuple[float, float]) -> MarkovDecisionProcess:
    # Every action returns to the sole state with probability one.
    # gamma=0 makes this one-decision model exactly equivalent to H=1 reward.
    transitions = np.ones((1, 2, 1), dtype=np.float64)
    rewards = np.asarray(action_rewards, dtype=np.float64).reshape(1, 2, 1)
    initial = np.ones(1, dtype=np.float64)
    return MarkovDecisionProcess(transitions, rewards, initial, gamma=0.0)


def main() -> None:
    family = Family([
        one_state_model((1.0, 0.0)),  # model 0: action 0 is specialist
        one_state_model((0.0, 1.0)),  # model 1: action 1 is specialist
    ])
    family.compute_vstar_values()
    np.testing.assert_allclose(family.vstar_values, [1.0, 1.0], atol=0.0, rtol=0.0)

    results: dict[str, dict[str, object]] = {}
    for k, expected_regret in ((1, 1.0), (2, 0.0)):
        solver = MultiPolicySolver(
            min_k=k,
            max_k=k,
            time_limit_per_node=30.0,
            time_out_per_k=60.0,
        )
        started = time.perf_counter()
        partition, policies, regret, lower_bound, solver_runtime = solver.solve(family)[k]
        wall_seconds = time.perf_counter() - started

        if partition is None or policies is None:
            raise RuntimeError(f"KAPS did not return a portfolio for k={k}")
        covered = sorted(index for cluster in partition.values() for index in cluster)
        if covered != [0, 1]:
            raise AssertionError(f"KAPS partition omitted or duplicated models: {partition}")
        if not math.isclose(float(regret), expected_regret, abs_tol=1e-9, rel_tol=0.0):
            raise AssertionError(
                f"k={k}: KAPS regret {regret!r}, expected {expected_regret}"
            )
        if lower_bound is None or float(lower_bound) > expected_regret + 1e-8:
            raise AssertionError(f"k={k}: invalid returned lower bound {lower_bound!r}")

        results[str(k)] = {
            "regret_float": float(regret),
            "lower_bound_float": float(lower_bound),
            "partition": {str(key): list(value) for key, value in partition.items()},
            "policies": [None if policy is None else policy.tolist() for policy in policies],
            "solver_runtime_seconds": float(solver_runtime),
            "wall_runtime_seconds": wall_seconds,
            "expected_regret_exact": "1" if k == 1 else "0",
        }

    print(json.dumps({
        "comparator": "SUMI-lab/KAPS",
        "fixture": "one-state-two-models-two-specialist-actions",
        "gamma": "0",
        "model_optimal_values": ["1", "1"],
        "results_by_k": results,
        "status": "PASS",
        "qualification": "KAPS is evaluated in float64; this fixture's values are exactly representable.",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
