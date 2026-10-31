#!/usr/bin/env python3
"""Distributionally robust randomized CRC over a frozen finite scenario set.

Optimizes worst-case expected hierarchy specificity over an ex-ante set of
finite-population type mixtures while enforcing the same universal
finite-exchangeability augmented-multiset risk constraints.

This is robustness of the utility objective across reference populations.
It is not a new distribution-shift validity theorem: the guarantee remains
the finite-exchangeability expected-risk guarantee under the stated loss.
"""
from __future__ import annotations
import argparse
import json
import math
import platform
import time
from typing import Iterable
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from randomized_global_crc_cifar import (
    RandomizedCommonDepthCRC,
    finite_population_state_weights,
    residual_test_probabilities,
    per_type_metrics,
)
from common_depth_cifar_crc import CommonDepthCIFARCRC


def _validate_scenario_pools(
    scenario_pools: dict[str, Iterable[int]], n_calibration: int
) -> dict[str, tuple[int, int, int]]:
    if not isinstance(scenario_pools, dict) or len(scenario_pools) < 2:
        raise ValueError("at least two named population scenarios are required")
    result = {}
    totals = set()
    for name, counts in scenario_pools.items():
        if not isinstance(name, str) or not name:
            raise ValueError("scenario names must be nonempty strings")
        values = tuple(int(x) for x in counts)
        if len(values) != 3 or any(x <= n_calibration for x in values):
            raise ValueError("each scenario needs three type counts > n_calibration")
        totals.add(sum(values))
        result[name] = values
    if len(totals) != 1:
        raise ValueError("all scenarios must have the same finite population size")
    return result


def _scenario_tables(n: int, pool: tuple[int, int, int],
                     states: list[tuple[int, int, int]],
                     actions: np.ndarray) -> dict:
    state_order, weights = finite_population_state_weights(n, pool)
    if state_order != states:
        raise ArithmeticError("scenario and LP calibration state orders differ")
    conditional = np.asarray(
        [residual_test_probabilities(pool, D) for D in states],
        dtype=np.float64,
    )
    metrics = [per_type_metrics(float(a)) for a in actions]
    specificity = np.asarray(
        [[float(np.dot(conditional[i], metrics[j]["specificity"]))
          for j in range(len(actions))]
         for i in range(len(states))],
        dtype=np.float64,
    )
    return {
        "weights": weights,
        "conditional_test_probabilities": conditional,
        "specificity_by_state_action": specificity,
        "metrics": metrics,
    }


class DistributionallyRobustRandomizedCRC:
    """Maximin-utility randomized CRC over a finite population-scenario set."""

    def __init__(
        self,
        n_calibration: int,
        alpha: float,
        scenario_pools: dict[str, Iterable[int]],
        action_grid: Iterable[float] = (0.0, 1.0, 2.0),
    ):
        self.scenarios = _validate_scenario_pools(scenario_pools, n_calibration)
        self.names = list(self.scenarios)
        self.nominal_name = self.names[0]
        self.base = RandomizedCommonDepthCRC(
            n_calibration, alpha, self.scenarios[self.nominal_name], action_grid
        )
        self.n = self.base.n
        self.alpha = self.base.alpha
        self.actions = self.base.actions
        self.states = self.base.states
        self.augmented = self.base.augmented
        self.losses = self.base.losses
        self.budget = self.base.budget
        self.index = self.base.index
        self.scenario_tables = {
            name: _scenario_tables(self.n, pool, self.states, self.actions)
            for name, pool in self.scenarios.items()
        }

    def _matrices(self):
        n_states = len(self.states)
        n_actions = len(self.actions)
        n_policy = n_states * n_actions
        n_variables = n_policy + 1  # final variable is worst-case utility t
        n_inequalities = len(self.augmented) + len(self.names)
        A_ub = lil_matrix((n_inequalities, n_variables), dtype=np.float64)
        b_ub = np.zeros(n_inequalities)

        # Universal expected-risk constraints; independent of scenario weights.
        for row, S in enumerate(self.augmented):
            for z, multiplicity in enumerate(S):
                if multiplicity:
                    D = list(S)
                    D[z] -= 1
                    state_index = self.index[tuple(D)]
                    start = state_index * n_actions
                    A_ub[row, start:start+n_actions] += multiplicity * self.losses[z]
            b_ub[row] = self.budget

        # Epigraph constraints: t <= expected specificity under every scenario.
        for s_index, name in enumerate(self.names):
            row = len(self.augmented) + s_index
            table = self.scenario_tables[name]
            coefficients = table["weights"][:, None] * table["specificity_by_state_action"]
            A_ub[row, :n_policy] = -coefficients.reshape(-1)
            A_ub[row, n_policy] = 1.0
            b_ub[row] = 0.0

        A_eq = lil_matrix((n_states, n_variables), dtype=np.float64)
        for i in range(n_states):
            A_eq[i, i*n_actions:(i+1)*n_actions] = 1.0
        b_eq = np.ones(n_states)
        return A_ub.tocsr(), b_ub, A_eq.tocsr(), b_eq

    def verify(self, probabilities: np.ndarray, tolerance: float = 1e-9) -> dict:
        return self.base.verify(probabilities, tolerance)

    def evaluate_scenario(self, probabilities: np.ndarray, name: str) -> dict:
        P = np.asarray(probabilities, dtype=np.float64)
        table = self.scenario_tables[name]
        weights = table["weights"]
        conditional = table["conditional_test_probabilities"]
        metrics = table["metrics"]
        total = {"risk": 0.0, "correct_specificity": 0.0, "mean_node_size": 0.0}
        for i in range(len(self.states)):
            for j in range(len(self.actions)):
                mass = weights[i] * P[i, j]
                if mass == 0.0:
                    continue
                total["risk"] += mass * float(
                    np.dot(conditional[i], np.asarray([
                        0.0 if z == 0 else (
                            0.5 * max(float(self.actions[j])-1.0, 0.0)
                            if z == 1 else 0.5*float(self.actions[j])
                        )
                        for z in range(3)
                    ]))
                )
                total["correct_specificity"] += mass * float(
                    table["specificity_by_state_action"][i, j]
                )
                total["mean_node_size"] += mass * float(
                    np.dot(conditional[i], metrics[j]["node_size"])
                )
        return total

    def solve(self) -> dict:
        A_ub, b_ub, A_eq, b_eq = self._matrices()
        n_states = len(self.states)
        n_actions = len(self.actions)
        n_policy = n_states * n_actions
        n_variables = n_policy + 1
        objective = np.zeros(n_variables, dtype=np.float64)
        objective[-1] = -1.0  # maximize t

        start = time.perf_counter()
        result = linprog(
            objective,
            A_ub=A_ub, b_ub=b_ub,
            A_eq=A_eq, b_eq=b_eq,
            bounds=[(0.0, 1.0)]*n_policy + [(0.0, 1.0)],
            method="highs",
            options={
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
            },
        )
        runtime = time.perf_counter() - start
        if not result.success or result.x is None:
            raise RuntimeError(f"robust randomized CRC LP failed: {result.message}")

        probabilities = result.x[:n_policy].reshape(n_states, n_actions)
        worst_case_utility = float(result.x[-1])
        validity = self.verify(probabilities)
        if not validity["valid"]:
            raise ArithmeticError(f"robust policy failed exact CRC checks: {validity}")

        scenario_metrics = {
            name: self.evaluate_scenario(probabilities, name)
            for name in self.names
        }
        actual_worst = min(
            x["correct_specificity"] for x in scenario_metrics.values()
        )
        if abs(actual_worst-worst_case_utility) > 1e-7:
            raise ArithmeticError("robust epigraph objective does not equal worst scenario utility")

        # KKT witness for maximin LP, including t as a primal variable.
        y_ub = np.asarray(result.ineqlin.marginals, dtype=np.float64)
        y_eq = np.asarray(result.eqlin.marginals, dtype=np.float64)
        y_lower = np.asarray(result.lower.marginals, dtype=np.float64)
        y_upper = np.asarray(result.upper.marginals, dtype=np.float64)
        x = np.asarray(result.x, dtype=np.float64)
        lower = np.zeros(n_variables)
        upper = np.ones(n_variables)
        inequality_slack = b_ub - A_ub@x
        equality_residual = A_eq@x - b_eq
        stationarity = objective - A_ub.T@y_ub - A_eq.T@y_eq - y_lower - y_upper
        dual_objective = float(
            np.dot(b_ub, y_ub) + np.dot(b_eq, y_eq)
            + np.dot(lower, y_lower) + np.dot(upper, y_upper)
        )
        complementarity = max(
            float(np.max(np.abs(y_ub*inequality_slack))),
            float(np.max(np.abs(y_lower*(x-lower)))),
            float(np.max(np.abs(y_upper*(upper-x)))),
        )
        kkt = {
            "primal_inequality_violation": float(np.max(np.maximum(-inequality_slack, 0.0))),
            "primal_equality_residual": float(np.max(np.abs(equality_residual))),
            "dual_sign_violation": float(max(
                np.max(np.maximum(y_ub, 0.0)),
                np.max(np.maximum(-y_lower, 0.0)),
                np.max(np.maximum(y_upper, 0.0)),
            )),
            "stationarity_max_abs": float(np.max(np.abs(stationarity))),
            "complementarity_max_abs": complementarity,
            "duality_gap_abs": float(abs(float(result.fun)-dual_objective)),
            "valid": True,
        }
        worst_kkt = max(v for k, v in kkt.items() if k != "valid")
        kkt["valid"] = bool(worst_kkt <= 1e-7)
        if not kkt["valid"]:
            raise ArithmeticError(f"robust CRC LP failed KKT checks: {kkt}")

        support = (probabilities > 1e-8).sum(axis=1)
        return {
            "probabilities": probabilities.tolist(),
            "worst_case_specificity": worst_case_utility,
            "scenario_metrics": scenario_metrics,
            "verification": validity,
            "solver": {
                "method": "HiGHS",
                "runtime_seconds": runtime,
                "objective": float(-result.fun),
                "n_variables": n_variables,
                "n_inequality_constraints": len(b_ub),
                "n_equality_constraints": len(b_eq),
                "kkt": kkt,
                "dual_witness": {
                    "inequality_marginals": y_ub.tolist(),
                    "equality_marginals": y_eq.tolist(),
                    "lower_marginals": y_lower.tolist(),
                    "upper_marginals": y_upper.tolist(),
                },
            },
            "randomization": {
                "states_with_multiple_actions": int(np.sum(support > 1)),
                "fraction_states_randomized": float(np.mean(support > 1)),
                "support_histogram": {
                    str(k): int(np.sum(support == k)) for k in np.unique(support)
                },
            },
        }


def evaluate_equal_share_for_scenario(
    engine: DistributionallyRobustRandomizedCRC,
    controller: dict[tuple[int, int, int], float],
    name: str,
) -> dict:
    table = engine.scenario_tables[name]
    w = table["weights"]
    q = table["conditional_test_probabilities"]
    total = {"risk": 0.0, "correct_specificity": 0.0, "mean_node_size": 0.0}
    for i, D in enumerate(engine.states):
        a = float(controller[D])
        metric = per_type_metrics(a)
        total["risk"] += w[i] * float(np.dot(q[i], metric["risk"]))
        total["correct_specificity"] += w[i] * float(np.dot(q[i], metric["specificity"]))
        total["mean_node_size"] += w[i] * float(np.dot(q[i], metric["node_size"]))
    return total


def run_comparison(
    n: int = 50,
    alpha: float = 0.10,
    scenario_pools: dict[str, tuple[int, int, int]] | None = None,
    equal_share_rounds: int = 500,
) -> dict:
    if scenario_pools is None:
        scenario_pools = {
            "nominal": (7300, 969, 1731),
            "less_superclass_correct": (7300, 500, 2200),
            "more_fine_errors": (6500, 1200, 2300),
            "higher_accuracy": (8000, 800, 1200),
            "mixed_shift": (7000, 1500, 1500),
        }
    robust = DistributionallyRobustRandomizedCRC(
        n, alpha, scenario_pools
    )
    solved = robust.solve()

    nominal = RandomizedCommonDepthCRC(
        n, alpha, scenario_pools["nominal"]
    ).solve()

    baseline = CommonDepthCIFARCRC(n, alpha)
    envelope = baseline.baseline("envelope")
    equal_share, history = baseline.recertify(
        envelope, rounds=equal_share_rounds
    )

    scenario_rows = {}
    for name in robust.names:
        nominal_metrics = robust.evaluate_scenario(
            np.asarray(nominal["probabilities"]), name
        )
        robust_metrics = solved["scenario_metrics"][name]
        equal_metrics = evaluate_equal_share_for_scenario(
            robust, equal_share, name
        )
        scenario_rows[name] = {
            "nominal_oracle_policy": nominal_metrics,
            "distributionally_robust_policy": robust_metrics,
            "equal_share_policy": equal_metrics,
            "robust_gain_vs_equal_share": (
                robust_metrics["correct_specificity"]
                - equal_metrics["correct_specificity"]
            ),
            "nominal_gain_vs_equal_share": (
                nominal_metrics["correct_specificity"]
                - equal_metrics["correct_specificity"]
            ),
        }

    return {
        "schema": "symthaea-distributionally-robust-crc-v1",
        "status": "FINITE_SCENARIO_ROBUST_UTILITY_BENCHMARK_NOT_SHIFT_VALIDITY_THEOREM",
        "protocol": {
            "n_calibration": n,
            "alpha": alpha,
            "scenario_pools": {k: list(v) for k, v in scenario_pools.items()},
            "scenario_ambiguity_set_frozen_ex_ante": True,
            "action_support": robust.actions.tolist(),
            "objective": "maximize minimum expected correct specificity over the finite scenario set",
            "risk_constraints": "all finite-exchangeability augmented-multiset constraints, independent of scenario weights",
        },
        "states": {
            "calibration": len(robust.states),
            "augmented": len(robust.augmented),
        },
        "worst_case_specificity": solved["worst_case_specificity"],
        "scenario_comparison": scenario_rows,
        "randomized_policy": {
            "states_with_multiple_actions": solved["randomization"]["states_with_multiple_actions"],
            "fraction_states_randomized": solved["randomization"]["fraction_states_randomized"],
            "support_histogram": solved["randomization"]["support_histogram"],
        },
        "verification": solved["verification"],
        "solver": solved["solver"],
        "limitations": [
            "finite objective scenario set only; no guarantee for distributions outside it",
            "the risk certificate remains finite-exchangeability expected risk, not covariate-shift robustness",
            "scenario populations must be chosen and frozen without inspecting the test outcomes",
            "three-type common-depth family only; feature-adaptive controller remains separate",
        ],
    }


def make_composition_grid_scenarios(
    population_total: int = 10000,
    min_count_per_type: int = 500,
    step: int = 500,
) -> dict[str, tuple[int, int, int]]:
    """All 3-type integer mixtures on a regular finite population grid."""
    if population_total <= 0 or min_count_per_type <= 0 or step <= 0:
        raise ValueError("population total, minimum count and step must be positive")
    remaining = population_total - 3 * min_count_per_type
    if remaining < 0 or remaining % step != 0:
        raise ValueError("grid parameters must tile the population total exactly")
    units = remaining // step
    scenarios = {}
    for x in range(units + 1):
        for y in range(units - x + 1):
            z = units - x - y
            counts = (
                min_count_per_type + step * x,
                min_count_per_type + step * y,
                min_count_per_type + step * z,
            )
            name = f"mix_{counts[0]}_{counts[1]}_{counts[2]}"
            scenarios[name] = counts
    return scenarios


def run_grid_robustness_stress(
    n: int = 50,
    alpha: float = 0.10,
    population_total: int = 10000,
    min_count_per_type: int = 500,
    step: int = 500,
    nominal_pool: tuple[int, int, int] = (7300, 969, 1731),
    equal_share_rounds: int = 500,
) -> dict:
    """Compare robust, nominal-oracle and equal-share policies on a dense pool grid."""
    scenarios = make_composition_grid_scenarios(
        population_total, min_count_per_type, step
    )
    robust = DistributionallyRobustRandomizedCRC(n, alpha, scenarios)
    solved = robust.solve()
    robust_P = np.asarray(solved["probabilities"], dtype=np.float64)

    nominal = RandomizedCommonDepthCRC(n, alpha, nominal_pool).solve()
    nominal_P = np.asarray(nominal["probabilities"], dtype=np.float64)

    equal_engine = CommonDepthCIFARCRC(n, alpha)
    envelope = equal_engine.baseline("envelope")
    equal_share, equal_history = equal_engine.recertify(
        envelope, rounds=equal_share_rounds
    )
    equal_validity = equal_engine.validity_report(equal_share)

    rows = {}
    for name in robust.names:
        rows[name] = {
            "robust": robust.evaluate_scenario(robust_P, name),
            "nominal_oracle": robust.evaluate_scenario(nominal_P, name),
            "equal_share": evaluate_equal_share_for_scenario(
                robust, equal_share, name
            ),
        }

    def worst(method: str, metric: str = "correct_specificity"):
        return min(
            ((values[method][metric], name) for name, values in rows.items()),
            key=lambda x: x[0],
        )

    def maximum(method: str, metric: str):
        return max(values[method][metric] for values in rows.values())

    robust_worst = worst("robust")
    nominal_worst = worst("nominal_oracle")
    equal_worst = worst("equal_share")

    return {
        "schema": "symthaea-distributionally-robust-crc-grid-v1",
        "status": "FINITE_GRID_MAXIMIN_REPLAY_NOT_CONTINUOUS_DISTRIBUTION_ROBUSTNESS",
        "protocol": {
            "n_calibration": n,
            "alpha": alpha,
            "population_total": population_total,
            "minimum_count_per_type": min_count_per_type,
            "step": step,
            "scenario_count": len(scenarios),
            "scenario_set": {k: list(v) for k, v in scenarios.items()},
            "scenario_grid_covers": (
                f"all integer 3-type mixtures with each type >= {min_count_per_type} "
                f"and counts on a {step}-example lattice"
            ),
            "action_support": robust.actions.tolist(),
            "objective": "maximize worst-case expected correct specificity over the scenario grid",
            "risk_constraints": "all augmented count-vector constraints, independent of scenario weights",
        },
        "state_counts": {
            "calibration": len(robust.states),
            "augmented": len(robust.augmented),
        },
        "worst_case": {
            "robust_specificity": robust_worst[0],
            "robust_scenario": robust_worst[1],
            "nominal_oracle_specificity": nominal_worst[0],
            "nominal_oracle_scenario": nominal_worst[1],
            "equal_share_specificity": equal_worst[0],
            "equal_share_scenario": equal_worst[1],
            "robust_gain_vs_equal_share_pp": 100.0 * (robust_worst[0] - equal_worst[0]),
            "robust_minus_nominal_oracle_pp": 100.0 * (robust_worst[0] - nominal_worst[0]),
        },
        "maximum_expected_risk_over_grid": {
            "robust": maximum("robust", "risk"),
            "nominal_oracle": maximum("nominal_oracle", "risk"),
            "equal_share": maximum("equal_share", "risk"),
        },
        "scenario_results": rows,
        "equal_share_verification": {
            "valid": equal_validity["valid"],
            "max_budget_excess": equal_validity["max_budget_excess"],
            "rounds_executed": len(equal_history) - 1,
        },
        "robust_verification": solved["verification"],
        "solver": solved["solver"],
        "limitations": [
            "finite lattice only; no guarantee for continuous mixtures between or outside grid points",
            "expected utility is robust only over the listed finite set",
            "finite-exchangeability expected-risk validity remains universal over all augmented count states",
            "scenario set must be frozen independently of the evaluated test outcomes",
            "common-depth three-type family only; confidence-adaptive family is separate",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run_comparison()
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps({
        "output": args.output,
        "worst_case_specificity": result["worst_case_specificity"],
        "scenario_results": result["scenario_comparison"],
        "max_augmented_violation": result["verification"]["max_budget_excess"],
        "solver": result["solver"],
    }, indent=2))
