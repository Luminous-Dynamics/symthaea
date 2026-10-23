#!/usr/bin/env python3
"""Adaptive probe-policy benchmark for action-dependent target-subset kernels.

This module imports the exact SCM likelihood table from the preceding pilot. It
compares fixed round-robin/random action allocation to adaptive graph-, target-
kernel-, and joint-information policies under the same probe count. The core uses
the standard library, with optional NumPy acceleration; worlds are synthetic only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
try:
    import numpy as _np
except ImportError:  # standard-library fallback keeps the benchmark portable
    _np = None
from action_subset_kernel_benchmark import (
    GRAPH_NAMES, KERNELS, MODES, TABLES, VARIABLES, WORLD,
    baseline_hard_do, graph_metrics, sample_categorical,
    true_kernels, logsumexp, softmax_log, encode_observation,
)

SCHEMA = "symthaea-adaptive-probe-policy-benchmark-v1"
BASE_KERNEL_SOURCE = {"path": "validation/action_subset_kernel_benchmark.py", "git_blob": "ce19dccec3a81f4c6a18563de88152019b966f5d", "sha256": "9d000f97b9cd7f16e4e86e9089c1ff230c2fc4ee4624dc7a5836a7c1ebbf6526", "bytes": 25398}
POLICIES = ("round_robin", "random", "greedy_graph_information", "greedy_kernel_information", "greedy_joint_information", "coverage_then_graph_information", "coverage_then_joint_information")
DEFAULT_REPLICATES = 8
DEFAULT_BUDGET = 4


def entropy(probabilities):
    out = 0.0
    for p in probabilities:
        if p > 0.0:
            out -= p * math.log2(p)
    return out


def posterior_state(data, mode):
    """Return graph posterior and graph-conditioned kernel posterior per action.

    Conditional on a graph, action-specific kernel priors/data factorize. We sum
    each kernel out to update the graph posterior, then retain the per-graph,
    per-action conditional posterior required for prospective information gain.
    """
    g_count, d_count = len(GRAPH_NAMES), len(KERNELS)
    per_action_loglike = {}
    for action in VARIABLES:
        rows = [r for r in data if r["action"] == action]
        g_rows = []
        likelihoods = TABLES[action][mode]
        for g in range(g_count):
            d_ll = []
            for d in range(d_count):
                dist = likelihoods[g][d]
                total = 0.0
                for row in rows:
                    p = dist[encode_observation(row, mode)]
                    if p <= 0.0:
                        total = -math.inf
                        break
                    total += math.log(p)
                d_ll.append(total)
            g_rows.append(d_ll)
        per_action_loglike[action] = g_rows
    log_evidence = {}
    for action in VARIABLES:
        log_evidence[action] = [logsumexp([x - math.log(d_count) for x in per_action_loglike[action][g]])
                                for g in range(g_count)]
    log_graph = [-math.log(g_count) + sum(log_evidence[a][g] for a in VARIABLES)
                 for g in range(g_count)]
    graph_p = softmax_log(log_graph)
    kernel_conditional = {}
    for action in VARIABLES:
        for_g = []
        for g in range(g_count):
            for_g.append(softmax_log([x - math.log(d_count) for x in per_action_loglike[action][g]]))
        kernel_conditional[action] = tuple(for_g)
    return {"graph_posterior": graph_p, "kernel_conditional": kernel_conditional}


def expected_information(data, action, mode, objective, state=None):
    """Exact one-step mutual information under current finite posterior."""
    if state is None:
        state = posterior_state(data, mode)
    graph_p = state["graph_posterior"]
    cond = state["kernel_conditional"][action]
    g_count, d_count = len(GRAPH_NAMES), len(KERNELS)
    if _np is not None:
        # Vectorized exact entropy reduction; no sampling or approximation is introduced.
        joint_np = _np.asarray([[graph_p[g] * cond[g][d] for d in range(d_count)]
                                for g in range(g_count)], dtype=float)
        joint_np /= joint_np.sum()
        likelihood_np = _np.asarray(TABLES[action][mode], dtype=float)  # graph × kernel × observation
        pred_np = (joint_np[:, :, None] * likelihood_np).sum(axis=(0, 1))
        prior_hg = entropy(graph_p)
        prior_hd = entropy(joint_np.sum(axis=0).tolist())
        prior_hjoint = entropy(joint_np.ravel().tolist())
        weighted_hg = weighted_hd = weighted_hjoint = 0.0
        for obs, p_obs in enumerate(pred_np):
            if p_obs <= 0.0:
                continue
            post = (joint_np * likelihood_np[:, :, obs]) / p_obs
            pg = post.sum(axis=1)
            pd = post.sum(axis=0)
            hg_terms = _np.where(pg > 0.0, pg * _np.log2(_np.maximum(pg, 1e-300)), 0.0)
            hd_terms = _np.where(pd > 0.0, pd * _np.log2(_np.maximum(pd, 1e-300)), 0.0)
            hj_terms = _np.where(post > 0.0, post * _np.log2(_np.maximum(post, 1e-300)), 0.0)
            weighted_hg -= float(p_obs) * float(hg_terms.sum())
            weighted_hd -= float(p_obs) * float(hd_terms.sum())
            weighted_hjoint -= float(p_obs) * float(hj_terms.sum())
        values = {"graph": prior_hg - weighted_hg,
                  "kernel": prior_hd - weighted_hd,
                  "joint": prior_hjoint - weighted_hjoint}
        if objective not in values:
            raise ValueError(f"unknown objective {objective}")
        result = values[objective]
        if result < -1e-8:
            raise AssertionError(f"negative mutual information {objective}={result}")
        return max(0.0, result)
    joint = [[graph_p[g] * cond[g][d] for d in range(d_count)] for g in range(g_count)]
    # Numerical normalization guard.
    total = sum(sum(row) for row in joint)
    joint = [[v / total for v in row] for row in joint]
    prior_hg = entropy(graph_p)
    prior_hd = entropy([sum(joint[g][d] for g in range(g_count)) for d in range(d_count)])
    prior_hjoint = entropy([p for row in joint for p in row])
    outcome_count = len(TABLES[action][mode][0][0])
    predictive = [sum(joint[g][d] * TABLES[action][mode][g][d][o]
                      for g in range(g_count) for d in range(d_count))
                  for o in range(outcome_count)]
    e_hg = e_hd = e_hjoint = 0.0
    for o, po in enumerate(predictive):
        if po <= 0.0:
            continue
        post_joint = [[joint[g][d] * TABLES[action][mode][g][d][o] / po
                       for d in range(d_count)] for g in range(g_count)]
        post_g = [sum(post_joint[g]) for g in range(g_count)]
        post_d = [sum(post_joint[g][d] for g in range(g_count)) for d in range(d_count)]
        e_hg += po * entropy(post_g)
        e_hd += po * entropy(post_d)
        e_hjoint += po * entropy([p for row in post_joint for p in row])
    values = {
        "graph": prior_hg - e_hg,
        "kernel": prior_hd - e_hd,
        "joint": prior_hjoint - e_hjoint,
    }
    for key, value in values.items():
        if value < -1e-9:
            raise AssertionError(f"negative mutual information {key}={value}")
        values[key] = max(0.0, value)
    return values[objective]


def choose_action(policy, data, step, mode, policy_rng, budget):
    if policy == "round_robin":
        return VARIABLES[step % len(VARIABLES)]
    if policy == "random":
        return policy_rng.choice(VARIABLES)
    objective = {
        "greedy_graph_information": "graph",
        "greedy_kernel_information": "kernel",
        "greedy_joint_information": "joint",
        "coverage_then_graph_information": "graph",
        "coverage_then_joint_information": "joint",
    }.get(policy)
    if objective is None:
        raise ValueError(f"unknown policy {policy}")
    candidates = tuple(VARIABLES)
    if policy.startswith("coverage_then_"):
        counts = [sum(1 for row in data if row["action"] == a) for a in VARIABLES]
        unseen = tuple(a for a, count in zip(VARIABLES, counts) if count == 0)
        if unseen:
            candidates = unseen
    state = posterior_state(data, mode)
    scores = {action: expected_information(data, action, mode, objective, state) for action in candidates}
    return min(candidates, key=lambda a: (-scores[a], a))


def sample_observation(graph_idx, action, actual_dist, mode, rng):
    subset = sample_categorical(actual_dist, rng)
    outcome = sample_categorical(WORLD[graph_idx][subset], rng)
    # For the noisy-target-flag channel, the imported pilot's sensor contract is used.
    if mode == "noisy_target_flag":
        from action_subset_kernel_benchmark import noisy_flag_prob
        flag = int(rng.random() < noisy_flag_prob(1, action, subset))
    else:
        flag = 0
    return {"action": action, "subset": subset, "y": outcome, "flag": flag}


def run_benchmark(replicates=DEFAULT_REPLICATES, budget=DEFAULT_BUDGET, modes=None):
    if replicates < 1 or budget < 1:
        raise ValueError("replicates and budget must be positive")
    modes = tuple(modes or MODES)
    if any(m not in MODES for m in modes):
        raise ValueError("unsupported observation mode")
    rows = []
    trial = 0
    for graph_idx, graph_name in enumerate(GRAPH_NAMES):
        for regime_idx, regime in enumerate(("in_support", "out_of_support")):
            actual_dists, _ = true_kernels(regime)
            for rep in range(replicates):
                base_seed = 20261023 + graph_idx * 100000 + regime_idx * 10000 + rep
                for mode_idx, mode in enumerate(modes):
                    for policy_idx, policy in enumerate(POLICIES):
                        trial += 1
                        # Independent policy randomness; a common world stream by paired seed
                        # reduces noise when policies happen to choose the same actions.
                        world_rng = random.Random(base_seed + mode_idx * 1000)
                        policy_rng = random.Random(base_seed + policy_idx * 10000 + mode_idx * 1000)
                        data = []
                        action_counts = [0, 0, 0]
                        for step in range(budget):
                            action = choose_action(policy, data, step, mode, policy_rng, budget)
                            row = sample_observation(graph_idx, action, actual_dists[action], mode, world_rng)
                            data.append(row)
                            action_counts[action] += 1
                        state = posterior_state(data, mode)
                        metrics = graph_metrics(state["graph_posterior"], graph_idx)
                        mean_kernels = []
                        for action in VARIABLES:
                            mean_d = [0.0] * 8
                            for g in range(len(GRAPH_NAMES)):
                                for d, p_d in enumerate(state["kernel_conditional"][action][g]):
                                    weight = state["graph_posterior"][g] * p_d
                                    for s, p_s in enumerate(KERNELS[d]):
                                        mean_d[s] += weight * p_s
                            mean_kernels.append(mean_d)
                        l1_by_action = [sum(abs(x-y) for x, y in zip(mean_kernels[a], actual_dists[a]))
                                        for a in VARIABLES]
                        hard = baseline_hard_do(data)
                        hard_metrics = graph_metrics(hard, graph_idx)
                        rows.append({
                            "trial_id": trial, "seed": base_seed, "graph": graph_name,
                            "graph_index": graph_idx, "kernel_regime": regime,
                            "observation_mode": mode, "policy": policy, "budget": budget,
                            **metrics,
                            "mean_target_kernel_l1": sum(l1_by_action) / len(l1_by_action),
                            "target_kernel_l1_by_action": l1_by_action,
                            "action_counts": action_counts,
                            "hard_do_graph_map_correct": hard_metrics["graph_map_correct"],
                            "hard_do_graph_log_loss_bits": hard_metrics["graph_log_loss_bits"],
                            "hard_do_graph_brier_score": hard_metrics["graph_brier_score"],
                        })
    aggregate = {}
    metric_keys = ("graph_map_correct", "graph_log_loss_bits", "graph_brier_score", "graph_true_posterior",
                   "mean_target_kernel_l1", "hard_do_graph_map_correct", "hard_do_graph_log_loss_bits",
                   "hard_do_graph_brier_score")
    for regime in ("in_support", "out_of_support"):
        for mode in modes:
            for policy in POLICIES:
                group = [r for r in rows if r["kernel_regime"] == regime and r["observation_mode"] == mode and r["policy"] == policy]
                item = {"n_trials": len(group)}
                for key in metric_keys:
                    vals = [float(r[key]) for r in group]
                    item[key + "_mean"] = sum(vals) / len(vals) if vals else None
                item["mean_action_counts"] = [sum(r["action_counts"][i] for r in group)/len(group) for i in VARIABLES]
                aggregate[f"{regime}/{mode}/{policy}"] = item
    return {
        "schema": SCHEMA,
        "status": "EXACT_POSTERIOR_ADAPTIVE_POLICY_PILOT_LOCAL_NOT_EXTERNAL_QUALIFICATION",
        "base_kernel_source": BASE_KERNEL_SOURCE,
        "protocol": {
            "graph_models": list(GRAPH_NAMES), "variables": 3,
            "kernel_grid": "uniform over 330 quarter-step distributions on all 8 subsets, independent by attempted target",
            "policies": list(POLICIES), "modes": list(modes), "budget_probes_per_trial": budget,
            "replicates_per_graph_regime": replicates,
            "truth_regimes": ["in_support: 0.5 intended singleton + 0.5 adjacent pair",
                              "out_of_support: 0.40 singleton + 0.35 adjacent pair + 0.25 all-three subset"],
            "selection": "exact one-step expected mutual information from current posterior; tie-break by lowest target index",
            "information_engine": "exact finite-hypothesis likelihood enumeration; optional NumPy vectorization for entropy reductions with a standard-library fallback",
            "metrics": "graph MAP accuracy, log loss, Brier score, posterior truth mass, mean target-kernel L1, action-use counts",
            "limitations": ["small three-variable synthetic SCM only", "no external ACDB/TCD-Arena or Choo et al. implementation run",
                            "fixed horizon compares equal probe counts, not physical cost/damage", "finite kernel prior misspecified out-of-grid"],
        },
        "aggregate_metrics": aggregate, "rows": rows,
    }


def self_test():
    empty_scores = {p: expected_information([], a, "exact_target_subset", "joint")
                    for p in ("greedy_joint_information",) for a in VARIABLES}
    assert all(math.isfinite(x) and x >= 0 for x in empty_scores.values())
    quick = run_benchmark(replicates=1, budget=2, modes=("exact_target_subset",))
    assert len(quick["rows"]) == len(GRAPH_NAMES) * 2 * len(POLICIES)
    assert all(0 <= r["graph_true_posterior"] <= 1 for r in quick["rows"])
    assert all(sum(r["action_counts"]) == 2 for r in quick["rows"])
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--source-git-blob", default="UNBOUND")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            print("ADAPTIVE_PROBE_POLICY_SELF_TESTS_PASSED")
            return 0
        result = run_benchmark(args.replicates, args.budget, tuple(args.modes.split(",")))
        result["source_git_blob"] = args.source_git_blob
        raw = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
        digest = hashlib.sha256(raw).hexdigest()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_bytes(raw)
            print(f"wrote {args.output} bytes={len(raw)} sha256={digest}")
        else:
            print(raw.decode(), end="")
        return 0
    except (ValueError, AssertionError, OSError, RuntimeError, OverflowError) as exc:
        print(f"NOT QUALIFIED: {exc}")
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
