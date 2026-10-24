#!/usr/bin/env python3
"""Exact finite-horizon active causal probing oracle on a reduced SCM game.

The benchmark intentionally uses a small 16-hypothesis model class so the Bellman
optimum can be enumerated exactly. It reuses the prior benchmark's exact SCM outcome
tables, but restricts graph hypotheses to chain/fork and each action kernel to one of
two off-target probabilities (0.1 or 0.6). The observation is the realized subset S
and the 3-bit outcome Y. This proves an exact oracle lane, not general SOTA.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, math, random
from pathlib import Path
from action_subset_kernel_benchmark import WORLD, GRAPH_NAMES, VARIABLES

SCHEMA = "symthaea-exact-finite-horizon-oracle-v1"
BASE_SOURCE_BLOB = "ce19dccec3a81f4c6a18563de88152019b966f5d"
GRAPHS = (0, 1)  # chain and fork
R_VALUES = (0.1, 0.6)
HYPOTHESES = tuple((g, *rs) for g in GRAPHS for rs in itertools.product(range(2), repeat=3))
H0 = math.log2(len(HYPOTHESES))
ACTIONS = tuple(VARIABLES)
HORIZONS = (1, 2, 3)
POLICIES = ("round_robin", "random_uniform", "greedy_graph_information", "greedy_kernel_information", "greedy_joint_information", "coverage_then_joint_information", "exact_bellman_oracle")

OBSERVATIONS = {a: tuple((s, y) for s in (1 << a, (1 << a) | (1 << ((a + 1) % 3))) for y in range(8)) for a in ACTIONS}

def entropy(ps):
    return -sum(p * math.log2(p) for p in ps if p > 0.0)

def likelihood(hyp, action, obs):
    graph_idx, *r_idx = hyp
    s, y = obs
    target_mask = 1 << action
    pair_mask = target_mask | (1 << ((action + 1) % 3))
    r = R_VALUES[r_idx[action]]
    if s == target_mask:
        p_subset = 1.0 - r
    elif s == pair_mask:
        p_subset = r
    else:
        return 0.0
    return p_subset * WORLD[graph_idx][s][y]

def predictive(posterior, action):
    return tuple((obs, sum(w * likelihood(h, action, obs) for h, w in zip(HYPOTHESES, posterior))) for obs in OBSERVATIONS[action])

def update(posterior, action, obs):
    values = tuple(w * likelihood(h, action, obs) for h, w in zip(HYPOTHESES, posterior))
    total = sum(values)
    if total <= 0.0:
        raise ValueError("zero-probability observation cannot update posterior")
    out = tuple(x / total for x in values)
    if not math.isclose(sum(out), 1.0, abs_tol=1e-12, rel_tol=0.0):
        raise AssertionError("posterior failed normalization")
    return out

def marginals(posterior):
    pg = [0.0] * len(GRAPHS)
    pk = [0.0] * 8
    for h, p in zip(HYPOTHESES, posterior):
        graph_idx, *r_idx = h
        pg[graph_idx] += p
        kidx = (r_idx[0] << 2) | (r_idx[1] << 1) | r_idx[2]
        pk[kidx] += p
    return pg, pk

def one_step_information(posterior, action):
    pg, pk = marginals(posterior)
    h_graph, h_kernel, h_joint = entropy(pg), entropy(pk), entropy(posterior)
    eg = ek = ej = 0.0
    for obs, po in predictive(posterior, action):
        if po <= 0.0:
            continue
        post = update(posterior, action, obs)
        p_graph, p_kernel = marginals(post)
        eg += po * entropy(p_graph)
        ek += po * entropy(p_kernel)
        ej += po * entropy(post)
    info = {"graph": h_graph - eg, "kernel": h_kernel - ek, "joint": h_joint - ej}
    for key, val in info.items():
        if val < -1e-9:
            raise AssertionError(f"negative MI: {key}={val}")
        info[key] = max(0.0, val)
    return info

def choose_action(policy, posterior, history):
    depth = len(history)
    if policy == "round_robin":
        return ACTIONS[depth % len(ACTIONS)]
    if policy == "coverage_then_joint_information":
        used = {a for a, _obs in history}
        untried = tuple(a for a in ACTIONS if a not in used)
        if untried:
            return min(untried, key=lambda a: (-one_step_information(posterior, a)["joint"], a))
        return min(ACTIONS, key=lambda a: (-one_step_information(posterior, a)["joint"], a))
    objective = {"greedy_graph_information": "graph", "greedy_kernel_information": "kernel", "greedy_joint_information": "joint"}.get(policy)
    if objective is None:
        raise ValueError(policy)
    return min(ACTIONS, key=lambda a: (-one_step_information(posterior, a)[objective], a))

def terminal_entropy(posterior):
    return entropy(posterior)

# Cache keys retain the full floating-point posterior tuple; no rounded-state merging.
def oracle_value(posterior, remaining, cache=None):
    if cache is None:
        cache = {}
    if remaining == 0:
        return terminal_entropy(posterior)
    key = (tuple(posterior), remaining)
    if key in cache:
        return cache[key]
    candidates = []
    for action in ACTIONS:
        value = 0.0
        for obs, po in predictive(posterior, action):
            if po > 0.0:
                value += po * oracle_value(update(posterior, action, obs), remaining - 1, cache)
        candidates.append((value, action))
    best = min(candidates)[0]
    cache[key] = best
    return best

def policy_value(posterior, remaining, policy, history=(), cache=None):
    if cache is None:
        cache = {}
    if remaining == 0:
        return terminal_entropy(posterior)
    if policy == "exact_bellman_oracle":
        return oracle_value(posterior, remaining, cache)
    if policy == "random_uniform":
        return sum(policy_value_after_action(posterior, remaining, a, policy, history, cache) for a in ACTIONS) / len(ACTIONS)
    action = choose_action(policy, posterior, history)
    return policy_value_after_action(posterior, remaining, action, policy, history, cache)

def policy_value_after_action(posterior, remaining, action, policy, history, cache):
    val = 0.0
    for obs, po in predictive(posterior, action):
        if po > 0.0:
            val += po * policy_value(update(posterior, action, obs), remaining - 1, policy, history + ((action, obs),), cache)
    return val

def first_action_distribution(policy, posterior, horizon):
    if policy == "exact_bellman_oracle":
        vals = []
        cache = {}
        for a in ACTIONS:
            exp = sum(po * oracle_value(update(posterior, a, obs), horizon - 1, cache)
                      for obs, po in predictive(posterior, a) if po > 0.0)
            vals.append((exp, a))
        return min(vals)[1], {a: v for v, a in vals}
    if policy == "random_uniform":
        return None, None
    a = choose_action(policy, posterior, ())
    return a, None

def run_benchmark(max_horizon=3):
    if max_horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    prior = tuple([1.0 / len(HYPOTHESES)] * len(HYPOTHESES))
    cache = {}
    rows = []
    for horizon in range(1, max_horizon + 1):
        oracle = oracle_value(prior, horizon, cache)
        for policy in POLICIES:
            expected_h = oracle if policy == "exact_bellman_oracle" else policy_value(prior, horizon, policy, cache=cache)
            gap = expected_h - oracle
            if gap < -1e-8:
                raise AssertionError(f"policy {policy} beats exact oracle by {-gap} bits")
            first_action, first_scores = first_action_distribution(policy, prior, horizon)
            rows.append({"horizon": horizon, "budget": horizon, "policy": policy,
                         "expected_terminal_joint_entropy_bits": expected_h,
                         "expected_joint_information_gain_bits": H0 - expected_h,
                         "oracle_terminal_joint_entropy_bits": oracle,
                         "regret_bits_vs_exact_oracle": max(0.0, gap),
                         "first_action": first_action,
                         "oracle_first_action_expected_terminal_entropy_by_action": first_scores,
                         "hypothesis_count": len(HYPOTHESES)})
    return {"schema": SCHEMA, "status": "EXACT_FINITE_HORIZON_ORACLE_LOCAL_PILOT",
            "protocol": {"graph_hypotheses": [GRAPH_NAMES[i] for i in GRAPHS],
                         "off_target_probabilities": list(R_VALUES),
                         "target_set_telemetry": "exact (S,Y) observed after each action",
                         "joint_hypotheses": len(HYPOTHESES), "prior": "uniform over 2 graphs × 2^3 action-kernel tuples",
                         "action_set": list(ACTIONS), "objective": "minimize expected terminal entropy of the joint graph/action-kernel hypothesis",
                         "oracle": "exact Bellman recursion over every action and every possible observation for horizons 1-3",
                         "limitations": ["Reduced 16-hypothesis game, not the 330-kernel full pilot", "Only chain and fork graphs",
                                         "Exact subset telemetry assumed", "Oracle optimizes joint entropy, not graph-only or kernel-only loss",
                                         "No published external baseline executed"]},
            "cache_entries": len(cache), "rows": rows}

def self_test():
    p = tuple([1.0 / len(HYPOTHESES)] * len(HYPOTHESES))
    assert len(HYPOTHESES) == 16
    for a in ACTIONS:
        assert math.isclose(sum(prob for _, prob in predictive(p, a)), 1.0, abs_tol=1e-12)
        for info in one_step_information(p, a).values():
            assert info >= 0.0
    quick = run_benchmark(max_horizon=2)
    assert len(quick["rows"]) == 2 * len(POLICIES)
    for horizon in (1, 2):
        oracle = [r for r in quick["rows"] if r["horizon"] == horizon and r["policy"] == "exact_bellman_oracle"][0]
        assert abs(oracle["regret_bits_vs_exact_oracle"]) <= 1e-10
        assert all(r["regret_bits_vs_exact_oracle"] >= 0.0 for r in quick["rows"] if r["horizon"] == horizon)
    return True

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--source-git-blob", default="UNBOUND")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test(); print("EXACT_FINITE_HORIZON_ORACLE_TESTS_PASSED"); return 0
        result = run_benchmark(args.horizon)
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
