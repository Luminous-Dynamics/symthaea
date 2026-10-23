#!/usr/bin/env python3
"""Exact small-SCM benchmark for action-dependent intervention target-set kernels.

Standard-library only. This is a research benchmark, not a production causal-discovery
system. It distinguishes an attempted target i from the realized target subset S.
All members of S are clamped to the action's requested value (1); other variables
follow their SCM equations. The finite learner prior contains all 330 distributions over the full 8-element power set whose masses are multiples of one quarter.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
import random
from pathlib import Path

SCHEMA = "symthaea-action-subset-kernel-benchmark-v1"
VARIABLES = (0, 1, 2)
SUBSETS = tuple(range(8))
SENSOR_SENSITIVITY = 0.90
SENSOR_SPECIFICITY = 0.90
CHILD_FLIP_PROB = 0.10
DEFAULT_REPLICATES = 24
DEFAULT_SAMPLES_PER_ACTION = 12

# Parent tuples plus a topological evaluation order. Root variables are fair Bernoulli;
# non-roots compute XOR(parents) with an independent 0.1 bit-flip noise.
MODELS = (
    {"name": "chain_0_to_1_to_2", "parents": ((), (0,), (1,)), "order": (0, 1, 2)},
    {"name": "fork_0_to_1_2", "parents": ((), (0,), (0,)), "order": (0, 1, 2)},
    {"name": "collider_0_1_to_2", "parents": ((), (), (0, 1)), "order": (0, 1, 2)},
    {"name": "reverse_chain_2_to_1_to_0", "parents": ((1,), (2,), ()), "order": (2, 1, 0)},
)
GRAPH_NAMES = tuple(m["name"] for m in MODELS)
MODES = ("outcome_only", "noisy_target_flag", "exact_target_subset")


def subset_mask(nodes):
    out = 0
    for node in nodes:
        if node not in VARIABLES:
            raise ValueError(f"invalid variable {node}")
        out |= 1 << node
    return out


def mask_nodes(mask):
    if not isinstance(mask, int) or mask < 0 or mask > 7:
        raise ValueError(f"subset mask must be in 0..7, got {mask!r}")
    return tuple(v for v in VARIABLES if mask & (1 << v))


def _weak_compositions(total, parts):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _weak_compositions(total - first, parts - 1):
            yield (first,) + rest


def kernel_catalog():
    """All categorical distributions with four integer mass units across 8 subsets."""
    counts_catalog = tuple(_weak_compositions(4, 8))
    distributions = tuple(tuple(count / 4.0 for count in counts) for counts in counts_catalog)
    assert len(distributions) == 330
    return distributions, counts_catalog

KERNELS, KERNEL_COUNTS = kernel_catalog()
KERNEL_INDEX = {counts: i for i, counts in enumerate(KERNEL_COUNTS)}


def structural_outcomes(model_idx, subset_mask_value):
    """Exact P(Y=y | graph, realized intervention subset), with outcomes as 3-bit masks."""
    model = MODELS[model_idx]
    parents, order = model["parents"], model["order"]
    forced = set(mask_nodes(subset_mask_value))
    roots = tuple(v for v in VARIABLES if not parents[v])
    children = tuple(v for v in VARIABLES if parents[v])
    result = [0.0] * 8
    # Exogenous root bits are fair; child bit-flips are independent Bernoulli(0.1).
    for root_bits in itertools.product((0, 1), repeat=len(roots)):
        root_values = dict(zip(roots, root_bits))
        p_roots = 0.5 ** len(roots)
        for flips in itertools.product((0, 1), repeat=len(children)):
            flip_values = dict(zip(children, flips))
            p_flips = math.prod(CHILD_FLIP_PROB if b else 1.0 - CHILD_FLIP_PROB for b in flips)
            x = [0, 0, 0]
            for node in order:
                if node in forced:
                    x[node] = 1
                elif not parents[node]:
                    x[node] = root_values[node]
                else:
                    parity = 0
                    for parent in parents[node]:
                        parity ^= x[parent]
                    x[node] = parity ^ flip_values[node]
            y = x[0] | (x[1] << 1) | (x[2] << 2)
            result[y] += p_roots * p_flips
    total = sum(result)
    if not math.isclose(total, 1.0, abs_tol=1e-12, rel_tol=0.0):
        raise AssertionError(f"SCM outcome distribution does not normalize: {total}")
    return tuple(result)


WORLD = tuple(tuple(structural_outcomes(g, s) for s in SUBSETS) for g in range(len(MODELS)))


def noisy_flag_prob(z, attempted_node, realized_subset):
    is_target_only = realized_subset == (1 << attempted_node)
    p_one = SENSOR_SENSITIVITY if is_target_only else 1.0 - SENSOR_SPECIFICITY
    return p_one if z == 1 else 1.0 - p_one


def observation_likelihoods(attempted_node, model_idx, kernel, mode):
    """P(observation category | graph, action-specific D_i) for all categories."""
    if mode not in MODES:
        raise ValueError(f"unknown observation mode {mode}")
    if mode == "outcome_only":
        likelihood = [0.0] * 8
        for s in SUBSETS:
            for y in SUBSETS:
                likelihood[y] += kernel[s] * WORLD[model_idx][s][y]
    elif mode == "noisy_target_flag":
        likelihood = [0.0] * 16  # category = y*2 + observed flag
        for s in SUBSETS:
            for y in SUBSETS:
                for z in (0, 1):
                    likelihood[y * 2 + z] += (
                        kernel[s] * WORLD[model_idx][s][y]
                        * noisy_flag_prob(z, attempted_node, s)
                    )
    else:
        likelihood = [0.0] * 64  # category = y*8 + exact realized subset mask
        for s in SUBSETS:
            for y in SUBSETS:
                likelihood[y * 8 + s] += kernel[s] * WORLD[model_idx][s][y]
    total = sum(likelihood)
    if not math.isclose(total, 1.0, abs_tol=1e-12, rel_tol=0.0):
        raise AssertionError(f"observation likelihood failed to normalize ({mode}): {total}")
    return tuple(likelihood)


def make_likelihood_tables():
    # tables[action][mode][graph][kernel hypothesis] -> categorical likelihood vector
    return {
        action: {
            mode: tuple(
                tuple(observation_likelihoods(action, g, kernel, mode) for kernel in KERNELS)
                for g in range(len(MODELS))
            )
            for mode in MODES
        }
        for action in VARIABLES
    }

TABLES = make_likelihood_tables()


def entropy(probabilities):
    total = sum(probabilities)
    if total <= 0.0:
        return 0.0
    out = 0.0
    for p in probabilities:
        if p > 0:
            q = p / total
            out -= q * math.log2(q)
    return out


def expected_information(action, mode, tables=TABLES):
    """Exact mutual information from one probe over uniform graph × kernel hypotheses."""
    rows = tables[action][mode]
    outcome_count = len(rows[0][0])
    graph_count, kernel_count = len(MODELS), len(KERNELS)
    prior = 1.0 / (graph_count * kernel_count)
    predictive = [sum(prior * rows[g][d][o] for g in range(graph_count)
                      for d in range(kernel_count)) for o in range(outcome_count)]
    hg0, hd0 = math.log2(graph_count), math.log2(kernel_count)
    hj0 = math.log2(graph_count * kernel_count)
    e_hg = e_hd = e_hjoint = 0.0
    for o, po in enumerate(predictive):
        if po <= 0:
            continue
        joint = [[prior * rows[g][d][o] / po for d in range(kernel_count)]
                 for g in range(graph_count)]
        pg = [sum(joint[g]) for g in range(graph_count)]
        pd = [sum(joint[g][d] for g in range(graph_count)) for d in range(kernel_count)]
        e_hg += po * entropy(pg)
        e_hd += po * entropy(pd)
        e_hjoint += po * entropy([p for row in joint for p in row])
    result = {
        "action_target": action,
        "mode": mode,
        "graph_information_bits": hg0 - e_hg,
        "action_kernel_information_bits": hd0 - e_hd,
        "joint_graph_kernel_information_bits": hj0 - e_hjoint,
        "prior_graph_entropy_bits": hg0,
        "prior_action_kernel_entropy_bits": hd0,
        "outcome_categories": outcome_count,
    }
    for key in ("graph_information_bits", "action_kernel_information_bits", "joint_graph_kernel_information_bits"):
        # Only remove harmless round-off; a substantive negative value is a bug.
        if result[key] < -1e-10:
            raise AssertionError(f"negative exact mutual information: {key}={result[key]}")
        result[key] = max(0.0, result[key])
    return result


def sample_categorical(probabilities, rng):
    r = rng.random()
    running = 0.0
    for i, p in enumerate(probabilities):
        running += p
        if r < running or i == len(probabilities) - 1:
            return i
    raise AssertionError("unreachable categorical sample")


def true_kernels(kernel_regime):
    distributions = []
    catalog_ids = []
    for i in VARIABLES:
        target = 1 << i
        pair = target | (1 << ((i + 1) % 3))
        if kernel_regime == "in_support":
            d = [0.0] * 8
            d[target] = 0.5
            d[pair] = 0.5
            distributions.append(tuple(d))
            counts = tuple(int(round(probability * 4)) for probability in d)
            catalog_ids.append(KERNEL_INDEX[counts])
        elif kernel_regime == "out_of_support":
            d = [0.0] * 8
            d[target] = 0.40
            d[pair] = 0.35
            d[7] += 0.25
            distributions.append(tuple(d))
            catalog_ids.append(None)
        else:
            raise ValueError(f"unknown kernel regime {kernel_regime}")
    return tuple(distributions), tuple(catalog_ids)


def make_dataset(model_idx, actual_distributions, seed, samples_per_action):
    """Generate paired latent target subsets/outcomes and a noisy sensor bit per row."""
    rng = random.Random(seed)
    data = []
    for action in VARIABLES:
        d = actual_distributions[action]
        for _ in range(samples_per_action):
            s = sample_categorical(d, rng)
            y = sample_categorical(WORLD[model_idx][s], rng)
            z = 1 if rng.random() < noisy_flag_prob(1, action, s) else 0
            data.append({"action": action, "y": y, "subset": s, "flag": z})
    return data


def logsumexp(values):
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return -math.inf
    m = max(finite)
    return m + math.log(sum(math.exp(v - m) for v in finite))


def softmax_log(log_weights):
    norm = logsumexp(log_weights)
    if not math.isfinite(norm):
        raise ValueError("all hypotheses assigned zero likelihood; posterior is undefined")
    vals = [math.exp(v - norm) if math.isfinite(v) else 0.0 for v in log_weights]
    total = sum(vals)
    return tuple(v / total for v in vals)


def encode_observation(row, mode):
    if mode == "outcome_only":
        return row["y"]
    if mode == "noisy_target_flag":
        return row["y"] * 2 + row["flag"]
    if mode == "exact_target_subset":
        return row["y"] * 8 + row["subset"]
    raise ValueError(mode)


def _safe_log(p):
    return math.log(p) if p > 0.0 else -math.inf


def posterior_fit(data, mode, tables=TABLES):
    """Exact Bayesian update; factorizes by action-specific D_i conditional on graph."""
    graph_count, kernel_count = len(MODELS), len(KERNELS)
    action_log_likelihoods = {}
    for action in VARIABLES:
        rows = [r for r in data if r["action"] == action]
        ll = [[0.0] * kernel_count for _ in range(graph_count)]
        for g in range(graph_count):
            for d in range(kernel_count):
                total = 0.0
                for row in rows:
                    obs = encode_observation(row, mode)
                    p = tables[action][mode][g][d][obs]
                    if p <= 0.0:
                        total = -math.inf
                        break
                    total += math.log(p)
                ll[g][d] = total
        action_log_likelihoods[action] = ll
    log_graph = []
    for g in range(graph_count):
        # Equal priors; constants -log(K) per action cancel in the normalized graph posterior.
        log_graph.append(sum(logsumexp(action_log_likelihoods[a][g]) for a in VARIABLES))
    graph_posterior = softmax_log(log_graph)
    kernel_posterior_by_graph = {}
    kernel_marginals = []
    for action in VARIABLES:
        conditional = []
        for g in range(graph_count):
            conditional.append(softmax_log(action_log_likelihoods[action][g]))
        kernel_posterior_by_graph[action] = tuple(conditional)
        marginal = [sum(graph_posterior[g] * conditional[g][d] for g in range(graph_count))
                    for d in range(kernel_count)]
        kernel_marginals.append(tuple(marginal))
    mean_distributions = []
    for action in VARIABLES:
        mean_d = [0.0] * 8
        for d, p_d in enumerate(kernel_marginals[action]):
            for s in SUBSETS:
                mean_d[s] += p_d * KERNELS[d][s]
        mean_distributions.append(tuple(mean_d))
    return {
        "graph_posterior": graph_posterior,
        "kernel_marginals": tuple(kernel_marginals),
        "posterior_mean_kernels": tuple(mean_distributions),
    }


def graph_posterior_baseline(data, likelihood_fn):
    """Graph posterior under a supplied action likelihood, ignoring telemetry."""
    log_weights = []
    for g in range(len(MODELS)):
        total = 0.0
        for row in data:
            p = likelihood_fn(g, row["action"], row["y"])
            if p <= 0:
                total = -math.inf
                break
            total += math.log(p)
        log_weights.append(total)
    return softmax_log(log_weights)


def baseline_hard_do(data):
    return graph_posterior_baseline(
        data,
        lambda g, action, y: WORLD[g][1 << action][y],
    )


def baseline_oracle_kernel(data, actual_distributions):
    def likelihood(g, action, y):
        return sum(actual_distributions[action][s] * WORLD[g][s][y] for s in SUBSETS)
    return graph_posterior_baseline(data, likelihood)


def brier_score(probabilities, true_idx):
    return sum((p - (1.0 if i == true_idx else 0.0)) ** 2 for i, p in enumerate(probabilities))


def graph_metrics(posterior, true_idx):
    truth_p = max(posterior[true_idx], 1e-300)
    guess = max(range(len(posterior)), key=lambda i: (posterior[i], -i))
    return {
        "graph_map_correct": guess == true_idx,
        "graph_log_loss_bits": -math.log2(truth_p),
        "graph_brier_score": brier_score(posterior, true_idx),
        "graph_true_posterior": posterior[true_idx],
    }


def run_benchmark(replicates=DEFAULT_REPLICATES, samples_per_action=DEFAULT_SAMPLES_PER_ACTION):
    if replicates < 1 or samples_per_action < 1:
        raise ValueError("replicates and samples_per_action must be positive")
    eig_rows = [expected_information(action, mode) for action in VARIABLES for mode in MODES]
    result_rows = []
    trial_id = 0
    for graph_idx in range(len(MODELS)):
        for kernel_regime_idx, kernel_regime in enumerate(("in_support", "out_of_support")):
            actual_dists, catalog_ids = true_kernels(kernel_regime)
            for rep in range(replicates):
                seed = 20261023 + graph_idx * 100000 + kernel_regime_idx * 10000 + rep
                data = make_dataset(graph_idx, actual_dists, seed, samples_per_action)
                hard_p = baseline_hard_do(data)
                oracle_p = baseline_oracle_kernel(data, actual_dists)
                for mode in MODES:
                    posterior = posterior_fit(data, mode)
                    gmetrics = graph_metrics(posterior["graph_posterior"], graph_idx)
                    hard_metrics = graph_metrics(hard_p, graph_idx)
                    oracle_metrics = graph_metrics(oracle_p, graph_idx)
                    kernel_l1 = []
                    exact_kernel_mass = []
                    for action in VARIABLES:
                        estimated = posterior["posterior_mean_kernels"][action]
                        truth = actual_dists[action]
                        kernel_l1.append(sum(abs(a - b) for a, b in zip(estimated, truth)))
                        if catalog_ids[action] is not None:
                            exact_kernel_mass.append(posterior["kernel_marginals"][action][catalog_ids[action]])
                    trial_id += 1
                    result_rows.append({
                        "trial_id": trial_id,
                        "seed": seed,
                        "true_graph": GRAPH_NAMES[graph_idx],
                        "true_graph_index": graph_idx,
                        "kernel_regime": kernel_regime,
                        "observation_mode": mode,
                        "samples_per_action": samples_per_action,
                        **gmetrics,
                        "mean_target_kernel_l1": sum(kernel_l1) / len(kernel_l1),
                        "per_action_target_kernel_l1": kernel_l1,
                        "mean_exact_kernel_posterior_mass": (
                            sum(exact_kernel_mass) / len(exact_kernel_mass) if exact_kernel_mass else None
                        ),
                        "hard_do_graph_map_correct": hard_metrics["graph_map_correct"],
                        "hard_do_graph_log_loss_bits": hard_metrics["graph_log_loss_bits"],
                        "hard_do_graph_brier_score": hard_metrics["graph_brier_score"],
                        "oracle_known_kernel_graph_map_correct": oracle_metrics["graph_map_correct"],
                        "oracle_known_kernel_graph_log_loss_bits": oracle_metrics["graph_log_loss_bits"],
                        "oracle_known_kernel_graph_brier_score": oracle_metrics["graph_brier_score"],
                    })
    group_metrics = {}
    keys = ("graph_map_correct", "graph_log_loss_bits", "graph_brier_score", "graph_true_posterior",
            "mean_target_kernel_l1", "mean_exact_kernel_posterior_mass", "hard_do_graph_map_correct",
            "hard_do_graph_log_loss_bits", "hard_do_graph_brier_score", "oracle_known_kernel_graph_map_correct",
            "oracle_known_kernel_graph_log_loss_bits", "oracle_known_kernel_graph_brier_score")
    for regime in ("in_support", "out_of_support"):
        for mode in MODES:
            group = [r for r in result_rows if r["kernel_regime"] == regime and r["observation_mode"] == mode]
            metrics = {}
            for key in keys:
                values = [float(r[key]) for r in group if r[key] is not None]
                metrics[key + "_mean"] = sum(values) / len(values) if values else None
            metrics["n_trials"] = len(group)
            group_metrics[f"{regime}/{mode}"] = metrics
    eig_group = {}
    for mode in MODES:
        group = [r for r in eig_rows if r["mode"] == mode]
        eig_group[mode] = {
            key: sum(r[key] for r in group) / len(group)
            for key in ("graph_information_bits", "action_kernel_information_bits", "joint_graph_kernel_information_bits")
        }
    return {
        "schema": SCHEMA,
        "status": "EXACT_SMALL_SCM_PILOT_LOCAL_NOT_EXTERNAL_QUALIFICATION",
        "protocol": {
            "variables": 3,
            "candidate_graph_models": list(GRAPH_NAMES),
            "action": "attempt intervention on node i with requested value 1, for i in {0,1,2}",
            "realized_subset_semantics": "every node in realized subset S is clamped to 1; all other nodes follow the structural equations",
            "structural_equations": "root nodes are independent fair Bernoulli; non-root nodes are XOR(parent values) with independent 0.1 bit-flip noise",
            "target_subset_universe": "all 8 subsets of {0,1,2}, including empty set and all multi-node subsets",
            "kernel_prior": "per action, uniform over all 330 categorical distributions on the 8 target subsets with probabilities in quarter increments; a disclosed finite grid, not the unrestricted continuous simplex",
            "truth_regimes": [
                "in_support: each action's kernel has 0.5 mass on its intended singleton and 0.5 on an adjacent pair",
                "out_of_support: each action has 0.40 mass on intended singleton, 0.35 on adjacent pair, 0.25 on all three nodes; this is outside the quarter-step learner grid",
            ],
            "telemetry": {
                "outcome_only": "observe full 3-bit outcome Y only",
                "noisy_target_flag": "observe Y and a flag for S == {attempted target}, sensitivity/specificity each 0.90",
                "exact_target_subset": "observe Y and the exact realized subset S",
            },
            "samples_per_action": samples_per_action,
            "replicates_per_graph_kernel_regime": replicates,
            "total_paired_worlds": len(MODELS) * 2 * replicates,
            "total_mode_rows": len(result_rows),
            "seed_policy": "seed = 20261023 + 100000*graph_index + 10000*kernel_regime_index + replicate_index",
            "baselines": {
                "hard_do": "misspecified graph posterior that treats each attempt as a perfect intervention on the named target and ignores telemetry",
                "oracle_known_kernel": "graph posterior that knows the true action-dependent subset distribution but observes Y only; upper-reference, not a deployable method",
            },
            "limitations": [
                "three-node Boolean worlds and a small graph family are a proof-of-execution fixture, not external qualification",
                "the finite Bayesian kernel prior uses quarter-step masses, not the unrestricted simplex; the out-of-support lane intentionally violates the grid",
                "no ACDB, TCD-Arena, or Choo et al. baseline implementation has been executed in this pilot",
                "telemetry flag noise is synthetic and should not be presented as a measured hardware sensor rate",
            ],
        },
        "exact_one_probe_expected_information": eig_rows,
        "expected_information_mean_by_mode": eig_group,
        "aggregate_metrics": group_metrics,
        "rows": result_rows,
    }


def self_test():
    catalog, pairs = kernel_catalog()
    assert len(catalog) == len(pairs) == 330
    assert all(math.isclose(sum(d), 1.0, abs_tol=1e-12) for d in catalog)
    assert all(sum(p > 0 for p in d) <= 4 for d in catalog)
    for g in range(len(MODELS)):
        for s in SUBSETS:
            assert math.isclose(sum(WORLD[g][s]), 1.0, abs_tol=1e-12)
    for action in VARIABLES:
        for mode in MODES:
            for g in range(len(MODELS)):
                for d in catalog:
                    likelihood = observation_likelihoods(action, g, d, mode)
                    assert math.isclose(sum(likelihood), 1.0, abs_tol=1e-12)
    # The full S observation contains Y and can synthesize either weaker observation,
    # so expected information for the joint (graph, D_i) state cannot decrease.
    for action in VARIABLES:
        infos = [expected_information(action, mode) for mode in MODES]
        for key in ("graph_information_bits", "action_kernel_information_bits", "joint_graph_kernel_information_bits"):
            assert infos[0][key] <= infos[1][key] + 1e-10, (action, key, infos)
            assert infos[1][key] <= infos[2][key] + 1e-10, (action, key, infos)
    dists, ids = true_kernels("in_support")
    assert all(i is not None for i in ids)
    assert all(math.isclose(sum(d), 1.0, abs_tol=1e-12) for d in dists)
    ood, ood_ids = true_kernels("out_of_support")
    assert all(i is None for i in ood_ids)
    data = make_dataset(0, dists, 20261023, 2)
    for mode in MODES:
        p = posterior_fit(data, mode)
        assert math.isclose(sum(p["graph_posterior"]), 1.0, abs_tol=1e-12)
        for marginal in p["kernel_marginals"]:
            assert math.isclose(sum(marginal), 1.0, abs_tol=1e-12)
        for mean_d in p["posterior_mean_kernels"]:
            assert math.isclose(sum(mean_d), 1.0, abs_tol=1e-12)
    quick = run_benchmark(replicates=1, samples_per_action=2)
    assert len(quick["rows"]) == 4 * 2 * 1 * len(MODES)
    assert len(quick["exact_one_probe_expected_information"]) == 9
    assert all(0.0 <= r["graph_true_posterior"] <= 1.0 for r in quick["rows"])
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--samples-per-action", type=int, default=DEFAULT_SAMPLES_PER_ACTION)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-git-blob", default="UNBOUND")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            print("ACTION_SUBSET_KERNEL_SELF_TESTS_PASSED")
            return 0
        result = run_benchmark(args.replicates, args.samples_per_action)
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
