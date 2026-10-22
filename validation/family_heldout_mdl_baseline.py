#!/usr/bin/env python3
"""Small-graph enumerative MDL reference baseline for family-held-out Boolean SCMs.

Exhaustively selects among all labeled DAGs for <=4 nodes, using a saturated
Boolean truth table per candidate parent set and known symmetric mechanism noise. Parent sets and truth tables have an explicit code length.
Compares equal-total-sample observational-only data with mixed observational +
hard-interventional data. This is a transparent discrete-model baseline, not a state-of-the-art
method or an ACDB reproduction.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
from pathlib import Path

from family_heldout_scm_benchmark import (
    MECHANISMS, TOPOLOGIES, MECHANISM_INDEX, TOPOLOGY_INDEX,
    SCM, graph_scores, make_scm, sample_scm, split_label, topological_order,
)

SCHEMA = "symthaea-family-heldout-mdl-baseline-v1"

def parent_options(node_count: int, node: int) -> tuple[tuple[int, ...], ...]:
    candidates = tuple(i for i in range(node_count) if i != node)
    result = []
    for mask in range(1 << len(candidates)):
        result.append(tuple(candidates[i] for i in range(len(candidates)) if (mask >> i) & 1))
    return tuple(result)


def enumerate_dags(node_count: int) -> tuple[tuple[tuple[int, ...], ...], ...]:
    if not 2 <= node_count <= 4:
        raise ValueError("exhaustive MDL baseline supports 2..=4 nodes")
    options = [parent_options(node_count, node) for node in range(node_count)]
    dags = []
    for parent_sets in itertools.product(*options):
        parents = tuple(tuple(ps) for ps in parent_sets)
        try:
            topological_order(parents)
        except ValueError:
            continue
        dags.append(parents)
    return tuple(dags)


def _node_counts(node: int, parents: tuple[int, ...], batches: list[dict], node_count: int):
    counts = [[0, 0] for _ in range(1 << len(parents))]
    n_effective = 0
    for batch in batches:
        intervention = batch.get("intervention")
        if intervention is not None and int(intervention[0]) == node:
            continue  # the intervened node's structural equation is replaced
        for row in batch["rows"]:
            context = 0
            for bit, parent in enumerate(parents):
                context |= (int(row[parent]) & 1) << bit
            outcome = int(row[node])
            counts[context][outcome] += 1
            n_effective += 1
    if n_effective < 1:
        raise ValueError(f"no usable samples for node {node}")
    return counts, n_effective


def local_mdl_score(node: int, parents: tuple[int, ...], batches: list[dict],
                    node_count: int, assumed_noise: float) -> tuple[float, int]:
    """Exact negative log-likelihood plus enumerative code length, all in bits."""
    if not 0.0 < assumed_noise < 0.5:
        raise ValueError("assumed_noise must be strictly between 0 and 0.5")
    counts, n_effective = _node_counts(node, parents, batches, node_count)
    log_correct, log_error = math.log2(1.0 - assumed_noise), math.log2(assumed_noise)
    log_likelihood_bits = 0.0
    for zeros, ones in counts:
        correct = max(zeros, ones)
        incorrect = min(zeros, ones)
        log_likelihood_bits += correct * log_correct + incorrect * log_error
    truth_table_bits = 1 << len(parents)
    parent_set_bits = math.log2(math.comb(node_count - 1, len(parents)))
    mdl_bits = -log_likelihood_bits + parent_set_bits + truth_table_bits
    return mdl_bits, truth_table_bits


def select_dag_mdl(batches: list[dict], node_count: int, assumed_noise: float = 0.1) -> dict:
    """Return minimum-MDL DAG, with deterministic tie-breaks by sparsity then parents."""
    if not batches:
        raise ValueError("at least one sample batch is required")
    options = [parent_options(node_count, node) for node in range(node_count)]
    local = {}
    for node in range(node_count):
        for parents in options[node]:
            local[(node, parents)] = local_mdl_score(node, parents, batches, node_count, assumed_noise)[0]
    best_parents = None
    best_key = None
    for parents in enumerate_dags(node_count):
        score = sum(local[(node, parents[node])] for node in range(node_count))
        edge_count = sum(len(ps) for ps in parents)
        key = (round(score, 12), edge_count, parents)
        if best_key is None or key < best_key:
            best_key, best_parents = key, parents
    if best_parents is None:
        raise AssertionError("no DAG candidate was enumerated")
    edges = tuple(sorted((parent, child) for child, ps in enumerate(best_parents) for parent in ps))
    return {"parents": best_parents, "edges": edges, "mdl_bits": best_key[0],
            "edge_count": len(edges), "candidate_dag_count": len(enumerate_dags(node_count))}


def sample_arms(scm: SCM, observation_count: int = 512, intervention_observation_count: int = 256,
                per_intervention_count: int = 32, noise: float = 0.1) -> tuple[list[dict], list[dict]]:
    """Create equal-total-row arms: observational only vs obs + all node/value hard-do batches."""
    if observation_count < 1 or intervention_observation_count < 1 or per_intervention_count < 1:
        raise ValueError("all sample counts must be positive")
    n = scm.node_count
    cell_seed = scm.seed * 100000 + TOPOLOGY_INDEX[scm.topology_family] * 1000 + MECHANISM_INDEX[scm.mechanism_family] * 100
    observational = [sample_scm(scm, observation_count, cell_seed + 1, noise_probability=noise)]
    mixed = [sample_scm(scm, intervention_observation_count, cell_seed + 2, noise_probability=noise)]
    action_id = 0
    for target in range(n):
        for value in (0, 1):
            mixed.append(sample_scm(scm, per_intervention_count, cell_seed + 10 + action_id,
                                    noise_probability=noise, intervention=(target, value)))
            action_id += 1
    obs_total = sum(len(batch["rows"]) for batch in observational)
    mixed_total = sum(len(batch["rows"]) for batch in mixed)
    if obs_total != mixed_total:
        raise AssertionError(f"sample budgets differ: observational={obs_total}, mixed={mixed_total}")
    return observational, mixed


def _mean(rows: list[dict], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows) if rows else 0.0


def _bootstrap_cluster_ci(clusters: dict[str, list[float]], seed: int, replicates: int = 2000) -> list[float] | None:
    """Percentile CI resampling topology×mechanism cells, not individual near-duplicate seeds."""
    values = [vals for vals in clusters.values() if vals]
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(replicates):
        picked = [rng.choice(values) for _ in range(len(values))]
        means.append(statistics.fmean(statistics.fmean(block) for block in picked))
    means.sort()
    return [means[int(0.025 * (replicates - 1))], means[int(0.975 * (replicates - 1))]]


def run_benchmark(cases_per_cell: int = 4, noise: float = 0.1,
                  observation_count: int = 512, intervention_observation_count: int = 256,
                  per_intervention_count: int = 32, source_git_blob: str = "UNBOUND") -> dict:
    suite_rows = []
    for topology in TOPOLOGIES:
        for mechanism in MECHANISMS:
            for seed in range(cases_per_cell):
                scm = make_scm(4, topology, mechanism, seed)
                obs_batches, mixed_batches = sample_arms(scm, observation_count,
                    intervention_observation_count, per_intervention_count, noise)
                obs_fit = select_dag_mdl(obs_batches, scm.node_count, assumed_noise=noise)
                mixed_fit = select_dag_mdl(mixed_batches, scm.node_count, assumed_noise=noise)
                truth = scm.edges()
                row = {"world_id": scm.world_id, "seed": seed,
                       "topology_family": topology, "mechanism_family": mechanism,
                       "split": split_label(topology, mechanism), "node_count": scm.node_count,
                       "total_samples_per_arm": observation_count,
                       "noise_probability": noise, "observational_mdl_bits": obs_fit["mdl_bits"],
                       "interventional_mdl_bits": mixed_fit["mdl_bits"]}
                for arm, fit in (("observational", obs_fit), ("observational_plus_interventions", mixed_fit)):
                    metrics = graph_scores(truth, fit["edges"], scm.node_count)
                    for key in ("skeleton_precision", "skeleton_recall", "skeleton_f1",
                                "directed_precision", "directed_recall", "directed_f1", "shd"):
                        row[f"{arm}_{key}"] = metrics[key]
                    row[f"{arm}_predicted_edges"] = [list(edge) for edge in fit["edges"]]
                row["paired_delta_skeleton_f1_intervention_minus_observation"] = (
                    row["observational_plus_interventions_skeleton_f1"] - row["observational_skeleton_f1"])
                row["paired_delta_directed_f1_intervention_minus_observation"] = (
                    row["observational_plus_interventions_directed_f1"] - row["observational_directed_f1"])
                row["paired_delta_shd_intervention_minus_observation"] = (
                    row["observational_plus_interventions_shd"] - row["observational_shd"])
                suite_rows.append(row)
    grouped = {}
    for split in ("train_id", "topology_ood", "mechanism_ood", "combined_ood"):
        picked = [row for row in suite_rows if row["split"] == split]
        summary = {"case_count": len(picked)}
        for arm in ("observational", "observational_plus_interventions"):
            summary[arm] = {metric: _mean(picked, f"{arm}_{metric}") for metric in (
                "skeleton_f1", "directed_f1", "shd", "skeleton_precision", "skeleton_recall")}
        paired = {}
        for metric in ("skeleton_f1", "directed_f1", "shd"):
            delta_key = f"paired_delta_{metric}_intervention_minus_observation"
            clusters = {}
            for row in picked:
                cluster = f'{row["topology_family"]}|{row["mechanism_family"]}'
                clusters.setdefault(cluster, []).append(row[delta_key])
            paired[metric] = {"mean_delta": _mean(picked, delta_key),
                              "cluster_bootstrap_95_ci": _bootstrap_cluster_ci(clusters, 1701 + len(split) + len(metric))}
        summary["paired_deltas"] = paired
        grouped[split] = summary
    return {
        "schema": SCHEMA,
        "status": "LOCAL_BASELINE_EVALUATION_NOT_SOTA",
        "source_git_blob": source_git_blob,
        "assumed_noise_equals_data_generating_noise": noise,
        "protocol": {
            "node_count": 4, "topology_families": list(TOPOLOGIES), "mechanism_families": list(MECHANISMS),
            "cases_per_topology_mechanism_cell": cases_per_cell,
            "total_worlds": len(suite_rows), "sample_budget_per_arm": observation_count,
            "observational_arm": f"{observation_count} observational samples",
            "interventional_arm": f"{intervention_observation_count} observational + {2*4*per_intervention_count} hard-do samples",
            "hard_do_actions": "every node intervened at 0 and 1; target equation replaced and target rows excluded from that node's local likelihood",
            "selection": "exhaustive labeled DAG enumeration (543 candidates for n=4); exact negative log-likelihood in bits plus log2(comb(n-1,k)) parent-set code bits and 2^k truth-table code bits per node; tie-break by fewer edges then lexicographic parent sets",
            "family_splits": "topology/mechanism family holdouts from family_heldout_scm_benchmark.py",
            "uncertainty": "paired percentile bootstrap over topology×mechanism cells; descriptive uncertainty over this finite generated suite, not population-level guarantee",
        },
        "split_counts": {key: grouped[key]["case_count"] for key in grouped},
        "summary_by_split": grouped,
        "rows": suite_rows,
        "limitations": [
            "Known noise probability is supplied to the scorer; unknown-noise estimation is not evaluated",
            "Four-node Boolean SCMs only; no latent confounding, intervention failure or continuous variables",
            "MDL estimator is a transparent finite truth-table baseline and is not a reproduction of PC, NOTEARS or ACDB agents",
            "Family groups are small and finite; bootstrap intervals are descriptive, not evidence of broad generalization",
            "No claim of outperforming SOTA; adapter parity and additional baseline implementations remain required",
        ],
    }


def self_test() -> None:
    dags = enumerate_dags(4)
    assert len(dags) == 543, f"expected 543 four-node DAGs, got {len(dags)}"
    assert all(len(set(ps)) == len(ps) for dag in dags for ps in dag)
    scm = make_scm(4, "chain", "and", 3)
    obs, mixed = sample_arms(scm, 24, 16, 1, 0.1)
    assert sum(len(b["rows"]) for b in obs) == sum(len(b["rows"]) for b in mixed) == 24
    assert len(mixed) == 1 + 2 * scm.node_count
    obs_fit = select_dag_mdl(obs, 4, 0.1)
    mixed_fit = select_dag_mdl(mixed, 4, 0.1)
    assert len(obs_fit["edges"]) <= 12 and len(mixed_fit["edges"]) <= 12
    try: enumerate_dags(5)
    except ValueError: pass
    else: raise AssertionError("unsupported exhaustive graph size not rejected")
    try: select_dag_mdl([], 4, 0.1)
    except ValueError: pass
    else: raise AssertionError("empty data not rejected")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-per-cell", type=int, default=4)
    parser.add_argument("--noise", type=float, default=0.1)
    parser.add_argument("--observations", type=int, default=512)
    parser.add_argument("--intervention-observations", type=int, default=256)
    parser.add_argument("--per-intervention", type=int, default=32)
    parser.add_argument("--source-git-blob", default="UNBOUND")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test(); print("FAMILY_HELDOUT_MDL_SELF_TESTS_PASSED"); return 0
        if not 1 <= args.cases_per_cell <= 20: raise ValueError("cases-per-cell must be in 1..=20")
        result = run_benchmark(args.cases_per_cell, args.noise, args.observations,
                               args.intervention_observations, args.per_intervention, args.source_git_blob)
        encoded = (json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_bytes(encoded)
            print(f"wrote {args.output} bytes={len(encoded)} sha256={hashlib.sha256(encoded).hexdigest()}")
        else: sys.stdout.buffer.write(encoded)
        return 0
    except (OSError, ValueError, AssertionError, RuntimeError) as error:
        print(f"NOT QUALIFIED: {error}", file=sys.stderr); return 2

if __name__ == "__main__": raise SystemExit(main())
