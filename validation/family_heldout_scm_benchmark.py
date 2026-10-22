#!/usr/bin/env python3
"""Deterministic family-held-out SCM suite and graph-recovery scoring contract.

Standard-library only. This generates ground-truth Boolean SCMs and benchmark
split metadata; it does not claim that a discovery algorithm has been evaluated.
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "symthaea-family-heldout-scm-v1"
TOPOLOGIES = ("chain", "fork", "collider", "diamond", "sparse_random", "dense_random")
MECHANISMS = ("and", "or", "xor", "threshold", "random_essential")
TOPOLOGY_HOLDOUT = frozenset(("collider", "diamond", "dense_random"))
MECHANISM_HOLDOUT = frozenset(("or", "xor"))
TOPOLOGY_INDEX = {name: i for i, name in enumerate(TOPOLOGIES)}
MECHANISM_INDEX = {name: i for i, name in enumerate(MECHANISMS)}

@dataclass(frozen=True)
class SCM:
    world_id: str
    seed: int
    node_count: int
    topology_family: str
    mechanism_family: str
    parents: tuple[tuple[int, ...], ...]
    truth_tables: tuple[int, ...]

    def edges(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted((parent, child) for child, ps in enumerate(self.parents) for parent in ps))

    def graph_key(self) -> tuple[tuple[int, ...], ...]:
        return self.parents


def split_label(topology: str, mechanism: str) -> str:
    topology_held = topology in TOPOLOGY_HOLDOUT
    mechanism_held = mechanism in MECHANISM_HOLDOUT
    if topology_held and mechanism_held:
        return "combined_ood"
    if topology_held:
        return "topology_ood"
    if mechanism_held:
        return "mechanism_ood"
    return "train_id"


def topology_parents(n: int, family: str, seed: int) -> tuple[tuple[int, ...], ...]:
    if n < 4:
        raise ValueError("node_count must be >= 4")
    if family not in TOPOLOGIES:
        raise ValueError(f"unknown topology family: {family}")
    parents = [set() for _ in range(n)]
    if family == "chain":
        for child in range(1, n): parents[child].add(child - 1)
    elif family == "fork":
        for child in range(1, n): parents[child].add(0)
    elif family == "collider":
        parents[n - 1].update((0, 1))
    elif family == "diamond":
        parents[1].add(0); parents[2].add(0)
        parents[3].update((1, 2))
        for child in range(4, n): parents[child].add(child - 1)
    else:
        rng = random.Random(seed)
        order = list(range(n)); rng.shuffle(order)
        probability = 0.25 if family == "sparse_random" else 0.65
        for later in range(1, n):
            child = order[later]
            for earlier in range(later):
                parent = order[earlier]
                if rng.random() < probability: parents[child].add(parent)
    result = tuple(tuple(sorted(ps)) for ps in parents)
    remaining = set(range(n))
    while remaining:
        ready = {node for node in remaining if not (set(result[node]) & remaining)}
        if not ready: raise AssertionError("generator emitted a directed cycle")
        remaining -= ready
    return result


def parent_is_essential(table: int, parent_count: int, parent_index: int) -> bool:
    for assignment in range(1 << parent_count):
        if (assignment >> parent_index) & 1: continue
        paired = assignment | (1 << parent_index)
        if ((table >> assignment) & 1) != ((table >> paired) & 1): return True
    return False


def mechanism_table(k: int, family: str, rng: random.Random) -> int:
    if family not in MECHANISMS: raise ValueError(f"unknown mechanism family: {family}")
    if k == 0: return rng.randrange(2)
    table = 0
    for assignment in range(1 << k):
        bits = assignment.bit_count()
        if family == "and": out = int(bits == k)
        elif family == "or": out = int(bits > 0)
        elif family == "xor": out = bits & 1
        elif family == "threshold": out = int(bits >= (k + 1) // 2)
        else: out = -1
        if out >= 0 and out: table |= 1 << assignment
    if family == "random_essential":
        for _ in range(10000):
            table = rng.getrandbits(1 << k)
            if all(parent_is_essential(table, k, p) for p in range(k)): break
        else: raise RuntimeError("could not sample an essential mechanism")
    if not all(parent_is_essential(table, k, p) for p in range(k)):
        raise AssertionError(f"mechanism {family} has an inessential parent at indegree {k}")
    return table


def make_scm(node_count: int, topology: str, mechanism: str, seed: int) -> SCM:
    ti, mi = TOPOLOGY_INDEX[topology], MECHANISM_INDEX[mechanism]
    graph_seed = seed * 1009 + ti * 9176 + 37
    mechanism_seed = seed * 7919 + mi * 6553 + 101
    parents = topology_parents(node_count, topology, graph_seed)
    rng = random.Random(mechanism_seed)
    tables = tuple(mechanism_table(len(ps), mechanism, rng) for ps in parents)
    identity = {"schema": SCHEMA, "seed": seed, "n": node_count, "topology": topology,
                "mechanism": mechanism, "parents": parents, "tables": tables}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return SCM(digest[:20], seed, node_count, topology, mechanism, parents, tables)


def mechanism_output(scm: SCM, node: int, assignment: tuple[int, ...]) -> int:
    parent_assignment = 0
    for bit, parent in enumerate(scm.parents[node]):
        parent_assignment |= (assignment[parent] & 1) << bit
    return (scm.truth_tables[node] >> parent_assignment) & 1


def topological_order(parents: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    remaining = set(range(len(parents)))
    order = []
    while remaining:
        ready = sorted(node for node in remaining if not (set(parents[node]) & remaining))
        if not ready:
            raise ValueError("cannot simulate a cyclic SCM")
        order.extend(ready)
        remaining -= set(ready)
    return tuple(order)


def sample_scm(scm: SCM, sample_count: int, seed: int, noise_probability: float = 0.0,
               intervention: tuple[int, int] | None = None) -> dict:
    """Sample observational rows or rows under one hard do(X_i=value) intervention."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    if not 0.0 <= noise_probability < 0.5:
        raise ValueError("noise_probability must be in [0, 0.5)")
    if intervention is not None:
        target, value = intervention
        if not 0 <= target < scm.node_count or value not in (0, 1):
            raise ValueError("intervention must be (valid_node, binary_value)")
    order = topological_order(scm.parents)
    rng = random.Random(seed)
    rows = []
    for _ in range(sample_count):
        values = [0] * scm.node_count
        for node in order:
            if intervention is not None and node == intervention[0]:
                values[node] = intervention[1]
                continue
            predicted = mechanism_output(scm, node, tuple(values))
            if rng.random() < noise_probability:
                predicted ^= 1
            values[node] = predicted
        rows.append(values)
    identity = {
        "schema": "symthaea-scm-samples-v1", "world_id": scm.world_id,
        "sample_count": sample_count, "seed": seed,
        "noise_probability": noise_probability,
        "intervention": None if intervention is None else list(intervention),
        "rows": rows,
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return {**identity, "samples_sha256": hashlib.sha256(canonical).hexdigest()}


def graph_scores(true_edges: tuple[tuple[int, int], ...] | list[tuple[int, int]],
                 predicted_edges: tuple[tuple[int, int], ...] | list[tuple[int, int]], node_count: int) -> dict:
    """Score a fully directed graph. A reversal is one SHD edit but one FP + one FN."""
    truth, prediction = set(true_edges), set(predicted_edges)
    for label, edges in (("true", truth), ("predicted", prediction)):
        for u, v in edges:
            if not (0 <= u < node_count and 0 <= v < node_count) or u == v:
                raise ValueError(f"invalid {label} edge {(u, v)}")
    if len(truth) != len(true_edges) or len(prediction) != len(predicted_edges):
        raise ValueError("duplicate directed edges are not allowed")
    true_skeleton = {tuple(sorted(edge)) for edge in truth}
    pred_skeleton = {tuple(sorted(edge)) for edge in prediction}
    sk_tp = len(true_skeleton & pred_skeleton)
    sk_fp, sk_fn = len(pred_skeleton - true_skeleton), len(true_skeleton - pred_skeleton)
    tp = len(truth & prediction)
    fp, fn = len(prediction - truth), len(truth - prediction)
    def ratio(num: int, den: int) -> float: return num / den if den else 0.0
    def f1(p: float, r: float) -> float: return 2*p*r/(p+r) if p+r else 0.0
    reversals = sum(1 for u, v in truth if (v, u) in prediction)
    shd = sk_fp + sk_fn + reversals
    return {
        "skeleton_tp": sk_tp, "skeleton_fp": sk_fp, "skeleton_fn": sk_fn,
        "skeleton_precision": ratio(sk_tp, sk_tp + sk_fp),
        "skeleton_recall": ratio(sk_tp, sk_tp + sk_fn),
        "skeleton_f1": f1(ratio(sk_tp, sk_tp + sk_fp), ratio(sk_tp, sk_tp + sk_fn)),
        "directed_tp": tp, "directed_fp": fp, "directed_fn": fn,
        "directed_precision": ratio(tp, tp + fp), "directed_recall": ratio(tp, tp + fn),
        "directed_f1": f1(ratio(tp, tp + fp), ratio(tp, tp + fn)),
        "shd": shd, "orientation_reversals": reversals,
        "true_edge_count": len(truth), "predicted_edge_count": len(prediction),
        "edge_count_delta": len(prediction) - len(truth),
    }


def layered_graph_scores(true_dag_edges, true_cpdag_directed_edges,
                         predicted_directed_edges, predicted_undirected_edges,
                         node_count: int, minimum_intervention_count: int,
                         interventions_used: int) -> dict:
    """Score skeleton, observationally compelled directions, full DAG, and budget separately.

    Undirected predictions recover an adjacency but not its direction. This is an
    ACDB-inspired layered contract; budget efficiency is reported as raw counts and
    a count ratio, not as a reproduction of ACDB's published scalar implementation.
    """
    true_dag = set(tuple(edge) for edge in true_dag_edges)
    cpdag_directed = set(tuple(edge) for edge in true_cpdag_directed_edges)
    pred_directed = set(tuple(edge) for edge in predicted_directed_edges)
    pred_undirected = set(tuple(edge) for edge in predicted_undirected_edges)
    # Reuse the strict directed-edge validation and duplicate checks.
    _ = graph_scores(list(true_dag), list(pred_directed), node_count)
    if (len(true_dag) != len(true_dag_edges) or len(cpdag_directed) != len(true_cpdag_directed_edges)
            or len(pred_directed) != len(predicted_directed_edges)):
        raise ValueError("duplicate truth or CPDAG directed edges are not allowed")
    if len(pred_undirected) != len(predicted_undirected_edges):
        raise ValueError("duplicate undirected predictions are not allowed")
    if not cpdag_directed.issubset(true_dag):
        raise ValueError("CPDAG compelled arrows must agree with the true DAG")
    for u, v in pred_undirected:
        if not (0 <= u < node_count and 0 <= v < node_count) or u >= v:
            raise ValueError("undirected edges must be in-range canonical pairs (u, v) with u < v")
    pred_directed_skeleton = {tuple(sorted(edge)) for edge in pred_directed}
    if len(pred_directed_skeleton) != len(pred_directed):
        raise ValueError("a predicted adjacency cannot contain both directed orientations")
    overlap = pred_directed_skeleton & pred_undirected
    if overlap:
        raise ValueError(f"a predicted adjacency cannot be both directed and undirected: {sorted(overlap)}")
    true_skeleton = {tuple(sorted(edge)) for edge in true_dag}
    pred_skeleton = pred_directed_skeleton | pred_undirected
    sk_tp = len(true_skeleton & pred_skeleton)
    sk_fp = len(pred_skeleton - true_skeleton)
    sk_fn = len(true_skeleton - pred_skeleton)
    def ratio(num: int, den: int) -> float: return num / den if den else 0.0
    def f1(p: float, r: float) -> float: return 2*p*r/(p+r) if p+r else 0.0
    sk_p, sk_r = ratio(sk_tp, sk_tp + sk_fp), ratio(sk_tp, sk_tp + sk_fn)
    directed_tp = len(true_dag & pred_directed)
    directed_fp, directed_fn = len(pred_directed - true_dag), len(true_dag - pred_directed)
    directed_p = ratio(directed_tp, directed_tp + directed_fp)
    directed_r = ratio(directed_tp, directed_tp + directed_fn)
    compelled_tp = len(cpdag_directed & pred_directed)
    compelled_fp, compelled_fn = len(pred_directed - cpdag_directed), len(cpdag_directed - pred_directed)
    compelled_p = ratio(compelled_tp, compelled_tp + compelled_fp)
    compelled_r = ratio(compelled_tp, compelled_tp + compelled_fn)
    reversals = sum(1 for u, v in true_dag if (v, u) in pred_directed)
    unresolved = len(pred_undirected & true_skeleton)
    dag_shd = sk_fp + sk_fn + reversals + unresolved
    if minimum_intervention_count < 0 or interventions_used < 0:
        raise ValueError("intervention counts must be nonnegative")
    ratio_to_minimum = (interventions_used / minimum_intervention_count
                        if minimum_intervention_count > 0 else (0.0 if interventions_used == 0 else None))
    return {
        "skeleton_tp": sk_tp, "skeleton_fp": sk_fp, "skeleton_fn": sk_fn,
        "skeleton_precision": sk_p, "skeleton_recall": sk_r, "skeleton_f1": f1(sk_p, sk_r),
        "compelled_tp": compelled_tp, "compelled_fp": compelled_fp, "compelled_fn": compelled_fn,
        "compelled_precision": compelled_p, "compelled_recall": compelled_r,
        "compelled_f1": f1(compelled_p, compelled_r),
        "directed_tp": directed_tp, "directed_fp": directed_fp, "directed_fn": directed_fn,
        "directed_precision": directed_p, "directed_recall": directed_r,
        "directed_f1": f1(directed_p, directed_r),
        "dag_shd": dag_shd, "orientation_reversals": reversals,
        "unresolved_true_adjacencies": unresolved,
        "orientation_coverage": ratio(len(pred_directed), len(pred_skeleton)),
        "minimum_intervention_count": minimum_intervention_count,
        "interventions_used": interventions_used,
        "intervention_count_to_minimum_ratio": ratio_to_minimum,
    }


def validate_scm(scm: SCM) -> None:
    if len(scm.parents) != scm.node_count or len(scm.truth_tables) != scm.node_count:
        raise AssertionError("node-array length mismatch")
    for child, ps in enumerate(scm.parents):
        if tuple(sorted(set(ps))) != ps or child in ps or any(p < 0 or p >= scm.node_count for p in ps):
            raise AssertionError("noncanonical parent list")
        if scm.truth_tables[child] < 0 or scm.truth_tables[child] >= (1 << (1 << len(ps))):
            raise AssertionError("truth table out of range")
        for pos in range(len(ps)):
            if not parent_is_essential(scm.truth_tables[child], len(ps), pos):
                raise AssertionError("declared parent is not functionally essential")
    remaining = set(range(scm.node_count))
    while remaining:
        ready = {node for node in remaining if not (set(scm.parents[node]) & remaining)}
        if not ready: raise AssertionError("SCM parents contain a directed cycle")
        remaining -= ready
    topology_parents(scm.node_count, scm.topology_family, scm.seed)


def build_suite(node_count: int = 6, cases_per_cell: int = 4) -> dict:
    if not 4 <= node_count <= 8: raise ValueError("node_count must be in 4..=8")
    if not 1 <= cases_per_cell <= 100: raise ValueError("cases_per_cell must be in 1..=100")
    rows = []
    for topology in TOPOLOGIES:
        for mechanism in MECHANISMS:
            for seed in range(cases_per_cell):
                scm = make_scm(node_count, topology, mechanism, seed)
                validate_scm(scm)
                rows.append({
                    "world_id": scm.world_id, "seed": seed, "node_count": node_count,
                    "topology_family": topology, "mechanism_family": mechanism,
                    "split": split_label(topology, mechanism),
                    "parents": [list(ps) for ps in scm.parents],
                    "directed_edges": [list(edge) for edge in scm.edges()],
                    "truth_tables": list(scm.truth_tables),
                })
    counts = {}
    for row in rows: counts[row["split"]] = counts.get(row["split"], 0) + 1
    return {
        "schema": SCHEMA,
        "status": "GENERATED_GROUND_TRUTH_SUITE_NOT_METHOD_PERFORMANCE",
        "node_count": node_count, "cases_per_topology_mechanism_cell": cases_per_cell,
        "topology_families": list(TOPOLOGIES), "mechanism_families": list(MECHANISMS),
        "topology_held_out_families": sorted(TOPOLOGY_HOLDOUT),
        "mechanism_held_out_families": sorted(MECHANISM_HOLDOUT),
        "split_counts": dict(sorted(counts.items())),
        "scoring_contract": "Fully directed DAG: skeleton P/R/F1; exact directed-edge P/R/F1; SHD counts an adjacency reversal as one edit; reversed directed edge counts as both FP and FN in directed F1.",
        "limitations": ["Boolean SCMs only", "finite fixed graph-family catalog", "not a causal-discovery algorithm", "no statistical uncertainty intervals", "not directly comparable to ACDB without its full protocol adapter"],
        "rows": rows,
    }


def self_test() -> None:
    a = build_suite(5, 2); b = build_suite(5, 2)
    assert a == b
    assert len(a["rows"]) == len(TOPOLOGIES) * len(MECHANISMS) * 2
    assert set(a["split_counts"]) == {"train_id", "topology_ood", "mechanism_ood", "combined_ood"}
    for row in a["rows"]:
        scm = SCM(row["world_id"], row["seed"], row["node_count"], row["topology_family"], row["mechanism_family"], tuple(tuple(x) for x in row["parents"]), tuple(row["truth_tables"]))
        validate_scm(scm)
        self_score = graph_scores(scm.edges(), scm.edges(), scm.node_count)
        assert self_score["shd"] == 0 and self_score["directed_f1"] == 1.0
    truth = [(0, 1), (1, 2)]
    reversed_prediction = [(1, 0), (1, 2)]
    score = graph_scores(truth, reversed_prediction, 3)
    assert score["skeleton_f1"] == 1.0 and score["directed_f1"] < 1.0
    assert score["shd"] == 1 and score["orientation_reversals"] == 1
    empty = graph_scores(truth, [], 3)
    assert empty["skeleton_recall"] == 0.0 and empty["shd"] == 2
    try: graph_scores(truth, [(0, 0)], 3)
    except ValueError: pass
    else: raise AssertionError("self-loop prediction not rejected")
    try: build_suite(3, 1)
    except ValueError: pass
    else: raise AssertionError("unsupported node count not rejected")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=6)
    parser.add_argument("--cases-per-cell", type=int, default=4)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test(); print("FAMILY_HELDOUT_SCM_SELF_TESTS_PASSED"); return 0
        payload = build_suite(args.nodes, args.cases_per_cell)
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_bytes(encoded)
            print(f"wrote {args.output} bytes={len(encoded)} sha256={hashlib.sha256(encoded).hexdigest()}")
        else: sys.stdout.buffer.write(encoded)
        return 0
    except (OSError, ValueError, AssertionError, RuntimeError) as error:
        print(f"NOT QUALIFIED: {error}", file=sys.stderr); return 2

if __name__ == "__main__": raise SystemExit(main())
