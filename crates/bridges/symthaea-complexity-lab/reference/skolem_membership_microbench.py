#!/usr/bin/env python3
"""Synthetic Skolem literal membership layout microbenchmark.

This compares Python representations only; it is not a CakeML or SCPOG benchmark.
"""
from __future__ import annotations
import array
import bisect
import json
import random
import sys
import time
from pathlib import Path

UNIVERSE = 65_536
NODES_PER_DEGREE = 128
DEGREES = (1, 4, 16, 64, 256, 1_024, 4_096)
QUERIES_PER_NODE = 96
SEED = 0xE7029
BITSET_BYTES = (UNIVERSE + 7) // 8
HYBRID_THRESHOLD = (UNIVERSE + 31) // 32


def deep_size(value, seen=None):
    if seen is None:
        seen = set()
    ident = id(value)
    if ident in seen:
        return 0
    seen.add(ident)
    size = sys.getsizeof(value)
    if isinstance(value, dict):
        size += sum(deep_size(k, seen) + deep_size(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple, set, frozenset)):
        size += sum(deep_size(x, seen) for x in value)
    return size


def make_bitset(values):
    bits = bytearray(BITSET_BYTES)
    for value in values:
        bits[value >> 3] |= 1 << (value & 7)
    return bits


def contains_sorted(values, query):
    i = bisect.bisect_left(values, query)
    return i < len(values) and values[i] == query


def contains_bitset(bits, query):
    return bool(bits[query >> 3] & (1 << (query & 7)))


def run_queries(representation, query_sets, kind):
    hits = 0
    start = time.perf_counter_ns()
    if kind == "hash_set":
        for node, queries in zip(representation, query_sets):
            hits += sum(q in node for q in queries)
    elif kind == "sorted_u32":
        for node, queries in zip(representation, query_sets):
            hits += sum(contains_sorted(node, q) for q in queries)
    elif kind == "dense_bitset":
        for node, queries in zip(representation, query_sets):
            hits += sum(contains_bitset(node, q) for q in queries)
    elif kind == "hybrid":
        for node, queries in zip(representation, query_sets):
            mode, payload = node
            if mode == "sparse":
                hits += sum(contains_sorted(payload, q) for q in queries)
            else:
                hits += sum(contains_bitset(payload, q) for q in queries)
    else:
        raise ValueError(kind)
    return hits, time.perf_counter_ns() - start


def main():
    rng = random.Random(SEED)
    node_values = []
    query_sets = []
    for degree in DEGREES:
        for _ in range(NODES_PER_DEGREE):
            values = sorted(rng.sample(range(UNIVERSE), degree))
            queries = []
            for _ in range(QUERIES_PER_NODE):
                queries.append(values[rng.randrange(degree)] if rng.random() < 0.5 else rng.randrange(UNIVERSE))
            node_values.append(values)
            query_sets.append(queries)

    representations = {
        "hash_set": [set(values) for values in node_values],
        "sorted_u32": [array.array("I", values) for values in node_values],
        "dense_bitset": [make_bitset(values) for values in node_values],
        "hybrid": [
            ("sparse", array.array("I", values)) if len(values) < HYBRID_THRESHOLD
            else ("dense", make_bitset(values))
            for values in node_values
        ],
    }

    rows = []
    reference_hits = None
    for kind, representation in representations.items():
        memory = deep_size(representation)
        hits, elapsed = run_queries(representation, query_sets, kind)
        if reference_hits is None:
            reference_hits = hits
        if hits != reference_hits:
            raise AssertionError(f"semantic mismatch: {kind} gave {hits}, expected {reference_hits}")
        rows.append({
            "representation": kind,
            "representation_bytes": memory,
            "representation_mib": round(memory / (1024 * 1024), 3),
            "lookup_queries": len(node_values) * QUERIES_PER_NODE,
            "membership_hits": hits,
            "lookup_time_ms": round(elapsed / 1_000_000, 3),
            "ns_per_query": round(elapsed / (len(node_values) * QUERIES_PER_NODE), 1),
            "semantics_match": True,
        })

    per_degree = []
    offset = 0
    for degree in DEGREES:
        count = NODES_PER_DEGREE
        sparse = [array.array("I", node_values[i]) for i in range(offset, offset + count)]
        sets = [set(node_values[i]) for i in range(offset, offset + count)]
        dense = [make_bitset(node_values[i]) for i in range(offset, offset + count)]
        per_degree.append({
            "degree": degree, "nodes": count,
            "sorted_u32_bytes": deep_size(sparse),
            "hash_set_bytes": deep_size(sets),
            "dense_bitset_bytes": deep_size(dense),
            "theoretical_dense_bytes_per_node": BITSET_BYTES,
            "hybrid_choice": "dense" if degree >= HYBRID_THRESHOLD else "sparse",
        })
        offset += count

    result = {
        "format": "symthaea.skolem-membership-layout-microbench",
        "version": 1,
        "claim_ceiling": "synthetic_python_representation_proxy_not_cakeml_or_sc_pog_benchmark".replace(" ", "_"),
        "parameters": {
            "universe_literals": UNIVERSE, "nodes_per_degree": NODES_PER_DEGREE,
            "degrees": list(DEGREES), "total_nodes": len(node_values),
            "queries_per_node": QUERIES_PER_NODE, "seed": SEED,
            "dense_bitset_bytes_per_node": BITSET_BYTES,
            "hybrid_threshold_degree": HYBRID_THRESHOLD,
        },
        "overall": rows,
        "per_degree": per_degree,
        "conclusion": "All representations returned identical membership results. Memory and time measurements are specific to CPython 3.13.5; the selected layout must be remeasured in CakeML before adoption.",
        "runtime": {"python": sys.version.split()[0], "platform": sys.platform},
    }
    out = Path(__file__).with_name("skolem_membership_layout_results.v1.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for row in rows:
        print(f"{row['representation']:12} {row['representation_mib']:8.3f} MiB {row['ns_per_query']:8.1f} ns/query hits={row['membership_hits']}")
    print(f"PASS identical membership semantics across {len(rows)} layouts; {len(node_values) * QUERIES_PER_NODE} queries")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
