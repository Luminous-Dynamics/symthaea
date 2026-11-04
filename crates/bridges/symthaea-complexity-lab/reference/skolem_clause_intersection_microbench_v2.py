#!/usr/bin/env python3
"""Synthetic SCPOG-shaped Skolem-node/clause membership microbenchmark.

Compares Python storage layouts only; not a CakeML/SCPOG implementation benchmark.
"""
from __future__ import annotations
import array, bisect, json, random, sys, time
from pathlib import Path

UNIVERSE = 65_536
NODES_PER_DEGREE = 96
DEGREES = (1, 4, 16, 64, 256, 1_024, 4_096)
CLAUSES_PER_NODE = 128
CLAUSE_LENGTHS = (3, 8, 16)
SEED = 0xE7029
DENSE_BYTES = (UNIVERSE + 3) // 4
HYBRID_THRESHOLD = (UNIVERSE + 15) // 16


def deep_size(value, seen=None):
    if seen is None: seen = set()
    ident = id(value)
    if ident in seen: return 0
    seen.add(ident)
    size = sys.getsizeof(value)
    if isinstance(value, dict): size += sum(deep_size(k, seen) + deep_size(v, seen) for k, v in value.items())
    elif isinstance(value, (list, tuple, set, frozenset)): size += sum(deep_size(x, seen) for x in value)
    return size


def lit_id(variable, positive):
    return 2 * variable + (1 if positive else 0)


def build_dense(values):
    state = bytearray(DENSE_BYTES)
    for value in values:
        variable, positive = value >> 1, bool(value & 1)
        code = 1 if positive else 2
        shift = (variable & 3) * 2
        state[variable >> 2] = (state[variable >> 2] & ~(3 << shift)) | (code << shift)
    return state


def has_sorted(values, query):
    i = bisect.bisect_left(values, query)
    return i < len(values) and values[i] == query


def has_dense(state, query):
    variable, positive = query >> 1, bool(query & 1)
    shift = (variable & 3) * 2
    code = (state[variable >> 2] >> shift) & 3
    return code == (1 if positive else 2)


def main():
    rng = random.Random(SEED)
    node_values, clause_sets = [], []
    for degree in DEGREES:
        for _ in range(NODES_PER_DEGREE):
            variables = rng.sample(range(UNIVERSE), degree)
            values = [lit_id(v, bool(rng.getrandbits(1))) for v in variables]
            node_values.append(sorted(values))
            node_vars = set(variables)
            clauses = []
            for qi in range(CLAUSES_PER_NODE):
                length = CLAUSE_LENGTHS[qi % len(CLAUSE_LENGTHS)]
                is_hit = (qi % 2 == 0)
                picked_vars = set()
                literals = []
                if is_hit:
                    member = values[rng.randrange(degree)]
                    picked_vars.add(member >> 1)
                    literals.append(member)
                while len(literals) < length:
                    variable = rng.randrange(UNIVERSE)
                    if variable in node_vars or variable in picked_vars: continue
                    picked_vars.add(variable)
                    literals.append(lit_id(variable, bool(rng.getrandbits(1))))
                rng.shuffle(literals)
                clauses.append(literals)
            clause_sets.append(clauses)

    reps = {
        "hash_set": [set(v) for v in node_values],
        "sorted_u32": [array.array("I", v) for v in node_values],
        "dense_2bit": [build_dense(v) for v in node_values],
        "hybrid": [("sparse", array.array("I", v)) if len(v) < HYBRID_THRESHOLD else ("dense", build_dense(v)) for v in node_values],
    }
    rows, reference_hits, reference_probes = [], None, None
    total_clauses = len(node_values) * CLAUSES_PER_NODE
    total_literal_count = sum(len(c) for group in clause_sets for c in group)
    for kind, representation in reps.items():
        start = time.perf_counter_ns()
        hits = probes = 0
        for node, clauses in zip(representation, clause_sets):
            for clause in clauses:
                clause_hit = False
                for lit in clause:
                    probes += 1
                    if kind == "hash_set": found = lit in node
                    elif kind == "sorted_u32": found = has_sorted(node, lit)
                    elif kind == "dense_2bit": found = has_dense(node, lit)
                    else:
                        mode, payload = node
                        found = has_sorted(payload, lit) if mode == "sparse" else has_dense(payload, lit)
                    if found:
                        clause_hit = True
                        break
                hits += int(clause_hit)
        elapsed = time.perf_counter_ns() - start
        memory = deep_size(representation)
        if reference_hits is None: reference_hits, reference_probes = hits, probes
        if hits != reference_hits or probes != reference_probes:
            raise AssertionError(f"semantic/probe mismatch for {kind}: hits={hits}, probes={probes}")
        rows.append({
            "representation": kind, "representation_bytes": memory,
            "representation_mib": round(memory / (1024 * 1024), 3),
            "queries_clauses": total_clauses, "candidate_literal_count": total_literal_count,
            "actual_literal_probes_after_short_circuit": probes, "clause_hits": hits,
            "lookup_time_ms": round(elapsed / 1_000_000, 3),
            "ns_per_clause": round(elapsed / total_clauses, 1), "semantics_match": True,
        })
    result = {
        "format": "symthaea.skolem-clause-intersection-layout-microbench", "version": 2,
        "claim_ceiling": "synthetic_python_sc_pog_shaped_lookup_proxy_not_cakeml_or_sc_pog_benchmark",
        "parameters": {
            "quantified_variable_universe": UNIVERSE, "literal_id_universe": UNIVERSE * 2,
            "nodes_per_degree": NODES_PER_DEGREE, "degrees": list(DEGREES), "total_nodes": len(node_values),
            "clauses_per_node": CLAUSES_PER_NODE, "clause_lengths": list(CLAUSE_LENGTHS),
            "hit_ratio": 0.5, "seed": SEED, "dense_2bit_bytes_per_node": DENSE_BYTES,
            "hybrid_threshold_degree": HYBRID_THRESHOLD,
        },
        "overall": rows,
        "conclusion": "This models the SCPOG reverse-check query shape: for each input clause, test whether any clause literal belongs to a Skolem node. Measurements are CPython-specific; CakeML has its own verified byte-array and array-based optimizations, so no CakeML performance claim follows.",
        "runtime": {"python": sys.version.split()[0], "platform": sys.platform},
    }
    out = Path(__file__).with_name("skolem_clause_intersection_results.v2.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for row in rows:
        print(f"{row['representation']:12} {row['representation_mib']:8.3f} MiB {row['ns_per_clause']:8.1f} ns/clause hits={row['clause_hits']} probes={row['actual_literal_probes_after_short_circuit']}")
    print(f"PASS {len(rows)} layouts agree on {total_clauses} clause-intersection queries")


if __name__ == "__main__":
    main()
