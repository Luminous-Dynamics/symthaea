#!/usr/bin/env python3
"""Context-budget frontier for compositional HDC under within-cluster overlap.

Independent NumPy pilot. Records are majority bundles of K role-bound binary
facts. Within groups of eight records, the first S fact slots are shared
exactly. A query exposes M odd-numbered fact cues; constituent bits can be
corrupted. Compare nearest bundled-record search with a role/slot-aware scorer.
This is not Symthaea Rust execution and not a production model.

Run: python3 hdc_context_budget_overlap_pilot.py --output-dir .
Requires Python 3.10+ and NumPy 2.x.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

DIM = 16_384
BYTES_PER_HV = DIM // 8
CORPUS_SIZES = (1_024,)
ARITIES = (7, 15)
NOISE_RATES = (0.0, 0.10)
SEEDS = (20261028, 20261029)
QUERIES_PER_CELL = 16
CLUSTER_WIDTH = 8
BITCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def build_facts(
    corpus_size: int,
    arity: int,
    shared_slots: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Build packed role-bound facts and majority-bundled record prototypes."""
    roles = rng.integers(
        0, 256, size=(arity, BYTES_PER_HV), dtype=np.uint8
    )
    facts = np.empty(
        (corpus_size, arity, BYTES_PER_HV), dtype=np.uint8
    )
    chunk_size = 64
    for start in range(0, corpus_size, chunk_size):
        stop = min(corpus_size, start + chunk_size)
        values = rng.integers(
            0, 256,
            size=(stop - start, arity, BYTES_PER_HV),
            dtype=np.uint8,
        )
        facts[start:stop] = np.bitwise_xor(
            values, roles[None, :, :]
        )

    if corpus_size % CLUSTER_WIDTH:
        raise ValueError("corpus size must be divisible by cluster width")

    # Exact within-cluster overlap: members share the first S role-bound facts.
    for start in range(0, corpus_size, CLUSTER_WIDTH):
        for slot in range(shared_slots):
            facts[start:start + CLUSTER_WIDTH, slot] = facts[start, slot]

    records = np.empty((corpus_size, BYTES_PER_HV), dtype=np.uint8)
    for start in range(0, corpus_size, chunk_size):
        stop = min(corpus_size, start + chunk_size)
        unpacked = np.unpackbits(
            facts[start:stop], axis=-1, bitorder="little"
        )
        bundled = unpacked.sum(axis=1) > (arity // 2)
        records[start:stop] = np.packbits(
            bundled, axis=-1, bitorder="little"
        )
    return facts, records


def corrupt(hv: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Flip exactly floor(DIM*rate) uniformly selected bits."""
    n_flip = math.floor(DIM * rate)
    bits = np.zeros(DIM, dtype=np.uint8)
    if n_flip:
        bits[rng.choice(DIM, size=n_flip, replace=False)] = 1
    mask = np.packbits(bits, bitorder="little")
    assert int(BITCOUNT[mask].sum(dtype=np.uint64)) == n_flip
    return np.bitwise_xor(hv, mask)


def query_bundle(query_facts: np.ndarray) -> np.ndarray:
    """Majority-bundle an odd number of observed fact cues."""
    if len(query_facts) % 2 == 0:
        raise ValueError("query cue count must be odd")
    bits = np.unpackbits(query_facts, axis=-1, bitorder="little")
    return np.packbits(
        bits.sum(axis=0) > (len(query_facts) // 2),
        bitorder="little",
    )


def full_scan(records: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Exact Hamming distances to bundled record prototypes."""
    xor = np.bitwise_xor(records, query[None, :])
    return np.bitwise_count(xor).sum(axis=1, dtype=np.uint32)


def component_scan(facts: np.ndarray, query_facts: np.ndarray) -> np.ndarray:
    """Slot-aware score: sum Hamming distances for matching cue/fact slots."""
    distances = np.zeros(len(facts), dtype=np.uint32)
    for slot, query in enumerate(query_facts):
        xor = np.bitwise_xor(facts[:, slot, :], query[None, :])
        distances += np.bitwise_count(xor).sum(axis=1, dtype=np.uint32)
    return distances


def rank_and_margin(
    distances: np.ndarray, target: int, score_scale: int
) -> tuple[int, float, int]:
    """Return target rank, top1/top2 similarity margin and winning ID."""
    winner = int(np.argmin(distances))  # stable lowest-ID-first tie behavior
    target_distance = int(distances[target])
    rank = (
        1
        + int(np.count_nonzero(distances < target_distance))
        + int(np.count_nonzero(
            (distances == target_distance)
            & (np.arange(len(distances)) < target)
        ))
    )
    two_best = np.partition(distances, 1)[:2]
    margin = float(
        (int(two_best.max()) - int(two_best.min())) / score_scale
    )
    return rank, margin, winner


def scalar_oracle_small(
    records: np.ndarray, query: np.ndarray
) -> np.ndarray:
    """Independent scalar-Hamming oracle for small deterministic fixtures."""
    query_bytes = [int(x) for x in query]
    return np.array(
        [
            sum(
                (int(byte) ^ qbyte).bit_count()
                for byte, qbyte in zip(row, query_bytes, strict=True)
            )
            for row in records
        ],
        dtype=np.uint32,
    )


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    oracle_checks = 0
    information_limit_cases = 0
    started = time.perf_counter()

    for seed in SEEDS:
        for corpus_size in CORPUS_SIZES:
            for arity in ARITIES:
                shared_options = (0, 1, 3) if arity == 7 else (0, 1, 3, 7)
                query_counts = (1, 3, 7)
                for shared_slots in shared_options:
                    rng = np.random.default_rng(
                        seed
                        + corpus_size * 101
                        + arity * 1009
                        + shared_slots * 65537
                    )
                    facts, records = build_facts(
                        corpus_size, arity, shared_slots, rng
                    )
                    targets = rng.permutation(corpus_size)[:QUERIES_PER_CELL]

                    for query_count in query_counts:
                        if query_count > arity:
                            continue
                        for noise_rate in NOISE_RATES:
                            bundled_hits = 0
                            component_hits = 0
                            bundled_ranks: list[int] = []
                            component_ranks: list[int] = []
                            bundled_margins: list[float] = []
                            component_margins: list[float] = []

                            for query_index, target_raw in enumerate(targets):
                                target = int(target_raw)
                                observed = np.stack([
                                    corrupt(
                                        facts[target, slot],
                                        noise_rate,
                                        rng,
                                    )
                                    for slot in range(query_count)
                                ])
                                query = query_bundle(observed)
                                bundled_distances = full_scan(records, query)
                                component_distances = component_scan(facts, observed)

                                rb, mb, wb = rank_and_margin(
                                    bundled_distances, target, DIM
                                )
                                rc, mc, wc = rank_and_margin(
                                    component_distances,
                                    target,
                                    DIM * query_count,
                                )
                                bundled_hits += int(wb == target)
                                component_hits += int(wc == target)
                                bundled_ranks.append(rb)
                                component_ranks.append(rc)
                                bundled_margins.append(mb)
                                component_margins.append(mc)

                                # Independently cross-check a 17-candidate fixture.
                                if query_index < 2:
                                    oracle = scalar_oracle_small(
                                        records[:17], query
                                    )
                                    assert np.array_equal(
                                        bundled_distances[:17], oracle
                                    )
                                    assert int(np.argmin(
                                        bundled_distances[:17]
                                    )) == int(np.argmin(oracle))
                                    oracle_checks += 1

                            if shared_slots > 0 and query_count <= shared_slots:
                                # Every provided cue is identical within each cluster.
                                information_limit_cases += (
                                    QUERIES_PER_CELL
                                )

                            for method, hits, ranks, margins in (
                                (
                                    "bundled_prototype",
                                    bundled_hits,
                                    bundled_ranks,
                                    bundled_margins,
                                ),
                                (
                                    "slot_aware_component_index",
                                    component_hits,
                                    component_ranks,
                                    component_margins,
                                ),
                            ):
                                rows.append({
                                    "seed": seed,
                                    "dimension_bits": DIM,
                                    "corpus_size": corpus_size,
                                    "bundle_arity": arity,
                                    "cluster_width": CLUSTER_WIDTH,
                                    "shared_fact_slots_per_cluster": shared_slots,
                                    "query_fact_count": query_count,
                                    "query_noise_rate": noise_rate,
                                    "method": method,
                                    "query_count": QUERIES_PER_CELL,
                                    "hit_at_1_count": hits,
                                    "hit_at_1": hits / QUERIES_PER_CELL,
                                    "median_target_rank": float(np.median(ranks)),
                                    "mean_target_rank": float(np.mean(ranks)),
                                    "mean_top1_top2_margin": float(np.mean(margins)),
                                    "information_limit_control": bool(
                                        shared_slots > 0
                                        and query_count <= shared_slots
                                    ),
                                    "interpretation": (
                                        "target identity not present in observed cues"
                                        if shared_slots > 0 and query_count <= shared_slots
                                        else "unique cue(s) exist; retrieval quality is empirical"
                                    ),
                                })
                    del facts, records

    result_path = output_dir / "context_budget_overlap_results.csv"
    with result_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "experiment": "DSE-011 compositional overlap and context-budget frontier",
        "seeds": list(SEEDS),
        "corpus_sizes": list(CORPUS_SIZES),
        "arities": list(ARITIES),
        "noise_rates": list(NOISE_RATES),
        "query_counts_per_cell": QUERIES_PER_CELL,
        "rows": len(rows),
        "scalar_oracle_fixture_checks": oracle_checks,
        "queries_in_information_limit_control_cells": information_limit_cases,
        "runtime_seconds": time.perf_counter() - started,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sha256_csv": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        "evidence_boundary": (
            "synthetic NumPy pilot; not Rust execution, production corpus results, "
            "or an SOTA claim"
        ),
    }
    (output_dir / "context_budget_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))


if __name__ == "__main__":
    main()
