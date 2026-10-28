#!/usr/bin/env python3
"""Deterministic synthetic HDC open-set and ambiguity-control pilot.

This is an independent NumPy reference workload, not Symthaea Rust execution.
It builds packed binary records by majority-bundling role-bound facts, calibrates
an acceptance threshold on clean known and independent absent cues, then freezes
it for corrupted known cues and open-set tests. An absent-value near-miss is
constructed to be byte-identical to a corrupted known cue; the experiment
asserts that a vector-only score policy cannot distinguish the two labels.

Requirements: Python 3.10+, NumPy 2.x (works with older NumPy versions too).
Run: python3 hdc_compositional_open_set_pilot.py --output-dir .
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
CORPUS_SIZES = (256, 1_024, 4_096)
ARITIES = (3, 7, 15)
NOISE_RATES = (0.00, 0.01, 0.02, 0.05, 0.10)
N_QUERIES = 32  # calibration/test sample; still a pilot, not a production guarantee
SEEDS = (20261028, 20261029)
BITCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def build_corpus(n: int, arity: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build packed BinaryHV analogues and preserve each record's first fact cue."""
    roles = rng.integers(0, 256, size=(arity, BYTES_PER_HV), dtype=np.uint8)
    records = np.empty((n, BYTES_PER_HV), dtype=np.uint8)
    cues = np.empty_like(records)
    chunk = 128
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        values = rng.integers(0, 256, size=(stop - start, arity, BYTES_PER_HV), dtype=np.uint8)
        facts = np.bitwise_xor(values, roles[None, :, :])
        cues[start:stop] = facts[:, 0, :]
        bits = np.unpackbits(facts, axis=-1, bitorder="little")
        bundled = bits.sum(axis=1) > (arity // 2)
        records[start:stop] = np.packbits(bundled, axis=-1, bitorder="little")
    return records, cues, roles[0].copy()


def flip_exact_bits(query: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Flip exactly floor(DIM*rate) randomly selected bits in a packed query."""
    n_flip = math.floor(DIM * rate)
    mask_bits = np.zeros(DIM, dtype=np.uint8)
    if n_flip:
        positions = rng.choice(DIM, size=n_flip, replace=False)
        mask_bits[positions] = 1
    mask = np.packbits(mask_bits, bitorder="little")
    result = np.bitwise_xor(query, mask)
    assert int(BITCOUNT[mask].sum(dtype=np.uint64)) == n_flip
    return result


def make_absent_cues(n: int, role: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Role-bound cues from random values not drawn from the stored corpus."""
    values = rng.integers(0, 256, size=(n, BYTES_PER_HV), dtype=np.uint8)
    return np.bitwise_xor(values, role[None, :])


def search_exact(records: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, int, float, float]:
    """Packed byte-XOR/popcount scan; NumPy argmin makes ties lowest-ID-first."""
    xor = np.bitwise_xor(records, query[None, :])
    distances = np.bitwise_count(xor).sum(axis=1, dtype=np.uint32)
    winner = int(np.argmin(distances))
    best_distance = int(distances[winner])
    if len(distances) > 1:
        two = np.partition(distances, 1)[:2]
        margin = float((int(max(two)) - int(min(two))) / DIM)
    else:
        margin = 1.0
    best_score = float((DIM - best_distance) / DIM)
    return distances, winner, best_score, margin


def target_rank(distances: np.ndarray, target_id: int) -> int:
    """Rank under distance ascending and lowest-ID-first tie-breaking."""
    target_distance = int(distances[target_id])
    better = int(np.count_nonzero(distances < target_distance))
    earlier_tie = int(np.count_nonzero((distances == target_distance) & (np.arange(len(distances)) < target_id)))
    return 1 + better + earlier_tie


def scalar_byte_oracle(records: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, int]:
    """Independent, deliberately simple Python-int.bit_count oracle for fixtures."""
    dists = np.empty(len(records), dtype=np.uint32)
    query_bytes = [int(x) for x in query]
    for i, row in enumerate(records):
        dists[i] = sum((int(x) ^ q).bit_count() for x, q in zip(row, query_bytes, strict=True))
    return dists, int(np.argmin(dists))


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if trials <= 0:
        return None, None
    p = successes / trials
    denom = 1 + z*z/trials
    center = (p + z*z/(2*trials)) / denom
    half = z * math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / denom
    return max(0.0, center-half), min(1.0, center+half)


def add_rate_fields(row: dict[str, Any], prefix: str, successes: int, trials: int) -> None:
    low, high = wilson_interval(successes, trials)
    row[f"{prefix}_count"] = successes
    row[f"{prefix}_n"] = trials
    row[f"{prefix}_rate"] = successes / trials if trials else ""
    row[f"{prefix}_wilson95_low"] = low if low is not None else ""
    row[f"{prefix}_wilson95_high"] = high if high is not None else ""


def run(output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    robustness_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    oracle_cases = 0
    ambiguity_twin_cases = 0
    tie_control_pass = False
    start_all = time.perf_counter()

    # A deterministic tie-control: exact duplicate vectors must choose the earliest ID.
    tie_rng = np.random.default_rng(20261028)
    tie_base = tie_rng.integers(0, 256, size=(17, BYTES_PER_HV), dtype=np.uint8)
    tie_query = tie_base[4].copy()
    tie_fixture = np.concatenate([tie_base, tie_base[4:5].copy()])
    tie_dists, tie_winner = scalar_byte_oracle(tie_fixture, tie_query)
    tie_control_pass = bool(tie_winner == 4 and tie_dists[4] == tie_dists[17] == 0)
    assert tie_control_pass

    for seed in SEEDS:
        for n in CORPUS_SIZES:
            for arity in ARITIES:
                rng = np.random.default_rng(seed + n * 1009 + arity * 65537)
                records, cues, role0 = build_corpus(n, arity, rng)
                order = rng.permutation(n)
                cal_ids = order[:N_QUERIES]
                test_ids = order[N_QUERIES:2*N_QUERIES]
                assert len(set(map(int, cal_ids)) & set(map(int, test_ids))) == 0

                # Calibration partition: clean known cues plus independent absent-value cues.
                cal_known_scores: list[float] = []
                for i in cal_ids:
                    _, _, score, _ = search_exact(records, cues[int(i)])
                    cal_known_scores.append(score)
                cal_unknown_cues = make_absent_cues(N_QUERIES, role0, rng)
                cal_unknown_scores: list[float] = []
                for q in cal_unknown_cues:
                    _, _, score, _ = search_exact(records, q)
                    cal_unknown_scores.append(score)

                # Freeze a zero-observed-false-accept threshold on calibration unknowns.
                threshold = max(cal_unknown_scores) + 1.0 / DIM
                cal_known_accept = sum(score >= threshold for score in cal_known_scores)
                cal_unknown_accept = sum(score >= threshold for score in cal_unknown_scores)
                assert cal_unknown_accept == 0

                # Independent held-out open-set control, not used for threshold selection.
                test_unknown_cues = make_absent_cues(N_QUERIES, role0, rng)
                test_unknown_scores = []
                for q in test_unknown_cues:
                    _, _, score, _ = search_exact(records, q)
                    test_unknown_scores.append(score)
                unknown_accepts = sum(score >= threshold for score in test_unknown_scores)

                test_ids_py = [int(i) for i in test_ids]
                per_noise: dict[float, list[dict[str, Any]]] = {}
                for noise in NOISE_RATES:
                    noise_rows: list[dict[str, Any]] = []
                    accepted = 0
                    hits = 0
                    ranks: list[int] = []
                    margins: list[float] = []
                    top_scores: list[float] = []
                    for qi, target_id in enumerate(test_ids_py):
                        q = flip_exact_bits(cues[target_id], noise, rng)
                        distances, winner, score, margin = search_exact(records, q)
                        rank = target_rank(distances, target_id)
                        is_accepted = score >= threshold
                        accepted += int(is_accepted)
                        hits += int(winner == target_id)
                        ranks.append(rank)
                        margins.append(margin)
                        top_scores.append(score)

                        # Independently verify distance and winner on a 17-candidate fixture.
                        if qi < 2:
                            sub = records[:17]
                            fast_dists = distances[:17]
                            oracle_dists, oracle_winner = scalar_byte_oracle(sub, q)
                            assert np.array_equal(fast_dists, oracle_dists)
                            assert int(np.argmin(fast_dists)) == oracle_winner
                            oracle_cases += 1

                        # The absent near-miss label is deliberately assigned to the exact same
                        # observed cue bytes. This is an ambiguity control, not a learnable label.
                        absent_value_twin = q.copy()
                        assert np.array_equal(q, absent_value_twin)
                        # Given identical input bytes and a deterministic search, its distances,
                        # winner, score and margin are identical by construction; re-running the
                        # same scan would add no independent evidence and double benchmark cost.
                        twin_winner, twin_score, twin_margin = winner, score, margin
                        assert twin_winner == winner and twin_score == score and twin_margin == margin
                        ambiguity_twin_cases += 1
                        noise_rows.append({
                            "noise_rate": noise,
                            "target_id": target_id,
                            "target_rank": rank,
                            "winner_id": winner,
                            "target_hit_at_1": int(winner == target_id),
                            "top_score": score,
                            "top1_top2_margin": margin,
                            "score_only_accept": int(is_accepted),
                            "same_byte_absent_twin_accept": int(twin_score >= threshold),
                        })
                    assert accepted == sum(row["score_only_accept"] for row in noise_rows)
                    assert accepted == sum(row["same_byte_absent_twin_accept"] for row in noise_rows)
                    per_noise[noise] = noise_rows

                    row: dict[str, Any] = {
                        "seed": seed,
                        "dimension_bits": DIM,
                        "corpus_size": n,
                        "bundle_arity": arity,
                        "noise_rate": noise,
                        "score_only_threshold": threshold,
                        "calibration_unknown_max_score": max(cal_unknown_scores),
                        "calibration_unknown_n": len(cal_unknown_scores),
                        "calibration_known_n": len(cal_known_scores),
                        "calibration_known_accept_count": cal_known_accept,
                        "calibration_known_accept_rate": cal_known_accept / N_QUERIES,
                        "mean_top_score": float(np.mean(top_scores)),
                        "mean_top1_top2_margin": float(np.mean(margins)),
                        "median_target_rank": float(np.median(ranks)),
                        "oracle_fixture_cases_so_far": oracle_cases,
                        "ambiguity_twin_cases_so_far": ambiguity_twin_cases,
                    }
                    add_rate_fields(row, "known_hit_at_1", hits, N_QUERIES)
                    add_rate_fields(row, "known_score_only_accept", accepted, N_QUERIES)
                    add_rate_fields(row, "same_byte_absent_twin_accept", accepted, N_QUERIES)
                    add_rate_fields(row, "heldout_independent_unknown_accept", unknown_accepts, N_QUERIES)
                    robustness_rows.append(row)

                cell_rows.append({
                    "seed": seed,
                    "dimension_bits": DIM,
                    "corpus_size": n,
                    "bundle_arity": arity,
                    "score_only_threshold": threshold,
                    "calibration_known_n": N_QUERIES,
                    "calibration_known_accept_count": cal_known_accept,
                    "calibration_known_accept_rate": cal_known_accept / N_QUERIES,
                    "calibration_independent_unknown_n": N_QUERIES,
                    "calibration_independent_unknown_accept_count": cal_unknown_accept,
                    "calibration_independent_unknown_max_score": max(cal_unknown_scores),
                    "heldout_independent_unknown_n": N_QUERIES,
                    "heldout_independent_unknown_accept_count": unknown_accepts,
                    "heldout_independent_unknown_accept_rate": unknown_accepts / N_QUERIES,
                    "evidence_note": "small synthetic pilot; threshold frozen; not a deployment FAR guarantee",
                })

    robust_path = output_dir / "compositional_robustness.csv"
    cell_path = output_dir / "open_set_calibration.csv"
    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        if not rows:
            raise ValueError(f"No rows to write to {path}")
        cols = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
    write_csv(robust_path, robustness_rows)
    write_csv(cell_path, cell_rows)

    elapsed = time.perf_counter() - start_all
    summary = {
        "experiment": "DSE-010 compositional open-set and observational ambiguity pilot",
        "seed_values": list(SEEDS),
        "dimension_bits": DIM,
        "corpus_sizes": list(CORPUS_SIZES),
        "bundle_arities": list(ARITIES),
        "noise_rates": list(NOISE_RATES),
        "queries_per_calibration_class": N_QUERIES,
        "robustness_rows": len(robustness_rows),
        "calibration_cells": len(cell_rows),
        "independent_scalar_oracle_fixture_cases": oracle_cases,
        "same_byte_ambiguity_twin_assertions": ambiguity_twin_cases,
        "duplicate_vector_lowest_id_tie_control_passed": tie_control_pass,
        "threshold_policy": "maximum calibration independent-unknown score + one discrete score quantum; frozen for held-out evaluation",
        "reproducibility_note": "timing is recorded only as total runtime in this manifest; measurement CSVs omit per-cell wall-clock values so their hashes are deterministic",
        "evidence_boundary": "NumPy synthetic HDC pilot only; not Rust execution, production acceptance calibration, energy measurement, or a population-level error-rate estimate",
        "runtime_seconds": elapsed,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "files": {
            robust_path.name: hashlib.sha256(robust_path.read_bytes()).hexdigest(),
            cell_path.name: hashlib.sha256(cell_path.read_bytes()).hexdigest(),
        },
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
