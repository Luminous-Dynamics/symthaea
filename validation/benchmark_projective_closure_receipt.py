"""Benchmark build and verification cost of authenticated closure receipts."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from fractions import Fraction
from typing import Any

from projective_closure_receipt import build_snapshot, make_score_receipt, verify_score_receipt


def _manifest_root(ids: list[str]) -> str:
    payload = json.dumps(sorted(ids), separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _linear_score(entries: list[dict[str, Any]]) -> Fraction:
    selected_count = sum(bool(item["selected"]) for item in entries)
    if selected_count == 0:
        return Fraction(0)
    threshold = Fraction(1, selected_count)
    baseline = sum(
        (max(Fraction(0), -item["b"]) for item in entries),
        Fraction(0),
    )
    prefix_count = sum(
        bool(item["selected"]) and item["b"] < threshold for item in entries
    )
    prefix_sum = sum(
        (
            item["b"]
            for item in entries
            if item["selected"] and item["b"] < threshold
        ),
        Fraction(0),
    )
    negative_sum = sum(
        (
            item["b"]
            for item in entries
            if item["selected"] and item["b"] < 0
        ),
        Fraction(0),
    )
    return baseline + Fraction(prefix_count, selected_count) - (
        prefix_sum - negative_sum
    )


def _trial(size: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    entries = [
        {
            "hypothesis_id": f"hypothesis-{i:09d}",
            "b": Fraction(rng.randint(-1_000_000, 1_000_000), 1_000_000),
            "selected": rng.random() < 0.3,
        }
        for i in range(size)
    ]
    if not any(item["selected"] for item in entries):
        entries[-1]["selected"] = True
    manifest = _manifest_root([item["hypothesis_id"] for item in entries])
    weight_digest = hashlib.sha256(b"weights-v1").hexdigest()
    config = {
        "alpha": "1/20",
        "weight_digest": weight_digest,
        "evidence_version": "evidence-v1",
        "tree_schema": 1,
    }

    start = time.perf_counter_ns()
    snapshot = build_snapshot(
        entries,
        manifest_root=manifest,
        sequence=1,
        configuration=config,
    )
    build_ms = (time.perf_counter_ns() - start) / 1e6

    start = time.perf_counter_ns()
    receipt = make_score_receipt(
        snapshot,
        alpha=Fraction(1, 20),
        weight_digest=weight_digest,
        evidence_version="evidence-v1",
    )
    proof_ms = (time.perf_counter_ns() - start) / 1e6

    encoded = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    proof_nodes = (
        len(receipt["prefix_lt_threshold"]["path"])
        + len(receipt["prefix_lt_zero"]["path"])
    )

    start = time.perf_counter_ns()
    authenticated_score = verify_score_receipt(
        receipt,
        expected_snapshot_hash=snapshot.snapshot_hash,
        expected_manifest_root=manifest,
        expected_weight_digest=weight_digest,
        expected_evidence_version="evidence-v1",
        expected_alpha=Fraction(1, 20),
    )
    verify_ms = (time.perf_counter_ns() - start) / 1e6

    start = time.perf_counter_ns()
    exact_scan_score = _linear_score(entries)
    scan_ms = (time.perf_counter_ns() - start) / 1e6
    if authenticated_score != exact_scan_score:
        raise AssertionError("authenticated score differs from independent scan")

    return {
        "n": size,
        "selected_count": snapshot.aggregate.selected_count,
        "build_ms": build_ms,
        "proof_generation_ms": proof_ms,
        "verification_ms": verify_ms,
        "independent_scan_ms": scan_ms,
        "proof_bytes": len(encoded),
        "prefix_proof_nodes": proof_nodes,
        "receipt_score_exactly_matches_scan": True,
    }


def run(
    sizes: tuple[int, ...] = (1_000, 10_000, 100_000),
    repeats: int = 3,
    seed: int = 9759,
) -> dict[str, Any]:
    if repeats < 1 or any(n < 1 for n in sizes):
        raise ValueError("repeats and sizes must be positive")
    rows: list[dict[str, Any]] = []
    for size in sizes:
        trials = [_trial(size, seed + size + rep) for rep in range(repeats)]
        row = {"n": size, "repeats": repeats}
        for field in (
            "build_ms",
            "proof_generation_ms",
            "verification_ms",
            "independent_scan_ms",
            "proof_bytes",
            "prefix_proof_nodes",
        ):
            row["median_" + field] = statistics.median(
                trial[field] for trial in trials
            )
        row["all_receipts_match_scan"] = all(
            trial["receipt_score_exactly_matches_scan"] for trial in trials
        )
        rows.append(row)
    return {
        "benchmark": "merkle_projective_closure_receipt",
        "version": 1,
        "seed": seed,
        "repeats": repeats,
        "rational_arithmetic": "fractions.Fraction",
        "hash": "SHA-256 over canonical JSON",
        "proofs_per_receipt": 2,
        "proof_type": "strict prefix proofs at 1/r and 0",
        "results": rows,
        "limitations": [
            "Python research prototype; not a Rust production benchmark.",
            "Build includes O(n log n) sorting; balanced tree assembly is linear after sorting.",
            "Proof verifies a score relative to a committed root, not truth/completeness of source evidence.",
            "Anti-rollback/fork prevention requires a trusted monotonic checkpoint or transparency log.",
            "The receipt proves feasibility only when the verified score is <= alpha; it does not prove optimizer optimality.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[1_000, 10_000, 100_000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=9759)
    parser.add_argument("--output", type=str, default="-")
    args = parser.parse_args()
    result = run(tuple(args.sizes), args.repeats, args.seed)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output == "-":
        print(rendered)
    else:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")


if __name__ == "__main__":
    main()
