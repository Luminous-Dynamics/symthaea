#!/usr/bin/env python3
"""Validate Choro-analysis metrics on independent synthetic fixtures only."""
from __future__ import annotations
import argparse, json, math, random, statistics, sys
from pathlib import Path
from typing import Sequence

SCHEMA = "muse-choro-synthetic-harness-report-v1"
SEED = 90210
PHRASE_LENGTH = 16
HYPERMETRIC_BOUNDARIES = {4, 8, 12, 16}

def validate_phrases(phrases: Sequence[Sequence[int]], phrase_length: int = PHRASE_LENGTH) -> None:
    if not phrases:
        raise ValueError("at least one phrase is required")
    if phrase_length < 1:
        raise ValueError("phrase_length must be positive")
    for i, phrase in enumerate(phrases):
        if not phrase:
            raise ValueError(f"phrase {i} has no events")
        if any(type(p) is not int or p < 1 or p > phrase_length for p in phrase):
            raise ValueError(f"phrase {i} contains an event outside 1..{phrase_length}")
        if len(set(phrase)) != len(phrase):
            raise ValueError(f"phrase {i} has duplicate event positions")

def boundary_alignment_rate(phrases: Sequence[Sequence[int]], phrase_length: int = PHRASE_LENGTH, boundaries: set[int] | None = None) -> float:
    validate_phrases(phrases, phrase_length)
    selected = HYPERMETRIC_BOUNDARIES if boundaries is None else boundaries
    if not selected or any(type(p) is not int or p < 1 or p > phrase_length for p in selected):
        raise ValueError("boundaries must be a non-empty set of positions within the phrase")
    total = sum(len(p) for p in phrases)
    return sum(position in selected for phrase in phrases for position in phrase) / total

def rotate_phrase(phrase: Sequence[int], offset: int, phrase_length: int = PHRASE_LENGTH) -> list[int]:
    if type(offset) is not int or phrase_length < 1:
        raise ValueError("offset must be an integer and phrase_length must be positive")
    if any(type(p) is not int or p < 1 or p > phrase_length for p in phrase):
        raise ValueError("phrase contains an event outside the phrase")
    shift = offset % phrase_length
    return sorted(((p - 1 + shift) % phrase_length) + 1 for p in phrase)

def conditional_transition_entropy(sequence: Sequence[str]) -> float:
    """Compute H(next_symbol | current_symbol) in bits for one sequence."""
    if len(sequence) < 2:
        raise ValueError("at least two symbols are required")
    if any(not isinstance(s, str) or not s.strip() for s in sequence):
        raise ValueError("symbols must be non-empty strings")
    from collections import Counter, defaultdict
    transitions: dict[str, Counter[str]] = defaultdict(Counter)
    for current, following in zip(sequence, sequence[1:]):
        transitions[current][following] += 1
    total = sum(sum(counts.values()) for counts in transitions.values())
    entropy = 0.0
    for counts in transitions.values():
        n = sum(counts.values())
        row_entropy = -sum((v / n) * math.log2(v / n) for v in counts.values())
        entropy += (n / total) * row_entropy
    return entropy

def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("correlation requires equal-length sequences with at least two values")
    values = list(xs) + list(ys)
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
        raise ValueError("correlation values must be finite numeric values")
    xbar, ybar = statistics.mean(xs), statistics.mean(ys)
    dx, dy = [x - xbar for x in xs], [y - ybar for y in ys]
    denom = (sum(x*x for x in dx) * sum(y*y for y in dy)) ** 0.5
    if denom == 0:
        raise ValueError("correlation is undefined for constant input")
    return sum(x*y for x, y in zip(dx, dy)) / denom

def run_synthetic_suite() -> dict:
    # Synthetic phase alignment. Not a universal claim about Choro.
    aligned = [[4, 8, 12, 16] for _ in range(64)]
    counterexample = [[1, 2, 3, 4] for _ in range(64)]
    phase_null = [rotate_phrase(aligned[i], i % PHRASE_LENGTH) for i in range(len(aligned))]
    aligned_rate = boundary_alignment_rate(aligned)
    counter_rate = boundary_alignment_rate(counterexample)
    null_rate = boundary_alignment_rate(phase_null)
    # An authored deterministic cycle vs. a seeded shuffle of the same symbols.
    cycle = ["I", "V", "vi", "IV"] * 16
    shuffled = list(cycle)
    random.Random(SEED).shuffle(shuffled)
    ordered_entropy = conditional_transition_entropy(cycle)
    shuffled_entropy = conditional_transition_entropy(shuffled)
    # Authored trend and its reversal. No values come from the Choro corpus.
    years = [1900 + 3*i for i in range(40)]
    modifications = [1 + i // 5 for i in range(40)]
    reversed_modifications = list(reversed(modifications))
    forward_r = pearson_correlation(years, modifications)
    reversed_r = pearson_correlation(years, reversed_modifications)
    checks = {
        "h1_aligned_fixture_exceeds_counterexample": aligned_rate > counter_rate,
        "h1_balanced_phase_null_matches_counterexample": abs(null_rate - counter_rate) < 1e-12,
        "h1_event_counts_preserved_by_phase_rotations": sum(map(len, aligned)) == sum(map(len, phase_null)),
        "h2_ordered_cycle_has_lower_transition_entropy": ordered_entropy < shuffled_entropy,
        "h2_shuffle_preserves_symbol_multiset": sorted(cycle) == sorted(shuffled),
        "h3_authored_positive_trend_detected": forward_r > 0.95,
        "h3_reversed_counterexample_detected": reversed_r < -0.95,
    }
    return {
        "schema": SCHEMA,
        "kind": "synthetic_fixture_validation_only_not_empirical_evidence",
        "seed": SEED,
        "source_data_used": False, "corpus_data_ingested": False,
        "runtime_capability_claimed": False, "empirical_results": None,
        "fixture_counts": {"phrases_per_h1_fixture": len(aligned), "events_per_phrase": 4, "h2_sequence_length": len(cycle), "h3_synthetic_records": len(years)},
        "results": {
            "h1_phrase_alignment": {"aligned_rate": aligned_rate, "counterexample_rate": counter_rate, "balanced_phase_rotation_null_rate": null_rate, "interpretation": "Distinguishes an authored phase-alignment fixture from a counterexample; does not claim these boundaries characterize Choro."},
            "h2_transition_context": {"ordered_conditional_entropy_bits": ordered_entropy, "shuffled_conditional_entropy_bits": shuffled_entropy, "interpretation": "A deterministic authored cycle has lower next-symbol uncertainty than a seeded shuffle of the same symbols; this is not evidence about Choro harmony."},
            "h3_temporal_trend": {"authored_positive_correlation": forward_r, "reversed_counterexample_correlation": reversed_r, "interpretation": "Detects an authored synthetic trend and its reversal; this is not a corpus-derived historical result."}
        },
        "checks": checks, "passed": all(checks.values()),
        "limitations": [
            "No Choro corpus, transcriptions, audio, copyrighted excerpts, or third-party dataset rows are used.",
            "Synthetic fixtures validate metric direction and guardrails only; they cannot validate a Choro-specific hypothesis.",
            "Rights clearance, data eligibility, held-out-by-composition evaluation and knowledgeable-practitioner review remain open gates."
        ]
    }

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="exit non-zero unless all synthetic checks pass")
    parser.add_argument("--output", type=Path, help="write JSON report to this path")
    args = parser.parse_args(argv)
    report = run_synthetic_suite()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if args.check and not report["passed"]:
        print("FAIL: one or more synthetic checks failed", file=sys.stderr)
        return 1
    if args.check:
        print("PASS: synthetic fixtures only; no empirical Choro claim", file=sys.stderr)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
