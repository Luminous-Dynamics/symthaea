#!/usr/bin/env python3
"""Matched-seed comparison of PITB contexts against the full-context baseline."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from pitb_support_landscape_v3 import ContractError, analyze, group_key, synthetic_rows

COST_FIELDS = ("model_calls", "tokens", "wall_ms", "checker_ms", "cost_microusd")


def mcnemar_exact_two_sided(candidate_only: int, baseline_only: int) -> Fraction:
    """Exact two-sided McNemar p-value using the conditional Binomial(d, 1/2) law."""
    if candidate_only < 0 or baseline_only < 0:
        raise ContractError("discordant pair counts must be non-negative")
    discordant = candidate_only + baseline_only
    if discordant == 0:
        return Fraction(1, 1)
    lower_tail_numerator = sum(math.comb(discordant, k)
                               for k in range(min(candidate_only, baseline_only) + 1))
    return min(Fraction(1, 1), Fraction(2 * lower_tail_numerator, 2 ** discordant))


def holm_adjust(p_values: dict[str, Fraction]) -> dict[str, Fraction]:
    """Holm step-down familywise adjustment, retaining exact rational values."""
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    running = Fraction(0, 1)
    adjusted = {}
    for rank, (key, p_value) in enumerate(ordered):
        candidate = min(Fraction(1, 1), (count - rank) * p_value)
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def _fraction_fields(value: Fraction) -> dict:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "value": float(value),
    }


def analyze_paired(rows: list[dict], min_replicates: int = 2,
                   max_variables: int = 8) -> dict:
    # Reuse the strict v3 ledger validator; paired statistics never operate on
    # incomplete, unpaired, over-budget, or otherwise rejected source rows.
    analyze(rows, min_replicates=min_replicates, max_variables=max_variables)
    groups = defaultdict(list)
    for row in rows:
        groups[group_key(row)].append(row)

    group_reports = []
    for gid, group_rows in sorted(groups.items()):
        first = group_rows[0]
        full_context = tuple(first["mutable_declarations"])
        by_context = defaultdict(dict)
        for row in group_rows:
            subset = tuple(row["requested_declarations"])
            by_context[subset][row["replicate_index"]] = row

        baseline_rows = by_context.get(full_context)
        if baseline_rows is None:
            raise ContractError(f"group {gid}: full-context baseline missing")
        replicate_ids = sorted(baseline_rows)
        baseline_success = {
            rep: baseline_rows[rep]["outcome"] == "verified_success"
            for rep in replicate_ids
        }

        comparisons = []
        raw_p_values = {}
        for subset in sorted(by_context, key=lambda item: (len(item), item)):
            if subset == full_context:
                continue
            candidate_rows = by_context[subset]
            first_candidate = candidate_rows[replicate_ids[0]]
            if not first_candidate["context_well_formed"]:
                continue
            both_success = candidate_only = baseline_only = both_fail = 0
            paired_cost_deltas = {field: 0 for field in COST_FIELDS}
            candidate_successes = baseline_successes = 0
            for rep in replicate_ids:
                candidate = candidate_rows[rep]
                candidate_pass = candidate["outcome"] == "verified_success"
                baseline_pass = baseline_success[rep]
                candidate_successes += int(candidate_pass)
                baseline_successes += int(baseline_pass)
                if candidate_pass and baseline_pass:
                    both_success += 1
                elif candidate_pass:
                    candidate_only += 1
                elif baseline_pass:
                    baseline_only += 1
                else:
                    both_fail += 1
                for field in COST_FIELDS:
                    paired_cost_deltas[field] += candidate[field] - baseline_rows[rep][field]

            n = len(replicate_ids)
            p_value = mcnemar_exact_two_sided(candidate_only, baseline_only)
            key = json.dumps(list(subset), separators=(",", ":"))
            raw_p_values[key] = p_value
            comparisons.append({
                "requested_declarations": list(subset),
                "replicate_pairs": n,
                "both_success": both_success,
                "candidate_only_success": candidate_only,
                "full_context_only_success": baseline_only,
                "both_fail": both_fail,
                "candidate_success_rate": candidate_successes / n,
                "full_context_success_rate": baseline_successes / n,
                "paired_success_rate_difference": (candidate_successes - baseline_successes) / n,
                "mcnemar_exact_two_sided_p": _fraction_fields(p_value),
                "mean_paired_resource_delta_candidate_minus_full": {
                    field: paired_cost_deltas[field] / n for field in COST_FIELDS
                },
            })

        adjusted = holm_adjust(raw_p_values)
        for comparison in comparisons:
            key = json.dumps(comparison["requested_declarations"], separators=(",", ":"))
            comparison["holm_adjusted_p"] = _fraction_fields(adjusted[key])

        group_reports.append({
            "group_id": gid,
            "study_id": first["study_id"],
            "target_id": first["target_id"],
            "mutable_declarations": first["mutable_declarations"],
            "replicate_count": first["replicate_count"],
            "full_context": list(full_context),
            "comparison_family_size": len(comparisons),
            "comparisons": comparisons,
            "interpretation_boundary": (
                "Matched-seed associations against the full-context baseline. "
                "McNemar inference assumes independent replicate pairs and a valid "
                "frozen pairing protocol; it does not prove logical necessity or "
                "causality. Holm adjustment controls familywise error across the "
                "reported candidate-context comparisons."
            ),
        })

    return {
        "schema_version": "pitb.support-landscape-paired-report.v0",
        "group_count": len(group_reports),
        "groups": group_reports,
    }


def self_test() -> None:
    budget = {
        "max_model_calls": 2, "max_tokens": 500, "max_wall_ms": 100,
        "max_checker_ms": 10, "max_cost_microusd": 10,
    }
    rows = synthetic_rows(
        "paired-self-test", "matched-outcomes", "component-a", ["a", "b"], {},
        lambda _subset: True, budget, replicate_count=10,
        success_counts={(): 0, ("a",): 8, ("b",): 9, ("a", "b"): 10},
    )
    report = analyze_paired(rows)
    group = report["groups"][0]
    by_subset = {tuple(item["requested_declarations"]): item for item in group["comparisons"]}
    a = by_subset[("a",)]
    assert a["both_success"] == 8
    assert a["candidate_only_success"] == 0
    assert a["full_context_only_success"] == 2
    assert a["both_fail"] == 0
    assert a["paired_success_rate_difference"] == -0.2
    assert a["mcnemar_exact_two_sided_p"]["numerator"] == 1
    assert a["mcnemar_exact_two_sided_p"]["denominator"] == 2
    assert "holm_adjusted_p" in a
    assert len(group["comparisons"]) == 3
    assert mcnemar_exact_two_sided(0, 0) == Fraction(1, 1)
    assert mcnemar_exact_two_sided(0, 10) == Fraction(1, 512)
    print("PITB_PAIRED_SUPPORT_SELF_TEST_PASS: matched contingency cells, exact McNemar, Holm adjustment, and paired cost deltas")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--min-replicates", type=int, default=2)
    parser.add_argument("--max-variables", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.min_replicates < 2:
            raise ContractError("--min-replicates must be >= 2")
        if not 1 <= args.max_variables <= 10:
            raise ContractError("--max-variables must be in [1,10]")
        if args.self_test:
            if args.outcomes or args.report:
                raise ContractError("--self-test cannot be combined with file paths")
            self_test()
            return 0
        if not args.outcomes or not args.report:
            raise ContractError("--outcomes and --report are required")
        rows = []
        for line_number, line in enumerate(args.outcomes.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ContractError(f"line {line_number}: {exc}") from exc
        if not rows:
            raise ContractError("empty outcome ledger")
        report = analyze_paired(rows, args.min_replicates, args.max_variables)
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"PITB_PAIRED_SUPPORT_ANALYSIS: groups={report['group_count']} output={args.report}")
        return 0
    except (ContractError, OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(f"PITB paired support analysis rejected: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
