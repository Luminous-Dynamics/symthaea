#!/usr/bin/env python3
"""Validate CETB RCAF JSON reports.

This is an isolated research-data validation utility, not production Symthaea code.
It performs schema validation when a schema path is supplied and always checks
cross-record semantic invariants that JSON Schema cannot express conveniently.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


class ReportValidationError(ValueError):
    """Raised when a report violates the RCAF evidence contract."""


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise ReportValidationError(message)


def _finite_number(value: Any, name: str) -> float:
    _need(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name}: expected number")
    number = float(value)
    _need(math.isfinite(number), f"{name}: value must be finite")
    return number


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    if n % 2:
        return ordered[n // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def exact_sign_test(gaps: list[float]) -> tuple[int, int, int, int, float]:
    """Return positive, negative, ties, n_non_tied and exact two-sided p-value."""
    positive = sum(value > 0.0 for value in gaps)
    negative = sum(value < 0.0 for value in gaps)
    ties = len(gaps) - positive - negative
    n = positive + negative
    if n == 0:
        return positive, negative, ties, n, 1.0

    # Exact binomial lower tail under p=0.5, evaluated in log space.
    k = min(positive, negative)
    log_terms = [
        math.lgamma(n + 1)
        - math.lgamma(j + 1)
        - math.lgamma(n - j + 1)
        - n * math.log(2.0)
        for j in range(k + 1)
    ]
    max_log = max(log_terms)
    lower_tail = math.exp(max_log) * sum(math.exp(term - max_log) for term in log_terms)
    return positive, negative, ties, n, min(1.0, 2.0 * lower_tail)


def validate_semantics(report: dict[str, Any], tolerance: float = 1e-9) -> None:
    """Validate participant-level and derived-value invariants."""
    _need(isinstance(report, dict), "report: root must be an object")
    _need(math.isfinite(tolerance) and tolerance >= 0.0, "tolerance must be finite and non-negative")

    _need(report.get("schema_version") == "rcaf.report/1.0.0", "schema_version must be rcaf.report/1.0.0")
    _need(report.get("evidence_status") in {
        "synthetic_fixture", "metadata_only", "signal_analysis", "independent_replay"
    }, "evidence_status is unknown")

    dataset = report.get("dataset")
    analysis = report.get("analysis")
    participants = report.get("participants")
    summary = report.get("summary")
    claim_limits = report.get("claim_limits")
    _need(isinstance(dataset, dict), "dataset must be an object")
    _need(isinstance(analysis, dict), "analysis must be an object")
    _need(isinstance(participants, list), "participants must be an array")
    _need(isinstance(summary, dict), "summary must be an object")
    _need(isinstance(claim_limits, dict), "claim_limits must be an object")

    # Dataset flow: each stage is a subset of the previous stage.
    flow = dataset.get("participant_flow")
    _need(isinstance(flow, dict), "dataset.participant_flow must be an object")
    flow_keys = ("acquired_or_recruited", "released", "qc_eligible", "analysis_included", "excluded")
    counts: dict[str, int] = {}
    for key in flow_keys:
        value = flow.get(key)
        _need(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
              f"participant_flow.{key} must be a non-negative integer")
        counts[key] = value
    _need(
        counts["acquired_or_recruited"] >= counts["released"] >= counts["qc_eligible"] >= counts["analysis_included"],
        "participant flow must satisfy acquired >= released >= QC eligible >= analysis included",
    )
    _need(counts["analysis_included"] == len(participants),
          "participant_flow.analysis_included must equal participant row count")

    if dataset.get("access_state") == "downloaded_verified":
        digest = dataset.get("file_manifest_sha256")
        _need(isinstance(digest, str) and re.fullmatch(r"[a-f0-9]{64}", digest) is not None,
              "downloaded_verified requires a lowercase SHA-256 file manifest digest")
    if report["evidence_status"] == "synthetic_fixture":
        _need(dataset.get("access_state") == "not_downloaded",
              "synthetic_fixture must not claim downloaded data")
        _need(dataset.get("doi") in (None, ""),
              "synthetic_fixture must not assert a real dataset DOI")

    # Frozen configuration and ROI alignment identity.
    roi_digest = analysis.get("roi_order_sha256")
    _need(isinstance(roi_digest, str) and re.fullmatch(r"[a-f0-9]{64}", roi_digest) is not None,
          "analysis.roi_order_sha256 must be a lowercase SHA-256 digest")
    segment_volumes = analysis.get("segment_volumes")
    _need(isinstance(segment_volumes, int) and not isinstance(segment_volumes, bool) and segment_volumes > 0,
          "analysis.segment_volumes must be a positive integer")
    neutral = _finite_number(analysis.get("neutral_sign_threshold"), "analysis.neutral_sign_threshold")
    _need(0.0 <= neutral <= 1.0, "analysis.neutral_sign_threshold must be in [0, 1]")

    # Participant identity and paired measurements.
    _need(len(participants) >= 2, "at least two participant rows are required")
    seen: set[str] = set()
    within: list[float] = []
    between: list[float] = []
    gaps: list[float] = []
    for index, participant in enumerate(participants):
        prefix = f"participants[{index}]"
        _need(isinstance(participant, dict), f"{prefix} must be an object")
        participant_id = participant.get("participant_key")
        _need(isinstance(participant_id, str) and participant_id.strip() != "",
              f"{prefix}.participant_key must be a non-empty string")
        _need(participant_id not in seen, f"{prefix}.participant_key is duplicated")
        seen.add(participant_id)

        w = _finite_number(participant.get("within_similarity"), f"{prefix}.within_similarity")
        b = _finite_number(participant.get("between_similarity"), f"{prefix}.between_similarity")
        g = _finite_number(participant.get("paired_gap"), f"{prefix}.paired_gap")
        _need(0.0 <= w <= 1.0, f"{prefix}.within_similarity must be in [0, 1]")
        _need(0.0 <= b <= 1.0, f"{prefix}.between_similarity must be in [0, 1]")
        _need(-1.0 <= g <= 1.0, f"{prefix}.paired_gap must be in [-1, 1]")
        _need(abs(g - (w - b)) <= tolerance, f"{prefix}.paired_gap does not equal within_similarity - between_similarity")

        for repeat_field in ("within_repeat_count", "between_repeat_count"):
            repeat_count = participant.get(repeat_field)
            _need(isinstance(repeat_count, int) and not isinstance(repeat_count, bool) and repeat_count > 0,
                  f"{prefix}.{repeat_field} must be a positive integer")

        within.append(w)
        between.append(b)
        gaps.append(g)

    # Recompute the headline summaries from the participant-level rows.
    n = len(participants)
    _need(summary.get("participant_count") == n, "summary.participant_count does not match participant rows")
    for field, expected, values in (
        ("median_within_similarity", _median(within), within),
        ("median_between_similarity", _median(between), between),
        ("median_paired_gap", _median(gaps), gaps),
    ):
        observed = _finite_number(summary.get(field), f"summary.{field}")
        _need(abs(observed - expected) <= tolerance, f"summary.{field} does not match participant rows")

    positive_fraction = sum(value > 0.0 for value in gaps) / n
    observed_fraction = _finite_number(summary.get("positive_gap_fraction"), "summary.positive_gap_fraction")
    _need(abs(observed_fraction - positive_fraction) <= tolerance,
          "summary.positive_gap_fraction does not match participant rows")

    sign = summary.get("sign_test")
    _need(isinstance(sign, dict), "summary.sign_test must be an object")
    p_pos, p_neg, p_ties, p_non_tied, p_exact = exact_sign_test(gaps)
    for field, expected in (
        ("positive_count", p_pos),
        ("negative_count", p_neg),
        ("tie_count", p_ties),
        ("non_tied_count", p_non_tied),
    ):
        _need(sign.get(field) == expected, f"summary.sign_test.{field} does not match participant rows")
    observed_p = _finite_number(sign.get("p_two_sided_exact"), "summary.sign_test.p_two_sided_exact")
    _need(0.0 <= observed_p <= 1.0, "summary.sign_test.p_two_sided_exact must be in [0, 1]")
    _need(abs(observed_p - p_exact) <= max(tolerance, 1e-12),
          "summary.sign_test.p_two_sided_exact does not match the exact binomial sign test")

    ci = summary.get("participant_bootstrap_ci")
    _need(isinstance(ci, dict), "summary.participant_bootstrap_ci must be an object")
    ci_lower = _finite_number(ci.get("lower"), "participant_bootstrap_ci.lower")
    ci_upper = _finite_number(ci.get("upper"), "participant_bootstrap_ci.upper")
    _need(-1.0 <= ci_lower <= ci_upper <= 1.0, "participant bootstrap interval is invalid")
    confidence = _finite_number(ci.get("confidence_level"), "participant_bootstrap_ci.confidence_level")
    _need(0.0 < confidence < 1.0, "participant bootstrap confidence_level must be in (0, 1)")
    draws = ci.get("draws")
    _need(isinstance(draws, int) and not isinstance(draws, bool) and draws >= 100,
          "participant bootstrap draws must be an integer >= 100")
    seed = ci.get("seed")
    _need(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
          "participant bootstrap seed must be a non-negative integer")

    edge = summary.get("edge_sign_transitions")
    if edge is not None:
        _need(isinstance(edge, dict), "summary.edge_sign_transitions must be an object")
        total = edge.get("total_edges")
        reversals = edge.get("polarity_reversal_count")
        changes = edge.get("sign_state_change_count")
        _need(isinstance(total, int) and not isinstance(total, bool) and total > 0,
              "edge_sign_transitions.total_edges must be positive")
        _need(isinstance(reversals, int) and 0 <= reversals <= total,
              "edge_sign_transitions.polarity_reversal_count is invalid")
        _need(isinstance(changes, int) and reversals <= changes <= total,
              "edge_sign_transitions.sign_state_change_count is invalid")
        edge_threshold = _finite_number(edge.get("neutral_threshold"), "edge_sign_transitions.neutral_threshold")
        _need(0.0 <= edge_threshold <= 1.0, "edge_sign_transitions.neutral_threshold must be in [0, 1]")

    _need(claim_limits.get("phenomenal_experience_inference") == "not_established_by_rcaf",
          "RCAF claim ceiling for phenomenal experience is missing or altered")
    _need(claim_limits.get("causal_interpretation") == "not_established_by_observational_similarity",
          "RCAF claim ceiling for causal interpretation is missing or altered")
    _need(claim_limits.get("clinical_use") == "not_validated_for_clinical_use",
          "RCAF claim ceiling for clinical use is missing or altered")


def validate_report(report_path: Path, schema_path: Path | None = None, tolerance: float = 1e-9) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if schema_path is not None:
        try:
            import jsonschema
        except ImportError as exc:
            raise ReportValidationError(
                "schema validation requires the optional 'jsonschema' package; install requirements-validation.txt"
            ) from exc
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        jsonschema.Draft202012Validator.check_schema(schema)
        errors = sorted(
            jsonschema.Draft202012Validator(schema).iter_errors(report),
            key=lambda error: (list(error.absolute_path), error.message),
        )
        if errors:
            first = errors[0]
            location = "/".join(str(part) for part in first.absolute_path) or "<root>"
            raise ReportValidationError(f"JSON Schema violation at {location}: {first.message}")
    validate_semantics(report, tolerance=tolerance)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="RCAF report JSON file")
    parser.add_argument("--schema", type=Path, help="optional Draft 2020-12 JSON Schema")
    parser.add_argument("--tolerance", type=float, default=1e-9)
    args = parser.parse_args(argv)
    try:
        validate_report(args.report, args.schema, args.tolerance)
    except (OSError, json.JSONDecodeError, ReportValidationError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print("PASS: RCAF report validation" + (" (schema + semantic)" if args.schema else " (semantic)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
