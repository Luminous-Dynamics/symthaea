#!/usr/bin/env python3
"""Fail-closed validation for an evidence-qualified symmetry benchmark scorecard.

This tool is benchmark/test infrastructure, not a production learner. It verifies
that the report accounts for every case in a frozen split and binds the run to
the exact protocol, case manifest, split and case-input bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "fixtures" / "v0.1" / "cases.json"
DEFAULT_SCHEMA = ROOT / "schemas" / "run_scorecard.schema.json"
DEFAULT_PROTOCOL = ROOT / "README.md"


class ScorecardError(ValueError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def expected_hashes(
    manifest: dict[str, Any],
    manifest_bytes: bytes,
    protocol_bytes: bytes,
    split: str,
    benchmark_track: str,
) -> tuple[str, str, str, list[dict[str, Any]]]:
    cases = [
        case
        for case in manifest["cases"]
        if case["split"] == split and case["track"] == benchmark_track
    ]
    cases.sort(key=lambda case: case["case_id"])
    if not cases:
        raise ScorecardError(
            f"manifest contains no cases for split={split!r}, track={benchmark_track!r}"
        )
    return (
        sha256(protocol_bytes),
        sha256(manifest_bytes),
        sha256(canonical_json([case["case_id"] for case in cases])),
        cases,
    )


def validate_scorecard(
    scorecard: dict[str, Any],
    manifest: dict[str, Any],
    manifest_bytes: bytes,
    protocol_bytes: bytes,
    schema: dict[str, Any],
) -> None:
    try:
        from jsonschema import Draft202012Validator
    except ImportError as error:
        raise ScorecardError(f"jsonschema is required: {error}") from error

    Draft202012Validator.check_schema(schema)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(scorecard),
        key=lambda error: (list(map(str, error.absolute_path)), error.message),
    )
    if errors:
        first = errors[0]
        location = "/".join(map(str, first.absolute_path)) or "<root>"
        raise ScorecardError(f"JSON Schema violation at {location}: {first.message}")

    protocol_hash, manifest_hash, split_hash, split_cases = expected_hashes(
        manifest,
        manifest_bytes,
        protocol_bytes,
        scorecard["split"],
        scorecard["benchmark_track"],
    )
    expected_input_hash = sha256(canonical_json(split_cases))
    expected_hash_values = {
        "protocol_digest_sha256": protocol_hash,
        "case_manifest_digest_sha256": manifest_hash,
        "split_digest_sha256": split_hash,
        "input_digest_sha256": expected_input_hash,
    }
    for field, expected in expected_hash_values.items():
        if scorecard[field] != expected:
            raise ScorecardError(
                f"{field} mismatch: report={scorecard[field]}, frozen={expected}"
            )

    denominator = scorecard["denominator"]
    results = scorecard["case_results"]
    cases_in_scope = len(split_cases)
    if denominator["cases_in_scope"] != cases_in_scope:
        raise ScorecardError(
            f"cases_in_scope must equal the frozen split size {cases_in_scope}"
        )
    if denominator["cases_attempted"] != len(results):
        raise ScorecardError("cases_attempted must equal case_results length")
    if denominator["cases_attempted"] > denominator["cases_in_scope"]:
        raise ScorecardError("cases_attempted exceeds cases_in_scope")

    expected_ids = {case["case_id"] for case in split_cases}
    result_ids = [result["case_id"] for result in results]
    if len(result_ids) != len(set(result_ids)):
        raise ScorecardError("case_results contains duplicate case IDs")
    unknown_ids = set(result_ids) - expected_ids
    if unknown_ids:
        raise ScorecardError(f"case_results contains cases outside the frozen split: {sorted(unknown_ids)}")

    if scorecard["execution_status"] == "completed" and set(result_ids) != expected_ids:
        missing = sorted(expected_ids - set(result_ids))
        raise ScorecardError(
            f"completed execution omits {len(missing)} in-scope cases: {missing[:5]}"
        )
    if scorecard["execution_status"] == "partial" and len(result_ids) >= len(expected_ids):
        raise ScorecardError("partial execution must leave at least one in-scope case unattempted")
    if scorecard["execution_status"] == "aborted" and results:
        # Aborted runs may record attempted cases, but must not label the run completed.
        pass

    status_counts = {
        "success": 0,
        "abstain": 0,
        "failure": 0,
        "out_of_domain": 0,
    }
    false_promotions = 0
    false_eliminations = 0
    for result in results:
        status_counts[result["status"]] += 1
        false_promotions += int(result["false_exact_promotion"])
        false_eliminations += int(result["false_elimination"])
        if result.get("elapsed_ms", 0) < 0:
            raise ScorecardError(f"negative elapsed_ms for {result['case_id']}")

    expected_denominator = {
        "completed": status_counts["success"],
        "abstained": status_counts["abstain"],
        "failed": status_counts["failure"],
        "out_of_domain": status_counts["out_of_domain"],
    }
    for key, expected in expected_denominator.items():
        if denominator[key] != expected:
            raise ScorecardError(
                f"denominator.{key}={denominator[key]} but case records sum to {expected}"
            )
    if sum(expected_denominator.values()) != denominator["cases_attempted"]:
        raise ScorecardError("outcome counts do not sum to cases_attempted")

    verification = scorecard["verification"]
    if verification["false_exact_promotions"] != false_promotions:
        raise ScorecardError("false_exact_promotions does not match per-case records")
    if verification["false_eliminations"] != false_eliminations:
        raise ScorecardError("false_eliminations does not match per-case records")
    if verification["lift_verified"] > verification["lift_attempts"]:
        raise ScorecardError("lift_verified exceeds lift_attempts")
    if verification["unverified_hard_prune_count"] != 0:
        raise ScorecardError("unverified hard-prune count must be zero")

    resources = scorecard["resources"]
    if resources["wall_time_ms"] < 0 or resources["peak_resident_memory_bytes"] < 0:
        raise ScorecardError("resource counters must be nonnegative")

    for metric in scorecard["metrics"]:
        interval = metric.get("interval")
        if interval is not None:
            if interval["lower"] > interval["upper"]:
                raise ScorecardError(f"metric {metric['name']} has reversed interval bounds")
            if not interval["lower"] <= metric["value"] <= interval["upper"]:
                raise ScorecardError(
                    f"metric {metric['name']} value falls outside its interval"
                )
            if interval["method"] != "none" and metric["sample_count"] == 0:
                raise ScorecardError(
                    f"metric {metric['name']} has an interval but zero samples"
                )


def self_test(manifest: dict[str, Any], manifest_bytes: bytes, protocol_bytes: bytes, schema: dict[str, Any]) -> None:
    """Exercise denominator, duplicate-ID, completeness and digest rejection."""
    track = "exact_finite_structural_stabilizer"
    split = "discovery"
    protocol_hash, manifest_hash, split_hash, cases = expected_hashes(
        manifest, manifest_bytes, protocol_bytes, split, track
    )
    input_hash = sha256(canonical_json(cases))
    first_id = cases[0]["case_id"]
    scorecard = {
        "schema_version": "0.1.0",
        "benchmark_id": "LUMINOUS-SYMM-QUALIFIED-DISCOVERY-001",
        "execution_id": "scorecard-validator-self-test",
        "execution_status": "partial",
        "benchmark_track": track,
        "split": split,
        "protocol_digest_sha256": protocol_hash,
        "case_manifest_digest_sha256": manifest_hash,
        "input_digest_sha256": input_hash,
        "split_digest_sha256": split_hash,
        "method": {
            "name": "validator-self-test",
            "version": "0",
            "method_family": "exact_oracle",
            "source_revision": "self-test",
            "configuration_digest_sha256": "0" * 64,
            "parameter_count": 0,
        },
        "environment": {
            "started_at": "2026-11-18T00:00:00Z",
            "os": "test",
            "architecture": "test",
            "cpu_model": "test",
            "gpu_model": None,
            "thread_count": 1,
            "memory_limit_bytes": 1024,
            "energy_meter": None,
            "software_versions": {"validator": "self-test"},
        },
        "denominator": {
            "cases_in_scope": len(cases),
            "cases_attempted": 1,
            "completed": 1,
            "abstained": 0,
            "failed": 0,
            "out_of_domain": 0,
            "denominator_rule": "all_in_scope_cases_including_abstentions_failures_and_out_of_domain",
        },
        "case_results": [{
            "case_id": first_id,
            "status": "success",
            "elapsed_ms": 1.0,
            "false_exact_promotion": False,
            "false_elimination": False,
            "verification_status": "passed",
        }],
        "metrics": [],
        "verification": {
            "false_exact_promotions": 0,
            "false_eliminations": 0,
            "missed_valid_solutions_after_lift": 0,
            "lift_attempts": 0,
            "lift_verified": 0,
            "unverified_hard_prune_count": 0,
        },
        "resources": {
            "wall_time_ms": 1.0,
            "peak_resident_memory_bytes": 1024,
        },
        "raw_artifacts": [{
            "artifact_id": "self-test",
            "sha256": "0" * 64,
            "media_type": "application/octet-stream",
        }],
    }
    validate_scorecard(scorecard, manifest, manifest_bytes, protocol_bytes, schema)

    def must_reject(candidate: dict[str, Any], label: str) -> None:
        try:
            validate_scorecard(candidate, manifest, manifest_bytes, protocol_bytes, schema)
        except ScorecardError:
            return
        raise ScorecardError(f"self-test mutation was incorrectly accepted: {label}")

    duplicate = json.loads(json.dumps(scorecard))
    duplicate["case_results"].append(dict(duplicate["case_results"][0]))
    duplicate["denominator"]["cases_attempted"] = 2
    duplicate["denominator"]["completed"] = 2
    must_reject(duplicate, "duplicate case IDs")

    missing_completed = json.loads(json.dumps(scorecard))
    missing_completed["execution_status"] = "completed"
    must_reject(missing_completed, "completed scorecard missing holdout cases")

    wrong_counts = json.loads(json.dumps(scorecard))
    wrong_counts["denominator"]["failed"] = 1
    must_reject(wrong_counts, "inconsistent denominator")

    wrong_digest = json.loads(json.dumps(scorecard))
    wrong_digest["input_digest_sha256"] = "f" * 64
    must_reject(wrong_digest, "mutated input digest")

    unsafe_prune = json.loads(json.dumps(scorecard))
    unsafe_prune["verification"]["unverified_hard_prune_count"] = 1
    must_reject(unsafe_prune, "unverified hard prune")

    print(
        "SYMM_Q_SCORECARD_SELF_TEST_PASS: complete split binding, unique cases, "
        "accounting, confidence intervals and zero-unverified-prune checks"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--scorecard", type=Path)
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()

    manifest_bytes = arguments.manifest.read_bytes()
    protocol_bytes = arguments.protocol.read_bytes()
    schema = json.loads(arguments.schema.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_bytes)
    if arguments.self_test:
        self_test(manifest, manifest_bytes, protocol_bytes, schema)
    if arguments.scorecard is not None:
        scorecard = json.loads(arguments.scorecard.read_bytes())
        validate_scorecard(scorecard, manifest, manifest_bytes, protocol_bytes, schema)
        print(f"SYMM_Q_SCORECARD_PASS: {scorecard['execution_id']}")
    elif not arguments.self_test:
        parser.error("provide --scorecard or --self-test")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ScorecardError as error:
        print(f"SYMM_Q_SCORECARD_FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
