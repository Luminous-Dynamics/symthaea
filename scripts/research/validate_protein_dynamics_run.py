#!/usr/bin/env python3
"""Fail-closed validator for Symthaea protein-dynamics benchmark run records.

Requires Python >=3.11 and jsonschema >=4.18,<5. This validates provenance,
accounting, and record integrity; it does not validate biological truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError:  # pragma: no cover - exercised by CLI install guidance
    print("ERROR: missing dependency 'jsonschema'; install jsonschema>=4.18,<5", file=sys.stderr)
    raise SystemExit(2)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCHEMA = REPO_ROOT / "docs/research/PROTEIN_DYNAMICS_RUN_RECORD_SCHEMA_v1.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path, label: str) -> tuple[Any | None, bytes | None, str | None]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return None, None, f"cannot read {label} {path}: {exc}"
    try:
        return json.loads(raw), raw, None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, raw, f"{label} is not valid UTF-8 JSON ({path}): {exc}"


def json_path(path: Any) -> str:
    parts = list(path)
    return "/" + "/".join(str(part).replace("~", "~0").replace("/", "~1") for part in parts)


def validate_record(
    record_path: Path,
    schema_path: Path,
    expected_targets_path: Path,
    artifact_root: Path,
) -> tuple[list[str], list[str], str | None]:
    errors: list[str] = []
    warnings: list[str] = []

    schema, _, err = read_json(schema_path, "JSON Schema")
    if err:
        return [err], warnings, None
    record, _, err = read_json(record_path, "run record")
    if err:
        return [err], warnings, None
    targets_doc, targets_raw, err = read_json(expected_targets_path, "expected-target manifest")
    if err:
        return [err], warnings, None

    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:  # jsonschema exposes several useful subclasses
        return [f"the run-record schema itself is invalid: {exc}"], warnings, None

    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors.extend(
        f"schema violation at {json_path(e.absolute_path)}: {e.message}"
        for e in sorted(validator.iter_errors(record), key=lambda e: json_path(e.absolute_path))
    )
    if not isinstance(record, dict):
        return errors or ["run record must be a JSON object"], warnings, None
    if not isinstance(targets_doc, dict) or set(targets_doc) != {"target_ids"}:
        errors.append("expected-target manifest must be an object containing only 'target_ids'")
        return errors, warnings, record.get("run_status")
    expected_ids = targets_doc.get("target_ids")
    if not isinstance(expected_ids, list) or not expected_ids or any(not isinstance(x, str) or not x for x in expected_ids):
        errors.append("expected-target manifest 'target_ids' must be a non-empty list of non-empty strings")
        return errors, warnings, record.get("run_status")
    if len(expected_ids) != len(set(expected_ids)):
        errors.append("expected-target manifest contains duplicate target IDs")
    if expected_ids != sorted(expected_ids):
        errors.append("expected-target IDs must be sorted lexicographically before freezing the manifest")

    prereg_for_hash = record.get("preregistration")
    expected_digest = prereg_for_hash.get("target_manifest_sha256") if isinstance(prereg_for_hash, dict) else None
    observed_digest = sha256_bytes(targets_raw or b"")
    if expected_digest != observed_digest:
        errors.append(
            "expected-target manifest SHA-256 does not match preregistration.target_manifest_sha256 "
            f"(record={expected_digest!r}, observed={observed_digest})"
        )

    status = record.get("run_status")
    rows = record.get("targets", [])
    if not isinstance(rows, list):
        rows = []
    invalid_id_rows = [index for index, row in enumerate(rows) if not isinstance(row, dict) or not isinstance(row.get("target_id"), str) or not row.get("target_id")]
    if invalid_id_rows:
        errors.append("target rows have missing or non-string target_id at indices: " + ", ".join(map(str, invalid_id_rows)))
    ids = [row.get("target_id") for row in rows if isinstance(row, dict) and isinstance(row.get("target_id"), str) and row.get("target_id")]
    if len(ids) != len(set(ids)):
        errors.append("run record contains duplicate target_id rows")
    expected_set = set(expected_ids)
    observed_set = set(ids)
    unexpected = sorted(observed_set - expected_set)
    missing = sorted(expected_set - observed_set)
    if unexpected:
        errors.append("run contains unregistered target IDs: " + ", ".join(unexpected))
    if missing:
        message = "expected targets absent from record: " + ", ".join(missing)
        if status == "complete":
            errors.append(message)
        else:
            warnings.append(message + " (permitted only because run_status is not complete)")

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        label = f"target[{index}] ({row.get('target_id', '<missing-id>')})"
        counts = [row.get(key) for key in ("candidate_conformers", "valid_conformers", "invalid_conformers")]
        if all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in counts):
            candidates, valid, invalid = counts
            if candidates != valid + invalid:
                errors.append(f"{label}: candidate_conformers must equal valid_conformers + invalid_conformers")
        outcome = row.get("outcome")
        metrics = row.get("metrics", [])
        diag = row.get("diagnostics", {})
        codes = diag.get("error_codes", []) if isinstance(diag, dict) else []
        if isinstance(outcome, str) and outcome in {"FAIL_EXECUTION", "FAIL_INVALID_OR_MISSING_METRICS"} and (
            not isinstance(codes, list) or not codes or not all(isinstance(code, str) and code for code in codes)
        ):
            errors.append(f"{label}: failure outcome must retain at least one diagnostics.error_codes entry")
        if status == "complete" and outcome == "NOT_RUN":
            errors.append(f"{label}: a complete run cannot contain a NOT_RUN target")
        if outcome == "PASS_WITH_VALID_METRICS" and not metrics:
            errors.append(f"{label}: PASS_WITH_VALID_METRICS requires at least one metric")
        metric_keys: set[tuple[Any, ...]] = set()
        for metric_index, metric in enumerate(metrics if isinstance(metrics, list) else []):
            if not isinstance(metric, dict):
                continue
            mlabel = f"{label} metric[{metric_index}]"
            value = metric.get("value")
            if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(float(value)):
                errors.append(f"{mlabel}: metric value must be finite (NaN/Infinity are never valid scores)")
            key = (metric.get("metric_name"), metric.get("observable_id"), metric.get("condition_id"), metric.get("evaluation_role"))
            if not all(value is None or isinstance(value, str) for value in key):
                errors.append(f"{mlabel}: metric identity fields must be strings (condition_id may be null)")
            elif key in metric_keys:
                errors.append(f"{mlabel}: duplicate metric identity within target")
            else:
                metric_keys.add(key)
            role, evidence, overlap = metric.get("evaluation_role"), row.get("evidence_class"), metric.get("observable_overlap")
            if role == "independent_held_out_observable" and overlap != "held_out_unseen":
                errors.append(f"{mlabel}: independent held-out metric must have observable_overlap=held_out_unseen")
            if role == "post_reweight_same_observable" and overlap != "used_for_reweighting":
                errors.append(f"{mlabel}: post-reweight metric must disclose observable_overlap=used_for_reweighting")
            if evidence == "simulation_generated" and role != "simulation_only":
                errors.append(f"{mlabel}: simulation-generated targets must use evaluation_role=simulation_only")
            if evidence != "simulation_generated" and role == "simulation_only":
                errors.append(f"{mlabel}: simulation_only metric attached to a non-simulation target")

    # Resource totals must reconcile with the per-target ledger, not just look plausible in isolation.
    resource = record.get("resource_accounting", {})
    if isinstance(resource, dict):
        reported_conformers = resource.get("candidate_conformers_generated")
        row_conformers = [
            row.get("candidate_conformers")
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("candidate_conformers"), int)
            and not isinstance(row.get("candidate_conformers"), bool) and row.get("candidate_conformers") >= 0
        ]
        if len(row_conformers) == len(rows) and isinstance(reported_conformers, int) and not isinstance(reported_conformers, bool):
            if sum(row_conformers) != reported_conformers:
                errors.append(
                    "resource_accounting.candidate_conformers_generated must equal the sum of target candidate_conformers "
                    f"(reported={reported_conformers}, target_sum={sum(row_conformers)})"
                )

    # Any artifact claimed in the record must resolve locally under the chosen root and match its hash.
    root = artifact_root.resolve()
    artifacts_value = record.get("output_artifacts", [])
    artifacts = artifacts_value if isinstance(artifacts_value, list) else []
    verified_count = 0
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            continue
        location = artifact.get("uri_or_relative_path")
        digest = artifact.get("sha256")
        label = f"output_artifacts[{index}] ({location!r})"
        if not isinstance(location, str) or not location:
            errors.append(f"{label}: missing local artifact path")
            continue
        parsed = urlparse(location)
        candidate = Path(location)
        if parsed.scheme or candidate.is_absolute():
            errors.append(f"{label}: verifier accepts local relative paths only; remote/absolute paths cannot be independently hashed")
            continue
        try:
            resolved = (root / candidate).resolve()
        except (OSError, RuntimeError, ValueError) as exc:
            errors.append(f"{label}: cannot resolve artifact path: {exc}")
            continue
        if not resolved.is_relative_to(root):
            errors.append(f"{label}: artifact path escapes artifact root")
            continue
        try:
            raw = resolved.read_bytes()
        except (OSError, ValueError) as exc:
            errors.append(f"{label}: cannot read artifact: {exc}")
            continue
        observed = sha256_bytes(raw)
        if observed != digest:
            errors.append(f"{label}: SHA-256 mismatch (record={digest!r}, observed={observed})")
        else:
            verified_count += 1

    attestations = record.get("attestations", {})
    if isinstance(attestations, dict):
        if attestations.get("all_claimed_output_hashes_verified") is True and verified_count != len(artifacts):
            errors.append("attestation claims all output hashes verified, but not every listed artifact verified locally")
        if status == "complete":
            for name in (
                "all_expected_targets_accounted",
                "failures_retained",
                "all_claimed_output_hashes_verified",
                "held_out_observables_sealed_until_scoring",
            ):
                if attestations.get(name) is not True:
                    errors.append(f"complete run requires attestations.{name}=true")
            if len(artifacts) == 0:
                errors.append("complete run must retain at least one output artifact")
            if missing or unexpected or len(ids) != len(expected_ids):
                errors.append("complete run must have an exact one-row-per-expected-target census")

    prereg = record.get("preregistration", {})
    created = record.get("created_at")
    frozen = prereg.get("frozen_at") if isinstance(prereg, dict) else None
    try:
        if created and frozen and datetime.fromisoformat(frozen.replace("Z", "+00:00")) > datetime.fromisoformat(created.replace("Z", "+00:00")):
            errors.append("preregistration.frozen_at must not be later than created_at")
    except (AttributeError, TypeError, ValueError):
        # JSON Schema's date-time format checker reports malformed values.
        pass

    return errors, warnings, status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, help="run-record JSON")
    parser.add_argument(
        "--expected-targets",
        required=True,
        type=Path,
        help='frozen JSON manifest: {"target_ids":[sorted, unique IDs...]}; its raw-byte SHA-256 must match preregistration',
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA, help=f"run-record JSON Schema (default: {DEFAULT_SCHEMA})")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=None,
        help="root for relative artifact paths (default: directory containing record)",
    )
    args = parser.parse_args(argv)
    artifact_root = args.artifact_root or args.record.resolve().parent
    try:
        errors, warnings, status = validate_record(args.record, args.schema, args.expected_targets, artifact_root)
    except Exception as exc:
        print(f"ERROR: validator encountered malformed input or an unexpected validation failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("INVALID_RUN_RECORD: validation failed closed", file=sys.stderr)
        return 2
    for warning in warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        print("INVALID_RUN_RECORD: one or more structural, provenance, outcome, or artifact checks failed", file=sys.stderr)
        return 2
    if status == "complete":
        print("VALID_COMPLETE_RUN_RECORD: record, frozen target census, local artifact hashes, and declared invariants verified; this is not biological validation")
    else:
        print(f"VALID_{str(status).upper()}_RECORD: record checks passed; this status does not indicate a completed benchmark or scientific pass")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
