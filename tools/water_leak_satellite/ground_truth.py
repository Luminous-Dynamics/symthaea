#!/usr/bin/env python3
"""Fail-closed structural validator for permissioned water-leak ground truth.

A PASS means only that the records satisfy this declared schema/label contract
and the requested rights scope. It does not independently verify a physical
leak, establish an acceptable research sample, or qualify a remote-sensing model.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

CORPUS_VERSION = "water-leak-ground-truth-corpus/v1"
RECORD_VERSION = "water-leak-ground-truth/v1"
OUTCOMES = {"confirmed_leak", "inspected_no_leak", "inconclusive", "unknown"}
SPLITS = {"train", "threshold_validation", "held_out_test", "prospective_field"}
METHODS = {
    "excavation_confirmed",
    "physical_flow_test",
    "repair_postchange_verified",
    "documented_no_leak_inspection",
    "acoustic_localization_only",
    "utility_record_only",
    "complaint_only",
    "unknown",
}
POSITIVE_METHODS = {
    "excavation_confirmed",
    "physical_flow_test",
    "repair_postchange_verified",
}
ALLOWED_USES = {
    "internal_research",
    "commercial_model_training",
    "commercial_inference",
    "public_redistribution",
}
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
SHA_RE = re.compile(r"^[a-f0-9]{64}$")
PURPOSES = tuple(sorted(ALLOWED_USES))

TOP_KEYS = {
    "schema_version", "record_id", "zone_id", "asset_segment_id", "observed_at",
    "outcome", "location", "verification", "rights", "evaluation_split",
    "split_group_id",
}


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and ID_RE.fullmatch(value) is not None


def _sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def _timestamp(value: Any) -> bool:
    return _parse_timestamp(value) is not None


def validate_record(record: Any, index: int = 0) -> list[str]:
    prefix = f"records[{index}]"
    errors: list[str] = []
    if not isinstance(record, dict):
        return [f"{prefix}: must be an object"]

    extra = set(record) - TOP_KEYS
    missing = TOP_KEYS - set(record)
    if extra:
        errors.append(f"{prefix}: unexpected keys: {', '.join(sorted(extra))}")
    if missing:
        errors.append(f"{prefix}: missing keys: {', '.join(sorted(missing))}")

    if record.get("schema_version") != RECORD_VERSION:
        errors.append(f"{prefix}.schema_version: expected {RECORD_VERSION}")
    for key in ("record_id", "zone_id", "asset_segment_id", "split_group_id"):
        if not _identifier(record.get(key)):
            errors.append(f"{prefix}.{key}: invalid or missing pseudonymous identifier")

    if not _timestamp(record.get("observed_at")):
        errors.append(f"{prefix}.observed_at: must be an ISO-8601 timestamp with timezone")

    outcome = record.get("outcome")
    if not isinstance(outcome, str) or outcome not in OUTCOMES:
        errors.append(f"{prefix}.outcome: unsupported outcome")

    split = record.get("evaluation_split")
    if not isinstance(split, str) or split not in SPLITS:
        errors.append(f"{prefix}.evaluation_split: unsupported split")

    location = record.get("location")
    if not isinstance(location, dict):
        errors.append(f"{prefix}.location: must be an object")
    else:
        allowed_location = {"longitude", "latitude", "horizontal_uncertainty_m", "coordinate_reference_system"}
        extra_location = set(location) - allowed_location
        if extra_location:
            errors.append(f"{prefix}.location: unexpected keys: {', '.join(sorted(extra_location))}")
        for key, low, high in (("longitude", -180.0, 180.0), ("latitude", -90.0, 90.0)):
            value = location.get(key)
            if not _finite_number(value) or not low <= value <= high:
                errors.append(f"{prefix}.location.{key}: invalid coordinate")
        uncertainty = location.get("horizontal_uncertainty_m")
        if not _finite_number(uncertainty) or uncertainty <= 0:
            errors.append(f"{prefix}.location.horizontal_uncertainty_m: must be finite and > 0")
        crs = location.get("coordinate_reference_system", "EPSG:4326")
        if crs != "EPSG:4326":
            errors.append(f"{prefix}.location.coordinate_reference_system: only EPSG:4326 is supported")

    verification = record.get("verification")
    verified_method = None
    independent = False
    post_change = False
    if not isinstance(verification, dict):
        errors.append(f"{prefix}.verification: must be an object")
    else:
        allowed_verification = {"method", "independent", "evidence_sha256", "post_repair_change_observed"}
        extra_verification = set(verification) - allowed_verification
        required_verification = {"method", "independent", "evidence_sha256"}
        missing_verification = required_verification - set(verification)
        if extra_verification:
            errors.append(f"{prefix}.verification: unexpected keys: {', '.join(sorted(extra_verification))}")
        if missing_verification:
            errors.append(f"{prefix}.verification: missing keys: {', '.join(sorted(missing_verification))}")
        verified_method = verification.get("method")
        if not isinstance(verified_method, str) or verified_method not in METHODS:
            errors.append(f"{prefix}.verification.method: unsupported method")
        independent = verification.get("independent") is True
        if not isinstance(verification.get("independent"), bool):
            errors.append(f"{prefix}.verification.independent: must be boolean")
        if not _sha256(verification.get("evidence_sha256")):
            errors.append(f"{prefix}.verification.evidence_sha256: must be a lowercase SHA-256 digest")
        if "post_repair_change_observed" in verification:
            if not isinstance(verification["post_repair_change_observed"], bool):
                errors.append(f"{prefix}.verification.post_repair_change_observed: must be boolean")
            else:
                post_change = verification["post_repair_change_observed"]

    if outcome == "confirmed_leak":
        # Type-check before set membership: malformed JSON may provide a list/dict,
        # which is unhashable and must be rejected as data, not crash validation.
        method_is_positive = isinstance(verified_method, str) and verified_method in POSITIVE_METHODS
        if not method_is_positive:
            errors.append(f"{prefix}: confirmed_leak requires a physical/repair confirmation method, not a proxy")
        if not independent:
            errors.append(f"{prefix}: confirmed_leak requires independent verification")
        if verified_method == "repair_postchange_verified" and not post_change:
            errors.append(f"{prefix}: repair_postchange_verified requires post_repair_change_observed=true")
    elif outcome == "inspected_no_leak":
        if verified_method != "documented_no_leak_inspection":
            errors.append(f"{prefix}: inspected_no_leak requires documented_no_leak_inspection")
        if not independent:
            errors.append(f"{prefix}: inspected_no_leak requires an independently recorded inspection")
    # inconclusive/unknown are valid records but are never converted to negative labels.

    rights = record.get("rights")
    if not isinstance(rights, dict):
        errors.append(f"{prefix}.rights: must be an object")
    else:
        required_rights = {"owner", "allowed_uses", "permission_reference_sha256"}
        allowed_rights = required_rights
        missing_rights = required_rights - set(rights)
        extra_rights = set(rights) - allowed_rights
        if missing_rights:
            errors.append(f"{prefix}.rights: missing keys: {', '.join(sorted(missing_rights))}")
        if extra_rights:
            errors.append(f"{prefix}.rights: unexpected keys: {', '.join(sorted(extra_rights))}")
        owner = rights.get("owner")
        if not isinstance(owner, str) or not owner.strip() or len(owner) > 256:
            errors.append(f"{prefix}.rights.owner: must identify the data owner")
        uses = rights.get("allowed_uses")
        if not isinstance(uses, list) or not uses or any(not isinstance(use, str) or use not in ALLOWED_USES for use in uses) or len(set(uses)) != len(uses):
            errors.append(f"{prefix}.rights.allowed_uses: must be a non-empty unique list of explicit permitted purposes")
        if not _sha256(rights.get("permission_reference_sha256")):
            errors.append(f"{prefix}.rights.permission_reference_sha256: must hash the permission/licence evidence")

    return errors


def label_class(record: dict[str, Any]) -> str:
    """Return positive, negative, or excluded under the strict label contract."""
    outcome = record.get("outcome")
    verification = record.get("verification", {})
    if not isinstance(verification, dict):
        return "excluded"
    method = verification.get("method")
    if outcome == "confirmed_leak" and verification.get("independent") is True and isinstance(method, str) and method in POSITIVE_METHODS:
        if method != "repair_postchange_verified" or verification.get("post_repair_change_observed") is True:
            return "positive"
    if outcome == "inspected_no_leak" and verification.get("independent") is True and method == "documented_no_leak_inspection":
        return "negative"
    return "excluded"


def validate_corpus(document: Any, purpose: str = "internal_research") -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    counts = {
        "records": 0, "positive": 0, "negative": 0, "excluded": 0,
        "purpose_eligible": 0,
    }
    if not isinstance(purpose, str) or purpose not in ALLOWED_USES:
        return [f"unsupported purpose: {purpose}"], counts
    if not isinstance(document, dict):
        return ["corpus: must be an object"], counts
    if set(document) != {"schema_version", "records"}:
        errors.append("corpus: required shape is exactly schema_version + records")
    if document.get("schema_version") != CORPUS_VERSION:
        errors.append(f"corpus.schema_version: expected {CORPUS_VERSION}")
    records = document.get("records")
    if not isinstance(records, list) or not records:
        errors.append("corpus.records: must be a non-empty list")
        return errors, counts

    counts["records"] = len(records)
    identifiers: set[str] = set()
    group_splits: dict[str, str] = {}
    zone_splits: dict[str, str] = {}
    segment_splits: dict[str, str] = {}
    temporal_by_split: dict[str, list[datetime]] = {}
    for index, record in enumerate(records):
        errors.extend(validate_record(record, index))
        if not isinstance(record, dict):
            counts["excluded"] += 1
            continue
        record_id = record.get("record_id")
        if isinstance(record_id, str):
            if record_id in identifiers:
                errors.append(f"records[{index}].record_id: duplicate record_id")
            identifiers.add(record_id)
        group = record.get("split_group_id")
        split = record.get("evaluation_split")
        if isinstance(group, str) and isinstance(split, str) and split in SPLITS:
            previous = group_splits.setdefault(group, split)
            if previous != split:
                errors.append(f"records[{index}].split_group_id: group crosses splits ({previous} and {split})")
        zone = record.get("zone_id")
        if isinstance(zone, str) and isinstance(split, str) and split in SPLITS:
            previous_zone_split = zone_splits.setdefault(zone, split)
            if previous_zone_split != split:
                errors.append(f"records[{index}].zone_id: zone crosses evaluation splits ({previous_zone_split} and {split})")
        segment = record.get("asset_segment_id")
        if isinstance(segment, str) and isinstance(split, str) and split in SPLITS:
            previous_segment_split = segment_splits.setdefault(segment, split)
            if previous_segment_split != split:
                errors.append(f"records[{index}].asset_segment_id: segment crosses evaluation splits ({previous_segment_split} and {split})")
        observed = _parse_timestamp(record.get("observed_at"))
        if observed is not None and isinstance(split, str) and split in SPLITS:
            temporal_by_split.setdefault(split, []).append(observed)
        cls = label_class(record)
        counts[cls] += 1
        rights = record.get("rights")
        uses = rights.get("allowed_uses", []) if isinstance(rights, dict) else []
        if isinstance(uses, list) and purpose in uses:
            counts["purpose_eligible"] += 1
        else:
            errors.append(f"records[{index}].rights: does not explicitly permit {purpose}")
    baseline_times = temporal_by_split.get("train", []) + temporal_by_split.get("threshold_validation", [])
    heldout_times = temporal_by_split.get("held_out_test", [])
    if baseline_times and heldout_times and min(heldout_times) <= max(baseline_times):
        errors.append("held_out_test: timestamps must be strictly later than all train/threshold_validation timestamps")
    return errors, counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--purpose", choices=PURPOSES, default="internal_research")
    args = parser.parse_args(argv)
    try:
        document = json.loads(args.corpus.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"FAIL: cannot read valid UTF-8 JSON corpus: {exc}", file=sys.stderr)
        return 2

    errors, counts = validate_corpus(document, args.purpose)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"FAIL: structural/rights contract rejected; summary={json.dumps(counts, sort_keys=True)}", file=sys.stderr)
        return 1
    print(f"PASS: schema/label/rights contract only; purpose={args.purpose}; summary={json.dumps(counts, sort_keys=True)}")
    print("This PASS does not prove the labels are true, the sample is representative, or a model is qualified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
