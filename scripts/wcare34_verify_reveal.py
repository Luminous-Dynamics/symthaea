#!/usr/bin/env python3
"""Verify WCARE-34 commit/reveal integrity before candidate execution.

This script performs no candidate scoring. It proves only that the revealed bytes
match the preregistered commitment for the frozen candidate epoch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
FREEZE_PATH = ROOT / "docs/release/evidence/WCARE34_CANDIDATE_FREEZE_V1.json"
ALLOWED_LINEAGES = {
    "ExternalHumanOrOrganization",
    "IndependentModelSession",
    "SameDevelopmentLineage",
    "Mixed",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def emit(receipt: dict, output: Path | None) -> int:
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    sys.stdout.write(encoded)
    return 0 if receipt["classification"] == "REVEAL_VERIFIED" else 1


def invalid(detail: str, *, output: Path | None, extra: dict | None = None) -> int:
    receipt = {
        "authority": "MeasurementOnly",
        "classification": "INVALID_HOLDOUT",
        "detail": detail,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        receipt.update(extra)
    return emit(receipt, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("commitment", type=Path)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("case_ids", type=Path)
    parser.add_argument("scoring_spec", type=Path)
    parser.add_argument("adjudication_spec", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    try:
        freeze_bytes = read_bytes(FREEZE_PATH)
        commitment_bytes = read_bytes(args.commitment)
        bundle_bytes = read_bytes(args.bundle)
        case_id_bytes = read_bytes(args.case_ids)
        scoring_bytes = read_bytes(args.scoring_spec)
        adjudication_bytes = read_bytes(args.adjudication_spec)
        freeze = json.loads(freeze_bytes)
        commitment = json.loads(commitment_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return invalid(f"artifact_read_or_parse_failed:{type(exc).__name__}", output=args.receipt)

    required = {
        "protocol_version",
        "candidate_epoch",
        "candidate_sha",
        "evaluator_lineage_class",
        "bundle_sha256",
        "bundle_byte_length",
        "case_count",
        "case_id_commitment_sha256",
        "scoring_spec_sha256",
        "adjudication_spec_sha256",
        "commitment_created_utc",
    }
    missing = sorted(required.difference(commitment))
    if missing:
        return invalid("commitment_missing_required_fields", output=args.receipt, extra={"missing": missing})

    for field in ("protocol_version", "candidate_epoch", "candidate_sha"):
        if commitment[field] != freeze[field]:
            return invalid(
                f"candidate_freeze_mismatch:{field}",
                output=args.receipt,
                extra={"committed": commitment[field], "frozen": freeze[field]},
            )

    if commitment["evaluator_lineage_class"] not in ALLOWED_LINEAGES:
        return invalid("invalid_evaluator_lineage_class", output=args.receipt)

    if commitment["bundle_byte_length"] != len(bundle_bytes):
        return invalid("bundle_byte_length_mismatch", output=args.receipt)
    if commitment["bundle_sha256"] != sha256_bytes(bundle_bytes):
        return invalid("bundle_sha256_mismatch", output=args.receipt)

    try:
        census_text = case_id_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return invalid("case_census_not_utf8", output=args.receipt)
    if not case_id_bytes.endswith(b"\n"):
        return invalid("case_census_missing_final_lf", output=args.receipt)
    if b"\r" in case_id_bytes:
        return invalid("case_census_not_lf_canonical", output=args.receipt)

    case_ids = census_text[:-1].split("\n")
    if not case_ids or any(not case_id for case_id in case_ids):
        return invalid("case_census_contains_blank_id", output=args.receipt)
    if len(set(case_ids)) != len(case_ids):
        return invalid("case_census_contains_duplicate_id", output=args.receipt)
    if commitment["case_count"] != len(case_ids):
        return invalid("case_count_mismatch", output=args.receipt)
    if commitment["case_id_commitment_sha256"] != sha256_bytes(case_id_bytes):
        return invalid("case_census_sha256_mismatch", output=args.receipt)
    if commitment["scoring_spec_sha256"] != sha256_bytes(scoring_bytes):
        return invalid("scoring_spec_sha256_mismatch", output=args.receipt)
    if commitment["adjudication_spec_sha256"] != sha256_bytes(adjudication_bytes):
        return invalid("adjudication_spec_sha256_mismatch", output=args.receipt)

    receipt = {
        "authority": "MeasurementOnly",
        "classification": "REVEAL_VERIFIED",
        "candidate_epoch": freeze["candidate_epoch"],
        "candidate_sha": freeze["candidate_sha"],
        "commitment_sha256": sha256_bytes(commitment_bytes),
        "bundle_sha256": sha256_bytes(bundle_bytes),
        "case_count": len(case_ids),
        "case_id_commitment_sha256": sha256_bytes(case_id_bytes),
        "scoring_spec_sha256": sha256_bytes(scoring_bytes),
        "adjudication_spec_sha256": sha256_bytes(adjudication_bytes),
        "evaluator_lineage_class": commitment["evaluator_lineage_class"],
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_execution_authorized_by_this_receipt": False,
        "phenomenal_experience_established": False,
        "suffering_established": False,
        "moral_patienthood_established": False,
        "veto_authority_granted": False,
        "self_preservation_authority_granted": False,
    }
    return emit(receipt, args.receipt)


if __name__ == "__main__":
    raise SystemExit(main())
