#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import verify_vart_001_archive_capsule as archive


def capsule() -> dict:
    data = {
        "schema": archive.SCHEMA,
        "experiment_id": archive.EXPERIMENT_ID,
        "status": "qualified",
        "development_use": "historical_only_no_tuning",
        "subject_source": {
            "head": "1" * 40,
            "tree": "2" * 40,
            "source_closure_receipt_sha256": "3" * 64,
        },
        "instrument_source": {
            "head": "4" * 40,
            "tree": "5" * 40,
            "qualification_receipt_sha256": "6" * 64,
            "source_closure_receipt_sha256": "7" * 64,
        },
        "freeze_v3_sha256": "8" * 64,
        "campaign_receipt_sha256": "9" * 64,
        "first_unblinding_analysis_receipt_sha256": "a" * 64,
        "post_unblinding_claim_audit_receipt_sha256": "b" * 64,
        "final_bounded_claim_packet_sha256": "c" * 64,
        "evidence_closure_sha256": "d" * 64,
        "superseded_lineages": [
            {"kind": "confirmatory_freeze", "sha256": "e" * 64, "superseded_by_sha256": "f" * 64}
        ],
        "artifact_bindings": [
            {"name": "freeze_v3", "sha256": "8" * 64},
            {"name": "sealed_evidence", "sha256": "d" * 64}
        ],
        "external_archive_anchor": {"required": True, "manifest_sha256": None, "record_locator": "test://anchor"},
        "claim_ceiling": "benchmark_family_bounded",
        "general_creativity_claim_authorized": False,
        "general_intelligence_claim_authorized": False,
        "claim_authorized": False,
    }
    data["external_archive_anchor"]["manifest_sha256"] = hashlib.sha256(archive.canonical_without_self_anchor(data)).hexdigest()
    return data


def write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def expect_reject(path: Path, data: dict, needle: str) -> None:
    write(path, data)
    try:
        archive.verify(path)
    except archive.Reject as exc:
        assert needle in str(exc), (needle, str(exc))
        return
    raise AssertionError(f"expected rejection containing {needle}")


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "capsule.json"
        good = capsule()
        write(path, good)
        assert archive.verify(path)["verdict"] == "VART_001_ARCHIVE_CAPSULE_QUALIFIED"

        bad = capsule()
        bad["development_use"] = "benchmark_tuning_allowed"
        expect_reject(path, bad, "anti-tuning")

        bad = capsule()
        bad["general_creativity_claim_authorized"] = True
        expect_reject(path, bad, "general creativity claim")

        bad = capsule()
        bad["artifact_bindings"][0]["sha256"] = "0" * 64
        expect_reject(path, bad, "ARCHIVE_MANIFEST_ANCHOR_MISMATCH")

        bad = capsule()
        bad["subject_source"]["head"] = "short"
        expect_reject(path, bad, "subject.head")

    print("PASS: VART-001 archive capsule acceptance + tamper rejection")


if __name__ == "__main__":
    main()
