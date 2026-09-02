#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "symthaea.vart-001.archive-capsule.v1"
EXPERIMENT_ID = "VART-WORLD-CREATIVE-001"
HEX = set("0123456789abcdef")


class Reject(RuntimeError):
    pass


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise Reject(msg)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise Reject(f"missing capsule: {path}") from exc
    except json.JSONDecodeError as exc:
        raise Reject(f"invalid JSON: {exc}") from exc


def require_sha(value: Any, name: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 64
        and all(c in HEX for c in value),
        f"{name} must be lowercase sha256 hex",
    )
    return value


def require_git_id(value: Any, name: str) -> str:
    require(
        isinstance(value, str)
        and len(value) == 40
        and all(c in HEX for c in value),
        f"{name} must be lowercase 40-hex git identity",
    )
    return value


def canonical_without_self_anchor(data: dict[str, Any]) -> bytes:
    clone = json.loads(json.dumps(data))
    anchor = clone.get("external_archive_anchor")
    if isinstance(anchor, dict):
        anchor["manifest_sha256"] = None
    return json.dumps(clone, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def verify(path: Path) -> dict[str, Any]:
    data = read_json(path)
    require(isinstance(data, dict), "capsule must be an object")
    require(data.get("schema") == SCHEMA, "unexpected archive schema")
    require(data.get("experiment_id") == EXPERIMENT_ID, "experiment identity mismatch")
    require(data.get("status") == "qualified", "archive capsule is not qualified")
    require(data.get("development_use") == "historical_only_no_tuning", "VART-001 anti-tuning boundary missing")
    require(data.get("claim_ceiling") == "benchmark_family_bounded", "claim ceiling broadened")
    require(data.get("general_creativity_claim_authorized") is False, "general creativity claim must remain unauthorized")
    require(data.get("general_intelligence_claim_authorized") is False, "general intelligence claim must remain unauthorized")
    require(data.get("claim_authorized") is False, "archive capsule itself cannot authorize a new claim")

    subject = data.get("subject_source")
    instrument = data.get("instrument_source")
    require(isinstance(subject, dict) and isinstance(instrument, dict), "source sections missing")
    require_git_id(subject.get("head"), "subject.head")
    require_git_id(subject.get("tree"), "subject.tree")
    require_sha(subject.get("source_closure_receipt_sha256"), "subject.source_closure_receipt_sha256")
    require_git_id(instrument.get("head"), "instrument.head")
    require_git_id(instrument.get("tree"), "instrument.tree")
    require_sha(instrument.get("qualification_receipt_sha256"), "instrument.qualification_receipt_sha256")
    require_sha(instrument.get("source_closure_receipt_sha256"), "instrument.source_closure_receipt_sha256")

    required_shas = [
        "freeze_v3_sha256",
        "campaign_receipt_sha256",
        "first_unblinding_analysis_receipt_sha256",
        "post_unblinding_claim_audit_receipt_sha256",
        "final_bounded_claim_packet_sha256",
        "evidence_closure_sha256",
    ]
    for key in required_shas:
        require_sha(data.get(key), key)

    lineages = data.get("superseded_lineages")
    require(isinstance(lineages, list) and lineages, "superseded lineage history missing")
    seen: set[str] = set()
    for i, item in enumerate(lineages):
        require(isinstance(item, dict), f"superseded_lineages[{i}] must be object")
        current = require_sha(item.get("sha256"), f"superseded_lineages[{i}].sha256")
        nxt = require_sha(item.get("superseded_by_sha256"), f"superseded_lineages[{i}].superseded_by_sha256")
        require(current not in seen, "duplicate superseded lineage")
        require(current != nxt, "lineage cannot supersede itself")
        seen.add(current)

    artifacts = data.get("artifact_bindings")
    require(isinstance(artifacts, list) and artifacts, "artifact bindings missing")
    names: set[str] = set()
    for i, item in enumerate(artifacts):
        require(isinstance(item, dict), f"artifact_bindings[{i}] must be object")
        name = item.get("name")
        require(isinstance(name, str) and name, f"artifact_bindings[{i}].name missing")
        require(name not in names, f"duplicate artifact binding: {name}")
        names.add(name)
        require_sha(item.get("sha256"), f"artifact_bindings[{i}].sha256")

    anchor = data.get("external_archive_anchor")
    require(isinstance(anchor, dict) and anchor.get("required") is True, "external archive anchor required")
    expected = require_sha(anchor.get("manifest_sha256"), "external_archive_anchor.manifest_sha256")
    locator = anchor.get("record_locator")
    require(isinstance(locator, str) and locator, "external archive anchor locator missing")
    reconstructed = hashlib.sha256(canonical_without_self_anchor(data)).hexdigest()
    require(expected == reconstructed, "ARCHIVE_MANIFEST_ANCHOR_MISMATCH")

    return {
        "verdict": "VART_001_ARCHIVE_CAPSULE_QUALIFIED",
        "artifact_count": len(artifacts),
        "manifest_sha256": reconstructed,
        "claim_ceiling": data["claim_ceiling"],
        "development_use": data["development_use"],
        "claim_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the immutable VART-001 archive capsule")
    parser.add_argument("capsule", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    try:
        result = verify(args.capsule.resolve())
    except (Reject, OSError, ValueError) as exc:
        payload = {"verdict": "VART_001_ARCHIVE_CAPSULE_REJECT", "detail": str(exc), "claim_authorized": False}
        if args.json:
            print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        else:
            print(f"REJECT: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        print(result["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
