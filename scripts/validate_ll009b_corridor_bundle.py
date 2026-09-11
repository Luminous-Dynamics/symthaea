#!/usr/bin/env python3
"""Validate/hash-bind an LL-009B corridor evidence bundle.

This tool composes evidence receipts; it does not validate the underlying
physics. Each producing subsystem remains responsible for its own semantic
validation. LL-009B prevents a downstream trade study from silently mixing
artifacts from different study, frame, epoch, or synthetic/real-data lineages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

BUNDLE_SCHEMA = "ll009b.corridor-bundle.v1"


class BundleError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BundleError(f"expected JSON object: {path}")
    return value


def nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def load_schema(path: Path) -> dict[str, Any]:
    schema = load_json(path)
    if schema.get("schema_version") != "ll009b.corridor-bundle-schema.v1":
        raise BundleError("unsupported LL-009B schema")
    roles = schema.get("required_roles_for_trade_study")
    classes = schema.get("allowed_evidence_classes")
    if not isinstance(roles, list) or not roles or len(set(roles)) != len(roles):
        raise BundleError("schema required roles must be unique/non-empty")
    if not isinstance(classes, list) or not classes:
        raise BundleError("schema allowed evidence classes must be non-empty")
    return schema


def validate_artifact(entry: dict[str, Any], allowed_classes: set[str]) -> None:
    for field in (
        "role",
        "path",
        "sha256",
        "evidence_class",
        "lineage_id",
        "study_id",
        "frame_contract_id",
        "epoch_contract_id",
    ):
        if not nonempty(entry.get(field)):
            raise BundleError(f"artifact missing {field}")
    if entry["evidence_class"] not in allowed_classes:
        raise BundleError(f"artifact {entry['role']}: invalid evidence_class")
    digest = entry["sha256"].lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise BundleError(f"artifact {entry['role']}: sha256 must be 64 lowercase hex chars")
    if not isinstance(entry.get("synthetic_only", False), bool):
        raise BundleError(f"artifact {entry['role']}: synthetic_only must be bool")


def validate_bundle(
    bundle_path: Path,
    schema_path: Path,
    artifact_root: Path | None,
) -> dict[str, Any]:
    schema = load_schema(schema_path)
    bundle = load_json(bundle_path)
    if bundle.get("schema_version") != BUNDLE_SCHEMA:
        raise BundleError(f"bundle schema_version must be {BUNDLE_SCHEMA}")
    for field in ("bundle_id", "study_id", "frame_contract_id", "epoch_contract_id", "requested_promotion"):
        if not nonempty(bundle.get(field)):
            raise BundleError(f"bundle missing {field}")

    artifacts = bundle.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise BundleError("bundle artifacts must be non-empty")

    allowed_classes = set(schema["allowed_evidence_classes"])
    by_role: dict[str, dict[str, Any]] = {}
    verified: list[dict[str, Any]] = []
    for entry in artifacts:
        if not isinstance(entry, dict):
            raise BundleError("artifact entries must be objects")
        validate_artifact(entry, allowed_classes)
        role = entry["role"]
        if role in by_role:
            raise BundleError(f"duplicate artifact role: {role}")
        by_role[role] = entry
        if entry["study_id"] != bundle["study_id"]:
            raise BundleError(f"artifact {role}: study_id mismatch")
        if entry["frame_contract_id"] != bundle["frame_contract_id"]:
            raise BundleError(f"artifact {role}: frame contract mismatch")
        if entry["epoch_contract_id"] != bundle["epoch_contract_id"]:
            raise BundleError(f"artifact {role}: epoch contract mismatch")

        record = {
            "role": role,
            "lineage_id": entry["lineage_id"],
            "declared_sha256": entry["sha256"].lower(),
            "evidence_class": entry["evidence_class"],
            "synthetic_only": bool(entry.get("synthetic_only", False)),
        }
        if artifact_root is not None:
            path = artifact_root / entry["path"]
            if not path.is_file():
                raise BundleError(f"artifact {role}: missing file {path}")
            actual = sha256_file(path)
            if actual != entry["sha256"].lower():
                raise BundleError(
                    f"artifact {role}: sha256 mismatch expected={entry['sha256']} actual={actual}"
                )
            record["verified_path"] = str(path)
            record["bytes"] = path.stat().st_size
        verified.append(record)

    promotion = bundle["requested_promotion"]
    if promotion == "trade_study_ready_for_real_site_comparison":
        required = set(schema["required_roles_for_trade_study"])
        missing = sorted(required.difference(by_role))
        if missing:
            raise BundleError(f"trade-study promotion missing roles: {', '.join(missing)}")
        forbidden = set(
            schema.get("promotion_rules", {})
            .get("trade_study_ready_for_real_site_comparison", {})
            .get("forbid_evidence_classes", [])
        )
        for role in sorted(required):
            entry = by_role[role]
            if entry["evidence_class"] in forbidden or entry.get("synthetic_only", False):
                raise BundleError(f"trade-study promotion forbids synthetic-only artifact: {role}")
    elif promotion != "research_bundle":
        raise BundleError(f"unknown requested_promotion: {promotion}")

    return {
        "schema_version": "ll009b.corridor-bundle-receipt.v1",
        "status": "pass",
        "bundle_id": bundle["bundle_id"],
        "study_id": bundle["study_id"],
        "requested_promotion": promotion,
        "frame_contract_id": bundle["frame_contract_id"],
        "epoch_contract_id": bundle["epoch_contract_id"],
        "bundle_manifest_sha256": sha256_file(bundle_path),
        "schema_sha256": sha256_file(schema_path),
        "artifacts": verified,
        "non_claim": "Bundle closure is provenance/referential evidence, not operational launch authorization.",
    }


def self_test() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        schema_path = root / "schema.json"
        schema_path.write_text(json.dumps({
            "schema_version": "ll009b.corridor-bundle-schema.v1",
            "required_roles_for_trade_study": ["terrain_pack_receipt", "frame_bridge_receipt"],
            "allowed_evidence_classes": ["synthetic", "derived_real_data"],
            "promotion_rules": {
                "trade_study_ready_for_real_site_comparison": {
                    "forbid_evidence_classes": ["synthetic"]
                }
            },
        }), encoding="utf-8")
        data = root / "terrain.json"
        data.write_text('{"ok":true}\n', encoding="utf-8")
        digest = sha256_file(data)
        base = {
            "role": "terrain_pack_receipt",
            "path": "terrain.json",
            "sha256": digest,
            "evidence_class": "derived_real_data",
            "lineage_id": "terrain-v1",
            "study_id": "study-1",
            "frame_contract_id": "frame-1",
            "epoch_contract_id": "epoch-1",
            "synthetic_only": False,
        }
        bundle_path = root / "bundle.json"
        bundle_path.write_text(json.dumps({
            "schema_version": BUNDLE_SCHEMA,
            "bundle_id": "bundle-1",
            "study_id": "study-1",
            "frame_contract_id": "frame-1",
            "epoch_contract_id": "epoch-1",
            "requested_promotion": "research_bundle",
            "artifacts": [base],
        }), encoding="utf-8")
        receipt = validate_bundle(bundle_path, schema_path, root)
        assert receipt["status"] == "pass"

        bad = dict(base)
        bad["study_id"] = "other-study"
        bundle_path.write_text(json.dumps({
            "schema_version": BUNDLE_SCHEMA,
            "bundle_id": "bundle-2",
            "study_id": "study-1",
            "frame_contract_id": "frame-1",
            "epoch_contract_id": "epoch-1",
            "requested_promotion": "research_bundle",
            "artifacts": [bad],
        }), encoding="utf-8")
        try:
            validate_bundle(bundle_path, schema_path, root)
        except BundleError:
            pass
        else:
            raise AssertionError("mixed study lineage must fail")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("configs/lunar_transport/ll009b_corridor_bundle_schema.json"),
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        self_test()
        print("LL-009B bundle self-test: PASS")
        return 0
    if args.bundle is None:
        raise BundleError("--bundle is required unless --self-test is used")
    receipt = validate_bundle(args.bundle, args.schema, args.artifact_root)
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BundleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
