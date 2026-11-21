"""Test-only validator for split-local semiconductor contrastive-pair manifests."""
from __future__ import annotations
from datetime import datetime
import re
from typing import Any

INPUT_SCHEMA = "luminous.semi-bench.contrastive-pair-manifest.v1"
OUTPUT_SCHEMA = "luminous.semi-bench.contrastive-pair-validation.v1"
SPLITS = {"train", "validation", "test"}
BASIS = {"spatial", "density", "structure", "production_proximity"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PairManifestError(ValueError):
    """Pair identity, split, or provenance contract violation."""


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PairManifestError(f"{field}: expected non-empty string")
    return value


def _time(value: Any, field: str, basis: str) -> datetime:
    raw = _text(value, field)
    try:
        value = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PairManifestError(f"{field}: invalid ISO-8601 timestamp") from exc
    offset = value.tzinfo is not None and value.utcoffset() is not None
    if basis == "offset_aware" and not offset:
        raise PairManifestError(f"{field}: timezone offset required")
    if basis == "source_local_naive" and offset:
        raise PairManifestError(f"{field}: timezone offset forbidden")
    return value


def validate_pair_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, dict) or manifest.get("schema") != INPUT_SCHEMA:
        raise PairManifestError("schema: unsupported or missing")
    dataset_id = _text(manifest.get("dataset_id"), "dataset_id")
    split_digest = _text(manifest.get("split_manifest_sha256"), "split_manifest_sha256")
    if not SHA256_RE.fullmatch(split_digest):
        raise PairManifestError("split_manifest_sha256: invalid SHA-256")
    time_basis = manifest.get("time_basis")
    if time_basis not in ("offset_aware", "source_local_naive"):
        raise PairManifestError("time_basis: invalid")
    clock_domain = None
    if time_basis == "source_local_naive":
        clock_domain = _text(manifest.get("clock_domain_id"), "clock_domain_id")
    embargo = manifest.get("time_embargo_minutes")
    if isinstance(embargo, bool) or not isinstance(embargo, (int, float)) or embargo < 0:
        raise PairManifestError("time_embargo_minutes: expected non-negative number")

    assignments: dict[str, dict[str, Any]] = {}
    raw_assignments = manifest.get("sample_assignments")
    if not isinstance(raw_assignments, list) or not raw_assignments:
        raise PairManifestError("sample_assignments: expected non-empty array")
    for i, row in enumerate(raw_assignments):
        if not isinstance(row, dict):
            raise PairManifestError(f"sample_assignments[{i}]: expected object")
        sid = _text(row.get("sample_id"), f"sample_assignments[{i}].sample_id")
        if sid in assignments:
            raise PairManifestError(f"sample_assignments[{i}]: duplicate sample identity")
        split = row.get("split")
        if split not in SPLITS:
            raise PairManifestError(f"sample_assignments[{i}]: invalid split")
        record: dict[str, Any] = {"split": split}
        if row.get("group_id") is not None:
            record["group_id"] = _text(row["group_id"], f"sample_assignments[{i}].group_id")
        if row.get("observed_at") is not None:
            record["observed_at"] = _time(row["observed_at"], f"sample_assignments[{i}].observed_at", time_basis)
        assignments[sid] = record

    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise PairManifestError("pairs: expected non-empty array")
    pair_ids: set[str] = set()
    edges: set[frozenset[str]] = set()
    split_counts = {s: 0 for s in sorted(SPLITS)}
    relations = {"positive": 0, "negative": 0}
    bases = {b: 0 for b in sorted(BASIS)}
    for i, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise PairManifestError(f"pairs[{i}]: expected object")
        pid = _text(pair.get("pair_id"), f"pairs[{i}].pair_id")
        if pid in pair_ids:
            raise PairManifestError(f"pairs[{i}]: duplicate pair identity")
        pair_ids.add(pid)
        left_id = _text(pair.get("left_sample_id"), f"pairs[{i}].left_sample_id")
        right_id = _text(pair.get("right_sample_id"), f"pairs[{i}].right_sample_id")
        if left_id == right_id:
            raise PairManifestError(f"pairs[{i}]: self-pair")
        if left_id not in assignments or right_id not in assignments:
            raise PairManifestError(f"pairs[{i}]: orphan sample reference")
        edge = frozenset((left_id, right_id))
        if edge in edges:
            raise PairManifestError(f"pairs[{i}]: duplicate unordered pair")
        edges.add(edge)
        scope = pair.get("pair_scope")
        if scope not in SPLITS:
            raise PairManifestError(f"pairs[{i}]: invalid pair_scope")
        left, right = assignments[left_id], assignments[right_id]
        if left["split"] != right["split"] or left["split"] != scope:
            raise PairManifestError(f"pairs[{i}]: cross-split pair or pair_scope mismatch")
        relation = pair.get("relation")
        if relation not in relations:
            raise PairManifestError(f"pairs[{i}]: invalid relation")
        basis = pair.get("basis")
        if not isinstance(basis, list) or not basis or len(set(basis)) != len(basis) or any(x not in BASIS for x in basis):
            raise PairManifestError(f"pairs[{i}]: invalid basis")
        reason = pair.get("proximity_reason")
        if "production_proximity" in basis:
            if reason == "same_lot":
                if not left.get("group_id") or left.get("group_id") != right.get("group_id"):
                    raise PairManifestError(f"pairs[{i}]: same_lot requires matching group identity")
            elif reason == "time_near":
                if "observed_at" not in left or "observed_at" not in right:
                    raise PairManifestError(f"pairs[{i}]: time_near lacks timestamps")
                delta = abs((left["observed_at"] - right["observed_at"]).total_seconds()) / 60
                if delta > float(embargo):
                    raise PairManifestError(f"pairs[{i}]: time_near exceeds embargo")
            else:
                raise PairManifestError(f"pairs[{i}]: proximity reason must be same_lot or time_near")
        elif reason is not None:
            raise PairManifestError(f"pairs[{i}]: proximity_reason without production_proximity")
        split_counts[scope] += 1
        relations[relation] += 1
        for item in basis:
            bases[item] += 1

    return {
        "schema": OUTPUT_SCHEMA,
        "dataset_id": dataset_id,
        "split_manifest_sha256": split_digest,
        "time_basis": time_basis,
        "clock_domain_id": clock_domain,
        "time_embargo_minutes": float(embargo),
        "sample_count": len(assignments),
        "pair_count": len(pairs),
        "pair_counts_by_scope": split_counts,
        "relation_counts": relations,
        "basis_counts": bases,
        "guarantees": {
            "unique_sample_identity": True,
            "unique_pair_identity": True,
            "unique_unordered_pair": True,
            "no_self_pairs": True,
            "all_pair_endpoints_in_declared_scope": True,
            "production_proximity_has_supporting_lot_or_time_evidence": True,
        },
        "claim_limits": {
            "pair_manifest_integrity_only": True,
            "does_not_prove_pair_semantic_correctness": True,
            "does_not_prove_embedding_quality_or_model_performance": True,
            "does_not_establish_causal_production_relationship": True,
        },
    }


def main() -> int:
    import argparse
    import json
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="contrastive-pair manifest JSON")
    parser.add_argument("--output", type=Path, help="write validation report JSON")
    args = parser.parse_args()
    try:
        subject = json.loads(args.input.read_text(encoding="utf-8"))
        result = validate_pair_manifest(subject)
    except (OSError, json.JSONDecodeError, PairManifestError) as exc:
        print(f"SEMI-BENCH contrastive-pair validator REJECT: {exc}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
