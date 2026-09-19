#!/usr/bin/env python3
"""Validate an auditable mathematical retrieval candidate-set artifact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

VERSION = "math-retrieval-candidate-set-v1"
AUTHORITY = "MeasurementOnly"
ROOT = {
    "version",
    "candidate_set_id",
    "authority",
    "corpus_snapshot_sha256",
    "knowledge_boundary_sha256",
    "candidate_eligibility_policy_sha256",
    "source_identity_kind",
    "canonical_order",
    "candidate_count",
    "candidates",
}


class ValidationError(ValueError):
    pass


def sha(value: object, where: str) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise ValidationError(f"{where}: sha256:<64 lowercase hex> required")
    if any(ch not in "0123456789abcdef" for ch in value[7:]):
        raise ValidationError(f"{where}: invalid SHA-256")
    return value


def validate(doc: object) -> None:
    if not isinstance(doc, dict) or set(doc) != ROOT:
        raise ValidationError("root: exact fields required")
    if doc["version"] != VERSION or doc["authority"] != AUTHORITY:
        raise ValidationError("root: version/authority invariant failed")
    if not isinstance(doc["candidate_set_id"], str) or not doc["candidate_set_id"].strip():
        raise ValidationError("candidate_set_id: non-empty string required")
    for field in (
        "corpus_snapshot_sha256",
        "knowledge_boundary_sha256",
        "candidate_eligibility_policy_sha256",
    ):
        sha(doc[field], field)
    if doc["source_identity_kind"] != "SourceObjectDigest":
        raise ValidationError("source_identity_kind: SourceObjectDigest required")
    if doc["canonical_order"] != "SourceObjectDigestAscending":
        raise ValidationError("canonical_order: SourceObjectDigestAscending required")
    count = doc["candidate_count"]
    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise ValidationError("candidate_count: positive integer required")
    candidates = doc["candidates"]
    if not isinstance(candidates, list) or not candidates:
        raise ValidationError("candidates: non-empty list required")
    checked = [sha(value, f"candidates[{i}]") for i, value in enumerate(candidates)]
    if len(checked) != count:
        raise ValidationError("candidate_count does not equal len(candidates)")
    if len(set(checked)) != len(checked):
        raise ValidationError("candidates: duplicates are forbidden")
    if checked != sorted(checked):
        raise ValidationError("candidates: SourceObjectDigestAscending order required")


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def fixture() -> dict:
    d = lambda n: f"sha256:{n:064x}"
    return {
        "version": VERSION,
        "candidate_set_id": "fixture",
        "authority": AUTHORITY,
        "corpus_snapshot_sha256": d(100),
        "knowledge_boundary_sha256": d(101),
        "candidate_eligibility_policy_sha256": d(102),
        "source_identity_kind": "SourceObjectDigest",
        "canonical_order": "SourceObjectDigestAscending",
        "candidate_count": 3,
        "candidates": [d(1), d(2), d(3)],
    }


def self_test() -> None:
    valid = fixture()
    validate(valid)
    attacks = [
        lambda d: d["candidates"].reverse(),
        lambda d: d["candidates"].__setitem__(1, d["candidates"][0]),
        lambda d: d.__setitem__("candidate_count", 2),
        lambda d: d.__setitem__("source_identity_kind", "PayloadDigest"),
        lambda d: d.__setitem__("canonical_order", "InsertionOrder"),
    ]
    for mutate in attacks:
        candidate = copy.deepcopy(valid)
        mutate(candidate)
        try:
            validate(candidate)
        except ValidationError:
            continue
        raise AssertionError("candidate-set adversarial self-test unexpectedly passed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("math-retrieval candidate-set v1 self-test: PASS")
        return 0
    if args.path is None:
        parser.error("path required unless --self-test")
    try:
        raw = args.path.read_bytes()
        doc = json.loads(raw.decode("utf-8"))
        validate(doc)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValidationError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({
        "version": "math-retrieval-candidate-set-validation-report-v1",
        "authority": AUTHORITY,
        "candidate_set_sha256": digest_bytes(raw),
        "candidate_count": doc["candidate_count"],
        "all_checks_passed": True,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
