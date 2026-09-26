#!/usr/bin/env python3
"""Independent stdlib validator for CIV-DOMAIN-GAP-001A source/data artifacts.

This validates architecture/data invariants only. A PASS does not establish
industrial capability, evidence maturity, safety, service sufficiency,
resource allocation, procurement, or physical execution authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

PARENT_SCHEMA = "civ-value-000-critical-domain-matrix-v1"
CHILD_SCHEMA = "civ-domain-gap-001a-owner-coverage-matrix-v1"
PARENT_SHA256 = "67abb09708153f10879d06121213dc1dae0f5de558adae857e670fee51cc5ae6"
CHILD_SHA256 = "45056ab59a16137b48325338a552b109b40eaef263936af96fb52eec1fab892a"
PARENT_HEAD = "acc9db77d7522f7213781020f9a0880815234699"
EXPECTED_DOMAIN_COUNT = 27

ALLOWED_DISPOSITIONS = {
    "ExistingDedicatedOwner",
    "ExistingSharedOwnersSufficient",
    "SharedOwnersNeedDomainProfile",
    "DedicatedDomainRootOpened",
    "DedicatedDomainLikelyMissing",
    "AuditUnresolved",
}
EXPECTED_DISPOSITION_COUNTS = {
    "ExistingDedicatedOwner": 3,
    "ExistingSharedOwnersSufficient": 3,
    "SharedOwnersNeedDomainProfile": 10,
    "DedicatedDomainRootOpened": 7,
    "DedicatedDomainLikelyMissing": 4,
    "AuditUnresolved": 0,
}
REQUIRED_ROOT_CODES = {"PROD", "COMP", "WATER", "AGR", "GRID", "HLTH", "CHEM"}
FORBIDDEN_KEY_FRAGMENTS = (
    "priority_score",
    "importance_score",
    "readiness_score",
    "self_sufficiency_score",
    "resilience_score",
    "bootstrap_score",
    "civilization_score",
)

ROW_FIELDS = {
    "id", "p", "t", "u", "m", "f", "c", "s", "o", "l", "y",
    "d", "q", "r", "g", "e", "x",
}
OWNER_LIST_FIELDS = {"u", "m", "f", "c", "s", "o", "l", "y", "r", "g"}


class ValidationError(Exception):
    pass


def fail(message: str) -> None:
    raise ValidationError(message)


def read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        fail(f"cannot read {path}: {exc}")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_json(raw: bytes, path: Path) -> dict[str, Any]:
    try:
        obj = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"invalid UTF-8/JSON in {path}: {exc}")
    if not isinstance(obj, dict):
        fail(f"{path} root must be a JSON object")
    return obj


def canonical_bytes(obj: dict[str, Any]) -> bytes:
    text = json.dumps(
        obj,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    return text.encode("utf-8")


def assert_exact_hash(raw: bytes, expected: str, label: str) -> None:
    actual = sha256_hex(raw)
    if actual != expected:
        fail(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def assert_canonical(raw: bytes, obj: dict[str, Any], label: str) -> None:
    expected = canonical_bytes(obj)
    if raw != expected:
        fail(f"{label} bytes are not canonical compact sorted-key JSON + final newline")


def recursive_keys(value: Any):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from recursive_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_keys(child)


def validate_parent(parent: dict[str, Any]) -> tuple[list[str], dict[str, str]]:
    if parent.get("schema") != PARENT_SCHEMA:
        fail(f"parent schema must be {PARENT_SCHEMA!r}")

    domains = parent.get("domains")
    if not isinstance(domains, list) or len(domains) != EXPECTED_DOMAIN_COUNT:
        fail(f"parent must contain exactly {EXPECTED_DOMAIN_COUNT} domains")

    ids: list[str] = []
    tags: dict[str, str] = {}
    for index, row in enumerate(domains):
        if not isinstance(row, dict):
            fail(f"parent domains[{index}] must be an object")
        domain_id = row.get("id")
        tag_code = row.get("t")
        if not isinstance(domain_id, str) or not domain_id:
            fail(f"parent domains[{index}] has invalid id")
        if not isinstance(tag_code, str) or not tag_code:
            fail(f"parent domain {domain_id!r} has invalid tag code")
        ids.append(domain_id)
        tags[domain_id] = tag_code

    if len(set(ids)) != len(ids):
        fail("parent contains duplicate domain IDs")

    return ids, tags


def validate_owner_list(row: dict[str, Any], field: str, owner_codes: set[str]) -> None:
    value = row[field]
    if not isinstance(value, list):
        fail(f"{row['id']}: field {field!r} must be a list")
    if any(not isinstance(item, str) or not item for item in value):
        fail(f"{row['id']}: field {field!r} contains invalid owner ref")
    if len(value) != len(set(value)):
        fail(f"{row['id']}: field {field!r} contains duplicate owner refs")
    unknown = sorted(set(value) - owner_codes)
    if unknown:
        fail(f"{row['id']}: field {field!r} contains unknown owner codes {unknown}")


def validate_child(
    child: dict[str, Any],
    parent_ids: list[str],
    parent_tags: dict[str, str],
) -> None:
    if child.get("schema") != CHILD_SCHEMA:
        fail(f"child schema must be {CHILD_SCHEMA!r}")

    parent_ref = child.get("parent")
    if not isinstance(parent_ref, dict):
        fail("child parent binding must be an object")
    expected_parent = {
        "domains": EXPECTED_DOMAIN_COUNT,
        "head": PARENT_HEAD,
        "path": "docs/release/evidence/civ-value-000-critical-domain-matrix-v1.json",
        "pr": 5951,
        "schema": PARENT_SCHEMA,
        "sha256": PARENT_SHA256,
    }
    if parent_ref != expected_parent:
        fail(f"child parent binding mismatch: expected {expected_parent!r}, got {parent_ref!r}")

    codes = child.get("codes")
    if not isinstance(codes, dict):
        fail("child codes must be an object")
    owners = codes.get("owners")
    if not isinstance(owners, dict) or not owners:
        fail("child owner codebook must be a non-empty object")
    owner_codes = set(owners)
    if any(not isinstance(k, str) or not isinstance(v, str) or not v for k, v in owners.items()):
        fail("child owner codebook contains invalid entries")

    dispositions = codes.get("dispositions")
    if not isinstance(dispositions, list) or set(dispositions) != ALLOWED_DISPOSITIONS:
        fail("child disposition codebook does not match frozen vocabulary")
    if len(dispositions) != len(set(dispositions)):
        fail("child disposition codebook contains duplicates")

    row_fields = codes.get("row_fields")
    if not isinstance(row_fields, dict) or set(row_fields) != ROW_FIELDS:
        fail("child row-field codebook does not match frozen row shape")

    tags = codes.get("tags")
    if tags != {
        "H": "human_essential",
        "M": "shared_industrial_multiplier",
        "S": "symthaea_continuity",
    }:
        fail("child tag codebook mismatch")

    maturity = codes.get("evidence_maturity")
    if not isinstance(maturity, dict) or "AuditUnresolved" not in maturity:
        fail("child must define the conservative AuditUnresolved evidence-maturity state")

    claim_ceiling = codes.get("claim_ceiling")
    if not isinstance(claim_ceiling, dict) or "coverage_only" not in claim_ceiling:
        fail("child must define the coverage_only claim ceiling")

    domains = child.get("domains")
    if not isinstance(domains, list) or len(domains) != EXPECTED_DOMAIN_COUNT:
        fail(f"child must contain exactly {EXPECTED_DOMAIN_COUNT} domains")

    child_ids: list[str] = []
    disposition_counts: Counter[str] = Counter()
    dedicated_root_codes: set[str] = set()

    for index, row in enumerate(domains):
        if not isinstance(row, dict):
            fail(f"child domains[{index}] must be an object")
        if set(row) != ROW_FIELDS:
            fail(
                f"child domains[{index}] field set mismatch: "
                f"missing={sorted(ROW_FIELDS-set(row))}, extra={sorted(set(row)-ROW_FIELDS)}"
            )

        domain_id = row["id"]
        if not isinstance(domain_id, str) or not domain_id:
            fail(f"child domains[{index}] has invalid id")
        child_ids.append(domain_id)

        expected_parent_ref = f"{PARENT_SCHEMA}:{domain_id}"
        if row["p"] != expected_parent_ref:
            fail(f"{domain_id}: parent-domain ref mismatch")
        if domain_id not in parent_tags:
            fail(f"{domain_id}: domain ID absent from parent")
        if row["t"] != parent_tags[domain_id]:
            fail(
                f"{domain_id}: critical-to tag mismatch: "
                f"parent={parent_tags[domain_id]!r}, child={row['t']!r}"
            )

        disposition = row["d"]
        if disposition not in ALLOWED_DISPOSITIONS:
            fail(f"{domain_id}: unknown coverage disposition {disposition!r}")
        disposition_counts[disposition] += 1

        if row["e"] != "AuditUnresolved":
            fail(f"{domain_id}: source audit may not promote evidence maturity beyond AuditUnresolved")
        if row["x"] != "coverage_only":
            fail(f"{domain_id}: claim ceiling must remain coverage_only")

        if not isinstance(row["q"], list) or any(not isinstance(x, str) or not x for x in row["q"]):
            fail(f"{domain_id}: missing_theorems must be a list of non-empty strings")

        for field in OWNER_LIST_FIELDS:
            validate_owner_list(row, field, owner_codes)

        if disposition in {"ExistingDedicatedOwner", "DedicatedDomainRootOpened"} and not row["r"]:
            fail(f"{domain_id}: {disposition} requires a dedicated-domain root ref")
        if disposition == "DedicatedDomainLikelyMissing" and row["r"]:
            fail(f"{domain_id}: DedicatedDomainLikelyMissing must not claim a dedicated root")

        dedicated_root_codes.update(row["r"])

    if child_ids != parent_ids:
        fail("child domain ID sequence must exactly match the frozen parent domain sequence")
    if len(set(child_ids)) != len(child_ids):
        fail("child contains duplicate domain IDs")

    if dict(disposition_counts) != {k: v for k, v in EXPECTED_DISPOSITION_COUNTS.items() if v}:
        normalized = {k: disposition_counts.get(k, 0) for k in EXPECTED_DISPOSITION_COUNTS}
        if normalized != EXPECTED_DISPOSITION_COUNTS:
            fail(
                f"coverage disposition census mismatch: "
                f"expected {EXPECTED_DISPOSITION_COUNTS}, got {normalized}"
            )

    if not REQUIRED_ROOT_CODES.issubset(dedicated_root_codes):
        missing_roots = sorted(REQUIRED_ROOT_CODES - dedicated_root_codes)
        fail(f"new dedicated-root refs missing from matrix: {missing_roots}")

    all_keys = [key.lower() for key in recursive_keys(child)]
    for forbidden in FORBIDDEN_KEY_FRAGMENTS:
        if forbidden in all_keys:
            fail(f"forbidden score/authority-laundering key present: {forbidden}")

    authority = child.get("authority")
    if authority != "analysis_only_no_priority_allocation_procurement_or_execution_authority":
        fail("child authority ceiling mismatch")

    nonclaims = child.get("nonclaims")
    if not isinstance(nonclaims, list) or not nonclaims:
        fail("child must retain explicit nonclaims")

    rules = child.get("rules")
    if not isinstance(rules, list) or not rules:
        fail("child must retain explicit ownership/anti-duplication rules")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parent",
        type=Path,
        default=Path("docs/release/evidence/civ-value-000-critical-domain-matrix-v1.json"),
    )
    parser.add_argument(
        "--child",
        type=Path,
        default=Path("docs/release/evidence/civ-domain-gap-001a-owner-coverage-matrix-v1.json"),
    )
    args = parser.parse_args()

    try:
        parent_raw = read_bytes(args.parent)
        child_raw = read_bytes(args.child)

        assert_exact_hash(parent_raw, PARENT_SHA256, "parent matrix")
        assert_exact_hash(child_raw, CHILD_SHA256, "owner matrix")

        parent = parse_json(parent_raw, args.parent)
        child = parse_json(child_raw, args.child)

        assert_canonical(parent_raw, parent, "parent matrix")
        assert_canonical(child_raw, child, "owner matrix")

        parent_ids, parent_tags = validate_parent(parent)
        validate_child(child, parent_ids, parent_tags)
    except ValidationError as exc:
        print(f"FAIL_CIV_DOMAIN_GAP_001A_SOURCE_VALIDATION: {exc}")
        return 1

    print("PASS_CIV_DOMAIN_GAP_001A_SOURCE_VALIDATION")
    print(f"parent_sha256={PARENT_SHA256}")
    print(f"owner_matrix_sha256={CHILD_SHA256}")
    print(f"domain_count={EXPECTED_DOMAIN_COUNT}")
    print("claim_ceiling=source_data_architecture_only")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
