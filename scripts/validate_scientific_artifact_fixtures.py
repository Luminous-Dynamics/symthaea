#!/usr/bin/env python3
"""Validate byte-backed scientific artifact contract fixtures (stdlib only).

This is a fixture cross-field validator, not a full JSON Schema implementation
or a production parser.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/research/artifact-intake"
SCHEMA_PATH = BASE / "SCIENTIFIC_ARTIFACT_MANIFEST_SCHEMA_V0.json"
FIXTURE_PATH = BASE / "SCIENTIFIC_ARTIFACT_MANIFEST_FIXTURES_V0.json"
HEX_256 = re.compile(r"^[0-9a-f]{64}$")
ALLOWED_DISPOSITIONS = {
    "PARSED", "PARTIAL", "UNSUPPORTED", "MALFORMED",
    "LIMIT_EXCEEDED", "INTEGRITY_FAILED",
}
FAIL_CLOSED = {"UNSUPPORTED", "MALFORMED", "LIMIT_EXCEEDED", "INTEGRITY_FAILED"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_json_pointer(document: Any, pointer: str) -> Any:
    if pointer == "":
        return document
    require(pointer.startswith("/"), f"invalid JSON Pointer: {pointer!r}")
    current = document
    for raw_part in pointer[1:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            require(part.isdigit(), f"non-index JSON Pointer segment: {part!r}")
            index = int(part)
            require(index < len(current), f"JSON Pointer index out of range: {part}")
            current = current[index]
        elif isinstance(current, dict):
            require(part in current, f"JSON Pointer key missing: {part!r}")
            current = current[part]
        else:
            raise ValueError(f"JSON Pointer traverses a scalar at {part!r}")
    return current


def validate_schema_document() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        "manifest schema must declare JSON Schema Draft 2020-12",
    )
    require(schema.get("type") == "object", "manifest schema root must be an object")
    props = schema.get("properties")
    require(isinstance(props, dict), "manifest schema has no properties object")
    for field in ("schema_version", "artifact_id", "acquisition", "media", "raw", "parser", "parse"):
        require(field in props, f"manifest schema is missing root property {field!r}")
    require("observation" in schema.get("$defs", {}), "manifest schema is missing observation definition")
    print("schema document: valid JSON; required manifest contract declarations present")


def validate_fixture(case: dict[str, Any], allowed_mismatches: set[str]) -> None:
    case_id = case["case_id"]
    manifest = case["manifest"]
    parse = manifest["parse"]
    raw = manifest["raw"]
    acquisition = manifest["acquisition"]
    usage = parse["usage"]
    limits = parse["limits"]
    disposition = parse["disposition"]

    require(disposition in ALLOWED_DISPOSITIONS, f"{case_id}: unknown parse disposition")
    artifact_bytes = case["fixture_artifact_bytes_utf8"].encode("utf-8")
    raw_sha = sha256_hex(artifact_bytes)

    require(raw["sha256"] == raw_sha, f"{case_id}: raw SHA-256 does not match fixture bytes")
    require(raw["byte_length"] == len(artifact_bytes), f"{case_id}: raw byte length mismatch")
    require(manifest["artifact_id"] == "sha256:" + raw_sha, f"{case_id}: artifact_id mismatch")
    require(raw["capture_ref"] == "cas://sha256/" + raw_sha, f"{case_id}: capture_ref mismatch")

    receipt_sha = acquisition["receipt_content_sha256"]
    if case_id in allowed_mismatches:
        require(receipt_sha != raw_sha, f"{case_id}: expected an intentional acquisition digest mismatch")
        require(disposition == "INTEGRITY_FAILED", f"{case_id}: digest mismatch must fail integrity")
    else:
        require(receipt_sha == raw_sha, f"{case_id}: acquisition receipt does not bind raw bytes")

    require(usage["bytes_read"] <= limits["max_input_bytes"], f"{case_id}: byte limit exceeded in receipt")
    require(usage["elapsed_ms"] <= limits["max_elapsed_ms"], f"{case_id}: time limit exceeded in receipt")
    require(usage["observations_emitted"] <= limits["max_observations"], f"{case_id}: observation limit exceeded")
    observations = parse["observations"]
    require(usage["observations_emitted"] == len(observations), f"{case_id}: observation count mismatch")
    require(usage["bytes_read"] <= raw["byte_length"], f"{case_id}: bytes_read exceeds captured artifact length")
    require(case.get("expected_disposition") == disposition, f"{case_id}: expected disposition mismatch")

    derived_bytes = case.get("fixture_derived_payload_utf8")
    if disposition in {"PARSED", "PARTIAL"}:
        require(isinstance(derived_bytes, str), f"{case_id}: parsed/partial case needs derived fixture bytes")
        require(parse["derived_payload_sha256"] == sha256_hex(derived_bytes.encode("utf-8")),
                f"{case_id}: derived payload SHA-256 mismatch")
    else:
        require(derived_bytes is None, f"{case_id}: failure fixture must not carry derived payload bytes")
        require(parse["derived_payload_sha256"] is None, f"{case_id}: failure disposition has a derived digest")
        require(observations == [], f"{case_id}: failure disposition emitted observations")

    for observation in observations:
        locator = observation["locator"]
        if locator["kind"] == "CSV_CELL":
            start, end = locator["byte_start"], locator["byte_end"]
            require(0 <= start < end <= len(artifact_bytes), f"{case_id}: CSV cell span out of range")
            exact = artifact_bytes[start:end].decode("utf-8")
            require(exact == observation["raw_lexeme"], f"{case_id}: CSV cell span does not resolve to raw_lexeme")
        elif locator["kind"] == "JSON_POINTER":
            document = json.loads(artifact_bytes.decode("utf-8"))
            target = resolve_json_pointer(document, locator["pointer"])
            require(target is not None, f"{case_id}: JSON Pointer resolves to null")
            lexeme = observation["raw_lexeme"].encode("utf-8")
            require(lexeme in artifact_bytes, f"{case_id}: JSON raw lexeme not present in source bytes")
        else:
            raise ValueError(f"{case_id}: unsupported fixture locator {locator['kind']!r}")

    if disposition == "PARTIAL":
        require(case.get("policy_usable") is False, f"{case_id}: partial output must be diagnostic-only")

    print(f"fixture {case_id}: {disposition} contract checks passed")


def main() -> int:
    try:
        validate_schema_document()
        fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        require(fixtures.get("fixture_class") == "synthetic_contract_with_real_fixture_bytes",
                "fixture class must state that literal bytes are present")
        cases = fixtures.get("cases")
        require(isinstance(cases, list) and len(cases) >= 7, "expected at least seven fixture cases")
        ids = [case["case_id"] for case in cases]
        require(len(ids) == len(set(ids)), "fixture case IDs must be unique")
        mismatch_ids = {case["case_id"] for case in cases if case["case_id"] == "receipt_digest_mismatch"}
        require(len(mismatch_ids) == 1, "expected exactly one named digest-mismatch negative control")
        for case in cases:
            validate_fixture(case, mismatch_ids)
        print(f"artifact fixture contract: {len(cases)} cases passed")
        return 0
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"artifact fixture contract failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
