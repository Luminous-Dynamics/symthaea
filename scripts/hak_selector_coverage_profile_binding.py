#!/usr/bin/env python3
"""Bind a HAK-016 coverage record to the exact HAK-015 profile artifact.

Audit/evidence tooling only. This closes ProfileName != ProfileContent for
portable interpretation of selector coverage. It does not authenticate a
provider, establish semantic truth, or grant runtime authority.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import hak_canonical_json as canonical
import hak_selector_coverage as coverage

SCHEMA_VERSION = "hak.selector-coverage-profile-binding.v1"
DIGEST_DOMAIN = SCHEMA_VERSION
PROFILE_ID = "hak.canonical-json.v1"
PROFILE_ARTIFACT_REF = (
    "git:Luminous-Dynamics/symthaea@cf440ce4f813bb30a6b1948e5caac96feda10610:"
    "docs/architecture/hak/canonical-json-v1.profile.json"
)
PROFILE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs/architecture/hak/canonical-json-v1.profile.json"
)
EXPECTED_PROFILE_RAW_SHA256 = (
    "sha256:f76152db3cce567bb4d24ca46b5ed94d95db698daa8882a6fcdb38828d40e830"
)
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


class SelectorCoverageProfileBindingError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectorCoverageProfileBindingError(message)


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def current_profile_bytes() -> bytes:
    return PROFILE_PATH.read_bytes()


def validate_profile_bytes(raw: bytes) -> None:
    _require(sha256_bytes(raw) == EXPECTED_PROFILE_RAW_SHA256,
             "inherited HAK-015 profile bytes do not match the bound raw SHA-256")
    try:
        profile = canonical.parse_strict_json(raw)
    except Exception as exc:
        raise SelectorCoverageProfileBindingError(
            f"bound HAK-015 profile bytes are not valid hak.canonical-json.v1 input: {exc}"
        ) from exc
    _require(isinstance(profile, dict) and profile.get("profile_id") == PROFILE_ID,
             "bound HAK-015 profile content does not declare the expected profile_id")


def _validate_coverage_digest(record: dict[str, Any]) -> None:
    _require(isinstance(record, dict), "coverage record must be object")
    _require(record.get("schema_version") == coverage.SCHEMA_VERSION,
             "coverage schema_version mismatch")
    _require(record.get("canonicalization_profile") == PROFILE_ID,
             "coverage record does not name the bound HAK-015 profile")
    digest = record.get("coverage_digest")
    _require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
             "coverage_digest invalid")
    _require(digest == coverage.compute_coverage_digest(record),
             "coverage_digest does not match coverage record content")


def compute_binding_digest(binding: dict[str, Any]) -> str:
    payload = {key: value for key, value in binding.items() if key != "binding_digest"}
    return canonical.hak_sha256(DIGEST_DOMAIN, payload)


def derive_profile_binding(record: dict[str, Any]) -> dict[str, Any]:
    _validate_coverage_digest(record)
    validate_profile_bytes(current_profile_bytes())
    binding: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "coverage": {
            "schema_version": coverage.SCHEMA_VERSION,
            "coverage_digest": record["coverage_digest"],
        },
        "canonicalization_profile": {
            "profile_id": PROFILE_ID,
            "artifact_ref": PROFILE_ARTIFACT_REF,
            "raw_sha256": EXPECTED_PROFILE_RAW_SHA256,
        },
        "binding_semantics": {
            "profile_name_is_not_profile_identity": True,
            "raw_digest_identifies_exact_profile_bytes": True,
            "artifact_ref_identifies_profile_provenance": True,
        },
    }
    binding["binding_digest"] = compute_binding_digest(binding)
    return binding


def validate_profile_binding(
    binding: dict[str, Any],
    record: dict[str, Any],
) -> None:
    _validate_coverage_digest(record)
    validate_profile_bytes(current_profile_bytes())
    _require(isinstance(binding, dict), "profile binding must be object")
    _require(binding.get("schema_version") == SCHEMA_VERSION,
             "profile binding schema_version invalid")
    expected = derive_profile_binding(record)
    _require(binding == expected,
             "profile binding is not the deterministic HAK-016 binding for this coverage record/profile artifact")
    digest = binding.get("binding_digest")
    _require(isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
             "binding_digest invalid")
    _require(digest == compute_binding_digest(binding), "binding_digest mismatch")
