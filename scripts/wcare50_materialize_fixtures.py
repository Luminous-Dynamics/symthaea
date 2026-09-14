#!/usr/bin/env python3
"""Materialize the frozen WCARE-50 independent Ed25519 fixture package.

The package bytes were produced independently of the Rust verifier with Python
cryptography's Ed25519 implementation. This helper does not sign or regenerate
fixtures; it decodes the exact frozen package and verifies its commitments.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import zlib

ARCHIVE_SHA256 = "81ea12f18312bcd8542cdc8e3066fe85d1f32f329c06e06e6870ac2f304820d9"
PACKAGE_SHA256 = "c6b66a25b5a04649306d81a4275019885edddd84d90bd4510a522441c3a8a9d9"
PACKAGE_VERSION = "wcare50-wcare42-executable-qualification-fixtures-v1"
PUBLIC_KEY = "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"
EXPECTED_CASES = (
    "accepted_provenance",
    "accepted_relation",
    "accepted_pre_result",
    "signature_valid_issuer_untrusted",
    "invalid_signature",
    "altered_subject_replay",
    "altered_policy_binding",
    "subject_scope_unauthorized",
    "unauthorized_strength",
    "expired_attestation",
    "revoked_issuer",
)
EXPECTED_FILES = {"envelope.json", "policy.json", "plan.json", "result.json", "subject.json"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_package(archive_path: Path) -> tuple[dict, bytes]:
    encoded = b"".join(archive_path.read_bytes().split())
    compressed = base64.b64decode(encoded, validate=True)
    if sha256(compressed) != ARCHIVE_SHA256:
        raise SystemExit("fixture_archive_sha256_mismatch")
    package_bytes = zlib.decompress(compressed)
    if sha256(package_bytes) != PACKAGE_SHA256:
        raise SystemExit("fixture_package_sha256_mismatch")
    package = json.loads(package_bytes.decode("utf-8"))
    if not isinstance(package, dict):
        raise SystemExit("fixture_package_not_object")
    return package, package_bytes


def validate(package: dict) -> None:
    if package.get("package_version") != PACKAGE_VERSION:
        raise SystemExit("fixture_package_version_mismatch")
    if package.get("authority") != "MeasurementOnly":
        raise SystemExit("fixture_authority_mismatch")
    if package.get("producer") != "python-cryptography-ed25519":
        raise SystemExit("fixture_producer_mismatch")
    if package.get("synthetic_only") is not True:
        raise SystemExit("fixture_synthetic_boundary_missing")
    if package.get("issuer_public_key_ed25519_hex") != PUBLIC_KEY:
        raise SystemExit("fixture_public_key_mismatch")
    if package.get("wcare42_verifier_source_blob") != "1c300a455f054d118e55556aac81b623824629bc":
        raise SystemExit("fixture_verifier_blob_mismatch")
    if package.get("wcare42_qualifier_blob") != "8a80d1a74e409503a72b4607425cc61d8072ca2e":
        raise SystemExit("fixture_qualifier_blob_mismatch")

    cases = package.get("cases")
    if not isinstance(cases, list):
        raise SystemExit("fixture_cases_missing")
    names = [case.get("name") for case in cases if isinstance(case, dict)]
    if names != list(EXPECTED_CASES):
        raise SystemExit(f"fixture_case_census_mismatch:{names!r}")

    for case in cases:
        if set(case) != {
            "name",
            "evaluation_utc",
            "expected_disposition",
            "expected_exit_code",
            "expected_result_bound_by_signature",
            "expected_checks",
            "files",
            "file_sha256",
        }:
            raise SystemExit(f"fixture_case_fields_mismatch:{case.get('name')}")
        files = case["files"]
        digests = case["file_sha256"]
        if set(files) != EXPECTED_FILES or set(digests) != EXPECTED_FILES:
            raise SystemExit(f"fixture_file_census_mismatch:{case['name']}")
        for name in sorted(EXPECTED_FILES):
            raw = files[name]
            if not isinstance(raw, str) or not raw.endswith("\n"):
                raise SystemExit(f"fixture_file_not_canonical_text:{case['name']}:{name}")
            if sha256(raw.encode("utf-8")) != digests[name]:
                raise SystemExit(f"fixture_file_digest_mismatch:{case['name']}:{name}")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise SystemExit(f"fixture_file_not_json_object:{case['name']}:{name}")
        checks = case["expected_checks"]
        expected_check_keys = {
            "signature_valid",
            "subject_binding_valid",
            "canonicalization_valid",
            "issuer_key_present",
            "key_valid_at_issue_time",
            "revocation_policy_satisfied",
            "attestation_current_at_evaluation",
            "scope_authorized",
            "claimed_strength_authorized",
            "issuer_trusted_for_claim",
        }
        if set(checks) != expected_check_keys or not all(isinstance(v, bool) for v in checks.values()):
            raise SystemExit(f"fixture_expected_checks_invalid:{case['name']}")
        if case["expected_disposition"] not in {
            "ATTESTATION_ACCEPTED",
            "SIGNATURE_VALID_ISSUER_UNTRUSTED",
            "ATTESTATION_REJECTED",
        }:
            raise SystemExit(f"fixture_disposition_invalid:{case['name']}")
        if case["expected_exit_code"] not in {0, 2, 4}:
            raise SystemExit(f"fixture_exit_code_invalid:{case['name']}")


def materialize(package: dict, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=False)
    for case in package["cases"]:
        case_dir = destination / case["name"]
        case_dir.mkdir()
        for name, content in case["files"].items():
            (case_dir / name).write_bytes(content.encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--package-json", type=Path)
    args = parser.parse_args()

    package, package_bytes = load_package(args.archive)
    validate(package)
    if args.package_json is not None:
        args.package_json.write_bytes(package_bytes)
    if args.out is not None:
        materialize(package, args.out)
    print(f"WCARE50_FIXTURE_PACKAGE_SHA256={PACKAGE_SHA256}")
    print(f"WCARE50_FIXTURE_ARCHIVE_SHA256={ARCHIVE_SHA256}")
    print("PASS_WCARE50_FROZEN_FIXTURE_PACKAGE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
