#!/usr/bin/env python3
"""Independent OpenSSL-backed verifier for the synthetic DSSE evidence envelope."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

PAYLOAD_TYPE = "application/vnd.luminous.connectome-evidence+json;version=1"
SPKI_ED25519_PREFIX = bytes.fromhex("302a300506032b6570032100")
SCOPE_NOTE = (
    "This receipt establishes that the listed trusted keys signed the exact DSSE payload bytes "
    "under the pinned trust policy at the declared verification time. It does not establish signer "
    "honesty, observation accuracy, protocol validity, sample representativeness, evidence "
    "independence or biological truth. The trust-policy pin must be anchored out-of-band. The "
    "statement issue time is signer-asserted unless separately timestamped by an independently "
    "trusted transparency service."
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def valid_hex(value: str, length: int) -> bool:
    return len(value) == length and all(char in "0123456789abcdef" for char in value)


def dsse_pae(payload_type: str, payload: bytes) -> bytes:
    return (
        f"DSSEv1 {len(payload_type.encode())} {payload_type} {len(payload)} ".encode()
        + payload
    )


def verify_signature_openssl(
    key_id: str,
    public_key_hex: str,
    signature_b64: str,
    pae: bytes,
    temp_dir: Path,
    index: int,
) -> None:
    assert valid_hex(public_key_hex, 64), f"malformed public key for {key_id}"
    public_key_der = SPKI_ED25519_PREFIX + bytes.fromhex(public_key_hex)
    signature = base64.b64decode(signature_b64, validate=True)
    assert len(signature) == 64, f"signature from {key_id} must be 64 bytes"

    public_path = temp_dir / f"pub-{index}.der"
    signature_path = temp_dir / f"sig-{index}.bin"
    pae_path = temp_dir / "dsse-pae.bin"
    public_path.write_bytes(public_key_der)
    signature_path.write_bytes(signature)
    pae_path.write_bytes(pae)
    result = subprocess.run(
        [
            "openssl",
            "pkeyutl",
            "-verify",
            "-pubin",
            "-inkey",
            str(public_path),
            "-keyform",
            "DER",
            "-rawin",
            "-in",
            str(pae_path),
            "-sigfile",
            str(signature_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"OpenSSL rejected signature for {key_id}: {result.stdout} {result.stderr}"
    )


def run(
    envelope_bytes: bytes,
    policy_bytes: bytes,
    expected_policy_sha256: str,
    verification_time_unix: int,
) -> dict:
    assert valid_hex(expected_policy_sha256, 64)
    actual_policy_sha256 = sha256(policy_bytes)
    assert actual_policy_sha256 == expected_policy_sha256, "trust policy pin mismatch"

    envelope = json.loads(envelope_bytes)
    policy = json.loads(policy_bytes)
    assert envelope["payload_type"] == PAYLOAD_TYPE
    assert envelope["signatures"], "DSSE envelope has no signatures"

    payload_bytes = base64.b64decode(envelope["payload"], validate=True)
    statement = json.loads(payload_bytes)
    assert statement["schema_version"] == 1
    for name in (
        "statement_id",
        "statement_type",
        "claim_scope",
        "query_id",
        "observer_record_id",
    ):
        assert isinstance(statement[name], str) and statement[name].strip()
    for name, length in (
        ("dataset_revision_commit_sha", 40),
        ("query_spec_sha256", 64),
        ("subject_sha256", 64),
        ("evidence_sha256", 64),
        ("protocol_sha256", 64),
    ):
        assert valid_hex(statement[name], length), f"invalid payload digest {name}"
    assert statement["issued_at_unix"] > 0
    assert statement["issued_at_unix"] <= verification_time_unix
    assert statement["expires_at_unix"] is None or statement["expires_at_unix"] > verification_time_unix

    assert policy["schema_version"] == 1 and policy["policy_id"].strip()
    assert policy["minimum_signatures"] > 0
    required_roles = policy["required_roles"]
    assert len(required_roles) == len(set(required_roles))
    assert all(isinstance(role, str) and role.strip() for role in required_roles)
    keys = {key["key_id"]: key for key in policy["trusted_keys"]}
    assert len(keys) == len(policy["trusted_keys"]), "duplicate trust key IDs"
    public_key_material = [key["public_key_hex"] for key in policy["trusted_keys"]]
    assert len(public_key_material) == len(set(public_key_material)), "duplicate public-key material under distinct key IDs"
    assert policy["minimum_signatures"] <= len(keys)

    pae = dsse_pae(envelope["payload_type"], payload_bytes)
    seen = set()
    covered_roles = set()
    verified_signers = []
    with tempfile.TemporaryDirectory(prefix="connectome-dsse-verify-") as temp:
        temp_dir = Path(temp)
        for index, signature_entry in enumerate(envelope["signatures"]):
            key_id = signature_entry["keyid"]
            assert key_id and key_id not in seen, "empty or duplicate signer key ID"
            seen.add(key_id)
            assert key_id in keys, f"untrusted signer {key_id}"
            key = keys[key_id]
            assert key["valid_from_unix"] <= verification_time_unix
            assert key["valid_until_unix"] is None or verification_time_unix <= key["valid_until_unix"]
            assert key["revoked_at_unix"] is None or verification_time_unix < key["revoked_at_unix"]
            assert statement["issued_at_unix"] >= key["valid_from_unix"]
            assert key["valid_until_unix"] is None or statement["issued_at_unix"] <= key["valid_until_unix"]
            assert key["role"].strip()
            verify_signature_openssl(
                key_id, key["public_key_hex"], signature_entry["sig"], pae, temp_dir, index
            )
            covered_roles.add(key["role"])
            verified_signers.append({"key_id": key_id, "role": key["role"]})

    assert len(verified_signers) >= policy["minimum_signatures"]
    assert set(required_roles).issubset(covered_roles), "missing required role"
    verified_signers.sort(key=lambda item: item["key_id"])
    sorted_covered_roles = sorted(covered_roles)

    return {
        "schema_version": 1,
        "tool": "connectome-evidence-attestation-audit",
        "envelope_sha256": sha256(envelope_bytes),
        "payload_sha256": sha256(payload_bytes),
        "trust_policy_sha256": actual_policy_sha256,
        "verification_time_unix": verification_time_unix,
        "payload_type": envelope["payload_type"],
        "statement_id": statement["statement_id"],
        "statement_type": statement["statement_type"],
        "claim_scope": statement["claim_scope"],
        "dataset_revision_commit_sha": statement["dataset_revision_commit_sha"],
        "query_id": statement["query_id"],
        "query_spec_sha256": statement["query_spec_sha256"],
        "subject_sha256": statement["subject_sha256"],
        "evidence_sha256": statement["evidence_sha256"],
        "protocol_sha256": statement["protocol_sha256"],
        "observer_record_id": statement["observer_record_id"],
        "issued_at_unix": statement["issued_at_unix"],
        "expires_at_unix": statement["expires_at_unix"],
        "required_roles": required_roles,
        "covered_roles": sorted_covered_roles,
        "minimum_signatures": policy["minimum_signatures"],
        "verified_signer_count": len(verified_signers),
        "verified_signers": verified_signers,
        "policy_accepted": True,
        "status": "SIGNED_EVIDENCE_VERIFIED_UNDER_PINNED_TRUST_POLICY",
        "signature_claim": "ED25519_SIGNATURES_VALID_OVER_DSSE_PAE",
        "trust_policy_claim": "TRUST_POLICY_MATCHED_CALLER_SUPPLIED_SHA256_PIN",
        "scientific_truth_claim": "NOT_ESTABLISHED",
        "scope_note": SCOPE_NOTE,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envelope", required=True, type=Path)
    parser.add_argument("--trust-policy", required=True, type=Path)
    parser.add_argument("--expected-trust-policy-sha256", required=True)
    parser.add_argument("--verification-time-unix", required=True, type=int)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    try:
        envelope_bytes = args.envelope.read_bytes()
        policy_bytes = args.trust_policy.read_bytes()
        expected = run(
            envelope_bytes,
            policy_bytes,
            args.expected_trust_policy_sha256,
            args.verification_time_unix,
        )
        actual = json.loads(args.report.read_text(encoding="utf-8"))
        assert actual == expected, "Rust signed-evidence receipt differs from independent OpenSSL oracle"
        print(
            f"verified {expected['status']}: "
            f"signers={expected['verified_signer_count']}; roles={','.join(expected['covered_roles'])}; "
            f"trust_policy_sha256={expected['trust_policy_sha256']}"
        )
        return 0
    except Exception as exc:
        print(f"signed evidence oracle failure: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
