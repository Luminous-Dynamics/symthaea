#!/usr/bin/env python3
"""Adversarial WCARE-43 RFC3161 campaign using a synthetic local TSA.

The fixture verifies protocol mechanics only. It cannot establish production
external preregistration.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "docs/release/evidence/fixtures/wcare43"
VERIFIER = ROOT / "scripts/wcare43_rfc3161_verify.py"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def write_json(path: Path, value: object) -> str:
    data = canonical(value)
    path.write_bytes(data)
    return sha(data)


def run_bytes(argv: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def openssl_identity() -> tuple[str, str, str]:
    raw = shutil.which("openssl")
    if not raw:
        raise AssertionError("openssl_not_found_for_wcare43_selftest")
    path = str(Path(raw).resolve())
    exe_sha = sha(Path(path).read_bytes())
    version = run_bytes([path, "version"])
    assert version.returncode == 0, version.stderr.decode("utf-8", "replace")
    return path, exe_sha, sha(version.stdout)


def tsa_der_sha(openssl: str, tsa_pem: Path) -> str:
    result = run_bytes([openssl, "x509", "-in", str(tsa_pem), "-outform", "DER"])
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")
    return sha(result.stdout)


def decode_fixture_response(path: Path) -> bytes:
    encoded = (FIX / "synthetic_response.tsr.b64").read_text().strip()
    data = base64.b64decode(encoded, validate=True)
    path.write_bytes(data)
    return data


def make_evaluation(
    tmp: Path,
    *,
    openssl: str,
    openssl_sha: str,
    openssl_version_sha: str,
    plan: Path,
    result: Path,
    response: Path,
    capsule_a: Path,
    capsule_b: Path,
    trust: Path,
    untrusted: Path,
    expected_signer_sha: str,
    service_identity_sha: str | None = None,
    synthetic: bool = True,
) -> tuple[list[str], dict, dict, dict]:
    verifier_sha = sha(VERIFIER.read_bytes())
    plan_sha = sha(plan.read_bytes())
    result_sha = sha(result.read_bytes())
    response_sha = sha(response.read_bytes())
    trust_sha = sha(trust.read_bytes())
    untrusted_sha = sha(untrusted.read_bytes())
    backend_id = "wcare43.rfc3161.openssl.v1"

    policy = {
        "protocol_version": "wcare43-rfc3161-preregistration-verifier-v1",
        "policy_id": "wcare43.synthetic.fixture.policy",
        "backend_id": backend_id,
        "openssl_executable_sha256": openssl_sha,
        "openssl_version_output_sha256": openssl_version_sha,
        "trust_anchor_bundle_sha256": trust_sha,
        "untrusted_bundle_sha256": untrusted_sha,
        "expected_tsa_signer_certificate_der_sha256": expected_signer_sha,
        "accepted_tsa_policy_labels": [],
        "message_imprint_algorithm": "sha256",
        "certificate_purpose": "timestampsign",
        "revocation_mode": "OfflineStaticNoRevocation",
        "allow_empty_untrusted_bundle": False,
        "synthetic_fixture_policy": synthetic,
    }
    policy_path = tmp / "policy.json"
    policy_sha = write_json(policy_path, policy)

    auth = {
        "protocol_version": "wcare41-authenticated-preregistration-v1",
        "wcare40_plan_sha256": plan_sha,
        "wcare40_result_sha256": result_sha,
        "wcare40_frontdoor_sha256": "1" * 64,
        "wcare40_core_verifier_sha256": "2" * 64,
        "builder_verifier": {
            "backend_id": "unused.builder.backend",
            "executable_sha256": "3" * 64,
            "policy_sha256": "4" * 64,
        },
        "temporal_verifier": {
            "backend_id": backend_id,
            "executable_sha256": verifier_sha,
            "policy_sha256": policy_sha,
        },
        "require_complete_builder_attestation_coverage": True,
        "require_temporal_preregistration": True,
        "evaluation_utc": "2026-09-14T00:00:00Z",
    }
    auth_path = tmp / "authentication-plan.json"
    write_json(auth_path, auth)

    manifest = json.loads((FIX / "fixture_manifest.json").read_text())
    proof = {
        "protocol_version": "wcare41-authenticated-preregistration-v1",
        "wcare40_plan_sha256": plan_sha,
        "backend_id": backend_id,
        "backend_executable_sha256": verifier_sha,
        "backend_policy_sha256": policy_sha,
        "service_identity_commitment_sha256": service_identity_sha or expected_signer_sha,
        "proof_artifact_sha256": response_sha,
        "commitment_time_utc": manifest["commitment_time_utc"],
        "inclusion_commitment_sha256": None,
        "proof_format": "rfc3161-timestamp-response-der",
        "proof_version": "rfc3161-v1",
    }
    proof_path = tmp / "proof.json"
    write_json(proof_path, proof)

    argv = [
        sys.executable,
        str(VERIFIER),
        str(auth_path),
        str(proof_path),
        str(policy_path),
        str(plan),
        str(result),
        str(response),
        str(trust),
        str(untrusted),
        "--final-capsule",
        f"a={capsule_a}",
        "--final-capsule",
        f"b={capsule_b}",
    ]
    return argv, policy, auth, proof


def execute(argv: list[str]) -> tuple[int, dict]:
    result = run_bytes(argv)
    try:
        value = json.loads(result.stdout)
    except Exception as exc:
        raise AssertionError(
            f"invalid verifier JSON: rc={result.returncode} stdout={result.stdout!r} stderr={result.stderr!r}"
        ) from exc
    return result.returncode, value


def dynamic_result(tmp: Path, plan: Path, a: Path, b: Path) -> Path:
    value = {
        "protocol_version": "wcare40-execution-replication-v1",
        "plan_sha256": sha(plan.read_bytes()),
        "disposition": "REPLICATION_SUPPORTED",
        "subject_eligible_replica_ids": ["a", "b"],
        "final_capsule_sha256_by_replica": {
            "a": sha(a.read_bytes()),
            "b": sha(b.read_bytes()),
        },
    }
    path = tmp / "dynamic-w40-result.json"
    write_json(path, value)
    return path


def dynamic_capsule(path: Path, started: str) -> None:
    write_json(path, {
        "protocol_version": "wcare39-execution-capsule-v1",
        "capsule_phase": "FINAL",
        "classification": "QUALIFIED_EXECUTION",
        "environment_integrity": "QUALIFIED",
        "commands": [{"started_utc": started}],
    })


def main() -> int:
    manifest = json.loads((FIX / "fixture_manifest.json").read_text())
    assert manifest["synthetic"] is True
    assert manifest["private_key_material_in_repository"] is False
    assert manifest["external_temporal_authority_established"] is False

    plan = FIX / "synthetic_plan.json"
    result = FIX / "synthetic_wcare40_result.json"
    capsule_a = FIX / "a.final.json"
    capsule_b = FIX / "b.final.json"
    root = FIX / "synthetic_root.pem"
    tsa = FIX / "synthetic_tsa.pem"
    assert sha(plan.read_bytes()) == manifest["wcare40_plan_sha256"]
    assert sha(result.read_bytes()) == manifest["wcare40_result_sha256"]
    assert sha(root.read_bytes()) == manifest["trust_anchor_pem_sha256"]
    assert sha(tsa.read_bytes()) == manifest["untrusted_tsa_pem_sha256"]
    assert sha(capsule_a.read_bytes()) == manifest["final_capsule_sha256_by_replica"]["a"]
    assert sha(capsule_b.read_bytes()) == manifest["final_capsule_sha256_by_replica"]["b"]

    openssl, openssl_sha, openssl_version_sha = openssl_identity()
    signer_sha = tsa_der_sha(openssl, tsa)
    assert signer_sha == manifest["tsa_signer_certificate_der_sha256"]

    with tempfile.TemporaryDirectory(prefix="wcare43-selftest-") as raw_tmp:
        tmp = Path(raw_tmp)
        response = tmp / "response.tsr"
        response_bytes = decode_fixture_response(response)
        assert sha(response_bytes) == manifest["timestamp_response_der_sha256"]

        baseline = tmp / "baseline"
        baseline.mkdir()
        argv, _, _, _ = make_evaluation(
            baseline,
            openssl=openssl,
            openssl_sha=openssl_sha,
            openssl_version_sha=openssl_version_sha,
            plan=plan,
            result=result,
            response=response,
            capsule_a=capsule_a,
            capsule_b=capsule_b,
            trust=root,
            untrusted=tsa,
            expected_signer_sha=signer_sha,
            synthetic=True,
        )
        rc, value = execute(argv)
        assert rc == 1, value
        assert value["disposition"] == "NOT_ESTABLISHED", value
        assert value["detail"] == "synthetic_tsa_fixture_cannot_establish_external_preregistration", value
        assert value["token_cryptographically_verified"] is True, value
        assert value["plan_message_imprint_verified"] is True, value
        assert value["message_imprint_algorithm"] == "sha256", value
        assert value["precedence_strictly_before_all_replicas"] is True, value
        assert value["preregistration_temporal_precedence_established"] is False, value
        assert value["external_temporal_authority_established"] is False, value

        wrong_plan_dir = tmp / "wrong-plan"
        wrong_plan_dir.mkdir()
        wrong_plan = wrong_plan_dir / "plan.json"
        wrong_plan.write_text('{"protocol_version":"wcare40-execution-replication-v1","replica_slots":[{"replica_id":"different"}]}')
        wrong_result = dynamic_result(wrong_plan_dir, wrong_plan, capsule_a, capsule_b)
        argv, _, _, _ = make_evaluation(
            wrong_plan_dir,
            openssl=openssl,
            openssl_sha=openssl_sha,
            openssl_version_sha=openssl_version_sha,
            plan=wrong_plan,
            result=wrong_result,
            response=response,
            capsule_a=capsule_a,
            capsule_b=capsule_b,
            trust=root,
            untrusted=tsa,
            expected_signer_sha=signer_sha,
        )
        rc, value = execute(argv)
        assert rc == 4 and value["disposition"] == "INVALID", value
        assert "rfc3161_verification_failed" in value["detail"], value

        modified_dir = tmp / "modified-token"
        modified_dir.mkdir()
        modified_response = modified_dir / "response.tsr"
        damaged = bytearray(response_bytes)
        damaged[-1] ^= 0x01
        modified_response.write_bytes(damaged)
        argv, _, _, _ = make_evaluation(
            modified_dir,
            openssl=openssl,
            openssl_sha=openssl_sha,
            openssl_version_sha=openssl_version_sha,
            plan=plan,
            result=result,
            response=modified_response,
            capsule_a=capsule_a,
            capsule_b=capsule_b,
            trust=root,
            untrusted=tsa,
            expected_signer_sha=signer_sha,
        )
        rc, value = execute(argv)
        assert rc == 4 and value["disposition"] == "INVALID", value

        identity_dir = tmp / "untrusted-identity"
        identity_dir.mkdir()
        argv, _, _, _ = make_evaluation(
            identity_dir,
            openssl=openssl,
            openssl_sha=openssl_sha,
            openssl_version_sha=openssl_version_sha,
            plan=plan,
            result=result,
            response=response,
            capsule_a=capsule_a,
            capsule_b=capsule_b,
            trust=root,
            untrusted=tsa,
            expected_signer_sha="0" * 64,
            service_identity_sha="0" * 64,
        )
        rc, value = execute(argv)
        assert rc == 1 and value["disposition"] == "NOT_ESTABLISHED", value
        assert value["detail"] == "tsa_identity_not_authorized_by_bound_policy", value

        equal_dir = tmp / "equal-time"
        equal_dir.mkdir()
        equal_a = equal_dir / "a.final.json"
        equal_b = equal_dir / "b.final.json"
        dynamic_capsule(equal_a, manifest["commitment_time_utc"])
        dynamic_capsule(equal_b, "2026-09-13T23:05:00Z")
        equal_result = dynamic_result(equal_dir, plan, equal_a, equal_b)
        argv, _, _, _ = make_evaluation(
            equal_dir,
            openssl=openssl,
            openssl_sha=openssl_sha,
            openssl_version_sha=openssl_version_sha,
            plan=plan,
            result=equal_result,
            response=response,
            capsule_a=equal_a,
            capsule_b=equal_b,
            trust=root,
            untrusted=tsa,
            expected_signer_sha=signer_sha,
        )
        rc, value = execute(argv)
        assert rc == 1 and value["disposition"] == "NOT_ESTABLISHED", value
        assert value["detail"] == "timestamp_not_strictly_before_all_qualifying_replicas", value
        assert value["precedence_strictly_before_all_replicas"] is False, value

        malformed_dir = tmp / "malformed"
        malformed_dir.mkdir()
        malformed = malformed_dir / "response.tsr"
        malformed.write_bytes(response_bytes[:64])
        argv, _, _, _ = make_evaluation(
            malformed_dir,
            openssl=openssl,
            openssl_sha=openssl_sha,
            openssl_version_sha=openssl_version_sha,
            plan=plan,
            result=result,
            response=malformed,
            capsule_a=capsule_a,
            capsule_b=capsule_b,
            trust=root,
            untrusted=tsa,
            expected_signer_sha=signer_sha,
        )
        rc, value = execute(argv)
        assert rc == 4 and value["disposition"] == "INVALID", value

    print(json.dumps({
        "authority": "MeasurementOnly",
        "classification": "PASS_WCARE43_SELFTEST",
        "real_external_preregistration_established": False,
        "synthetic_valid_early_token_verified": True,
        "wrong_plan_rejected": True,
        "modified_token_rejected": True,
        "untrusted_tsa_identity_not_established": True,
        "equal_time_not_established": True,
        "malformed_response_rejected": True,
        "runtime_authority_granted": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
