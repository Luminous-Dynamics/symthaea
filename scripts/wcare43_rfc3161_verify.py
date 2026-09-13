#!/usr/bin/env python3
"""WCARE-43 RFC 3161 temporal preregistration verifier.

MeasurementOnly. Standard-library orchestration around a preregistered OpenSSL
backend. This script never grants runtime authority.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any

VERIFIER_PROTOCOL = "wcare43-rfc3161-preregistration-verifier-v1"
WCARE41_PROTOCOL = "wcare41-authenticated-preregistration-v1"
WCARE40_PROTOCOL = "wcare40-execution-replication-v1"
WCARE39_PROTOCOL = "wcare39-execution-capsule-v1"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
UTC20 = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z$")
PEM_CERT = re.compile(
    rb"-----BEGIN CERTIFICATE-----\r?\n.*?-----END CERTIFICATE-----\r?\n?",
    re.DOTALL,
)


class InvalidEvidence(Exception):
    pass


class IndeterminateInfrastructure(Exception):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise InvalidEvidence(f"read_failed:{path}:{exc}") from exc


def load_json(path: Path) -> tuple[dict[str, Any], bytes, str]:
    data = read_bytes(path)
    try:
        value = json.loads(data)
    except Exception as exc:
        raise InvalidEvidence(f"json_parse_failed:{path}:{exc}") from exc
    if not isinstance(value, dict):
        raise InvalidEvidence(f"json_not_object:{path}")
    return value, data, sha256_bytes(data)


def require_hex64(value: Any, label: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise InvalidEvidence(f"invalid_sha256:{label}")
    return value


def require_token(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or not re.fullmatch(r"[A-Za-z0-9._:-]+", value):
        raise InvalidEvidence(f"invalid_token:{label}")
    return value


def parse_utc20(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not UTC20.fullmatch(value):
        raise InvalidEvidence(f"invalid_utc:{label}")
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def canonical_utc(dt: datetime, preserve_fraction: bool = False) -> str:
    dt = dt.astimezone(timezone.utc)
    if preserve_fraction and dt.microsecond:
        return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond:06d}".rstrip("0") + "Z"
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_openssl_gen_time(value: str) -> datetime:
    value = value.strip()
    for fmt in ("%b %d %H:%M:%S %Y GMT", "%b %d %H:%M:%S.%f %Y GMT"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    raise InvalidEvidence(f"unparseable_rfc3161_gen_time:{value}")


def command_env() -> dict[str, str]:
    env = dict(os.environ)
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    env["TZ"] = "UTC"
    return env


def run(argv: list[str], *, input_bytes: bytes | None = None, allow_failure: bool = False) -> subprocess.CompletedProcess[bytes]:
    try:
        result = subprocess.run(
            argv,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=command_env(),
            check=False,
            shell=False,
        )
    except OSError as exc:
        raise IndeterminateInfrastructure(f"backend_execution_failed:{argv[0]}:{exc}") from exc
    if not allow_failure and result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip().replace("\n", " | ")
        raise InvalidEvidence(f"backend_command_failed:{argv[1:3]}:{result.returncode}:{detail}")
    return result


def base_result(detail: str = "uninitialized") -> dict[str, Any]:
    return {
        "authority": "MeasurementOnly",
        "verifier_protocol_version": VERIFIER_PROTOCOL,
        "wcare41_protocol_version": WCARE41_PROTOCOL,
        "disposition": "INVALID",
        "detail": detail,
        "wcare41_authentication_plan_sha256": None,
        "wcare43_verifier_sha256": None,
        "wcare40_plan_sha256": None,
        "wcare40_result_sha256": None,
        "wcare41_temporal_proof_package_sha256": None,
        "backend_policy_sha256": None,
        "timestamp_response_sha256": None,
        "trust_anchor_bundle_sha256": None,
        "untrusted_bundle_sha256": None,
        "openssl_executable_sha256": None,
        "openssl_version_output_sha256": None,
        "service_identity_commitment_sha256": None,
        "tsa_signer_certificate_der_sha256": None,
        "tsa_policy_label": None,
        "token_cryptographically_verified": False,
        "plan_message_imprint_verified": False,
        "certificate_time_validation_performed": False,
        "tsa_identity_policy_met": False,
        "tsa_policy_label_allowed": False,
        "revocation_mode": None,
        "revocation_status_checked": False,
        "synthetic_fixture_policy": False,
        "external_temporal_authority_established": False,
        "commitment_time_utc": None,
        "qualifying_replica_ids": [],
        "final_capsule_sha256_by_replica": {},
        "replica_start_utc_by_replica": {},
        "earliest_qualifying_replica_start_utc": None,
        "precedence_strictly_before_all_replicas": False,
        "wcare40_replication_supported": False,
        "preregistration_temporal_precedence_established": False,
        "builder_authentication_established": False,
        "reviewer_independence_established": False,
        "subject_correctness_established": False,
        "global_tsa_trust_established": False,
        "network_isolation_established": False,
        "sandbox_enforcement_established": False,
        "phenomenal_experience_established": False,
        "suffering_established": False,
        "moral_patienthood_established": False,
        "binding_consent_established": False,
        "veto_authority_granted": False,
        "self_preservation_authority_granted": False,
        "runtime_authority_granted": False,
    }


def parse_final_capsules(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in values:
        if "=" not in item:
            raise InvalidEvidence("invalid_final_capsule_argument")
        replica_id, raw_path = item.split("=", 1)
        if not replica_id or not raw_path or replica_id in result:
            raise InvalidEvidence(f"duplicate_or_invalid_final_capsule:{replica_id}")
        result[replica_id] = Path(raw_path)
    return result


def extract_candidate_time_and_policy(openssl: str, response: Path) -> tuple[datetime, str]:
    text = run([openssl, "ts", "-reply", "-in", str(response), "-text"]).stdout.decode("utf-8", "strict")
    time_match = re.search(r"^Time stamp:\s*(.+?)\s*$", text, re.MULTILINE)
    policy_match = re.search(r"^Policy OID:\s*(.+?)\s*$", text, re.MULTILINE)
    if not time_match or not policy_match:
        raise InvalidEvidence("rfc3161_text_missing_time_or_policy")
    return parse_openssl_gen_time(time_match.group(1)), policy_match.group(1).strip()


def extract_tsa_signer_der_sha256(openssl: str, response: Path) -> str:
    with tempfile.TemporaryDirectory(prefix="wcare43-token-") as raw_tmp:
        tmp = Path(raw_tmp)
        token = tmp / "token.der"
        certs = tmp / "certs.pem"
        run([openssl, "ts", "-reply", "-in", str(response), "-token_out", "-out", str(token)])
        extracted = run([openssl, "pkcs7", "-inform", "DER", "-in", str(token), "-print_certs"]).stdout
        certs.write_bytes(extracted)
        blocks = PEM_CERT.findall(extracted)
        if not blocks:
            raise InvalidEvidence("timestamp_token_contains_no_certificates")
        candidates: list[str] = []
        for index, block in enumerate(blocks):
            cert_path = tmp / f"cert-{index}.pem"
            cert_path.write_bytes(block)
            purpose = run([openssl, "x509", "-in", str(cert_path), "-noout", "-purpose"]).stdout.decode("utf-8", "strict")
            if re.search(r"^Time Stamp signing\s*:\s*Yes\s*$", purpose, re.MULTILINE):
                der = run([openssl, "x509", "-in", str(cert_path), "-outform", "DER"]).stdout
                candidates.append(sha256_bytes(der))
        if len(candidates) != 1:
            raise InvalidEvidence(f"timestamp_signer_candidate_count:{len(candidates)}")
        return candidates[0]


def earliest_replica_starts(
    w40: dict[str, Any],
    supplied: dict[str, Path],
    out: dict[str, Any],
) -> dict[str, datetime]:
    replica_ids = w40.get("subject_eligible_replica_ids")
    expected_hashes = w40.get("final_capsule_sha256_by_replica")
    if not isinstance(replica_ids, list) or not replica_ids or not isinstance(expected_hashes, dict):
        raise InvalidEvidence("wcare40_missing_subject_eligible_replica_evidence")
    if len(replica_ids) != len(set(replica_ids)) or any(not isinstance(x, str) or not x for x in replica_ids):
        raise InvalidEvidence("wcare40_invalid_subject_eligible_replica_ids")
    if set(supplied) != set(replica_ids):
        raise InvalidEvidence("final_capsule_argument_census_mismatch")

    starts: dict[str, datetime] = {}
    for replica_id in replica_ids:
        capsule, capsule_bytes, capsule_sha = load_json(supplied[replica_id])
        if capsule_sha != require_hex64(expected_hashes.get(replica_id), f"wcare40_final_capsule:{replica_id}"):
            raise InvalidEvidence(f"final_capsule_sha256_mismatch:{replica_id}")
        if capsule.get("protocol_version") != WCARE39_PROTOCOL:
            raise InvalidEvidence(f"final_capsule_protocol_mismatch:{replica_id}")
        if capsule.get("capsule_phase") != "FINAL" or capsule.get("classification") != "QUALIFIED_EXECUTION" or capsule.get("environment_integrity") != "QUALIFIED":
            raise InvalidEvidence(f"final_capsule_not_qualified:{replica_id}")
        commands = capsule.get("commands")
        if not isinstance(commands, list):
            raise InvalidEvidence(f"final_capsule_commands_missing:{replica_id}")
        parsed: list[datetime] = []
        for command in commands:
            if not isinstance(command, dict):
                raise InvalidEvidence(f"final_capsule_command_invalid:{replica_id}")
            started = command.get("started_utc")
            if started is not None:
                parsed.append(parse_utc20(started, f"command_started_utc:{replica_id}"))
        if not parsed:
            raise InvalidEvidence(f"final_capsule_has_no_started_command:{replica_id}")
        starts[replica_id] = min(parsed)
        out["final_capsule_sha256_by_replica"][replica_id] = capsule_sha
        out["replica_start_utc_by_replica"][replica_id] = canonical_utc(starts[replica_id])
    out["qualifying_replica_ids"] = list(replica_ids)
    out["earliest_qualifying_replica_start_utc"] = canonical_utc(min(starts.values()))
    return starts


def verify(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    out = base_result()
    try:
        auth_plan, auth_plan_bytes, auth_plan_sha = load_json(Path(args.authentication_plan))
        proof, proof_bytes, proof_sha = load_json(Path(args.proof_package))
        policy, policy_bytes, policy_sha = load_json(Path(args.backend_policy))
        w40_plan, plan_bytes, plan_sha = load_json(Path(args.wcare40_plan))
        w40_result, result_bytes, result_sha = load_json(Path(args.wcare40_result))
        response_path = Path(args.timestamp_response)
        response_bytes = read_bytes(response_path)
        response_sha = sha256_bytes(response_bytes)
        trust_path = Path(args.trust_anchor_bundle)
        trust_bytes = read_bytes(trust_path)
        trust_sha = sha256_bytes(trust_bytes)
        untrusted_path = None if args.untrusted_bundle == "-" else Path(args.untrusted_bundle)
        untrusted_bytes = b"" if untrusted_path is None else read_bytes(untrusted_path)
        untrusted_sha = None if untrusted_path is None else sha256_bytes(untrusted_bytes)

        out.update({
            "wcare41_authentication_plan_sha256": auth_plan_sha,
            "wcare40_plan_sha256": plan_sha,
            "wcare40_result_sha256": result_sha,
            "wcare41_temporal_proof_package_sha256": proof_sha,
            "backend_policy_sha256": policy_sha,
            "timestamp_response_sha256": response_sha,
            "trust_anchor_bundle_sha256": trust_sha,
            "untrusted_bundle_sha256": untrusted_sha,
        })

        if auth_plan.get("protocol_version") != WCARE41_PROTOCOL or proof.get("protocol_version") != WCARE41_PROTOCOL:
            raise InvalidEvidence("wcare41_protocol_mismatch")
        if policy.get("protocol_version") != VERIFIER_PROTOCOL:
            raise InvalidEvidence("wcare43_policy_protocol_mismatch")
        if w40_plan.get("protocol_version") != WCARE40_PROTOCOL or w40_result.get("protocol_version") != WCARE40_PROTOCOL:
            raise InvalidEvidence("wcare40_protocol_mismatch")

        verifier_sha = sha256_bytes(Path(__file__).resolve().read_bytes())
        out["wcare43_verifier_sha256"] = verifier_sha
        temporal_plan = auth_plan.get("temporal_verifier")
        if not isinstance(temporal_plan, dict):
            raise InvalidEvidence("authentication_plan_temporal_verifier_missing")
        backend_id = require_token(policy.get("backend_id"), "backend_policy.backend_id")
        if require_token(temporal_plan.get("backend_id"), "authentication_plan.temporal_verifier.backend_id") != backend_id:
            raise InvalidEvidence("authentication_plan_backend_id_mismatch")
        if require_hex64(temporal_plan.get("executable_sha256"), "authentication_plan.temporal_verifier.executable_sha256") != verifier_sha:
            raise InvalidEvidence("authentication_plan_verifier_sha256_mismatch")
        if require_hex64(temporal_plan.get("policy_sha256"), "authentication_plan.temporal_verifier.policy_sha256") != policy_sha:
            raise InvalidEvidence("authentication_plan_backend_policy_sha256_mismatch")

        if require_hex64(auth_plan.get("wcare40_plan_sha256"), "authentication_plan.wcare40_plan_sha256") != plan_sha:
            raise InvalidEvidence("authentication_plan_wcare40_plan_sha256_mismatch")
        if require_hex64(auth_plan.get("wcare40_result_sha256"), "authentication_plan.wcare40_result_sha256") != result_sha:
            raise InvalidEvidence("authentication_plan_wcare40_result_sha256_mismatch")
        if require_hex64(proof.get("wcare40_plan_sha256"), "proof.wcare40_plan_sha256") != plan_sha:
            raise InvalidEvidence("proof_wcare40_plan_sha256_mismatch")
        if require_token(proof.get("backend_id"), "proof.backend_id") != backend_id:
            raise InvalidEvidence("proof_backend_id_mismatch")
        if require_hex64(proof.get("backend_executable_sha256"), "proof.backend_executable_sha256") != verifier_sha:
            raise InvalidEvidence("proof_verifier_sha256_mismatch")
        if require_hex64(proof.get("backend_policy_sha256"), "proof.backend_policy_sha256") != policy_sha:
            raise InvalidEvidence("proof_backend_policy_sha256_mismatch")
        if require_hex64(proof.get("proof_artifact_sha256"), "proof.proof_artifact_sha256") != response_sha:
            raise InvalidEvidence("timestamp_response_sha256_mismatch")
        if proof.get("proof_format") != "rfc3161-timestamp-response-der" or proof.get("proof_version") != "rfc3161-v1":
            raise InvalidEvidence("unsupported_rfc3161_proof_profile")
        if proof.get("inclusion_commitment_sha256") is not None:
            raise InvalidEvidence("rfc3161_v1_inclusion_commitment_must_be_null")

        if require_hex64(policy.get("trust_anchor_bundle_sha256"), "policy.trust_anchor_bundle_sha256") != trust_sha:
            raise InvalidEvidence("trust_anchor_bundle_sha256_mismatch")
        expected_untrusted = policy.get("untrusted_bundle_sha256")
        allow_empty = policy.get("allow_empty_untrusted_bundle") is True
        if untrusted_path is None:
            if expected_untrusted is not None or not allow_empty:
                raise InvalidEvidence("untrusted_bundle_required_or_policy_mismatch")
        else:
            if require_hex64(expected_untrusted, "policy.untrusted_bundle_sha256") != untrusted_sha:
                raise InvalidEvidence("untrusted_bundle_sha256_mismatch")

        if policy.get("message_imprint_algorithm") != "sha256" or policy.get("certificate_purpose") != "timestampsign":
            raise InvalidEvidence("unsupported_backend_policy_profile")
        if policy.get("revocation_mode") != "OfflineStaticNoRevocation":
            raise InvalidEvidence("unsupported_revocation_mode")
        out["revocation_mode"] = "OfflineStaticNoRevocation"
        out["synthetic_fixture_policy"] = policy.get("synthetic_fixture_policy") is True

        if require_hex64(w40_result.get("plan_sha256"), "wcare40_result.plan_sha256") != plan_sha:
            raise InvalidEvidence("wcare40_result_plan_sha256_mismatch")
        out["wcare40_replication_supported"] = w40_result.get("disposition") == "REPLICATION_SUPPORTED"

        openssl_path_raw = shutil.which("openssl")
        if not openssl_path_raw:
            raise IndeterminateInfrastructure("openssl_not_found")
        openssl_path = str(Path(openssl_path_raw).resolve())
        openssl_sha = sha256_bytes(read_bytes(Path(openssl_path)))
        version = run([openssl_path, "version"])
        if version.returncode != 0:
            raise IndeterminateInfrastructure("openssl_version_failed")
        version_sha = sha256_bytes(version.stdout)
        out["openssl_executable_sha256"] = openssl_sha
        out["openssl_version_output_sha256"] = version_sha
        if require_hex64(policy.get("openssl_executable_sha256"), "policy.openssl_executable_sha256") != openssl_sha:
            raise InvalidEvidence("openssl_executable_sha256_mismatch")
        if require_hex64(policy.get("openssl_version_output_sha256"), "policy.openssl_version_output_sha256") != version_sha:
            raise InvalidEvidence("openssl_version_output_sha256_mismatch")

        candidate_time, tsa_policy_label = extract_candidate_time_and_policy(openssl_path, response_path)
        out["tsa_policy_label"] = tsa_policy_label
        claimed_commitment = proof.get("commitment_time_utc")
        if not isinstance(claimed_commitment, str):
            raise InvalidEvidence("proof_commitment_time_missing")
        if claimed_commitment != canonical_utc(candidate_time, preserve_fraction=True):
            raise InvalidEvidence("proof_commitment_time_does_not_match_token")

        verify_argv = [
            openssl_path,
            "ts",
            "-verify",
            "-digest",
            plan_sha,
            "-in",
            str(response_path),
            "-CAfile",
            str(trust_path),
        ]
        if untrusted_path is not None:
            verify_argv += ["-untrusted", str(untrusted_path)]
        verify_argv += ["-purpose", "timestampsign", "-attime", str(int(candidate_time.timestamp()))]
        verified = run(verify_argv, allow_failure=True)
        if verified.returncode != 0:
            detail = verified.stderr.decode("utf-8", "replace").strip().replace("\n", " | ")
            raise InvalidEvidence(f"rfc3161_verification_failed:{detail}")
        out["token_cryptographically_verified"] = True
        out["plan_message_imprint_verified"] = True
        out["certificate_time_validation_performed"] = True
        out["commitment_time_utc"] = canonical_utc(candidate_time, preserve_fraction=True)

        signer_sha = extract_tsa_signer_der_sha256(openssl_path, response_path)
        out["tsa_signer_certificate_der_sha256"] = signer_sha
        service_identity = require_hex64(proof.get("service_identity_commitment_sha256"), "proof.service_identity_commitment_sha256")
        out["service_identity_commitment_sha256"] = service_identity
        expected_signer = require_hex64(policy.get("expected_tsa_signer_certificate_der_sha256"), "policy.expected_tsa_signer_certificate_der_sha256")
        out["tsa_identity_policy_met"] = signer_sha == expected_signer == service_identity

        allowed_labels = policy.get("accepted_tsa_policy_labels")
        if not isinstance(allowed_labels, list) or any(not isinstance(x, str) or not x for x in allowed_labels) or len(allowed_labels) != len(set(allowed_labels)):
            raise InvalidEvidence("invalid_accepted_tsa_policy_labels")
        out["tsa_policy_label_allowed"] = not allowed_labels or tsa_policy_label in allowed_labels

        supplied_capsules = parse_final_capsules(args.final_capsule)
        starts = earliest_replica_starts(w40_result, supplied_capsules, out)
        precedence = all(candidate_time < start for start in starts.values())
        out["precedence_strictly_before_all_replicas"] = precedence

        if not out["tsa_identity_policy_met"]:
            out["disposition"] = "NOT_ESTABLISHED"
            out["detail"] = "tsa_identity_not_authorized_by_bound_policy"
            return out, 1
        if not out["tsa_policy_label_allowed"]:
            out["disposition"] = "NOT_ESTABLISHED"
            out["detail"] = "tsa_policy_label_not_allowed"
            return out, 1
        if not out["wcare40_replication_supported"]:
            out["disposition"] = "NOT_ESTABLISHED"
            out["detail"] = "wcare40_result_not_replication_supported"
            return out, 1
        if not precedence:
            out["disposition"] = "NOT_ESTABLISHED"
            out["detail"] = "timestamp_not_strictly_before_all_qualifying_replicas"
            return out, 1

        out["disposition"] = "ESTABLISHED"
        out["detail"] = "exact_rfc3161_plan_commitment_precedes_all_qualifying_replicas_under_bound_policy"
        out["preregistration_temporal_precedence_established"] = True
        return out, 0
    except IndeterminateInfrastructure as exc:
        out["disposition"] = "INDETERMINATE"
        out["detail"] = str(exc)
        return out, 3
    except InvalidEvidence as exc:
        out["disposition"] = "INVALID"
        out["detail"] = str(exc)
        return out, 4
    except Exception as exc:
        out["disposition"] = "INVALID"
        out["detail"] = f"unexpected_verifier_error:{type(exc).__name__}:{exc}"
        return out, 4


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("authentication_plan")
    p.add_argument("proof_package")
    p.add_argument("backend_policy")
    p.add_argument("wcare40_plan")
    p.add_argument("wcare40_result")
    p.add_argument("timestamp_response")
    p.add_argument("trust_anchor_bundle")
    p.add_argument("untrusted_bundle", help="PEM bundle path or '-' when policy permits empty")
    p.add_argument("--final-capsule", action="append", default=[], metavar="REPLICA_ID=PATH")
    return p


def main() -> int:
    args = parser().parse_args()
    result, exit_code = verify(args)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
