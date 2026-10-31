#!/usr/bin/env python3
"""Typed, fail-closed proof-carrying risk certificate manifest (research v1).

This module validates certificate semantics and evidence binding. It does not
prove statistical assumptions such as marginal e-value validity or exchangeability;
those remain explicit assumptions in the manifest and must be independently justified.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import re
from typing import Any

SCHEMA = "symthaea-risk-certificate-v1"
DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
RISK_KINDS = {"expected_risk", "high_probability_risk", "fdr", "supfdr", "fwer"}
PROOF_LANES = {
    "exact_finite_exchangeability",
    "analytical_fdr_theorem",
    "exact_intersection_closure",
    "theorem_backed_eprocess",
    "high_probability_bound",
    "simulation_only",
}
STATUSES = {"pass", "fail", "inconclusive"}
FORBIDDEN_KEYS = {
    "labels", "raw_labels", "calibration_data", "calibration_examples",
    "test_examples", "raw_features", "raw_predictions", "raw_evalues",
    "private_key", "password", "secret",
}


class CertificateError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def parse_exact_fraction(value: Any, field: str) -> Fraction:
    if not isinstance(value, str):
        raise CertificateError(f"{field} must be an exact rational string")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise CertificateError(f"{field} is not a valid rational") from exc


def _walk_reject_raw_payloads(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key.lower() in FORBIDDEN_KEYS:
                raise CertificateError(f"raw/private payload field is forbidden: {path}.{key}")
            _walk_reject_raw_payloads(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _walk_reject_raw_payloads(nested, f"{path}[{index}]")


def _validate_digest_map(digests: Any, field: str) -> None:
    if not isinstance(digests, dict) or not digests:
        raise CertificateError(f"{field} must be a nonempty digest map")
    for name, digest in digests.items():
        if not isinstance(name, str) or not name:
            raise CertificateError(f"{field} contains an invalid digest name")
        if not isinstance(digest, str) or not DIGEST_RE.fullmatch(digest):
            raise CertificateError(f"{field}.{name} must be a lowercase SHA-256 digest")


def _validate_manifest(manifest: dict) -> None:
    if manifest.get("schema") != SCHEMA:
        raise CertificateError("unsupported certificate schema")
    _walk_reject_raw_payloads(manifest)

    target = manifest.get("risk_target")
    assumptions = manifest.get("assumptions")
    scope = manifest.get("scope")
    method = manifest.get("method")
    evidence = manifest.get("evidence")
    lane = manifest.get("proof_lane")

    if not all(isinstance(x, dict) for x in (target, assumptions, scope, method, evidence)):
        raise CertificateError("risk_target, assumptions, scope, method and evidence must be objects")
    kind = target.get("kind")
    if kind not in RISK_KINDS:
        raise CertificateError("unsupported risk target kind")
    level = parse_exact_fraction(target.get("level"), "risk_target.level")
    if not 0 < level < 1:
        raise CertificateError("risk_target.level must be in (0,1)")
    if not isinstance(target.get("metric"), str) or not target["metric"]:
        raise CertificateError("risk_target.metric is required")
    if lane not in PROOF_LANES:
        raise CertificateError("unsupported proof lane")
    if not isinstance(method.get("name"), str) or not method["name"]:
        raise CertificateError("method.name is required")
    if not isinstance(method.get("version"), str) or not method["version"]:
        raise CertificateError("method.version is required")
    _validate_digest_map(evidence, "evidence")

    if kind == "expected_risk":
        if lane != "exact_finite_exchangeability":
            raise CertificateError("expected_risk requires exact_finite_exchangeability in this v1 contract")
        if assumptions.get("sampling") != "finite_exchangeability":
            raise CertificateError("expected_risk certificate must declare finite exchangeability")
        if scope.get("universe") != "all_augmented_count_vectors":
            raise CertificateError("exact CRC scope must enumerate all augmented count vectors")
        if not isinstance(scope.get("n_calibration"), int) or scope["n_calibration"] < 1:
            raise CertificateError("expected_risk scope needs positive n_calibration")
        if not isinstance(scope.get("augmented_state_count"), int) or scope["augmented_state_count"] < 1:
            raise CertificateError("expected_risk scope needs augmented_state_count")
        if target.get("time_uniform") is not False:
            raise CertificateError("finite CRC expected risk is not a time-uniform guarantee")

    elif kind == "high_probability_risk":
        if lane != "high_probability_bound":
            raise CertificateError("high_probability_risk requires high_probability_bound")
        confidence = parse_exact_fraction(target.get("confidence_level"), "risk_target.confidence_level")
        if not 0 < confidence < 1:
            raise CertificateError("confidence_level must be in (0,1)")
        if assumptions.get("sampling") not in {"iid", "exchangeable"}:
            raise CertificateError("high-probability risk needs a declared sampling assumption")

    elif kind == "fdr":
        if lane not in {"analytical_fdr_theorem", "exact_intersection_closure"}:
            raise CertificateError("offline FDR needs a fixed-family FDR proof lane")
        if scope.get("fixed_family") is not True or scope.get("time_uniform") is not False:
            raise CertificateError("FDR certificate must be explicitly fixed-family and offline")
        if not isinstance(scope.get("family_size"), int) or scope["family_size"] < 1:
            raise CertificateError("FDR scope needs a positive family_size")
        if assumptions.get("marginal_evalues_valid") is not True:
            raise CertificateError("FDR certificate must state marginal e-value validity")
        if assumptions.get("dependence_scope") != "arbitrary_dependence":
            raise CertificateError("FDR e-value guarantee must state its dependence scope")

    elif kind == "supfdr":
        if lane != "theorem_backed_eprocess":
            raise CertificateError("SupFDR requires a theorem-backed e-process lane")
        if target.get("time_uniform") is not True or scope.get("online") is not True:
            raise CertificateError("SupFDR must be explicitly online and time-uniform")
        if assumptions.get("filtration") != "one_global_filtration":
            raise CertificateError("SupFDR requires one common global filtration")
        for key in ("predictable_arrivals", "optional_stopping_valid", "eprocess_supermartingale"):
            if assumptions.get(key) is not True:
                raise CertificateError(f"SupFDR assumption {key} must be true")
        if not isinstance(scope.get("horizon"), (int, str)) or scope["horizon"] in (0, ""):
            raise CertificateError("SupFDR needs an explicit positive or unbounded horizon")

    elif kind == "fwer":
        if lane not in {"exact_intersection_closure", "high_probability_bound"}:
            raise CertificateError("FWER needs a closure or high-probability proof lane")
        if not isinstance(scope.get("family_size"), int) or scope["family_size"] < 1:
            raise CertificateError("FWER scope needs a positive family_size")

    if lane == "simulation_only":
        raise CertificateError("simulation_only is not an admissible proof lane in v1")


def _validate_receipt(manifest: dict, manifest_digest: str, receipt: dict) -> None:
    if not isinstance(receipt, dict):
        raise CertificateError("receipt must be an object")
    if receipt.get("status") not in STATUSES:
        raise CertificateError("receipt status must be pass, fail or inconclusive")
    lane = manifest["proof_lane"]
    if receipt.get("proof_lane") != lane:
        raise CertificateError("receipt proof lane does not match manifest")
    if receipt.get("manifest_sha256") != manifest_digest:
        raise CertificateError("receipt is not bound to this exact manifest")
    if receipt.get("evidence_digests") != manifest["evidence"]:
        raise CertificateError("receipt evidence digests do not match manifest evidence")
    verifier = receipt.get("verifier")
    checks = receipt.get("checks")
    if not isinstance(verifier, dict) or not isinstance(checks, dict):
        raise CertificateError("receipt verifier and checks must be objects")
    if not isinstance(verifier.get("name"), str) or not verifier["name"]:
        raise CertificateError("receipt verifier.name is required")

    if lane == "exact_finite_exchangeability":
        if receipt["status"] == "pass":
            if checks.get("arithmetic") != "rational_exact":
                raise CertificateError("exact finite-exchangeability pass requires rational_exact arithmetic")
            if not isinstance(checks.get("augmented_state_count"), int) or checks["augmented_state_count"] < 1:
                raise CertificateError("exact CRC receipt needs augmented_state_count")
            if not isinstance(checks.get("table_validation_passes"), int) or checks["table_validation_passes"] < 1:
                raise CertificateError("exact CRC receipt needs table_validation_passes")
            excess = parse_exact_fraction(checks.get("max_excess"), "checks.max_excess")
            if excess > 0:
                raise CertificateError("exact CRC receipt cannot pass with positive budget excess")
            if not isinstance(checks.get("worst_augmented_count"), list):
                raise CertificateError("exact CRC receipt needs a worst augmented state witness")
    elif lane == "theorem_backed_eprocess":
        if receipt["status"] == "pass" and not (
            isinstance(checks.get("theorem_reference"), str)
            and checks["theorem_reference"]
            and checks.get("filtration_contract_checked") is True
            and checks.get("optional_stopping_contract_checked") is True
        ):
            raise CertificateError("SupFDR pass requires theorem and filtration checks")
    elif lane == "high_probability_bound":
        if receipt["status"] == "pass":
            if not checks.get("confidence_bound"):
                raise CertificateError("high-probability pass needs a confidence bound receipt")
            upper = parse_exact_fraction(checks["confidence_bound"], "checks.confidence_bound")
            target = parse_exact_fraction(manifest["risk_target"]["level"], "risk_target.level")
            if upper > target:
                raise CertificateError("high-probability upper bound exceeds target")

    if lane == "simulation_only" and receipt["status"] == "pass":
        raise CertificateError("simulation cannot produce a passing risk certificate")


def build_certificate(manifest: dict, receipt_without_binding: dict) -> dict:
    """Bind the receipt to the canonical manifest then validate the full envelope."""
    digest = sha256_json(manifest)
    receipt = dict(receipt_without_binding)
    receipt["manifest_sha256"] = digest
    certificate = {
        "schema": SCHEMA,
        "manifest": manifest,
        "manifest_sha256": digest,
        "receipt": receipt,
    }
    verify_certificate(certificate)
    return certificate


def verify_certificate(certificate: dict) -> dict:
    if not isinstance(certificate, dict) or certificate.get("schema") != SCHEMA:
        raise CertificateError("unsupported certificate envelope")
    manifest = certificate.get("manifest")
    receipt = certificate.get("receipt")
    if not isinstance(manifest, dict):
        raise CertificateError("certificate manifest is missing")
    _validate_manifest(manifest)
    digest = sha256_json(manifest)
    if certificate.get("manifest_sha256") != digest:
        raise CertificateError("manifest digest mismatch")
    _validate_receipt(manifest, digest, receipt)
    return {
        "valid": receipt["status"] == "pass",
        "manifest_sha256": digest,
        "risk_kind": manifest["risk_target"]["kind"],
        "proof_lane": manifest["proof_lane"],
        "receipt_status": receipt["status"],
    }
