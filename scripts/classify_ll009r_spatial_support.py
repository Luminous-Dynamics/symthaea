#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any

POLICY_SCHEMA = "ll009r.spatial-support-policy.v1"
AUDIT_SCHEMA = "ll009r.nested-resolution-audit-receipt.v1"
RECEIPT_SCHEMA = "ll009r.spatial-support-classification-receipt.v1"

SUPPORT_CLASSES = {
    "continuous_hard_bound",
    "empirical_multiscale_bound",
    "resolution_qualified",
    "sample_points_only",
    "unknown",
}
CLAIM_CLASSES = {
    "continuous_hard_bound",
    "risk_qualified_horizon",
    "empirical_sampled_horizon",
    "descriptive_only",
}


class RError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n"
    ).encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_json(path: pathlib.Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise RError(f"{label} must contain object")
    return value


def validate_policy(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != POLICY_SCHEMA:
        raise RError(f"schema_version must be {POLICY_SCHEMA}")
    policy = dict(value)
    for key in ("study_id", "requested_claim_class"):
        if not isinstance(policy.get(key), str) or not policy[key]:
            raise RError(f"missing {key}")
    if policy["requested_claim_class"] not in CLAIM_CLASSES:
        raise RError("unsupported requested_claim_class")
    layers = policy.get("layers")
    if not isinstance(layers, list) or not layers:
        raise RError("layers required")
    seen = set()
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise RError(f"layers[{index}] must be object")
        layer_id = layer.get("layer_id")
        support_class = layer.get("support_class")
        basis = layer.get("evidence_basis")
        if not isinstance(layer_id, str) or not layer_id:
            raise RError(f"layers[{index}] missing layer_id")
        if layer_id in seen:
            raise RError(f"duplicate layer_id {layer_id}")
        if support_class not in SUPPORT_CLASSES:
            raise RError(f"{layer_id}: unsupported support_class")
        if not isinstance(basis, str) or not basis:
            raise RError(f"{layer_id}: evidence_basis required")
        seen.add(layer_id)
    return policy


def validate_audit(audit: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    if audit.get("schema_version") != AUDIT_SCHEMA or audit.get("status") != "pass":
        raise RError("nested audit receipt must be passing LL-009R receipt")
    if audit.get("study_id") != policy["study_id"]:
        raise RError("nested audit study mismatch")
    coarse = audit.get("coarse_layer_id")
    fine = audit.get("fine_layer_id")
    if not isinstance(coarse, str) or not isinstance(fine, str):
        raise RError("nested audit layer IDs required")
    if coarse == fine:
        raise RError("nested audit coarse/fine layer IDs must differ")
    if audit.get("support_semantics") != "empirical_multiscale_bound":
        raise RError("nested audit must retain empirical_multiscale_bound semantics")
    return audit


def classify(
    policy: dict[str, Any], audit: dict[str, Any] | None = None
) -> dict[str, Any]:
    policy = validate_policy(policy)
    layers = [dict(item) for item in policy["layers"]]
    audit_applied = False

    if audit is not None:
        audit = validate_audit(audit, policy)
        coarse_id = audit["coarse_layer_id"]
        fine_id = audit["fine_layer_id"]
        ids = {item["layer_id"] for item in layers}
        if coarse_id not in ids or fine_id not in ids:
            raise RError("nested audit references layer absent from policy")
        for layer in layers:
            if layer["layer_id"] == coarse_id:
                original = layer["support_class"]
                if original == "unknown":
                    raise RError("cannot upgrade unknown spatial support from empirical audit")
                if original != "continuous_hard_bound":
                    layer["support_class_before_audit"] = original
                    layer["support_class"] = "empirical_multiscale_bound"
                    layer["nested_audit_sha256"] = audit["receipt_sha256"]
                    layer["observed_positive_excursion_margin_deg"] = audit[
                        "max_positive_fine_minus_coarse_deg"
                    ]
                    audit_applied = True

    classes = {layer["support_class"] for layer in layers}
    requested = policy["requested_claim_class"]
    if requested == "continuous_hard_bound":
        allowed = classes == {"continuous_hard_bound"}
        reason = (
            "all admitted layers provide continuous hard support bounds"
            if allowed
            else "continuous hard horizon requires continuous_hard_bound support on every admitted layer"
        )
    elif requested == "risk_qualified_horizon":
        allowed = "unknown" not in classes and "sample_points_only" not in classes
        reason = (
            "all layers have at least resolution-qualified or empirical/hard support"
            if allowed
            else "risk-qualified horizon blocks unknown and sample-points-only spatial support"
        )
    elif requested == "empirical_sampled_horizon":
        allowed = "unknown" not in classes
        reason = (
            "all layers have explicit spatial-support semantics"
            if allowed
            else "empirical sampled horizon blocks unknown spatial support"
        )
    else:
        allowed = True
        reason = "descriptive-only claim is permitted with explicit support-class disclosure"

    strongest_common = "unknown"
    if classes == {"continuous_hard_bound"}:
        strongest_common = "continuous_hard_bound"
    elif "unknown" not in classes and "sample_points_only" not in classes:
        if "resolution_qualified" in classes:
            strongest_common = "resolution_qualified"
        else:
            strongest_common = "empirical_multiscale_bound"
    elif "unknown" not in classes:
        strongest_common = "sample_points_only"

    result = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "pass" if allowed else "blocked",
        "study_id": policy["study_id"],
        "requested_claim_class": requested,
        "claim_allowed": allowed,
        "reason": reason,
        "strongest_common_spatial_support": strongest_common,
        "layers": layers,
        "audit_applied": audit_applied,
        "policy_sha256": sha256_bytes(canonical_bytes(policy)),
        "non_claims": [
            "Spatial-support classification is independent of vertical/RMS/ensemble uncertainty semantics.",
            "An empirical multiscale audit bounds only observed coarse-vs-fine skyline differences over its exact overlap; it is not a continuous-terrain theorem.",
            "Effective resolution, sampled slope, RMS roughness and Hurst products are not relabeled as hard bounds without a source-supported theorem.",
        ],
    }
    if audit is not None:
        result["audit_receipt_sha256"] = audit["receipt_sha256"]
    result["receipt_sha256"] = sha256_bytes(canonical_bytes(result))
    return result


def write_immutable(path: pathlib.Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise RError(f"refusing to overwrite differing output {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def self_test() -> None:
    policy = {
        "schema_version": POLICY_SCHEMA,
        "study_id": "study-r",
        "requested_claim_class": "continuous_hard_bound",
        "layers": [
            {
                "layer_id": "near",
                "support_class": "sample_points_only",
                "evidence_basis": "synthetic",
            },
            {
                "layer_id": "far",
                "support_class": "resolution_qualified",
                "evidence_basis": "synthetic",
            },
        ],
    }
    blocked = classify(policy)
    assert blocked["status"] == "blocked"
    assert blocked["strongest_common_spatial_support"] == "sample_points_only"

    audit = {
        "schema_version": AUDIT_SCHEMA,
        "status": "pass",
        "study_id": "study-r",
        "coarse_layer_id": "far",
        "fine_layer_id": "near",
        "support_semantics": "empirical_multiscale_bound",
        "max_positive_fine_minus_coarse_deg": 1.25,
    }
    audit["receipt_sha256"] = sha256_bytes(canonical_bytes(audit))
    still_blocked = classify(policy, audit)
    assert still_blocked["status"] == "blocked"
    far = [x for x in still_blocked["layers"] if x["layer_id"] == "far"][0]
    assert far["support_class"] == "empirical_multiscale_bound"

    risk = json.loads(json.dumps(policy))
    risk["requested_claim_class"] = "risk_qualified_horizon"
    assert classify(risk, audit)["status"] == "blocked"

    empirical = json.loads(json.dumps(policy))
    empirical["requested_claim_class"] = "empirical_sampled_horizon"
    assert classify(empirical, audit)["status"] == "pass"

    hard = json.loads(json.dumps(policy))
    hard["layers"] = [
        {
            "layer_id": "near",
            "support_class": "continuous_hard_bound",
            "evidence_basis": "synthetic theorem",
        },
        {
            "layer_id": "far",
            "support_class": "continuous_hard_bound",
            "evidence_basis": "synthetic theorem",
        },
    ]
    assert classify(hard)["status"] == "pass"
    print("LL-009R spatial-support classifier self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description="LL-009R spatial-support claim classifier")
    parser.add_argument("--policy")
    parser.add_argument("--audit")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        if not args.policy or not args.output:
            raise RError("--policy and --output are required")
        policy = safe_json(pathlib.Path(args.policy), "policy")
        audit = safe_json(pathlib.Path(args.audit), "audit") if args.audit else None
        receipt = classify(policy, audit)
        write_immutable(pathlib.Path(args.output), receipt)
        print(json.dumps(receipt, sort_keys=True, indent=2))
        return 0
    except (OSError, json.JSONDecodeError, RError) as exc:
        raise SystemExit(f"LL-009R classifier failure: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
