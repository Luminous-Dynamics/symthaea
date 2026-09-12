#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import tempfile
from typing import Any

POLICY_SCHEMA = "ll009o.uncertainty-semantics-policy.v1"
LOCK_SCHEMA = "ll009n.nasa-source-lock.v1"
L_SCHEMA = "ll009l.terrain-sample-pack.v1"
M_SCHEMA = "ll009m.radial-uncertainty-receipt.v1"
RECEIPT_SCHEMA = "ll009o.uncertainty-semantics-receipt.v1"
SEMANTICS = {"hard_upper_bound", "rms_error", "empirical_ensemble", "unknown"}
CLAIMS = {"deterministic_upper_bound", "risk_qualified", "descriptive_only"}


class OError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, separators=(",", ": ")) + "\n").encode()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_obj(path: pathlib.Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise OError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise OError(f"{label} must contain a JSON object")
    return value


def require_hex(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise OError(f"{label} must be lowercase SHA-256 hex")
    return value


def validate_policy(policy: dict) -> None:
    if policy.get("schema_version") != POLICY_SCHEMA:
        raise OError(f"policy schema must be {POLICY_SCHEMA}")
    if not isinstance(policy.get("study_id"), str) or not policy["study_id"]:
        raise OError("policy study_id required")
    if policy.get("requested_claim_class") not in CLAIMS:
        raise OError(f"requested_claim_class must be one of {sorted(CLAIMS)}")

    site = policy.get("site_vertical_uncertainty")
    if not isinstance(site, dict) or site.get("semantics_class") not in SEMANTICS:
        raise OError("site_vertical_uncertainty semantics_class invalid")
    if site.get("source_id") is not None and not isinstance(site.get("source_id"), str):
        raise OError("site source_id must be string or null")
    if not isinstance(site.get("evidence_basis"), str) or not site["evidence_basis"]:
        raise OError("site_vertical_uncertainty evidence_basis required")

    layers = policy.get("layers")
    if not isinstance(layers, list) or not layers:
        raise OError("policy layers required")
    seen = set()
    for item in layers:
        if not isinstance(item, dict) or not isinstance(item.get("layer_id"), str) or not item["layer_id"]:
            raise OError("each policy layer needs layer_id")
        if item["layer_id"] in seen:
            raise OError(f"duplicate policy layer {item['layer_id']}")
        seen.add(item["layer_id"])
        if item.get("semantics_class") not in SEMANTICS:
            raise OError(f"{item['layer_id']}: invalid semantics_class")
        if not isinstance(item.get("uncertainty_source_id"), str) or not item["uncertainty_source_id"]:
            raise OError(f"{item['layer_id']}: uncertainty_source_id required")
        if not isinstance(item.get("evidence_basis"), str) or not item["evidence_basis"]:
            raise OError(f"{item['layer_id']}: evidence_basis required")


def source_index(lock: dict) -> dict[str, dict]:
    if lock.get("schema_version") != LOCK_SCHEMA:
        raise OError(f"source lock schema must be {LOCK_SCHEMA}")
    files = lock.get("files")
    if not isinstance(files, list) or not files:
        raise OError("source lock files missing")
    result = {}
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("source_id"), str):
            raise OError("invalid source lock entry")
        source_id = entry["source_id"]
        if source_id in result:
            raise OError(f"duplicate source lock id {source_id}")
        require_hex(entry.get("sha256"), f"source {source_id} sha256")
        result[source_id] = entry
    return result


def l_layer_index(pack: dict) -> dict[str, dict]:
    if pack.get("schema_version") != L_SCHEMA:
        raise OError(f"L pack schema must be {L_SCHEMA}")
    layers = pack.get("layers")
    if not isinstance(layers, list) or not layers:
        raise OError("L pack layers missing")
    result = {}
    for layer in layers:
        layer_id = layer.get("layer_id")
        if not isinstance(layer_id, str) or layer_id in result:
            raise OError("invalid/duplicate L layer_id")
        require_hex(
            layer.get("uncertainty_source_sha256"),
            f"L layer {layer_id} uncertainty hash",
        )
        result[layer_id] = layer
    return result


def verify_m(m_receipt: dict, l_pack_path: pathlib.Path, l_pack: dict) -> dict[str, dict]:
    if m_receipt.get("schema_version") != M_SCHEMA or m_receipt.get("status") != "pass":
        raise OError("M receipt missing/pass schema mismatch")
    if m_receipt.get("l_pack_sha256") != sha256_file(l_pack_path):
        raise OError("M receipt does not bind exact L pack")
    if m_receipt.get("study_id") != l_pack.get("study_id"):
        raise OError("M/L study mismatch")
    layers = m_receipt.get("layers")
    if not isinstance(layers, list) or not layers:
        raise OError("M receipt layers missing")
    result = {}
    for layer in layers:
        layer_id = layer.get("layer_id")
        if not isinstance(layer_id, str) or layer_id in result:
            raise OError("invalid/duplicate M layer_id")
        require_hex(
            layer.get("uncertainty_source_sha256"),
            f"M layer {layer_id} uncertainty hash",
        )
        result[layer_id] = layer
    return result


def classify(classes: list[str]) -> tuple[bool, bool, str]:
    if any(value == "unknown" for value in classes):
        return False, False, "insufficient_uncertainty_semantics"
    if all(value == "hard_upper_bound" for value in classes):
        return True, True, "deterministic_upper_bound_eligible"
    if any(value in {"rms_error", "empirical_ensemble"} for value in classes):
        return False, True, "risk_qualified_only"
    return False, False, "insufficient_uncertainty_semantics"


def run(
    policy_path: pathlib.Path,
    source_lock_path: pathlib.Path,
    l_pack_path: pathlib.Path,
    m_receipt_path: pathlib.Path,
) -> dict:
    policy = read_obj(policy_path, "policy")
    source_lock = read_obj(source_lock_path, "source lock")
    l_pack = read_obj(l_pack_path, "L pack")
    m_receipt = read_obj(m_receipt_path, "M receipt")
    validate_policy(policy)

    sources = source_index(source_lock)
    l_layers = l_layer_index(l_pack)
    m_layers = verify_m(m_receipt, l_pack_path, l_pack)
    if policy["study_id"] != l_pack.get("study_id") or policy["study_id"] != source_lock.get("study_id"):
        raise OError("policy/source-lock/L study mismatch")

    policy_layers = {item["layer_id"]: item for item in policy["layers"]}
    if set(policy_layers) != set(l_layers) or set(m_layers) != set(l_layers):
        raise OError("policy/L/M layer sets differ")

    classes = []
    output_layers = []
    for layer_id in sorted(l_layers):
        semantic = policy_layers[layer_id]
        source = sources.get(semantic["uncertainty_source_id"])
        if source is None:
            raise OError(f"{layer_id}: uncertainty source id absent from exact source lock")
        source_hash = require_hex(source.get("sha256"), f"{layer_id} source lock hash")
        l_hash = l_layers[layer_id]["uncertainty_source_sha256"]
        m_hash = m_layers[layer_id]["uncertainty_source_sha256"]
        if source_hash != l_hash or source_hash != m_hash:
            raise OError(
                f"{layer_id}: uncertainty source hash does not bind N/L/M to same bytes"
            )
        if source.get("role") not in {
            "elevation_rms_uncertainty_m",
            "surface_height_error_m",
            "vertical_uncertainty_m",
        }:
            raise OError(
                f"{layer_id}: locked source role is not a recognized vertical-uncertainty role"
            )
        semantics_class = semantic["semantics_class"]
        classes.append(semantics_class)
        output_layers.append(
            {
                "layer_id": layer_id,
                "semantics_class": semantics_class,
                "uncertainty_source_id": semantic["uncertainty_source_id"],
                "uncertainty_source_sha256": source_hash,
                "evidence_basis": semantic["evidence_basis"],
                "deterministic_upper_bound_eligible": semantics_class == "hard_upper_bound",
                "requires_statistical_or_ensemble_closure": semantics_class
                in {"rms_error", "empirical_ensemble"},
            }
        )

    site = policy["site_vertical_uncertainty"]
    site_class = site["semantics_class"]
    if site.get("source_id"):
        source = sources.get(site["source_id"])
        if source is None:
            raise OError("site uncertainty source absent from exact source lock")
        site_hash = require_hex(source.get("sha256"), "site uncertainty source hash")
    else:
        site_hash = None
        if site_class != "unknown":
            raise OError(
                "site uncertainty semantics require source_id unless semantics_class=unknown"
            )
    classes.append(site_class)

    deterministic, risk_eligible, classification = classify(classes)
    requested = policy["requested_claim_class"]
    if requested == "deterministic_upper_bound":
        allowed = deterministic
    elif requested == "risk_qualified":
        allowed = risk_eligible
    else:
        allowed = True

    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": "pass" if allowed else "blocked",
        "study_id": policy["study_id"],
        "requested_claim_class": requested,
        "classification": classification,
        "requested_claim_allowed": allowed,
        "deterministic_upper_bound_eligible": deterministic,
        "risk_qualified_horizon_eligible": risk_eligible,
        "policy_sha256": sha256_file(policy_path),
        "source_lock_sha256": sha256_file(source_lock_path),
        "l_pack_sha256": sha256_file(l_pack_path),
        "m_receipt_sha256": sha256_file(m_receipt_path),
        "site_vertical_uncertainty": {
            "semantics_class": site_class,
            "source_id": site.get("source_id"),
            "source_sha256": site_hash,
            "evidence_basis": site["evidence_basis"],
            "deterministic_upper_bound_eligible": site_class == "hard_upper_bound",
        },
        "layers": output_layers,
        "claim_rule": (
            "Unqualified deterministic terrain-upper-bound claims require hard_upper_bound "
            "semantics for every terrain and site uncertainty input. RMS and empirical "
            "ensembles require a separately executed statistical/ensemble closure."
        ),
        "non_claims": [
            "LL-009O classifies uncertainty meaning; it does not turn RMS error into a hard bound.",
            "A risk-qualified classification does not choose a confidence level, tail model, familywise sky risk, or clone-ensemble statistic.",
            "This receipt does not close unresolved/subpixel terrain between raster support points.",
        ],
    }
    receipt["receipt_sha256"] = sha256_bytes(canonical_bytes(receipt))
    return receipt


def write_immutable(path: pathlib.Path, value: dict) -> None:
    payload = canonical_bytes(value)
    if path.exists():
        if path.read_bytes() != payload:
            raise OError(f"refusing to overwrite differing immutable output {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def self_test() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        uncertainty_hash = hashlib.sha256(b"rms-source").hexdigest()
        source_lock = {
            "schema_version": LOCK_SCHEMA,
            "study_id": "study-o",
            "files": [
                {
                    "source_id": "uncertainty",
                    "role": "elevation_rms_uncertainty_m",
                    "sha256": uncertainty_hash,
                }
            ],
        }
        l_pack = {
            "schema_version": L_SCHEMA,
            "study_id": "study-o",
            "layers": [
                {
                    "layer_id": "near",
                    "uncertainty_source_sha256": uncertainty_hash,
                }
            ],
        }
        l_pack_path = root / "l-pack.json"
        l_pack_path.write_bytes(canonical_bytes(l_pack))
        m_receipt = {
            "schema_version": M_SCHEMA,
            "status": "pass",
            "study_id": "study-o",
            "l_pack_sha256": sha256_file(l_pack_path),
            "layers": [
                {
                    "layer_id": "near",
                    "uncertainty_source_sha256": uncertainty_hash,
                }
            ],
        }
        policy = {
            "schema_version": POLICY_SCHEMA,
            "study_id": "study-o",
            "requested_claim_class": "deterministic_upper_bound",
            "site_vertical_uncertainty": {
                "semantics_class": "rms_error",
                "source_id": "uncertainty",
                "evidence_basis": "synthetic RMS",
            },
            "layers": [
                {
                    "layer_id": "near",
                    "uncertainty_source_id": "uncertainty",
                    "semantics_class": "rms_error",
                    "evidence_basis": "synthetic RMS",
                }
            ],
        }
        lock_path = root / "source-lock.json"
        m_path = root / "m-receipt.json"
        policy_path = root / "policy.json"
        lock_path.write_bytes(canonical_bytes(source_lock))
        m_path.write_bytes(canonical_bytes(m_receipt))
        policy_path.write_bytes(canonical_bytes(policy))

        blocked = run(policy_path, lock_path, l_pack_path, m_path)
        assert blocked["status"] == "blocked"
        assert not blocked["deterministic_upper_bound_eligible"]

        policy["requested_claim_class"] = "risk_qualified"
        policy_path.write_bytes(canonical_bytes(policy))
        risk = run(policy_path, lock_path, l_pack_path, m_path)
        assert risk["status"] == "pass"
        assert risk["classification"] == "risk_qualified_only"

        policy["requested_claim_class"] = "deterministic_upper_bound"
        policy["layers"][0]["semantics_class"] = "hard_upper_bound"
        policy["site_vertical_uncertainty"]["semantics_class"] = "hard_upper_bound"
        policy_path.write_bytes(canonical_bytes(policy))
        hard = run(policy_path, lock_path, l_pack_path, m_path)
        assert hard["status"] == "pass"
        assert hard["deterministic_upper_bound_eligible"]

        m_receipt["layers"][0]["uncertainty_source_sha256"] = "0" * 64
        m_path.write_bytes(canonical_bytes(m_receipt))
        try:
            run(policy_path, lock_path, l_pack_path, m_path)
        except OError as exc:
            assert "same bytes" in str(exc)
        else:
            raise OError("self-test expected N/L/M hash mismatch rejection")
        print("LL-009O self-test: PASS")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="LL-009O uncertainty-semantics claim guard"
    )
    parser.add_argument("--policy")
    parser.add_argument("--source-lock")
    parser.add_argument("--l-pack")
    parser.add_argument("--m-receipt")
    parser.add_argument("--output")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        if args.self_test:
            self_test()
            return 0
        if not all(
            (args.policy, args.source_lock, args.l_pack, args.m_receipt, args.output)
        ):
            parser.error(
                "--policy --source-lock --l-pack --m-receipt --output required"
            )
        receipt = run(
            pathlib.Path(args.policy),
            pathlib.Path(args.source_lock),
            pathlib.Path(args.l_pack),
            pathlib.Path(args.m_receipt),
        )
        write_immutable(pathlib.Path(args.output), receipt)
        print(json.dumps(receipt, sort_keys=True, indent=2))
        return 0 if receipt["status"] == "pass" else 2
    except (OError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
