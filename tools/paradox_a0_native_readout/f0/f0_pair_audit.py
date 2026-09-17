#!/usr/bin/env python3
"""Label-blind technical-pair auditor for PARADOX A0-R F0.

Consumes exactly two sealed worker-response JSON files. It independently
recomputes raw-channel, scientific-payload, and provenance commitments before
constructing an acyclic pair receipt. No semantic labels, splits, oracle data,
probe state, predictions, or scores are accepted.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from typing import Any

WORKER_SCHEMA = "PARADOX-A0R-F0-WORKER-RESPONSE-V1"
PAIR_SCHEMA = "PARADOX-A0R-F0-TECHNICAL-PAIR-RECEIPT-V1"
SCIENCE_DOMAIN = b"PARADOX-A0R-F0-SCIENTIFIC-V1"
RECEIPT_DOMAIN = b"PARADOX-A0R-F0-RECEIPT-V1"
FEATURE_DOMAIN = b"PARADOX-A0R-F0-FEATURE-BUNDLE-V1"
PAIR_DOMAIN = b"PARADOX-A0R-F0-TECHNICAL-PAIR-V1"

TOP_KEYS = {
    "schema_version",
    "raw_feature_bundle",
    "scientific_payload",
    "scientific_payload_sha256",
    "provenance_envelope",
    "receipt_binding_sha256",
}
RAW_KEYS = {
    "recurrent_f32le_hex",
    "thought_f32le_hex",
    "wisdom_hv_hex",
    "sealed_feature_bundle_sha256",
}
SCIENCE_KEYS = {
    "measurement_cycle_index",
    "recurrent_length",
    "recurrent_f32le_sha256",
    "recurrent_nonfinite_count",
    "recurrent_all_zero",
    "thought_vector_length",
    "thought_vector_f32le_sha256",
    "thought_vector_nonfinite_count",
    "wisdom_hv_byte_length",
    "wisdom_hv_sha256",
    "sealed_feature_bundle_sha256",
    "recurrent_masking_enabled",
    "spectral_entropy_masking_enabled",
    "effective_dim_fraction_override_is_none",
    "measurement_validity",
    "invalidity_reasons",
}
PROVENANCE_KEYS = {
    "production_subject_sha",
    "g2b_subject_sha",
    "a0_subject_sha",
    "m0_subject_sha",
    "f0_subject_sha",
    "opaque_measurement_id",
    "opaque_base_fixture_id",
    "opaque_transform_id",
    "technical_pair_id",
    "technical_replicate_index",
    "service_instance_id",
    "runner_config_projection_sha256",
    "executable_sha256",
    "environment_capsule_sha256",
    "source_binding_sha256",
    "scientific_payload_sha256",
}
FORBIDDEN_FRAGMENTS = (
    "label",
    "condition",
    "expected_response",
    "oracle",
    "capability_atom",
    "score",
    "split",
    "probe",
    "prediction",
    "preprocessing",
    "semantic_fixture",
)


def die(code: str) -> "NoReturn":
    raise ValueError(code)


def exact_keys(obj: dict[str, Any], expected: set[str], where: str) -> None:
    if set(obj) != expected:
        die(f"{where}_KEY_SET_MISMATCH")


def reject_forbidden_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in FORBIDDEN_FRAGMENTS):
                die(f"SEMANTIC_KEY_FORBIDDEN:{key}")
            reject_forbidden_keys(child)
    elif isinstance(value, list):
        for child in value:
            reject_forbidden_keys(child)


def require_hex(value: Any, n: int, code: str) -> str:
    if not isinstance(value, str) or len(value) != n:
        die(code)
    try:
        bytes.fromhex(value)
    except ValueError:
        die(code)
    return value.lower()


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def frame(name: str, value: bytes) -> bytes:
    key = name.encode("utf-8")
    return len(key).to_bytes(4, "little") + key + len(value).to_bytes(8, "little") + value


def domain_hash(domain: bytes, payload: bytes) -> str:
    h = hashlib.sha256()
    h.update(len(domain).to_bytes(8, "little"))
    h.update(domain)
    h.update(len(payload).to_bytes(8, "little"))
    h.update(payload)
    return h.hexdigest()


def hash_frames(domain: bytes, fields: list[tuple[str, bytes]]) -> str:
    return domain_hash(domain, b"".join(frame(name, value) for name, value in fields))


def scientific_bytes(s: dict[str, Any]) -> bytes:
    reasons = s["invalidity_reasons"]
    if not isinstance(reasons, list) or not all(isinstance(x, str) for x in reasons):
        die("INVALIDITY_REASONS_MALFORMED")
    fields: list[tuple[str, bytes]] = [
        ("schema", b"PARADOX-A0R-F0-SCIENTIFIC-PAYLOAD-V1"),
        ("measurement_cycle_index", int(s["measurement_cycle_index"]).to_bytes(8, "little")),
        ("recurrent_length", int(s["recurrent_length"]).to_bytes(8, "little")),
        ("recurrent_sha256", str(s["recurrent_f32le_sha256"]).encode()),
        ("recurrent_nonfinite_count", int(s["recurrent_nonfinite_count"]).to_bytes(8, "little")),
        ("recurrent_all_zero", bytes([1 if s["recurrent_all_zero"] else 0])),
        ("thought_length", int(s["thought_vector_length"]).to_bytes(8, "little")),
        ("thought_sha256", str(s["thought_vector_f32le_sha256"]).encode()),
        ("thought_nonfinite_count", int(s["thought_vector_nonfinite_count"]).to_bytes(8, "little")),
        ("wisdom_byte_length", int(s["wisdom_hv_byte_length"]).to_bytes(8, "little")),
        ("wisdom_sha256", str(s["wisdom_hv_sha256"]).encode()),
        ("feature_bundle_sha256", str(s["sealed_feature_bundle_sha256"]).encode()),
        ("recurrent_masking_enabled", bytes([1 if s["recurrent_masking_enabled"] else 0])),
        ("spectral_entropy_masking_enabled", bytes([1 if s["spectral_entropy_masking_enabled"] else 0])),
        ("effective_dim_fraction_override_is_none", bytes([1 if s["effective_dim_fraction_override_is_none"] else 0])),
        ("measurement_validity", str(s["measurement_validity"]).encode()),
    ]
    fields.extend(("invalidity_reason", reason.encode()) for reason in reasons)
    return b"".join(frame(name, value) for name, value in fields)


def provenance_bytes(p: dict[str, Any]) -> bytes:
    ordered = [
        ("schema", "PARADOX-A0R-F0-PROVENANCE-ENVELOPE-V1"),
        ("production_subject_sha", p["production_subject_sha"]),
        ("g2b_subject_sha", p["g2b_subject_sha"]),
        ("a0_subject_sha", p["a0_subject_sha"]),
        ("m0_subject_sha", p["m0_subject_sha"]),
        ("f0_subject_sha", p["f0_subject_sha"]),
        ("opaque_measurement_id", p["opaque_measurement_id"]),
        ("opaque_base_fixture_id", p["opaque_base_fixture_id"]),
        ("opaque_transform_id", p["opaque_transform_id"]),
        ("technical_pair_id", p["technical_pair_id"]),
        ("service_instance_id", p["service_instance_id"]),
        ("runner_config_projection_sha256", p["runner_config_projection_sha256"]),
        ("executable_sha256", p["executable_sha256"]),
        ("environment_capsule_sha256", p["environment_capsule_sha256"]),
        ("source_binding_sha256", p["source_binding_sha256"]),
        ("scientific_payload_sha256", p["scientific_payload_sha256"]),
    ]
    out = b"".join(frame(name, str(value).encode()) for name, value in ordered)
    out += frame("technical_replicate_index", int(p["technical_replicate_index"]).to_bytes(8, "little"))
    return out


def verify_worker_receipt(doc: Any) -> dict[str, Any]:
    if not isinstance(doc, dict):
        die("WORKER_RECEIPT_NOT_OBJECT")
    reject_forbidden_keys(doc)
    exact_keys(doc, TOP_KEYS, "TOP")
    if doc["schema_version"] != WORKER_SCHEMA:
        die("WORKER_SCHEMA_MISMATCH")

    raw = doc["raw_feature_bundle"]
    science = doc["scientific_payload"]
    prov = doc["provenance_envelope"]
    if not all(isinstance(x, dict) for x in (raw, science, prov)):
        die("WORKER_SUBOBJECT_TYPE_FAILURE")
    exact_keys(raw, RAW_KEYS, "RAW")
    exact_keys(science, SCIENCE_KEYS, "SCIENCE")
    exact_keys(prov, PROVENANCE_KEYS, "PROVENANCE")

    recurrent = bytes.fromhex(raw["recurrent_f32le_hex"])
    thought = bytes.fromhex(raw["thought_f32le_hex"])
    wisdom = bytes.fromhex(raw["wisdom_hv_hex"])
    if len(recurrent) != int(science["recurrent_length"]) * 4:
        die("RAW_RECURRENT_LENGTH_MISMATCH")
    if len(thought) != int(science["thought_vector_length"]) * 4:
        die("RAW_THOUGHT_LENGTH_MISMATCH")
    if len(wisdom) != int(science["wisdom_hv_byte_length"]):
        die("RAW_WISDOM_LENGTH_MISMATCH")
    if sha256_hex(recurrent) != require_hex(science["recurrent_f32le_sha256"], 64, "RECURRENT_DIGEST_INVALID"):
        die("RAW_RECURRENT_DIGEST_MISMATCH")
    if sha256_hex(thought) != require_hex(science["thought_vector_f32le_sha256"], 64, "THOUGHT_DIGEST_INVALID"):
        die("RAW_THOUGHT_DIGEST_MISMATCH")
    if sha256_hex(wisdom) != require_hex(science["wisdom_hv_sha256"], 64, "WISDOM_DIGEST_INVALID"):
        die("RAW_WISDOM_DIGEST_MISMATCH")

    feature_hash = hash_frames(
        FEATURE_DOMAIN,
        [("recurrent_f32le", recurrent), ("thought_f32le", thought), ("wisdom_hv", wisdom)],
    )
    if feature_hash != require_hex(raw["sealed_feature_bundle_sha256"], 64, "RAW_FEATURE_DIGEST_INVALID"):
        die("RAW_FEATURE_BUNDLE_DIGEST_MISMATCH")
    if feature_hash != science["sealed_feature_bundle_sha256"]:
        die("SCIENCE_FEATURE_BUNDLE_DIGEST_MISMATCH")

    science_hash = domain_hash(SCIENCE_DOMAIN, scientific_bytes(science))
    if science_hash != require_hex(doc["scientific_payload_sha256"], 64, "SCIENCE_DIGEST_INVALID"):
        die("SCIENTIFIC_PAYLOAD_DIGEST_MISMATCH")
    if science_hash != prov["scientific_payload_sha256"]:
        die("PROVENANCE_SCIENCE_BINDING_MISMATCH")

    receipt_hash = domain_hash(RECEIPT_DOMAIN, provenance_bytes(prov))
    if receipt_hash != require_hex(doc["receipt_binding_sha256"], 64, "RECEIPT_DIGEST_INVALID"):
        die("PROVENANCE_RECEIPT_DIGEST_MISMATCH")
    return doc


def pair_receipt(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    receipts = sorted([a, b], key=lambda r: int(r["provenance_envelope"]["technical_replicate_index"]))
    p0, p1 = (r["provenance_envelope"] for r in receipts)
    indices = [int(p0["technical_replicate_index"]), int(p1["technical_replicate_index"])]
    reasons: list[str] = []
    if indices != [0, 1]:
        reasons.append("REPLICATE_INDEX_SET_INVALID")
    if p0["technical_pair_id"] != p1["technical_pair_id"]:
        reasons.append("TECHNICAL_PAIR_ID_MISMATCH")
    if p0["service_instance_id"] == p1["service_instance_id"]:
        reasons.append("SERVICE_INSTANCE_REUSE")
    if a["scientific_payload_sha256"] != b["scientific_payload_sha256"]:
        reasons.append("SCIENTIFIC_PAYLOAD_MISMATCH")
    if a["receipt_binding_sha256"] == b["receipt_binding_sha256"]:
        reasons.append("PROVENANCE_BINDING_NOT_DISTINCT")
    if a["scientific_payload"]["measurement_validity"] != "ELIGIBLE_F0_TRANSPORT" or b["scientific_payload"]["measurement_validity"] != "ELIGIBLE_F0_TRANSPORT":
        reasons.append("PEER_INVALID")

    status = "ELIGIBLE_F0_TECHNICAL_PAIR" if not reasons else "INVALID_MEASUREMENT_NONDETERMINISTIC"
    pair = {
        "schema_version": PAIR_SCHEMA,
        "technical_pair_id": p0["technical_pair_id"] if p0["technical_pair_id"] == p1["technical_pair_id"] else None,
        "replicate_indices": indices,
        "worker_receipt_binding_sha256": [receipts[0]["receipt_binding_sha256"], receipts[1]["receipt_binding_sha256"]],
        "service_instance_ids": [p0["service_instance_id"], p1["service_instance_id"]],
        "scientific_payload_sha256": [receipts[0]["scientific_payload_sha256"], receipts[1]["scientific_payload_sha256"]],
        "status": status,
        "invalidity_reasons": reasons,
        "statistical_n_increment": 0,
    }
    canonical = json.dumps(pair, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    pair["pair_receipt_sha256"] = domain_hash(PAIR_DOMAIN, canonical)
    return pair


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: f0_pair_audit.py REPLICATE0.json REPLICATE1.json", file=sys.stderr)
        return 2
    try:
        docs = [json.loads(pathlib.Path(path).read_text(encoding="utf-8")) for path in argv[1:]]
        verified = [verify_worker_receipt(doc) for doc in docs]
        receipt = pair_receipt(*verified)
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
        return 0 if receipt["status"] == "ELIGIBLE_F0_TECHNICAL_PAIR" else 3
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(json.dumps({"schema_version": PAIR_SCHEMA, "status": "INVALID_MEASUREMENT_NONDETERMINISTIC", "invalidity_reasons": [str(exc)]}, sort_keys=True, separators=(",", ":")))
        return 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
