#!/usr/bin/env python3
"""Independent label-blind technical-pair auditor for PARADOX A0-R F0 V3."""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import struct
import sys
import unicodedata
import uuid
from typing import Any

WORKER_SCHEMA = "PARADOX-A0R-F0-WORKER-RESPONSE-V3"
PAIR_SCHEMA = "PARADOX-A0R-F0-TECHNICAL-PAIR-RECEIPT-V2"
BINDING_SCHEMA = "PARADOX-A0R-F0-EXECUTION-BINDING-V1"
SCIENCE_DOMAIN = b"PARADOX-A0R-F0-SCIENTIFIC-V1"
RECEIPT_DOMAIN = b"PARADOX-A0R-F0-RECEIPT-V2"
FEATURE_DOMAIN = b"PARADOX-A0R-F0-FEATURE-BUNDLE-V1"
PAIR_DOMAIN = b"PARADOX-A0R-F0-TECHNICAL-PAIR-V2"
BINDING_DOMAIN = b"PARADOX-A0R-F0-EXECUTION-BINDING-V1"
PRODUCTION_SHA = "eb73527d05a913e79d1f05135ad6b06c1da8e2ee"
G2B_SHA = "09d83a1d1fddbbbd30e4eba7cc95946c8eab871f"
A0_SHA = "8cc2651576f0068a1ccac3ef21c8a0a0eb3c2afb"
M0_SHA = "306b0471a0854d2e77c1d50438fc493806fe1564"
CONFIG_SHA = "2cd99b06cf2cf11cfc0612c0db818355099b48b04bd687ca66a31a08680be1f9"
U64_MAX = (1 << 64) - 1

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
    "agent_visible_event_count",
    "agent_visible_events_sha256",
    "runner_config_projection_sha256",
    "executable_sha256",
    "environment_capsule_sha256",
    "source_binding_sha256",
    "scientific_payload_sha256",
}
BINDING_KEYS = {
    "schema_version",
    "f0_subject_sha",
    "opaque_measurement_id",
    "opaque_base_fixture_id",
    "opaque_transform_id",
    "technical_pair_id",
    "agent_visible_event_count",
    "agent_visible_events_sha256",
    "measurement_cycle_index",
    "runner_config_projection_sha256",
    "executable_sha256",
    "environment_capsule_sha256",
    "source_binding_sha256",
}
PAIR_INVARIANT_FIELDS = (
    "production_subject_sha",
    "g2b_subject_sha",
    "a0_subject_sha",
    "m0_subject_sha",
    "f0_subject_sha",
    "opaque_measurement_id",
    "opaque_base_fixture_id",
    "opaque_transform_id",
    "technical_pair_id",
    "agent_visible_event_count",
    "agent_visible_events_sha256",
    "runner_config_projection_sha256",
    "executable_sha256",
    "environment_capsule_sha256",
    "source_binding_sha256",
    "scientific_payload_sha256",
)


class AuditFailure(ValueError):
    pass


def die(code: str) -> None:
    raise AuditFailure(code)


def reject_float_token(_: str) -> None:
    die("JSON_FLOAT_TOKEN_FORBIDDEN")


def reject_constant(token: str) -> None:
    die(f"JSON_NONFINITE_CONSTANT_FORBIDDEN:{token}")


def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            die(f"JSON_DUPLICATE_KEY:{key}")
        result[key] = value
    return result


def require_nfc(value: Any) -> None:
    if isinstance(value, str):
        if unicodedata.normalize("NFC", value) != value:
            die("JSON_STRING_NOT_NFC")
    elif isinstance(value, list):
        for item in value:
            require_nfc(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            require_nfc(key)
            require_nfc(item)


def strict_json_load(path: pathlib.Path) -> Any:
    text = path.read_text(encoding="utf-8")
    value = json.loads(
        text,
        object_pairs_hook=unique_object,
        parse_float=reject_float_token,
        parse_constant=reject_constant,
    )
    require_nfc(value)
    return value


def exact_keys(obj: dict[str, Any], expected: set[str], where: str) -> None:
    if set(obj) != expected:
        die(f"{where}_KEY_SET_MISMATCH")


def require_str(value: Any, code: str) -> str:
    if type(value) is not str:
        die(code)
    return value


def require_bool(value: Any, code: str) -> bool:
    if type(value) is not bool:
        die(code)
    return value


def require_uint(value: Any, code: str) -> int:
    if type(value) is not int or not 0 <= value <= U64_MAX:
        die(code)
    return value


def require_lower_hex(value: Any, length: int, code: str) -> str:
    value = require_str(value, code)
    if len(value) != length or any(ch not in "0123456789abcdef" for ch in value):
        die(code)
    return value


def require_opaque_id(value: Any, code: str) -> str:
    return require_lower_hex(value, 32, code)


def require_hex_bytes(value: Any, code: str) -> bytes:
    value = require_str(value, code)
    if len(value) % 2 or any(ch not in "0123456789abcdef" for ch in value):
        die(code)
    return bytes.fromhex(value)


def require_uuid4(value: Any) -> str:
    value = require_str(value, "SERVICE_INSTANCE_ID_INVALID")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise AuditFailure("SERVICE_INSTANCE_ID_INVALID") from exc
    if parsed.version != 4 or str(parsed) != value:
        die("SERVICE_INSTANCE_ID_INVALID")
    return value


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


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def decode_f32le(raw: bytes, code: str) -> list[float]:
    if len(raw) % 4:
        die(code)
    return [struct.unpack("<f", raw[i : i + 4])[0] for i in range(0, len(raw), 4)]


def validate_binding(binding: Any) -> dict[str, Any]:
    if type(binding) is not dict:
        die("EXECUTION_BINDING_NOT_OBJECT")
    exact_keys(binding, BINDING_KEYS, "EXECUTION_BINDING")
    if require_str(binding["schema_version"], "BINDING_SCHEMA_TYPE") != BINDING_SCHEMA:
        die("BINDING_SCHEMA_MISMATCH")
    require_lower_hex(binding["f0_subject_sha"], 40, "BINDING_F0_SHA_INVALID")
    for field in (
        "opaque_measurement_id",
        "opaque_base_fixture_id",
        "opaque_transform_id",
        "technical_pair_id",
    ):
        require_opaque_id(binding[field], f"BINDING_OPAQUE_ID_INVALID:{field}")
    event_count = require_uint(binding["agent_visible_event_count"], "BINDING_EVENT_COUNT_INVALID")
    if event_count == 0:
        die("BINDING_EVENT_COUNT_ZERO")
    require_lower_hex(binding["agent_visible_events_sha256"], 64, "BINDING_EVENT_DIGEST_INVALID")
    measurement_cycle = require_uint(binding["measurement_cycle_index"], "BINDING_MEASUREMENT_CYCLE_INVALID")
    if measurement_cycle != event_count - 1:
        die("BINDING_MEASUREMENT_CYCLE_NOT_FINAL")
    if require_str(binding["runner_config_projection_sha256"], "BINDING_CONFIG_TYPE") != CONFIG_SHA:
        die("BINDING_CONFIG_MISMATCH")
    for field in ("executable_sha256", "environment_capsule_sha256", "source_binding_sha256"):
        require_lower_hex(binding[field], 64, f"BINDING_DIGEST_INVALID:{field}")
    return binding


def scientific_bytes(s: dict[str, Any]) -> bytes:
    fields: list[tuple[str, bytes]] = [
        ("schema", b"PARADOX-A0R-F0-SCIENTIFIC-PAYLOAD-V1"),
        ("measurement_cycle_index", s["measurement_cycle_index"].to_bytes(8, "little")),
        ("recurrent_length", s["recurrent_length"].to_bytes(8, "little")),
        ("recurrent_sha256", s["recurrent_f32le_sha256"].encode()),
        ("recurrent_nonfinite_count", s["recurrent_nonfinite_count"].to_bytes(8, "little")),
        ("recurrent_all_zero", bytes([1 if s["recurrent_all_zero"] else 0])),
        ("thought_length", s["thought_vector_length"].to_bytes(8, "little")),
        ("thought_sha256", s["thought_vector_f32le_sha256"].encode()),
        ("thought_nonfinite_count", s["thought_vector_nonfinite_count"].to_bytes(8, "little")),
        ("wisdom_byte_length", s["wisdom_hv_byte_length"].to_bytes(8, "little")),
        ("wisdom_sha256", s["wisdom_hv_sha256"].encode()),
        ("feature_bundle_sha256", s["sealed_feature_bundle_sha256"].encode()),
        ("recurrent_masking_enabled", bytes([1 if s["recurrent_masking_enabled"] else 0])),
        ("spectral_entropy_masking_enabled", bytes([1 if s["spectral_entropy_masking_enabled"] else 0])),
        ("effective_dim_fraction_override_is_none", bytes([1 if s["effective_dim_fraction_override_is_none"] else 0])),
        ("measurement_validity", s["measurement_validity"].encode()),
    ]
    fields.extend(("invalidity_reason", reason.encode()) for reason in s["invalidity_reasons"])
    return b"".join(frame(name, value) for name, value in fields)


def provenance_bytes(p: dict[str, Any]) -> bytes:
    ordered = [
        ("schema", "PARADOX-A0R-F0-PROVENANCE-ENVELOPE-V2"),
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
        ("agent_visible_events_sha256", p["agent_visible_events_sha256"]),
        ("runner_config_projection_sha256", p["runner_config_projection_sha256"]),
        ("executable_sha256", p["executable_sha256"]),
        ("environment_capsule_sha256", p["environment_capsule_sha256"]),
        ("source_binding_sha256", p["source_binding_sha256"]),
        ("scientific_payload_sha256", p["scientific_payload_sha256"]),
    ]
    out = b"".join(frame(name, value.encode()) for name, value in ordered)
    out += frame("technical_replicate_index", p["technical_replicate_index"].to_bytes(8, "little"))
    out += frame("agent_visible_event_count", p["agent_visible_event_count"].to_bytes(8, "little"))
    return out


def validate_science_types(science: dict[str, Any]) -> None:
    for field in (
        "measurement_cycle_index",
        "recurrent_length",
        "recurrent_nonfinite_count",
        "thought_vector_length",
        "thought_vector_nonfinite_count",
        "wisdom_hv_byte_length",
    ):
        science[field] = require_uint(science[field], f"SCIENCE_UINT_TYPE_INVALID:{field}")
    for field in (
        "recurrent_all_zero",
        "recurrent_masking_enabled",
        "spectral_entropy_masking_enabled",
        "effective_dim_fraction_override_is_none",
    ):
        science[field] = require_bool(science[field], f"SCIENCE_BOOL_TYPE_INVALID:{field}")
    for field in (
        "recurrent_f32le_sha256",
        "thought_vector_f32le_sha256",
        "wisdom_hv_sha256",
        "sealed_feature_bundle_sha256",
    ):
        science[field] = require_lower_hex(science[field], 64, f"SCIENCE_DIGEST_INVALID:{field}")
    science["measurement_validity"] = require_str(science["measurement_validity"], "SCIENCE_VALIDITY_TYPE_INVALID")
    reasons = science["invalidity_reasons"]
    if type(reasons) is not list or not all(type(reason) is str for reason in reasons):
        die("INVALIDITY_REASONS_TYPE_INVALID")


def validate_provenance_types(prov: dict[str, Any]) -> None:
    for field in (
        "production_subject_sha",
        "g2b_subject_sha",
        "a0_subject_sha",
        "m0_subject_sha",
        "f0_subject_sha",
    ):
        prov[field] = require_lower_hex(prov[field], 40, f"PROVENANCE_SHA_INVALID:{field}")
    for field in (
        "opaque_measurement_id",
        "opaque_base_fixture_id",
        "opaque_transform_id",
        "technical_pair_id",
    ):
        prov[field] = require_opaque_id(prov[field], f"PROVENANCE_OPAQUE_ID_INVALID:{field}")
    prov["technical_replicate_index"] = require_uint(prov["technical_replicate_index"], "REPLICATE_INDEX_TYPE_INVALID")
    if prov["technical_replicate_index"] not in (0, 1):
        die("REPLICATE_INDEX_INVALID")
    prov["agent_visible_event_count"] = require_uint(prov["agent_visible_event_count"], "EVENT_COUNT_TYPE_INVALID")
    prov["service_instance_id"] = require_uuid4(prov["service_instance_id"])
    for field in (
        "agent_visible_events_sha256",
        "runner_config_projection_sha256",
        "executable_sha256",
        "environment_capsule_sha256",
        "source_binding_sha256",
        "scientific_payload_sha256",
    ):
        prov[field] = require_lower_hex(prov[field], 64, f"PROVENANCE_DIGEST_INVALID:{field}")


def verify_worker_receipt(doc: Any, binding: dict[str, Any]) -> dict[str, Any]:
    if type(doc) is not dict:
        die("WORKER_RECEIPT_NOT_OBJECT")
    exact_keys(doc, TOP_KEYS, "TOP")
    if require_str(doc["schema_version"], "WORKER_SCHEMA_TYPE") != WORKER_SCHEMA:
        die("WORKER_SCHEMA_MISMATCH")

    raw = doc["raw_feature_bundle"]
    science = doc["scientific_payload"]
    prov = doc["provenance_envelope"]
    if not all(type(obj) is dict for obj in (raw, science, prov)):
        die("WORKER_SUBOBJECT_TYPE_FAILURE")
    exact_keys(raw, RAW_KEYS, "RAW")
    exact_keys(science, SCIENCE_KEYS, "SCIENCE")
    exact_keys(prov, PROVENANCE_KEYS, "PROVENANCE")
    validate_science_types(science)
    validate_provenance_types(prov)

    recurrent = require_hex_bytes(raw["recurrent_f32le_hex"], "RAW_RECURRENT_HEX_INVALID")
    thought = require_hex_bytes(raw["thought_f32le_hex"], "RAW_THOUGHT_HEX_INVALID")
    wisdom = require_hex_bytes(raw["wisdom_hv_hex"], "RAW_WISDOM_HEX_INVALID")
    recurrent_values = decode_f32le(recurrent, "RAW_RECURRENT_ALIGNMENT_INVALID")
    thought_values = decode_f32le(thought, "RAW_THOUGHT_ALIGNMENT_INVALID")

    recurrent_len = len(recurrent_values)
    thought_len = len(thought_values)
    wisdom_len = len(wisdom)
    recurrent_nonfinite = sum(not math.isfinite(value) for value in recurrent_values)
    thought_nonfinite = sum(not math.isfinite(value) for value in thought_values)
    recurrent_all_zero = all(
        recurrent[i : i + 4] == b"\x00\x00\x00\x00"
        for i in range(0, len(recurrent), 4)
    )

    if science["recurrent_length"] != recurrent_len:
        die("RECURRENT_LENGTH_MISMATCH")
    if science["thought_vector_length"] != thought_len:
        die("THOUGHT_LENGTH_MISMATCH")
    if science["wisdom_hv_byte_length"] != wisdom_len:
        die("WISDOM_LENGTH_MISMATCH")
    if science["recurrent_nonfinite_count"] != recurrent_nonfinite:
        die("NONFINITE_COUNT_MISMATCH:RECURRENT")
    if science["thought_vector_nonfinite_count"] != thought_nonfinite:
        die("NONFINITE_COUNT_MISMATCH:THOUGHT")
    if science["recurrent_all_zero"] != recurrent_all_zero:
        die("RECURRENT_ALL_ZERO_MISMATCH")

    if sha256_hex(recurrent) != science["recurrent_f32le_sha256"]:
        die("RAW_RECURRENT_DIGEST_MISMATCH")
    if sha256_hex(thought) != science["thought_vector_f32le_sha256"]:
        die("RAW_THOUGHT_DIGEST_MISMATCH")
    if sha256_hex(wisdom) != science["wisdom_hv_sha256"]:
        die("RAW_WISDOM_DIGEST_MISMATCH")

    feature_hash = hash_frames(
        FEATURE_DOMAIN,
        [("recurrent_f32le", recurrent), ("thought_f32le", thought), ("wisdom_hv", wisdom)],
    )
    raw_feature_hash = require_lower_hex(
        raw["sealed_feature_bundle_sha256"], 64, "RAW_FEATURE_DIGEST_INVALID"
    )
    if feature_hash != raw_feature_hash or feature_hash != science["sealed_feature_bundle_sha256"]:
        die("RAW_FEATURE_BUNDLE_DIGEST_MISMATCH")

    if science["recurrent_masking_enabled"] is not False:
        die("CONFIG_DIAGNOSTIC_DRIFT:RECURRENT_MASK")
    if science["spectral_entropy_masking_enabled"] is not False:
        die("CONFIG_DIAGNOSTIC_DRIFT:SPECTRAL_MASK")
    if science["effective_dim_fraction_override_is_none"] is not True:
        die("CONFIG_DIAGNOSTIC_DRIFT:EFFECTIVE_DIM_OVERRIDE")

    expected_reasons: list[str] = []
    if recurrent_len != 256:
        expected_reasons.append("INVALID_MEASUREMENT_DIMENSION_RECURRENT")
    if thought_len != 32:
        expected_reasons.append("INVALID_MEASUREMENT_DIMENSION_THOUGHT")
    if recurrent_nonfinite or thought_nonfinite:
        expected_reasons.append("INVALID_MEASUREMENT_NONFINITE")
    if wisdom_len != 2048:
        expected_reasons.append("INVALID_MEASUREMENT_DIMENSION_WISDOM_HV")
    if science["invalidity_reasons"] != expected_reasons:
        die("INVALIDITY_REASON_MISMATCH")
    expected_validity = "ELIGIBLE_F0_TRANSPORT" if not expected_reasons else "INVALID_F0_TRANSPORT"
    if science["measurement_validity"] != expected_validity:
        die("MEASUREMENT_VALIDITY_MISMATCH")

    science_hash = domain_hash(SCIENCE_DOMAIN, scientific_bytes(science))
    if science_hash != require_lower_hex(doc["scientific_payload_sha256"], 64, "SCIENCE_DIGEST_INVALID"):
        die("SCIENTIFIC_PAYLOAD_DIGEST_MISMATCH")

    if prov["production_subject_sha"] != PRODUCTION_SHA:
        die("FIXED_PROVENANCE_MISMATCH:PRODUCTION")
    if prov["g2b_subject_sha"] != G2B_SHA:
        die("FIXED_PROVENANCE_MISMATCH:G2B")
    if prov["a0_subject_sha"] != A0_SHA:
        die("FIXED_PROVENANCE_MISMATCH:A0")
    if prov["m0_subject_sha"] != M0_SHA:
        die("FIXED_PROVENANCE_MISMATCH:M0")
    if prov["runner_config_projection_sha256"] != CONFIG_SHA:
        die("FIXED_PROVENANCE_MISMATCH:CONFIG")
    if prov["scientific_payload_sha256"] != science_hash:
        die("PROVENANCE_SCIENCE_BINDING_MISMATCH")

    binding_pairs = (
        ("f0_subject_sha", "f0_subject_sha"),
        ("opaque_measurement_id", "opaque_measurement_id"),
        ("opaque_base_fixture_id", "opaque_base_fixture_id"),
        ("opaque_transform_id", "opaque_transform_id"),
        ("technical_pair_id", "technical_pair_id"),
        ("agent_visible_event_count", "agent_visible_event_count"),
        ("agent_visible_events_sha256", "agent_visible_events_sha256"),
        ("runner_config_projection_sha256", "runner_config_projection_sha256"),
        ("executable_sha256", "executable_sha256"),
        ("environment_capsule_sha256", "environment_capsule_sha256"),
        ("source_binding_sha256", "source_binding_sha256"),
    )
    for prov_field, binding_field in binding_pairs:
        if prov[prov_field] != binding[binding_field]:
            die(f"EXECUTION_BINDING_MISMATCH:{prov_field}")
    if science["measurement_cycle_index"] != binding["measurement_cycle_index"]:
        die("EXECUTION_BINDING_MISMATCH:measurement_cycle_index")

    receipt_hash = domain_hash(RECEIPT_DOMAIN, provenance_bytes(prov))
    if receipt_hash != require_lower_hex(doc["receipt_binding_sha256"], 64, "RECEIPT_DIGEST_INVALID"):
        die("PROVENANCE_RECEIPT_DIGEST_MISMATCH")
    return doc


def pair_receipt(binding: dict[str, Any], a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    receipts = sorted(
        [a, b],
        key=lambda receipt: receipt["provenance_envelope"]["technical_replicate_index"],
    )
    p0, p1 = (receipt["provenance_envelope"] for receipt in receipts)
    reasons: list[str] = []

    indices = [p0["technical_replicate_index"], p1["technical_replicate_index"]]
    if indices != [0, 1]:
        reasons.append("REPLICATE_INDEX_SET_INVALID")
    for field in PAIR_INVARIANT_FIELDS:
        if p0[field] != p1[field]:
            reasons.append(f"PEER_BINDING_MISMATCH:{field}")
    if p0["service_instance_id"] == p1["service_instance_id"]:
        reasons.append("SERVICE_INSTANCE_REUSE")
    if receipts[0]["scientific_payload_sha256"] != receipts[1]["scientific_payload_sha256"]:
        reasons.append("SCIENTIFIC_PAYLOAD_MISMATCH")
    if receipts[0]["receipt_binding_sha256"] == receipts[1]["receipt_binding_sha256"]:
        reasons.append("PROVENANCE_BINDING_NOT_DISTINCT")
    if any(
        receipt["scientific_payload"]["measurement_validity"] != "ELIGIBLE_F0_TRANSPORT"
        for receipt in receipts
    ):
        reasons.append("PEER_INVALID")

    status = "ELIGIBLE_F0_TECHNICAL_PAIR" if not reasons else "INVALID_MEASUREMENT_NONDETERMINISTIC"
    binding_hash = domain_hash(BINDING_DOMAIN, canonical_json(binding))
    pair = {
        "schema_version": PAIR_SCHEMA,
        "execution_binding_sha256": binding_hash,
        "technical_pair_id": binding["technical_pair_id"],
        "replicate_indices": indices,
        "worker_receipt_binding_sha256": [
            receipts[0]["receipt_binding_sha256"],
            receipts[1]["receipt_binding_sha256"],
        ],
        "service_instance_ids": [p0["service_instance_id"], p1["service_instance_id"]],
        "scientific_payload_sha256": [
            receipts[0]["scientific_payload_sha256"],
            receipts[1]["scientific_payload_sha256"],
        ],
        "status": status,
        "invalidity_reasons": reasons,
        "statistical_n_increment": 0,
    }
    pair["pair_receipt_sha256"] = domain_hash(PAIR_DOMAIN, canonical_json(pair))
    return pair


def main(argv: list[str]) -> int:
    if len(argv) != 4:
        print(
            "usage: f0_pair_audit.py EXECUTION_BINDING.json REPLICATE0.json REPLICATE1.json",
            file=sys.stderr,
        )
        return 2
    try:
        binding = validate_binding(strict_json_load(pathlib.Path(argv[1])))
        docs = [strict_json_load(pathlib.Path(path)) for path in argv[2:]]
        verified = [verify_worker_receipt(doc, binding) for doc in docs]
        receipt = pair_receipt(binding, *verified)
        print(json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False))
        return 0 if receipt["status"] == "ELIGIBLE_F0_TECHNICAL_PAIR" else 3
    except (OSError, json.JSONDecodeError, AuditFailure, TypeError, UnicodeError) as exc:
        print(
            json.dumps(
                {
                    "schema_version": PAIR_SCHEMA,
                    "status": "INVALID_MEASUREMENT_NONDETERMINISTIC",
                    "invalidity_reasons": [str(exc)],
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
