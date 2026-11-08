#!/usr/bin/env python3
"""Independent standard-library oracle for connectome as-of ledger commitments.

This implementation intentionally does not import or call the Rust encoder.
It fixes the V1 binary format so fixture roots can be reproduced independently.
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
HASH_SUITE = "SHA-256"
GENESIS_DOMAIN = b"SYTHAEA/CONNECTOME/ASOF-LEDGER-GENESIS/V1\0"
ENTRY_DOMAIN = b"SYTHAEA/CONNECTOME/ASOF-LEDGER-ENTRY/V1\0"
INPUT_PREFIX_DOMAIN = b"SYTHAEA/CONNECTOME/ASOF-INPUT-PREFIX/V1\0"
HISTORY_PREFIX_DOMAIN = b"SYTHAEA/CONNECTOME/ASOF-HISTORY-PREFIX/V1\0"
CHECKPOINT_DOMAIN = b"SYTHAEA/CONNECTOME/ASOF-CHECKPOINT/V1\0"

AVAILABILITY = {
    "APPROVED": 0,
    "PROVISIONAL": 1,
    "DISPUTED": 2,
    "FAILED": 3,
    "MISSING": 4,
    "OUT_OF_SUPPORT": 5,
    "NOT_ASSESSED": 6,
}
SPLIT = {"EXPLORATION": 0, "FIT": 1, "CALIBRATION": 2, "CONFIRMATION": 3}
DECISION_KIND = {"DISCOVERY": 0, "CONFIRMATION": 1}
PAYLOAD_KIND = {"EVIDENCE": 1, "FAMILY_FREEZE": 2, "DECISION": 3}


def u8(value: int) -> bytes:
    return struct.pack(">B", value)


def u32(value: int) -> bytes:
    return struct.pack(">I", value)


def u64(value: int) -> bytes:
    return struct.pack(">Q", value)


def i64(value: int) -> bytes:
    return struct.pack(">q", value)


def blob(value: bytes) -> bytes:
    return u64(len(value)) + value


def string(value: str) -> bytes:
    return blob(value.encode("utf-8"))


def digest(value: str) -> bytes:
    raw = bytes.fromhex(value)
    if len(raw) != 32 or len(value) != 64 or value.lower() != value:
        raise ValueError(f"invalid lowercase SHA-256 digest: {value!r}")
    return raw


def option_string(value: str | None) -> bytes:
    return u8(0) if value is None else u8(1) + string(value)


def option_i64(value: int | None) -> bytes:
    return u8(0) if value is None else u8(1) + i64(value)


def string_list(values: list[str]) -> bytes:
    return u64(len(values)) + b"".join(string(value) for value in values)


def payload_bytes(kind: str, payload: dict[str, Any]) -> bytes:
    out = bytearray()
    if kind == "EVIDENCE":
        out += u8(1)
        out += string(payload["event_id"])
        out += string(payload["evidence_family_id"])
        out += string(payload["source_stream"])
        out += digest(payload["source_artifact_sha256"])
        out += u32(payload["revision"])
        out += option_string(payload["supersedes_event_id"])
        out += option_i64(payload["event_time_ms"])
        out += option_i64(payload["acquired_at_ms"])
        out += option_i64(payload["annotated_at_ms"])
        out += option_i64(payload["qc_at_ms"])
        out += u8(AVAILABILITY[payload["availability_state"]])
        out += u8(SPLIT[payload["split"]])
        out += option_string(payload["confirmatory_family_id"])
    elif kind == "FAMILY_FREEZE":
        out += u8(2)
        out += string(payload["family_id"])
        out += digest(payload["family_commitment_sha256"])
        out += u8(1 if payload["confirmation_lockbox_closed"] else 0)
    elif kind == "DECISION":
        out += u8(3)
        out += string(payload["decision_id"])
        out += string(payload["campaign_id"])
        out += u8(DECISION_KIND[payload["kind"]])
        out += digest(payload["policy_version_sha256"])
        out += digest(payload["input_prefix_sha256"])
        out += digest(payload["driver_receipt_sha256"])
        out += digest(payload["history_prefix_sha256"])
        out += digest(payload["action_sha256"])
        out += digest(payload["feasibility_receipt_sha256"])
        out += string_list(payload["available_event_ids"])
        out += string_list(payload["consumed_event_ids"])
        out += option_string(payload["confirmatory_family_id"])
    else:
        raise ValueError(f"unknown ledger payload kind {kind!r}")
    return bytes(out)


def genesis_root(ledger: dict[str, Any]) -> str:
    preimage = (
        GENESIS_DOMAIN
        + u32(SCHEMA_VERSION)
        + string(HASH_SUITE)
        + string(ledger["campaign_id"])
        + string(ledger["global_filtration_id"])
        + digest(ledger["global_filtration_sha256"])
        + string(ledger["ledger_epoch_id"])
    )
    return hashlib.sha256(preimage).hexdigest()


def prefix_digest(domain: bytes, event_ids: list[str], roots: dict[str, str]) -> str:
    encoded = bytearray(domain + u32(SCHEMA_VERSION) + u64(len(event_ids)))
    for event_id in event_ids:
        encoded += string(event_id)
        if event_id in roots:
            encoded += u8(1) + digest(roots[event_id])
        else:
            encoded += u8(0)
    return hashlib.sha256(encoded).hexdigest()


def entry_root(
    ledger: dict[str, Any], sequence: int, previous_root: str, kind: str, payload: dict[str, Any]
) -> str:
    payload_encoded = payload_bytes(kind, payload)
    preimage = (
        ENTRY_DOMAIN
        + u32(SCHEMA_VERSION)
        + string(HASH_SUITE)
        + string(ledger["campaign_id"])
        + string(ledger["global_filtration_id"])
        + digest(ledger["global_filtration_sha256"])
        + string(ledger["ledger_epoch_id"])
        + u64(sequence)
        + digest(previous_root)
        + u8(PAYLOAD_KIND[kind])
        + blob(payload_encoded)
    )
    return hashlib.sha256(preimage).hexdigest()


def checkpoint_sha256(ledger: dict[str, Any]) -> str:
    preimage = (
        CHECKPOINT_DOMAIN
        + u32(SCHEMA_VERSION)
        + string(HASH_SUITE)
        + string(ledger["campaign_id"])
        + string(ledger["global_filtration_id"])
        + string(ledger["ledger_epoch_id"])
        + u64(len(ledger["entries"]))
        + digest(ledger["ledger_chain_root_sha256"])
    )
    return hashlib.sha256(preimage).hexdigest()


def verify_manifest(document: dict[str, Any]) -> dict[str, Any]:
    entries = document["entries"]
    roots: dict[str, str] = {}
    ordered_event_ids: list[str] = []
    previous = genesis_root(document)
    computed_entry_roots: list[str] = []
    decision_input_roots: list[str] = []
    decision_history_roots: list[str] = []

    for index, entry in enumerate(entries, start=1):
        if entry["sequence"] != index:
            raise ValueError(f"entry sequence mismatch at {index}")
        kind = entry["kind"]
        payload = entry["payload"]
        if kind == "DECISION":
            expected_input = prefix_digest(INPUT_PREFIX_DOMAIN, ordered_event_ids, roots)
            expected_history = prefix_digest(HISTORY_PREFIX_DOMAIN, payload["consumed_event_ids"], roots)
            if payload["available_event_ids"] != ordered_event_ids:
                raise ValueError(f"decision {payload['decision_id']} availability prefix mismatch")
            if payload["input_prefix_sha256"] != expected_input:
                raise ValueError(f"decision {payload['decision_id']} input prefix digest mismatch")
            if payload["history_prefix_sha256"] != expected_history:
                raise ValueError(f"decision {payload['decision_id']} history prefix digest mismatch")
            decision_input_roots.append(expected_input)
            decision_history_roots.append(expected_history)

        computed = entry_root(document, index, previous, kind, payload)
        if entry["entry_commitment_sha256"] != computed:
            raise ValueError(f"entry commitment mismatch at sequence {index}")
        entry["entry_commitment_sha256"] = computed
        previous = computed
        computed_entry_roots.append(computed)

        if kind == "EVIDENCE":
            event_id = payload["event_id"]
            if event_id in roots:
                raise ValueError(f"duplicate evidence event ID {event_id}")
            roots[event_id] = computed
            ordered_event_ids.append(event_id)

    if document["ledger_chain_root_sha256"] != previous:
        raise ValueError("terminal ledger chain root mismatch")

    return {
        "entry_commitment_sha256": computed_entry_roots,
        "decision_input_prefix_sha256": decision_input_roots,
        "decision_history_prefix_sha256": decision_history_roots,
        "ledger_chain_root_sha256": previous,
        "terminal_checkpoint_sha256": checkpoint_sha256(document),
        "sha256_empty": hashlib.sha256(b"").hexdigest(),
        "sha256_abc": hashlib.sha256(b"abc").hexdigest(),
    }


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: verify_ledger_commitments.py <ledger.json>", file=sys.stderr)
        return 2
    document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    result = verify_manifest(document)
    expected = document.get("oracle_expectations", {})
    for key, value in result.items():
        if key in expected and expected[key] != value:
            raise SystemExit(f"oracle mismatch for {key}: expected {expected[key]!r}, got {value!r}")
    print("independent canonical ledger commitment replay: PASS")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
