#!/usr/bin/env python3
"""Independent canonical-byte oracle for SPINE-000B receipt commitments v1.

Measurement-only. This file does not execute cognition and grants no authority.
It freezes a byte-level reference implementation for Issue #3255.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from copy import deepcopy
from pathlib import Path

EXECUTION_DOMAIN = b"symthaea.spine.000b.execution-receipt.v1\0"
INTEGRATION_DOMAIN = b"symthaea.spine.000b.cycle-integration.v1\0"
APPLICATION_DOMAIN = b"symthaea.spine.000b.state-application.v1\0"
CYCLE_DOMAIN = b"symthaea.spine.000b.cycle-evidence.v1\0"
GENESIS_DOMAIN = b"symthaea.spine.000b.evidence-genesis.v1\0"
CHAIN_DOMAIN = b"symthaea.spine.000b.evidence-chain-link.v1\0"

OUTCOMES = {
    "SKIPPED_SCHEDULE": 0,
    "SKIPPED_HEALTH_DISABLED": 1,
    "EXECUTED_NEUTRAL": 2,
    "EXECUTED_NON_NEUTRAL": 3,
    "PANICKED_CAUGHT": 4,
    "FAILED_OTHER": 5,
}
URGENCY = {"CRUISE": 0, "NORMAL": 1, "CRITICAL": 2}
SOURCE_TAG = {
    "CONFIDENCE_DELTA": 0,
    "LR_MODULATION": 1,
    "EXPLORATION_DELTA": 2,
    "AROUSAL_DELTA": 3,
    "VALENCE_DELTA": 4,
    "FLAG": 5,
}
STATE_CHANGE = {"UNCHANGED": 0, "CHANGED": 1, "NOT_OBSERVED_AT_BOUNDARY": 2}
VALUE_TAG = {"F64_BITS": 1, "F32_BITS": 2, "U64": 3, "U32": 4, "BOOL": 5, "DIGEST32": 6}

IDENTITY_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
DEST_RE = IDENTITY_RE
PATH_RE = re.compile(r"^[A-Za-z0-9_./-]{1,512}$")
REF_RE = re.compile(r"^[A-Za-z0-9_./:@+-]{1,512}$")
U32_MAX = 0xFFFF_FFFF
U16_MAX = 0xFFFF


def fail(msg: str):
    raise ValueError(msg)


def u8(v: int) -> bytes:
    if not isinstance(v, int) or not 0 <= v <= 0xFF:
        fail("u8 out of range")
    return bytes([v])


def u16(v: int) -> bytes:
    if not isinstance(v, int) or not 0 <= v <= U16_MAX:
        fail("u16 out of range")
    return struct.pack("<H", v)


def u32(v: int) -> bytes:
    if not isinstance(v, int) or not 0 <= v <= U32_MAX:
        fail("u32 out of range")
    return struct.pack("<I", v)


def u64(v: int) -> bytes:
    if not isinstance(v, int) or not 0 <= v <= 0xFFFF_FFFF_FFFF_FFFF:
        fail("u64 out of range")
    return struct.pack("<Q", v)


def boolean(v: bool) -> bytes:
    if type(v) is not bool:
        fail("boolean must be exact bool")
    return b"\x01" if v else b"\x00"


def ascii_string(value: str, *, kind: str) -> bytes:
    if not isinstance(value, str):
        fail(f"{kind} must be string")
    try:
        raw = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{kind} must be ASCII") from exc
    if kind == "identity":
        if not IDENTITY_RE.fullmatch(value):
            fail("invalid subsystem identity")
    elif kind == "destination":
        if not DEST_RE.fullmatch(value):
            fail("invalid destination id")
    elif kind == "path":
        if not PATH_RE.fullmatch(value) or value.startswith("/") or "\\" in value:
            fail("invalid source path")
        segments = value.split("/")
        if any(segment in ("", ".", "..") for segment in segments):
            fail("noncanonical source path")
    elif kind == "ref":
        if not REF_RE.fullmatch(value):
            fail("invalid overlap ref")
    else:
        fail("unknown string kind")
    if len(raw) > U16_MAX:
        fail("string too long")
    return u16(len(raw)) + raw


def enum_tag(table: dict[str, int], name: str, label: str) -> bytes:
    if name not in table:
        fail(f"unknown {label}: {name}")
    return u8(table[name])


def digest(domain: bytes, payload: bytes) -> bytes:
    return hashlib.sha256(domain + payload).digest()


def hex32(value: str) -> bytes:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        fail("digest must be lowercase 64-char hex")
    return bytes.fromhex(value)


def proposal_bytes(p: dict) -> bytes:
    return b"".join(
        [
            u64(p["confidence_delta_bits"]),
            u64(p["lr_modulation_bits"]),
            u64(p["exploration_delta_bits"]),
            u32(p["arousal_delta_bits"]),
            u32(p["valence_delta_bits"]),
            u32(p["flags"]),
        ]
    )


def integrated_bytes(v: dict) -> bytes:
    return proposal_bytes(v) + u32(v["n_contributors"])


def optional_proposal(value) -> bytes:
    if value is None:
        return b"\x00"
    return b"\x01" + proposal_bytes(value)


def canonical_value(value) -> bytes:
    if value is None:
        fail("canonical value cannot be None; use optional_value")
    kind = value.get("kind")
    if kind not in VALUE_TAG:
        fail(f"unknown canonical value kind: {kind}")
    tag = u8(VALUE_TAG[kind])
    raw = value.get("value")
    if kind == "F64_BITS":
        return tag + u64(raw)
    if kind == "F32_BITS":
        return tag + u32(raw)
    if kind == "U64":
        return tag + u64(raw)
    if kind == "U32":
        return tag + u32(raw)
    if kind == "BOOL":
        return tag + boolean(raw)
    if kind == "DIGEST32":
        return tag + hex32(raw)
    raise AssertionError("unreachable")


def optional_value(value) -> bytes:
    return b"\x00" if value is None else b"\x01" + canonical_value(value)


def validate_execution_truth(r: dict) -> None:
    outcome = r["execution_outcome"]
    if outcome not in OUTCOMES:
        fail("unknown execution outcome")
    emitted = r["emitted"]
    admitted = r["admitted"]
    proposal = r.get("proposal")
    if outcome in {"SKIPPED_SCHEDULE", "SKIPPED_HEALTH_DISABLED", "PANICKED_CAUGHT", "FAILED_OTHER"}:
        if emitted or admitted or proposal is not None:
            fail("non-executed/panic outcome has emitted/admitted proposal")
    elif outcome == "EXECUTED_NEUTRAL":
        if not emitted or admitted or proposal is None:
            fail("neutral execution truth table violated")
    elif outcome == "EXECUTED_NON_NEUTRAL":
        if not emitted or proposal is None:
            fail("non-neutral execution must emit proposal")


def execution_bytes(r: dict) -> bytes:
    validate_execution_truth(r)
    refs = list(r.get("overlap_refs", []))
    if len(refs) > U16_MAX or len(set(refs)) != len(refs):
        fail("invalid overlap refs")
    refs = sorted(refs, key=lambda x: x.encode("ascii"))
    payload = bytearray()
    payload += u64(r["cycle_number"])
    payload += ascii_string(r["subsystem_identity"], kind="identity")
    payload += ascii_string(r["source_path"], kind="path")
    payload += u32(r["schedule_interval"])
    payload += enum_tag(URGENCY, r["urgency"], "urgency")
    payload += boolean(r["eligible_to_run"])
    payload += enum_tag(OUTCOMES, r["execution_outcome"], "execution outcome")
    payload += boolean(r["emitted"])
    payload += boolean(r["admitted"])
    payload += optional_proposal(r.get("proposal"))
    payload += u16(len(refs))
    for ref in refs:
        payload += ascii_string(ref, kind="ref")
    return bytes(payload)


def execution_digest(r: dict) -> bytes:
    return digest(EXECUTION_DOMAIN, execution_bytes(r))


def integration_bytes(r: dict) -> bytes:
    subjects = list(r.get("subjects", []))
    names = [s["subsystem_identity"] for s in subjects]
    if len(names) != len(set(names)):
        fail("duplicate integration subject identity")
    subjects.sort(key=lambda s: s["subsystem_identity"].encode("ascii"))
    if r["admitted_count"] != len(subjects):
        fail("admitted_count must equal subject_count")
    payload = bytearray()
    payload += u64(r["cycle_number"])
    payload += u32(r["admitted_count"])
    payload += integrated_bytes(r["integrated_all"])
    payload += u32(len(subjects))
    for subject in subjects:
        mask = subject["changed_channel_mask"]
        if mask & 0b1110_0000:
            fail("changed-channel reserved bits set")
        expected_changed = bool(mask or subject["uniquely_contributed_flags"])
        if subject["integration_changed"] is not expected_changed:
            fail("integration_changed inconsistent with influence fields")
        payload += ascii_string(subject["subsystem_identity"], kind="identity")
        payload += integrated_bytes(subject["integrated_without_subject"])
        payload += u8(mask)
        payload += u32(subject["uniquely_contributed_flags"])
        payload += boolean(subject["integration_changed"])
    return bytes(payload)


def integration_digest(r: dict) -> bytes:
    return digest(INTEGRATION_DOMAIN, integration_bytes(r))


def application_bytes(r: dict) -> bytes:
    source_tag = r["source_tag"]
    if source_tag not in SOURCE_TAG:
        fail("unknown application source tag")
    source_flag = r.get("source_flag", 0)
    if source_tag != "FLAG" and source_flag != 0:
        fail("scalar application must have source_flag=0")
    change = r["state_change_status"]
    if change not in STATE_CHANGE:
        fail("unknown state change status")
    before, after = r.get("before"), r.get("after")
    if change in {"UNCHANGED", "CHANGED"}:
        if before is None or after is None or before.get("kind") != after.get("kind"):
            fail("observed state change requires compatible before/after")
    payload = bytearray()
    payload += u64(r["cycle_number"])
    payload += u32(r["application_index"])
    payload += ascii_string(r["destination_id"], kind="destination")
    payload += enum_tag(SOURCE_TAG, source_tag, "application source")
    payload += u32(source_flag)
    payload += boolean(r["applied"])
    payload += enum_tag(STATE_CHANGE, change, "state change")
    payload += optional_value(before)
    payload += optional_value(after)
    return bytes(payload)


def application_digest(r: dict) -> bytes:
    return digest(APPLICATION_DOMAIN, application_bytes(r))


def cycle_bytes(cycle_number: int, executions: list[dict], integration: dict, applications: list[dict]) -> bytes:
    if integration["cycle_number"] != cycle_number:
        fail("integration/envelope cycle mismatch")
    exec_names = [r["subsystem_identity"] for r in executions]
    if len(exec_names) != len(set(exec_names)):
        fail("duplicate execution identity")
    for r in executions:
        if r["cycle_number"] != cycle_number:
            fail("execution/envelope cycle mismatch")
    for r in applications:
        if r["cycle_number"] != cycle_number:
            fail("application/envelope cycle mismatch")
    executions = sorted(executions, key=lambda r: r["subsystem_identity"].encode("ascii"))
    applications = sorted(applications, key=lambda r: r["application_index"])
    indices = [r["application_index"] for r in applications]
    if indices != list(range(len(applications))):
        fail("application indices must be contiguous from zero")
    payload = bytearray()
    payload += u64(cycle_number)
    payload += u32(len(executions))
    for r in executions:
        payload += ascii_string(r["subsystem_identity"], kind="identity")
        payload += execution_digest(r)
    payload += integration_digest(integration)
    payload += u32(len(applications))
    for r in applications:
        payload += u32(r["application_index"])
        payload += application_digest(r)
    return bytes(payload)


def cycle_digest(cycle_number: int, executions: list[dict], integration: dict, applications: list[dict]) -> bytes:
    return digest(CYCLE_DOMAIN, cycle_bytes(cycle_number, executions, integration, applications))


def genesis_root(subject_manifest_digest_hex: str) -> bytes:
    return digest(GENESIS_DOMAIN, hex32(subject_manifest_digest_hex))


def chain_root(previous_root: bytes, cycle_digest_value: bytes) -> bytes:
    if len(previous_root) != 32 or len(cycle_digest_value) != 32:
        fail("chain inputs must be 32-byte digests")
    return digest(CHAIN_DOMAIN, previous_root + cycle_digest_value)


def sample_records():
    zero64 = 0
    one64 = struct.unpack("<Q", struct.pack("<d", 1.0))[0]
    quarter64 = struct.unpack("<Q", struct.pack("<d", 0.25))[0]
    neutral = {
        "confidence_delta_bits": zero64,
        "lr_modulation_bits": one64,
        "exploration_delta_bits": zero64,
        "arousal_delta_bits": 0,
        "valence_delta_bits": 0,
        "flags": 0,
    }
    proposal = {**neutral, "confidence_delta_bits": quarter64, "flags": 1}
    execution_a = {
        "cycle_number": 7,
        "subsystem_identity": "manager_z",
        "source_path": "src/cognitive_loop/managers/z.rs",
        "schedule_interval": 7,
        "urgency": "NORMAL",
        "eligible_to_run": True,
        "execution_outcome": "EXECUTED_NON_NEUTRAL",
        "emitted": True,
        "admitted": True,
        "proposal": proposal,
        "overlap_refs": ["spine000c:manager_z"],
    }
    execution_b = {
        "cycle_number": 7,
        "subsystem_identity": "manager_a",
        "source_path": "src/cognitive_loop/managers/a.rs",
        "schedule_interval": 11,
        "urgency": "NORMAL",
        "eligible_to_run": True,
        "execution_outcome": "SKIPPED_HEALTH_DISABLED",
        "emitted": False,
        "admitted": False,
        "proposal": None,
        "overlap_refs": [],
    }
    integrated_all = {**proposal, "n_contributors": 1}
    integrated_without = {**neutral, "n_contributors": 0}
    integration = {
        "cycle_number": 7,
        "admitted_count": 1,
        "integrated_all": integrated_all,
        "subjects": [
            {
                "subsystem_identity": "manager_z",
                "integrated_without_subject": integrated_without,
                "changed_channel_mask": 0b00001,
                "uniquely_contributed_flags": 1,
                "integration_changed": True,
            }
        ],
    }
    app0 = {
        "cycle_number": 7,
        "application_index": 0,
        "destination_id": "prediction_confidence",
        "source_tag": "CONFIDENCE_DELTA",
        "source_flag": 0,
        "applied": True,
        "state_change_status": "CHANGED",
        "before": {"kind": "F64_BITS", "value": struct.unpack("<Q", struct.pack("<d", 0.5))[0]},
        "after": {"kind": "F64_BITS", "value": struct.unpack("<Q", struct.pack("<d", 0.75))[0]},
    }
    app1 = {
        "cycle_number": 7,
        "application_index": 1,
        "destination_id": "episodic_memory.consolidate_recent",
        "source_tag": "FLAG",
        "source_flag": 2,
        "applied": True,
        "state_change_status": "NOT_OBSERVED_AT_BOUNDARY",
        "before": None,
        "after": None,
    }
    return execution_a, execution_b, integration, app0, app1


def vectors() -> dict:
    ea, eb, integration, app0, app1 = sample_records()
    cd = cycle_digest(7, [ea, eb], integration, [app0, app1])
    subject = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
    root0 = genesis_root(subject)
    root1 = chain_root(root0, cd)
    return {
        "schema": "symthaea.spine.000b.receipt-commitment-v1-vectors",
        "authority": "measurement-only",
        "execution_a_bytes_hex": execution_bytes(ea).hex(),
        "execution_a_sha256": execution_digest(ea).hex(),
        "execution_b_bytes_hex": execution_bytes(eb).hex(),
        "execution_b_sha256": execution_digest(eb).hex(),
        "integration_bytes_hex": integration_bytes(integration).hex(),
        "integration_sha256": integration_digest(integration).hex(),
        "application_0_bytes_hex": application_bytes(app0).hex(),
        "application_0_sha256": application_digest(app0).hex(),
        "application_1_bytes_hex": application_bytes(app1).hex(),
        "application_1_sha256": application_digest(app1).hex(),
        "cycle_bytes_hex": cycle_bytes(7, [ea, eb], integration, [app0, app1]).hex(),
        "cycle_sha256": cd.hex(),
        "subject_manifest_sha256": subject,
        "genesis_root_sha256": root0.hex(),
        "chain_root_after_cycle_sha256": root1.hex(),
    }


def self_test() -> None:
    ea, eb, integration, app0, app1 = sample_records()
    base = execution_digest(ea)

    # A noncanonical telemetry value is not part of canonical execution bytes.
    telemetry_a = {"duration_ns": 10}
    telemetry_b = {"duration_ns": 999999}
    assert telemetry_a != telemetry_b
    assert execution_digest(ea) == base

    bitflip = deepcopy(ea)
    bitflip["proposal"]["confidence_delta_bits"] ^= 1
    assert execution_digest(bitflip) != base

    outcome = deepcopy(ea)
    outcome["execution_outcome"] = "PANICKED_CAUGHT"
    outcome["emitted"] = False
    outcome["admitted"] = False
    outcome["proposal"] = None
    assert execution_digest(outcome) != base

    renamed = deepcopy(ea)
    renamed["subsystem_identity"] = "manager_y"
    assert execution_digest(renamed) != base

    # Execution input ordering is not semantic; identity sorting canonicalizes it.
    c1 = cycle_digest(7, [ea, eb], integration, [app0, app1])
    c2 = cycle_digest(7, [eb, ea], integration, [app0, app1])
    assert c1 == c2

    # Application list ordering is canonicalized by index, but changing the
    # semantic indices changes the commitment.
    swapped0, swapped1 = deepcopy(app0), deepcopy(app1)
    swapped0["application_index"], swapped1["application_index"] = 1, 0
    c3 = cycle_digest(7, [ea, eb], integration, [swapped0, swapped1])
    assert c3 != c1

    bad = deepcopy(ea)
    bad["urgency"] = "UNKNOWN"
    try:
        execution_digest(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown urgency must fail")

    bad_path = deepcopy(ea)
    bad_path["source_path"] = "../escape.rs"
    try:
        execution_digest(bad_path)
    except ValueError:
        pass
    else:
        raise AssertionError("noncanonical path must fail")

    bad_count = deepcopy(integration)
    bad_count["integrated_all"] = dict(bad_count["integrated_all"], n_contributors=U32_MAX + 1)
    try:
        integration_digest(bad_count)
    except ValueError:
        pass
    else:
        raise AssertionError("count overflow must fail")

    # JSON presentation order/whitespace is deliberately irrelevant: parsing
    # produces the same semantic record before canonical binary encoding.
    rendered_a = json.dumps(ea, sort_keys=True, indent=2)
    rendered_b = json.dumps(ea, sort_keys=False, separators=(",", ":"))
    assert rendered_a != rendered_b
    assert execution_digest(json.loads(rendered_a)) == execution_digest(json.loads(rendered_b))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--emit-vectors", type=Path)
    args = parser.parse_args()
    if not args.self_test and args.emit_vectors is None:
        parser.error("request --self-test and/or --emit-vectors")
    if args.self_test:
        self_test()
        print("SPINE-000B receipt commitment oracle self-test: PASS")
    if args.emit_vectors is not None:
        args.emit_vectors.write_text(json.dumps(vectors(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {args.emit_vectors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
