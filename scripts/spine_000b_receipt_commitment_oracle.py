#!/usr/bin/env python3
"""Independent canonical-byte oracle for SPINE-000B receipt commitments v1.

The application record binds the production operation, the exact applied
operand, and whether the integrated source was a non-identity scalar, a set
flag, or a clear flag. Measurement-only: this does not execute cognition and
confers no causal, epistemic, safety, or action authority.
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
SOURCE_CONDITION = {
    "SCALAR_NON_IDENTITY": 0,
    "FLAG_SET": 1,
    "FLAG_CLEAR": 2,
}
STATE_CHANGE = {"UNCHANGED": 0, "CHANGED": 1, "NOT_OBSERVED_AT_BOUNDARY": 2}
VALUE_TAG = {"F64_BITS": 1, "F32_BITS": 2, "U64": 3, "U32": 4, "BOOL": 5, "DIGEST32": 6}

IDENTITY_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
PATH_RE = re.compile(r"^[A-Za-z0-9_./-]{1,512}$")
REF_RE = re.compile(r"^[A-Za-z0-9_./:@+-]{1,512}$")
U16_MAX = 0xFFFF
U32_MAX = 0xFFFF_FFFF


def fail(message: str):
    raise ValueError(message)


def u8(value: int) -> bytes:
    if not isinstance(value, int) or not 0 <= value <= 0xFF:
        fail("u8 out of range")
    return bytes([value])


def u16(value: int) -> bytes:
    if not isinstance(value, int) or not 0 <= value <= U16_MAX:
        fail("u16 out of range")
    return struct.pack("<H", value)


def u32(value: int) -> bytes:
    if not isinstance(value, int) or not 0 <= value <= U32_MAX:
        fail("u32 out of range")
    return struct.pack("<I", value)


def u64(value: int) -> bytes:
    if not isinstance(value, int) or not 0 <= value <= 0xFFFF_FFFF_FFFF_FFFF:
        fail("u64 out of range")
    return struct.pack("<Q", value)


def boolean(value: bool) -> bytes:
    if type(value) is not bool:
        fail("boolean must be exact bool")
    return b"\x01" if value else b"\x00"


def canonical_string(value: str, kind: str) -> bytes:
    if not isinstance(value, str):
        fail(f"{kind} must be a string")
    try:
        raw = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{kind} must be ASCII") from exc
    if kind in {"identity", "destination", "operation"}:
        if not IDENTITY_RE.fullmatch(value):
            fail(f"invalid {kind}")
    elif kind == "path":
        if not PATH_RE.fullmatch(value) or value.startswith("/") or "\\" in value:
            fail("invalid source path")
        if any(part in {"", ".", ".."} for part in value.split("/")):
            fail("noncanonical source path")
    elif kind == "ref":
        if not REF_RE.fullmatch(value):
            fail("invalid overlap ref")
    else:
        fail("unknown canonical string kind")
    if len(raw) > U16_MAX:
        fail("canonical string too long")
    return u16(len(raw)) + raw


def enum_tag(table: dict[str, int], name: str, label: str) -> bytes:
    if name not in table:
        fail(f"unknown {label}: {name}")
    return u8(table[name])


def sha256(domain: bytes, payload: bytes) -> bytes:
    return hashlib.sha256(domain + payload).digest()


def hex32(value: str) -> bytes:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        fail("digest must be lowercase 64-char hex")
    return bytes.fromhex(value)


def proposal_bytes(value: dict) -> bytes:
    return b"".join(
        (
            u64(value["confidence_delta_bits"]),
            u64(value["lr_modulation_bits"]),
            u64(value["exploration_delta_bits"]),
            u32(value["arousal_delta_bits"]),
            u32(value["valence_delta_bits"]),
            u32(value["flags"]),
        )
    )


def integrated_bytes(value: dict) -> bytes:
    return proposal_bytes(value) + u32(value["n_contributors"])


def optional_proposal(value) -> bytes:
    return b"\x00" if value is None else b"\x01" + proposal_bytes(value)


def canonical_value(value: dict) -> bytes:
    kind = value.get("kind")
    if kind not in VALUE_TAG:
        fail(f"unknown canonical value kind: {kind}")
    out = bytearray(u8(VALUE_TAG[kind]))
    raw = value.get("value")
    if kind == "F64_BITS":
        out += u64(raw)
    elif kind == "F32_BITS":
        out += u32(raw)
    elif kind == "U64":
        out += u64(raw)
    elif kind == "U32":
        out += u32(raw)
    elif kind == "BOOL":
        out += boolean(raw)
    elif kind == "DIGEST32":
        out += hex32(raw)
    return bytes(out)


def optional_value(value) -> bytes:
    return b"\x00" if value is None else b"\x01" + canonical_value(value)


def value_kind(value):
    return None if value is None else value.get("kind")


def validate_execution(record: dict) -> None:
    outcome = record["execution_outcome"]
    if outcome not in OUTCOMES:
        fail("unknown execution outcome")
    emitted = record["emitted"]
    admitted = record["admitted"]
    proposal = record.get("proposal")
    if outcome in {"SKIPPED_SCHEDULE", "SKIPPED_HEALTH_DISABLED", "PANICKED_CAUGHT", "FAILED_OTHER"}:
        if emitted or admitted or proposal is not None:
            fail("skipped/panic/failed execution truth table violated")
    elif outcome == "EXECUTED_NEUTRAL":
        if not emitted or admitted or proposal is None:
            fail("neutral execution truth table violated")
    elif outcome == "EXECUTED_NON_NEUTRAL":
        if not emitted or proposal is None:
            fail("non-neutral execution truth table violated")


def execution_bytes(record: dict) -> bytes:
    validate_execution(record)
    refs = list(record.get("overlap_refs", []))
    if len(refs) > U16_MAX or len(refs) != len(set(refs)):
        fail("invalid overlap refs")
    for ref in refs:
        canonical_string(ref, "ref")
    refs.sort(key=lambda ref: ref.encode("ascii"))
    out = bytearray()
    out += u64(record["cycle_number"])
    out += canonical_string(record["subsystem_identity"], "identity")
    out += canonical_string(record["source_path"], "path")
    out += u32(record["schedule_interval"])
    out += enum_tag(URGENCY, record["urgency"], "urgency")
    out += boolean(record["eligible_to_run"])
    out += enum_tag(OUTCOMES, record["execution_outcome"], "execution outcome")
    out += boolean(record["emitted"])
    out += boolean(record["admitted"])
    out += optional_proposal(record.get("proposal"))
    out += u16(len(refs))
    for ref in refs:
        out += canonical_string(ref, "ref")
    return bytes(out)


def execution_digest(record: dict) -> bytes:
    return sha256(EXECUTION_DOMAIN, execution_bytes(record))


def integration_bytes(record: dict) -> bytes:
    subjects = list(record.get("subjects", []))
    names = [subject["subsystem_identity"] for subject in subjects]
    if len(names) != len(set(names)):
        fail("duplicate integration subject identity")
    subjects.sort(key=lambda subject: subject["subsystem_identity"].encode("ascii"))
    if record["admitted_count"] != len(subjects):
        fail("admitted_count must equal subject_count")
    out = bytearray()
    out += u64(record["cycle_number"])
    out += u32(record["admitted_count"])
    out += integrated_bytes(record["integrated_all"])
    out += u32(len(subjects))
    for subject in subjects:
        mask = subject["changed_channel_mask"]
        if mask & 0b1110_0000:
            fail("changed-channel reserved bits set")
        unique = subject["uniquely_contributed_flags"]
        if subject["integration_changed"] is not bool(mask or unique):
            fail("integration_changed inconsistent with influence fields")
        out += canonical_string(subject["subsystem_identity"], "identity")
        out += integrated_bytes(subject["integrated_without_subject"])
        out += u8(mask)
        out += u32(unique)
        out += boolean(subject["integration_changed"])
    return bytes(out)


def integration_digest(record: dict) -> bytes:
    return sha256(INTEGRATION_DOMAIN, integration_bytes(record))


def application_bytes(record: dict) -> bytes:
    source = record["source_tag"]
    condition = record["source_condition"]
    if source not in SOURCE_TAG:
        fail("unknown application source tag")
    if condition not in SOURCE_CONDITION:
        fail("unknown application source condition")
    source_flag = record.get("source_flag", 0)
    if source != "FLAG":
        if source_flag != 0 or condition != "SCALAR_NON_IDENTITY":
            fail("scalar application source semantics invalid")
    else:
        if source_flag == 0 or condition not in {"FLAG_SET", "FLAG_CLEAR"}:
            fail("flag application source semantics invalid")

    change = record["state_change_status"]
    if change not in STATE_CHANGE:
        fail("unknown state change status")
    before = record.get("before")
    after = record.get("after")
    if change in {"UNCHANGED", "CHANGED"}:
        if before is None or after is None or value_kind(before) != value_kind(after):
            fail("observed state change requires compatible before/after")

    out = bytearray()
    out += u64(record["cycle_number"])
    out += u32(record["application_index"])
    out += canonical_string(record["operation_id"], "operation")
    out += canonical_string(record["destination_id"], "destination")
    out += enum_tag(SOURCE_TAG, source, "application source")
    out += u32(source_flag)
    out += enum_tag(SOURCE_CONDITION, condition, "application source condition")
    out += boolean(record["applied"])
    out += optional_value(record.get("applied_argument"))
    out += enum_tag(STATE_CHANGE, change, "state change")
    out += optional_value(before)
    out += optional_value(after)
    return bytes(out)


def application_digest(record: dict) -> bytes:
    return sha256(APPLICATION_DOMAIN, application_bytes(record))


def cycle_bytes(cycle_number: int, executions: list[dict], integration: dict, applications: list[dict]) -> bytes:
    if integration["cycle_number"] != cycle_number:
        fail("integration/envelope cycle mismatch")
    execution_names = [record["subsystem_identity"] for record in executions]
    if len(execution_names) != len(set(execution_names)):
        fail("duplicate execution identity")
    if any(record["cycle_number"] != cycle_number for record in executions):
        fail("execution/envelope cycle mismatch")
    if any(record["cycle_number"] != cycle_number for record in applications):
        fail("application/envelope cycle mismatch")
    executions = sorted(executions, key=lambda record: record["subsystem_identity"].encode("ascii"))
    applications = sorted(applications, key=lambda record: record["application_index"])
    if [record["application_index"] for record in applications] != list(range(len(applications))):
        fail("application indices must be contiguous from zero")
    out = bytearray()
    out += u64(cycle_number)
    out += u32(len(executions))
    for record in executions:
        out += canonical_string(record["subsystem_identity"], "identity")
        out += execution_digest(record)
    out += integration_digest(integration)
    out += u32(len(applications))
    for record in applications:
        out += u32(record["application_index"])
        out += application_digest(record)
    return bytes(out)


def cycle_digest(cycle_number: int, executions: list[dict], integration: dict, applications: list[dict]) -> bytes:
    return sha256(CYCLE_DOMAIN, cycle_bytes(cycle_number, executions, integration, applications))


def genesis_root(subject_manifest_digest_hex: str) -> bytes:
    return sha256(GENESIS_DOMAIN, hex32(subject_manifest_digest_hex))


def chain_root(previous_root: bytes, cycle_digest_value: bytes) -> bytes:
    if len(previous_root) != 32 or len(cycle_digest_value) != 32:
        fail("chain inputs must be 32-byte digests")
    return sha256(CHAIN_DOMAIN, previous_root + cycle_digest_value)


def sample_records():
    f64_bits = lambda value: struct.unpack("<Q", struct.pack("<d", value))[0]
    f32_bits = lambda value: struct.unpack("<I", struct.pack("<f", value))[0]
    neutral = {
        "confidence_delta_bits": f64_bits(0.0),
        "lr_modulation_bits": f64_bits(1.0),
        "exploration_delta_bits": f64_bits(0.0),
        "arousal_delta_bits": f32_bits(0.0),
        "valence_delta_bits": f32_bits(0.0),
        "flags": 0,
    }
    proposal = {**neutral, "confidence_delta_bits": f64_bits(0.25), "flags": 1}
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
    integration = {
        "cycle_number": 7,
        "admitted_count": 1,
        "integrated_all": {**proposal, "n_contributors": 1},
        "subjects": [
            {
                "subsystem_identity": "manager_z",
                "integrated_without_subject": {**neutral, "n_contributors": 0},
                "changed_channel_mask": 1,
                "uniquely_contributed_flags": 1,
                "integration_changed": True,
            }
        ],
    }
    application_0 = {
        "cycle_number": 7,
        "application_index": 0,
        "operation_id": "feedback.adjust_confidence",
        "destination_id": "prediction_confidence",
        "source_tag": "CONFIDENCE_DELTA",
        "source_flag": 0,
        "source_condition": "SCALAR_NON_IDENTITY",
        "applied": True,
        "applied_argument": {"kind": "F32_BITS", "value": f32_bits(0.25)},
        "state_change_status": "CHANGED",
        "before": {"kind": "F64_BITS", "value": f64_bits(0.5)},
        "after": {"kind": "F64_BITS", "value": f64_bits(0.75)},
    }
    application_1 = {
        "cycle_number": 7,
        "application_index": 1,
        "operation_id": "episodic_memory.consolidate_recent",
        "destination_id": "fep.episodic_memory",
        "source_tag": "FLAG",
        "source_flag": 2,
        "source_condition": "FLAG_SET",
        "applied": True,
        "applied_argument": None,
        "state_change_status": "NOT_OBSERVED_AT_BOUNDARY",
        "before": None,
        "after": None,
    }
    return execution_a, execution_b, integration, application_0, application_1


def vectors() -> dict:
    ea, eb, integration, app0, app1 = sample_records()
    cycle = cycle_digest(7, [ea, eb], integration, [app0, app1])
    subject = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f"
    root0 = genesis_root(subject)
    root1 = chain_root(root0, cycle)
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
        "cycle_sha256": cycle.hex(),
        "subject_manifest_sha256": subject,
        "genesis_root_sha256": root0.hex(),
        "chain_root_after_cycle_sha256": root1.hex(),
    }


def self_test() -> None:
    ea, eb, integration, app0, app1 = sample_records()
    base = execution_digest(ea)
    bitflip = deepcopy(ea)
    bitflip["proposal"]["confidence_delta_bits"] ^= 1
    assert execution_digest(bitflip) != base
    renamed = deepcopy(ea)
    renamed["subsystem_identity"] = "manager_y"
    assert execution_digest(renamed) != base

    cycle_a = cycle_digest(7, [ea, eb], integration, [app0, app1])
    cycle_b = cycle_digest(7, [eb, ea], integration, [app0, app1])
    assert cycle_a == cycle_b

    swapped0, swapped1 = deepcopy(app0), deepcopy(app1)
    swapped0["application_index"], swapped1["application_index"] = 1, 0
    assert cycle_digest(7, [ea, eb], integration, [swapped0, swapped1]) != cycle_a

    changed_arg = deepcopy(app0)
    changed_arg["applied_argument"]["value"] ^= 1
    assert application_digest(changed_arg) != application_digest(app0)
    changed_op = deepcopy(app0)
    changed_op["operation_id"] = "feedback.scale_confidence"
    assert application_digest(changed_op) != application_digest(app0)
    changed_condition = deepcopy(app1)
    changed_condition["source_condition"] = "FLAG_CLEAR"
    assert application_digest(changed_condition) != application_digest(app1)

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

    bad_scalar = deepcopy(app0)
    bad_scalar["source_condition"] = "FLAG_SET"
    try:
        application_digest(bad_scalar)
    except ValueError:
        pass
    else:
        raise AssertionError("scalar/condition mismatch must fail")

    bad_count = deepcopy(integration)
    bad_count["integrated_all"] = dict(bad_count["integrated_all"], n_contributors=U32_MAX + 1)
    try:
        integration_digest(bad_count)
    except ValueError:
        pass
    else:
        raise AssertionError("count overflow must fail")

    a = json.dumps(ea, sort_keys=True, indent=2)
    b = json.dumps(ea, sort_keys=False, separators=(",", ":"))
    assert a != b
    assert execution_digest(json.loads(a)) == execution_digest(json.loads(b))

    telemetry_a = {"duration_ns": 1}
    telemetry_b = {"duration_ns": 999}
    assert telemetry_a != telemetry_b and execution_digest(ea) == base


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
