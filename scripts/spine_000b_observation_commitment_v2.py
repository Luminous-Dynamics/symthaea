#!/usr/bin/env python3
"""SPINE-000B-C2 canonical observation commitment v2 reference encoder.

Measurement-only. This is a reference implementation for frozen synthetic
vectors. Runtime evidence and causal load are deliberately out of scope.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from dataclasses import dataclass
from pathlib import Path

EXECUTION_V2_DOMAIN = b"symthaea.spine.000b.execution-receipt.v2\0"
GUARD_EVENT_DOMAIN = b"symthaea.spine.000b.guard-witness-event.v1\0"
GUARD_BUNDLE_DOMAIN = b"symthaea.spine.000b.guard-witness-bundle.v1\0"
OBS_CYCLE_DOMAIN = b"symthaea.spine.000b.observation-cycle.v1\0"
OBS_GENESIS_DOMAIN = b"symthaea.spine.000b.observation-genesis.v1\0"
OBS_CHAIN_DOMAIN = b"symthaea.spine.000b.observation-chain-link.v1\0"

IDENTITY_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
PATH_RE = re.compile(r"^[A-Za-z0-9_./-]+$")
VALID_PREDICATE_IDS = set(range(1, 8))
MAX_GUARD_WITNESSES = 16

EXECUTION_OUTCOME = {
    "SKIPPED_SCHEDULE": 0,
    "SKIPPED_HEALTH_DISABLED": 1,
    "EXECUTED_NEUTRAL": 2,
    "EXECUTED_NON_NEUTRAL": 3,
    "PANICKED_CAUGHT": 4,
    "FAILED_OTHER": 5,
}
URGENCY = {"CRUISE": 0, "NORMAL": 1, "CRITICAL": 2}


def u8(v: int) -> bytes:
    if not 0 <= v <= 0xFF:
        raise ValueError("u8 overflow")
    return struct.pack("<B", v)


def u16(v: int) -> bytes:
    if not 0 <= v <= 0xFFFF:
        raise ValueError("u16 overflow")
    return struct.pack("<H", v)


def u32(v: int) -> bytes:
    if not 0 <= v <= 0xFFFF_FFFF:
        raise ValueError("u32 overflow")
    return struct.pack("<I", v)


def u64(v: int) -> bytes:
    if not 0 <= v <= 0xFFFF_FFFF_FFFF_FFFF:
        raise ValueError("u64 overflow")
    return struct.pack("<Q", v)


def bool_byte(v: bool) -> bytes:
    if type(v) is not bool:
        raise ValueError("canonical bool must be bool")
    return b"\x01" if v else b"\x00"


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def canonical_identity(value: str, *, max_len: int = 128) -> bytes:
    raw = value.encode("ascii")
    if not 1 <= len(raw) <= max_len or not IDENTITY_RE.fullmatch(value):
        raise ValueError(f"invalid canonical identity: {value!r}")
    return u16(len(raw)) + raw


def canonical_path(value: str) -> bytes:
    raw = value.encode("ascii")
    if not 1 <= len(raw) <= 512 or not PATH_RE.fullmatch(value):
        raise ValueError(f"invalid canonical path: {value!r}")
    if value.startswith("/") or "\\" in value:
        raise ValueError("path must be repo-relative POSIX")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("noncanonical path segment")
    return u16(len(raw)) + raw


@dataclass(frozen=True)
class ProposalBitsV2:
    confidence_delta_bits: int
    lr_modulation_bits: int
    exploration_delta_bits: int
    arousal_delta_bits: int
    valence_delta_bits: int
    flags: int
    reserved: int

    def encode(self) -> bytes:
        return b"".join(
            [
                u64(self.confidence_delta_bits),
                u64(self.lr_modulation_bits),
                u64(self.exploration_delta_bits),
                u32(self.arousal_delta_bits),
                u32(self.valence_delta_bits),
                u32(self.flags),
                u32(self.reserved),
            ]
        )


@dataclass(frozen=True)
class ExecutionReceiptV2:
    cycle_number: int
    subsystem_identity: str
    source_path: str
    schedule_interval: int
    urgency: str
    eligible_to_run: bool
    execution_outcome: str
    emitted: bool
    admitted: bool
    proposal: ProposalBitsV2 | None
    overlap_refs: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.urgency not in URGENCY:
            raise ValueError("unknown urgency")
        if self.execution_outcome not in EXECUTION_OUTCOME:
            raise ValueError("unknown execution outcome")
        if tuple(sorted(self.overlap_refs, key=lambda x: x.encode("ascii"))) != self.overlap_refs:
            raise ValueError("overlap refs must be bytewise sorted")
        if len(set(self.overlap_refs)) != len(self.overlap_refs):
            raise ValueError("duplicate overlap ref")

        outcome = self.execution_outcome
        no_emit = {
            "SKIPPED_SCHEDULE",
            "SKIPPED_HEALTH_DISABLED",
            "PANICKED_CAUGHT",
            "FAILED_OTHER",
        }
        if outcome in no_emit:
            if self.emitted or self.admitted or self.proposal is not None:
                raise ValueError("non-execution outcome cannot emit/admit proposal")
        elif outcome == "EXECUTED_NEUTRAL":
            if not self.emitted or self.admitted or self.proposal is None:
                raise ValueError("EXECUTED_NEUTRAL truth-table violation")
        elif outcome == "EXECUTED_NON_NEUTRAL":
            if not self.emitted or not self.admitted or self.proposal is None:
                raise ValueError("EXECUTED_NON_NEUTRAL truth-table violation")

    def canonical_bytes(self) -> bytes:
        self.validate()
        out = bytearray()
        out += u64(self.cycle_number)
        out += canonical_identity(self.subsystem_identity)
        out += canonical_path(self.source_path)
        out += u32(self.schedule_interval)
        out += u8(URGENCY[self.urgency])
        out += bool_byte(self.eligible_to_run)
        out += u8(EXECUTION_OUTCOME[self.execution_outcome])
        out += bool_byte(self.emitted)
        out += bool_byte(self.admitted)
        if self.proposal is None:
            out += b"\x00"
        else:
            out += b"\x01" + self.proposal.encode()
        out += u16(len(self.overlap_refs))
        for ref in self.overlap_refs:
            out += canonical_identity(ref)
        return bytes(out)

    def digest(self) -> bytes:
        return sha256(EXECUTION_V2_DOMAIN + self.canonical_bytes())


@dataclass(frozen=True)
class GuardWitnessEventV1:
    cycle_number: int
    witness_index: int
    predicate_id: int
    outcome: bool

    def canonical_bytes(self) -> bytes:
        if self.predicate_id not in VALID_PREDICATE_IDS:
            raise ValueError("unknown predicate ID")
        return (
            u64(self.cycle_number)
            + u16(self.witness_index)
            + u16(self.predicate_id)
            + bool_byte(self.outcome)
        )

    def digest(self) -> bytes:
        return sha256(GUARD_EVENT_DOMAIN + self.canonical_bytes())


@dataclass(frozen=True)
class GuardWitnessBundleV1:
    cycle_number: int
    overflow: bool
    events: tuple[GuardWitnessEventV1, ...]

    def validate(self) -> None:
        if len(self.events) > MAX_GUARD_WITNESSES:
            raise ValueError("guard witness capacity exceeded")
        pred_ids: list[int] = []
        for expected_index, event in enumerate(self.events):
            if event.cycle_number != self.cycle_number:
                raise ValueError("guard event cycle mismatch")
            if event.witness_index != expected_index:
                raise ValueError("guard witness indices must be contiguous from zero")
            event.canonical_bytes()
            pred_ids.append(event.predicate_id)
        if len(pred_ids) != len(set(pred_ids)):
            raise ValueError("duplicate predicate ID in guard bundle")

    def canonical_bytes(self) -> bytes:
        self.validate()
        out = bytearray()
        out += u64(self.cycle_number)
        out += bool_byte(self.overflow)
        out += u16(len(self.events))
        for event in self.events:
            out += u16(event.witness_index)
            out += event.digest()
        return bytes(out)

    def digest(self) -> bytes:
        return sha256(GUARD_BUNDLE_DOMAIN + self.canonical_bytes())


@dataclass(frozen=True)
class ExecutionEntry:
    subsystem_identity: str
    digest: bytes


@dataclass(frozen=True)
class ApplicationEntry:
    application_index: int
    digest: bytes


@dataclass(frozen=True)
class ObservationCycleV1:
    cycle_number: int
    executions: tuple[ExecutionEntry, ...]
    integration_digest: bytes
    applications: tuple[ApplicationEntry, ...]
    guard_bundle: GuardWitnessBundleV1
    manager_observer_overflow: bool
    application_observer_overflow: bool
    guard_observer_overflow: bool
    observer_buffers_complete: bool

    def validate(self) -> None:
        if len(self.integration_digest) != 32:
            raise ValueError("integration digest must be 32 bytes")
        identities = [entry.subsystem_identity for entry in self.executions]
        if len(identities) != len(set(identities)):
            raise ValueError("duplicate execution identity")
        for entry in self.executions:
            canonical_identity(entry.subsystem_identity)
            if len(entry.digest) != 32:
                raise ValueError("execution digest must be 32 bytes")
        for expected_index, app in enumerate(self.applications):
            if app.application_index != expected_index:
                raise ValueError("application indices must be contiguous from zero")
            if len(app.digest) != 32:
                raise ValueError("application digest must be 32 bytes")
        if self.guard_bundle.cycle_number != self.cycle_number:
            raise ValueError("guard bundle cycle mismatch")
        self.guard_bundle.validate()
        if self.guard_observer_overflow != self.guard_bundle.overflow:
            raise ValueError("guard overflow envelope/bundle mismatch")
        expected_complete = not (
            self.manager_observer_overflow
            or self.application_observer_overflow
            or self.guard_observer_overflow
        )
        if self.observer_buffers_complete != expected_complete:
            raise ValueError("observer_buffers_complete inconsistent with overflow bits")

    def canonical_bytes(self) -> bytes:
        self.validate()
        sorted_exec = sorted(self.executions, key=lambda e: e.subsystem_identity.encode("ascii"))
        out = bytearray()
        out += u64(self.cycle_number)
        out += u32(len(sorted_exec))
        for entry in sorted_exec:
            out += canonical_identity(entry.subsystem_identity)
            out += entry.digest
        out += self.integration_digest
        out += u32(len(self.applications))
        for app in self.applications:
            out += u32(app.application_index)
            out += app.digest
        out += self.guard_bundle.digest()
        out += bool_byte(self.manager_observer_overflow)
        out += bool_byte(self.application_observer_overflow)
        out += bool_byte(self.guard_observer_overflow)
        out += bool_byte(self.observer_buffers_complete)
        return bytes(out)

    def digest(self) -> bytes:
        return sha256(OBS_CYCLE_DOMAIN + self.canonical_bytes())


def observation_genesis(subject_manifest_digest: bytes) -> bytes:
    if len(subject_manifest_digest) != 32:
        raise ValueError("subject manifest digest must be 32 bytes")
    return sha256(OBS_GENESIS_DOMAIN + subject_manifest_digest)


def observation_chain_link(previous_root: bytes, cycle_digest: bytes) -> bytes:
    if len(previous_root) != 32 or len(cycle_digest) != 32:
        raise ValueError("chain inputs must be 32-byte digests")
    return sha256(OBS_CHAIN_DOMAIN + previous_root + cycle_digest)


def neutral_proposal(*, reserved: int = 0) -> ProposalBitsV2:
    return ProposalBitsV2(
        confidence_delta_bits=struct.unpack("<Q", struct.pack("<d", 0.0))[0],
        lr_modulation_bits=struct.unpack("<Q", struct.pack("<d", 1.0))[0],
        exploration_delta_bits=struct.unpack("<Q", struct.pack("<d", 0.0))[0],
        arousal_delta_bits=struct.unpack("<I", struct.pack("<f", 0.0))[0],
        valence_delta_bits=struct.unpack("<I", struct.pack("<f", 0.0))[0],
        flags=0,
        reserved=reserved,
    )


def sample_vectors() -> dict[str, str | int | bool]:
    neutral = ExecutionReceiptV2(
        cycle_number=42,
        subsystem_identity="drive_manager",
        source_path="src/cognitive_loop/managers/drive_manager.rs",
        schedule_interval=1,
        urgency="NORMAL",
        eligible_to_run=True,
        execution_outcome="EXECUTED_NEUTRAL",
        emitted=True,
        admitted=False,
        proposal=neutral_proposal(reserved=0),
    )
    reserved = ExecutionReceiptV2(
        cycle_number=42,
        subsystem_identity="drive_manager",
        source_path="src/cognitive_loop/managers/drive_manager.rs",
        schedule_interval=1,
        urgency="NORMAL",
        eligible_to_run=True,
        execution_outcome="EXECUTED_NEUTRAL",
        emitted=True,
        admitted=False,
        proposal=neutral_proposal(reserved=1),
    )
    memory = ExecutionReceiptV2(
        cycle_number=42,
        subsystem_identity="memory_manager",
        source_path="src/cognitive_loop/managers/memory_manager.rs",
        schedule_interval=1,
        urgency="NORMAL",
        eligible_to_run=False,
        execution_outcome="SKIPPED_SCHEDULE",
        emitted=False,
        admitted=False,
        proposal=None,
    )

    guard_true = GuardWitnessEventV1(42, 0, 1, True)
    guard_false = GuardWitnessEventV1(42, 0, 1, False)
    guard_events = (
        GuardWitnessEventV1(42, 0, 1, True),
        GuardWitnessEventV1(42, 1, 2, True),
        GuardWitnessEventV1(42, 2, 3, False),
    )
    bundle = GuardWitnessBundleV1(42, False, guard_events)

    observation = ObservationCycleV1(
        cycle_number=42,
        executions=(
            ExecutionEntry("memory_manager", memory.digest()),
            ExecutionEntry("drive_manager", neutral.digest()),
        ),
        integration_digest=bytes.fromhex("11" * 32),
        applications=(
            ApplicationEntry(0, bytes.fromhex("22" * 32)),
            ApplicationEntry(1, bytes.fromhex("33" * 32)),
        ),
        guard_bundle=bundle,
        manager_observer_overflow=False,
        application_observer_overflow=False,
        guard_observer_overflow=False,
        observer_buffers_complete=True,
    )
    overflow_observation = ObservationCycleV1(
        cycle_number=42,
        executions=observation.executions,
        integration_digest=observation.integration_digest,
        applications=observation.applications,
        guard_bundle=bundle,
        manager_observer_overflow=True,
        application_observer_overflow=False,
        guard_observer_overflow=False,
        observer_buffers_complete=False,
    )

    root0 = observation_genesis(bytes.fromhex("44" * 32))
    root1 = observation_chain_link(root0, observation.digest())

    return {
        "schema": "symthaea.spine.000b.observation-commitment-v2-vectors.v1",
        "neutral_execution_bytes_hex": neutral.canonical_bytes().hex(),
        "neutral_execution_digest_hex": neutral.digest().hex(),
        "reserved_execution_bytes_hex": reserved.canonical_bytes().hex(),
        "reserved_execution_digest_hex": reserved.digest().hex(),
        "guard_true_bytes_hex": guard_true.canonical_bytes().hex(),
        "guard_true_digest_hex": guard_true.digest().hex(),
        "guard_false_bytes_hex": guard_false.canonical_bytes().hex(),
        "guard_false_digest_hex": guard_false.digest().hex(),
        "guard_bundle_bytes_hex": bundle.canonical_bytes().hex(),
        "guard_bundle_digest_hex": bundle.digest().hex(),
        "observation_cycle_bytes_hex": observation.canonical_bytes().hex(),
        "observation_cycle_digest_hex": observation.digest().hex(),
        "overflow_observation_cycle_bytes_hex": overflow_observation.canonical_bytes().hex(),
        "overflow_observation_cycle_digest_hex": overflow_observation.digest().hex(),
        "observation_genesis_root_hex": root0.hex(),
        "observation_chain_root_1_hex": root1.hex(),
    }


def self_test() -> None:
    vectors = sample_vectors()
    assert vectors["neutral_execution_bytes_hex"] != vectors["reserved_execution_bytes_hex"]
    assert vectors["neutral_execution_digest_hex"] != vectors["reserved_execution_digest_hex"]
    assert vectors["guard_true_digest_hex"] != vectors["guard_false_digest_hex"]
    assert vectors["observation_cycle_digest_hex"] != vectors["overflow_observation_cycle_digest_hex"]

    # Absence of a witness is distinct from an evaluated false witness.
    empty_bundle = GuardWitnessBundleV1(42, False, ())
    false_bundle = GuardWitnessBundleV1(42, False, (GuardWitnessEventV1(42, 0, 1, False),))
    assert empty_bundle.digest() != false_bundle.digest()

    # Duplicate predicate IDs fail closed.
    try:
        GuardWitnessBundleV1(
            42,
            False,
            (
                GuardWitnessEventV1(42, 0, 1, True),
                GuardWitnessEventV1(42, 1, 1, False),
            ),
        ).canonical_bytes()
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate guard predicate IDs must fail")

    # Index gaps fail closed.
    try:
        GuardWitnessBundleV1(42, False, (GuardWitnessEventV1(42, 1, 1, True),)).canonical_bytes()
    except ValueError:
        pass
    else:
        raise AssertionError("guard witness index gaps must fail")

    # Execution input order canonicalizes by subsystem identity.
    base = sample_observation_for_tests(reverse=False)
    reversed_input = sample_observation_for_tests(reverse=True)
    assert base.canonical_bytes() == reversed_input.canonical_bytes()
    assert base.digest() == reversed_input.digest()

    # Application order/index is semantic and cannot be reordered silently.
    try:
        bad = ObservationCycleV1(
            cycle_number=base.cycle_number,
            executions=base.executions,
            integration_digest=base.integration_digest,
            applications=tuple(reversed(base.applications)),
            guard_bundle=base.guard_bundle,
            manager_observer_overflow=False,
            application_observer_overflow=False,
            guard_observer_overflow=False,
            observer_buffers_complete=True,
        )
        bad.canonical_bytes()
    except ValueError:
        pass
    else:
        raise AssertionError("application reordering must fail validation")

    # Guard overflow duplicated in bundle/envelope must agree.
    try:
        bad = ObservationCycleV1(
            cycle_number=base.cycle_number,
            executions=base.executions,
            integration_digest=base.integration_digest,
            applications=base.applications,
            guard_bundle=base.guard_bundle,
            manager_observer_overflow=False,
            application_observer_overflow=False,
            guard_observer_overflow=True,
            observer_buffers_complete=False,
        )
        bad.canonical_bytes()
    except ValueError:
        pass
    else:
        raise AssertionError("guard overflow mismatch must fail")

    # Chain position matters.
    root0 = observation_genesis(bytes.fromhex("44" * 32))
    root1 = observation_chain_link(root0, base.digest())
    root2 = observation_chain_link(root1, base.digest())
    assert root1 != root2


def sample_observation_for_tests(*, reverse: bool) -> ObservationCycleV1:
    neutral = ExecutionReceiptV2(
        42,
        "drive_manager",
        "src/cognitive_loop/managers/drive_manager.rs",
        1,
        "NORMAL",
        True,
        "EXECUTED_NEUTRAL",
        True,
        False,
        neutral_proposal(),
    )
    memory = ExecutionReceiptV2(
        42,
        "memory_manager",
        "src/cognitive_loop/managers/memory_manager.rs",
        1,
        "NORMAL",
        False,
        "SKIPPED_SCHEDULE",
        False,
        False,
        None,
    )
    executions = (
        ExecutionEntry("drive_manager", neutral.digest()),
        ExecutionEntry("memory_manager", memory.digest()),
    )
    if reverse:
        executions = tuple(reversed(executions))
    bundle = GuardWitnessBundleV1(
        42,
        False,
        (
            GuardWitnessEventV1(42, 0, 1, True),
            GuardWitnessEventV1(42, 1, 2, False),
        ),
    )
    return ObservationCycleV1(
        42,
        executions,
        bytes.fromhex("11" * 32),
        (ApplicationEntry(0, bytes.fromhex("22" * 32)),),
        bundle,
        False,
        False,
        False,
        True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--write-vectors", type=Path)
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("SPINE-000B-C2 observation commitment self-test: PASS")
        print("authority=measurement-only")
        print("runtime_evidence_claimed=false")
        print("causal_load_claimed=false")

    if args.write_vectors:
        args.write_vectors.parent.mkdir(parents=True, exist_ok=True)
        args.write_vectors.write_text(
            json.dumps(sample_vectors(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.write_vectors}")

    if not args.self_test and not args.write_vectors:
        parser.error("request --self-test and/or --write-vectors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
