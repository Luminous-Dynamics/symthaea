#!/usr/bin/env python3
"""Independent PIE-002H-A full ProcessDefinition content-commitment oracle.

Freezes a canonical, bounded, versioned, domain-separated SHA-256 commitment
over the complete PIE ProcessDefinition semantic surface.

Content identity != currentness, authenticity, truth, applicability, feasibility,
dispatch, or execution authority.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import struct
from dataclasses import dataclass, replace

DOMAIN = b"symthaea.pie.process-content.v1\x00"
MAX_STRING_BYTES = 4096
MAX_COLLECTION_ITEMS = 1024
MAX_CANONICAL_BYTES = 1_048_576
BASELINE_SHA256 = "423cb41801ddfeb75e2c5d4a13d7edb3f108a2825f2965d4e669ee934736440d"


def _sized_text(value: str, label: str) -> bytes:
    if not isinstance(value, str):
        raise ValueError(f"{label}: text required")
    encoded = value.encode("utf-8")
    if len(encoded) > MAX_STRING_BYTES:
        raise ValueError(f"{label}: string too large")
    return encoded


def _required_text(value: str, label: str) -> bytes:
    encoded = _sized_text(value, label)
    if not value.strip():
        raise ValueError(f"{label}: required")
    return encoded


def _canonical_id(value: str, label: str) -> bytes:
    encoded = _required_text(value, label)
    if value != value.strip():
        raise ValueError(f"{label}: surrounding whitespace is non-canonical")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise ValueError(f"{label}: control character is non-canonical")
    return encoded


def _u8(value: int) -> bytes:
    return struct.pack(">B", value)


def _u32(value: int) -> bytes:
    return struct.pack(">I", value)


def _lp(value: bytes) -> bytes:
    return _u32(len(value)) + value


def _text(value: str, label: str = "text") -> bytes:
    return _lp(_sized_text(value, label))


def _required_text_bytes(value: str, label: str) -> bytes:
    return _lp(_required_text(value, label))


def _id(value: str, label: str = "id") -> bytes:
    return _lp(_canonical_id(value, label))


def _optional_text(value: str | None, label: str) -> bytes:
    return b"\x00" if value is None else b"\x01" + _text(value, label)


@dataclass(frozen=True)
class Range:
    minimum: float
    maximum: float


def _range(value: Range, label: str) -> bytes:
    if (
        not math.isfinite(value.minimum)
        or not math.isfinite(value.maximum)
        or value.minimum < 0.0
        or value.maximum < 0.0
        or value.minimum > value.maximum
    ):
        raise ValueError(f"{label}: invalid range")

    def encode_f64(number: float) -> bytes:
        # PIE's Rust scalar equality treats -0.0 == +0.0. Commit that semantic
        # value once rather than allowing two encodings for equal quantities.
        canonical = 0.0 if number == 0.0 else number
        return struct.pack(">d", canonical)

    return encode_f64(value.minimum) + encode_f64(value.maximum)


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    evidence_class: int
    source: str
    note: str | None = None


def _evidence(value: Evidence) -> bytes:
    _canonical_id(value.evidence_id, "evidence_id")
    if not 0 <= value.evidence_class <= 6:
        raise ValueError("evidence_class: unknown discriminant")
    if value.evidence_class != 0 and not value.source.strip():
        raise ValueError("non-hypothesis evidence requires source")
    return (
        _id(value.evidence_id, "evidence_id")
        + _u8(value.evidence_class)
        + _text(value.source, "evidence_source")
        + _optional_text(value.note, "evidence_note")
    )


def _multiset(values: tuple, encoder, label: str) -> bytes:
    if len(values) > MAX_COLLECTION_ITEMS:
        raise ValueError(f"{label}: too many items")
    encoded = sorted(encoder(value) for value in values)
    return _u32(len(encoded)) + b"".join(_lp(item) for item in encoded)


def _evidence_set(values: tuple[Evidence, ...]) -> bytes:
    ids: set[str] = set()
    for item in values:
        if item.evidence_id in ids:
            raise ValueError(f"duplicate evidence id: {item.evidence_id}")
        ids.add(item.evidence_id)
    return _multiset(values, _evidence, "evidence")


@dataclass(frozen=True)
class ProcessDefinition:
    process_id: str
    name: str
    # Input tuple: material_key, optional (grade_label, specification_ref), role, mass range.
    inputs: tuple
    # Output tuple: material_key, grade, physical_form, role, mass, disposition.
    outputs: tuple
    # Utility tuple: utility-kind discriminant, range.
    utilities: tuple
    # Equipment tuple: class key, quantity, criticality, evidence tuple.
    equipment: tuple
    # Environment tuple: constraint-kind discriminant, payload.
    environment: tuple
    evidence: tuple[Evidence, ...]


def _grade(value: tuple[str, str | None]) -> bytes:
    label, specification_ref = value
    return _required_text_bytes(label, "material_grade") + _optional_text(
        specification_ref, "material_specification_ref"
    )


def _input(value: tuple) -> bytes:
    material_key, required_grade, role, mass = value
    if not 0 <= role <= 4:
        raise ValueError("process_input_role: unknown discriminant")
    grade = b"\x00" if required_grade is None else b"\x01" + _grade(required_grade)
    return (
        _id(material_key, "process_input_material_key")
        + grade
        + _u8(role)
        + _range(mass, "input_mass_kg")
    )


def _disposition(value: tuple[int, str | None]) -> bytes:
    kind, target = value
    if not 0 <= kind <= 3:
        raise ValueError("output_disposition: unknown discriminant")
    if kind in (1, 2):
        if target is None:
            raise ValueError("output disposition target required")
        return _u8(kind) + _id(target, "output_destination_id")
    if target is not None:
        raise ValueError("non-target output disposition carries target")
    return _u8(kind)


def _output(value: tuple) -> bytes:
    material_key, grade, physical_form, role, mass, disposition = value
    if not 0 <= physical_form <= 12 or not 0 <= role <= 4:
        raise ValueError("process_output: unknown enum discriminant")
    return (
        _id(material_key, "process_output_material_key")
        + _grade(grade)
        + _u8(physical_form)
        + _u8(role)
        + _range(mass, "output_mass_kg")
        + _disposition(disposition)
    )


def _utility(value: tuple[int, Range]) -> bytes:
    kind, quantity = value
    if not 0 <= kind <= 4:
        raise ValueError("utility_demand: unknown discriminant")
    return _u8(kind) + _range(quantity, "utility")


def _equipment(value: tuple) -> bytes:
    equipment_class, quantity, criticality, evidence = value
    if not 1 <= quantity <= 0xFFFFFFFF or not 0 <= criticality <= 2:
        raise ValueError("equipment requirement invalid")
    return (
        _id(equipment_class, "equipment_class")
        + _u32(quantity)
        + _u8(criticality)
        + _evidence_set(evidence)
    )


def _environment(value: tuple) -> bytes:
    kind, payload = value
    if not 0 <= kind <= 4:
        raise ValueError("environment constraint: unknown discriminant")
    if kind == 0:  # Body(CelestialBody)
        if not isinstance(payload, int) or not 0 <= payload <= 3:
            raise ValueError("celestial body: unknown discriminant")
        return _u8(kind) + _u8(payload)
    if kind in (1, 2, 3):  # Temperature, Pressure, Gravity
        if not isinstance(payload, Range):
            raise ValueError("numeric environment constraint requires range")
        return _u8(kind) + _range(payload, "environment_range")
    if payload is not None:  # VacuumCompatible
        raise ValueError("vacuum-compatible constraint has no payload")
    return _u8(kind)


def canonical_process_bytes(process: ProcessDefinition) -> bytes:
    # Commitment admission is deliberately stricter than legacy PIE-000:
    # durable IDs/references must already be in canonical lexical form.
    _canonical_id(process.process_id, "process_id")
    _required_text(process.name, "process_name")
    if not process.inputs:
        raise ValueError("process requires at least one input")
    if not process.outputs:
        raise ValueError("process requires at least one output")

    payload = (
        DOMAIN
        + _id(process.process_id, "process_id")
        + _required_text_bytes(process.name, "process_name")
        + _multiset(process.inputs, _input, "inputs")
        + _multiset(process.outputs, _output, "outputs")
        + _multiset(process.utilities, _utility, "utilities")
        + _multiset(process.equipment, _equipment, "equipment")
        + _multiset(process.environment, _environment, "environment")
        + _evidence_set(process.evidence)
    )
    if len(payload) > MAX_CANONICAL_BYTES:
        raise ValueError("canonical process content exceeds byte budget")
    return payload


def process_content_sha256(process: ProcessDefinition) -> str:
    return hashlib.sha256(canonical_process_bytes(process)).hexdigest()


def _r(minimum: float, maximum: float | None = None) -> Range:
    return Range(minimum, minimum if maximum is None else maximum)


def baseline_process() -> ProcessDefinition:
    literature = Evidence("ev-a", 1, "doi:a", "basis")
    hypothesis = Evidence("ev-b", 0, "", "exploratory")
    return ProcessDefinition(
        process_id="proc-1",
        name="Reference reduction",
        inputs=(
            ("regolith", None, 0, _r(10.0, 12.0)),
            ("argon", None, 3, _r(1.0, 1.2)),
        ),
        outputs=(
            ("metal", ("Al-6061", "ASTM-B209"), 3, 0, _r(6.0, 7.0), (0, None)),
            ("tailings", ("tailings", None), 1, 3, _r(4.0, 5.0), (1, "tailings-vault")),
        ),
        utilities=(
            (0, _r(100.0, 120.0)),
            (3, _r(20.0, 25.0)),
            (4, _r(9.0, 11.0)),
            (1, _r(50.0, 60.0)),
        ),
        equipment=(("reactor-v1", 2, 0, (literature,)),),
        environment=((0, 1), (1, _r(240.0, 320.0)), (4, None)),
        evidence=(literature, hypothesis),
    )


def _must_fail(callable_, label: str) -> None:
    try:
        callable_()
    except ValueError:
        return
    raise AssertionError(label)


def self_test() -> None:
    base = baseline_process()
    digest = process_content_sha256(base)
    assert digest == BASELINE_SHA256

    # Vec ordering is not identity-bearing in this commitment profile; all
    # records are committed as canonical multisets. Multiplicity is retained.
    reordered = replace(
        base,
        inputs=tuple(reversed(base.inputs)),
        outputs=tuple(reversed(base.outputs)),
        utilities=tuple(reversed(base.utilities)),
        equipment=tuple(reversed(base.equipment)),
        environment=tuple(reversed(base.environment)),
        evidence=tuple(reversed(base.evidence)),
    )
    assert process_content_sha256(reordered) == digest
    assert process_content_sha256(replace(base, utilities=base.utilities + (base.utilities[1],))) != digest

    # Every major ProcessDefinition surface participates in identity.
    assert process_content_sha256(replace(base, name=base.name + " v2")) != digest
    assert process_content_sha256(
        replace(base, utilities=((0, _r(101.0, 121.0)),) + base.utilities[1:])
    ) != digest
    assert process_content_sha256(
        replace(base, inputs=(("regolith", None, 0, _r(10.0, 13.0)),) + base.inputs[1:])
    ) != digest
    assert process_content_sha256(
        replace(base, outputs=(
            ("metal", ("Al-6061", "ASTM-B209"), 3, 0, _r(6.0, 7.5), (0, None)),
        ) + base.outputs[1:])
    ) != digest
    assert process_content_sha256(
        replace(base, equipment=(("reactor-v1", 3, 0, base.equipment[0][3]),))
    ) != digest
    assert process_content_sha256(
        replace(base, environment=((0, 2),) + base.environment[1:])
    ) != digest
    assert process_content_sha256(
        replace(base, evidence=(
            Evidence("ev-a", 1, "doi:a", "revised interpretation"),
            base.evidence[1],
        ))
    ) != digest

    # Semantic signed zero has exactly one canonical encoding.
    zero_positive = replace(base, utilities=((2, Range(0.0, 0.0)),) + base.utilities)
    zero_negative = replace(base, utilities=((2, Range(-0.0, +0.0)),) + base.utilities)
    assert process_content_sha256(zero_positive) == process_content_sha256(zero_negative)

    # Durable lexical IDs fail closed rather than silently trim/normalize.
    _must_fail(
        lambda: process_content_sha256(replace(base, process_id=" proc-1")),
        "leading-whitespace process identity must fail",
    )
    _must_fail(
        lambda: process_content_sha256(
            replace(base, inputs=(("regolith ", None, 0, _r(10.0, 12.0)),) + base.inputs[1:])
        ),
        "non-canonical material key must fail",
    )
    _must_fail(
        lambda: process_content_sha256(replace(base, evidence=(Evidence(" ev-a", 1, "doi:a"),))),
        "non-canonical evidence id must fail",
    )
    _must_fail(
        lambda: process_content_sha256(
            replace(base, outputs=(
                ("metal", ("Al-6061", "ASTM-B209"), 3, 0, _r(6.0, 7.0), (2, " next-process")),
            ) + base.outputs[1:])
        ),
        "non-canonical process reference must fail",
    )

    duplicate_evidence = replace(base, evidence=(base.evidence[0], base.evidence[0]))
    _must_fail(lambda: process_content_sha256(duplicate_evidence), "duplicate evidence id must fail")

    nonfinite = replace(base, utilities=((0, Range(1.0, math.inf)),) + base.utilities[1:])
    _must_fail(lambda: process_content_sha256(nonfinite), "non-finite quantity must fail")

    too_many = replace(
        base,
        utilities=tuple((1, _r(float(index))) for index in range(MAX_COLLECTION_ITEMS + 1)),
    )
    _must_fail(lambda: process_content_sha256(too_many), "collection budget must fail")

    too_long = replace(base, name="x" * (MAX_STRING_BYTES + 1))
    _must_fail(lambda: process_content_sha256(too_long), "string budget must fail")

    print(BASELINE_SHA256)
    print("ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.error("only --self-test is supported")


if __name__ == "__main__":
    main()
