#!/usr/bin/env python3
"""Independent PIE-002F evidence-bearing supply/recovery-context oracle.

Proves one narrow property: every external numerical fact required by PIE-002D
is explicit, numerically valid, and carries explicit validated EvidenceRef
lineage. Evidence-bearing does not imply truth, freshness, independence,
applicability, feasibility, dispatch, or authority.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from enum import IntEnum

MAX_EVIDENCE_PER_FACT = 64
MAX_TEXT_BYTES = 4096


class EvidenceClass(IntEnum):
    HYPOTHESIS = 0
    LITERATURE_MODEL = 1
    VENDOR_PROJECTION = 2
    LAB_MEASURED = 3
    RELEVANT_ENVIRONMENT_MEASURED = 4
    INTEGRATED_DEMONSTRATION = 5
    QUALIFIED = 6


def _bounded_text(value: str, label: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{label}: text required")
    if len(value.encode("utf-8")) > MAX_TEXT_BYTES:
        raise ValueError(f"{label}: text too large")


def _canonical_id(value: str, label: str) -> None:
    _bounded_text(value, label)
    if not value or not value.strip():
        raise ValueError(f"{label}: required")
    if value != value.strip():
        raise ValueError(f"{label}: surrounding whitespace is non-canonical")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise ValueError(f"{label}: control character is non-canonical")


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    evidence_class: EvidenceClass
    source: str
    note: str | None = None

    def validate(self) -> None:
        _canonical_id(self.evidence_id, "evidence_id")
        if not isinstance(self.evidence_class, EvidenceClass):
            raise ValueError("evidence_class: unknown")
        _bounded_text(self.source, "evidence_source")
        if self.note is not None:
            _bounded_text(self.note, "evidence_note")
        if self.evidence_class is not EvidenceClass.HYPOTHESIS and not self.source.strip():
            raise ValueError("non-hypothesis evidence requires source")


@dataclass(frozen=True)
class Range:
    minimum: float
    maximum: float

    def validate(self, label: str, *, strictly_positive: bool = False) -> None:
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError(f"{label}: non-finite range")
        if self.minimum < 0.0 or self.maximum < 0.0 or self.minimum > self.maximum:
            raise ValueError(f"{label}: invalid range")
        if strictly_positive and self.minimum <= 0.0:
            raise ValueError(f"{label}: lower bound must be strictly positive")


@dataclass(frozen=True)
class FractionRange:
    minimum: float
    maximum: float

    def validate(self, label: str) -> None:
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError(f"{label}: non-finite fraction")
        if self.minimum < 0.0 or self.maximum > 1.0 or self.minimum > self.maximum:
            raise ValueError(f"{label}: expected 0 <= min <= max <= 1")


@dataclass(frozen=True)
class EvidenceBoundRange:
    value: Range
    evidence: tuple[EvidenceRef, ...]

    def validate(self, label: str, *, strictly_positive: bool = False) -> None:
        self.value.validate(label, strictly_positive=strictly_positive)
        _validate_evidence(self.evidence, label)


@dataclass(frozen=True)
class EvidenceBoundFraction:
    value: FractionRange
    evidence: tuple[EvidenceRef, ...]

    def validate(self, label: str) -> None:
        self.value.validate(label)
        _validate_evidence(self.evidence, label)


def _validate_evidence(evidence: tuple[EvidenceRef, ...], label: str) -> None:
    if not evidence:
        raise ValueError(f"{label}: evidence required")
    if len(evidence) > MAX_EVIDENCE_PER_FACT:
        raise ValueError(f"{label}: too many evidence refs")
    ids: set[str] = set()
    for item in evidence:
        item.validate()
        if item.evidence_id in ids:
            raise ValueError(f"{label}: duplicate evidence id {item.evidence_id}")
        ids.add(item.evidence_id)


@dataclass(frozen=True)
class BareSupplyRecoveryContext:
    recoverable_energy_j: Range
    recovery_duration_s: Range
    storage_acceptance_j: Range
    storage_charge_power_w: Range
    storage_discharge_power_w: Range
    recovery_delivery_fraction: FractionRange
    available_energy_capacity_j: Range
    available_sustained_power_w: Range
    available_peak_power_w: Range


@dataclass(frozen=True)
class EvidenceBearingSupplyRecoveryContext:
    recoverable_energy_j: EvidenceBoundRange
    recovery_duration_s: EvidenceBoundRange
    storage_acceptance_j: EvidenceBoundRange
    storage_charge_power_w: EvidenceBoundRange
    storage_discharge_power_w: EvidenceBoundRange
    recovery_delivery_fraction: EvidenceBoundFraction
    available_energy_capacity_j: EvidenceBoundRange
    available_sustained_power_w: EvidenceBoundRange
    available_peak_power_w: EvidenceBoundRange

    def validate(self) -> None:
        self.recoverable_energy_j.validate("recoverable_energy_j")
        self.recovery_duration_s.validate("recovery_duration_s", strictly_positive=True)
        self.storage_acceptance_j.validate("storage_acceptance_j")
        self.storage_charge_power_w.validate("storage_charge_power_w")
        self.storage_discharge_power_w.validate("storage_discharge_power_w")
        self.recovery_delivery_fraction.validate("recovery_delivery_fraction")
        self.available_energy_capacity_j.validate("available_energy_capacity_j")
        self.available_sustained_power_w.validate("available_sustained_power_w")
        self.available_peak_power_w.validate("available_peak_power_w")

    def bare_context(self) -> BareSupplyRecoveryContext:
        self.validate()
        return BareSupplyRecoveryContext(
            recoverable_energy_j=self.recoverable_energy_j.value,
            recovery_duration_s=self.recovery_duration_s.value,
            storage_acceptance_j=self.storage_acceptance_j.value,
            storage_charge_power_w=self.storage_charge_power_w.value,
            storage_discharge_power_w=self.storage_discharge_power_w.value,
            recovery_delivery_fraction=self.recovery_delivery_fraction.value,
            available_energy_capacity_j=self.available_energy_capacity_j.value,
            available_sustained_power_w=self.available_sustained_power_w.value,
            available_peak_power_w=self.available_peak_power_w.value,
        )


@dataclass(frozen=True)
class EvidenceBearingContextReceipt:
    bare_context: BareSupplyRecoveryContext
    recoverable_energy_evidence: tuple[EvidenceRef, ...]
    recovery_duration_evidence: tuple[EvidenceRef, ...]
    storage_acceptance_evidence: tuple[EvidenceRef, ...]
    storage_charge_power_evidence: tuple[EvidenceRef, ...]
    storage_discharge_power_evidence: tuple[EvidenceRef, ...]
    recovery_delivery_fraction_evidence: tuple[EvidenceRef, ...]
    available_energy_capacity_evidence: tuple[EvidenceRef, ...]
    available_sustained_power_evidence: tuple[EvidenceRef, ...]
    available_peak_power_evidence: tuple[EvidenceRef, ...]


def validate_evidence_bearing_context(
    context: EvidenceBearingSupplyRecoveryContext,
) -> EvidenceBearingContextReceipt:
    bare = context.bare_context()
    return EvidenceBearingContextReceipt(
        bare_context=bare,
        recoverable_energy_evidence=context.recoverable_energy_j.evidence,
        recovery_duration_evidence=context.recovery_duration_s.evidence,
        storage_acceptance_evidence=context.storage_acceptance_j.evidence,
        storage_charge_power_evidence=context.storage_charge_power_w.evidence,
        storage_discharge_power_evidence=context.storage_discharge_power_w.evidence,
        recovery_delivery_fraction_evidence=context.recovery_delivery_fraction.evidence,
        available_energy_capacity_evidence=context.available_energy_capacity_j.evidence,
        available_sustained_power_evidence=context.available_sustained_power_w.evidence,
        available_peak_power_evidence=context.available_peak_power_w.evidence,
    )


def _r(lo: float, hi: float | None = None) -> Range:
    return Range(lo, lo if hi is None else hi)


def _ev(
    eid: str,
    cls: EvidenceClass = EvidenceClass.HYPOTHESIS,
    source: str = "",
) -> EvidenceRef:
    return EvidenceRef(eid, cls, source, "synthetic fixture")


def _b(value: Range, *evidence: EvidenceRef) -> EvidenceBoundRange:
    return EvidenceBoundRange(value, evidence)


def baseline_context() -> EvidenceBearingSupplyRecoveryContext:
    shared = _ev("study-a", EvidenceClass.LITERATURE_MODEL, "doi:study-a")
    measured = _ev("lab-a", EvidenceClass.LAB_MEASURED, "report:lab-a")
    hypothesis = _ev("hyp-a")
    return EvidenceBearingSupplyRecoveryContext(
        recoverable_energy_j=_b(_r(30.0, 40.0), shared),
        recovery_duration_s=_b(_r(4.0, 5.0), measured),
        storage_acceptance_j=_b(_r(50.0, 60.0), shared),
        storage_charge_power_w=_b(_r(10.0, 12.0), measured),
        storage_discharge_power_w=_b(_r(8.0, 9.0), measured),
        recovery_delivery_fraction=EvidenceBoundFraction(FractionRange(0.8, 0.9), (shared,)),
        available_energy_capacity_j=_b(_r(120.0, 140.0), hypothesis),
        available_sustained_power_w=_b(_r(15.0, 20.0), shared),
        available_peak_power_w=_b(_r(30.0, 35.0), measured),
    )


def _must_fail(callable_, label: str) -> None:
    try:
        callable_()
    except ValueError:
        return
    raise AssertionError(label)


def self_test() -> None:
    context = baseline_context()
    receipt = validate_evidence_bearing_context(context)

    assert receipt.bare_context == BareSupplyRecoveryContext(
        recoverable_energy_j=_r(30.0, 40.0),
        recovery_duration_s=_r(4.0, 5.0),
        storage_acceptance_j=_r(50.0, 60.0),
        storage_charge_power_w=_r(10.0, 12.0),
        storage_discharge_power_w=_r(8.0, 9.0),
        recovery_delivery_fraction=FractionRange(0.8, 0.9),
        available_energy_capacity_j=_r(120.0, 140.0),
        available_sustained_power_w=_r(15.0, 20.0),
        available_peak_power_w=_r(30.0, 35.0),
    )

    assert receipt.recoverable_energy_evidence == context.recoverable_energy_j.evidence
    assert receipt.storage_acceptance_evidence == context.storage_acceptance_j.evidence
    assert (
        receipt.recoverable_energy_evidence[0].evidence_id
        == receipt.storage_acceptance_evidence[0].evidence_id
        == "study-a"
    )
    assert (
        receipt.available_energy_capacity_evidence[0].evidence_class
        is EvidenceClass.HYPOTHESIS
    )

    missing = replace(
        context,
        available_peak_power_w=EvidenceBoundRange(_r(30.0, 35.0), ()),
    )
    _must_fail(
        lambda: validate_evidence_bearing_context(missing),
        "missing evidence must fail",
    )

    bad_source = _ev("lab-b", EvidenceClass.LAB_MEASURED, "")
    malformed = replace(
        context,
        recovery_duration_s=_b(_r(4.0, 5.0), bad_source),
    )
    _must_fail(
        lambda: validate_evidence_bearing_context(malformed),
        "measured evidence needs source",
    )

    dup_a = _ev("dup", EvidenceClass.LITERATURE_MODEL, "doi:a")
    dup_b = _ev("dup", EvidenceClass.LAB_MEASURED, "report:b")
    duplicated = replace(
        context,
        recoverable_energy_j=_b(_r(30.0, 40.0), dup_a, dup_b),
    )
    _must_fail(
        lambda: validate_evidence_bearing_context(duplicated),
        "duplicate evidence id must fail",
    )

    shared = _ev("shared", EvidenceClass.LITERATURE_MODEL, "doi:shared")
    reused = replace(
        context,
        recoverable_energy_j=_b(_r(30.0, 40.0), shared),
        storage_acceptance_j=_b(_r(50.0, 60.0), shared),
    )
    reused_receipt = validate_evidence_bearing_context(reused)
    assert (
        reused_receipt.recoverable_energy_evidence[0]
        == reused_receipt.storage_acceptance_evidence[0]
    )

    invalid_numeric = replace(
        context,
        recovery_duration_s=_b(_r(0.0, 5.0), _ev("time")),
    )
    _must_fail(
        lambda: validate_evidence_bearing_context(invalid_numeric),
        "zero-inclusive recovery duration",
    )
    bad_fraction = replace(
        context,
        recovery_delivery_fraction=EvidenceBoundFraction(
            FractionRange(0.8, 1.1),
            (_ev("frac"),),
        ),
    )
    _must_fail(
        lambda: validate_evidence_bearing_context(bad_fraction),
        "invalid fraction",
    )

    aliased = replace(
        context,
        recoverable_energy_j=_b(_r(30.0, 40.0), _ev(" study-a")),
    )
    _must_fail(
        lambda: validate_evidence_bearing_context(aliased),
        "non-canonical evidence id",
    )

    too_many = tuple(_ev(f"e-{i}") for i in range(MAX_EVIDENCE_PER_FACT + 1))
    oversized = replace(
        context,
        recoverable_energy_j=EvidenceBoundRange(_r(30.0, 40.0), too_many),
    )
    _must_fail(
        lambda: validate_evidence_bearing_context(oversized),
        "evidence budget",
    )

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
