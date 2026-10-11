#!/usr/bin/env python3
"""Independent PIE-003 composition / grade / constituent-balance oracle.

Research-only reference implementation. It imports no Symthaea code and makes no
claim about lunar/Mars process performance, purity, chemistry, or qualification.

Fractions are mass fractions on [0, 1]. Explicit constituent fractions describe
identified constituents. `unknown_fraction` is uncharacterized remainder and is
not silently assigned to any declared constituent. For an undeclared constituent,
the possible fraction is conservatively [0, unknown_fraction.max].
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from enum import Enum


class GradeStatus(str, Enum):
    PASS = "Pass"
    FAIL = "Fail"
    INDETERMINATE = "Indeterminate"


class BalanceStatus(str, Enum):
    EXACT_BALANCED = "ExactBalanced"
    EXACT_UNBALANCED = "ExactUnbalanced"
    POSSIBLE = "PossibleWithUncertainty"
    IMPOSSIBLE = "ImpossibleWithinBounds"


@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def validate(self, upper: float | None = None) -> None:
        if not (math.isfinite(self.lo) and math.isfinite(self.hi)):
            raise ValueError("interval endpoints must be finite")
        if self.lo < 0 or self.lo > self.hi:
            raise ValueError("invalid nonnegative interval")
        if upper is not None and self.hi > upper:
            raise ValueError("interval exceeds physical upper bound")

    @property
    def exact(self) -> bool:
        return self.lo == self.hi


@dataclass(frozen=True)
class Constituent:
    constituent_id: str
    fraction: Interval


@dataclass(frozen=True)
class Composition:
    constituents: tuple[Constituent, ...]
    unknown_fraction: Interval

    def validate(self, tolerance: float = 1e-12) -> None:
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("invalid composition tolerance")
        self.unknown_fraction.validate(1.0)
        ids: set[str] = set()
        for item in self.constituents:
            if not item.constituent_id.strip():
                raise ValueError("empty constituent id")
            if item.constituent_id in ids:
                raise ValueError("duplicate constituent id")
            ids.add(item.constituent_id)
            item.fraction.validate(1.0)
        minimum = sum(x.fraction.lo for x in self.constituents) + self.unknown_fraction.lo
        maximum = sum(x.fraction.hi for x in self.constituents) + self.unknown_fraction.hi
        if minimum > 1.0 + tolerance or maximum < 1.0 - tolerance:
            raise ValueError("composition intervals cannot contain a total fraction of 1")

    def declared_fraction(self, constituent_id: str) -> Interval | None:
        for item in self.constituents:
            if item.constituent_id == constituent_id:
                return item.fraction
        return None

    def possible_fraction(self, constituent_id: str) -> Interval:
        declared = self.declared_fraction(constituent_id)
        if declared is not None:
            return declared
        return Interval(0.0, self.unknown_fraction.hi)

    @property
    def fully_characterized_exact(self) -> bool:
        return (
            self.unknown_fraction == Interval(0.0, 0.0)
            and all(x.fraction.exact for x in self.constituents)
        )


@dataclass(frozen=True)
class Constraint:
    constituent_id: str
    minimum: float | None = None
    maximum: float | None = None

    def validate(self) -> None:
        if not self.constituent_id.strip() or (self.minimum is None and self.maximum is None):
            raise ValueError("invalid grade constraint")
        for value in (self.minimum, self.maximum):
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("grade fractions must lie in [0, 1]")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("grade minimum exceeds maximum")


@dataclass(frozen=True)
class GradeSpec:
    constraints: tuple[Constraint, ...]
    max_unknown_fraction: float | None = None


@dataclass(frozen=True)
class GradeReport:
    status: str
    reasons: tuple[str, ...]


def assess_grade(composition: Composition, spec: GradeSpec) -> GradeReport:
    composition.validate()
    seen: set[str] = set()
    indeterminate: list[str] = []

    for constraint in spec.constraints:
        constraint.validate()
        if constraint.constituent_id in seen:
            raise ValueError("duplicate grade constraint")
        seen.add(constraint.constituent_id)
        observed = composition.possible_fraction(constraint.constituent_id)

        if constraint.minimum is not None:
            if observed.hi < constraint.minimum:
                return GradeReport(
                    GradeStatus.FAIL.value,
                    (f"{constraint.constituent_id} cannot reach minimum",),
                )
            if observed.lo < constraint.minimum:
                indeterminate.append(
                    f"{constraint.constituent_id} is not established above minimum"
                )

        if constraint.maximum is not None:
            if observed.lo > constraint.maximum:
                return GradeReport(
                    GradeStatus.FAIL.value,
                    (f"{constraint.constituent_id} exceeds maximum",),
                )
            if observed.hi > constraint.maximum:
                indeterminate.append(
                    f"{constraint.constituent_id} is not established below maximum"
                )

    if spec.max_unknown_fraction is not None:
        limit = spec.max_unknown_fraction
        if not math.isfinite(limit) or not 0 <= limit <= 1:
            raise ValueError("invalid unknown-fraction limit")
        if composition.unknown_fraction.lo > limit:
            return GradeReport(
                GradeStatus.FAIL.value,
                ("unknown fraction necessarily exceeds maximum",),
            )
        if composition.unknown_fraction.hi > limit:
            indeterminate.append("unknown fraction is not established below maximum")

    if indeterminate:
        return GradeReport(GradeStatus.INDETERMINATE.value, tuple(indeterminate))
    return GradeReport(
        GradeStatus.PASS.value,
        ("all admissible compositions satisfy grade",),
    )


@dataclass(frozen=True)
class Lot:
    mass_kg: float
    composition: Composition


def mix_lots(lots: tuple[Lot, ...]) -> Composition:
    if not lots:
        raise ValueError("no lots")
    total = 0.0
    ids: set[str] = set()
    for lot in lots:
        if not math.isfinite(lot.mass_kg) or lot.mass_kg <= 0:
            raise ValueError("lot mass must be positive")
        lot.composition.validate()
        total += lot.mass_kg
        ids.update(x.constituent_id for x in lot.composition.constituents)

    mixed: list[Constituent] = []
    for constituent_id in sorted(ids):
        lo = 0.0
        hi = 0.0
        for lot in lots:
            declared = lot.composition.declared_fraction(constituent_id)
            if declared is not None:
                lo += lot.mass_kg * declared.lo
                hi += lot.mass_kg * declared.hi
        mixed.append(Constituent(constituent_id, Interval(lo / total, hi / total)))

    unknown = Interval(
        sum(lot.mass_kg * lot.composition.unknown_fraction.lo for lot in lots) / total,
        sum(lot.mass_kg * lot.composition.unknown_fraction.hi for lot in lots) / total,
    )
    result = Composition(tuple(mixed), unknown)
    result.validate()
    return result


@dataclass(frozen=True)
class Stream:
    mass_kg: Interval
    composition: Composition


@dataclass(frozen=True)
class BalanceReport:
    status: str
    input_kg: tuple[float, float]
    output_kg: tuple[float, float]
    residual_kg: tuple[float, float]
    tolerance_kg: float


def constituent_balance(
    constituent_id: str,
    inputs: tuple[Stream, ...],
    outputs: tuple[Stream, ...],
    absolute_tolerance_kg: float = 0.0,
    relative_tolerance: float = 0.0,
) -> BalanceReport:
    if not constituent_id.strip() or not inputs or not outputs:
        raise ValueError("constituent and both stream sets are required")
    for value in (absolute_tolerance_kg, relative_tolerance):
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid tolerance")

    def total(streams: tuple[Stream, ...]) -> tuple[float, float, bool]:
        lo = hi = 0.0
        exact = True
        for stream in streams:
            stream.mass_kg.validate()
            stream.composition.validate()
            fraction = stream.composition.possible_fraction(constituent_id)
            lo += stream.mass_kg.lo * fraction.lo
            hi += stream.mass_kg.hi * fraction.hi
            exact &= (
                stream.mass_kg.exact
                and stream.composition.fully_characterized_exact
            )
        return lo, hi, exact

    in_lo, in_hi, in_exact = total(inputs)
    out_lo, out_hi, out_exact = total(outputs)
    tolerance = max(
        absolute_tolerance_kg,
        relative_tolerance * max(in_hi, out_hi),
    )
    residual = (out_lo - in_hi, out_hi - in_lo)

    if in_exact and out_exact:
        status = (
            BalanceStatus.EXACT_BALANCED
            if abs(out_lo - in_lo) <= tolerance
            else BalanceStatus.EXACT_UNBALANCED
        )
    else:
        disjoint = in_hi + tolerance < out_lo or out_hi + tolerance < in_lo
        status = BalanceStatus.IMPOSSIBLE if disjoint else BalanceStatus.POSSIBLE

    return BalanceReport(
        status.value,
        (in_lo, in_hi),
        (out_lo, out_hi),
        residual,
        tolerance,
    )


def self_test() -> None:
    exact = Composition(
        (
            Constituent("A", Interval(0.6, 0.6)),
            Constituent("B", Interval(0.4, 0.4)),
        ),
        Interval(0.0, 0.0),
    )
    exact.validate()

    try:
        Composition(
            (Constituent("A", Interval(0.2, 0.3)),),
            Interval(0.0, 0.1),
        ).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("physically impossible composition must fail")

    spec = GradeSpec(
        (Constraint("A", minimum=0.55),),
        max_unknown_fraction=0.01,
    )
    assert assess_grade(exact, spec).status == GradeStatus.PASS.value
    assert (
        assess_grade(
            exact,
            GradeSpec((Constraint("A", minimum=0.7),)),
        ).status
        == GradeStatus.FAIL.value
    )

    uncertain = Composition(
        (
            Constituent("A", Interval(0.5, 0.65)),
            Constituent("B", Interval(0.3, 0.45)),
        ),
        Interval(0.0, 0.1),
    )
    uncertain.validate()
    assert assess_grade(uncertain, spec).status == GradeStatus.INDETERMINATE.value

    assert (
        assess_grade(
            exact,
            GradeSpec((Constraint("C", minimum=0.01),)),
        ).status
        == GradeStatus.FAIL.value
    )

    incomplete = Composition(
        (Constituent("A", Interval(0.8, 0.8)),),
        Interval(0.2, 0.2),
    )
    assert (
        assess_grade(
            incomplete,
            GradeSpec((Constraint("C", minimum=0.01),)),
        ).status
        == GradeStatus.INDETERMINATE.value
    )

    pure_a = Composition(
        (Constituent("A", Interval(1.0, 1.0)),),
        Interval(0.0, 0.0),
    )
    pure_b = Composition(
        (Constituent("B", Interval(1.0, 1.0)),),
        Interval(0.0, 0.0),
    )
    mixed = mix_lots((Lot(6.0, pure_a), Lot(4.0, pure_b)))
    assert mixed.declared_fraction("A") == Interval(0.6, 0.6)
    assert mixed.declared_fraction("B") == Interval(0.4, 0.4)

    source = Stream(Interval(10.0, 10.0), exact)
    out_a = Stream(Interval(6.0, 6.0), pure_a)
    out_b = Stream(Interval(4.0, 4.0), pure_b)
    assert (
        constituent_balance("A", (source,), (out_a, out_b)).status
        == BalanceStatus.EXACT_BALANCED.value
    )
    assert (
        constituent_balance("B", (source,), (out_a, out_b)).status
        == BalanceStatus.EXACT_BALANCED.value
    )

    wrong_a = Stream(Interval(5.0, 5.0), pure_a)
    wrong_b = Stream(Interval(5.0, 5.0), pure_b)
    assert (
        constituent_balance("A", (source,), (wrong_a, wrong_b)).status
        == BalanceStatus.EXACT_UNBALANCED.value
    )

    uncertain_in = Stream(Interval(9.0, 11.0), uncertain)
    uncertain_out = Stream(Interval(5.0, 7.0), pure_a)
    assert (
        constituent_balance("A", (uncertain_in,), (uncertain_out,)).status
        == BalanceStatus.POSSIBLE.value
    )

    unknown_stream = Stream(Interval(10.0, 10.0), incomplete)
    c_out = Stream(
        Interval(0.1, 0.1),
        Composition(
            (Constituent("C", Interval(1.0, 1.0)),),
            Interval(0.0, 0.0),
        ),
    )
    assert (
        constituent_balance("C", (unknown_stream,), (c_out,)).status
        == BalanceStatus.POSSIBLE.value
    )

    assert (
        assess_grade(
            incomplete,
            GradeSpec(
                (Constraint("A", minimum=0.75),),
                max_unknown_fraction=0.05,
            ),
        ).status
        == GradeStatus.FAIL.value
    )

    try:
        Composition(
            (
                Constituent("A", Interval(0.5, 0.5)),
                Constituent("A", Interval(0.5, 0.5)),
            ),
            Interval(0.0, 0.0),
        ).validate()
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate constituent ids must fail")

    for bad in (
        Interval(-0.1, 0.2),
        Interval(0.1, float("nan")),
        Interval(0.8, 1.1),
    ):
        try:
            bad.validate(1.0)
        except ValueError:
            pass
        else:
            raise AssertionError("malformed fraction must fail")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("ok")
        return
    parser.error("--self-test is required in PIE-003A")


if __name__ == "__main__":
    main()
