#!/usr/bin/env python3
"""Independent PIE Phase-0 resource-access oracle.

Research-only semantics for occurrence -> acquisition -> transport -> delivered
feedstock. No Moon/Mars abundance, mining yield, transport performance, or
operational authority is implied.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from enum import Enum


class Feasibility(str, Enum):
    GUARANTEED = "Guaranteed"
    POSSIBLE = "Possible"
    IMPOSSIBLE = "Impossible"


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
            raise ValueError("interval exceeds upper bound")


@dataclass(frozen=True)
class ResourceOccurrence:
    resource_id: str
    site_id: str
    in_place_mass_kg: Interval


@dataclass(frozen=True)
class AcquisitionModel:
    recovered_fraction: Interval


@dataclass(frozen=True)
class TransportEdge:
    source_site: str
    target_site: str
    capacity_kg_per_window: Interval
    delivered_fraction: Interval
    energy_j_per_shipped_kg: Interval
    transit_time_s: Interval
    distance_m: float
    available: bool = True


@dataclass(frozen=True)
class AccessResult:
    recoverable_at_source_kg: tuple[float, float]
    deliverable_at_target_kg: tuple[float, float]
    transport_energy_at_delivery_limit_j: tuple[float, float]
    feasibility: str


def mul(a: Interval, b: Interval) -> Interval:
    return Interval(a.lo * b.lo, a.hi * b.hi)


def assess_access(
    occurrence: ResourceOccurrence,
    acquisition: AcquisitionModel,
    target_site: str,
    demand_kg: Interval,
    transport: TransportEdge | None = None,
) -> AccessResult:
    if not occurrence.resource_id.strip() or not occurrence.site_id.strip() or not target_site.strip():
        raise ValueError("resource/site ids are required")
    occurrence.in_place_mass_kg.validate()
    acquisition.recovered_fraction.validate(1.0)
    demand_kg.validate()

    recovered = mul(occurrence.in_place_mass_kg, acquisition.recovered_fraction)

    if occurrence.site_id == target_site:
        delivered = recovered
        energy = Interval(0.0, 0.0)
    else:
        if transport is None or not transport.available:
            return AccessResult(
                (recovered.lo, recovered.hi),
                (0.0, 0.0),
                (0.0, 0.0),
                Feasibility.IMPOSSIBLE.value,
            )
        if transport.source_site != occurrence.site_id or transport.target_site != target_site:
            raise ValueError("transport edge does not connect occurrence to target")
        transport.capacity_kg_per_window.validate()
        transport.delivered_fraction.validate(1.0)
        transport.energy_j_per_shipped_kg.validate()
        transport.transit_time_s.validate()
        if not math.isfinite(transport.distance_m) or transport.distance_m < 0:
            raise ValueError("invalid transport distance")

        shipped = Interval(
            min(recovered.lo, transport.capacity_kg_per_window.lo),
            min(recovered.hi, transport.capacity_kg_per_window.hi),
        )
        delivered = mul(shipped, transport.delivered_fraction)
        energy = mul(shipped, transport.energy_j_per_shipped_kg)

    if delivered.lo >= demand_kg.hi:
        status = Feasibility.GUARANTEED
    elif delivered.hi < demand_kg.lo:
        status = Feasibility.IMPOSSIBLE
    else:
        status = Feasibility.POSSIBLE

    return AccessResult(
        (recovered.lo, recovered.hi),
        (delivered.lo, delivered.hi),
        (energy.lo, energy.hi),
        status.value,
    )


def self_test() -> None:
    occurrence = ResourceOccurrence("resource", "mine", Interval(100.0, 120.0))
    acquisition = AcquisitionModel(Interval(0.5, 0.6))
    edge = TransportEdge(
        "mine", "plant",
        Interval(40.0, 80.0),
        Interval(0.9, 0.95),
        Interval(10.0, 12.0),
        Interval(100.0, 120.0),
        500_000.0,
        True,
    )

    local = assess_access(occurrence, acquisition, "mine", Interval(40.0, 40.0))
    assert local.recoverable_at_source_kg == (50.0, 72.0)
    assert local.deliverable_at_target_kg == (50.0, 72.0)
    assert local.transport_energy_at_delivery_limit_j == (0.0, 0.0)
    assert local.feasibility == Feasibility.GUARANTEED.value

    missing = assess_access(occurrence, acquisition, "plant", Interval(1.0, 1.0))
    assert missing.deliverable_at_target_kg == (0.0, 0.0)
    assert missing.feasibility == Feasibility.IMPOSSIBLE.value

    result = assess_access(occurrence, acquisition, "plant", Interval(30.0, 30.0), edge)
    assert result.deliverable_at_target_kg == (36.0, 68.39999999999999)
    assert result.feasibility == Feasibility.GUARANTEED.value

    uncertain = assess_access(occurrence, acquisition, "plant", Interval(50.0, 50.0), edge)
    assert uncertain.feasibility == Feasibility.POSSIBLE.value

    impossible = assess_access(occurrence, acquisition, "plant", Interval(70.0, 70.0), edge)
    assert impossible.feasibility == Feasibility.IMPOSSIBLE.value

    exact_occ = ResourceOccurrence("resource", "mine", Interval(120.0, 120.0))
    exact_acq = AcquisitionModel(Interval(0.6, 0.6))
    exact_edge = TransportEdge(
        "mine", "plant",
        Interval(80.0, 80.0), Interval(0.95, 0.95),
        Interval(10.0, 10.0), Interval(100.0, 100.0), 500_000.0, True,
    )
    exact = assess_access(exact_occ, exact_acq, "plant", Interval(60.0, 60.0), exact_edge)
    assert exact.feasibility == Feasibility.GUARANTEED.value
    wider = assess_access(occurrence, acquisition, "plant", Interval(60.0, 60.0), edge)
    assert wider.feasibility == Feasibility.POSSIBLE.value

    closed = TransportEdge(**{**edge.__dict__, "available": False})
    assert assess_access(occurrence, acquisition, "plant", Interval(1.0, 1.0), closed).feasibility == Feasibility.IMPOSSIBLE.value

    try:
        wrong = TransportEdge(**{**edge.__dict__, "target_site": "other"})
        assess_access(occurrence, acquisition, "plant", Interval(1.0, 1.0), wrong)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched transport edge must fail")

    try:
        assess_access(occurrence, AcquisitionModel(Interval(0.9, 1.1)), "mine", Interval(1.0, 1.0))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid recovery fraction must fail")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("ok")
        return
    parser.error("--self-test is required in the reference oracle")


if __name__ == "__main__":
    main()
