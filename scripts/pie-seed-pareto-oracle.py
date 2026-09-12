#!/usr/bin/env python3
"""PIE-009A independent synthetic seed-package / Pareto oracle.

This oracle is implementation-independent and uses synthetic fixtures only.
It does not claim Moon/Mars process performance, economics, or autonomous authority.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Iterable


@dataclass(frozen=True)
class SeedItem:
    item_id: str
    mass_t: float
    volume_m3: float
    capabilities: tuple[str, ...]
    critical_closure_gain: float
    productive_closure_gain: float
    useful_output_t_per_year: float
    resilience_gain: float
    imported_blocker_mass_t: float
    commissioning_days: float


@dataclass(frozen=True)
class SeedUniverse:
    items: tuple[SeedItem, ...]
    require_power_for_output: bool = True


@dataclass(frozen=True)
class PackageResult:
    item_ids: tuple[str, ...]
    seed_mass_t: float
    seed_volume_m3: float
    useful_output_t_per_year: float
    critical_closure: float
    productive_closure: float
    resilience: float
    imported_blocker_mass_t: float
    commissioning_days: float
    feasible: bool
    reasons: tuple[str, ...]


def _finite(*values: float) -> bool:
    return all(math.isfinite(v) for v in values)


def validate_universe(universe: SeedUniverse) -> None:
    ids: set[str] = set()
    for item in universe.items:
        if not item.item_id or item.item_id in ids:
            raise ValueError("seed item IDs must be unique and non-empty")
        ids.add(item.item_id)
        vals = (
            item.mass_t,
            item.volume_m3,
            item.critical_closure_gain,
            item.productive_closure_gain,
            item.useful_output_t_per_year,
            item.resilience_gain,
            item.imported_blocker_mass_t,
            item.commissioning_days,
        )
        if not _finite(*vals):
            raise ValueError("seed item values must be finite")
        if item.mass_t <= 0 or item.volume_m3 < 0:
            raise ValueError("seed mass must be positive and volume non-negative")
        if not (0 <= item.critical_closure_gain <= 1):
            raise ValueError("critical closure gain must be within [0,1]")
        if not (0 <= item.productive_closure_gain <= 1):
            raise ValueError("productive closure gain must be within [0,1]")
        if item.useful_output_t_per_year < 0 or item.resilience_gain < 0:
            raise ValueError("output/resilience gains must be non-negative")
        if item.commissioning_days < 0:
            raise ValueError("commissioning time must be non-negative")


def _ids(items: Iterable[SeedItem]) -> set[str]:
    return {x.item_id for x in items}


def evaluate_package(
    universe: SeedUniverse,
    selected_ids: Iterable[str],
    mass_budget_t: float,
    volume_budget_m3: float,
    local_control_alternative: bool = False,
) -> PackageResult:
    validate_universe(universe)
    if not _finite(mass_budget_t, volume_budget_m3) or mass_budget_t < 0 or volume_budget_m3 < 0:
        raise ValueError("budgets must be finite and non-negative")

    by_id = {x.item_id: x for x in universe.items}
    requested = tuple(sorted(set(selected_ids)))
    if any(item_id not in by_id for item_id in requested):
        raise ValueError("package references unknown seed item")
    items = [by_id[item_id] for item_id in requested]
    ids = _ids(items)

    mass = sum(x.mass_t for x in items)
    volume = sum(x.volume_m3 for x in items)
    reasons: list[str] = []
    if mass > mass_budget_t:
        reasons.append("mass_budget_exceeded")
    if volume > volume_budget_m3:
        reasons.append("volume_budget_exceeded")

    raw_output = sum(x.useful_output_t_per_year for x in items)
    useful_output = raw_output
    if universe.require_power_for_output and "power" not in ids:
        useful_output = 0.0
        if raw_output > 0:
            reasons.append("no_power")
    if "excavator" not in ids:
        useful_output = min(useful_output, 20.0)
    if "refinery" not in ids:
        useful_output = min(useful_output, 30.0)

    critical = min(1.0, sum(x.critical_closure_gain for x in items))
    productive = min(1.0, sum(x.productive_closure_gain for x in items))

    if "electronics" not in ids and not local_control_alternative:
        critical = min(critical, 0.55)
        productive = min(productive, 0.55)
    if "machine_shop" not in ids:
        productive = min(productive, 0.35)
    if "metrology" not in ids:
        productive = min(productive, 0.65)
    if "spares" not in ids:
        critical = min(critical, 0.70)

    resilience = min(1.0, sum(x.resilience_gain for x in items) / 4.0)
    blockers = max(0.0, sum(x.imported_blocker_mass_t for x in items))
    if local_control_alternative and "electronics" not in ids:
        blockers = max(0.0, blockers - 0.5)
        critical = min(1.0, critical + 0.15)
        productive = min(1.0, productive + 0.15)

    commissioning = max((x.commissioning_days for x in items), default=0.0)
    return PackageResult(
        item_ids=requested,
        seed_mass_t=mass,
        seed_volume_m3=volume,
        useful_output_t_per_year=useful_output,
        critical_closure=critical,
        productive_closure=productive,
        resilience=resilience,
        imported_blocker_mass_t=blockers,
        commissioning_days=commissioning,
        feasible=not reasons,
        reasons=tuple(reasons),
    )


def dominates(a: PackageResult, b: PackageResult) -> bool:
    """True only when a is no worse in every objective and better in at least one.

    Minimize: seed mass, volume, imported blocker mass, commissioning time.
    Maximize: useful output, critical closure, productive closure, resilience.
    """
    if not a.feasible:
        return False
    if not b.feasible:
        return True

    no_worse = (
        a.seed_mass_t <= b.seed_mass_t
        and a.seed_volume_m3 <= b.seed_volume_m3
        and a.imported_blocker_mass_t <= b.imported_blocker_mass_t
        and a.commissioning_days <= b.commissioning_days
        and a.useful_output_t_per_year >= b.useful_output_t_per_year
        and a.critical_closure >= b.critical_closure
        and a.productive_closure >= b.productive_closure
        and a.resilience >= b.resilience
    )
    strict = (
        a.seed_mass_t < b.seed_mass_t
        or a.seed_volume_m3 < b.seed_volume_m3
        or a.imported_blocker_mass_t < b.imported_blocker_mass_t
        or a.commissioning_days < b.commissioning_days
        or a.useful_output_t_per_year > b.useful_output_t_per_year
        or a.critical_closure > b.critical_closure
        or a.productive_closure > b.productive_closure
        or a.resilience > b.resilience
    )
    return no_worse and strict


def pareto_frontier(results: Iterable[PackageResult]) -> list[PackageResult]:
    feasible = [r for r in results if r.feasible]
    return [
        candidate
        for candidate in feasible
        if not any(
            dominates(other, candidate)
            for other in feasible
            if other != candidate
        )
    ]


def enumerate_packages(
    universe: SeedUniverse,
    mass_budget_t: float,
    volume_budget_m3: float,
    local_control_alternative: bool = False,
) -> list[PackageResult]:
    validate_universe(universe)
    ids = [x.item_id for x in universe.items]
    out: list[PackageResult] = []
    for n in range(len(ids) + 1):
        for chosen in combinations(ids, n):
            out.append(
                evaluate_package(
                    universe,
                    chosen,
                    mass_budget_t,
                    volume_budget_m3,
                    local_control_alternative,
                )
            )
    return out


def synthetic_universe() -> SeedUniverse:
    return SeedUniverse(
        (
            SeedItem("power", 4, 5, ("power",), 0.10, 0.05, 0, 0.30, 0.5, 2),
            SeedItem("excavator", 5, 8, ("excavate",), 0.10, 0.05, 50, 0.20, 1.0, 5),
            SeedItem("refinery", 8, 10, ("refine",), 0.15, 0.10, 100, 0.20, 2.0, 10),
            SeedItem("machine_shop", 7, 9, ("fabricate",), 0.20, 0.30, 30, 0.40, 1.5, 8),
            SeedItem("metrology", 3, 3, ("verify",), 0.10, 0.15, 0, 0.30, 0.3, 4),
            SeedItem("spares", 2, 2, ("maintain",), 0.10, 0.05, 0, 0.50, -1.0, 0),
            SeedItem("electronics", 1, 1, ("control",), 0.15, 0.15, 0, 0.20, -2.0, 0),
            SeedItem("recycler", 4, 5, ("recycle",), 0.10, 0.15, 20, 0.40, 0.2, 12),
        )
    )


def self_test() -> None:
    u = synthetic_universe()

    small = enumerate_packages(u, 20, 30)
    large = enumerate_packages(u, 30, 50)
    small_ids = {r.item_ids for r in small if r.feasible}
    large_ids = {r.item_ids for r in large if r.feasible}
    assert small_ids <= large_ids

    all_items = [x.item_id for x in u.items]
    over = evaluate_package(u, all_items, 10, 10)
    assert not over.feasible
    assert "mass_budget_exceeded" in over.reasons
    assert "volume_budget_exceeded" in over.reasons

    base_ids = ["power", "machine_shop", "metrology", "spares", "electronics"]
    with_e = evaluate_package(u, base_ids, 30, 50)
    without_e = evaluate_package(u, [x for x in base_ids if x != "electronics"], 30, 50)
    assert without_e.critical_closure <= with_e.critical_closure
    assert without_e.productive_closure <= with_e.productive_closure

    alt = evaluate_package(
        u,
        [x for x in base_ids if x != "electronics"],
        30,
        50,
        local_control_alternative=True,
    )
    assert alt.critical_closure >= without_e.critical_closure
    assert alt.productive_closure >= without_e.productive_closure
    assert alt.imported_blocker_mass_t <= without_e.imported_blocker_mass_t

    results = enumerate_packages(u, 30, 50)
    front = pareto_frontier(results)
    assert len(front) > 1
    assert any(r.useful_output_t_per_year >= 150 for r in front)
    assert any(r.productive_closure >= 0.8 for r in front)

    bulk = evaluate_package(u, ["power", "excavator", "refinery"], 30, 50)
    closure = evaluate_package(
        u, ["power", "machine_shop", "metrology", "spares", "electronics"], 30, 50
    )
    assert bulk.useful_output_t_per_year > closure.useful_output_t_per_year
    assert bulk.productive_closure < closure.productive_closure

    no_power = evaluate_package(u, ["excavator", "refinery"], 30, 50)
    assert no_power.useful_output_t_per_year == 0
    assert "no_power" in no_power.reasons

    bad = SeedUniverse((u.items[0], u.items[0]))
    try:
        validate_universe(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate item IDs must fail")

    for candidate in front:
        assert candidate.feasible
        assert not any(
            dominates(other, candidate)
            for other in results
            if other.feasible and other != candidate
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mass-budget-t", type=float, default=30.0)
    parser.add_argument("--volume-budget-m3", type=float, default=50.0)
    parser.add_argument("--local-control-alternative", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("ok")
        return

    universe = synthetic_universe()
    results = enumerate_packages(
        universe,
        args.mass_budget_t,
        args.volume_budget_m3,
        args.local_control_alternative,
    )
    front = pareto_frontier(results)
    payload = {
        "mass_budget_t": args.mass_budget_t,
        "volume_budget_m3": args.volume_budget_m3,
        "feasible_packages": sum(1 for r in results if r.feasible),
        "pareto_frontier": [asdict(r) for r in front],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
