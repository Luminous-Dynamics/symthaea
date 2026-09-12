#!/usr/bin/env python3
"""Independent PIE Phase-0 equipment lifecycle / spares oracle."""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class MaintenancePolicy:
    interval_h: int
    downtime_h: int
    spare_id: str
    spare_qty: int


@dataclass(frozen=True)
class Machine:
    machine_id: str
    policy: MaintenancePolicy | None
    locally_repairable: bool
    locally_reproducible: bool


@dataclass(frozen=True)
class SpareProduction:
    hour: int
    spare_id: str
    qty: int


@dataclass(frozen=True)
class Failure:
    hour: int
    machine_ids: tuple[str, ...]
    spare_id: str
    spare_qty_per_machine: int
    repair_downtime_h: int


@dataclass
class State:
    operational: bool = True
    stopped_at_h: int | None = None
    downtime_h: int = 0
    spares_consumed: int = 0


@dataclass(frozen=True)
class MachineReport:
    machine_id: str
    operational_end: bool
    stopped_at_h: int | None
    uptime_h: int
    downtime_h: int
    spares_consumed: int
    locally_repairable: bool
    locally_reproducible: bool


def validate_machine(machine: Machine) -> None:
    if not machine.machine_id.strip():
        raise ValueError("machine id required")
    if machine.policy is not None:
        p = machine.policy
        if p.interval_h <= 0 or p.downtime_h < 0 or p.spare_qty < 0 or not p.spare_id.strip():
            raise ValueError("invalid maintenance policy")


def simulate(
    horizon_h: int,
    machines: tuple[Machine, ...],
    initial_spares: dict[str, int],
    productions: tuple[SpareProduction, ...] = (),
    failures: tuple[Failure, ...] = (),
) -> tuple[list[MachineReport], dict[str, int]]:
    if horizon_h <= 0:
        raise ValueError("horizon must be positive")
    ids = [m.machine_id for m in machines]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate machine id")
    for machine in machines:
        validate_machine(machine)

    spares: defaultdict[str, int] = defaultdict(int)
    for spare_id, qty in initial_spares.items():
        if not spare_id.strip() or qty < 0:
            raise ValueError("invalid spare inventory")
        spares[spare_id] += qty

    events: defaultdict[int, list[tuple[str, object]]] = defaultdict(list)
    for production in productions:
        if production.hour < 0 or production.hour > horizon_h or production.qty < 0 or not production.spare_id.strip():
            raise ValueError("invalid spare-production event")
        events[production.hour].append(("produce", production))

    for failure in failures:
        if (
            failure.hour < 0
            or failure.hour > horizon_h
            or failure.spare_qty_per_machine < 0
            or failure.repair_downtime_h < 0
            or not failure.spare_id.strip()
        ):
            raise ValueError("invalid failure event")
        for machine_id in failure.machine_ids:
            if machine_id not in ids:
                raise ValueError("failure references unknown machine")
        events[failure.hour].append(("failure", failure))

    for machine in machines:
        if machine.policy is None:
            continue
        due = machine.policy.interval_h
        while due <= horizon_h:
            events[due].append(("service", machine.machine_id))
            due += machine.policy.interval_h

    states = {machine.machine_id: State() for machine in machines}
    by_id = {machine.machine_id: machine for machine in machines}

    for hour in sorted(events):
        # Material produced at a timestamp may be consumed by service/failure work
        # at that same timestamp; future output cannot be consumed earlier.
        for kind, payload in events[hour]:
            if kind == "produce":
                production = payload
                spares[production.spare_id] += production.qty

        for kind, payload in events[hour]:
            if kind != "service":
                continue
            machine_id = payload
            state = states[machine_id]
            machine = by_id[machine_id]
            policy = machine.policy
            if not state.operational or policy is None:
                continue
            if policy.spare_qty and spares[policy.spare_id] < policy.spare_qty:
                state.operational = False
                state.stopped_at_h = hour
                continue
            spares[policy.spare_id] -= policy.spare_qty
            state.spares_consumed += policy.spare_qty
            state.downtime_h += policy.downtime_h

        for kind, payload in events[hour]:
            if kind != "failure":
                continue
            failure = payload
            # Stable ordering makes scarce-spare allocation deterministic in the
            # reference model. Production policy may later use a richer allocator.
            for machine_id in sorted(failure.machine_ids):
                state = states[machine_id]
                machine = by_id[machine_id]
                if not state.operational:
                    continue
                if (
                    not machine.locally_repairable
                    or spares[failure.spare_id] < failure.spare_qty_per_machine
                ):
                    state.operational = False
                    state.stopped_at_h = hour
                    continue
                spares[failure.spare_id] -= failure.spare_qty_per_machine
                state.spares_consumed += failure.spare_qty_per_machine
                state.downtime_h += failure.repair_downtime_h

    reports: list[MachineReport] = []
    for machine in machines:
        state = states[machine.machine_id]
        active_span = state.stopped_at_h if state.stopped_at_h is not None else horizon_h
        uptime = max(0, active_span - state.downtime_h)
        reports.append(
            MachineReport(
                machine.machine_id,
                state.operational,
                state.stopped_at_h,
                uptime,
                state.downtime_h,
                state.spares_consumed,
                machine.locally_repairable,
                machine.locally_reproducible,
            )
        )

    return reports, dict(spares)


def self_test() -> None:
    policy = MaintenancePolicy(100, 10, "seal", 1)
    machine = Machine("m1", policy, True, False)

    reports, _ = simulate(250, (machine,), {"seal": 2})
    report = reports[0]
    assert report.operational_end
    assert report.downtime_h == 20
    assert report.uptime_h == 230
    assert report.spares_consumed == 2

    reports, _ = simulate(250, (machine,), {"seal": 1})
    report = reports[0]
    assert not report.operational_end
    assert report.stopped_at_h == 200
    assert report.uptime_h == 190

    reports, _ = simulate(
        250,
        (machine,),
        {"seal": 1},
        (SpareProduction(150, "seal", 1),),
    )
    report = reports[0]
    assert report.operational_end
    assert not report.locally_reproducible

    failure = Failure(50, ("m1",), "bearing", 1, 5)
    reports, _ = simulate(100, (machine,), {"seal": 2}, (), (failure,))
    assert not reports[0].operational_end
    assert reports[0].stopped_at_h == 50

    reports, _ = simulate(100, (machine,), {"seal": 2, "bearing": 1}, (), (failure,))
    report = reports[0]
    assert report.operational_end
    assert report.downtime_h == 15
    assert not report.locally_reproducible

    m2 = Machine("m2", None, True, False)
    m3 = Machine("m3", None, True, False)
    correlated = Failure(20, ("m2", "m3"), "controller", 1, 4)
    reports, _ = simulate(100, (m2, m3), {"controller": 1}, (), (correlated,))
    assert sum(report.operational_end for report in reports) == 1
    assert sum(report.stopped_at_h == 20 for report in reports) == 1

    nonrepairable = Machine("nr", None, False, False)
    nonrepairable_failure = Failure(10, ("nr",), "part", 1, 1)
    reports, _ = simulate(50, (nonrepairable,), {"part": 5}, (), (nonrepairable_failure,))
    assert not reports[0].operational_end

    try:
        simulate(0, (machine,), {"seal": 1})
    except ValueError:
        pass
    else:
        raise AssertionError("nonpositive campaign horizon must fail")


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
