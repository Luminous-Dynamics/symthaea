#!/usr/bin/env python3
"""Independent PIE-009S metrology sustainment oracle."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Channel:
    channel_id: str
    max_age_steps: int
    critical: bool = True


@dataclass(frozen=True)
class CalibrationAction:
    channel_id: str


@dataclass
class Campaign:
    channels: dict[str, Channel]
    last_calibration_step: dict[str, int]
    bench_slots_per_step: int
    standard_uses: int


def _qualified(channel: Channel, last_step: int, step: int) -> bool:
    return step - last_step <= channel.max_age_steps


def _validate(campaign: Campaign) -> None:
    if not campaign.channels:
        raise ValueError("empty channel registry")
    if campaign.bench_slots_per_step < 0 or campaign.standard_uses < 0:
        raise ValueError("negative campaign resource")
    if set(campaign.channels) != set(campaign.last_calibration_step):
        raise ValueError("last-calibration registry mismatch")
    for channel_id, channel in campaign.channels.items():
        if channel.channel_id != channel_id or not channel_id:
            raise ValueError("invalid channel identifier")
        if channel.max_age_steps < 0:
            raise ValueError("negative qualification age")
        if campaign.last_calibration_step[channel_id] < 0:
            raise ValueError("negative calibration step")


def simulate(campaign: Campaign, schedule: dict[int, list[CalibrationAction]], steps: int) -> list[dict]:
    _validate(campaign)
    if steps < 0:
        raise ValueError("negative campaign duration")

    last_calibration = dict(campaign.last_calibration_step)
    standard_uses = campaign.standard_uses
    pending: list[str] = []
    records: list[dict] = []

    for step in range(steps):
        # Calibrations completed in the previous campaign step become effective only
        # at this new boundary.
        for channel_id in pending:
            last_calibration[channel_id] = step
        pending = []

        opening = {
            channel_id: _qualified(channel, last_calibration[channel_id], step)
            for channel_id, channel in campaign.channels.items()
        }
        critical_opening = {
            channel_id: opening[channel_id]
            for channel_id, channel in campaign.channels.items()
            if channel.critical
        }

        actions = schedule.get(step, [])
        if len(actions) > campaign.bench_slots_per_step:
            raise ValueError("calibration bench capacity exceeded")

        seen: set[str] = set()
        for action in actions:
            if action.channel_id not in campaign.channels:
                raise ValueError("unknown calibration target")
            if action.channel_id in seen:
                raise ValueError("duplicate calibration target")
            seen.add(action.channel_id)

        if len(actions) > standard_uses:
            raise ValueError("reference/standard uses exhausted")
        standard_uses -= len(actions)
        pending = [action.channel_id for action in actions]

        records.append(
            {
                "step": step,
                "opening_qualified": opening,
                "all_critical_qualified": all(critical_opening.values()),
                "calibrations_started": tuple(pending),
                "standard_uses_remaining": standard_uses,
            }
        )

    return records


def run_self_test() -> str:
    channels = {
        "oxygen_pressure": Channel("oxygen_pressure", 2, True),
        "bus_voltage": Channel("bus_voltage", 2, True),
    }

    campaign = Campaign(
        channels=channels,
        last_calibration_step={"oxygen_pressure": 0, "bus_voltage": 0},
        bench_slots_per_step=1,
        standard_uses=4,
    )
    staggered = {
        1: [CalibrationAction("oxygen_pressure")],
        2: [CalibrationAction("bus_voltage")],
        4: [CalibrationAction("oxygen_pressure")],
        5: [CalibrationAction("bus_voltage")],
    }
    records = simulate(campaign, staggered, 7)
    assert all(record["all_critical_qualified"] for record in records)

    delayed = {2: [CalibrationAction("oxygen_pressure")]}
    records = simulate(
        Campaign(
            channels=channels,
            last_calibration_step={"oxygen_pressure": 0, "bus_voltage": 0},
            bench_slots_per_step=1,
            standard_uses=2,
        ),
        delayed,
        4,
    )
    assert records[3]["opening_qualified"]["bus_voltage"] is False
    assert records[3]["all_critical_qualified"] is False

    # A calibration started in the already-lapsed opening state at step 2 does not
    # retroactively qualify that opening state.
    one_channel = {"x": Channel("x", 1, True)}
    records = simulate(
        Campaign(one_channel, {"x": 0}, 1, 1),
        {2: [CalibrationAction("x")]},
        3,
    )
    assert records[2]["opening_qualified"]["x"] is False

    # Standard inventory is single-spend.
    try:
        simulate(
            Campaign(
                channels,
                {"oxygen_pressure": 0, "bus_voltage": 0},
                2,
                1,
            ),
            {1: [CalibrationAction("oxygen_pressure"), CalibrationAction("bus_voltage")]},
            2,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("standard inventory must fail closed")

    # Bench capacity is finite.
    try:
        simulate(
            Campaign(
                channels,
                {"oxygen_pressure": 0, "bus_voltage": 0},
                1,
                2,
            ),
            {1: [CalibrationAction("oxygen_pressure"), CalibrationAction("bus_voltage")]},
            2,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("bench over-capacity must fail closed")

    malformed_schedules = [
        {0: [CalibrationAction("oxygen_pressure"), CalibrationAction("oxygen_pressure")]},
        {0: [CalibrationAction("missing")]},
    ]
    for malformed in malformed_schedules:
        try:
            simulate(
                Campaign(
                    channels,
                    {"oxygen_pressure": 0, "bus_voltage": 0},
                    2,
                    2,
                ),
                malformed,
                1,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("malformed schedule must fail closed")

    return "ok"


if __name__ == "__main__":
    print(run_self_test())
