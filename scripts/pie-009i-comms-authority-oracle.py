#!/usr/bin/env python3
"""PIE-009I independent communications-latency / authority oracle."""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

class Authority(Enum):
    LOCAL_SAFETY = "local_safety"
    LOCAL_OPERATOR = "local_operator"
    EARTH_OPERATOR = "earth_operator"
    ADVISORY = "advisory"

@dataclass(frozen=True)
class PlantState:
    step: int
    version: int
    essential_floor: float
    nonessential_enabled: bool
    failed_units: Tuple[str, ...]

@dataclass(frozen=True)
class Command:
    command_id: str
    issue_step: int
    nominal_arrival_step: int
    expiry_step: int
    expected_state_version: Optional[int]
    authority: Authority
    action: str
    target: Optional[str] = None

@dataclass(frozen=True)
class CommandReceipt:
    command_id: str
    status: str
    applied_step: Optional[int]
    reason: str

@dataclass(frozen=True)
class StepInput:
    link_open: bool = True
    newly_failed: Tuple[str, ...] = ()

@dataclass(frozen=True)
class StepOutput:
    state: PlantState
    receipts: Tuple[CommandReceipt, ...]
    local_action: str

PROTECTED_ACTIONS = {"disable_essential_protection", "force_below_essential_floor"}

def validate_command(c: Command) -> None:
    if not c.command_id:
        raise ValueError("command id required")
    if c.issue_step < 0 or c.nominal_arrival_step < c.issue_step or c.expiry_step < c.nominal_arrival_step:
        raise ValueError("invalid command timing")
    if c.expected_state_version is not None and c.expected_state_version < 0:
        raise ValueError("invalid expected version")
    if not c.action:
        raise ValueError("action required")

def run(initial: PlantState, steps: Tuple[StepInput, ...], commands: Tuple[Command, ...]) -> Tuple[StepOutput, ...]:
    for c in commands:
        validate_command(c)
    if len({c.command_id for c in commands}) != len(commands):
        raise ValueError("duplicate command id")
    if initial.step != 0 or initial.version < 0 or not (0 <= initial.essential_floor <= 1):
        raise ValueError("invalid initial state")

    pending = {c.command_id: c for c in commands}
    state = initial
    outputs = []

    for step, inp in enumerate(steps):
        if step != state.step:
            raise AssertionError("state step drift")

        failed = set(state.failed_units)
        changed = False
        for unit in inp.newly_failed:
            if not unit:
                raise ValueError("empty failed unit id")
            if unit not in failed:
                failed.add(unit)
                changed = True

        # Local safety/autonomy observes current local state immediately.
        # It cannot know future link state or future failures.
        local_action = "preserve_essential"
        nonessential = state.nonessential_enabled
        essential_floor = state.essential_floor
        if failed:
            nonessential = False
            essential_floor = max(essential_floor, 1.0)
            if nonessential != state.nonessential_enabled or essential_floor != state.essential_floor:
                changed = True

        receipts = []
        # Delivery is possible only at/after nominal arrival and while the link is open.
        # Undelivered commands may wait, but expire at their explicit deadline.
        for cid in sorted(list(pending)):
            c = pending[cid]
            if step > c.expiry_step:
                receipts.append(CommandReceipt(cid, "Rejected", None, "expired before delivery"))
                del pending[cid]
                continue
            if step < c.nominal_arrival_step or not inp.link_open:
                continue

            if c.expected_state_version is not None and c.expected_state_version != state.version + (1 if changed else 0):
                receipts.append(CommandReceipt(cid, "Rejected", None, "stale state-version precondition"))
                del pending[cid]
                continue

            if c.action in PROTECTED_ACTIONS and c.authority not in (Authority.LOCAL_SAFETY,):
                receipts.append(CommandReceipt(cid, "Rejected", None, "authority cannot override local essential protection"))
                del pending[cid]
                continue

            if c.authority == Authority.ADVISORY:
                receipts.append(CommandReceipt(cid, "AdvisoryOnly", step, "recorded but not authoritative"))
                del pending[cid]
                continue

            if c.action == "enable_nonessential":
                if failed:
                    receipts.append(CommandReceipt(cid, "Rejected", None, "local failed-state guard blocks nonessential enable"))
                else:
                    nonessential = True
                    changed = changed or (not state.nonessential_enabled)
                    receipts.append(CommandReceipt(cid, "Applied", step, "accepted"))
            elif c.action == "disable_nonessential":
                changed = changed or state.nonessential_enabled
                nonessential = False
                receipts.append(CommandReceipt(cid, "Applied", step, "accepted"))
            elif c.action == "repair_unit":
                if not c.target or c.target not in failed:
                    receipts.append(CommandReceipt(cid, "Rejected", None, "repair target is not currently failed"))
                else:
                    failed.remove(c.target)
                    changed = True
                    receipts.append(CommandReceipt(cid, "Applied", step, "repair applied in synthetic reference"))
            elif c.action in PROTECTED_ACTIONS:
                # Only LOCAL_SAFETY reaches here. The synthetic safety action is explicit.
                essential_floor = 0.0
                changed = True
                receipts.append(CommandReceipt(cid, "Applied", step, "local safety authority action"))
            else:
                receipts.append(CommandReceipt(cid, "Rejected", None, "unknown action"))
            del pending[cid]

        next_version = state.version + (1 if changed else 0)
        next_state = PlantState(
            step=step + 1,
            version=next_version,
            essential_floor=essential_floor,
            nonessential_enabled=nonessential,
            failed_units=tuple(sorted(failed)),
        )
        outputs.append(StepOutput(
            state=PlantState(
                step=step,
                version=next_version,
                essential_floor=essential_floor,
                nonessential_enabled=nonessential,
                failed_units=tuple(sorted(failed)),
            ),
            receipts=tuple(receipts),
            local_action=local_action,
        ))
        state = next_state

    # Commands that were never deliverable within the supplied horizon remain pending;
    # the caller must not treat them as executed.
    return tuple(outputs)

def self_test() -> None:
    init = PlantState(0, 0, 1.0, True, ())

    # 1. Remote command cannot execute before arrival.
    c = Command("c1", 0, 2, 4, None, Authority.EARTH_OPERATOR, "disable_nonessential")
    out = run(init, (StepInput(), StepInput(), StepInput()), (c,))
    assert not out[0].receipts and not out[1].receipts
    assert out[2].receipts[0].status == "Applied"

    # 2. Link outage delays delivery; expiry causes fail-closed rejection.
    c2 = Command("c2", 0, 1, 2, None, Authority.EARTH_OPERATOR, "disable_nonessential")
    out2 = run(init, (StepInput(), StepInput(False), StepInput(False), StepInput(True)), (c2,))
    assert out2[3].receipts[0].status == "Rejected"
    assert "expired" in out2[3].receipts[0].reason

    # 3. Intervening local failure changes state version; stale command is rejected.
    c3 = Command("c3", 0, 2, 4, 0, Authority.EARTH_OPERATOR, "enable_nonessential")
    out3 = run(init, (StepInput(), StepInput(newly_failed=("pump",)), StepInput()), (c3,))
    assert out3[2].receipts[0].status == "Rejected"
    assert "stale" in out3[2].receipts[0].reason

    # 4. Earth/advisory authority cannot disable local essential protection.
    bad = Command("bad", 0, 0, 1, None, Authority.EARTH_OPERATOR, "disable_essential_protection")
    adv = Command("adv", 0, 0, 1, None, Authority.ADVISORY, "disable_nonessential")
    out4 = run(init, (StepInput(),), (bad, adv))
    statuses = {r.command_id: r.status for r in out4[0].receipts}
    assert statuses["bad"] == "Rejected"
    assert statuses["adv"] == "AdvisoryOnly"

    # 5. Local safety behavior continues during communication outage.
    out5 = run(init, (StepInput(False, ("water",)), StepInput(False)), ())
    assert out5[0].local_action == "preserve_essential"
    assert out5[0].state.nonessential_enabled is False
    assert out5[0].state.essential_floor == 1.0

    # 6. Remote request to enable nonessential work is blocked while a local failure remains.
    c6 = Command("c6", 0, 0, 2, None, Authority.EARTH_OPERATOR, "enable_nonessential")
    out6 = run(init, (StepInput(True, ("water",)),), (c6,))
    assert out6[0].receipts[0].status == "Rejected"

    # 7. Local repair can restore state, and later remote command may apply if version matches.
    repair = Command("repair", 0, 0, 1, None, Authority.LOCAL_OPERATOR, "repair_unit", "water")
    enable = Command("enable", 0, 1, 2, 1, Authority.EARTH_OPERATOR, "enable_nonessential")
    out7 = run(init, (StepInput(True, ("water",)), StepInput()), (repair, enable))
    assert out7[0].receipts[0].status == "Applied"
    assert out7[1].receipts[0].status == "Applied"

    # 8. Future link/failure differences cannot affect earlier local behavior.
    prefix_a = run(init, (StepInput(False), StepInput(False), StepInput(True)), ())
    prefix_b = run(init, (StepInput(False), StepInput(False), StepInput(True, ("future",))), ())
    assert prefix_a[0].local_action == prefix_b[0].local_action
    assert prefix_a[1].local_action == prefix_b[1].local_action

    # 9. Invalid timing fails closed.
    try:
        run(init, (StepInput(),), (Command("x", 2, 1, 3, None, Authority.EARTH_OPERATOR, "disable_nonessential"),))
    except ValueError:
        pass
    else:
        raise AssertionError("invalid command timing must fail")

    # 10. Duplicate command ids fail closed.
    d = Command("dup", 0, 0, 1, None, Authority.EARTH_OPERATOR, "disable_nonessential")
    try:
        run(init, (StepInput(),), (d, d))
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate commands must fail")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        self_test()
        print("ok")
    else:
        p.error("--self-test required")
