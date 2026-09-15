#!/usr/bin/env python3
"""Pre-implementation verifier for SPINE-000B-I1 observer architecture v1.

This establishes only that the preregistered non-interference architecture is
consistent with the current source ordering and manager census. It does not
claim runtime observer non-interference; no runtime observer exists in this
tranche.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

CONTRACT = Path("docs/research/SPINE_000B_OBSERVER_NONINTERFERENCE_V1.md")
DYNAMICS = Path("src/cognitive_loop/cycle_phase_dynamics/mod.rs")
GENESIS_TEST = Path("tests/genesis_determinism.rs")
CENSUS = Path("scripts/spine_census.py")
CAPACITY = 64

RUN_SUBSYSTEM_RE = re.compile(r'run_subsystem!\([^,]+,\s*"([^"]+)"\s*,\s*snapshot\)')


def fail(message: str) -> None:
    raise SystemExit(f"SPINE-000B-I1 ARCHITECTURE FAIL: {message}")


def main() -> int:
    for path in (CONTRACT, DYNAMICS, GENESIS_TEST, CENSUS):
        if not path.is_file():
            fail(f"missing subject file: {path}")

    contract = CONTRACT.read_text(encoding="utf-8")
    dynamics = DYNAMICS.read_text(encoding="utf-8")
    genesis = GENESIS_TEST.read_text(encoding="utf-8")

    required_contract = (
        "Two-stage observer architecture",
        "Stage A — bounded hot-path capture",
        "Stage B — deferred derivation",
        "v1 capacity is **64 manager events per cycle**",
        "existing IDs never change meaning",
        "IDs are never reused",
        "OFF-A",
        "OFF-B",
        "A statistically nonsignificant difference is not evidence of equivalence",
        "I1-STATIC",
        "I1-DYNAMIC",
    )
    for phrase in required_contract:
        if phrase not in contract:
            fail(f"contract missing required phrase: {phrase}")

    macro_pos = dynamics.find("macro_rules! run_subsystem")
    manager_call_pos = dynamics.find("run_subsystem!(self.drive_manager")
    elapsed_pos = dynamics.find("let elapsed_us = cycle_start.elapsed().as_micros() as u64;")
    budget_pos = dynamics.find("let available_us = 20_000u64.saturating_sub(elapsed_us);")
    if min(macro_pos, manager_call_pos, elapsed_pos, budget_pos) < 0:
        fail("could not locate frozen manager/budget source anchors")
    if not (macro_pos < manager_call_pos < elapsed_pos < budget_pos):
        fail("manager seam no longer precedes the frozen wall-clock budget decision")

    # The non-interference campaign must not pretend independent full-loop runs
    # are already exact. Bind the current repository's documented baseline.
    if "const F32_TOLERANCE: f32 = 0.15;" not in genesis:
        fail("existing full-loop determinism baseline changed; I1 campaign design must be reviewed")
    if "excluding timing info" not in genesis:
        fail("genesis determinism test no longer documents timing exclusion")

    result = subprocess.run(
        [sys.executable, str(CENSUS)],
        check=True,
        capture_output=True,
        text=True,
    )
    census = json.loads(result.stdout)
    if census.get("authority") != "measurement-only":
        fail("SPINE census authority drifted")
    if census.get("causal_load_claimed") is not False:
        fail("SPINE census unexpectedly claims causal load")
    manager_count = census.get("manager_count")
    if not isinstance(manager_count, int) or manager_count <= 0:
        fail("invalid manager_count from static census")
    if manager_count > CAPACITY:
        fail(f"manager census {manager_count} exceeds Stage-A capacity {CAPACITY}")
    duplicates = census.get("duplicate_subsystem_names")
    if duplicates:
        fail("duplicate subsystem names in current census: " + ", ".join(duplicates))

    census_names = {
        record.get("subsystem_name")
        for record in census.get("managers", [])
        if isinstance(record, dict) and record.get("subsystem_name")
    }
    live_names = RUN_SUBSYSTEM_RE.findall(dynamics)
    if not live_names:
        fail("no live run_subsystem calls found")
    if len(live_names) > CAPACITY:
        fail(f"live manager call count {len(live_names)} exceeds Stage-A capacity {CAPACITY}")
    if len(set(live_names)) != len(live_names):
        duplicates = sorted({name for name in live_names if live_names.count(name) > 1})
        fail("live manager identity occurs at multiple run_subsystem callsites: " + ", ".join(duplicates))

    unknown = sorted(set(live_names) - census_names)
    if unknown:
        fail("live manager names missing from static census: " + ", ".join(unknown))

    # This branch freezes design only. If runtime observer symbols appear here,
    # they must move to a dedicated implementation lineage with stronger checks.
    forbidden_preimplementation = (
        "SpineObserverBuffer",
        "spine_observer_buffer",
        "ProposalInfluenceReceipt",
        "CycleIntegrationReceipt",
        "StateApplicationReceipt",
    )
    present = [symbol for symbol in forbidden_preimplementation if symbol in dynamics]
    if present:
        fail("runtime observer implementation appeared in preregistration subject: " + ", ".join(present))

    print("SPINE-000B-I1 observer architecture preflight: PASS")
    print(f"manager_census_count={manager_count}")
    print(f"live_run_subsystem_calls={len(live_names)}")
    print(f"stage_a_capacity={CAPACITY}")
    print("source_order=manager_seam_before_wall_clock_budget_gate")
    print("status=OBSERVER_ARCHITECTURE_FROZEN_IMPLEMENTATION_PENDING")
    print("authority=measurement-only")
    print("noninterference_claimed=false")
    print("causal_load_claimed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
