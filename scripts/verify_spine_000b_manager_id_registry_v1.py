#!/usr/bin/env python3
"""Verify SPINE-000B-I1R1 stable runtime manager IDs against live Phase-B source."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

REGISTRY = Path("docs/research/SPINE_000B_MANAGER_ID_REGISTRY_V1.json")
DYNAMICS = Path("src/cognitive_loop/cycle_phase_dynamics/mod.rs")
ALLOWED_STATUS = {"ACTIVE", "RETIRED"}
RUN_RE = re.compile(
    r'run_subsystem!\(\s*[^,]+,\s*"([^"]+)"\s*,\s*snapshot\s*\)',
    re.MULTILINE,
)


def fail(msg: str) -> None:
    raise ValueError(msg)


def source_runtime_names(source: str) -> list[str]:
    names = RUN_RE.findall(source)
    if not names:
        fail("no live run_subsystem attribution labels found")
    if len(names) != len(set(names)):
        fail(f"duplicate live runtime attribution label: {names}")
    return names


def validate_shape(reg: dict) -> None:
    required = {
        "schema", "authority", "causal_load_claimed", "source_path",
        "identity_source", "reserved_ids", "observer_capacity", "append_only", "entries",
    }
    if set(reg) != required:
        fail(f"registry fields drifted missing={sorted(required-set(reg))} extra={sorted(set(reg)-required)}")
    if reg["schema"] != "symthaea.spine.000b.manager-id-registry.v1":
        fail("schema drift")
    if reg["authority"] != "measurement-only" or reg["causal_load_claimed"] is not False:
        fail("authority boundary drift")
    if reg["source_path"] != "src/cognitive_loop/cycle_phase_dynamics/mod.rs":
        fail("source path drift")
    if reg["reserved_ids"] != [0]:
        fail("ID 0 must be the sole v1 reserved ID")
    if reg["append_only"] is not True:
        fail("registry must be append-only")
    if not isinstance(reg["observer_capacity"], int) or reg["observer_capacity"] <= 0:
        fail("invalid observer capacity")
    entries = reg["entries"]
    if not isinstance(entries, list) or not entries:
        fail("entries missing")
    ids = [e.get("id") for e in entries]
    names = [e.get("runtime_name") for e in entries]
    if any(not isinstance(i, int) or i <= 0 or i > 0xFFFF for i in ids):
        fail("manager IDs must be positive u16")
    if len(ids) != len(set(ids)):
        fail("duplicate manager ID")
    if len(names) != len(set(names)):
        fail("duplicate runtime name")
    if ids != list(range(1, len(entries) + 1)):
        fail("initial v1 manager IDs must be contiguous 1..N in frozen order")
    for e in entries:
        if set(e) != {"id", "runtime_name", "status"}:
            fail("entry shape drift")
        if not isinstance(e["runtime_name"], str) or not e["runtime_name"]:
            fail("runtime name must be non-empty string")
        if e["status"] not in ALLOWED_STATUS:
            fail(f"unknown manager status {e['status']}")
    if len(entries) > reg["observer_capacity"]:
        fail("registry population exceeds manager observer capacity")


def validate_current_source(reg: dict, source_names: list[str]) -> None:
    active = [e for e in reg["entries"] if e["status"] == "ACTIVE"]
    active_names = [e["runtime_name"] for e in active]
    if active_names != source_names:
        missing = [n for n in source_names if n not in active_names]
        extra = [n for n in active_names if n not in source_names]
        fail(
            "v1 source/registry execution order mismatch "
            f"missing={missing} extra={extra} source={source_names} registry={active_names}"
        )


def validate_append(old: dict, new: dict) -> None:
    """Validate a hypothetical future append-only registry against this v1 prefix."""
    validate_shape(new)
    old_entries = old["entries"]
    new_entries = new["entries"]
    if len(new_entries) < len(old_entries):
        fail("append-only registry cannot shrink")
    for before, after in zip(old_entries, new_entries):
        if before["id"] != after["id"] or before["runtime_name"] != after["runtime_name"]:
            fail("historical manager ID/name mapping changed")
        # ACTIVE may transition to RETIRED, but RETIRED may never return to ACTIVE.
        if before["status"] == "RETIRED" and after["status"] != "RETIRED":
            fail("retired manager ID was reused/reactivated")
    expected_next = len(old_entries) + 1
    for e in new_entries[len(old_entries):]:
        if e["id"] != expected_next:
            fail("new manager IDs must append monotonically")
        expected_next += 1


def must_reject(fn, label: str) -> None:
    try:
        fn()
    except ValueError:
        return
    fail(f"negative control did not reject: {label}")


def controls(reg: dict, source_names: list[str]) -> None:
    x = copy.deepcopy(reg); x["entries"][1]["id"] = x["entries"][0]["id"]
    must_reject(lambda: validate_shape(x), "duplicate id")
    x = copy.deepcopy(reg); x["entries"][1]["runtime_name"] = x["entries"][0]["runtime_name"]
    must_reject(lambda: validate_shape(x), "duplicate name")
    x = copy.deepcopy(reg); x["entries"][0]["id"] = 0
    must_reject(lambda: validate_shape(x), "reserved zero")
    x = copy.deepcopy(reg); x["observer_capacity"] = len(x["entries"]) - 1
    must_reject(lambda: validate_shape(x), "capacity underflow")
    x = copy.deepcopy(reg); x["entries"].pop()
    must_reject(lambda: validate_current_source(x, source_names), "missing live manager")
    x = copy.deepcopy(reg); x["entries"][0], x["entries"][1] = x["entries"][1], x["entries"][0]
    must_reject(lambda: validate_shape(x), "renumber/reorder initial IDs")

    appended = copy.deepcopy(reg)
    appended["entries"].append({"id": len(reg["entries"]) + 1, "runtime_name": "future_manager", "status": "ACTIVE"})
    validate_append(reg, appended)

    renumbered = copy.deepcopy(appended)
    renumbered["entries"][0]["id"] = 2
    must_reject(lambda: validate_append(reg, renumbered), "historical renumber")

    retired = copy.deepcopy(reg)
    retired["entries"][0]["status"] = "RETIRED"
    validate_append(reg, retired)
    reactivated = copy.deepcopy(retired)
    reactivated["entries"][0]["status"] = "ACTIVE"
    must_reject(lambda: validate_append(retired, reactivated), "retired ID reuse")


def main() -> int:
    reg = json.loads(REGISTRY.read_text(encoding="utf-8"))
    source = DYNAMICS.read_text(encoding="utf-8")
    names = source_runtime_names(source)
    validate_shape(reg)
    validate_current_source(reg, names)
    controls(reg, names)
    print("SPINE-000B-I1R1 manager ID registry verifier: PASS")
    print(f"runtime_manager_count={len(names)}")
    print(f"observer_capacity={reg['observer_capacity']}")
    print(f"execution_order={','.join(names)}")
    print("authority=measurement-only")
    print("runtime_execution_claimed=false")
    print("causal_load_claimed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
