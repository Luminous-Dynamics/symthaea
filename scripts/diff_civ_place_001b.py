#!/usr/bin/env python3
"""Differential gate for the CIV-PLACE-001B Python/Rust leaf kernel."""
from __future__ import annotations
import importlib.util, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_ORACLE = ROOT / "scripts" / "validate_civ_place_001b.py"
FIXTURE = ROOT / "docs" / "engineering" / "fixtures" / "civ-place-001b.json"

def load_reference():
    spec = importlib.util.spec_from_file_location("civ_place_reference", PYTHON_ORACLE)
    if spec is None or spec.loader is None:
        raise SystemExit("unable to load Python reference oracle")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def run_rust():
    proc = subprocess.run(
        ["cargo","run","-p","symthaea-civ-place","--bin","civ-place-001b-oracle","--quiet"],
        cwd=ROOT, text=True, capture_output=True, check=False)
    if proc.returncode:
        print(proc.stdout, end=""); print(proc.stderr, end="", file=sys.stderr)
        raise SystemExit("Rust leaf oracle failed")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines: raise SystemExit("Rust leaf oracle emitted no JSON")
    return json.loads(lines[-1])

def main() -> int:
    reference = load_reference()
    reference.main()

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in fixture["nodes"]}
    projections = {p["id"]: p for p in fixture["projections"]}
    services = {s["id"]: s for s in fixture["services"]}

    python_services = {
        sid: reference.evaluate(service, nodes, fixture["dependencies"], projections)
        for sid, service in sorted(services.items())
    }
    python_cases = {
        case["id"]: reference.derive(case, fixture, python_services)
        for case in fixture["hostile_cases"]
    }

    rust = run_rust()
    expected = {
        "profile": fixture["profile"],
        "schema_version": fixture["schema_version"],
        "services": python_services,
        "hostile_cases": python_cases,
    }
    actual = {
        "profile": rust["profile"],
        "schema_version": rust["schema_version"],
        "services": rust["services"],
        "hostile_cases": rust["hostile_cases"],
    }
    if actual != expected:
        print("CIV-PLACE differential FAIL: Python/Rust semantic output differs", file=sys.stderr)
        print(json.dumps({"expected":expected,"actual":actual}, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    print(json.dumps({
        "result":"PASS",
        "profile":rust["profile"],
        "fixture_sha256":rust["fixture_sha256"],
        "services":len(rust["services"]),
        "hostile_cases":len(rust["hostile_cases"]),
        "python_reference":"PASS",
        "rust_leaf":"PASS",
        "differential":"PASS",
        "authority_claim":"none",
    }, sort_keys=True, separators=(",",":")))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
