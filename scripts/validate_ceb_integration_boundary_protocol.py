#!/usr/bin/env python3
"""Static validation for the CEB end-to-end integration boundary protocol.

This checks research-document structure only. It does not build, execute or
benchmark candidate code, tasks, tests, service stacks or graders.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

DOC = "docs/research/CEB_END_TO_END_INTEGRATION_BOUNDARY_PROTOCOL_V0_2026-11-07.md"
LEVELS = (
    "B0 — Static/source",
    "B1 — Component",
    "B2 — Workspace",
    "B3 — Interface",
    "B4 — System",
    "B5 — Operational",
    "B6 — Evolution",
)
SCENARIOS = tuple(f"E2E-S{i:02d}" for i in range(1, 13))
GATES = ("#10064", "#10369")


def validate_document(doc: str) -> list[str]:
    errors: list[str] = []
    for level in LEVELS:
        if level not in doc:
            errors.append(f"missing verification boundary {level}")
    scenario_rows = [
        line for line in doc.splitlines()
        if re.match(r"^\| E2E-S\d{2} \|", line)
    ]
    ids = [
        re.match(r"^\| (E2E-S\d{2}) \|", line).group(1)
        for line in scenario_rows
    ]
    if ids != list(SCENARIOS):
        errors.append(f"must define E2E-S01..E2E-S12 once and in order; got {ids}")
    for scenario, line in zip(ids, scenario_rows):
        if line.count("|") < 4:
            errors.append(f"{scenario}: incomplete scenario row")

    for required in (
        "SWE-Serve",
        "SWE-Cycle",
        "E2E-SWE",
        "69.4%",
        "45.9%",
        "from isolated tasks to the integrated FullCycle",
        "whole-repository generation",
        "paired boundary",
        "smallest sufficient",
        "false boundary promotion",
        "positive/negative verifier controls",
        "No new primary lane",
        "no new",
        "no receipt",
    ):
        if required.lower() not in doc.lower():
            errors.append(f"missing required integration-boundary concept: {required}")
    for gate in GATES:
        if gate not in doc:
            errors.append(f"missing assurance gate {gate}")
    if "14-lane / 56-primary-metric" not in doc:
        errors.append("must preserve the 14-lane / 56-primary-metric portfolio")
    if "never silently certified" not in doc and "cannot silently certify" not in doc:
        errors.append("must state that narrower evidence cannot certify a broader claim")
    if "No candidate is run" not in doc and "no candidate is run" not in doc.lower():
        errors.append("must retain the no-candidate-execution claim boundary")
    return errors


def self_test() -> list[str]:
    doc = "\n".join((
        *LEVELS,
        *[f"| {scenario} | synthetic | boundary | stop |" for scenario in SCENARIOS],
        "SWE-Serve SWE-Cycle E2E-SWE 69.4% 45.9% from isolated tasks to the integrated FullCycle whole-repository generation",
        "paired boundary smallest sufficient false boundary promotion positive/negative verifier controls",
        "No new primary lane no new no receipt #10064 #10369 14-lane / 56-primary-metric",
        "cannot silently certify a broader claim; No candidate is run",
    ))
    if validate_document(doc):
        return ["minimal valid integration protocol rejected"]
    mutations = (
        ("missing B6", doc.replace("B6 — Evolution", "B6 removed")),
        ("missing scenario", doc.replace("| E2E-S12 |", "| E2E-SXX |")),
        ("duplicate scenario", doc.replace("| E2E-S02 |", "| E2E-S01 |")),
        ("missing E2E comparator", doc.replace("E2E-SWE", "OTHER")),
        ("missing critical gate", doc.replace("#10064", "gate")),
        ("portfolio drift", doc.replace("14-lane / 56-primary-metric", "13-lane / 45-metric")),
        ("missing claim boundary", doc.replace("No candidate is run", "candidate executed")),
    )
    for label, mutated in mutations:
        if not validate_document(mutated):
            return [f"negative self-test accepted mutation: {label}"]
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        failures = self_test()
        if failures:
            for failure in failures:
                print(f"SELF_TEST_FAIL: {failure}", file=sys.stderr)
            return 1
        print("CEB_INTEGRATION_BOUNDARY_SELF_TEST_PASS: 7 negative controls")
    path = args.repo_root / DOC
    try:
        doc = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot read {DOC}: {exc}", file=sys.stderr)
        return 1
    errors = validate_document(doc)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("CEB_INTEGRATION_BOUNDARY_PASS: levels=7 scenarios=12 primary_metrics=0")
    print(
        "CEB_INTEGRATION_BOUNDARY_CLAIM_BOUNDARY: documentation validation only; "
        "no candidate code, builds, tests, services, graders or benchmark tasks executed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
