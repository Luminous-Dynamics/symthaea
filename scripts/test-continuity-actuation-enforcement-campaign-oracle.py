#!/usr/bin/env python3
"""Execute the frozen actuation-enforcement campaign oracle parity contract.

Standard-library only. This is an independent structural regression harness, not a
verifier, qualification result, or source of execution authority.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[1]
ORACLE = ROOT / "scripts" / "continuity-actuation-enforcement-campaign-oracle.py"
FIXTURE = ROOT / "tests" / "fixtures" / "continuity" / "actuation_enforcement_campaign_v1.json"
EXPECTED = ROOT / "tests" / "fixtures" / "continuity" / "actuation_enforcement_campaign_v1.expected.json"


def run(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ORACLE), *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def load_json(path: pathlib.Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def main() -> int:
    for path in (ORACLE, FIXTURE, EXPECTED):
        if not path.is_file():
            print(f"FAIL: required parity artifact missing: {path.relative_to(ROOT)}", file=sys.stderr)
            return 2

    result = run([str(FIXTURE.relative_to(ROOT))])
    if result.returncode != 0:
        print("FAIL: positive campaign fixture was rejected", file=sys.stderr)
        if result.stdout:
            print(result.stdout, file=sys.stderr, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        return 3

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        print(
            f"FAIL: expected exactly one JSON summary line from oracle, got {len(lines)}",
            file=sys.stderr,
        )
        return 4

    try:
        observed = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        print(f"FAIL: oracle summary is not valid JSON: {exc}", file=sys.stderr)
        return 5

    expected = load_json(EXPECTED)
    if observed != expected:
        print("FAIL: campaign oracle output drifted from frozen expected fixture", file=sys.stderr)
        print("expected=" + json.dumps(expected, sort_keys=True), file=sys.stderr)
        print("observed=" + json.dumps(observed, sort_keys=True), file=sys.stderr)
        return 6

    self_test = run(["--self-test"])
    if self_test.returncode != 0:
        print("FAIL: campaign oracle adversarial self-tests failed", file=sys.stderr)
        if self_test.stdout:
            print(self_test.stdout, file=sys.stderr, end="")
        if self_test.stderr:
            print(self_test.stderr, file=sys.stderr, end="")
        return 7

    if self_test.stdout.strip() != "PASS: campaign oracle self-tests":
        print("FAIL: unexpected campaign oracle self-test output", file=sys.stderr)
        print(self_test.stdout, file=sys.stderr, end="")
        return 8

    print(
        "PASS: actuation-enforcement campaign oracle matches frozen fixture and adversarial self-tests"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
