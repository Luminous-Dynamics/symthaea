#!/usr/bin/env python3
"""Execute the frozen actuation-enforcement campaign oracle parity contract.

Standard-library only. This is an independent structural regression harness, not a
verifier, qualification result, or source of execution authority.
"""

from __future__ import annotations

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
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


def parse_single_summary(result: subprocess.CompletedProcess[str], context: str) -> dict[str, Any]:
    if result.returncode != 0:
        raise AssertionError(
            f"{context}: oracle rejected qualifying input: {result.stderr.strip()}"
        )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise AssertionError(f"{context}: expected one JSON summary line, got {len(lines)}")
    try:
        value = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise AssertionError(f"{context}: oracle summary is invalid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise AssertionError(f"{context}: oracle summary must be an object")
    return value


def write_case(directory: pathlib.Path, name: str, manifest: dict[str, Any]) -> pathlib.Path:
    path = directory / f"{name}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, sort_keys=True, separators=(",", ":"))
        fh.write("\n")
    return path


def require_denied(directory: pathlib.Path, name: str, manifest: dict[str, Any]) -> None:
    path = write_case(directory, name, manifest)
    result = run([str(path)])
    if result.returncode == 0:
        raise AssertionError(f"{name}: malformed/ambiguous campaign was accepted")
    if "DENY:" not in result.stderr:
        raise AssertionError(f"{name}: rejection did not use fail-closed DENY path")


def require_identity_drift(
    directory: pathlib.Path,
    name: str,
    manifest: dict[str, Any],
    baseline_hash: str,
) -> None:
    path = write_case(directory, name, manifest)
    observed = parse_single_summary(run([str(path)]), name)
    if observed.get("canonical_preimage_sha256") == baseline_hash:
        raise AssertionError(f"{name}: qualifying lineage drift did not change campaign preimage")


def main() -> int:
    try:
        for path in (ORACLE, FIXTURE, EXPECTED):
            if not path.is_file():
                raise AssertionError(
                    f"required parity artifact missing: {path.relative_to(ROOT)}"
                )

        fixture = load_json(FIXTURE)
        expected = load_json(EXPECTED)
        if not isinstance(fixture, dict) or not isinstance(expected, dict):
            raise AssertionError("fixture and expected output must be JSON objects")

        observed = parse_single_summary(run([str(FIXTURE.relative_to(ROOT))]), "positive fixture")
        if observed != expected:
            raise AssertionError(
                "positive fixture drifted: expected="
                + json.dumps(expected, sort_keys=True)
                + " observed="
                + json.dumps(observed, sort_keys=True)
            )

        baseline_hash = expected.get("canonical_preimage_sha256")
        if not isinstance(baseline_hash, str) or len(baseline_hash) != 64:
            raise AssertionError("expected fixture lacks canonical_preimage_sha256")

        with tempfile.TemporaryDirectory(prefix="symthaea-campaign-oracle-") as temp:
            directory = pathlib.Path(temp)

            reordered = copy.deepcopy(fixture)
            reordered["records"] = list(reversed(reordered["records"]))
            reordered_summary = parse_single_summary(
                run([str(write_case(directory, "reordered", reordered))]),
                "reordered records",
            )
            if reordered_summary != expected:
                raise AssertionError("record order changed canonical campaign semantics")

            out_of_window = copy.deepcopy(fixture)
            out_of_window["records"][0]["observed_at_unix_ms"] = fixture["started_at_unix_ms"] - 1
            require_denied(directory, "out_of_window", out_of_window)

            duplicate_obligation = copy.deepcopy(fixture)
            duplicate_obligation["records"][1]["obligation"] = duplicate_obligation["records"][0]["obligation"]
            require_denied(directory, "duplicate_obligation", duplicate_obligation)

            duplicate_record_id = copy.deepcopy(fixture)
            duplicate_record_id["records"][8]["record_id"] = duplicate_record_id["records"][7]["record_id"]
            require_denied(directory, "duplicate_record_id", duplicate_record_id)

            zero_environment = copy.deepcopy(fixture)
            zero_environment["environment_manifest_digest"] = "00" * 32
            require_denied(directory, "zero_environment", zero_environment)

            shadow_top_level = copy.deepcopy(fixture)
            shadow_top_level["shadow_policy"] = "permit"
            require_denied(directory, "shadow_top_level", shadow_top_level)

            shadow_record = copy.deepcopy(fixture)
            shadow_record["records"][0]["shadow_result"] = "satisfied"
            require_denied(directory, "shadow_record", shadow_record)

            zero_backend_generation = copy.deepcopy(fixture)
            zero_backend_generation["backend_generation"] = 0
            require_denied(directory, "zero_backend_generation", zero_backend_generation)

            reversed_interval = copy.deepcopy(fixture)
            reversed_interval["started_at_unix_ms"] = fixture["ended_at_unix_ms"] + 1
            require_denied(directory, "reversed_interval", reversed_interval)

            toolchain_drift = copy.deepcopy(fixture)
            toolchain_drift["toolchain_realization_digest"] = "99" * 32
            require_identity_drift(directory, "toolchain_drift", toolchain_drift, baseline_hash)

            backend_generation_drift = copy.deepcopy(fixture)
            backend_generation_drift["backend_generation"] += 1
            require_identity_drift(
                directory,
                "backend_generation_drift",
                backend_generation_drift,
                baseline_hash,
            )

            enforcement_profile_substitution = copy.deepcopy(fixture)
            enforcement_profile_substitution["enforcement_profile_id"] = "99" * 32
            require_identity_drift(
                directory,
                "enforcement_profile_substitution",
                enforcement_profile_substitution,
                baseline_hash,
            )

            boundary_implementation_drift = copy.deepcopy(fixture)
            boundary_implementation_drift["boundary_implementation_digest"] = "98" * 32
            require_identity_drift(
                directory,
                "boundary_implementation_drift",
                boundary_implementation_drift,
                baseline_hash,
            )

        self_test = run(["--self-test"])
        if self_test.returncode != 0:
            raise AssertionError(
                "oracle internal adversarial self-tests failed: " + self_test.stderr.strip()
            )
        if self_test.stdout.strip() != "PASS: campaign oracle self-tests":
            raise AssertionError("unexpected oracle internal self-test output")

    except (OSError, json.JSONDecodeError, AssertionError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    print(
        "PASS: campaign oracle matches frozen fixture and external/internal adversarial tests"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
