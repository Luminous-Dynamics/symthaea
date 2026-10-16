#!/usr/bin/env python3
"""Create a deterministic, exact-head CI record for run-bundle validator tests.

This record is intended to be the predicate of a separately signed GitHub
artifact attestation. It records test execution metadata and the exact hashes
of inputs to the test job. It does not attest that an economic simulation is
true, or that this test suite validates production authority.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPORT_SCHEMA = "civos.socioeconomic_interoperability.validator_execution_record.v1"
CLAIM_CEILING = "synthetic_deterministic_only"
SOURCE_PATHS = (
    "scripts/validate_socioeconomic_run_bundle.py",
    "scripts/test_validate_socioeconomic_run_bundle.py",
    "scripts/write_socioeconomic_run_bundle_ci_report.py",
    "scripts/test_write_socioeconomic_run_bundle_ci_report.py",
    "docs/research/finance/SYM-FIN-SYS-016_run_bundle_manifest_v1.schema.json",
    "docs/research/finance/SYM-FIN-SYS-016_validator_execution_record_v1.schema.json",
    "scripts/socioeconomic_run_receipt_requirements.txt",
    ".github/workflows/socioeconomic-run-bundle.yml",
)
TEST_PATHS = (
    "scripts/test_validate_socioeconomic_run_bundle.py",
    "scripts/test_write_socioeconomic_run_bundle_ci_report.py",
)
REQUIRED_ENV = (
    "GITHUB_SHA",
    "CIVOS_SOURCE_REVISION",
    "GITHUB_REPOSITORY",
    "GITHUB_WORKFLOW",
    "GITHUB_RUN_ID",
    "GITHUB_RUN_ATTEMPT",
    "GITHUB_EVENT_NAME",
    "GITHUB_JOB",
    "RUNNER_OS",
)


class ReportFailure(Exception):
    """Unable to produce a trustworthy CI execution record."""


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def count_test_functions(test_source: bytes) -> int:
    try:
        tree = ast.parse(test_source.decode("utf-8"))
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ReportFailure(f"test source is not valid UTF-8 Python: {exc}") from exc
    return sum(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        for node in ast.walk(tree)
    )


def build_report(
    workspace: str | Path,
    env: dict[str, str],
    test_log_relative_path: str,
) -> dict[str, Any]:
    root = Path(workspace).resolve(strict=True)
    missing = [name for name in REQUIRED_ENV if not env.get(name, "").strip()]
    if missing:
        raise ReportFailure(f"required workflow metadata missing: {', '.join(missing)}")

    source_revision = env["CIVOS_SOURCE_REVISION"]
    if re.fullmatch(r"[0-9a-f]{40}", source_revision) is None:
        raise ReportFailure("CIVOS_SOURCE_REVISION must be a full lowercase 40-character commit SHA")
    event_sha = env["GITHUB_SHA"]
    if re.fullmatch(r"[0-9a-f]{40}", event_sha) is None:
        raise ReportFailure("GITHUB_SHA must be a full lowercase 40-character commit SHA")

    try:
        checked_out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReportFailure(f"cannot establish checked-out Git revision: {exc}") from exc
    if checked_out != source_revision:
        raise ReportFailure(
            f"checked-out revision {checked_out!r} does not match CIVOS_SOURCE_REVISION {source_revision!r}"
        )

    files: list[dict[str, Any]] = []
    for relative in (*SOURCE_PATHS, test_log_relative_path):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ReportFailure(f"unsafe evidence path: {relative!r}")
        file_path = root / path
        if file_path.is_symlink() or not file_path.is_file():
            raise ReportFailure(f"required evidence file is missing or unsafe: {relative}")
        try:
            file_path.resolve(strict=True).relative_to(root)
        except (ValueError, OSError) as exc:
            raise ReportFailure(f"evidence path escapes workspace: {relative}") from exc
        data = file_path.read_bytes()
        if relative == test_log_relative_path and not data:
            raise ReportFailure("test log is empty")
        files.append(
            {
                "path": relative,
                "digest": sha256_bytes(data),
                "byte_length": len(data),
                "role": "test_log" if relative == test_log_relative_path else "test_input",
            }
        )

    test_count = sum(
        count_test_functions((root / relative).read_bytes())
        for relative in TEST_PATHS
    )
    if test_count < 1:
        raise ReportFailure("no test functions found in the test sources")

    try:
        version_result = subprocess.run(
            [sys.executable, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
        python_version = (version_result.stdout or version_result.stderr).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReportFailure(f"cannot establish Python version: {exc}") from exc

    return {
        "schema": REPORT_SCHEMA,
        "claim_ceiling": CLAIM_CEILING,
        "result": "passed",
        "scope": "run_bundle_validator_and_execution_record_unit_tests",
        "repository": env["GITHUB_REPOSITORY"],
        "source_revision": source_revision,
        "event_sha": event_sha,
        "workflow": env["GITHUB_WORKFLOW"],
        "job": env["GITHUB_JOB"],
        "event": env["GITHUB_EVENT_NAME"],
        "run_id": env["GITHUB_RUN_ID"],
        "run_attempt": env["GITHUB_RUN_ATTEMPT"],
        "runner_os": env["RUNNER_OS"],
        "python_version": python_version,
        "test_commands": [
            "python3 scripts/test_validate_socioeconomic_run_bundle.py",
            "python3 scripts/test_write_socioeconomic_run_bundle_ci_report.py",
        ],
        "authored_test_function_count": test_count,
        "artifacts": sorted(files, key=lambda item: item["path"]),
        "authority_boundary": {
            "proves_simulation_truth": False,
            "can_authorize_real_actions": False,
            "can_reserve_real_resources": False,
            "can_settle_real_assets": False,
            "production_ledger_import_allowed": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--test-log", default="evidence/test-output.txt")
    parser.add_argument("--output", default="evidence/qualification-report.json")
    args = parser.parse_args()
    root = Path(args.root).resolve(strict=True)
    try:
        report = build_report(root, dict(os.environ), args.test_log)
        output = Path(args.output)
        if output.is_absolute() or ".." in output.parts:
            raise ReportFailure("output must be a relative path within the workspace")
        output_path = root / output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists() or output_path.is_symlink():
            raise ReportFailure(f"refusing to overwrite existing report: {args.output}")
        output_path.write_text(
            json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    except (ReportFailure, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "PASS: deterministic CI execution record written; "
        f"{report['authored_test_function_count']} test functions inventoried; "
        "signed attestation must be verified separately."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
