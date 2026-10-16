#!/usr/bin/env python3
"""Create an exact-head CI execution report and its separate attestation predicate.

The report is the attestation subject. A distinct bounded predicate names the
report digest and summarizes the tested source revision and observed test counts.
Neither artifact attests that an economic simulation is true or production
authority is valid.
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
PREDICATE_SCHEMA = "civos.socioeconomic_interoperability.validator_attestation_predicate.v1"
CLAIM_CEILING = "synthetic_deterministic_only"
SOURCE_PATHS = (
    "scripts/validate_socioeconomic_run_bundle.py",
    "scripts/test_validate_socioeconomic_run_bundle.py",
    "scripts/write_socioeconomic_run_bundle_ci_report.py",
    "scripts/test_write_socioeconomic_run_bundle_ci_report.py",
    "docs/research/finance/SYM-FIN-SYS-016_run_bundle_manifest_v1.schema.json",
    "docs/research/finance/SYM-FIN-SYS-016_validator_execution_record_v1.schema.json",
    "docs/research/finance/SYM-FIN-SYS-016_validator_attestation_predicate_v1.schema.json",
    "scripts/socioeconomic_run_receipt_requirements.txt",
    ".github/workflows/socioeconomic-run-bundle.yml",
)
TEST_PATHS = (
    "scripts/test_validate_socioeconomic_run_bundle.py",
    "scripts/test_write_socioeconomic_run_bundle_ci_report.py",
)
EXPECTED_TEST_COUNTS = (14, 11)
REQUIRED_ENV = (
    "GITHUB_SHA",
    "CIVOS_SOURCE_REVISION",
    "GITHUB_REPOSITORY",
    "GITHUB_WORKFLOW",
    "GITHUB_WORKFLOW_REF",
    "GITHUB_WORKFLOW_SHA",
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


def safe_workspace_file(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ReportFailure(f"unsafe evidence path: {relative!r}")
    cursor = root
    for part in path.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ReportFailure(f"symlink components are forbidden in evidence paths: {relative}")
    try:
        resolved = cursor.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise ReportFailure(f"evidence path is missing or escapes workspace: {relative}") from exc
    if not resolved.is_file():
        raise ReportFailure(f"evidence path is not a regular file: {relative}")
    return resolved


def parse_unittest_summaries(log_text: str) -> list[int]:
    matches = re.findall(r"Ran ([0-9]+) tests? in .*\n\nOK", log_text)
    return [int(match) for match in matches]


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
    workflow_sha = env["GITHUB_WORKFLOW_SHA"]
    if re.fullmatch(r"[0-9a-f]{40}", workflow_sha) is None:
        raise ReportFailure("GITHUB_WORKFLOW_SHA must be a full lowercase 40-character commit SHA")

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
        file_path = safe_workspace_file(root, relative)
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

    per_suite_test_counts = [
        count_test_functions(safe_workspace_file(root, relative).read_bytes())
        for relative in TEST_PATHS
    ]
    if tuple(per_suite_test_counts) != EXPECTED_TEST_COUNTS:
        raise ReportFailure(
            "versioned test inventory changed: "
            f"expected {list(EXPECTED_TEST_COUNTS)}, got {per_suite_test_counts}"
        )
    test_count = sum(per_suite_test_counts)
    if test_count < 1:
        raise ReportFailure("no test functions found in the test sources")
    log_text = safe_workspace_file(root, test_log_relative_path).read_text(encoding="utf-8")
    executed_counts = parse_unittest_summaries(log_text)
    if executed_counts != per_suite_test_counts:
        raise ReportFailure(
            "test log summaries do not match the discovered test suites: "
            f"expected {per_suite_test_counts}, observed {executed_counts}"
        )

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
        "workflow_ref": env["GITHUB_WORKFLOW_REF"],
        "workflow_sha": workflow_sha,
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
        "executed_test_count": sum(executed_counts),
        "executed_test_counts_by_suite": executed_counts,
        "artifacts": sorted(files, key=lambda item: item["path"]),
        "authority_boundary": {
            "proves_simulation_truth": False,
            "can_authorize_real_actions": False,
            "can_reserve_real_resources": False,
            "can_settle_real_assets": False,
            "production_ledger_import_allowed": False,
        },
    }


def build_attestation_predicate(report: dict[str, Any], report_bytes: bytes) -> dict[str, Any]:
    """Build a distinct custom predicate bound to the exact execution-report bytes."""
    return {
        "schema": PREDICATE_SCHEMA,
        "claim_ceiling": CLAIM_CEILING,
        "scope": report["scope"],
        "result": report["result"],
        "repository": report["repository"],
        "source_revision": report["source_revision"],
        "event_sha": report["event_sha"],
        "workflow": report["workflow"],
        "workflow_ref": report["workflow_ref"],
        "workflow_sha": report["workflow_sha"],
        "run_id": report["run_id"],
        "run_attempt": report["run_attempt"],
        "execution_record_digest": sha256_bytes(report_bytes),
        "executed_test_count": report["executed_test_count"],
        "executed_test_counts_by_suite": report["executed_test_counts_by_suite"],
        "authority_boundary": {
            "proves_simulation_truth": False,
            "can_authorize_real_actions": False,
            "can_reserve_real_resources": False,
            "can_settle_real_assets": False,
            "production_ledger_import_allowed": False,
        },
    }


def safe_output_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ReportFailure(f"output must be a safe relative path within the workspace: {relative}")
    parent = root
    for part in path.parts[:-1]:
        parent = parent / part
        if parent.is_symlink():
            raise ReportFailure(f"symlink components are forbidden in report output: {relative}")
        if parent.exists() and not parent.is_dir():
            raise ReportFailure(f"report output parent is not a directory: {relative}")
    parent.mkdir(parents=True, exist_ok=True)
    try:
        parent.resolve(strict=True).relative_to(root)
    except (ValueError, OSError) as exc:
        raise ReportFailure(f"report output escapes workspace: {relative}") from exc
    output_path = parent / path.parts[-1]
    if output_path.exists() or output_path.is_symlink():
        raise ReportFailure(f"refusing to overwrite existing report: {relative}")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--test-log", default="evidence/test-output.txt")
    parser.add_argument("--output", default="evidence/qualification-report.json")
    parser.add_argument("--predicate-output", default="evidence/attestation-predicate.json")
    args = parser.parse_args()
    root = Path(args.root).resolve(strict=True)
    try:
        if Path(args.output) == Path(args.predicate_output):
            raise ReportFailure("report and predicate outputs must be distinct files")
        report = build_report(root, dict(os.environ), args.test_log)
        output_path = safe_output_path(root, args.output)
        predicate_path = safe_output_path(root, args.predicate_output)
        report_bytes = (json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        predicate = build_attestation_predicate(report, report_bytes)
        predicate_bytes = (json.dumps(predicate, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        output_path.write_bytes(report_bytes)
        predicate_path.write_bytes(predicate_bytes)
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
