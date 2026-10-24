#!/usr/bin/env python3
"""Run the frozen synthetic casting policy diagnostic and replay-score its output.

This source-locked harness is diagnostic tooling, not an external SOTA or
physical-validation run. All artifacts must be written outside the repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOCK_REL = Path("docs/research/CASTING_MULTIFIDELITY_POLICY_PRERUN_LOCK_V1.json")
VERIFIER_REL = Path("tools/casting-benchmark/validate_scenario_prerun_lock.py")
RUNNER_REL = Path("tools/casting-benchmark/multifidelity_scenario_policy_bakeoff.py")
SCORER_REL = Path("tools/casting-benchmark/score_scenario_policy_bakeoff.py")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo_root: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if proc.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def output_outside_repo(value: Path, repo_root: Path, label: str) -> Path:
    path = value.expanduser().resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError:
        return path
    raise ValueError(f"{label} must be outside the repository")


def validate_output_contract(result: Any, preregistration: dict[str, Any]) -> None:
    if not isinstance(result, dict):
        raise ValueError("runner output must be a JSON object")
    design = preregistration["design"]
    expected = {
        "benchmark_id": preregistration["benchmark_id"],
        "synthetic_fixture_only": True,
        "scenario_ids": design["scenario_ids"],
        "dimensions": design["dimensions"],
        "severities": design["severities"],
        "seeds": design["paired_seeds"],
        "design_count": design["design_count"],
        "cost_budget": design["cost_budget"],
        "fidelity_costs": design["fidelity_costs"],
        "policies": design["policy_ids"],
        "run_count": design["expected_run_count"],
        "summary_count": design["expected_cell_summary_count"],
        "paired_comparison_count": design["expected_paired_comparison_count"],
    }
    for key, expected_value in expected.items():
        if result.get(key) != expected_value:
            raise ValueError(
                f"runner output contract mismatch for {key}: "
                f"expected {expected_value!r}, got {result.get(key)!r}"
            )
    runs = result.get("paired_runs")
    if not isinstance(runs, list) or len(runs) != design["expected_run_count"]:
        raise ValueError("paired_runs must retain every preregistered run record")


def run_capture(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--lock-report", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    try:
        paths = {
            "output": output_outside_repo(args.output, repo_root, "output"),
            "receipt": output_outside_repo(args.receipt, repo_root, "receipt"),
            "lock_report": output_outside_repo(args.lock_report, repo_root, "lock report"),
            "bundle": output_outside_repo(args.bundle, repo_root, "bundle"),
        }
        if len(set(paths.values())) != len(paths):
            raise ValueError("all four output paths must be distinct")
        for label, path in paths.items():
            if path.exists():
                raise ValueError(f"{label} path already exists; refusing to overwrite: {path}")
        for relative in (LOCK_REL, VERIFIER_REL, RUNNER_REL, SCORER_REL):
            if not (repo_root / relative).is_file():
                raise ValueError(f"required tool is missing: {relative}")
        manifest = json.loads((repo_root / LOCK_REL).read_text(encoding="utf-8"))
        preregistration = manifest["preregistration"]
        command = preregistration["planned_command_argv"]
        if not isinstance(command, list) or len(command) < 2 or command[1] != str(RUNNER_REL):
            raise ValueError("preregistered runner command does not match the required runner path")
        if command[0] != "python3":
            raise ValueError("preregistered interpreter token must be python3")
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(json.dumps({"valid": False, "status": "LOCKED_RUN_REFUSED", "error": str(exc)}, sort_keys=True))
        return 2

    started = utc_now()
    head_before = git(repo_root, "rev-parse", "HEAD")
    verifier_command = [
        sys.executable, str(repo_root / VERIFIER_REL),
        "--repo-root", str(repo_root), "--lock", str(repo_root / LOCK_REL),
    ]
    pre = run_capture(verifier_command, repo_root)
    if pre.returncode != 0:
        print(pre.stdout, file=sys.stderr)
        print(pre.stderr, file=sys.stderr)
        return 2
    try:
        pre_report = json.loads(pre.stdout)
        if pre_report.get("valid") is not True or pre_report.get("execution_head_sha") != head_before:
            raise ValueError("pre-run verifier report is invalid or refers to a different HEAD")
        paths["lock_report"].write_text(json.dumps(pre_report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"PRE_RUN_LOCK_REPORT_REFUSED: {exc}", file=sys.stderr)
        return 2

    runner_command = [sys.executable, *command[1:]]
    runner_code = 2
    runner_error = ""
    scorer_code: int | None = None
    contract_error: str | None = None
    post_report: dict[str, Any] | None = None
    try:
        with paths["output"].open("xb") as raw_output:
            runner = subprocess.run(
                runner_command, cwd=repo_root, stdout=raw_output,
                stderr=subprocess.PIPE, check=False,
            )
        runner_code = runner.returncode
        runner_error = runner.stderr.decode("utf-8", errors="replace")
        if runner_code == 0:
            result = json.loads(paths["output"].read_text(encoding="utf-8"))
            validate_output_contract(result, preregistration)
            scorer_command = [
                sys.executable, str(repo_root / SCORER_REL),
                str(paths["output"]), "--receipt", str(paths["receipt"]),
            ]
            scorer = run_capture(scorer_command, repo_root)
            scorer_code = scorer.returncode
            if scorer_code != 0:
                runner_error += "\nSCORER STDERR:\n" + scorer.stderr
        else:
            contract_error = "runner returned a non-zero exit code"
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        contract_error = str(exc)

    try:
        post = run_capture(verifier_command, repo_root)
        if post.returncode == 0:
            post_report = json.loads(post.stdout)
        else:
            runner_error += "\nPOST_RUN_LOCK VERIFICATION:\n" + post.stderr + post.stdout
    except (OSError, json.JSONDecodeError):
        post_report = None

    same_head = git(repo_root, "rev-parse", "HEAD") == head_before
    valid_post = (
        post_report is not None
        and post_report.get("valid") is True
        and post_report.get("execution_head_sha") == head_before
        and post_report.get("lock_sha256") == pre_report.get("lock_sha256")
    )
    output_exists = paths["output"].is_file()
    receipt_exists = paths["receipt"].is_file()
    trace_replay_pass = (
        runner_code == 0 and scorer_code == 0 and output_exists and receipt_exists
        and contract_error is None
    )
    success = trace_replay_pass and valid_post and same_head
    bundle: dict[str, Any] = {
        "schema_version": "1.0.0",
        "bundle_id": "CASTING-MF-POLICY-SCENARIOS-RUN-BUNDLE-001",
        "status": (
            "SYNTHETIC_TRACE_REPLAYED_NOT_SOTA_EVIDENCE" if success
            else "RUN_INCOMPLETE_OR_REJECTED"
        ),
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "execution_head_sha": head_before,
        "source_lock_report": pre_report,
        "post_run_lock_report": post_report,
        "execution_head_unchanged": same_head,
        "post_run_lock_valid": valid_post,
        "runner_argv": runner_command,
        "runner_return_code": runner_code,
        "runner_stderr": runner_error[-8000:],
        "output_contract_error": contract_error,
        "observed_run_count": None,
        "raw_result_path": str(paths["output"]),
        "raw_result_sha256": sha256_file(paths["output"]) if output_exists else None,
        "receipt_path": str(paths["receipt"]),
        "receipt_sha256": sha256_file(paths["receipt"]) if receipt_exists else None,
        "trace_replay_return_code": scorer_code,
        "claim_ceiling": "SYNTHETIC_TRACE_REPLAYED_NOT_SOTA_EVIDENCE",
        "non_claims": [
            "No external BoTorch or COCO baseline is run by this harness.",
            "No physical casting simulator or held-out foundry dataset is validated.",
            "A successful bundle establishes source-locked internal trace consistency only.",
        ],
    }
    if output_exists:
        try:
            loaded = json.loads(paths["output"].read_text(encoding="utf-8"))
            bundle["observed_run_count"] = loaded.get("run_count")
        except (OSError, json.JSONDecodeError):
            pass
    try:
        paths["bundle"].write_text(json.dumps(bundle, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except OSError as exc:
        print(json.dumps({"valid": False, "status": "BUNDLE_WRITE_FAILED", "error": str(exc)}, sort_keys=True))
        return 2

    print(json.dumps({
        "valid": success,
        "status": bundle["status"],
        "execution_head_sha": head_before,
        "run_count": bundle["observed_run_count"],
        "raw_result_sha256": bundle["raw_result_sha256"],
        "receipt_sha256": bundle["receipt_sha256"],
        "bundle_path": str(paths["bundle"]),
        "error": contract_error,
    }, sort_keys=True, indent=2))
    return 0 if success else 3


if __name__ == "__main__":
    raise SystemExit(main())
