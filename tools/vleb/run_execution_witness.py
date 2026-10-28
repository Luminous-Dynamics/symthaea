#!/usr/bin/env python3
"""Run one evaluator-pinned command and emit an execution witness receipt.

The plan and identities must come from trusted evaluator configuration. This
recorder does not decide whether program behavior is semantically correct.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

HEX = re.compile(r"^[a-f0-9]{64}$")
GIT_SHA = re.compile(r"^[a-f0-9]{40}$")
ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
MAX_TIMEOUT_SECONDS = 300


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def safe_relative(value: Any, *, allow_dot: bool = False) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("path must be a non-empty POSIX relative path")
    rel = PurePosixPath(value)
    if rel.is_absolute() or ".." in rel.parts or str(rel) != value:
        raise ValueError("absolute, traversal, or ambiguous path rejected")
    if value == "." and allow_dot:
        return rel
    if not rel.parts or any(part in ("", ".") for part in rel.parts):
        raise ValueError("non-canonical relative path rejected")
    return rel


def safe_existing_file(root: Path, relative: Any) -> Path:
    rel = safe_relative(relative)
    root = root.resolve(strict=True)
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink evidence path rejected")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("path must resolve to a regular file inside evidence root")
    return resolved


def safe_directory(root: Path, relative: Any) -> Path:
    rel = safe_relative(relative, allow_dot=True)
    root = root.resolve(strict=True)
    current = root
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink working-directory component rejected")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_dir():
        raise ValueError("working directory must be a directory inside evidence root")
    return resolved


def prepare_new_file(root: Path, relative: Any) -> Path:
    rel = safe_relative(relative)
    root = root.resolve(strict=True)
    current = root
    for part in rel.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink output-directory component rejected")
        if current.exists():
            if not current.is_dir():
                raise ValueError("output parent component is not a directory")
        else:
            current.mkdir()
    target = current / rel.parts[-1]
    if target.is_symlink() or target.exists():
        raise ValueError(f"refusing to overwrite existing output: {relative}")
    if not target.parent.resolve(strict=True).is_relative_to(root):
        raise ValueError("output parent escapes evidence root")
    return target


def validate_plan(plan: Any) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("execution plan root must be an object")
    required = {
        "schema_version", "task_id", "claim_id", "repository", "run",
        "candidate_revision", "source_sha256", "runtime", "verifier",
        "invocation", "source_path", "inputs", "outputs",
    }
    if set(plan) != required:
        raise ValueError(f"execution plan fields differ from required set: {sorted(set(plan) ^ required)}")
    if plan.get("schema_version") != "vleb-execution-witness-plan-v1":
        raise ValueError("unsupported execution witness plan schema_version")
    for field in ("task_id", "claim_id", "source_path"):
        if not isinstance(plan.get(field), str) or not plan[field]:
            raise ValueError(f"{field} must be a non-empty string")
    repo = plan["repository"]
    run = plan["run"]
    runtime = plan["runtime"]
    verifier = plan["verifier"]
    invocation = plan["invocation"]
    if not isinstance(repo, dict) or set(repo) != {"full_name", "id"}:
        raise ValueError("repository must contain only full_name and id")
    if not isinstance(repo["full_name"], str) or not repo["full_name"] or not isinstance(repo["id"], int) or isinstance(repo["id"], bool) or repo["id"] < 1:
        raise ValueError("invalid repository identity")
    if not isinstance(run, dict) or set(run) != {"run_id", "attempt"}:
        raise ValueError("run must contain only run_id and attempt")
    if not isinstance(run["run_id"], str) or not run["run_id"] or not isinstance(run["attempt"], int) or isinstance(run["attempt"], bool) or run["attempt"] < 1:
        raise ValueError("invalid run identity")
    if not isinstance(plan["candidate_revision"], str) or not GIT_SHA.fullmatch(plan["candidate_revision"]):
        raise ValueError("candidate_revision must be a lowercase Git SHA-1")
    if not isinstance(plan["source_sha256"], str) or not HEX.fullmatch(plan["source_sha256"]):
        raise ValueError("source_sha256 must be lowercase SHA-256")
    if not isinstance(runtime, dict) or set(runtime) != {"platform", "immutable_image_id", "network_mode"}:
        raise ValueError("runtime must contain platform, immutable_image_id and network_mode")
    if any(not isinstance(runtime[k], str) or not runtime[k] for k in runtime):
        raise ValueError("runtime values must be non-empty strings")
    if runtime["network_mode"] not in {"none", "restricted", "unrestricted", "unknown"}:
        raise ValueError("unsupported network_mode")
    if not isinstance(verifier, dict) or set(verifier) != {"id", "executable_path", "executable_sha256"}:
        raise ValueError("verifier must contain id, executable_path and executable_sha256")
    if not isinstance(verifier["id"], str) or not verifier["id"]:
        raise ValueError("verifier id must be non-empty")
    if not isinstance(verifier["executable_path"], str) or not Path(verifier["executable_path"]).is_absolute():
        raise ValueError("verifier executable_path must be absolute")
    if not isinstance(verifier["executable_sha256"], str) or not HEX.fullmatch(verifier["executable_sha256"]):
        raise ValueError("verifier executable_sha256 must be lowercase SHA-256")
    if not isinstance(invocation, dict) or set(invocation) != {
        "argv", "working_directory", "environment", "timeout_seconds", "expected_exit_code"
    }:
        raise ValueError("invocation has an unexpected or missing field")
    argv = invocation["argv"]
    if not isinstance(argv, list) or not argv or any(not isinstance(item, str) or "\x00" in item for item in argv):
        raise ValueError("argv must be a non-empty array of strings without NUL bytes")
    if argv[0] != verifier["executable_path"]:
        raise ValueError("argv[0] must equal the pinned verifier executable path")
    safe_relative(invocation["working_directory"], allow_dot=True)
    env = invocation["environment"]
    if not isinstance(env, dict) or any(
        not isinstance(key, str) or not ENV_KEY.fullmatch(key) or not isinstance(value, str) or "\x00" in value
        for key, value in env.items()
    ):
        raise ValueError("environment must be a map of valid names to string values")
    if any(re.search(r"(token|secret|password|credential|private.?key)", key, re.I) for key in env):
        raise ValueError("secret-like environment variable names are not accepted in witness plans")
    timeout = invocation["timeout_seconds"]
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0 or timeout > MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout_seconds must be in (0, {MAX_TIMEOUT_SECONDS}]")
    exit_code = invocation["expected_exit_code"]
    if not isinstance(exit_code, int) or isinstance(exit_code, bool) or exit_code < -255 or exit_code > 255:
        raise ValueError("expected_exit_code must be an integer in [-255, 255]")
    if not isinstance(plan["inputs"], list) or not plan["inputs"]:
        raise ValueError("inputs must be a non-empty array")
    input_paths: set[str] = set()
    for entry in plan["inputs"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError("each input must contain only path and sha256")
        safe_relative(entry["path"])
        if entry["path"] in input_paths:
            raise ValueError("duplicate input path")
        input_paths.add(entry["path"])
        if not isinstance(entry["sha256"], str) or not HEX.fullmatch(entry["sha256"]):
            raise ValueError("input sha256 must be lowercase SHA-256")
    if plan["source_path"] not in input_paths:
        raise ValueError("source_path must be included in the pinned input inventory")
    source_input = next(x for x in plan["inputs"] if x["path"] == plan["source_path"])
    if source_input["sha256"] != plan["source_sha256"]:
        raise ValueError("source input digest differs from source_sha256")
    outputs = plan["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != {"stdout_path", "stderr_path"}:
        raise ValueError("outputs must contain stdout_path and stderr_path")
    output_paths = {safe_relative(outputs["stdout_path"]), safe_relative(outputs["stderr_path"])}
    if len(output_paths) != 2 or input_paths.intersection(str(p) for p in output_paths):
        raise ValueError("output paths must be unique and must not overwrite an input")
    if not isinstance(plan["claim_id"], str) or not re.fullmatch(r"^[A-Z0-9][A-Z0-9._-]{1,127}$", plan["claim_id"]):
        raise ValueError("invalid claim_id")
    return plan


def expected_identity(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": plan["task_id"],
        "claim_id": plan["claim_id"],
        "repository": plan["repository"],
        "run": plan["run"],
        "candidate_revision": plan["candidate_revision"],
        "source_sha256": plan["source_sha256"],
        "runtime": plan["runtime"],
    }


def record_execution(
    *,
    plan_path: Path,
    evidence_root: Path,
    receipt_path: str,
    expected_plan_sha256: str,
    expected_runner_sha256: str,
    expected_verifier_sha256: str,
    trusted_identity: dict[str, Any],
) -> dict[str, Any]:
    for name, value in (
        ("plan", expected_plan_sha256),
        ("runner", expected_runner_sha256),
        ("verifier", expected_verifier_sha256),
    ):
        if not isinstance(value, str) or not HEX.fullmatch(value):
            raise ValueError(f"expected {name} digest must be lowercase SHA-256")
    observed_runner_sha = digest(Path(__file__).read_bytes())
    if observed_runner_sha != expected_runner_sha256:
        raise ValueError("execution witness runner differs from externally pinned source")
    root = evidence_root.resolve(strict=True)
    if plan_path.is_absolute():
        try:
            plan_relative = plan_path.relative_to(root).as_posix()
        except ValueError as exc:
            raise ValueError("execution plan must be located inside evidence root") from exc
    else:
        plan_relative = plan_path.as_posix()
    plan_real = safe_existing_file(root, plan_relative)
    raw_plan = plan_real.read_bytes()
    if digest(raw_plan) != expected_plan_sha256:
        raise ValueError("execution plan bytes differ from externally pinned digest")
    plan = validate_plan(json.loads(raw_plan.decode("utf-8", errors="strict")))
    if plan["verifier"]["executable_sha256"] != expected_verifier_sha256:
        raise ValueError("plan-selected verifier differs from externally pinned executable digest")
    exe = Path(plan["verifier"]["executable_path"]).resolve(strict=True)
    if not exe.is_file() or digest(exe.read_bytes()) != expected_verifier_sha256:
        raise ValueError("actual verifier executable bytes differ from external pin")
    for field in ("task_id", "claim_id", "repository", "run", "candidate_revision", "source_sha256", "runtime"):
        if trusted_identity.get(field) != plan[field]:
            raise ValueError(f"execution plan {field} differs from externally supplied identity")
    if Path(receipt_path).is_absolute():
        raise ValueError("receipt path must be relative to evidence root")
    receipt_rel = safe_relative(receipt_path)
    reserved = {str(receipt_rel), str(safe_relative(plan_relative))}
    if str(receipt_rel) in {plan["outputs"]["stdout_path"], plan["outputs"]["stderr_path"]}:
        raise ValueError("receipt path conflicts with a command output")
    if str(receipt_rel) in {item["path"] for item in plan["inputs"]}:
        raise ValueError("receipt path conflicts with an input")
    output_paths = {plan["outputs"]["stdout_path"], plan["outputs"]["stderr_path"]}
    if len(output_paths) != 2 or output_paths & reserved:
        raise ValueError("output paths must not overlap plan or receipt")
    cwd = safe_directory(root, plan["invocation"]["working_directory"])
    input_rows: list[dict[str, str]] = []
    for item in plan["inputs"]:
        resolved = safe_existing_file(root, item["path"])
        before = digest(resolved.read_bytes())
        if before != item["sha256"]:
            raise ValueError(f"input digest does not match plan: {item['path']}")
        input_rows.append({"path": item["path"], "sha256_before": before, "sha256_after": before})
    env = plan["invocation"]["environment"]
    env_sha = digest(canonical_json(env))
    started_wall = datetime.now(timezone.utc)
    start_monotonic = time.monotonic()
    timed_out = False
    exit_code: int | None
    stdout = b""
    stderr = b""
    try:
        completed = subprocess.run(
            plan["invocation"]["argv"],
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=plan["invocation"]["timeout_seconds"],
            check=False,
            shell=False,
        )
        exit_code = completed.returncode
        stdout, stderr = completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = None
        stdout = exc.stdout if isinstance(exc.stdout, bytes) else (exc.stdout or "").encode()
        stderr = exc.stderr if isinstance(exc.stderr, bytes) else (exc.stderr or "").encode()
    finished_monotonic = time.monotonic()
    finished_wall = datetime.now(timezone.utc)
    duration_ms = max(0, int((finished_monotonic - start_monotonic) * 1000))
    stdout_target = prepare_new_file(root, plan["outputs"]["stdout_path"])
    stderr_target = prepare_new_file(root, plan["outputs"]["stderr_path"])
    stdout_target.write_bytes(stdout)
    stderr_target.write_bytes(stderr)
    for item, row in zip(plan["inputs"], input_rows, strict=True):
        row["sha256_after"] = digest(safe_existing_file(root, item["path"]).read_bytes())
    receipt: dict[str, Any] = {
        "schema_version": "vleb-execution-witness-v1",
        **expected_identity(plan),
        "plan_path": str(safe_relative(plan_relative)),
        "plan_sha256": expected_plan_sha256,
        "verifier": {
            "id": plan["verifier"]["id"],
            "executable_path": str(exe),
            "executable_sha256": expected_verifier_sha256,
        },
        "invocation": {
            "argv": plan["invocation"]["argv"],
            "working_directory": plan["invocation"]["working_directory"],
            "environment_sha256": env_sha,
            "started_at": started_wall.isoformat(),
            "finished_at": finished_wall.isoformat(),
            "duration_ms": duration_ms,
            "expected_exit_code": plan["invocation"]["expected_exit_code"],
            "exit_code": exit_code,
            "timed_out": timed_out,
        },
        "inputs": input_rows,
        "outputs": {
            "stdout_path": plan["outputs"]["stdout_path"],
            "stdout_sha256": digest(stdout),
            "stderr_path": plan["outputs"]["stderr_path"],
            "stderr_sha256": digest(stderr),
        },
        "producer_sha256": expected_runner_sha256,
    }
    receipt_target = prepare_new_file(root, str(receipt_rel))
    receipt_target.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "receipt": receipt,
        "receipt_path": str(receipt_target),
        "receipt_sha256": digest(receipt_target.read_bytes()),
        "execution_contract_passed": (not timed_out and exit_code == plan["invocation"]["expected_exit_code"]),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "duration_ms": duration_ms,
    }


def expected_identity_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": plan["task_id"],
        "claim_id": plan["claim_id"],
        "repository": plan["repository"],
        "run": plan["run"],
        "candidate_revision": plan["candidate_revision"],
        "source_sha256": plan["source_sha256"],
        "runtime": plan["runtime"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="plan file inside the evidence root")
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--receipt-path", required=True, help="new receipt path relative to evidence root")
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-verifier-sha256", required=True)
    parser.add_argument("--expected-task-id", required=True)
    parser.add_argument("--expected-claim-id", required=True)
    parser.add_argument("--expected-repository", required=True)
    parser.add_argument("--expected-repository-id", type=int, required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-run-attempt", type=int, required=True)
    parser.add_argument("--expected-candidate-revision", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-runtime-image-id", required=True)
    parser.add_argument("--expected-platform", required=True)
    parser.add_argument("--expected-network-mode", choices=["none", "restricted", "unrestricted", "unknown"], required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    identity = {
        "task_id": args.expected_task_id,
        "claim_id": args.expected_claim_id,
        "repository": {"full_name": args.expected_repository, "id": args.expected_repository_id},
        "run": {"run_id": args.expected_run_id, "attempt": args.expected_run_attempt},
        "candidate_revision": args.expected_candidate_revision,
        "source_sha256": args.expected_source_sha256,
        "runtime": {
            "platform": args.expected_platform,
            "immutable_image_id": args.expected_runtime_image_id,
            "network_mode": args.expected_network_mode,
        },
    }
    try:
        result = record_execution(
            plan_path=args.plan,
            evidence_root=args.evidence_root,
            receipt_path=args.receipt_path,
            expected_plan_sha256=args.expected_plan_sha256,
            expected_runner_sha256=args.expected_runner_sha256,
            expected_verifier_sha256=args.expected_verifier_sha256,
            trusted_identity=identity,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, subprocess.SubprocessError) as exc:
        print(json.dumps({
            "schema_version": "vleb-execution-witness-run-v1",
            "eligible": False,
            "verdict": "fail",
            "issues": [{"code": "execution_witness_run_failed", "message": str(exc)}],
        }, indent=2))
        return 1
    print(json.dumps({
        "schema_version": "vleb-execution-witness-run-v1",
        "eligible": result["execution_contract_passed"],
        "verdict": "pass" if result["execution_contract_passed"] else "fail",
        "receipt_path": result["receipt_path"],
        "receipt_sha256": result["receipt_sha256"],
        "exit_code": result["exit_code"],
        "timed_out": result["timed_out"],
        "duration_ms": result["duration_ms"],
        "note": "execution-contract pass means only that the observed exit code matched the evaluator-pinned plan; semantic correctness requires a separate trusted verifier",
    }, indent=2))
    return 0 if result["execution_contract_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
