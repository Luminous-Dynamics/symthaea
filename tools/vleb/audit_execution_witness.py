#!/usr/bin/env python3
"""Independently audit evaluator-observed VLEB execution witness receipts.

A valid receipt binds the recorded process to a pinned plan, input/output bytes,
and external task/source/run/runtime identities. A valid receipt alone does not
prove that program behavior is correct.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError as exc:
    raise SystemExit("jsonschema==4.26.0 is required; refusing to skip schema validation") from exc

HEX = re.compile(r"^[a-f0-9]{64}$")
GIT_SHA = re.compile(r"^[a-f0-9]{40}$")


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


def safe_file(root: Path, relative: Any) -> Path:
    rel = safe_relative(relative)
    root_resolved = root.resolve(strict=True)
    current = root_resolved
    for part in rel.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink evidence path rejected")
    resolved = current.resolve(strict=True)
    if not resolved.is_relative_to(root_resolved) or not resolved.is_file():
        raise ValueError("path must be a regular file inside evidence root")
    return resolved


def validate_plan(plan: Any) -> dict[str, Any]:
    required = {
        "schema_version", "task_id", "claim_id", "repository", "run",
        "candidate_revision", "source_sha256", "runtime", "verifier",
        "invocation", "source_path", "inputs", "outputs",
    }
    if not isinstance(plan, dict) or set(plan) != required:
        raise ValueError("execution plan must be an object with the exact v1 field set")
    if plan["schema_version"] != "vleb-execution-witness-plan-v1":
        raise ValueError("unsupported execution plan schema")
    if not isinstance(plan["repository"], dict) or set(plan["repository"]) != {"full_name", "id"}:
        raise ValueError("invalid repository identity in plan")
    if not isinstance(plan["run"], dict) or set(plan["run"]) != {"run_id", "attempt"}:
        raise ValueError("invalid run identity in plan")
    if not isinstance(plan["runtime"], dict) or set(plan["runtime"]) != {"platform", "immutable_image_id", "network_mode"}:
        raise ValueError("invalid runtime identity in plan")
    if not isinstance(plan["verifier"], dict) or set(plan["verifier"]) != {"id", "executable_path", "executable_sha256"}:
        raise ValueError("invalid verifier identity in plan")
    if not isinstance(plan["invocation"], dict) or set(plan["invocation"]) != {
        "argv", "working_directory", "environment", "timeout_seconds", "expected_exit_code"
    }:
        raise ValueError("invalid invocation in plan")
    if not isinstance(plan["inputs"], list) or not plan["inputs"]:
        raise ValueError("plan input inventory must be non-empty")
    for row in plan["inputs"]:
        if not isinstance(row, dict) or set(row) != {"path", "sha256"}:
            raise ValueError("invalid input row in plan")
        safe_relative(row["path"])
        if not isinstance(row["sha256"], str) or not HEX.fullmatch(row["sha256"]):
            raise ValueError("invalid input digest in plan")
    if plan["source_path"] not in {row["path"] for row in plan["inputs"]}:
        raise ValueError("plan source_path is absent from its input inventory")
    source_input = next(row for row in plan["inputs"] if row["path"] == plan["source_path"])
    if source_input["sha256"] != plan["source_sha256"]:
        raise ValueError("plan source digest disagrees with source input pin")
    outputs = plan["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != {"stdout_path", "stderr_path"}:
        raise ValueError("invalid plan output paths")
    safe_relative(outputs["stdout_path"])
    safe_relative(outputs["stderr_path"])
    if outputs["stdout_path"] == outputs["stderr_path"]:
        raise ValueError("stdout and stderr paths must differ")
    invocation = plan["invocation"]
    if not isinstance(invocation["argv"], list) or not invocation["argv"] or any(not isinstance(x, str) for x in invocation["argv"]):
        raise ValueError("plan argv must be a non-empty string array")
    if not isinstance(invocation["environment"], dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in invocation["environment"].items()):
        raise ValueError("plan environment must be a string-to-string map")
    if invocation["argv"][0] != plan["verifier"]["executable_path"]:
        raise ValueError("plan argv[0] differs from verifier executable path")
    if not isinstance(plan["verifier"]["executable_sha256"], str) or not HEX.fullmatch(plan["verifier"]["executable_sha256"]):
        raise ValueError("invalid verifier executable digest")
    if not isinstance(plan["source_sha256"], str) or not HEX.fullmatch(plan["source_sha256"]):
        raise ValueError("invalid source digest")
    if not isinstance(plan["candidate_revision"], str) or not GIT_SHA.fullmatch(plan["candidate_revision"]):
        raise ValueError("invalid candidate revision")
    return plan


def identity_from_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": plan["task_id"],
        "claim_id": plan["claim_id"],
        "repository": plan["repository"],
        "run": plan["run"],
        "candidate_revision": plan["candidate_revision"],
        "source_sha256": plan["source_sha256"],
        "runtime": plan["runtime"],
    }


def audit_execution_receipt(
    *,
    receipt_path: Path,
    plan_path: Path,
    schema_path: Path,
    evidence_root: Path,
    expected_receipt_sha256: str,
    expected_auditor_sha256: str,
    expected_schema_sha256: str,
    expected_runner_sha256: str,
    expected_plan_sha256: str,
    expected_verifier_sha256: str,
    trusted_identity: dict[str, Any],
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    pins = {
        "receipt": expected_receipt_sha256,
        "auditor": expected_auditor_sha256,
        "schema": expected_schema_sha256,
        "runner": expected_runner_sha256,
        "plan": expected_plan_sha256,
        "verifier": expected_verifier_sha256,
    }
    for label, value in pins.items():
        if not isinstance(value, str) or not HEX.fullmatch(value):
            issues.append({"code": "invalid_trusted_pin", "message": f"{label} pin must be lowercase SHA-256"})
    if issues:
        return {"schema_version": "vleb-execution-witness-audit-v1", "evidence_valid": False, "issues": issues}

    try:
        observed_auditor = digest(Path(__file__).read_bytes())
        raw_schema = schema_path.read_bytes()
        observed_schema = digest(raw_schema)
        raw_receipt = receipt_path.read_bytes()
    except OSError as exc:
        return {
            "schema_version": "vleb-execution-witness-audit-v1",
            "evidence_valid": False,
            "verdict": "unresolved",
            "issues": [{"code": "trusted_input_unavailable", "message": str(exc)}],
        }
    if observed_auditor != expected_auditor_sha256:
        issues.append({"code": "auditor_pin_mismatch", "message": "running execution receipt auditor differs from evaluator pin"})
    if observed_schema != expected_schema_sha256:
        issues.append({"code": "schema_pin_mismatch", "message": "receipt schema differs from evaluator pin"})
    if digest(raw_receipt) != expected_receipt_sha256:
        issues.append({"code": "receipt_pin_mismatch", "message": "receipt bytes differ from evaluator-supplied pin"})
    if issues:
        return {"schema_version": "vleb-execution-witness-audit-v1", "evidence_valid": False, "verdict": "fail", "issues": issues}

    try:
        schema = json.loads(raw_schema.decode("utf-8", errors="strict"))
        receipt = json.loads(raw_receipt.decode("utf-8", errors="strict"))
        raw_plan = plan_path.read_bytes()
        plan_sha = digest(raw_plan)
        plan = validate_plan(json.loads(raw_plan.decode("utf-8", errors="strict")))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        return {
            "schema_version": "vleb-execution-witness-audit-v1",
            "evidence_valid": False,
            "verdict": "fail",
            "issues": [{"code": "execution_witness_input_invalid", "message": str(exc)}],
        }
    if plan_sha != expected_plan_sha256:
        issues.append({"code": "plan_pin_mismatch", "message": "execution plan bytes differ from evaluator pin"})
    try:
        Draft202012Validator.check_schema(schema)
        schema_errors = list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(receipt))
    except Exception as exc:
        return {
            "schema_version": "vleb-execution-witness-audit-v1",
            "evidence_valid": False,
            "verdict": "fail",
            "issues": [{"code": "schema_validator_error", "message": str(exc)}],
        }
    for error in schema_errors:
        loc = ".".join(map(str, error.absolute_path)) or "$"
        issues.append({"code": "receipt_schema_invalid", "message": f"{loc}: {error.message}"})

    if not isinstance(receipt, dict):
        issues.append({"code": "receipt_root_invalid", "message": "receipt root must be an object"})
        return {"schema_version": "vleb-execution-witness-audit-v1", "evidence_valid": False, "verdict": "fail", "issues": issues}
    if receipt.get("schema_version") != "vleb-execution-witness-v1":
        issues.append({"code": "receipt_version_invalid", "message": "unsupported witness receipt schema_version"})
    if receipt.get("plan_sha256") != expected_plan_sha256 or receipt.get("plan_sha256") != plan_sha:
        issues.append({"code": "plan_receipt_binding_mismatch", "message": "receipt plan digest does not bind exact plan bytes"})
    if receipt.get("producer_sha256") != expected_runner_sha256:
        issues.append({"code": "runner_pin_mismatch", "message": "receipt producer digest differs from evaluator-pinned runner"})
    if receipt.get("verifier", {}).get("executable_sha256") != expected_verifier_sha256:
        issues.append({"code": "verifier_pin_mismatch", "message": "receipt-selected verifier digest differs from evaluator pin"})
    if identity_from_plan(plan) != trusted_identity:
        issues.append({"code": "plan_identity_mismatch", "message": "pinned plan does not match evaluator-supplied task/run/source/runtime identity"})
    for field, expected in identity_from_plan(plan).items():
        if receipt.get(field) != expected:
            issues.append({"code": "receipt_identity_mismatch", "message": f"receipt field {field} does not match pinned plan"})
    if receipt.get("verifier", {}).get("id") != plan["verifier"]["id"]:
        issues.append({"code": "verifier_id_mismatch", "message": "receipt verifier ID differs from pinned plan"})
    try:
        observed_executable = digest(Path(plan["verifier"]["executable_path"]).resolve(strict=True).read_bytes())
        if observed_executable != expected_verifier_sha256:
            issues.append({"code": "verifier_executable_digest_mismatch", "message": "available verifier executable bytes differ from evaluator pin"})
    except OSError as exc:
        issues.append({"code": "verifier_executable_unavailable", "message": str(exc)})

    invocation = receipt.get("invocation", {})
    planned = plan["invocation"]
    env_sha = digest(canonical_json(planned["environment"]))
    for field, expected in (
        ("argv", planned["argv"]),
        ("working_directory", planned["working_directory"]),
        ("environment_sha256", env_sha),
        ("expected_exit_code", planned["expected_exit_code"]),
    ):
        if invocation.get(field) != expected:
            issues.append({"code": "invocation_plan_mismatch", "message": f"receipt invocation field {field} differs from pinned plan"})
    if receipt.get("plan_path") != str(safe_relative(plan_path.as_posix())):
        issues.append({"code": "plan_path_mismatch", "message": "receipt does not reference the expected relative plan path"})

    started = finished = None
    try:
        started = datetime.fromisoformat(invocation["started_at"].replace("Z", "+00:00"))
        finished = datetime.fromisoformat(invocation["finished_at"].replace("Z", "+00:00"))
        if started.tzinfo is None or finished.tzinfo is None or finished < started:
            raise ValueError("timestamps must be timezone-aware and nondecreasing")
        if not isinstance(invocation.get("duration_ms"), int) or isinstance(invocation.get("duration_ms"), bool) or invocation["duration_ms"] < 0:
            raise ValueError("duration_ms must be a nonnegative integer")
    except (KeyError, TypeError, ValueError) as exc:
        issues.append({"code": "execution_timing_invalid", "message": str(exc)})

    input_rows = receipt.get("inputs")
    expected_inputs = plan["inputs"]
    if not isinstance(input_rows, list) or len(input_rows) != len(expected_inputs):
        issues.append({"code": "input_inventory_mismatch", "message": "receipt input inventory differs from pinned plan"})
        input_rows = []
    else:
        for planned_input, observed in zip(expected_inputs, input_rows, strict=True):
            if not isinstance(observed, dict) or observed.get("path") != planned_input["path"]:
                issues.append({"code": "input_inventory_mismatch", "message": "receipt input path/order differs from pinned plan"})
                continue
            before = observed.get("sha256_before")
            after = observed.get("sha256_after")
            if before != planned_input["sha256"] or after != planned_input["sha256"]:
                issues.append({"code": "input_digest_mismatch", "message": f"input was changed or did not match plan: {planned_input['path']}"})
            try:
                actual = digest(safe_file(evidence_root, planned_input["path"]).read_bytes())
                if actual != planned_input["sha256"]:
                    issues.append({"code": "input_artifact_digest_mismatch", "message": f"input bytes differ from plan: {planned_input['path']}"})
            except (OSError, ValueError) as exc:
                issues.append({"code": "input_artifact_invalid", "message": str(exc)})

    outputs = receipt.get("outputs", {})
    plan_outputs = plan["outputs"]
    for key, path_key, sha_key in (
        ("stdout", "stdout_path", "stdout_sha256"),
        ("stderr", "stderr_path", "stderr_sha256"),
    ):
        if outputs.get(path_key) != plan_outputs[path_key]:
            issues.append({"code": "output_plan_mismatch", "message": f"{key} path differs from pinned plan"})
            continue
        try:
            observed_sha = digest(safe_file(evidence_root, outputs[path_key]).read_bytes())
            if observed_sha != outputs.get(sha_key):
                issues.append({"code": "output_digest_mismatch", "message": f"{key} bytes differ from receipt digest"})
        except (OSError, ValueError) as exc:
            issues.append({"code": "output_artifact_invalid", "message": f"{key}: {exc}"})

    timed_out = invocation.get("timed_out") is True
    actual_exit = invocation.get("exit_code")
    expected_exit = planned["expected_exit_code"]
    execution_contract_passed = not timed_out and actual_exit == expected_exit
    evidence_valid = not issues
    return {
        "schema_version": "vleb-execution-witness-audit-v1",
        "evidence_valid": evidence_valid,
        "verdict": "pass" if evidence_valid else "fail",
        "execution_contract_passed": execution_contract_passed,
        "observed_exit_code": actual_exit,
        "expected_exit_code": expected_exit,
        "timed_out": timed_out,
        "task_id": receipt.get("task_id"),
        "claim_id": receipt.get("claim_id"),
        "run_id": receipt.get("run", {}).get("run_id"),
        "run_attempt": receipt.get("run", {}).get("attempt"),
        "receipt_sha256": digest(raw_receipt),
        "plan_sha256": plan_sha,
        "issues": issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path(__file__).with_name("execution-witness-receipt.schema.json"))
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--expected-auditor-sha256", required=True)
    parser.add_argument("--expected-schema-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
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
    args = parser.parse_args(argv)
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
        result = audit_execution_receipt(
            receipt_path=args.receipt,
            plan_path=args.plan,
            schema_path=args.schema,
            evidence_root=args.evidence_root,
            expected_receipt_sha256=args.expected_receipt_sha256,
            expected_auditor_sha256=args.expected_auditor_sha256,
            expected_schema_sha256=args.expected_schema_sha256,
            expected_runner_sha256=args.expected_runner_sha256,
            expected_plan_sha256=args.expected_plan_sha256,
            expected_verifier_sha256=args.expected_verifier_sha256,
            trusted_identity=identity,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, subprocess.SubprocessError) as exc:
        result = {
            "schema_version": "vleb-execution-witness-audit-v1",
            "evidence_valid": False,
            "verdict": "unresolved",
            "issues": [{"code": "execution_witness_audit_unavailable", "message": str(exc)}],
        }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("evidence_valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
