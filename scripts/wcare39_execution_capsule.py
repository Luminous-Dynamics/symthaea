#!/usr/bin/env python3
"""WCARE-39 execution capsule capture and runner.

Dependency-free qualification tooling. It records exact source/tool/material state,
executes preregistered argv arrays without a shell, and refuses to mix evidence
across immutable environment drift.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import locale
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

PROTOCOL = "wcare39-execution-capsule-v1"
STANDARD_MATERIALS = {
    "Cargo.lock": True,
    "flake.lock": True,
    "rust-toolchain.toml": True,
}
SAFE_LITERAL_DENY_TOKENS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "COOKIE", "AUTH", "API_KEY", "PRIVATE_KEY")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
        timeout=30,
    )


def repo_root() -> Path:
    process = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if process.returncode != 0:
        raise RuntimeError("not_in_git_worktree")
    return Path(process.stdout.decode().strip()).resolve()


def read_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("json_root_not_object")
    return raw, value


def validate_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise ValueError(f"{name}_invalid_sha256")
    return value


def validate_plan(plan: dict[str, Any]) -> None:
    required = {
        "protocol_version", "subject_git_head", "subject_digests", "materials",
        "safe_environment", "network_policy_declared", "sandbox_policy_declared",
        "deterministic_seed_commitments", "stages",
    }
    allowed = required | {"notes"}
    if set(plan) - allowed:
        raise ValueError("plan_unknown_fields")
    if required - set(plan):
        raise ValueError("plan_missing_fields")
    if plan["protocol_version"] != PROTOCOL:
        raise ValueError("plan_protocol_mismatch")
    head = plan["subject_git_head"]
    if not isinstance(head, str) or len(head) != 40 or any(ch not in "0123456789abcdef" for ch in head):
        raise ValueError("plan_subject_git_head_invalid")
    digests = plan["subject_digests"]
    if not isinstance(digests, dict):
        raise ValueError("plan_subject_digests_not_object")
    for path, digest in digests.items():
        if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("plan_subject_digest_path_invalid")
        validate_sha256(digest, f"subject_digest:{path}")
    if not isinstance(plan["materials"], list):
        raise ValueError("plan_materials_not_array")
    seen_materials: set[str] = set()
    for entry in plan["materials"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "required"}:
            raise ValueError("plan_material_entry_invalid")
        path = entry["path"]
        if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("plan_material_path_invalid")
        if path in seen_materials:
            raise ValueError("plan_duplicate_material")
        seen_materials.add(path)
        if not isinstance(entry["required"], bool):
            raise ValueError("plan_material_required_not_boolean")
    if not isinstance(plan["safe_environment"], list):
        raise ValueError("plan_safe_environment_not_array")
    seen_env: set[str] = set()
    for entry in plan["safe_environment"]:
        if not isinstance(entry, dict) or set(entry) != {"key", "mode"}:
            raise ValueError("plan_safe_environment_entry_invalid")
        key, mode = entry["key"], entry["mode"]
        if not isinstance(key, str) or not key or not key.replace("_", "A").isalnum() or key.upper() != key:
            raise ValueError("plan_safe_environment_key_invalid")
        if key in seen_env:
            raise ValueError("plan_duplicate_environment_key")
        seen_env.add(key)
        if mode not in {"Literal", "Sha256"}:
            raise ValueError("plan_safe_environment_mode_invalid")
        if mode == "Literal" and any(token in key for token in SAFE_LITERAL_DENY_TOKENS):
            raise ValueError(f"sensitive_environment_literal_forbidden:{key}")
    if plan["network_policy_declared"] not in {"Unspecified", "Allow", "Deny", "Restricted"}:
        raise ValueError("plan_network_policy_invalid")
    if plan["sandbox_policy_declared"] not in {"Unspecified", "None", "Restricted", "DefaultDeny"}:
        raise ValueError("plan_sandbox_policy_invalid")
    seeds = plan["deterministic_seed_commitments"]
    if not isinstance(seeds, dict):
        raise ValueError("plan_seed_commitments_not_object")
    for name, digest in seeds.items():
        if not isinstance(name, str) or not name:
            raise ValueError("plan_seed_name_invalid")
        validate_sha256(digest, f"seed:{name}")
    stages = plan["stages"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("plan_stages_empty")
    seen_stages: set[str] = set()
    for stage in stages:
        expected = {"stage_id", "argv", "cwd", "timeout_seconds", "output_receipt_path"}
        if not isinstance(stage, dict) or set(stage) != expected:
            raise ValueError("plan_stage_fields_invalid")
        stage_id = stage["stage_id"]
        if not isinstance(stage_id, str) or not stage_id or stage_id in seen_stages:
            raise ValueError("plan_stage_id_invalid_or_duplicate")
        seen_stages.add(stage_id)
        argv = stage["argv"]
        if not isinstance(argv, list) or not argv or any(not isinstance(item, str) or "\x00" in item for item in argv):
            raise ValueError("plan_stage_argv_invalid")
        cwd = stage["cwd"]
        if not isinstance(cwd, str) or Path(cwd).is_absolute() or ".." in Path(cwd).parts:
            raise ValueError("plan_stage_cwd_invalid")
        timeout = stage["timeout_seconds"]
        if not isinstance(timeout, int) or isinstance(timeout, bool) or not 1 <= timeout <= 86400:
            raise ValueError("plan_stage_timeout_invalid")
        output = stage["output_receipt_path"]
        if output is not None and (not isinstance(output, str) or not output or Path(output).is_absolute() or ".." in Path(output).parts):
            raise ValueError("plan_stage_output_path_invalid")


def auto_materials(plan: dict[str, Any]) -> dict[str, bool]:
    materials = dict(STANDARD_MATERIALS)
    for entry in plan["materials"]:
        materials[entry["path"]] = materials.get(entry["path"], False) or entry["required"]
    invokes_auth = any(
        "wcare37" in " ".join(stage["argv"]).lower() or "wcare38" in " ".join(stage["argv"]).lower()
        for stage in plan["stages"]
    )
    if invokes_auth:
        materials["tools/wcare37_attestation_verifier/Cargo.lock"] = True
    return materials


def capture_materials(root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for relative, required in sorted(auto_materials(plan).items()):
        path = root / relative
        present = path.is_file()
        output.append({
            "path": relative,
            "required": required,
            "present": present,
            "sha256": sha256_file(path) if present else None,
        })
    return output


def capture_subject_digests(root: Path, plan: dict[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in sorted(plan["subject_digests"].items()):
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"subject_material_missing:{relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"subject_digest_mismatch:{relative}:{actual}")
        observed[relative] = actual
    return observed


def safe_environment(plan: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for entry in sorted(plan["safe_environment"], key=lambda item: item["key"]):
        key = entry["key"]
        if key not in os.environ:
            output.append({"key": key, "mode": "Absent", "value": None})
            continue
        value = os.environ[key]
        if entry["mode"] == "Literal":
            output.append({"key": key, "mode": "Literal", "value": value})
        else:
            output.append({"key": key, "mode": "Sha256", "value": sha256_bytes(value.encode())})
    return output


def tool_version(executable: str) -> str:
    base = Path(executable).name.lower()
    args = [executable, "--version"]
    if base in {"python", "python3", "python3.11", "python3.12", "python3.13"}:
        args = [executable, "--version"]
    try:
        proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return "<version-unavailable>"
    return proc.stdout.decode(errors="replace")[:4096].strip()


def resolve_tool(command: str, root: Path) -> dict[str, Any]:
    candidate = Path(command)
    if candidate.is_absolute() or "/" in command:
        path = candidate if candidate.is_absolute() else (root / candidate)
        resolved = path.resolve() if path.exists() else path
    else:
        found = shutil.which(command)
        if found is None:
            raise RuntimeError(f"tool_not_found:{command}")
        resolved = Path(found).resolve()
    if not resolved.is_file():
        raise RuntimeError(f"tool_not_file:{command}")
    version = tool_version(str(resolved))
    return {
        "role": command,
        "executable_path": str(resolved),
        "executable_sha256": sha256_file(resolved),
        "version_output_sha256": sha256_bytes(version.encode()),
        "version_output": version,
    }


def capture_tools(root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    commands = {"git", sys.executable}
    commands.update(stage["argv"][0] for stage in plan["stages"])
    return sorted((resolve_tool(command, root) for command in commands), key=lambda item: item["role"])


def capture_platform() -> dict[str, str]:
    try:
        current_locale = locale.setlocale(locale.LC_ALL, None) or ""
    except locale.Error:
        current_locale = "<locale-unavailable>"
    timezone_repr = json.dumps(
        {"tzname": list(time.tzname), "timezone": time.timezone, "daylight": time.daylight},
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "os": platform.system(),
        "kernel_release": platform.release(),
        "architecture": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "locale": current_locale,
        "timezone": timezone_repr,
    }


def command_skeleton(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "stage_id": stage["stage_id"],
            "argv": stage["argv"],
            "cwd": stage["cwd"],
            "timeout_seconds": stage["timeout_seconds"],
            "started_utc": None,
            "finished_utc": None,
            "exit_code": None,
            "termination_class": "NotRun",
            "stdout_sha256": None,
            "stderr_sha256": None,
            "output_receipt_sha256": None,
            "subject_outcome": "NOT_RUN",
        }
        for stage in plan["stages"]
    ]


def immutable_view(capsule: dict[str, Any]) -> dict[str, Any]:
    commands = [
        {"stage_id": item["stage_id"], "argv": item["argv"], "cwd": item["cwd"], "timeout_seconds": item["timeout_seconds"]}
        for item in capsule["commands"]
    ]
    return {
        "subject_git_head": capsule["subject_git_head"],
        "worktree_clean": capsule["worktree_clean"],
        "repository_root_commitment_sha256": capsule["repository_root_commitment_sha256"],
        "subject_digests": capsule["subject_digests"],
        "command_plan_sha256": capsule["command_plan_sha256"],
        "platform": capsule["platform"],
        "tools": capsule["tools"],
        "materials": capsule["materials"],
        "safe_environment": capsule["safe_environment"],
        "network_policy_declared": capsule["network_policy_declared"],
        "sandbox_policy_declared": capsule["sandbox_policy_declared"],
        "deterministic_seed_commitments": capsule["deterministic_seed_commitments"],
        "commands": commands,
    }


def compare_immutable(prepared: dict[str, Any], final: dict[str, Any]) -> list[str]:
    p, f = immutable_view(prepared), immutable_view(final)
    return sorted(key for key in p if p[key] != f[key])


def capture_base(root: Path, plan_raw: bytes, plan: dict[str, Any]) -> dict[str, Any]:
    current_head = git(root, "rev-parse", "HEAD").stdout.decode().strip()
    status = git(root, "status", "--porcelain=v1", "--untracked-files=normal").stdout
    worktree_clean = not status.strip()
    origin = git(root, "remote", "get-url", "origin", check=False)
    repo_identity = origin.stdout.strip() if origin.returncode == 0 and origin.stdout.strip() else root.name.encode()
    return {
        "protocol_version": PROTOCOL,
        "authority": "MeasurementOnly",
        "capsule_phase": "PREPARED",
        "classification": "CAPSULE_PREPARED",
        "environment_integrity": "QUALIFIED",
        "subject_outcome": "NOT_RUN",
        "subject_git_head": current_head,
        "worktree_clean": worktree_clean,
        "repository_root_commitment_sha256": sha256_bytes(repo_identity),
        "subject_digests": capture_subject_digests(root, plan),
        "command_plan_sha256": sha256_bytes(plan_raw),
        "platform": capture_platform(),
        "tools": capture_tools(root, plan),
        "materials": capture_materials(root, plan),
        "safe_environment": safe_environment(plan),
        "network_policy_declared": plan["network_policy_declared"],
        "sandbox_policy_declared": plan["sandbox_policy_declared"],
        "deterministic_seed_commitments": dict(sorted(plan["deterministic_seed_commitments"].items())),
        "commands": command_skeleton(plan),
        "prepared_capsule_sha256": None,
        "drift_fields": [],
        "created_utc": utc_now(),
        "network_isolation_established": False,
        "sandbox_enforcement_established": False,
        "independent_builder_established": False,
        "phenomenal_experience_established": False,
        "suffering_established": False,
        "moral_patienthood_established": False,
        "binding_consent_established": False,
        "veto_authority_granted": False,
        "self_preservation_authority_granted": False,
        "runtime_authority_granted": False,
    }


def qualify_prepared(capsule: dict[str, Any], plan: dict[str, Any]) -> None:
    if capsule["subject_git_head"] != plan["subject_git_head"]:
        capsule["classification"] = "INVALID_CAPSULE"
        capsule["environment_integrity"] = "INVALID"
        return
    if not capsule["worktree_clean"]:
        capsule["classification"] = "INVALID_CAPSULE"
        capsule["environment_integrity"] = "INVALID"
        return
    missing = [item["path"] for item in capsule["materials"] if item["required"] and not item["present"]]
    if missing:
        capsule["classification"] = "INFRASTRUCTURE_INDETERMINATE"
        capsule["environment_integrity"] = "INDETERMINATE"


def execute_stages(root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    records = command_skeleton(plan)
    stop = False
    for index, stage in enumerate(plan["stages"]):
        if stop:
            continue
        record = records[index]
        cwd = (root / stage["cwd"]).resolve()
        try:
            cwd.relative_to(root)
        except ValueError:
            record["termination_class"] = "InfrastructureFailure"
            record["subject_outcome"] = "INDETERMINATE"
            stop = True
            continue
        if not cwd.is_dir():
            record["termination_class"] = "InfrastructureFailure"
            record["subject_outcome"] = "INDETERMINATE"
            stop = True
            continue
        record["started_utc"] = utc_now()
        try:
            process = subprocess.run(
                stage["argv"],
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                timeout=stage["timeout_seconds"],
                check=False,
            )
            record["finished_utc"] = utc_now()
            record["exit_code"] = process.returncode
            record["termination_class"] = "Exited" if process.returncode >= 0 else "Signaled"
            record["stdout_sha256"] = sha256_bytes(process.stdout)
            record["stderr_sha256"] = sha256_bytes(process.stderr)
            record["subject_outcome"] = "PASS" if process.returncode == 0 else "FAIL"
            output = stage["output_receipt_path"]
            if output is not None:
                output_path = (root / output).resolve()
                try:
                    output_path.relative_to(root)
                except ValueError:
                    record["subject_outcome"] = "INVALID"
                    stop = True
                else:
                    if output_path.is_file():
                        record["output_receipt_sha256"] = sha256_file(output_path)
                    elif process.returncode == 0:
                        record["subject_outcome"] = "INVALID"
                        stop = True
            if process.returncode != 0:
                stop = True
        except subprocess.TimeoutExpired as exc:
            record["finished_utc"] = utc_now()
            record["termination_class"] = "TimedOut"
            record["stdout_sha256"] = sha256_bytes(exc.stdout or b"")
            record["stderr_sha256"] = sha256_bytes(exc.stderr or b"")
            record["subject_outcome"] = "INDETERMINATE"
            stop = True
        except OSError:
            record["finished_utc"] = utc_now()
            record["termination_class"] = "InfrastructureFailure"
            record["subject_outcome"] = "INDETERMINATE"
            stop = True
    return records


def aggregate_outcome(commands: list[dict[str, Any]]) -> str:
    outcomes = [entry["subject_outcome"] for entry in commands]
    if "INVALID" in outcomes:
        return "INVALID"
    if "INDETERMINATE" in outcomes:
        return "INDETERMINATE"
    if "FAIL" in outcomes:
        return "FAIL"
    if outcomes and all(value == "PASS" for value in outcomes):
        return "PASS"
    return "NOT_RUN"


def prepare(plan_path: Path) -> tuple[dict[str, Any], dict[str, Any], bytes, Path]:
    root = repo_root()
    raw, plan = read_json(plan_path)
    validate_plan(plan)
    capsule = capture_base(root, raw, plan)
    qualify_prepared(capsule, plan)
    return capsule, plan, raw, root


def run_plan(plan_path: Path) -> dict[str, Any]:
    prepared, plan, raw, root = prepare(plan_path)
    prepared_bytes = canonical_json_bytes(prepared)
    prepared_sha = sha256_bytes(prepared_bytes)
    if prepared["classification"] != "CAPSULE_PREPARED":
        return prepared

    command_results = execute_stages(root, plan)
    try:
        final = capture_base(root, raw, plan)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        final = dict(prepared)
        final["capsule_phase"] = "FINAL"
        final["classification"] = "INFRASTRUCTURE_INDETERMINATE"
        final["environment_integrity"] = "INDETERMINATE"
        final["subject_outcome"] = aggregate_outcome(command_results)
        final["commands"] = command_results
        final["prepared_capsule_sha256"] = prepared_sha
        final["drift_fields"] = [f"final_capture_failed:{type(exc).__name__}"]
        final["created_utc"] = utc_now()
        return final

    final["capsule_phase"] = "FINAL"
    final["commands"] = command_results
    final["subject_outcome"] = aggregate_outcome(command_results)
    final["prepared_capsule_sha256"] = prepared_sha
    drift = compare_immutable(prepared, final)
    final["drift_fields"] = drift
    if drift:
        final["classification"] = "ENVIRONMENT_DRIFT"
        final["environment_integrity"] = "DRIFTED"
    else:
        final["classification"] = "QUALIFIED_EXECUTION"
        final["environment_integrity"] = "QUALIFIED"
    final["created_utc"] = utc_now()
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_prepare = sub.add_parser("prepare")
    p_prepare.add_argument("plan", type=Path)
    p_run = sub.add_parser("run")
    p_run.add_argument("plan", type=Path)
    p_compare = sub.add_parser("compare")
    p_compare.add_argument("prepared", type=Path)
    p_compare.add_argument("final", type=Path)
    args = parser.parse_args()

    try:
        if args.command == "prepare":
            capsule, _plan, _raw, _root = prepare(args.plan)
            payload = capsule
        elif args.command == "run":
            payload = run_plan(args.plan)
        else:
            _pr, prepared = read_json(args.prepared)
            _fr, final = read_json(args.final)
            drift = compare_immutable(prepared, final)
            payload = {
                "authority": "MeasurementOnly",
                "protocol_version": PROTOCOL,
                "classification": "ENVIRONMENT_DRIFT" if drift else "QUALIFIED_EXECUTION",
                "drift_fields": drift,
                "runtime_authority_granted": False,
            }
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        payload = {
            "authority": "MeasurementOnly",
            "protocol_version": PROTOCOL,
            "classification": "INVALID_CAPSULE" if isinstance(exc, ValueError) else "INFRASTRUCTURE_INDETERMINATE",
            "detail": str(exc),
            "runtime_authority_granted": False,
        }
        sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        return 4 if payload["classification"] == "INVALID_CAPSULE" else 3

    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    classification = payload.get("classification")
    if classification in {"CAPSULE_PREPARED", "QUALIFIED_EXECUTION"}:
        return 0
    if classification == "ENVIRONMENT_DRIFT":
        return 2
    if classification == "INFRASTRUCTURE_INDETERMINATE":
        return 3
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
