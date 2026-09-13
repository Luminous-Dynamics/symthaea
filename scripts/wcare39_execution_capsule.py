#!/usr/bin/env python3
"""WCARE-39 execution capsule capture and runner.

Dependency-free qualification tooling. It records exact source/tool/material state,
persists PREPARED before launching evidence-producing commands, executes exact
argv arrays without a shell, and refuses to mix evidence across environment drift.
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
SAFE_LITERAL_DENY_TOKENS = (
    "TOKEN", "SECRET", "PASSWORD", "PASSWD", "COOKIE", "AUTH", "API_KEY", "PRIVATE_KEY",
)
KNOWN_VERSION_TOOLS = {
    "git", "cargo", "rustc", "rustup", "nix", "nix-shell", "bash", "sh",
}


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


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *args], cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=check, timeout=30,
    )


def repo_root() -> Path:
    process = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False, timeout=30,
    )
    if process.returncode != 0:
        raise RuntimeError("not_in_git_worktree")
    return Path(process.stdout.decode().strip()).resolve()


def root_relative(root: Path, relative: str) -> Path:
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"unsafe_relative_path:{relative}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path_escapes_repository:{relative}") from exc
    return resolved


def read_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("json_root_not_object")
    return raw, value


def valid_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def validate_plan(plan: dict[str, Any]) -> None:
    required = {
        "protocol_version", "subject_git_head", "subject_digests", "materials",
        "safe_environment", "network_policy_declared", "sandbox_policy_declared",
        "deterministic_seed_commitments", "stages",
    }
    if set(plan) - (required | {"notes"}) or required - set(plan):
        raise ValueError("plan_field_set_invalid")
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
        if not valid_sha(digest):
            raise ValueError(f"plan_subject_digest_invalid:{path}")

    if not isinstance(plan["materials"], list):
        raise ValueError("plan_materials_not_array")
    material_paths: set[str] = set()
    for item in plan["materials"]:
        if not isinstance(item, dict) or set(item) != {"path", "required"}:
            raise ValueError("plan_material_entry_invalid")
        path = item["path"]
        if not isinstance(path, str) or not path or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ValueError("plan_material_path_invalid")
        if path in material_paths:
            raise ValueError("plan_duplicate_material")
        material_paths.add(path)
        if not isinstance(item["required"], bool):
            raise ValueError("plan_material_required_not_boolean")

    if not isinstance(plan["safe_environment"], list):
        raise ValueError("plan_safe_environment_not_array")
    env_keys: set[str] = set()
    for item in plan["safe_environment"]:
        if not isinstance(item, dict) or set(item) != {"key", "mode"}:
            raise ValueError("plan_safe_environment_entry_invalid")
        key, mode = item["key"], item["mode"]
        if not isinstance(key, str) or not key or key.upper() != key or not key.replace("_", "A").isalnum():
            raise ValueError("plan_safe_environment_key_invalid")
        if key in env_keys:
            raise ValueError("plan_duplicate_environment_key")
        env_keys.add(key)
        if mode not in {"Literal", "Sha256"}:
            raise ValueError("plan_safe_environment_mode_invalid")
        if mode == "Literal" and any(token in key for token in SAFE_LITERAL_DENY_TOKENS):
            raise ValueError(f"sensitive_environment_literal_forbidden:{key}")

    if plan["network_policy_declared"] not in {"Unspecified", "Allow", "Deny", "Restricted"}:
        raise ValueError("plan_network_policy_invalid")
    if plan["sandbox_policy_declared"] not in {"Unspecified", "None", "Restricted", "DefaultDeny"}:
        raise ValueError("plan_sandbox_policy_invalid")

    seeds = plan["deterministic_seed_commitments"]
    if not isinstance(seeds, dict) or any(not isinstance(k, str) or not k or not valid_sha(v) for k, v in seeds.items()):
        raise ValueError("plan_seed_commitments_invalid")

    stages = plan["stages"]
    if not isinstance(stages, list) or not stages:
        raise ValueError("plan_stages_empty")
    stage_ids: set[str] = set()
    output_paths: set[str] = set()
    for stage in stages:
        expected = {"stage_id", "argv", "cwd", "timeout_seconds", "output_receipt_path"}
        if not isinstance(stage, dict) or set(stage) != expected:
            raise ValueError("plan_stage_fields_invalid")
        stage_id = stage["stage_id"]
        if not isinstance(stage_id, str) or not stage_id or stage_id in stage_ids:
            raise ValueError("plan_stage_id_invalid_or_duplicate")
        stage_ids.add(stage_id)
        argv = stage["argv"]
        if not isinstance(argv, list) or not argv or any(not isinstance(x, str) or "\x00" in x for x in argv):
            raise ValueError("plan_stage_argv_invalid")
        cwd = stage["cwd"]
        if not isinstance(cwd, str) or Path(cwd).is_absolute() or ".." in Path(cwd).parts:
            raise ValueError("plan_stage_cwd_invalid")
        timeout = stage["timeout_seconds"]
        if not isinstance(timeout, int) or isinstance(timeout, bool) or not 1 <= timeout <= 86400:
            raise ValueError("plan_stage_timeout_invalid")
        output = stage["output_receipt_path"]
        if output is not None:
            if not isinstance(output, str) or not output or Path(output).is_absolute() or ".." in Path(output).parts:
                raise ValueError("plan_stage_output_path_invalid")
            if output in output_paths:
                raise ValueError("plan_duplicate_output_receipt_path")
            output_paths.add(output)


def validate_file_bindings(root: Path, plan: dict[str, Any]) -> None:
    bound = set(plan["subject_digests"]) | {item["path"] for item in plan["materials"]}
    outputs = {item["output_receipt_path"] for item in plan["stages"] if item["output_receipt_path"] is not None}
    for stage in plan["stages"]:
        for arg in stage["argv"]:
            if arg in outputs or arg.startswith("-") or Path(arg).is_absolute() or ".." in Path(arg).parts:
                continue
            candidate = root / arg
            if candidate.is_file() and candidate.suffix.lower() in {".py", ".sh", ".rs", ".json", ".toml", ".lock", ".nix"}:
                if arg not in bound:
                    raise ValueError(f"unbound_command_file:{arg}")


def auto_materials(plan: dict[str, Any]) -> dict[str, bool]:
    materials = dict(STANDARD_MATERIALS)
    for item in plan["materials"]:
        materials[item["path"]] = materials.get(item["path"], False) or item["required"]
    invokes_auth = any(
        "wcare37" in " ".join(stage["argv"]).lower() or "wcare38" in " ".join(stage["argv"]).lower()
        for stage in plan["stages"]
    )
    if invokes_auth:
        materials["tools/wcare37_attestation_verifier/Cargo.lock"] = True
    return materials


def capture_materials(root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for relative, required in sorted(auto_materials(plan).items()):
        path = root_relative(root, relative)
        present = path.is_file()
        result.append({"path": relative, "required": required, "present": present, "sha256": sha256_file(path) if present else None})
    return result


def capture_subject_digests(root: Path, plan: dict[str, Any]) -> dict[str, str]:
    observed = {}
    for relative, expected in sorted(plan["subject_digests"].items()):
        path = root_relative(root, relative)
        if not path.is_file():
            raise RuntimeError(f"subject_material_missing:{relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"subject_digest_mismatch:{relative}:{actual}")
        observed[relative] = actual
    return observed


def safe_environment(plan: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for item in sorted(plan["safe_environment"], key=lambda x: x["key"]):
        key = item["key"]
        if key not in os.environ:
            result.append({"key": key, "mode": "Absent", "value": None})
            continue
        value = os.environ[key]
        if item["mode"] == "Literal":
            if len(value) > 2048 or any(ord(ch) < 32 and ch not in "\t" for ch in value):
                raise ValueError(f"unsafe_literal_environment_value:{key}")
            result.append({"key": key, "mode": "Literal", "value": value})
        else:
            result.append({"key": key, "mode": "Sha256", "value": sha256_bytes(value.encode())})
    return result


def ambient_environment_sha256() -> str:
    # A single aggregate commitment detects hidden environment drift without
    # publishing arbitrary key/value pairs or secret-bearing variable names.
    payload = b"\0".join(
        f"{key}={value}".encode(errors="surrogateescape")
        for key, value in sorted(os.environ.items())
    )
    return sha256_bytes(payload)


def privacy_safe_locator(path: Path, root: Path) -> str:
    try:
        return "repo:" + str(path.relative_to(root))
    except ValueError:
        raw = str(path)
        if raw.startswith("/nix/store/"):
            return raw
        return f"basename:{path.name};path_sha256:{sha256_bytes(raw.encode())}"


def version_output(path: Path) -> str:
    base = path.name.lower()
    safe = base in KNOWN_VERSION_TOOLS or base.startswith("python")
    if not safe:
        return "<not-invoked-unrecognized-tool>"
    try:
        proc = subprocess.run([str(path), "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False, timeout=15)
        return proc.stdout.decode(errors="replace")[:4096].strip()
    except (OSError, subprocess.TimeoutExpired):
        return "<version-unavailable>"


def resolve_executable(command: str, root: Path) -> Path:
    candidate = Path(command)
    if candidate.is_absolute() or "/" in command:
        path = candidate if candidate.is_absolute() else root_relative(root, command)
    else:
        found = shutil.which(command)
        if found is None:
            raise RuntimeError(f"tool_not_found:{command}")
        path = Path(found)
    resolved = path.resolve()
    if not resolved.is_file():
        raise RuntimeError(f"tool_not_file:{command}")
    return resolved


def capture_tools(root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    commands = {"git", sys.executable}
    commands.update(stage["argv"][0] for stage in plan["stages"])
    records = []
    for command in sorted(commands):
        path = resolve_executable(command, root)
        version = version_output(path)
        records.append({
            "role": command,
            "executable_path": privacy_safe_locator(path, root),
            "executable_sha256": sha256_file(path),
            "version_output_sha256": sha256_bytes(version.encode()),
            "version_output": version,
        })
    return records


def capture_platform() -> dict[str, str]:
    try:
        current_locale = locale.setlocale(locale.LC_ALL, None) or ""
    except locale.Error:
        current_locale = "<locale-unavailable>"
    tz = json.dumps({"tzname": list(time.tzname), "timezone": time.timezone, "daylight": time.daylight}, sort_keys=True, separators=(",", ":"))
    return {
        "os": platform.system(), "kernel_release": platform.release(), "architecture": platform.machine(),
        "python_implementation": platform.python_implementation(), "python_version": platform.python_version(),
        "locale": current_locale, "timezone": tz,
    }


def command_skeleton(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [{
        "stage_id": stage["stage_id"], "argv": stage["argv"], "cwd": stage["cwd"],
        "timeout_seconds": stage["timeout_seconds"], "started_utc": None, "finished_utc": None,
        "exit_code": None, "termination_class": "NotRun", "stdout_sha256": None,
        "stderr_sha256": None, "output_receipt_sha256": None, "subject_outcome": "NOT_RUN",
    } for stage in plan["stages"]]


def immutable_view(capsule: dict[str, Any]) -> dict[str, Any]:
    return {
        "subject_git_head": capsule["subject_git_head"], "worktree_clean": capsule["worktree_clean"],
        "repository_root_commitment_sha256": capsule["repository_root_commitment_sha256"],
        "subject_digests": capsule["subject_digests"], "command_plan_sha256": capsule["command_plan_sha256"],
        "platform": capsule["platform"], "tools": capsule["tools"], "materials": capsule["materials"],
        "safe_environment": capsule["safe_environment"], "ambient_environment_sha256": capsule["ambient_environment_sha256"],
        "network_policy_declared": capsule["network_policy_declared"], "sandbox_policy_declared": capsule["sandbox_policy_declared"],
        "deterministic_seed_commitments": capsule["deterministic_seed_commitments"],
        "commands": [{"stage_id": x["stage_id"], "argv": x["argv"], "cwd": x["cwd"], "timeout_seconds": x["timeout_seconds"]} for x in capsule["commands"]],
    }


def compare_immutable(prepared: dict[str, Any], final: dict[str, Any]) -> list[str]:
    left, right = immutable_view(prepared), immutable_view(final)
    return sorted(key for key in left if left[key] != right[key])


def capture_base(root: Path, plan_raw: bytes, plan: dict[str, Any]) -> dict[str, Any]:
    head = git(root, "rev-parse", "HEAD").stdout.decode().strip()
    clean = not git(root, "status", "--porcelain=v1", "--untracked-files=normal").stdout.strip()
    origin = git(root, "remote", "get-url", "origin", check=False)
    repo_identity = origin.stdout.strip() if origin.returncode == 0 and origin.stdout.strip() else root.name.encode()
    return {
        "protocol_version": PROTOCOL, "authority": "MeasurementOnly", "capsule_phase": "PREPARED",
        "classification": "CAPSULE_PREPARED", "environment_integrity": "QUALIFIED", "subject_outcome": "NOT_RUN",
        "subject_git_head": head, "worktree_clean": clean, "repository_root_commitment_sha256": sha256_bytes(repo_identity),
        "subject_digests": capture_subject_digests(root, plan), "command_plan_sha256": sha256_bytes(plan_raw),
        "platform": capture_platform(), "tools": capture_tools(root, plan), "materials": capture_materials(root, plan),
        "safe_environment": safe_environment(plan), "ambient_environment_sha256": ambient_environment_sha256(),
        "network_policy_declared": plan["network_policy_declared"], "sandbox_policy_declared": plan["sandbox_policy_declared"],
        "deterministic_seed_commitments": dict(sorted(plan["deterministic_seed_commitments"].items())),
        "commands": command_skeleton(plan), "prepared_capsule_sha256": None, "drift_fields": [], "created_utc": utc_now(),
        "network_isolation_established": False, "sandbox_enforcement_established": False,
        "independent_builder_established": False, "phenomenal_experience_established": False,
        "suffering_established": False, "moral_patienthood_established": False, "binding_consent_established": False,
        "veto_authority_granted": False, "self_preservation_authority_granted": False, "runtime_authority_granted": False,
    }


def qualify_prepared(capsule: dict[str, Any], plan: dict[str, Any]) -> None:
    if capsule["subject_git_head"] != plan["subject_git_head"] or not capsule["worktree_clean"]:
        capsule["classification"] = "INVALID_CAPSULE"
        capsule["environment_integrity"] = "INVALID"
        return
    if any(x["required"] and not x["present"] for x in capsule["materials"]):
        capsule["classification"] = "INFRASTRUCTURE_INDETERMINATE"
        capsule["environment_integrity"] = "INDETERMINATE"


def prepare(plan_path: Path) -> tuple[dict[str, Any], dict[str, Any], bytes, Path]:
    root = repo_root()
    raw, plan = read_json(plan_path)
    validate_plan(plan)
    validate_file_bindings(root, plan)
    capsule = capture_base(root, raw, plan)
    qualify_prepared(capsule, plan)
    return capsule, plan, raw, root


def ensure_ignored_evidence_path(root: Path, relative: str) -> Path:
    path = root_relative(root, relative)
    probe = str(Path(relative) / ".wcare39-probe")
    result = git(root, "check-ignore", "--no-index", "-q", probe, check=False)
    if result.returncode != 0:
        raise ValueError("evidence_directory_must_be_git_ignored")
    return path


def execute_stages(root: Path, plan: dict[str, Any]) -> list[dict[str, Any]]:
    records = command_skeleton(plan)
    stop = False
    for index, stage in enumerate(plan["stages"]):
        if stop:
            continue
        record = records[index]
        cwd = root_relative(root, stage["cwd"])
        if not cwd.is_dir():
            record["termination_class"] = "InfrastructureFailure"
            record["subject_outcome"] = "INDETERMINATE"
            stop = True
            continue
        output_relative = stage["output_receipt_path"]
        output_path = root_relative(root, output_relative) if output_relative is not None else None
        if output_path is not None and output_path.exists():
            record["termination_class"] = "InfrastructureFailure"
            record["subject_outcome"] = "INVALID"
            stop = True
            continue
        record["started_utc"] = utc_now()
        try:
            process = subprocess.run(
                stage["argv"], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                shell=False, timeout=stage["timeout_seconds"], check=False,
            )
            record["finished_utc"] = utc_now()
            record["exit_code"] = process.returncode
            record["termination_class"] = "Exited" if process.returncode >= 0 else "Signaled"
            record["stdout_sha256"] = sha256_bytes(process.stdout)
            record["stderr_sha256"] = sha256_bytes(process.stderr)
            record["subject_outcome"] = "PASS" if process.returncode == 0 else "FAIL"
            if output_path is not None:
                if output_path.is_file():
                    record["output_receipt_sha256"] = sha256_file(output_path)
                elif process.returncode == 0:
                    record["subject_outcome"] = "INVALID"
            if record["subject_outcome"] != "PASS":
                stop = True
        except subprocess.TimeoutExpired as exc:
            record["finished_utc"] = utc_now(); record["termination_class"] = "TimedOut"
            record["stdout_sha256"] = sha256_bytes(exc.stdout or b""); record["stderr_sha256"] = sha256_bytes(exc.stderr or b"")
            record["subject_outcome"] = "INDETERMINATE"; stop = True
        except OSError:
            record["finished_utc"] = utc_now(); record["termination_class"] = "InfrastructureFailure"
            record["subject_outcome"] = "INDETERMINATE"; stop = True
    return records


def aggregate_outcome(commands: list[dict[str, Any]]) -> str:
    outcomes = [x["subject_outcome"] for x in commands]
    for value in ("INVALID", "INDETERMINATE", "FAIL"):
        if value in outcomes:
            return value
    return "PASS" if outcomes and all(x == "PASS" for x in outcomes) else "NOT_RUN"


def run_plan(plan_path: Path, evidence_dir_relative: str) -> dict[str, Any]:
    prepared, plan, raw, root = prepare(plan_path)
    if prepared["classification"] != "CAPSULE_PREPARED":
        return prepared
    evidence_dir = ensure_ignored_evidence_path(root, evidence_dir_relative)
    prepared_path = evidence_dir / "prepared.json"
    final_path = evidence_dir / "final.json"
    if prepared_path.exists() or final_path.exists():
        raise ValueError("execution_capsule_output_already_exists")
    prepared_bytes = canonical_json_bytes(prepared)
    atomic_write(prepared_path, prepared_bytes)  # durable before first command launch
    prepared_sha = sha256_bytes(prepared_bytes)

    commands = execute_stages(root, plan)
    try:
        final = capture_base(root, raw, plan)
        final["capsule_phase"] = "FINAL"
        final["commands"] = commands
        final["subject_outcome"] = aggregate_outcome(commands)
        final["prepared_capsule_sha256"] = prepared_sha
        drift = compare_immutable(prepared, final)
        final["drift_fields"] = drift
        final["classification"] = "ENVIRONMENT_DRIFT" if drift else "QUALIFIED_EXECUTION"
        final["environment_integrity"] = "DRIFTED" if drift else "QUALIFIED"
        final["created_utc"] = utc_now()
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        final = dict(prepared)
        final.update({
            "capsule_phase": "FINAL", "classification": "INFRASTRUCTURE_INDETERMINATE",
            "environment_integrity": "INDETERMINATE", "subject_outcome": aggregate_outcome(commands),
            "commands": commands, "prepared_capsule_sha256": prepared_sha,
            "drift_fields": [f"final_capture_failed:{type(exc).__name__}"], "created_utc": utc_now(),
        })
    atomic_write(final_path, canonical_json_bytes(final))
    return final


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_prepare = sub.add_parser("prepare"); p_prepare.add_argument("plan", type=Path)
    p_run = sub.add_parser("run"); p_run.add_argument("plan", type=Path); p_run.add_argument("evidence_dir")
    p_compare = sub.add_parser("compare"); p_compare.add_argument("prepared", type=Path); p_compare.add_argument("final", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            payload, _plan, _raw, _root = prepare(args.plan)
        elif args.command == "run":
            payload = run_plan(args.plan, args.evidence_dir)
        else:
            _a, prepared = read_json(args.prepared); _b, final = read_json(args.final)
            drift = compare_immutable(prepared, final)
            payload = {"authority": "MeasurementOnly", "protocol_version": PROTOCOL,
                       "classification": "ENVIRONMENT_DRIFT" if drift else "QUALIFIED_EXECUTION",
                       "drift_fields": drift, "runtime_authority_granted": False}
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        classification = "INVALID_CAPSULE" if isinstance(exc, ValueError) else "INFRASTRUCTURE_INDETERMINATE"
        payload = {"authority": "MeasurementOnly", "protocol_version": PROTOCOL,
                   "classification": classification, "detail": str(exc), "runtime_authority_granted": False}
        sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        return 4 if classification == "INVALID_CAPSULE" else 3
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return {"CAPSULE_PREPARED": 0, "QUALIFIED_EXECUTION": 0, "ENVIRONMENT_DRIFT": 2,
            "INFRASTRUCTURE_INDETERMINATE": 3}.get(payload.get("classification"), 4)


if __name__ == "__main__":
    raise SystemExit(main())
