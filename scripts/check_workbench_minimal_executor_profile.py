#!/usr/bin/env python3
"""Independent consistency checker for the minimal executor vs qualified #681."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import workbench_minimal_isolated_executor as executor

QUALIFIED_PROFILE_SHA256 = "sha256:bc446fa7613c7197585bfb6fe7c250d4c6fd583eca0434c64bec8e722a62f6b4"
QUALIFIED_PROFILE_PR = 681
QUALIFIED_PROFILE_HEAD = "ca34837f957d178a48b8229c6a258059c2596492"
QUALIFIED_PROFILE_RUN = 34225954077
PROFILE_SCHEMA = "symthaea-workbench-invocation-isolation-profile-v1"


class ProfileConsistencyError(ValueError):
    pass


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ProfileConsistencyError(f"profile: duplicate key {key}")
        out[key] = value
    return out


def exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ProfileConsistencyError(f"{label}: closed-world schema mismatch")
    return value


def strict_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ProfileConsistencyError(f"{label}: boolean required")
    return value


def load_qualified_profile(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if digest_bytes(raw) != QUALIFIED_PROFILE_SHA256:
        raise ProfileConsistencyError("profile: qualified file root mismatch")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProfileConsistencyError("profile: valid UTF-8 JSON required") from exc
    if not isinstance(value, dict):
        raise ProfileConsistencyError("profile: object required")
    return value


def verify_consistency(profile: dict[str, Any]) -> dict[str, Any]:
    top = exact(
        profile,
        {
            "schema", "status", "parent_execution_capsule_profile", "qualification_platform",
            "program_binding", "process_environment", "per_invocation_isolation",
            "scientific_path_gate", "numerical_execution_gate", "authority",
        },
        "profile",
    )
    if top["schema"] != PROFILE_SCHEMA or top["status"] != "candidate_profile_only":
        raise ProfileConsistencyError("profile: schema/status mismatch")
    if top["qualification_platform"] != "x86_64-linux":
        raise ProfileConsistencyError("profile: x86_64-linux required")

    program = exact(
        top["program_binding"],
        {"root_source", "relative_main_program", "require_absolute_exec_path", "host_path_lookup_allowed"},
        "program binding",
    )
    if program != {
        "root_source": "independently-verified-closure-root",
        "relative_main_program": "bin/wb_command",
        "require_absolute_exec_path": True,
        "host_path_lookup_allowed": False,
    }:
        raise ProfileConsistencyError("program binding: exact qualified contract required")

    process = exact(
        top["process_environment"],
        {"stage", "inherit_host_environment", "fixed", "dynamic_bindings", "descendant_environment"},
        "process environment",
    )
    if process["stage"] != "root-main-program-execve":
        raise ProfileConsistencyError("process environment: execve stage mismatch")
    if strict_bool(process["inherit_host_environment"], "inherit host environment") is not False:
        raise ProfileConsistencyError("process environment: host inheritance forbidden")
    if process["fixed"] != dict(executor.FIXED_ENTRY_ENVIRONMENT):
        raise ProfileConsistencyError("process environment: executor fixed entries differ from profile")

    expected_dynamic = {
        variable: f"invocation.{role}"
        for variable, role in executor.DYNAMIC_ENTRY_BINDINGS.items()
    }
    if process["dynamic_bindings"] != expected_dynamic:
        raise ProfileConsistencyError("process environment: executor dynamic bindings differ from profile")

    descendant = exact(
        process["descendant_environment"],
        {
            "entry_environment_is_final_environment",
            "verified_program_may_transform_environment",
            "host_injection_after_execve_allowed",
            "post_entry_environment_equivalence_assumed",
        },
        "descendant environment",
    )
    if descendant != {
        "entry_environment_is_final_environment": False,
        "verified_program_may_transform_environment": True,
        "host_injection_after_execve_allowed": False,
        "post_entry_environment_equivalence_assumed": False,
    }:
        raise ProfileConsistencyError("descendant environment: boundary mismatch")

    isolation = top["per_invocation_isolation"]
    expected_policies = {
        "cwd": "cwd_policy",
        "home": "home_policy",
        "xdg_config_home": "xdg_config_home_policy",
        "xdg_cache_home": "xdg_cache_home_policy",
        "xdg_data_home": "xdg_data_home_policy",
        "xdg_state_home": "xdg_state_home_policy",
        "xdg_runtime_dir": "xdg_runtime_dir_policy",
        "xdg_config_dirs": "xdg_config_dirs_policy",
        "xdg_data_dirs": "xdg_data_dirs_policy",
        "tmpdir": "tmpdir_policy",
    }
    for role in executor.ROOT_ROLES:
        key = expected_policies[role]
        if isolation.get(key) != "fresh-private-empty-per-invocation":
            raise ProfileConsistencyError(f"isolation: root policy mismatch: {role}")
    expected_scalar_isolation = {
        "dynamic_environment_must_match_created_roots": True,
        "dynamic_environment_paths_must_be_absolute": True,
        "directory_mode_octal": "0700",
        "umask_octal": "0077",
        "reuse_allowed": False,
        "stale_destination_fails_closed": True,
        "cleanup_must_be_confirmed": True,
        "stdin_policy": "devnull",
        "stdout_policy": "retain-exact-bytes",
        "stderr_policy": "retain-exact-bytes",
        "close_fds": True,
        "pass_fds": [],
    }
    for key, expected in expected_scalar_isolation.items():
        if isolation.get(key) != expected or type(isolation.get(key)) is not type(expected):
            raise ProfileConsistencyError(f"isolation: scalar contract mismatch: {key}")
    if executor.DIRECTORY_MODE != 0o700 or executor.PROCESS_UMASK != 0o077:
        raise ProfileConsistencyError("executor: mode/umask constants differ from profile")

    path_gate = top["scientific_path_gate"]
    for key in (
        "snapshot_root_path_is_scientific_identity",
        "scratch_root_path_is_scientific_identity",
        "working_directory_path_is_scientific_identity",
        "bytewise_equivalence_assumed",
        "semantic_equivalence_assumed",
    ):
        if strict_bool(path_gate.get(key), f"scientific path gate {key}") is not False:
            raise ProfileConsistencyError(f"scientific path gate: premature authority: {key}")
    if path_gate.get("required_perturbation_axes") != [
        "snapshot_root", "scratch_root", "working_directory"
    ]:
        raise ProfileConsistencyError("scientific path gate: perturbation axes mismatch")

    numerical = top["numerical_execution_gate"]
    if numerical.get("same_x86_64_implies_equivalence") is not False:
        raise ProfileConsistencyError("numerical gate: x86_64 equivalence must remain false")
    if numerical.get("same_closure_implies_cross_cpu_equivalence") is not False:
        raise ProfileConsistencyError("numerical gate: closure/cross-CPU equivalence must remain false")
    if numerical.get("same_host_repeatability_required") is not True:
        raise ProfileConsistencyError("numerical gate: same-host repeatability must remain required")
    if numerical.get("cross_cpu_equivalence_required_before_transfer") is not True:
        raise ProfileConsistencyError("numerical gate: cross-CPU transfer gate must remain required")

    authority = top["authority"]
    if authority.get("invocation_contract_defined") is not True:
        raise ProfileConsistencyError("authority: invocation contract must remain defined")
    for key, value in authority.items():
        if key == "invocation_contract_defined":
            continue
        if strict_bool(value, f"authority {key}") is not False:
            raise ProfileConsistencyError(f"authority: premature promotion: {key}")

    return {
        "schema": "symthaea-workbench-minimal-executor-profile-consistency-v1",
        "qualified_profile_pr": QUALIFIED_PROFILE_PR,
        "qualified_profile_head": QUALIFIED_PROFILE_HEAD,
        "qualified_profile_run": QUALIFIED_PROFILE_RUN,
        "qualified_profile_sha256": QUALIFIED_PROFILE_SHA256,
        "consistent": True,
        "authority": {
            "invocation_contract_defined": True,
            "executor_contract_consistent": True,
            "real_workbench_execution_performed": False,
            "scientific_execution_qualified": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify_consistency(load_qualified_profile(args.profile))
    except (ProfileConsistencyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
