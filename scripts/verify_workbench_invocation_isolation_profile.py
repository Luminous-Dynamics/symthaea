#!/usr/bin/env python3
"""Closed-world verifier for Workbench invocation isolation profile v1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "symthaea-workbench-invocation-isolation-profile-v1"
PARENT_SCHEMA = "symthaea-workbench-execution-capsule-profile-v1"

TOP_KEYS = {
    "schema", "status", "parent_execution_capsule_profile", "qualification_platform",
    "program_binding", "process_environment", "per_invocation_isolation",
    "scientific_path_gate", "numerical_execution_gate", "authority",
}
PROGRAM_KEYS = {"root_source", "relative_main_program", "require_absolute_exec_path", "host_path_lookup_allowed"}
ENV_KEYS = {"stage", "inherit_host_environment", "fixed", "dynamic_bindings", "descendant_environment"}
FIXED_ENV_KEYS = {"LANG", "LC_ALL", "TZ", "OMP_NUM_THREADS", "OMP_DYNAMIC", "PATH"}
DYNAMIC_ENV_KEYS = {
    "PWD", "HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME",
    "XDG_RUNTIME_DIR", "XDG_CONFIG_DIRS", "XDG_DATA_DIRS", "TMPDIR", "TMP", "TEMP",
}
DESCENDANT_ENV_KEYS = {
    "entry_environment_is_final_environment", "verified_program_may_transform_environment",
    "host_injection_after_execve_allowed", "post_entry_environment_equivalence_assumed",
}
ISOLATION_KEYS = {
    "cwd_policy", "home_policy", "xdg_config_home_policy", "xdg_cache_home_policy",
    "xdg_data_home_policy", "xdg_state_home_policy", "xdg_runtime_dir_policy",
    "xdg_config_dirs_policy", "xdg_data_dirs_policy", "tmpdir_policy",
    "dynamic_environment_must_match_created_roots", "dynamic_environment_paths_must_be_absolute",
    "directory_mode_octal", "umask_octal", "reuse_allowed", "stale_destination_fails_closed",
    "cleanup_must_be_confirmed", "stdin_policy", "stdout_policy", "stderr_policy", "close_fds", "pass_fds",
}
PATH_GATE_KEYS = {
    "snapshot_root_path_is_scientific_identity", "scratch_root_path_is_scientific_identity",
    "working_directory_path_is_scientific_identity", "bytewise_equivalence_assumed",
    "semantic_equivalence_assumed", "required_perturbation_axes",
}
NUMERIC_GATE_KEYS = {
    "same_x86_64_implies_equivalence", "same_closure_implies_cross_cpu_equivalence",
    "runtime_cpu_dispatch_known", "known_dispatch_modes", "same_host_repeatability_required",
    "cross_cpu_equivalence_required_before_transfer", "diagnostic_context",
}
AUTHORITY_KEYS = {
    "invocation_contract_defined", "invocation_executed", "path_equivalence_established",
    "same_host_repeatability_established", "cross_cpu_equivalence_established",
    "workbench_execution_qualified", "transform_executed", "atlas_correctness_established",
    "fmq010_established", "neural_alignment_established", "consciousness_evidence",
}

EXPECTED_FIXED_ENV = {
    "LANG": "C",
    "LC_ALL": "C",
    "TZ": "UTC0",
    "OMP_NUM_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "PATH": "",
}
EXPECTED_DYNAMIC_ENV = {
    "PWD": "invocation.cwd",
    "HOME": "invocation.home",
    "XDG_CONFIG_HOME": "invocation.xdg_config_home",
    "XDG_CACHE_HOME": "invocation.xdg_cache_home",
    "XDG_DATA_HOME": "invocation.xdg_data_home",
    "XDG_STATE_HOME": "invocation.xdg_state_home",
    "XDG_RUNTIME_DIR": "invocation.xdg_runtime_dir",
    "XDG_CONFIG_DIRS": "invocation.xdg_config_dirs",
    "XDG_DATA_DIRS": "invocation.xdg_data_dirs",
    "TMPDIR": "invocation.tmpdir",
    "TMP": "invocation.tmpdir",
    "TEMP": "invocation.tmpdir",
}
EXPECTED_DESCENDANT_ENV = {
    "entry_environment_is_final_environment": False,
    "verified_program_may_transform_environment": True,
    "host_injection_after_execve_allowed": False,
    "post_entry_environment_equivalence_assumed": False,
}
EXPECTED_DISPATCH_MODES = ["AVX512FMA", "AVX512", "AVX", "SSE2", "NAIVE"]
EXPECTED_DIAGNOSTICS = [
    "cpu_vendor", "cpu_family", "cpu_model", "cpu_stepping", "cpu_flags_digest", "kernel_release"
]
EXPECTED_PERTURBATIONS = ["snapshot_root", "scratch_root", "working_directory"]


class ContractError(ValueError):
    pass


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ContractError(f"JSON object: duplicate key: {key}")
        out[key] = value
    return out


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)


def exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ContractError(f"{label}: closed-world schema mismatch")
    return value


def strict_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ContractError(f"{label}: boolean required")
    return value


def verify_parent_binding(profile: dict[str, Any], parent: Any) -> None:
    if not isinstance(parent, dict) or parent.get("schema") != PARENT_SCHEMA:
        raise ContractError("parent profile: schema mismatch")
    if parent.get("qualification_platform") != profile["qualification_platform"]:
        raise ContractError("parent profile: qualification platform mismatch")

    package = parent.get("nixpkgs_package")
    if not isinstance(package, dict) or package.get("main_program") != "wb_command":
        raise ContractError("parent profile: wb_command main program required")

    env = parent.get("execution_environment")
    if not isinstance(env, dict):
        raise ContractError("parent profile: execution environment required")
    expected_parent_env = {
        "lang": "C",
        "lc_all": "C",
        "timezone": "UTC0",
        "omp_num_threads": "1",
        "omp_dynamic": "FALSE",
        "home_policy": "fresh-private-empty",
        "xdg_config_home_policy": "fresh-private-empty",
        "xdg_cache_home_policy": "fresh-private-empty",
        "tmpdir_policy": "fresh-private-empty",
    }
    if env != expected_parent_env:
        raise ContractError("parent profile: exact v1 execution environment mismatch")


def verify_profile(value: Any, parent: Any) -> dict[str, Any]:
    profile = exact(value, TOP_KEYS, "invocation isolation profile")
    if profile["schema"] != SCHEMA or profile["status"] != "candidate_profile_only":
        raise ContractError("invocation isolation profile: schema/status mismatch")
    if profile["parent_execution_capsule_profile"] != PARENT_SCHEMA:
        raise ContractError("invocation isolation profile: parent schema mismatch")
    if profile["qualification_platform"] != "x86_64-linux":
        raise ContractError("invocation isolation profile: v1 requires x86_64-linux")

    program = exact(profile["program_binding"], PROGRAM_KEYS, "program binding")
    if strict_bool(program["require_absolute_exec_path"], "require_absolute_exec_path") is not True:
        raise ContractError("program binding: absolute executable path required")
    if strict_bool(program["host_path_lookup_allowed"], "host_path_lookup_allowed") is not False:
        raise ContractError("program binding: host PATH lookup forbidden")
    if program["root_source"] != "independently-verified-closure-root" or program["relative_main_program"] != "bin/wb_command":
        raise ContractError("program binding: exact verified closure-root wb_command binding required")

    env = exact(profile["process_environment"], ENV_KEYS, "process environment")
    if env["stage"] != "root-main-program-execve":
        raise ContractError("process environment: exact execve stage required")
    if strict_bool(env["inherit_host_environment"], "inherit_host_environment") is not False:
        raise ContractError("process environment: host inheritance forbidden")
    fixed = exact(env["fixed"], FIXED_ENV_KEYS, "fixed process environment")
    if fixed != EXPECTED_FIXED_ENV:
        raise ContractError("process environment: exact fixed environment mismatch")
    dynamic = exact(env["dynamic_bindings"], DYNAMIC_ENV_KEYS, "dynamic process environment")
    if dynamic != EXPECTED_DYNAMIC_ENV:
        raise ContractError("process environment: exact per-invocation dynamic bindings required")
    descendant = exact(env["descendant_environment"], DESCENDANT_ENV_KEYS, "descendant environment")
    for key in DESCENDANT_ENV_KEYS:
        strict_bool(descendant[key], f"descendant environment {key}")
    if descendant != EXPECTED_DESCENDANT_ENV:
        raise ContractError("descendant environment: exact non-equivalence boundary required")

    iso = exact(profile["per_invocation_isolation"], ISOLATION_KEYS, "per-invocation isolation")
    for key, expected in {
        "dynamic_environment_must_match_created_roots": True,
        "dynamic_environment_paths_must_be_absolute": True,
        "reuse_allowed": False,
        "stale_destination_fails_closed": True,
        "cleanup_must_be_confirmed": True,
        "close_fds": True,
    }.items():
        if strict_bool(iso[key], f"per-invocation isolation {key}") is not expected:
            raise ContractError(f"per-invocation isolation: boolean contract mismatch: {key}")
    expected_iso = {
        "cwd_policy": "fresh-private-empty-per-invocation",
        "home_policy": "fresh-private-empty-per-invocation",
        "xdg_config_home_policy": "fresh-private-empty-per-invocation",
        "xdg_cache_home_policy": "fresh-private-empty-per-invocation",
        "xdg_data_home_policy": "fresh-private-empty-per-invocation",
        "xdg_state_home_policy": "fresh-private-empty-per-invocation",
        "xdg_runtime_dir_policy": "fresh-private-empty-per-invocation",
        "xdg_config_dirs_policy": "fresh-private-empty-per-invocation",
        "xdg_data_dirs_policy": "fresh-private-empty-per-invocation",
        "tmpdir_policy": "fresh-private-empty-per-invocation",
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
    if iso != expected_iso:
        raise ContractError("per-invocation isolation: exact v1 contract mismatch")

    path_gate = exact(profile["scientific_path_gate"], PATH_GATE_KEYS, "scientific path gate")
    for key in (
        "snapshot_root_path_is_scientific_identity", "scratch_root_path_is_scientific_identity",
        "working_directory_path_is_scientific_identity", "bytewise_equivalence_assumed",
        "semantic_equivalence_assumed",
    ):
        if strict_bool(path_gate[key], f"scientific path gate {key}") is not False:
            raise ContractError(f"scientific path gate: premature promotion forbidden: {key}")
    if path_gate["required_perturbation_axes"] != EXPECTED_PERTURBATIONS:
        raise ContractError("scientific path gate: exact perturbation axes required")

    numeric = exact(profile["numerical_execution_gate"], NUMERIC_GATE_KEYS, "numerical execution gate")
    if strict_bool(numeric["same_x86_64_implies_equivalence"], "same_x86_64_implies_equivalence") is not False:
        raise ContractError("numerical gate: x86_64 equivalence cannot be assumed")
    if strict_bool(numeric["same_closure_implies_cross_cpu_equivalence"], "same_closure_implies_cross_cpu_equivalence") is not False:
        raise ContractError("numerical gate: closure identity cannot imply cross-CPU equivalence")
    if strict_bool(numeric["runtime_cpu_dispatch_known"], "runtime_cpu_dispatch_known") is not True:
        raise ContractError("numerical gate: runtime CPU dispatch fact must remain explicit")
    if numeric["known_dispatch_modes"] != EXPECTED_DISPATCH_MODES:
        raise ContractError("numerical gate: exact known CPU dispatch modes required")
    if strict_bool(numeric["same_host_repeatability_required"], "same_host_repeatability_required") is not True:
        raise ContractError("numerical gate: same-host repeatability must be required")
    if strict_bool(numeric["cross_cpu_equivalence_required_before_transfer"], "cross_cpu_equivalence_required_before_transfer") is not True:
        raise ContractError("numerical gate: cross-CPU transfer requires evidence")
    if numeric["diagnostic_context"] != EXPECTED_DIAGNOSTICS:
        raise ContractError("numerical gate: exact diagnostic context required")

    authority = exact(profile["authority"], AUTHORITY_KEYS, "authority")
    for key in AUTHORITY_KEYS:
        strict_bool(authority[key], f"authority {key}")
    if authority["invocation_contract_defined"] is not True:
        raise ContractError("authority: invocation contract must be defined")
    for key in AUTHORITY_KEYS - {"invocation_contract_defined"}:
        if authority[key] is not False:
            raise ContractError(f"authority escalation forbidden: {key}")

    verify_parent_binding(profile, parent)
    return profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--parent-profile", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        profile = verify_profile(load(args.profile), load(args.parent_profile))
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "schema": SCHEMA,
        "status": "validated-profile-only",
        "qualification_platform": profile["qualification_platform"],
        "authority": profile["authority"],
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
