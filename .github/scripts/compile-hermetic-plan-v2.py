#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Compile a closed hermetic qualification profile into a deterministic bwrap plan.

QUAL-INFRA-003B is deliberately non-executing: this module never launches bwrap,
Cargo, or subject code. It validates trusted materialized inputs and emits both
a stable semantic plan identity and an execution-instance plan whose live
enforcement belongs to a later qualification tranche.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

POLICY_SCHEMA = "symthaea.qualification-hermetic-policy.v2"
PROFILES_SCHEMA = "symthaea.qualification-hermetic-profiles.v2"
REQUEST_SCHEMA = "symthaea.qualification-hermetic-plan-request.v2"
PLAN_SCHEMA = "symthaea.qualification-hermetic-plan.v2"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")

POLICY_FIELDS = {
    "schema_id",
    "authority",
    "runtime_authority",
    "physical_claim",
    "runner",
    "backend",
    "phase_model",
    "dependency_layout",
    "required_profile_fields",
    "namespace",
    "mounts",
    "environment",
    "adversarial_gates",
    "nonclaims",
}
BACKEND_FIELDS = {
    "primary",
    "package",
    "setuid_bwrap_allowed",
    "unprivileged_user_namespace_required",
    "apparmor",
}
PACKAGE_FIELDS = {"name", "ubuntu_release", "reviewed_version", "binary"}
APPARMOR_FIELDS = {
    "policy",
    "preferred_profile",
    "global_unprivileged_userns_restriction_disable_allowed",
    "fail_closed_if_profile_unavailable",
}
PHASE_FIELDS = {"materialization", "qualification"}
MATERIALIZATION_FIELDS = {"network", "subject_execution", "required_outputs"}
QUALIFICATION_FIELDS = {
    "network",
    "subject_execution",
    "cargo_offline",
    "immutable_materialization_required",
}
DEPENDENCY_LAYOUT_FIELDS = {
    "strategy",
    "vendor_path",
    "cargo_config_path",
    "cargo_home_path",
    "dependency_content_writable_by_subject",
}
NAMESPACE_FIELDS = {
    "unshare",
    "die_with_parent",
    "new_session",
    "clear_environment",
    "capabilities",
    "no_new_privileges",
}
POLICY_MOUNT_FIELDS = {
    "/",
    "/src",
    "/toolchain",
    "/deps",
    "/work",
    "/home",
    "/tmp",
    "/proc",
    "/dev",
    "control_plane",
    "evidence_output",
    "host_tmp",
    "host_workspace_other_than_subject",
}
POLICY_ENV_FIELDS = {
    "locale",
    "timezone",
    "home",
    "tmpdir",
    "cargo_home",
    "cargo_target_dir",
    "path",
    "cargo_net_offline",
    "git_config_global",
    "git_config_nosystem",
}
PROFILES_FIELDS = {"schema_id", "profiles"}
REQUEST_FIELDS = {
    "schema_id",
    "qualification_id",
    "source",
    "runtime_rootfs",
    "toolchain",
    "dependencies",
    "writable",
    "trusted",
    "bubblewrap",
}
SOURCE_FIELDS = {"host_path", "subject_sha", "tree_sha"}
DIGESTED_DIR_FIELDS = {"host_path", "sha256"}
DEPENDENCY_FIELDS = {"host_path", "manifest_sha256"}
WRITABLE_FIELDS = {"work", "home", "tmp"}
TRUSTED_FIELDS = {"control_plane", "evidence"}
BWRAP_FIELDS = {"host_path", "sha256"}
PROFILE_FIELDS = {
    "description",
    "network",
    "working_directory",
    "allowed_executables",
    "environment",
    "commands",
}
ALLOWED_ENV_KEYS = {
    "CARGO_HOME",
    "CARGO_NET_OFFLINE",
    "CARGO_TARGET_DIR",
    "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_NOSYSTEM",
    "HOME",
    "LANG",
    "LC_ALL",
    "PATH",
    "RUSTC",
    "RUSTDOC",
    "RUSTFMT",
    "TMPDIR",
    "TZ",
}
EXPECTED_ENVIRONMENT = {
    "CARGO_HOME": "/work/cargo-home",
    "CARGO_NET_OFFLINE": "true",
    "CARGO_TARGET_DIR": "/work/target",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "HOME": "/home",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/toolchain/bin:/usr/bin:/bin",
    "RUSTC": "/toolchain/bin/rustc",
    "RUSTDOC": "/toolchain/bin/rustdoc",
    "RUSTFMT": "/toolchain/bin/rustfmt",
    "TMPDIR": "/tmp",
    "TZ": "UTC",
}
EXPECTED_PROFILE_FIELDS = {
    "runtime_rootfs_digest",
    "bubblewrap_binary_digest",
    "rust_toolchain_digest",
    "cargo_vendor_manifest_digest",
    "source_subject_sha",
    "commands",
}
EXPECTED_MATERIALIZATION_OUTPUTS = {
    "runtime_rootfs",
    "rust_toolchain",
    "cargo_vendor_tree",
    "cargo_config",
    "materialization_manifest",
}
EXPECTED_DEPENDENCY_LAYOUT = {
    "strategy": "cargo-vendor",
    "vendor_path": "/deps/vendor",
    "cargo_config_path": "/deps/cargo-config.toml",
    "cargo_home_path": "/work/cargo-home",
    "dependency_content_writable_by_subject": False,
}
EXPECTED_MOUNT_POLICY = {
    "/": "READ_ONLY_PINNED_RUNTIME_ROOTFS",
    "/src": "READ_ONLY_EXACT_SUBJECT",
    "/toolchain": "READ_ONLY_PINNED_TOOLCHAIN",
    "/deps": "READ_ONLY_MATERIALIZED_VENDOR_AND_CONFIG",
    "/work": "PRIVATE_WRITABLE",
    "/home": "PRIVATE_WRITABLE",
    "/tmp": "PRIVATE_WRITABLE",
    "/proc": "PRIVATE_PROC",
    "/dev": "MINIMAL_DEVICE_SET",
    "control_plane": "NOT_VISIBLE",
    "evidence_output": "NOT_VISIBLE_TO_SUBJECT",
    "host_tmp": "NOT_VISIBLE",
    "host_workspace_other_than_subject": "NOT_VISIBLE",
}


class PlanError(RuntimeError):
    pass


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _exact_fields(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlanError(f"{label} must be an object")
    keys = set(value)
    if keys != expected:
        raise PlanError(
            f"{label} fields mismatch: missing={sorted(expected - keys)} unknown={sorted(keys - expected)}"
        )
    return value


def _hex(value: Any, pattern: re.Pattern[str], label: str) -> str:
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise PlanError(f"{label} has invalid hex identity")
    if set(value) == {"0"}:
        raise PlanError(f"{label} must not be all zero")
    return value


def _canonical_existing_path(value: Any, label: str, *, directory: bool) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise PlanError(f"{label} must be a non-empty path string")
    raw = Path(value)
    if not raw.is_absolute() or ".." in raw.parts:
        raise PlanError(f"{label} must be an absolute path without '..'")
    if raw.is_symlink():
        raise PlanError(f"{label} root must not itself be a symlink")
    try:
        resolved = raw.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PlanError(f"{label} does not exist") from exc
    if directory and not resolved.is_dir():
        raise PlanError(f"{label} must be a directory")
    if not directory and not resolved.is_file():
        raise PlanError(f"{label} must be a file")
    return resolved


def _nested(a: Path, b: Path) -> bool:
    if a == b:
        return True
    try:
        a.relative_to(b)
        return True
    except ValueError:
        pass
    try:
        b.relative_to(a)
        return True
    except ValueError:
        return False


def _require_disjoint(paths: dict[str, Path]) -> None:
    items = list(paths.items())
    for index, (name_a, path_a) in enumerate(items):
        for name_b, path_b in items[index + 1 :]:
            if _nested(path_a, path_b):
                raise PlanError(
                    f"host roots must be disjoint: {name_a}={path_a} {name_b}={path_b}"
                )


def _validate_dependency_materialization(root: Path) -> None:
    vendor = root / "vendor"
    config = root / "cargo-config.toml"
    if vendor.is_symlink() or config.is_symlink():
        raise PlanError("dependency vendor/config roots must not be symlinks")
    if not vendor.is_dir():
        raise PlanError("dependency materialization must contain vendor directory")
    if not config.is_file():
        raise PlanError("dependency materialization must contain cargo-config.toml")


def _validate_policy(policy: Any) -> dict[str, Any]:
    policy = _exact_fields(policy, POLICY_FIELDS, "policy")
    if policy["schema_id"] != POLICY_SCHEMA:
        raise PlanError("hermetic policy schema mismatch")
    if policy["authority"] != "sandbox-policy-definition-only":
        raise PlanError("hermetic policy authority mismatch")
    if policy["runtime_authority"] != "NONE" or policy["physical_claim"] != "NONE":
        raise PlanError("hermetic policy nonclaims mismatch")
    if policy["runner"] != "ubuntu-24.04":
        raise PlanError("hermetic policy runner mismatch")

    backend = _exact_fields(policy["backend"], BACKEND_FIELDS, "policy.backend")
    package = _exact_fields(backend["package"], PACKAGE_FIELDS, "policy.backend.package")
    apparmor = _exact_fields(backend["apparmor"], APPARMOR_FIELDS, "policy.backend.apparmor")
    if backend["primary"] != "bubblewrap":
        raise PlanError("v2 policy requires bubblewrap backend")
    if package != {
        "name": "bubblewrap",
        "ubuntu_release": "24.04",
        "reviewed_version": "0.9.0-1ubuntu0.3",
        "binary": "/usr/bin/bwrap",
    }:
        raise PlanError("bubblewrap package policy mismatch")
    if backend["setuid_bwrap_allowed"] is not False:
        raise PlanError("setuid bubblewrap is not admitted")
    if backend["unprivileged_user_namespace_required"] is not True:
        raise PlanError("unprivileged user namespaces must be required")
    if apparmor != {
        "policy": "profile-scoped-only",
        "preferred_profile": "bwrap-userns-restrict",
        "global_unprivileged_userns_restriction_disable_allowed": False,
        "fail_closed_if_profile_unavailable": True,
    }:
        raise PlanError("AppArmor policy mismatch")

    phase = _exact_fields(policy["phase_model"], PHASE_FIELDS, "policy.phase_model")
    materialization = _exact_fields(
        phase["materialization"], MATERIALIZATION_FIELDS, "policy.phase_model.materialization"
    )
    qualification = _exact_fields(
        phase["qualification"], QUALIFICATION_FIELDS, "policy.phase_model.qualification"
    )
    if materialization["network"] != "TRUSTED_CONTROL_PLANE_ONLY":
        raise PlanError("materialization network policy mismatch")
    if materialization["subject_execution"] is not False:
        raise PlanError("subject execution is forbidden during materialization")
    if set(materialization["required_outputs"]) != EXPECTED_MATERIALIZATION_OUTPUTS:
        raise PlanError("materialization output contract mismatch")
    if qualification != {
        "network": "NONE",
        "subject_execution": True,
        "cargo_offline": True,
        "immutable_materialization_required": True,
    }:
        raise PlanError("qualification phase policy mismatch")

    dependency_layout = _exact_fields(
        policy["dependency_layout"], DEPENDENCY_LAYOUT_FIELDS, "policy.dependency_layout"
    )
    if dependency_layout != EXPECTED_DEPENDENCY_LAYOUT:
        raise PlanError("dependency layout policy mismatch")

    namespace = _exact_fields(policy["namespace"], NAMESPACE_FIELDS, "policy.namespace")
    mounts = _exact_fields(policy["mounts"], POLICY_MOUNT_FIELDS, "policy.mounts")
    environment = _exact_fields(policy["environment"], POLICY_ENV_FIELDS, "policy.environment")
    if mounts != EXPECTED_MOUNT_POLICY:
        raise PlanError("policy mount map mismatch")
    if environment != {
        "locale": "C.UTF-8",
        "timezone": "UTC",
        "home": "/home",
        "tmpdir": "/tmp",
        "cargo_home": "/work/cargo-home",
        "cargo_target_dir": "/work/target",
        "path": "/toolchain/bin:/usr/bin:/bin",
        "cargo_net_offline": True,
        "git_config_global": "/dev/null",
        "git_config_nosystem": True,
    }:
        raise PlanError("policy environment mismatch")
    if set(namespace["unshare"]) != {"user", "ipc", "pid", "uts", "cgroup", "network"}:
        raise PlanError("policy namespace set mismatch")
    if namespace != {
        "unshare": namespace["unshare"],
        "die_with_parent": True,
        "new_session": True,
        "clear_environment": True,
        "capabilities": "NONE",
        "no_new_privileges": True,
    }:
        raise PlanError("policy namespace semantics mismatch")
    if set(policy["required_profile_fields"]) != EXPECTED_PROFILE_FIELDS:
        raise PlanError("policy required_profile_fields mismatch")
    if not isinstance(policy["adversarial_gates"], list) or not policy["adversarial_gates"]:
        raise PlanError("policy adversarial gates must be non-empty")
    if not isinstance(policy["nonclaims"], list) or not policy["nonclaims"]:
        raise PlanError("policy nonclaims must be non-empty")
    return policy


def _validate_profiles(profiles: Any) -> dict[str, Any]:
    profiles = _exact_fields(profiles, PROFILES_FIELDS, "profiles registry")
    if profiles["schema_id"] != PROFILES_SCHEMA:
        raise PlanError("hermetic profile registry schema mismatch")
    table = profiles["profiles"]
    if not isinstance(table, dict) or not table:
        raise PlanError("hermetic profile registry is empty")
    return profiles


def _validate_profile(profile: Any) -> dict[str, Any]:
    profile = _exact_fields(profile, PROFILE_FIELDS, "profile")
    if profile["network"] != "NONE":
        raise PlanError("hermetic profile network must be NONE")
    if profile["working_directory"] != "/src":
        raise PlanError("hermetic profile working directory must be /src")
    allowed = profile["allowed_executables"]
    if allowed != ["/toolchain/bin/cargo"]:
        raise PlanError("v2 profile executable set mismatch")
    if profile["environment"] != EXPECTED_ENVIRONMENT:
        raise PlanError("v2 profile environment mismatch")
    commands = profile["commands"]
    if not isinstance(commands, list) or not commands:
        raise PlanError("v2 profile commands must be a non-empty list")
    required_prefix = ["/toolchain/bin/cargo", "--offline", "--config", "/deps/cargo-config.toml"]
    for command in commands:
        if not isinstance(command, list) or not command:
            raise PlanError("every command must be a non-empty argv list")
        if not all(isinstance(arg, str) and arg and "\0" not in arg for arg in command):
            raise PlanError("command argv contains invalid values")
        if command[0] not in allowed:
            raise PlanError(f"command executable is not admitted: {command[0]}")
        if command[:4] != required_prefix:
            raise PlanError("every Cargo command must bind offline mode and trusted vendor config")
    return profile


def _load_request(request: Any) -> dict[str, Any]:
    request = _exact_fields(request, REQUEST_FIELDS, "request")
    if request["schema_id"] != REQUEST_SCHEMA:
        raise PlanError("plan request schema mismatch")
    _exact_fields(request["source"], SOURCE_FIELDS, "request.source")
    _exact_fields(request["runtime_rootfs"], DIGESTED_DIR_FIELDS, "request.runtime_rootfs")
    _exact_fields(request["toolchain"], DIGESTED_DIR_FIELDS, "request.toolchain")
    _exact_fields(request["dependencies"], DEPENDENCY_FIELDS, "request.dependencies")
    _exact_fields(request["writable"], WRITABLE_FIELDS, "request.writable")
    _exact_fields(request["trusted"], TRUSTED_FIELDS, "request.trusted")
    _exact_fields(request["bubblewrap"], BWRAP_FIELDS, "request.bubblewrap")
    return request


def compile_plan(policy: Any, profiles: Any, request: Any) -> dict[str, Any]:
    policy = _validate_policy(policy)
    profiles = _validate_profiles(profiles)
    request = _load_request(request)

    qualification_id = request["qualification_id"]
    table = profiles["profiles"]
    if not isinstance(qualification_id, str) or qualification_id not in table:
        raise PlanError(f"qualification profile is not registered: {qualification_id}")
    profile = _validate_profile(table[qualification_id])

    source = request["source"]
    runtime = request["runtime_rootfs"]
    toolchain = request["toolchain"]
    dependencies = request["dependencies"]
    writable = request["writable"]
    trusted = request["trusted"]
    bubblewrap = request["bubblewrap"]

    _hex(source["subject_sha"], SHA40, "source.subject_sha")
    _hex(source["tree_sha"], SHA40, "source.tree_sha")
    _hex(runtime["sha256"], SHA256, "runtime_rootfs.sha256")
    _hex(toolchain["sha256"], SHA256, "toolchain.sha256")
    _hex(dependencies["manifest_sha256"], SHA256, "dependencies.manifest_sha256")
    _hex(bubblewrap["sha256"], SHA256, "bubblewrap.sha256")

    host_paths = {
        "source": _canonical_existing_path(source["host_path"], "source.host_path", directory=True),
        "runtime_rootfs": _canonical_existing_path(runtime["host_path"], "runtime_rootfs.host_path", directory=True),
        "toolchain": _canonical_existing_path(toolchain["host_path"], "toolchain.host_path", directory=True),
        "dependencies": _canonical_existing_path(dependencies["host_path"], "dependencies.host_path", directory=True),
        "work": _canonical_existing_path(writable["work"], "writable.work", directory=True),
        "home": _canonical_existing_path(writable["home"], "writable.home", directory=True),
        "tmp": _canonical_existing_path(writable["tmp"], "writable.tmp", directory=True),
        "control_plane": _canonical_existing_path(trusted["control_plane"], "trusted.control_plane", directory=True),
        "evidence": _canonical_existing_path(trusted["evidence"], "trusted.evidence", directory=True),
    }
    _require_disjoint(host_paths)
    _validate_dependency_materialization(host_paths["dependencies"])

    expected_bwrap = policy["backend"]["package"]["binary"]
    if not isinstance(bubblewrap["host_path"], str) or bubblewrap["host_path"] != expected_bwrap:
        raise PlanError(f"bubblewrap path mismatch: expected {expected_bwrap}")
    bwrap_path = expected_bwrap

    mounts = [
        {"mode": "ro-bind", "host": str(host_paths["runtime_rootfs"]), "guest": "/"},
        {"mode": "ro-bind", "host": str(host_paths["source"]), "guest": "/src"},
        {"mode": "ro-bind", "host": str(host_paths["toolchain"]), "guest": "/toolchain"},
        {"mode": "ro-bind", "host": str(host_paths["dependencies"]), "guest": "/deps"},
        {"mode": "bind", "host": str(host_paths["work"]), "guest": "/work"},
        {"mode": "bind", "host": str(host_paths["home"]), "guest": "/home"},
        {"mode": "bind", "host": str(host_paths["tmp"]), "guest": "/tmp"},
        {"mode": "proc", "guest": "/proc"},
        {"mode": "dev", "guest": "/dev"},
    ]
    if len({mount["guest"] for mount in mounts}) != len(mounts):
        raise PlanError("duplicate guest mount destination")

    namespace_flags = [
        "--unshare-user",
        "--unshare-ipc",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-uts",
        "--unshare-cgroup",
        "--die-with-parent",
        "--new-session",
        "--clearenv",
        "--cap-drop",
        "ALL",
    ]
    prefix = [bwrap_path, *namespace_flags]
    for mount in mounts:
        mode = mount["mode"]
        if mode in {"ro-bind", "bind"}:
            prefix.extend([f"--{mode}", mount["host"], mount["guest"]])
        elif mode in {"proc", "dev"}:
            prefix.extend([f"--{mode}", mount["guest"]])
        else:
            raise PlanError(f"internal unsupported mount mode: {mode}")
    for key, value in sorted(profile["environment"].items()):
        prefix.extend(["--setenv", key, value])
    prefix.extend(["--chdir", profile["working_directory"]])

    commands = [list(command) for command in profile["commands"]]
    invocations = [prefix + ["--"] + command for command in commands]
    semantic_mounts = [{"mode": mount["mode"], "guest": mount["guest"]} for mount in mounts]

    semantic_core: dict[str, Any] = {
        "schema_id": PLAN_SCHEMA,
        "policy_canonical_sha256": canonical_sha256(policy),
        "profiles_canonical_sha256": canonical_sha256(profiles),
        "qualification_id": qualification_id,
        "identities": {
            "source_subject_sha": source["subject_sha"],
            "source_tree_sha": source["tree_sha"],
            "runtime_rootfs_sha256": runtime["sha256"],
            "rust_toolchain_sha256": toolchain["sha256"],
            "cargo_vendor_manifest_sha256": dependencies["manifest_sha256"],
            "bubblewrap_sha256": bubblewrap["sha256"],
        },
        "backend_path": bwrap_path,
        "namespace_flags": namespace_flags,
        "network": "NONE",
        "mounts": semantic_mounts,
        "environment": dict(sorted(profile["environment"].items())),
        "working_directory": profile["working_directory"],
        "commands": commands,
    }
    semantic_plan_sha256 = canonical_sha256(semantic_core)

    execution_core: dict[str, Any] = {
        "semantic_plan_sha256": semantic_plan_sha256,
        "request_canonical_sha256": canonical_sha256(request),
        "mounts": mounts,
        "trusted_paths_excluded": {
            "control_plane": str(host_paths["control_plane"]),
            "evidence": str(host_paths["evidence"]),
        },
        "bubblewrap_invocations": invocations,
    }
    execution_plan_sha256 = canonical_sha256(execution_core)

    return {
        "schema_id": PLAN_SCHEMA,
        "authority": "deterministic-sandbox-plan-only",
        "runtime_authority": "NONE",
        "backend_execution_claim": "NONE",
        "semantic_plan_sha256": semantic_plan_sha256,
        "execution_plan_sha256": execution_plan_sha256,
        "semantic": semantic_core,
        "execution": execution_core,
        "nonclaims": [
            "plan_compilation_is_not_bubblewrap_execution",
            "plan_compilation_is_not_namespace_enforcement",
            "plan_compilation_is_not_source_qualification",
            "plan_compilation_is_not_runtime_authority",
            "plan_compilation_is_not_hardware_observation",
            "plan_compilation_is_not_physical_safety",
        ],
    }


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    plan = compile_plan(load_json(args.policy), load_json(args.profiles), load_json(args.request))
    Path(args.output).write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(plan["semantic_plan_sha256"])
    print(plan["execution_plan_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
