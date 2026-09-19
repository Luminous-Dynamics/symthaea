#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Apply QUAL-INFRA-003B1 containment hardening to deterministic v2 plans.

This module is deliberately non-executing. It imports the reviewed QUAL-INFRA-003B
compiler, validates additional filesystem invariants, replaces the generic /dev
devtmpfs with a closed device capability set, hard-disables nested user
namespaces, and recomputes semantic/execution plan identities.

It does not launch Bubblewrap, Cargo, or subject code.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

BASE_SCRIPT = Path(__file__).with_name("compile-hermetic-plan-v2.py")
HARDENING_REVISION = "qual-infra-003b1"
DEVICE_ALLOWLIST = (
    "/dev/null",
    "/dev/zero",
    "/dev/random",
    "/dev/urandom",
)
STDIO_SYMLINKS = (
    ("/proc/self/fd", "/dev/fd"),
    ("/proc/self/fd/0", "/dev/stdin"),
    ("/proc/self/fd/1", "/dev/stdout"),
    ("/proc/self/fd/2", "/dev/stderr"),
)

_spec = importlib.util.spec_from_file_location("hermetic_plan_v2_base", BASE_SCRIPT)
if _spec is None or _spec.loader is None:
    raise RuntimeError("unable to load QUAL-INFRA-003B base compiler")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

PlanError = base.PlanError
REQUEST_SCHEMA = base.REQUEST_SCHEMA


def _within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _validate_confined_symlinks(root: Path, label: str) -> None:
    """Require mounted data-tree symlinks to be relative and tree-confined."""
    for entry in root.rglob("*"):
        if not entry.is_symlink():
            continue
        raw_target = entry.readlink()
        if raw_target.is_absolute():
            raise PlanError(
                f"{label} contains absolute symlink: {entry} -> {raw_target}"
            )
        try:
            target = entry.resolve(strict=True)
        except (FileNotFoundError, RuntimeError, OSError) as exc:
            raise PlanError(f"{label} contains unresolved symlink: {entry}") from exc
        if not _within(target, root):
            raise PlanError(
                f"{label} symlink escapes mounted tree: {entry} -> {target}"
            )


def _validate_additional_inputs(request: Any) -> dict[str, Any]:
    request = base._load_request(request)
    source = base._canonical_existing_path(
        request["source"]["host_path"], "source.host_path", directory=True
    )
    dependencies = base._canonical_existing_path(
        request["dependencies"]["host_path"],
        "dependencies.host_path",
        directory=True,
    )
    _validate_confined_symlinks(source, "source")
    _validate_confined_symlinks(dependencies, "dependencies")
    return request


def _harden_namespace_flags(flags: list[str]) -> list[str]:
    hardened = list(flags)
    if "--unshare-all" in hardened or "--share-net" in hardened:
        raise PlanError("003B1 refuses soft/override namespace flags")
    try:
        user_index = hardened.index("--unshare-user")
    except ValueError as exc:
        raise PlanError("003B1 requires hard --unshare-user") from exc
    if "--disable-userns" not in hardened:
        hardened.insert(user_index + 1, "--disable-userns")
    if hardened.count("--disable-userns") != 1:
        raise PlanError("003B1 requires exactly one --disable-userns")
    return hardened


def _harden_mounts(mounts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace base `--dev /dev` with a tmpfs and a closed device allowlist."""
    hardened: list[dict[str, Any]] = []
    replaced = 0
    for mount in mounts:
        if mount.get("mode") == "dev" and mount.get("guest") == "/dev":
            replaced += 1
            hardened.append({"mode": "tmpfs", "guest": "/dev"})
            for device in DEVICE_ALLOWLIST:
                hardened.append(
                    {"mode": "dev-bind", "host": device, "guest": device}
                )
            for target, guest in STDIO_SYMLINKS:
                hardened.append(
                    {"mode": "symlink", "target": target, "guest": guest}
                )
            continue
        if mount.get("mode") == "dev":
            raise PlanError(f"003B1 refuses generic devtmpfs mount: {mount}")
        hardened.append(dict(mount))
    if replaced != 1:
        raise PlanError(
            f"003B1 expected exactly one base /dev mount, observed {replaced}"
        )
    guests = [mount["guest"] for mount in hardened]
    if len(set(guests)) != len(guests):
        raise PlanError("003B1 produced duplicate guest mount destination")
    return hardened


def _semantic_mount(mount: dict[str, Any]) -> dict[str, Any]:
    mode = mount["mode"]
    guest = mount["guest"]
    if mode in {"ro-bind", "bind", "proc", "tmpfs"}:
        return {"mode": mode, "guest": guest}
    if mode == "dev-bind":
        return {"mode": mode, "device": mount["host"], "guest": guest}
    if mode == "symlink":
        return {"mode": mode, "target": mount["target"], "guest": guest}
    raise PlanError(f"003B1 unsupported semantic mount mode: {mode}")


def _build_prefix(
    backend_path: str,
    namespace_flags: list[str],
    mounts: list[dict[str, Any]],
    environment: dict[str, str],
    working_directory: str,
) -> list[str]:
    prefix = [backend_path, *namespace_flags]
    for mount in mounts:
        mode = mount["mode"]
        if mode in {"ro-bind", "bind", "dev-bind"}:
            prefix.extend([f"--{mode}", mount["host"], mount["guest"]])
        elif mode in {"proc", "tmpfs"}:
            prefix.extend([f"--{mode}", mount["guest"]])
        elif mode == "symlink":
            prefix.extend(["--symlink", mount["target"], mount["guest"]])
        else:
            raise PlanError(f"003B1 unsupported execution mount mode: {mode}")
    for key, value in sorted(environment.items()):
        prefix.extend(["--setenv", key, value])
    prefix.extend(["--chdir", working_directory])
    return prefix


def compile_plan(policy: Any, profiles: Any, request: Any) -> dict[str, Any]:
    request = _validate_additional_inputs(request)
    plan = base.compile_plan(policy, profiles, request)

    namespace_flags = _harden_namespace_flags(plan["semantic"]["namespace_flags"])
    mounts = _harden_mounts(plan["execution"]["mounts"])

    semantic = dict(plan["semantic"])
    semantic["hardening_revision"] = HARDENING_REVISION
    semantic["namespace_flags"] = namespace_flags
    semantic["mounts"] = [_semantic_mount(mount) for mount in mounts]
    semantic["device_allowlist"] = list(DEVICE_ALLOWLIST)
    semantic_plan_sha256 = base.canonical_sha256(semantic)

    prefix = _build_prefix(
        semantic["backend_path"],
        namespace_flags,
        mounts,
        semantic["environment"],
        semantic["working_directory"],
    )
    invocations = [
        prefix + ["--"] + list(command) for command in semantic["commands"]
    ]

    execution = dict(plan["execution"])
    execution["hardening_revision"] = HARDENING_REVISION
    execution["semantic_plan_sha256"] = semantic_plan_sha256
    execution["mounts"] = mounts
    execution["bubblewrap_invocations"] = invocations
    execution_plan_sha256 = base.canonical_sha256(execution)

    hardened = dict(plan)
    hardened["compiler_hardening_revision"] = HARDENING_REVISION
    hardened["semantic_plan_sha256"] = semantic_plan_sha256
    hardened["execution_plan_sha256"] = execution_plan_sha256
    hardened["semantic"] = semantic
    hardened["execution"] = execution
    hardened["nonclaims"] = list(plan["nonclaims"]) + [
        "003b1_plan_hardening_is_not_runtime_containment",
        "device_allowlist_definition_is_not_device_isolation_evidence",
        "symlink_validation_is_not_namespace_escape_execution_evidence",
    ]
    return hardened


def load_json(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True)
    parser.add_argument("--profiles", required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    plan = compile_plan(
        load_json(args.policy),
        load_json(args.profiles),
        load_json(args.request),
    )
    Path(args.output).write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(plan["semantic_plan_sha256"])
    print(plan["execution_plan_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
