#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bind pre-materialized MuJoCo 3.8.0 evidence into a hardened v2 plan.

QUAL-INFRA-003B2 is deliberately non-executing. It imports the reviewed 003B1
compiler, validates the trusted MuJoCo runtime manifest plus the actual
libmujoco.so bytes, injects only the explicit dynamic-link directory required by
mujoco-rs, and recomputes semantic/execution plan identities.

003B2 does not establish that the materialization was reproduced from the
official release asset. That stronger proposition belongs to QUAL-INFRA-003D.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any

BASE_SCRIPT = Path(__file__).with_name("compile-hermetic-plan-v2b1.py")
BINDING_REVISION = "qual-infra-003b2"
MUJOCO_PROFILE_ID = "symthaea-humanoid-mujoco-hermetic-v2"
MUJOCO_RUNTIME_SCHEMA = "symthaea.qualification-mujoco-runtime.v1"
EXTRACTED_TREE_DIGEST_ALGORITHM = "symthaea.extracted-tree.sha256.v1"
MUJOCO_VERSION = "3.8.0"
MUJOCO_RS_COMPATIBILITY = "4.0.1+mj-3.8.0"
MUJOCO_RELEASE_COMMIT = "34d69ad4cb1a21846b8297e2bc5e68a4938276c1"
MUJOCO_RUNTIME_SUBDIR = Path("mujoco-3.8.0")
MUJOCO_GUEST_DYNAMIC_LINK_DIR = "/deps/mujoco-3.8.0/lib"
SHA256 = re.compile(r"^[0-9a-f]{64}$")

MANIFEST_FIELDS = {
    "schema_id",
    "platform",
    "version",
    "mujoco_rs_compatibility",
    "upstream_release_commit",
    "upstream_asset_id",
    "upstream_asset_name",
    "upstream_asset_size",
    "upstream_asset_sha256",
    "extracted_tree_digest_algorithm",
    "extracted_tree_sha256",
    "libmujoco_sha256",
}

EXPECTED_ASSETS = {
    "x86_64-linux": {
        "upstream_asset_id": 404891937,
        "upstream_asset_name": "mujoco-3.8.0-linux-x86_64.tar.gz",
        "upstream_asset_size": 20812715,
        "upstream_asset_sha256": "2be88c6f92a06c3eaffdb47d3a6d3fbf159fbc057e9d272d592fb194e41fefab",
    },
    "aarch64-linux": {
        "upstream_asset_id": 404891905,
        "upstream_asset_name": "mujoco-3.8.0-linux-aarch64.tar.gz",
        "upstream_asset_size": 20674425,
        "upstream_asset_sha256": "adc4a7856d2b8d42ba4e889b57cbceb13a329c869f410cf1ad110b153c4745e4",
    },
}

_spec = importlib.util.spec_from_file_location("hermetic_plan_v2b1", BASE_SCRIPT)
if _spec is None or _spec.loader is None:
    raise RuntimeError("unable to load QUAL-INFRA-003B1 compiler")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

PlanError = base.PlanError
REQUEST_SCHEMA = base.REQUEST_SCHEMA


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hex_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise PlanError(f"{label} must be lowercase SHA-256 hex")
    if set(value) == {"0"}:
        raise PlanError(f"{label} must not be all zero")
    return value


def _load_runtime_manifest(dependencies_root: Path) -> tuple[dict[str, Any], Path, Path, str]:
    runtime_root = dependencies_root / MUJOCO_RUNTIME_SUBDIR
    manifest_path = runtime_root / "runtime-manifest.json"
    lib_path = runtime_root / "lib" / "libmujoco.so"

    if runtime_root.is_symlink() or manifest_path.is_symlink():
        raise PlanError("MuJoCo runtime root/manifest must not be symlinks")
    if not runtime_root.is_dir():
        raise PlanError("MuJoCo runtime directory is missing")
    if not manifest_path.is_file():
        raise PlanError("MuJoCo runtime manifest is missing")

    base._validate_confined_symlinks(runtime_root, "MuJoCo runtime")

    if not lib_path.exists():
        raise PlanError("MuJoCo runtime must provide lib/libmujoco.so")
    try:
        resolved_lib = lib_path.resolve(strict=True)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        raise PlanError("MuJoCo libmujoco.so cannot be resolved") from exc
    try:
        resolved_lib.relative_to(runtime_root.resolve(strict=True))
    except ValueError as exc:
        raise PlanError("MuJoCo libmujoco.so escapes the materialized runtime") from exc
    if not resolved_lib.is_file():
        raise PlanError("MuJoCo libmujoco.so target must be a regular file")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PlanError("MuJoCo runtime manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_FIELDS:
        raise PlanError("MuJoCo runtime manifest fields mismatch")
    if manifest["schema_id"] != MUJOCO_RUNTIME_SCHEMA:
        raise PlanError("MuJoCo runtime manifest schema mismatch")
    if manifest["extracted_tree_digest_algorithm"] != EXTRACTED_TREE_DIGEST_ALGORITHM:
        raise PlanError("MuJoCo extracted-tree digest algorithm mismatch")
    if manifest["version"] != MUJOCO_VERSION:
        raise PlanError("MuJoCo runtime version mismatch")
    if manifest["mujoco_rs_compatibility"] != MUJOCO_RS_COMPATIBILITY:
        raise PlanError("MuJoCo wrapper compatibility mismatch")
    if manifest["upstream_release_commit"] != MUJOCO_RELEASE_COMMIT:
        raise PlanError("MuJoCo upstream release commit mismatch")

    platform = manifest["platform"]
    if platform not in EXPECTED_ASSETS:
        raise PlanError(f"unsupported MuJoCo qualification platform: {platform}")
    expected_asset = EXPECTED_ASSETS[platform]
    for field, expected in expected_asset.items():
        if manifest[field] != expected:
            raise PlanError(f"MuJoCo runtime {field} mismatch")

    _hex_sha256(manifest["upstream_asset_sha256"], "upstream_asset_sha256")
    _hex_sha256(manifest["extracted_tree_sha256"], "extracted_tree_sha256")
    expected_lib_sha = _hex_sha256(manifest["libmujoco_sha256"], "libmujoco_sha256")
    actual_lib_sha = _file_sha256(resolved_lib)
    if actual_lib_sha != expected_lib_sha:
        raise PlanError("MuJoCo libmujoco.so bytes do not match runtime manifest")

    manifest_file_sha = _file_sha256(manifest_path)
    return manifest, manifest_path, resolved_lib, manifest_file_sha


def compile_plan(policy: Any, profiles: Any, request: Any) -> dict[str, Any]:
    if not isinstance(request, dict) or request.get("qualification_id") != MUJOCO_PROFILE_ID:
        raise PlanError("003B2 accepts only the closed humanoid MuJoCo profile")

    plan = base.compile_plan(policy, profiles, request)
    dependencies_root = base.base._canonical_existing_path(
        request["dependencies"]["host_path"],
        "dependencies.host_path",
        directory=True,
    )
    manifest, manifest_path, resolved_lib, manifest_file_sha = _load_runtime_manifest(
        dependencies_root
    )

    semantic = dict(plan["semantic"])
    environment = dict(semantic["environment"])
    if "MUJOCO_DOWNLOAD_DIR" in environment or "MUJOCO_DYNAMIC_LINK_DIR" in environment:
        raise PlanError("MuJoCo link/download variables must be injected only by 003B2")
    environment["MUJOCO_DYNAMIC_LINK_DIR"] = MUJOCO_GUEST_DYNAMIC_LINK_DIR
    semantic["environment"] = environment
    semantic["mujoco_binding_revision"] = BINDING_REVISION
    semantic["mujoco_runtime"] = {
        "schema_id": manifest["schema_id"],
        "platform": manifest["platform"],
        "version": manifest["version"],
        "mujoco_rs_compatibility": manifest["mujoco_rs_compatibility"],
        "upstream_release_commit": manifest["upstream_release_commit"],
        "upstream_asset_id": manifest["upstream_asset_id"],
        "upstream_asset_name": manifest["upstream_asset_name"],
        "upstream_asset_size": manifest["upstream_asset_size"],
        "upstream_asset_sha256": manifest["upstream_asset_sha256"],
        "extracted_tree_digest_algorithm": manifest["extracted_tree_digest_algorithm"],
        "extracted_tree_sha256": manifest["extracted_tree_sha256"],
        "libmujoco_sha256": manifest["libmujoco_sha256"],
        "runtime_manifest_sha256": manifest_file_sha,
        "guest_dynamic_link_dir": MUJOCO_GUEST_DYNAMIC_LINK_DIR,
        "download_dir_policy": "ABSENT",
    }
    semantic_plan_sha256 = _canonical_sha256(semantic)

    prefix = base._build_prefix(
        semantic["backend_path"],
        semantic["namespace_flags"],
        plan["execution"]["mounts"],
        environment,
        semantic["working_directory"],
    )
    invocations = [
        prefix + ["--"] + list(command) for command in semantic["commands"]
    ]
    if any("MUJOCO_DOWNLOAD_DIR" in arg for argv in invocations for arg in argv):
        raise PlanError("003B2 refuses MUJOCO_DOWNLOAD_DIR in Bubblewrap argv")

    execution = dict(plan["execution"])
    execution["mujoco_binding_revision"] = BINDING_REVISION
    execution["semantic_plan_sha256"] = semantic_plan_sha256
    execution["mujoco_runtime"] = {
        "manifest_host_path": str(manifest_path.resolve(strict=True)),
        "libmujoco_host_path": str(resolved_lib),
        "runtime_manifest_sha256": manifest_file_sha,
        "libmujoco_sha256": manifest["libmujoco_sha256"],
    }
    execution["bubblewrap_invocations"] = invocations
    execution_plan_sha256 = _canonical_sha256(execution)

    hardened = dict(plan)
    hardened["mujoco_binding_revision"] = BINDING_REVISION
    hardened["semantic_plan_sha256"] = semantic_plan_sha256
    hardened["execution_plan_sha256"] = execution_plan_sha256
    hardened["semantic"] = semantic
    hardened["execution"] = execution
    hardened["nonclaims"] = list(plan["nonclaims"]) + [
        "003b2_manifest_binding_is_not_003d_materialization_qualification",
        "003b2_explicit_link_path_is_not_mujoco_runtime_execution_evidence",
        "003b2_compile_profile_is_not_hum_dyn_or_hum_wrench_behavioral_qualification",
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
        load_json(args.policy), load_json(args.profiles), load_json(args.request)
    )
    Path(args.output).write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(plan["semantic_plan_sha256"])
    print(plan["execution_plan_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
