#!/usr/bin/env python3
"""Validate the non-authorizing Workbench execution-capsule profile against flake.lock."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROFILE_SCHEMA = "symthaea-workbench-execution-capsule-profile-v1"
PROFILE_KEYS = {
    "schema",
    "status",
    "qualification_platform",
    "flake_selection",
    "nixpkgs_package",
    "execution_environment",
    "future_closure_receipt",
    "authority",
}
FLAKE_KEYS = {"root_input", "locked_node", "rev", "nar_hash"}
PACKAGE_KEYS = {
    "attribute",
    "package_path",
    "package_blob_sha",
    "pname",
    "version",
    "main_program",
    "upstream_tag",
    "upstream_source_hash",
}
ENV_KEYS = {
    "lang",
    "lc_all",
    "timezone",
    "omp_num_threads",
    "omp_dynamic",
    "home_policy",
    "xdg_config_home_policy",
    "xdg_cache_home_policy",
    "tmpdir_policy",
}
CLOSURE_KEYS = {
    "store_prefix",
    "require_recursive_runtime_closure",
    "require_per_path_nar_hash",
    "require_per_path_references",
    "closure_digest_algorithm",
    "closure_digest_serialization",
    "require_workbench_file_sha256",
    "require_workbench_version_output_sha256",
    "require_nix_version",
    "require_platform_identity",
}
AUTHORITY_KEYS = {
    "flake_selection_bound",
    "nixpkgs_package_metadata_pinned",
    "closure_realized",
    "closure_qualified",
    "transform_executed",
    "scientific_execution_qualified",
    "atlas_correctness_established",
    "fmq010_established",
    "neural_alignment_established",
    "consciousness_evidence",
}


class ContractError(ValueError):
    pass


def exact(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ContractError(f"{label}: closed-world schema mismatch")
    return value


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(profile_path: Path, flake_lock_path: Path) -> dict[str, Any]:
    profile = exact(load(profile_path), PROFILE_KEYS, "profile")
    if profile["schema"] != PROFILE_SCHEMA or profile["status"] != "candidate_profile_only":
        raise ContractError("profile: identity/status mismatch")
    if profile["qualification_platform"] != "x86_64-linux":
        raise ContractError("profile: v1 qualification platform must be x86_64-linux")

    selection = exact(profile["flake_selection"], FLAKE_KEYS, "flake selection")
    package = exact(profile["nixpkgs_package"], PACKAGE_KEYS, "nixpkgs package")
    environment = exact(profile["execution_environment"], ENV_KEYS, "execution environment")
    closure = exact(profile["future_closure_receipt"], CLOSURE_KEYS, "closure receipt")
    authority = exact(profile["authority"], AUTHORITY_KEYS, "authority")

    lock = load(flake_lock_path)
    if not isinstance(lock, dict) or not isinstance(lock.get("nodes"), dict):
        raise ContractError("flake.lock: nodes missing")
    root_name = lock.get("root")
    root = lock["nodes"].get(root_name, {})
    root_inputs = root.get("inputs", {}) if isinstance(root, dict) else {}
    if selection["root_input"] not in root_inputs:
        raise ContractError("flake.lock: root nixpkgs input missing")
    locked_node = root_inputs[selection["root_input"]]
    if locked_node != selection["locked_node"]:
        raise ContractError("flake.lock: locked nixpkgs node mismatch")
    locked = lock["nodes"].get(locked_node, {}).get("locked", {})
    if locked.get("rev") != selection["rev"] or locked.get("narHash") != selection["nar_hash"]:
        raise ContractError("flake.lock: nixpkgs revision/NAR root mismatch")

    if package != {
        "attribute": "connectome-workbench",
        "package_path": "pkgs/by-name/co/connectome-workbench/package.nix",
        "package_blob_sha": "01e021edc0f7795f946015bb2a69103cbffaa0ba",
        "pname": "connectome-workbench",
        "version": "2.1.0",
        "main_program": "wb_command",
        "upstream_tag": "v2.1.0",
        "upstream_source_hash": "sha256-f1T0i4x7rr3u/3ZvJ4cEAb377e7YcaGMKa2uUslVqR0=",
    }:
        raise ContractError("profile: pinned nixpkgs package metadata mismatch")

    if environment != {
        "lang": "C",
        "lc_all": "C",
        "timezone": "UTC0",
        "omp_num_threads": "1",
        "omp_dynamic": "FALSE",
        "home_policy": "fresh-private-empty",
        "xdg_config_home_policy": "fresh-private-empty",
        "xdg_cache_home_policy": "fresh-private-empty",
        "tmpdir_policy": "fresh-private-empty",
    }:
        raise ContractError("profile: execution environment drift")

    expected_closure = {
        "store_prefix": "/nix/store/",
        "require_recursive_runtime_closure": True,
        "require_per_path_nar_hash": True,
        "require_per_path_references": True,
        "closure_digest_algorithm": "sha256",
        "closure_digest_serialization": "canonical-json-sorted-store-paths-v1",
        "require_workbench_file_sha256": True,
        "require_workbench_version_output_sha256": True,
        "require_nix_version": True,
        "require_platform_identity": True,
    }
    if closure != expected_closure:
        raise ContractError("profile: future closure receipt contract mismatch")

    if authority != {
        "flake_selection_bound": True,
        "nixpkgs_package_metadata_pinned": True,
        "closure_realized": False,
        "closure_qualified": False,
        "transform_executed": False,
        "scientific_execution_qualified": False,
        "atlas_correctness_established": False,
        "fmq010_established": False,
        "neural_alignment_established": False,
        "consciousness_evidence": False,
    }:
        raise ContractError("profile: authority escalation or drift")
    return profile


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--flake-lock", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        validate(args.profile, args.flake_lock)
    except (ContractError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"profile": PROFILE_SCHEMA, "status": "validated-profile-only"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
