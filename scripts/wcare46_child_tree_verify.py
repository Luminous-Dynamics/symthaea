#!/usr/bin/env python3
"""Verify exact WCARE-42/WCARE-43 child-tree integration without promoting execution claims."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

PROTOCOL = "wcare46-selective-child-tree-integration-v1"
W45 = "a53595ad9ef7ebd16eac8e047f70666323083f86"
W42 = "4bb84790bab3dc6a6d2e30d0ef081950a3954717"
W43 = "9e4bfd621e3c48ba6335f95b3dd1fd7b36dccf25"
INTEGRATION = "bc8701b207ac6ddc024f76d4bd84767a120e2804"
INTEGRATION_TREE = "842881c38aa4d148aad9b5f13fe84a8b36556ef4"
LOCK_PATH = "tools/wcare42_builder_attestation_verifier/Cargo.lock"
MANIFEST_PATH = "docs/release/evidence/WCARE46_CHILD_TREE_MANIFEST_V1.json"


class InvalidIntegration(Exception):
    pass


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 0:
        raise InvalidIntegration(f"git_failed:{' '.join(args)}:{result.stderr.strip()}")
    return result.stdout


def base() -> dict[str, Any]:
    return {
        "authority": "MeasurementOnly",
        "protocol_version": PROTOCOL,
        "classification": "INVALID_INTEGRATION",
        "detail": "uninitialized",
        "wcare45_parent_head": W45,
        "wcare42_source_head": W42,
        "wcare43_source_head": W43,
        "integration_commit": INTEGRATION,
        "integration_tree": INTEGRATION_TREE,
        "exact_24_file_diff_verified": False,
        "exact_path_mode_blob_census_verified": False,
        "wcare42_child_tree_present": False,
        "wcare43_child_tree_present": False,
        "wcare42_standalone_lock_present": False,
        "wcare45_preflight_classification": None,
        "child_execution_indeterminate": True,
        "child_reexecution_established": False,
        "child_verifier_lineage_established": False,
        "wcare42_executable_qualification_established": False,
        "wcare43_external_execution_lineage_established": False,
        "builder_authentication_established": False,
        "preregistration_temporal_precedence_established": False,
        "authenticated_preregistered_replication_established": False,
        "runtime_authority_granted": False,
    }


def load_manifest(root: Path) -> dict[str, Any]:
    try:
        value = json.loads((root / MANIFEST_PATH).read_text())
    except Exception as exc:
        raise InvalidIntegration(f"manifest_invalid:{exc}") from exc
    if not isinstance(value, dict):
        raise InvalidIntegration("manifest_not_object")
    required = {
        "protocol_version", "authority", "wcare45_parent_head", "wcare42_source_head",
        "wcare43_source_head", "integration_commit", "integration_tree", "wcare42_file_count",
        "wcare43_file_count", "total_file_count", "entries",
        "wcare42_standalone_lock_expected_absent", "child_reexecution_established",
        "child_verifier_lineage_established", "builder_authentication_established",
        "preregistration_temporal_precedence_established",
        "authenticated_preregistered_replication_established", "runtime_authority_granted",
    }
    if set(value) != required:
        raise InvalidIntegration(f"manifest_key_set_mismatch:{sorted(set(value) ^ required)}")
    if value["protocol_version"] != PROTOCOL or value["authority"] != "MeasurementOnly":
        raise InvalidIntegration("manifest_protocol_or_authority_mismatch")
    if value["wcare45_parent_head"] != W45 or value["wcare42_source_head"] != W42 or value["wcare43_source_head"] != W43:
        raise InvalidIntegration("manifest_source_head_mismatch")
    if value["integration_commit"] != INTEGRATION or value["integration_tree"] != INTEGRATION_TREE:
        raise InvalidIntegration("manifest_integration_subject_mismatch")
    if value["wcare42_file_count"] != 9 or value["wcare43_file_count"] != 15 or value["total_file_count"] != 24:
        raise InvalidIntegration("manifest_file_count_mismatch")
    if value["wcare42_standalone_lock_expected_absent"] is not True:
        raise InvalidIntegration("manifest_lock_absence_not_required")
    for field in (
        "child_reexecution_established", "child_verifier_lineage_established",
        "builder_authentication_established", "preregistration_temporal_precedence_established",
        "authenticated_preregistered_replication_established", "runtime_authority_granted",
    ):
        if value[field] is not False:
            raise InvalidIntegration(f"manifest_overclaim:{field}")
    return value


def verify_entries(root: Path, manifest: dict[str, Any]) -> tuple[set[str], int, int]:
    entries = manifest["entries"]
    if not isinstance(entries, list) or len(entries) != 24:
        raise InvalidIntegration("manifest_entries_not_24")
    expected_paths: set[str] = set()
    w42_count = 0
    w43_count = 0
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"source", "path", "mode", "blob_sha"}:
            raise InvalidIntegration("manifest_entry_shape_invalid")
        source = entry["source"]
        path = entry["path"]
        mode = entry["mode"]
        blob = entry["blob_sha"]
        if source not in {"WCARE42", "WCARE43"} or not isinstance(path, str) or mode != "100644":
            raise InvalidIntegration(f"manifest_entry_semantics_invalid:{path}")
        if not isinstance(blob, str) or len(blob) != 40 or any(c not in "0123456789abcdef" for c in blob):
            raise InvalidIntegration(f"manifest_blob_invalid:{path}")
        if path in expected_paths:
            raise InvalidIntegration(f"duplicate_manifest_path:{path}")
        expected_paths.add(path)
        if source == "WCARE42":
            w42_count += 1
        else:
            w43_count += 1
        line = git(root, "ls-tree", INTEGRATION, "--", path).strip()
        parts = line.split(None, 3)
        if len(parts) != 4:
            raise InvalidIntegration(f"integration_path_missing:{path}")
        actual_mode, actual_type, actual_blob, actual_path = parts
        if (actual_mode, actual_type, actual_blob, actual_path) != (mode, "blob", blob, path):
            raise InvalidIntegration(f"integration_tuple_mismatch:{path}:{line}")
    if w42_count != 9 or w43_count != 15:
        raise InvalidIntegration("manifest_source_count_mismatch")
    return expected_paths, w42_count, w43_count


def verify_diff(root: Path, expected_paths: set[str]) -> None:
    parents = git(root, "show", "-s", "--format=%P", INTEGRATION).strip().split()
    if parents != [W45]:
        raise InvalidIntegration("integration_parent_mismatch")
    tree = git(root, "show", "-s", "--format=%T", INTEGRATION).strip()
    if tree != INTEGRATION_TREE:
        raise InvalidIntegration("integration_tree_mismatch")
    diff = git(root, "diff-tree", "--no-commit-id", "--name-status", "-r", INTEGRATION)
    actual_paths: set[str] = set()
    lines = [line for line in diff.splitlines() if line.strip()]
    if len(lines) != 24:
        raise InvalidIntegration(f"integration_diff_count:{len(lines)}")
    for line in lines:
        parts = line.split("\t")
        if len(parts) != 2 or parts[0] != "A":
            raise InvalidIntegration(f"integration_non_addition:{line}")
        actual_paths.add(parts[1])
    if actual_paths != expected_paths:
        raise InvalidIntegration("integration_diff_path_set_mismatch")


def lock_present(root: Path, ref: str) -> bool:
    check = subprocess.run(["git", "cat-file", "-e", f"{ref}:{LOCK_PATH}"], cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return check.returncode == 0


def run_preflight(root: Path) -> dict[str, Any]:
    result = subprocess.run([sys.executable, "scripts/wcare45-preflight.py"], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if result.returncode != 3:
        raise InvalidIntegration(f"wcare45_preflight_exit:{result.returncode}:{result.stderr.strip()}")
    try:
        value = json.loads(result.stdout)
    except Exception as exc:
        raise InvalidIntegration(f"wcare45_preflight_json_invalid:{exc}") from exc
    if value.get("classification") != "CHILD_EXECUTION_INDETERMINATE":
        raise InvalidIntegration("wcare45_preflight_not_indeterminate")
    if value.get("wcare42_child_tree_present") is not True or value.get("wcare43_child_tree_present") is not True:
        raise InvalidIntegration("wcare45_child_tree_presence_not_true")
    if value.get("wcare42_standalone_lock_present") is not False:
        raise InvalidIntegration("wcare45_lock_unexpectedly_present")
    detail = value.get("detail", "")
    if "wcare42_child_tree_not_integrated" in detail or "wcare43_child_tree_not_integrated" in detail:
        raise InvalidIntegration("old_child_tree_absence_blocker_remains")
    for required in ("wcare42_standalone_lock_missing", "child_verifiers_not_reexecuted_by_wcare45_preflight"):
        if required not in detail:
            raise InvalidIntegration(f"required_blocker_missing:{required}")
    for field in (
        "child_verifier_lineage_established", "wcare42_executable_qualification_established",
        "wcare43_external_execution_lineage_established", "builder_authentication_established",
        "preregistration_temporal_precedence_established",
        "authenticated_preregistered_replication_established", "runtime_authority_granted",
    ):
        if value.get(field) is not False:
            raise InvalidIntegration(f"wcare45_preflight_overclaim:{field}")
    return value


def main() -> int:
    root_result = subprocess.run(["git", "rev-parse", "--show-toplevel"], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if root_result.returncode != 0:
        print(json.dumps(base() | {"detail": "not_in_git_worktree"}, sort_keys=True, separators=(",", ":")))
        return 4
    root = Path(root_result.stdout.strip())
    out = base()
    try:
        manifest = load_manifest(root)
        paths, _, _ = verify_entries(root, manifest)
        out["exact_path_mode_blob_census_verified"] = True
        verify_diff(root, paths)
        out["exact_24_file_diff_verified"] = True
        if lock_present(root, INTEGRATION) or lock_present(root, "HEAD"):
            raise InvalidIntegration("wcare42_standalone_lock_present")
        preflight = run_preflight(root)
        out["wcare42_child_tree_present"] = True
        out["wcare43_child_tree_present"] = True
        out["wcare42_standalone_lock_present"] = False
        out["wcare45_preflight_classification"] = preflight["classification"]
        out["classification"] = "PASS_WCARE46_CHILD_TREE_INTEGRATION"
        out["detail"] = "exact_24_child_blobs_integrated;lock_and_reexecution_blockers_preserved"
        print(json.dumps(out, sort_keys=True, separators=(",", ":")))
        return 0
    except InvalidIntegration as exc:
        out["detail"] = str(exc)
        print(json.dumps(out, sort_keys=True, separators=(",", ":")))
        return 4
    except Exception as exc:
        out["detail"] = f"unexpected_error:{type(exc).__name__}:{exc}"
        print(json.dumps(out, sort_keys=True, separators=(",", ":")))
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
