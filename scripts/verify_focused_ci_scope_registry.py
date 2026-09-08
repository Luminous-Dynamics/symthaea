#!/usr/bin/env python3
"""Validate the focused-CI registry and admit changed paths fail-closed."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "symthaea-focused-ci-scope-registry-v1"
TOP_KEYS = {
    "schema",
    "default_policy",
    "unknown_path_policy",
    "ambiguous_owner_policy",
    "mixed_composition_group_policy",
    "scientific_authority",
    "meta_qualifier_workflow",
    "scopes",
}
SCOPE_KEYS = {
    "id",
    "composition_group",
    "composable",
    "global_ci_eligible",
    "qualification_source_mode",
    "qualification_source_identity_qualified",
    "executing_workflow_identity_qualified",
    "merge_compatibility_separate",
    "scientific_authority",
    "focused_workflow",
    "focused_workflow_git_blob",
    "files",
}
ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
WORKFLOW_RE = re.compile(r"^\.github/workflows/[a-z0-9][a-z0-9-]*\.yml$")
GIT_BLOB_RE = re.compile(r"^[0-9a-f]{40}$")

CONTROL_PLANE_EXACT = {
    ".cargo/config",
    ".cargo/config.toml",
    ".github/CODEOWNERS",
    ".github/workflows/ci-source-tree-identity.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/focused-ci-scope-registry.yml",
    ".github/workflows/workbench-provenance-ci-scope.yml",
    "Cargo.lock",
    "Cargo.toml",
    "data/ci/ci_source_tree_identity_profile_v1.json",
    "data/ci/focused_ci_scope_registry_v1.json",
    "docs/ci/CI_SOURCE_TREE_IDENTITY_V1.md",
    "docs/ci/CI_WORKFLOW_DEFINITION_IDENTITY_V1.md",
    "docs/ci/FOCUSED_CI_SCOPE_REGISTRY_V1.md",
    "docs/ci/WORKBENCH_PROVENANCE_CI_SCOPE_V1.md",
    "flake.lock",
    "flake.nix",
    "rust-toolchain",
    "rust-toolchain.toml",
    "scripts/check-class-a-changes.sh",
    "scripts/collect_ci_changed_paths.py",
    "scripts/classify_workbench_provenance_ci_scope.py",
    "scripts/render_workbench_provenance_ci_paths_ignore.py",
    "scripts/test_classify_workbench_provenance_ci_scope.py",
    "scripts/test_collect_ci_changed_paths.py",
    "scripts/test_render_workbench_provenance_ci_paths_ignore.py",
    "scripts/test_verify_ci_source_tree_identity.py",
    "scripts/test_verify_ci_workflow_definition_identity.py",
    "scripts/test_verify_focused_ci_scope_registry.py",
    "scripts/verify_ci_source_tree_identity.py",
    "scripts/verify_ci_workflow_definition_identity.py",
    "scripts/verify_focused_ci_scope_registry.py",
}
CONTROL_PLANE_PREFIXES = (".github/actions/",)


class RegistryError(ValueError):
    pass


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _exact_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise RegistryError(f"{label}: exact boolean required")
    return value


def _canonical_repo_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RegistryError(f"{label}: non-empty string required")
    if "\x00" in value or "\n" in value or "\r" in value or "\\" in value:
        raise RegistryError(f"{label}: control/backslash spelling forbidden")
    if value.startswith("/"):
        raise RegistryError(f"{label}: absolute path forbidden")
    pure = PurePosixPath(value)
    if value != "/".join(pure.parts):
        raise RegistryError(f"{label}: exact canonical POSIX spelling required")
    if any(part in {"", ".", ".."} for part in pure.parts):
        raise RegistryError(f"{label}: traversal/noncanonical component forbidden")
    return value


def _is_control_plane(path: str) -> bool:
    return path in CONTROL_PLANE_EXACT or any(
        path.startswith(prefix) for prefix in CONTROL_PLANE_PREFIXES
    )


def _closed_keys(obj: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise RegistryError(f"{label}: object required")
    keys = set(obj)
    if keys != expected:
        raise RegistryError(f"{label}: keys mismatch: {sorted(keys ^ expected)}")
    return obj


def validate_registry(raw: Any) -> dict[str, Any]:
    registry = _closed_keys(raw, TOP_KEYS, "registry")
    if registry["schema"] != SCHEMA:
        raise RegistryError("registry: schema mismatch")
    for key in ("default_policy", "unknown_path_policy", "mixed_composition_group_policy"):
        if registry[key] != "full-ci-required":
            raise RegistryError(f"registry: {key} must fail closed")
    if registry["ambiguous_owner_policy"] != "registry-invalid":
        raise RegistryError("registry: ambiguous_owner_policy must invalidate registry")
    if _exact_bool(registry["scientific_authority"], "registry.scientific_authority"):
        raise RegistryError("registry: scientific authority forbidden")

    meta = _canonical_repo_path(
        registry["meta_qualifier_workflow"], "registry.meta_qualifier_workflow"
    )
    if not WORKFLOW_RE.fullmatch(meta):
        raise RegistryError("registry: meta qualifier must be a workflow .yml path")
    if meta not in CONTROL_PLANE_EXACT:
        raise RegistryError("registry: meta qualifier must remain control-plane protected")

    scopes = registry["scopes"]
    if not isinstance(scopes, list) or not scopes:
        raise RegistryError("registry.scopes: non-empty array required")

    ids: set[str] = set()
    owners: dict[str, str] = {}
    normalized: list[dict[str, Any]] = []

    for index, candidate in enumerate(scopes):
        label = f"registry.scopes[{index}]"
        scope = _closed_keys(candidate, SCOPE_KEYS, label)

        scope_id = scope["id"]
        if not isinstance(scope_id, str) or not ID_RE.fullmatch(scope_id):
            raise RegistryError(f"{label}.id: canonical kebab-case required")
        if scope_id in ids:
            raise RegistryError(f"registry: duplicate scope id: {scope_id}")
        ids.add(scope_id)

        group = scope["composition_group"]
        if not isinstance(group, str) or not ID_RE.fullmatch(group):
            raise RegistryError(
                f"{label}.composition_group: canonical kebab-case required"
            )

        composable = _exact_bool(scope["composable"], f"{label}.composable")
        eligible = _exact_bool(
            scope["global_ci_eligible"], f"{label}.global_ci_eligible"
        )
        source_mode = scope["qualification_source_mode"]
        if source_mode != "exact-pr-head":
            raise RegistryError(
                f"{label}.qualification_source_mode: exact-pr-head required in v1"
            )
        source_qualified = _exact_bool(
            scope["qualification_source_identity_qualified"],
            f"{label}.qualification_source_identity_qualified",
        )
        executing_qualified = _exact_bool(
            scope["executing_workflow_identity_qualified"],
            f"{label}.executing_workflow_identity_qualified",
        )
        merge_separate = _exact_bool(
            scope["merge_compatibility_separate"],
            f"{label}.merge_compatibility_separate",
        )
        if not merge_separate:
            raise RegistryError(
                f"{label}: theorem qualification and merge compatibility must remain separate"
            )
        if eligible and not source_qualified:
            raise RegistryError(
                f"{label}: global CI eligibility requires qualified source identity"
            )
        if eligible and not executing_qualified:
            raise RegistryError(
                f"{label}: global CI eligibility requires qualified executing workflow identity"
            )
        if _exact_bool(scope["scientific_authority"], f"{label}.scientific_authority"):
            raise RegistryError(f"{label}: scientific authority forbidden")

        workflow = _canonical_repo_path(
            scope["focused_workflow"], f"{label}.focused_workflow"
        )
        if not WORKFLOW_RE.fullmatch(workflow):
            raise RegistryError(
                f"{label}: focused_workflow must be .github/workflows/*.yml"
            )
        if workflow == meta or workflow == ".github/workflows/ci.yml":
            raise RegistryError(
                f"{label}: focused workflow cannot be a routing/global workflow"
            )
        workflow_blob = scope["focused_workflow_git_blob"]
        if not isinstance(workflow_blob, str) or not GIT_BLOB_RE.fullmatch(workflow_blob):
            raise RegistryError(
                f"{label}.focused_workflow_git_blob: 40 lowercase hex required"
            )

        files_raw = scope["files"]
        if not isinstance(files_raw, list) or not files_raw:
            raise RegistryError(f"{label}.files: non-empty array required")
        files: list[str] = []
        local_seen: set[str] = set()
        for raw_path in files_raw:
            path = _canonical_repo_path(raw_path, f"{label}.files")
            if path in local_seen:
                raise RegistryError(f"{label}: duplicate file: {path}")
            if _is_control_plane(path):
                raise RegistryError(
                    f"{label}: control-plane ownership forbidden: {path}"
                )
            if path in owners:
                raise RegistryError(
                    f"registry: ambiguous owner for {path}: {owners[path]} and {scope_id}"
                )
            local_seen.add(path)
            owners[path] = scope_id
            files.append(path)

        if files != sorted(files):
            raise RegistryError(f"{label}.files: canonical sorted order required")
        if workflow not in local_seen:
            raise RegistryError(
                f"{label}: focused workflow must be owned by its scope"
            )

        normalized.append(
            {
                "id": scope_id,
                "composition_group": group,
                "composable": composable,
                "global_ci_eligible": eligible,
                "qualification_source_mode": source_mode,
                "qualification_source_identity_qualified": source_qualified,
                "executing_workflow_identity_qualified": executing_qualified,
                "merge_compatibility_separate": True,
                "scientific_authority": False,
                "focused_workflow": workflow,
                "focused_workflow_git_blob": workflow_blob,
                "files": files,
            }
        )

    return {
        "schema": SCHEMA,
        "default_policy": "full-ci-required",
        "unknown_path_policy": "full-ci-required",
        "ambiguous_owner_policy": "registry-invalid",
        "mixed_composition_group_policy": "full-ci-required",
        "scientific_authority": False,
        "meta_qualifier_workflow": meta,
        "scopes": normalized,
    }


def _observe_qualifier_blobs(
    scopes: list[dict[str, Any]], repository_root: Path
) -> tuple[dict[str, str], dict[str, str]]:
    try:
        root = repository_root.resolve(strict=True)
    except OSError as exc:
        return {}, {"<repository-root>": f"unavailable: {exc.__class__.__name__}"}

    observed: dict[str, str] = {}
    errors: dict[str, str] = {}
    for scope in scopes:
        workflow = scope["focused_workflow"]
        target = root.joinpath(*PurePosixPath(workflow).parts)
        try:
            if target.is_symlink():
                raise RegistryError("symlink forbidden")
            resolved = target.resolve(strict=True)
            resolved.relative_to(root)
            if not resolved.is_file():
                raise RegistryError("regular file required")
            observed[workflow] = git_blob_sha(resolved.read_bytes())
        except (OSError, ValueError, RegistryError) as exc:
            errors[workflow] = f"{exc.__class__.__name__}: {exc}"
    return observed, errors


def admit_changed_paths(
    registry_raw: Any,
    changed_raw: Any,
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    registry = validate_registry(registry_raw)
    if not isinstance(changed_raw, list):
        raise RegistryError("changed paths: JSON array required")

    changed: list[str] = []
    seen: set[str] = set()
    for raw in changed_raw:
        path = _canonical_repo_path(raw, "changed path")
        if path in seen:
            raise RegistryError(f"changed paths: duplicate path: {path}")
        seen.add(path)
        changed.append(path)
    changed.sort()

    scope_by_id = {scope["id"]: scope for scope in registry["scopes"]}
    owner_by_path = {
        path: scope["id"]
        for scope in registry["scopes"]
        for path in scope["files"]
    }

    reasons: list[str] = []
    touched_ids: set[str] = set()
    unowned: list[str] = []
    control_plane: list[str] = []
    changed_qualifiers: list[str] = []

    if not changed:
        reasons.append("empty-diff")

    for path in changed:
        if _is_control_plane(path):
            control_plane.append(path)
            continue
        owner = owner_by_path.get(path)
        if owner is None:
            unowned.append(path)
            continue
        touched_ids.add(owner)
        if path == scope_by_id[owner]["focused_workflow"]:
            changed_qualifiers.append(path)

    if control_plane:
        reasons.append("control-plane-change")
    if unowned:
        reasons.append("unowned-path")
    if changed_qualifiers:
        reasons.append("focused-qualifier-change")

    touched = [scope_by_id[scope_id] for scope_id in sorted(touched_ids)]
    if any(not scope["global_ci_eligible"] for scope in touched):
        reasons.append("scope-not-global-ci-eligible")
    if any(not scope["qualification_source_identity_qualified"] for scope in touched):
        reasons.append("source-identity-not-qualified")
    if any(not scope["executing_workflow_identity_qualified"] for scope in touched):
        reasons.append("executing-workflow-identity-not-qualified")

    groups = sorted({scope["composition_group"] for scope in touched})
    if len(groups) > 1:
        reasons.append("mixed-composition-groups")
    if len(touched) > 1 and any(not scope["composable"] for scope in touched):
        reasons.append("noncomposable-scope-mix")

    expected_blobs = {
        scope["focused_workflow"]: scope["focused_workflow_git_blob"]
        for scope in touched
    }
    observed_blobs: dict[str, str] = {}
    blob_errors: dict[str, str] = {}
    blob_mismatches: list[str] = []
    qualifier_blobs_verified = False

    if touched:
        if repository_root is None:
            reasons.append("head-qualifier-blob-unverified")
        else:
            observed_blobs, blob_errors = _observe_qualifier_blobs(
                touched, repository_root
            )
            if blob_errors:
                reasons.append("head-qualifier-blob-unavailable")
            blob_mismatches = sorted(
                workflow
                for workflow, expected in expected_blobs.items()
                if observed_blobs.get(workflow) is not None
                and observed_blobs[workflow] != expected
            )
            if blob_mismatches:
                reasons.append("head-qualifier-blob-drift")
            qualifier_blobs_verified = (
                not blob_errors
                and not blob_mismatches
                and observed_blobs == expected_blobs
            )

    global_ci_required = bool(reasons) or not touched

    required_workflows = sorted(
        {scope["focused_workflow"] for scope in touched}
    )
    if touched:
        required_workflows.append(registry["meta_qualifier_workflow"])
        required_workflows = sorted(set(required_workflows))

    return {
        "schema": "symthaea-focused-ci-admission-v1",
        "global_ci_required": global_ci_required,
        "global_ci_may_skip": not global_ci_required,
        "changed_paths": changed,
        "touched_scopes": [scope["id"] for scope in touched],
        "composition_groups": groups,
        "required_workflows": required_workflows,
        "unowned_paths": unowned,
        "control_plane_paths": control_plane,
        "changed_qualifier_workflows": changed_qualifiers,
        "expected_qualifier_blobs": expected_blobs,
        "head_observed_qualifier_blobs": observed_blobs,
        "head_qualifier_blob_errors": blob_errors,
        "head_qualifier_blob_mismatches": blob_mismatches,
        "head_qualifier_blobs_verified": qualifier_blobs_verified,
        "qualifier_blobs_verified": qualifier_blobs_verified,
        "reasons": sorted(set(reasons)),
        "scientific_authority": False,
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise RegistryError(f"JSON object: duplicate key: {key}")
        out[key] = value
    return out


def load_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--changed-paths-json", type=Path)
    parser.add_argument("--repository-root", type=Path)
    args = parser.parse_args(argv)

    try:
        registry = load_json(args.registry)
        if args.changed_paths_json is None:
            result: Any = validate_registry(registry)
        else:
            result = admit_changed_paths(
                registry,
                load_json(args.changed_paths_json),
                repository_root=args.repository_root,
            )
    except (OSError, json.JSONDecodeError, RegistryError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    if args.changed_paths_json is not None and result["global_ci_required"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
