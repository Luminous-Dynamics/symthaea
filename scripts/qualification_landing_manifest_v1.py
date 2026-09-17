#!/usr/bin/env python3
"""Validate the QUAL-001A1 historical landing-source provenance manifest.

This validator checks only provenance-graph integrity. It does not import, execute,
or qualify any historical implementation and cannot transfer historical PASS.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "symthaea.qualification-landing-source-manifest.v1"
ERA = "QERA-001"
STATUSES = {
    "NormativeCandidate",
    "CompatibilityAdapter",
    "TestVector",
    "HistoricalOnly",
    "DomainOwned",
}
TOP_KEYS = {
    "schema",
    "qualification_era_target",
    "normative_framing_role",
    "external_primitives",
    "entries",
}
ENTRY_KEYS = {
    "logical_role",
    "source_branch",
    "source_commit_sha",
    "source_path",
    "source_blob_id",
    "semantic_schema_or_domain",
    "normative_status",
    "required_entry_roles",
    "required_external_primitives",
    "golden_vector_refs",
    "claim_ceiling",
    "migration_or_equivalence_ref",
    "notes",
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
ROLE = re.compile(r"^[a-z][a-z0-9_]*$")
RECEIPT_V1_DOMAIN = "symthaea.qualification-receipt-core.v1"


class ManifestError(ValueError):
    """Landing manifest is malformed or violates convergence invariants."""


def _exact_keys(value: dict[str, Any], expected: set[str], where: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ManifestError(
            f"{where}: schema mismatch missing={sorted(expected-observed)} "
            f"unknown={sorted(observed-expected)}"
        )


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{where}: expected non-empty string")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7f for ch in value):
        raise ManifestError(f"{where}: control characters are forbidden")
    return value


def _sorted_unique_strings(value: Any, where: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ManifestError(f"{where}: expected array of non-empty strings")
    if value != sorted(value) or len(value) != len(set(value)):
        raise ManifestError(f"{where}: must be sorted and unique")
    if not allow_empty and not value:
        raise ManifestError(f"{where}: must not be empty")
    return value


def _canonical_repo_path(value: Any, where: str) -> str:
    text = _text(value, where)
    if text.startswith("/") or text.endswith("/") or "\\" in text:
        raise ManifestError(f"{where}: expected canonical repository-relative POSIX path")
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ManifestError(f"{where}: path contains non-canonical/traversal segment")
    return text


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise ManifestError(f"git {' '.join(args)} failed: {detail}")
    return proc.stdout.strip()


def _git_exists(repo: Path, *args: str) -> bool:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.returncode == 0


def _verify_source(repo: Path, entry: dict[str, Any]) -> None:
    role = entry["logical_role"]
    commit = entry["source_commit_sha"]
    path = entry["source_path"]
    expected_blob = entry["source_blob_id"]

    actual_type = _git(repo, "cat-file", "-t", commit)
    if actual_type != "commit":
        raise ManifestError(f"{role}: source object is not a commit: {actual_type}")

    actual_blob = _git(repo, "rev-parse", f"{commit}:{path}")
    if actual_blob != expected_blob:
        raise ManifestError(
            f"{role}: source blob mismatch expected={expected_blob} actual={actual_blob}"
        )
    blob_type = _git(repo, "cat-file", "-t", actual_blob)
    if blob_type != "blob":
        raise ManifestError(f"{role}: source path did not resolve to a blob")


def _local_python_import_paths(repo: Path, entry: dict[str, Any]) -> set[str]:
    source_path = entry["source_path"]
    if not source_path.startswith("scripts/") or not source_path.endswith(".py"):
        return set()

    commit = entry["source_commit_sha"]
    raw = _git(repo, "show", f"{commit}:{source_path}")
    try:
        tree = ast.parse(raw, filename=source_path)
    except SyntaxError as error:
        raise ManifestError(
            f"{entry['logical_role']}: historical Python source does not parse: {error}"
        ) from error

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module.split(".", 1)[0])

    local_paths: set[str] = set()
    for module in modules:
        candidate = f"scripts/{module}.py"
        if _git_exists(repo, "cat-file", "-e", f"{commit}:{candidate}"):
            local_paths.add(candidate)
    return local_paths


def _verify_import_closure(repo: Path, entries: dict[str, dict[str, Any]]) -> None:
    path_to_roles: dict[str, list[str]] = {}
    for role, item in entries.items():
        path_to_roles.setdefault(item["source_path"], []).append(role)

    for role, item in entries.items():
        if item["normative_status"] == "TestVector":
            continue
        if not item["source_path"].startswith("scripts/") or not item["source_path"].endswith(".py"):
            continue

        discovered_roles: set[str] = set()
        for dependency_path in sorted(_local_python_import_paths(repo, item)):
            candidates = sorted(path_to_roles.get(dependency_path, []))
            if not candidates:
                raise ManifestError(
                    f"{role}: unrepresented local import {dependency_path}"
                )
            if len(candidates) != 1:
                raise ManifestError(
                    f"{role}: ambiguous local import {dependency_path}: roles={candidates}"
                )
            dependency_role = candidates[0]
            dependency = entries[dependency_role]
            actual_dependency_blob = _git(
                repo,
                "rev-parse",
                f"{item['source_commit_sha']}:{dependency_path}",
            )
            if actual_dependency_blob != dependency["source_blob_id"]:
                raise ManifestError(
                    f"{role}: dependency blob version skew for {dependency_role} "
                    f"at {dependency_path}: importer saw {actual_dependency_blob}, "
                    f"manifest selected {dependency['source_blob_id']}"
                )
            discovered_roles.add(dependency_role)

        declared_roles = set(item["required_entry_roles"])
        if discovered_roles != declared_roles:
            missing = sorted(discovered_roles - declared_roles)
            extra = sorted(declared_roles - discovered_roles)
            raise ManifestError(
                f"{role}: required_entry_roles must equal direct local Python imports "
                f"(missing={missing}, extra={extra})"
            )


def _reject_cycles(entries: dict[str, dict[str, Any]]) -> None:
    visiting: set[str] = set()
    done: set[str] = set()

    def visit(role: str) -> None:
        if role in done:
            return
        if role in visiting:
            raise ManifestError(f"dependency cycle includes {role}")
        visiting.add(role)
        for dep in entries[role]["required_entry_roles"]:
            visit(dep)
        visiting.remove(role)
        done.add(role)

    for role in sorted(entries):
        visit(role)


def validate_manifest(value: Any, *, repo: Path | None = None, verify_git: bool = True) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError("manifest: expected object")
    _exact_keys(value, TOP_KEYS, "manifest")
    if value["schema"] != SCHEMA:
        raise ManifestError(f"manifest.schema: expected {SCHEMA}")
    if value["qualification_era_target"] != ERA:
        raise ManifestError(f"manifest.qualification_era_target: expected {ERA}")

    external = _sorted_unique_strings(
        value["external_primitives"], "manifest.external_primitives", allow_empty=False
    )

    raw_entries = value["entries"]
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ManifestError("manifest.entries: expected non-empty array")

    roles: list[str] = []
    normalized: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(raw_entries):
        where = f"manifest.entries[{index}]"
        if not isinstance(item, dict):
            raise ManifestError(f"{where}: expected object")
        _exact_keys(item, ENTRY_KEYS, where)

        role = _text(item["logical_role"], f"{where}.logical_role")
        if ROLE.fullmatch(role) is None:
            raise ManifestError(f"{where}.logical_role: invalid role spelling")
        if role in normalized:
            raise ManifestError(f"manifest.entries: duplicate logical_role {role}")
        roles.append(role)

        commit = _text(item["source_commit_sha"], f"{where}.source_commit_sha")
        blob = _text(item["source_blob_id"], f"{where}.source_blob_id")
        if HEX40.fullmatch(commit) is None:
            raise ManifestError(f"{where}.source_commit_sha: expected 40 lowercase hex")
        if HEX40.fullmatch(blob) is None:
            raise ManifestError(f"{where}.source_blob_id: expected Git SHA-1 blob id")

        status = item["normative_status"]
        if status not in STATUSES:
            raise ManifestError(f"{where}.normative_status: unsupported {status!r}")

        migration = item["migration_or_equivalence_ref"]
        if migration is not None:
            _text(migration, f"{where}.migration_or_equivalence_ref")
        if status in {"CompatibilityAdapter", "HistoricalOnly"} and migration is None:
            raise ManifestError(f"{role}: {status} requires migration_or_equivalence_ref")

        normalized[role] = {
            **item,
            "source_branch": _text(item["source_branch"], f"{where}.source_branch"),
            "source_path": _canonical_repo_path(item["source_path"], f"{where}.source_path"),
            "semantic_schema_or_domain": _text(
                item["semantic_schema_or_domain"], f"{where}.semantic_schema_or_domain"
            ),
            "required_entry_roles": _sorted_unique_strings(
                item["required_entry_roles"], f"{where}.required_entry_roles"
            ),
            "required_external_primitives": _sorted_unique_strings(
                item["required_external_primitives"], f"{where}.required_external_primitives"
            ),
            "golden_vector_refs": _sorted_unique_strings(
                item["golden_vector_refs"], f"{where}.golden_vector_refs"
            ),
            "claim_ceiling": _sorted_unique_strings(
                item["claim_ceiling"], f"{where}.claim_ceiling", allow_empty=False
            ),
            "notes": _sorted_unique_strings(item["notes"], f"{where}.notes"),
        }

    if roles != sorted(roles):
        raise ManifestError("manifest.entries: entries must be sorted by logical_role")

    framing_role = _text(value["normative_framing_role"], "manifest.normative_framing_role")
    if framing_role not in normalized:
        raise ManifestError("normative_framing_role does not name an entry")
    framing = normalized[framing_role]
    if framing["normative_status"] != "NormativeCandidate":
        raise ManifestError("normative framing role must be NormativeCandidate")

    framing_candidates = [
        role
        for role, item in normalized.items()
        if item["normative_status"] == "NormativeCandidate"
        and "qualification-framing" in item["semantic_schema_or_domain"]
    ]
    if framing_candidates != [framing_role]:
        raise ManifestError(
            f"exactly one normative framing candidate required; found {framing_candidates}"
        )

    for role, item in normalized.items():
        for dep in item["required_entry_roles"]:
            if dep not in normalized:
                raise ManifestError(f"{role}: missing required entry role {dep}")
            if normalized[dep]["normative_status"] == "DomainOwned":
                raise ManifestError(f"{role}: generic entry may not depend on DomainOwned role {dep}")
        for ext in item["required_external_primitives"]:
            if ext not in external:
                raise ManifestError(f"{role}: undeclared external primitive {ext}")
        for ref in item["golden_vector_refs"]:
            if ref not in normalized:
                raise ManifestError(f"{role}: missing golden-vector role {ref}")
            if normalized[ref]["normative_status"] != "TestVector":
                raise ManifestError(f"{role}: golden-vector ref {ref} is not TestVector")

        if item["normative_status"] == "NormativeCandidate" and not item["golden_vector_refs"]:
            raise ManifestError(f"{role}: normative identity-bearing candidate requires golden vectors")
        if (
            item["semantic_schema_or_domain"] == RECEIPT_V1_DOMAIN
            and item["normative_status"] == "NormativeCandidate"
        ):
            raise ManifestError(
                f"{role}: historical receipt-core V1 cannot be normative without #2469 migration"
            )

    _reject_cycles(normalized)

    if verify_git:
        if repo is None:
            raise ManifestError("repository path required when verify_git=true")
        repo = repo.resolve()
        if not (repo / ".git").exists() and _git(repo, "rev-parse", "--is-inside-work-tree") != "true":
            raise ManifestError(f"not a Git worktree: {repo}")
        for role in sorted(normalized):
            _verify_source(repo, normalized[role])
        _verify_import_closure(repo, normalized)

    return {
        "schema": SCHEMA,
        "qualification_era_target": ERA,
        "normative_framing_role": framing_role,
        "external_primitives": external,
        "entries": [normalized[role] for role in sorted(normalized)],
    }


def canonical_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument(
        "--no-git-verify",
        action="store_true",
        help="validate schema/graph only; do not resolve historical Git objects",
    )
    args = parser.parse_args(argv)

    try:
        value = json.loads(args.manifest.read_text(encoding="utf-8"))
        normalized = validate_manifest(value, repo=args.repo, verify_git=not args.no_git_verify)
    except (OSError, json.JSONDecodeError, ManifestError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    digest = hashlib.sha256(canonical_bytes(normalized)).hexdigest()
    print("PASS: qualification landing source manifest is structurally consistent")
    print(f"entries: {len(normalized['entries'])}")
    print(f"manifest_sha256: {digest}")
    print("qualification_authority: NOT ESTABLISHED")
    print("pass_transfer: FORBIDDEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
