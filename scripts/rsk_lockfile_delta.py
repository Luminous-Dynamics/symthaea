#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Fail-closed qualifier for RSK Cargo.lock repair candidates.

Cargo must generate the candidate on a pinned executor. This verifier permits
only the exact RSK workspace path-package additions plus a narrowly constrained
class of Cargo feature-unification dependency-edge additions on pre-existing
packages. Such an edge is admissible only when its target was already pinned in
the baseline lockfile, Cargo metadata contains the exact source->target edge,
and the source package is reachable from an RSK package in that resolved graph.

No package identity, version, source, checksum, dependency deletion, unrelated
edge addition, or new external package is admitted by this exception.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import re
import sys
import tomllib
from typing import Any

EXPECTED_RSK: dict[str, set[str]] = {
    "symthaea-replicator-semantics": set(),
    "symthaea-replicator-safety": {
        "symthaea-replicator-semantics",
        "sha2",
        "serde_json",
        "hex",
    },
    "symthaea-replicator-ledger": {"symthaea-replicator-safety"},
}
EXPECTED_VERSION = "0.1.0"


class LockfileDeltaError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "lockfile_delta_rejected",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse(data: bytes) -> dict[str, Any]:
    try:
        parsed = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise LockfileDeltaError(
            f"invalid Cargo.lock TOML: {exc}", code="invalid_lockfile"
        ) from exc
    if not isinstance(parsed.get("package"), list):
        raise LockfileDeltaError(
            "Cargo.lock must contain [[package]] records", code="invalid_lockfile"
        )
    return parsed


def _canonical_record(record: dict[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _dependency_name(raw: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise LockfileDeltaError(
            "package dependency entries must be non-empty strings",
            code="invalid_dependency_entry",
        )
    return raw.split(" ", 1)[0]


def _identity(record: dict[str, Any]) -> tuple[str, str, str | None]:
    name = record.get("name")
    version = record.get("version")
    source = record.get("source")
    if not isinstance(name, str) or not name:
        raise LockfileDeltaError("package name must be a non-empty string", code="invalid_package")
    if not isinstance(version, str) or not version:
        raise LockfileDeltaError(
            f"{name}: package version must be a non-empty string", code="invalid_package"
        )
    if source is not None and not isinstance(source, str):
        raise LockfileDeltaError(
            f"{name}: package source must be a string when present", code="invalid_package"
        )
    return name, version, source


def _identity_text(identity: tuple[str, str, str | None]) -> str:
    name, version, source = identity
    return f"{name} {version}" + (f" ({source})" if source is not None else "")


def _records_by_identity(
    packages: list[dict[str, Any]],
) -> dict[tuple[str, str, str | None], dict[str, Any]]:
    out: dict[tuple[str, str, str | None], dict[str, Any]] = {}
    for record in packages:
        ident = _identity(record)
        if ident in out:
            raise LockfileDeltaError(
                f"duplicate package identity: {_identity_text(ident)}",
                code="duplicate_package_identity",
            )
        out[ident] = record
    return out


def _records_by_name(
    packages: list[dict[str, Any]], names: set[str]
) -> dict[str, list[dict[str, Any]]]:
    out = {name: [] for name in names}
    for record in packages:
        name = record.get("name")
        if name in out:
            out[name].append(record)
    return out


def _validate_rsk_record(name: str, record: dict[str, Any]) -> None:
    if record.get("name") != name:
        raise LockfileDeltaError(f"{name}: record name mismatch")
    if record.get("version") != EXPECTED_VERSION:
        raise LockfileDeltaError(
            f"{name}: expected version {EXPECTED_VERSION}, got {record.get('version')!r}"
        )
    if "source" in record or "checksum" in record:
        raise LockfileDeltaError(
            f"{name}: workspace path package must not have source/checksum"
        )
    deps_raw = record.get("dependencies", [])
    if not isinstance(deps_raw, list):
        raise LockfileDeltaError(f"{name}: dependencies must be a list")
    deps = {_dependency_name(dep) for dep in deps_raw}
    if len(deps) != len(deps_raw):
        raise LockfileDeltaError(f"{name}: duplicate dependency names are not allowed")
    expected = EXPECTED_RSK[name]
    if deps != expected:
        raise LockfileDeltaError(
            f"{name}: dependency set mismatch; expected {sorted(expected)}, got {sorted(deps)}"
        )


_DEP_RE = re.compile(r"^(?P<name>\S+)(?: (?P<version>\S+))?(?: \((?P<source>.+)\))?$")


def _resolve_lock_dependency(
    raw: str,
    packages: dict[tuple[str, str, str | None], dict[str, Any]],
) -> tuple[str, str, str | None]:
    if not isinstance(raw, str) or not raw:
        raise LockfileDeltaError(
            "package dependency entries must be non-empty strings",
            code="invalid_dependency_entry",
        )
    match = _DEP_RE.fullmatch(raw)
    if match is None:
        raise LockfileDeltaError(
            f"cannot parse Cargo.lock dependency entry {raw!r}",
            code="ambiguous_dependency_target",
        )
    name = match.group("name")
    version = match.group("version")
    source = match.group("source")
    candidates = [
        ident
        for ident in packages
        if ident[0] == name
        and (version is None or ident[1] == version)
        and (source is None or ident[2] == source)
    ]
    if len(candidates) != 1:
        raise LockfileDeltaError(
            f"dependency entry {raw!r} resolves to {len(candidates)} baseline packages",
            code="ambiguous_dependency_target",
        )
    return candidates[0]


def _metadata_indexes(metadata: dict[str, Any]) -> tuple[
    dict[tuple[str, str, str | None], str],
    dict[str, tuple[str, str, str | None]],
    dict[str, list[str]],
]:
    packages = metadata.get("packages")
    resolve = metadata.get("resolve")
    if not isinstance(packages, list) or not isinstance(resolve, dict):
        raise LockfileDeltaError(
            "Cargo metadata must contain packages and resolve",
            code="invalid_cargo_metadata",
        )
    nodes = resolve.get("nodes")
    if not isinstance(nodes, list):
        raise LockfileDeltaError(
            "Cargo metadata resolve.nodes must be a list",
            code="invalid_cargo_metadata",
        )

    by_identity: dict[tuple[str, str, str | None], str] = {}
    by_id: dict[str, tuple[str, str, str | None]] = {}
    for package in packages:
        if not isinstance(package, dict):
            raise LockfileDeltaError(
                "Cargo metadata package entries must be objects",
                code="invalid_cargo_metadata",
            )
        package_id = package.get("id")
        name = package.get("name")
        version = package.get("version")
        source = package.get("source")
        if (
            not isinstance(package_id, str)
            or not isinstance(name, str)
            or not isinstance(version, str)
            or (source is not None and not isinstance(source, str))
        ):
            raise LockfileDeltaError(
                "Cargo metadata package identity is malformed",
                code="invalid_cargo_metadata",
            )
        ident = (name, version, source)
        if ident in by_identity or package_id in by_id:
            raise LockfileDeltaError(
                f"Cargo metadata contains duplicate identity for {_identity_text(ident)}",
                code="ambiguous_cargo_metadata",
            )
        by_identity[ident] = package_id
        by_id[package_id] = ident

    adjacency: dict[str, list[str]] = {package_id: [] for package_id in by_id}
    seen_nodes: set[str] = set()
    for node in nodes:
        if not isinstance(node, dict) or not isinstance(node.get("id"), str):
            raise LockfileDeltaError(
                "Cargo metadata resolve node is malformed",
                code="invalid_cargo_metadata",
            )
        source_id = node["id"]
        if source_id not in by_id:
            raise LockfileDeltaError(
                f"Cargo metadata resolve references unknown source id {source_id!r}",
                code="invalid_cargo_metadata",
            )
        if source_id in seen_nodes:
            raise LockfileDeltaError(
                f"Cargo metadata contains duplicate resolve node {source_id!r}",
                code="ambiguous_cargo_metadata",
            )
        seen_nodes.add(source_id)
        deps = node.get("deps")
        if not isinstance(deps, list):
            raise LockfileDeltaError(
                f"Cargo metadata node {source_id!r} has malformed deps",
                code="invalid_cargo_metadata",
            )
        targets: list[str] = []
        for dep in deps:
            if not isinstance(dep, dict) or not isinstance(dep.get("pkg"), str):
                raise LockfileDeltaError(
                    f"Cargo metadata node {source_id!r} has malformed dependency",
                    code="invalid_cargo_metadata",
                )
            target_id = dep["pkg"]
            if target_id not in by_id:
                raise LockfileDeltaError(
                    f"Cargo metadata dependency references unknown id {target_id!r}",
                    code="invalid_cargo_metadata",
                )
            targets.append(target_id)
        adjacency[source_id] = targets
    return by_identity, by_id, adjacency


def _proof_paths_from_rsk(
    metadata: dict[str, Any],
) -> tuple[
    dict[tuple[str, str, str | None], str],
    dict[str, tuple[str, str, str | None]],
    dict[str, list[str]],
    dict[str, list[str]],
]:
    by_identity, by_id, adjacency = _metadata_indexes(metadata)
    roots: list[str] = []
    for name in sorted(EXPECTED_RSK):
        ident = (name, EXPECTED_VERSION, None)
        package_id = by_identity.get(ident)
        if package_id is None:
            raise LockfileDeltaError(
                f"Cargo metadata is missing RSK root {_identity_text(ident)}",
                code="missing_rsk_metadata_root",
            )
        roots.append(package_id)

    paths: dict[str, list[str]] = {}
    queue: deque[str] = deque()
    for root in roots:
        if root not in paths:
            paths[root] = [root]
            queue.append(root)
    while queue:
        source = queue.popleft()
        for target in adjacency.get(source, []):
            if target not in paths:
                paths[target] = paths[source] + [target]
                queue.append(target)
    return by_identity, by_id, adjacency, paths


def _load_metadata(data: bytes | None) -> dict[str, Any] | None:
    if data is None:
        return None
    try:
        parsed = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LockfileDeltaError(
            f"invalid Cargo metadata JSON: {exc}", code="invalid_cargo_metadata"
        ) from exc
    if not isinstance(parsed, dict):
        raise LockfileDeltaError(
            "Cargo metadata must be a JSON object", code="invalid_cargo_metadata"
        )
    return parsed


def _classify_non_rsk_changes(
    old_packages: list[dict[str, Any]],
    new_packages: list[dict[str, Any]],
    metadata: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    names = set(EXPECTED_RSK)
    old_map = _records_by_identity(
        [record for record in old_packages if record.get("name") not in names]
    )
    new_map = _records_by_identity(
        [record for record in new_packages if record.get("name") not in names]
    )

    removed = sorted(set(old_map) - set(new_map), key=_identity_text)
    added = sorted(set(new_map) - set(old_map), key=_identity_text)
    if removed or added:
        raise LockfileDeltaError(
            "non-RSK package graph drift detected"
            + (f"; removed={len(removed)}" if removed else "")
            + (f"; added={len(added)}" if added else ""),
            code="non_rsk_package_identity_drift",
            details={
                "removed": [_identity_text(item) for item in removed],
                "added": [_identity_text(item) for item in added],
            },
        )

    changed = [
        ident
        for ident in sorted(old_map, key=_identity_text)
        if _canonical_record(old_map[ident]) != _canonical_record(new_map[ident])
    ]
    if not changed:
        return []

    prechecked: list[tuple[tuple[str, str, str | None], list[str]]] = []
    for ident in changed:
        old_record = old_map[ident]
        new_record = new_map[ident]
        old_without_deps = {k: v for k, v in old_record.items() if k != "dependencies"}
        new_without_deps = {k: v for k, v in new_record.items() if k != "dependencies"}
        if old_without_deps != new_without_deps:
            raise LockfileDeltaError(
                f"non-RSK package graph drift detected; {_identity_text(ident)} changed "
                "fields other than dependencies",
                code="non_rsk_package_record_mutation",
            )

        old_deps = old_record.get("dependencies", [])
        new_deps = new_record.get("dependencies", [])
        if not isinstance(old_deps, list) or not isinstance(new_deps, list):
            raise LockfileDeltaError(
                f"{_identity_text(ident)}: dependencies must be lists",
                code="invalid_dependency_entry",
            )
        if len(set(old_deps)) != len(old_deps) or len(set(new_deps)) != len(new_deps):
            raise LockfileDeltaError(
                f"{_identity_text(ident)}: duplicate dependency entries are not allowed",
                code="invalid_dependency_entry",
            )
        removed_deps = sorted(set(old_deps) - set(new_deps))
        added_deps = sorted(set(new_deps) - set(old_deps))
        if removed_deps:
            raise LockfileDeltaError(
                f"non-RSK package graph drift detected; {_identity_text(ident)} "
                f"removed dependency edges {removed_deps}",
                code="dependency_edge_removal",
            )
        if not added_deps:
            raise LockfileDeltaError(
                f"non-RSK package graph drift detected; {_identity_text(ident)} changed "
                "dependency ordering/representation without adding a proved edge",
                code="unexplained_dependency_record_change",
            )
        prechecked.append((ident, added_deps))

    if metadata is None:
        raise LockfileDeltaError(
            "non-RSK package graph drift detected; Cargo metadata proof is required "
            "for any pre-existing dependency-edge addition",
            code="cargo_metadata_required",
        )

    by_identity, by_id, adjacency, paths = _proof_paths_from_rsk(metadata)
    allowed: list[dict[str, Any]] = []

    for ident, added_deps in prechecked:
        source_id = by_identity.get(ident)
        if source_id is None:
            raise LockfileDeltaError(
                f"Cargo metadata does not contain {_identity_text(ident)}",
                code="metadata_identity_missing",
            )
        if source_id not in paths:
            raise LockfileDeltaError(
                f"{_identity_text(ident)} is not reachable from an RSK metadata root",
                code="edge_outside_rsk_closure",
            )

        for raw_dep in added_deps:
            target_ident = _resolve_lock_dependency(raw_dep, old_map)
            target_id = by_identity.get(target_ident)
            if target_id is None:
                raise LockfileDeltaError(
                    f"Cargo metadata does not contain already-pinned target "
                    f"{_identity_text(target_ident)}",
                    code="metadata_identity_missing",
                )
            if target_id not in adjacency.get(source_id, []):
                raise LockfileDeltaError(
                    f"Cargo metadata lacks edge {_identity_text(ident)} -> "
                    f"{_identity_text(target_ident)}",
                    code="edge_absent_from_metadata",
                )

            path_ids = paths[source_id] + [target_id]
            path = [_identity_text(by_id[package_id]) for package_id in path_ids]
            allowed.append(
                {
                    "source": _identity_text(ident),
                    "dependency_entry": raw_dep,
                    "target": _identity_text(target_ident),
                    "proof_path": path,
                    "target_preexisting_in_baseline": True,
                    "source_reachable_from_rsk_root": True,
                    "edge_present_in_cargo_metadata": True,
                }
            )

    return allowed


def qualify_lockfile_delta(
    before: bytes,
    after: bytes,
    cargo_metadata: bytes | None = None,
) -> dict[str, Any]:
    old = _parse(before)
    new = _parse(after)
    metadata = _load_metadata(cargo_metadata)

    if old.get("version") != new.get("version"):
        raise LockfileDeltaError(
            f"lockfile format version changed: {old.get('version')!r} -> {new.get('version')!r}",
            code="lockfile_version_changed",
        )
    if old.get("version") != 4:
        raise LockfileDeltaError(
            f"expected Cargo.lock version = 4, got {old.get('version')!r}",
            code="unsupported_lockfile_version",
        )

    old_top = {key: value for key, value in old.items() if key != "package"}
    new_top = {key: value for key, value in new.items() if key != "package"}
    if old_top != new_top:
        raise LockfileDeltaError(
            "non-package top-level Cargo.lock data changed",
            code="lockfile_top_level_changed",
        )

    names = set(EXPECTED_RSK)
    old_packages = old["package"]
    new_packages = new["package"]
    old_rsk = _records_by_name(old_packages, names)
    new_rsk = _records_by_name(new_packages, names)

    for name in sorted(names):
        if len(old_rsk[name]) > 1:
            raise LockfileDeltaError(
                f"{name}: before lock contains duplicate package records"
            )
        if len(new_rsk[name]) != 1:
            raise LockfileDeltaError(
                f"{name}: after lock must contain exactly one package record, got {len(new_rsk[name])}"
            )
        _validate_rsk_record(name, new_rsk[name][0])
        if old_rsk[name]:
            _validate_rsk_record(name, old_rsk[name][0])
            if _canonical_record(old_rsk[name][0]) != _canonical_record(new_rsk[name][0]):
                raise LockfileDeltaError(f"{name}: pre-existing RSK record changed")

    allowed_edges = _classify_non_rsk_changes(old_packages, new_packages, metadata)

    added_rsk = sorted(name for name in names if not old_rsk[name])
    status = "rsk-path-records-added" if added_rsk else "already-qualified-no-rsk-delta"
    if allowed_edges:
        status = (
            "rsk-path-records-and-proved-feature-edges-added"
            if added_rsk
            else "proved-feature-edges-added"
        )

    return {
        "schema": "symthaea.rsk.lockfile-delta-report.v2",
        "status": status,
        "before_sha256": _sha256(before),
        "after_sha256": _sha256(after),
        "lockfile_version": old["version"],
        "added_rsk_packages": added_rsk,
        "rsk_packages_after": {
            name: {
                "version": new_rsk[name][0]["version"],
                "dependencies": sorted(
                    _dependency_name(dep)
                    for dep in new_rsk[name][0].get("dependencies", [])
                ),
            }
            for name in sorted(names)
        },
        "non_rsk_package_identities_unchanged": True,
        "non_rsk_non_dependency_fields_unchanged": True,
        "dependency_edge_removals": [],
        "allowed_preexisting_dependency_edge_additions": allowed_edges,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--cargo-metadata")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)
    try:
        metadata = (
            Path(args.cargo_metadata).read_bytes()
            if args.cargo_metadata is not None
            else None
        )
        report = qualify_lockfile_delta(
            Path(args.before).read_bytes(),
            Path(args.after).read_bytes(),
            metadata,
        )
    except (OSError, LockfileDeltaError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    encoded = json.dumps(report, sort_keys=True, indent=2) + "\n"
    if args.json_out:
        Path(args.json_out).write_text(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
