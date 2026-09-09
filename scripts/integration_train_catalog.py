#!/usr/bin/env python3
"""Validate content-addressed integration-train catalogs and their graph composition."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any

import integration_train_manifest as train

SCHEMA = "symthaea.integration-train-catalog.v1"
DOMAIN = b"symthaea.integration-train-catalog.v1\0"
MAX_TRAINS = 64
MAX_EDGES = 256
_ALLOWED_RELATIONS = {"linear_successor", "parallel_successor"}
_ALLOWED_STATUS = "SourceManifestOnly"
_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def _require_exact_keys(value: dict[str, Any], required: set[str], optional: set[str], *, where: str) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise train.TrainManifestError(f"{where}: missing fields: {', '.join(missing)}")
    if unknown:
        raise train.TrainManifestError(f"{where}: unknown fields: {', '.join(unknown)}")


def _require_string(value: Any, *, where: str) -> str:
    if not isinstance(value, str):
        raise train.TrainManifestError(f"{where}: expected string")
    if value != value.strip():
        raise train.TrainManifestError(f"{where}: leading/trailing whitespace is not canonical")
    if not value:
        raise train.TrainManifestError(f"{where}: must not be empty")
    if unicodedata.normalize("NFC", value) != value:
        raise train.TrainManifestError(f"{where}: text must use Unicode NFC normalization")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise train.TrainManifestError(f"{where}: control characters are not canonical")
    if len(value.encode("utf-8")) > train.MAX_TEXT_BYTES:
        raise train.TrainManifestError(f"{where}: exceeds {train.MAX_TEXT_BYTES} UTF-8 bytes")
    return value


def _require_id(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise train.TrainManifestError(f"{where}: expected sha256:<64 lowercase hex>")
    return value


def _require_manifest_path(value: Any, *, where: str) -> str:
    text = _require_string(value, where=where)
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or text in {".", ".."} or path.as_posix() != text or "\\" in text or not text.endswith(".json"):
        raise train.TrainManifestError(f"{where}: non-canonical JSON repository path {text!r}")
    return text


def _require_branch(value: Any, *, where: str) -> str:
    text = _require_string(value, where=where)
    if text.startswith("/") or text.endswith("/") or "//" in text or ".." in text or "@{" in text or "\\" in text or text.endswith(".lock"):
        raise train.TrainManifestError(f"{where}: non-canonical Git branch name {text!r}")
    return text


def _canonical_payload_from_normalized(catalog: dict[str, Any]) -> bytes:
    payload = {key: value for key, value in catalog.items() if key != "catalog_id"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _compute_catalog_id_from_normalized(catalog: dict[str, Any]) -> str:
    digest = hashlib.sha256(DOMAIN + _canonical_payload_from_normalized(catalog)).hexdigest()
    return f"sha256:{digest}"


def normalize_catalog(catalog: Any, *, verify_declared_id: bool = True, require_id: bool = False) -> dict[str, Any]:
    if not isinstance(catalog, dict):
        raise train.TrainManifestError("catalog: expected object")
    _require_exact_keys(catalog, {"schema", "program_id", "trains", "edges", "non_claims"}, {"catalog_id"}, where="catalog")
    if catalog["schema"] != SCHEMA:
        raise train.TrainManifestError(f"catalog.schema: expected {SCHEMA!r}")
    if require_id and "catalog_id" not in catalog:
        raise train.TrainManifestError("catalog.catalog_id: required but absent")

    program_id = _require_string(catalog["program_id"], where="catalog.program_id")
    non_claims = train._require_sorted_unique_strings(catalog["non_claims"], where="catalog.non_claims")
    if not non_claims:
        raise train.TrainManifestError("catalog.non_claims: must contain at least one explicit non-claim")

    raw_trains = catalog["trains"]
    if not isinstance(raw_trains, list) or not raw_trains:
        raise train.TrainManifestError("catalog.trains: expected non-empty list")
    if len(raw_trains) > MAX_TRAINS:
        raise train.TrainManifestError(f"catalog.trains: exceeds V1 bound of {MAX_TRAINS}")

    trains: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    seen_ids: set[str] = set()
    seen_locations: set[tuple[str, str]] = set()

    for index, raw in enumerate(raw_trains):
        where = f"catalog.trains[{index}]"
        if not isinstance(raw, dict):
            raise train.TrainManifestError(f"{where}: expected object")
        _require_exact_keys(raw, {"name", "train_id", "manifest_branch", "manifest_path", "base_subject", "cumulative_tip_sha", "role", "status"}, set(), where=where)
        name = _require_string(raw["name"], where=f"{where}.name")
        train_id = _require_id(raw["train_id"], where=f"{where}.train_id")
        branch = _require_branch(raw["manifest_branch"], where=f"{where}.manifest_branch")
        path = _require_manifest_path(raw["manifest_path"], where=f"{where}.manifest_path")
        base_subject = train._require_sha(raw["base_subject"], where=f"{where}.base_subject")
        tip = train._require_sha(raw["cumulative_tip_sha"], where=f"{where}.cumulative_tip_sha")
        role = _require_string(raw["role"], where=f"{where}.role")
        status = _require_string(raw["status"], where=f"{where}.status")
        if status != _ALLOWED_STATUS:
            raise train.TrainManifestError(f"{where}.status: V1 requires {_ALLOWED_STATUS!r}")
        if name in seen_names:
            raise train.TrainManifestError(f"{where}.name: duplicate train {name!r}")
        if train_id in seen_ids:
            raise train.TrainManifestError(f"{where}.train_id: duplicate train identity {train_id}")
        location = (branch, path)
        if location in seen_locations:
            raise train.TrainManifestError(f"{where}: duplicate manifest branch/path {branch}:{path}")
        trains.append({"name": name, "train_id": train_id, "manifest_branch": branch, "manifest_path": path, "base_subject": base_subject, "cumulative_tip_sha": tip, "role": role, "status": status})
        seen_names.add(name)
        seen_ids.add(train_id)
        seen_locations.add(location)

    if [item["name"] for item in trains] != sorted(item["name"] for item in trains):
        raise train.TrainManifestError("catalog.trains: must be lexicographically sorted by name")

    by_name = {item["name"]: item for item in trains}
    raw_edges = catalog["edges"]
    if not isinstance(raw_edges, list):
        raise train.TrainManifestError("catalog.edges: expected list")
    if len(raw_edges) > MAX_EDGES:
        raise train.TrainManifestError(f"catalog.edges: exceeds V1 bound of {MAX_EDGES}")

    edges: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    linear_in: dict[str, int] = {}
    linear_out: dict[str, int] = {}

    for index, raw in enumerate(raw_edges):
        where = f"catalog.edges[{index}]"
        if not isinstance(raw, dict):
            raise train.TrainManifestError(f"{where}: expected object")
        _require_exact_keys(raw, {"from", "to", "relation", "boundary_sha"}, set(), where=where)
        source = _require_string(raw["from"], where=f"{where}.from")
        target = _require_string(raw["to"], where=f"{where}.to")
        relation = _require_string(raw["relation"], where=f"{where}.relation")
        boundary = train._require_sha(raw["boundary_sha"], where=f"{where}.boundary_sha")
        if source == target:
            raise train.TrainManifestError(f"{where}: self edge is not allowed")
        if source not in by_name or target not in by_name:
            raise train.TrainManifestError(f"{where}: edge references unknown train {source!r}->{target!r}")
        if relation not in _ALLOWED_RELATIONS:
            raise train.TrainManifestError(f"{where}.relation: expected one of {sorted(_ALLOWED_RELATIONS)}")
        pair = (source, target)
        if pair in seen_pairs:
            raise train.TrainManifestError(f"{where}: duplicate edge {source!r}->{target!r}")
        if boundary != by_name[source]["cumulative_tip_sha"]:
            raise train.TrainManifestError(f"{where}.boundary_sha: does not equal {source}.cumulative_tip_sha")
        if boundary != by_name[target]["base_subject"]:
            raise train.TrainManifestError(f"{where}.boundary_sha: does not equal {target}.base_subject")
        if relation == "linear_successor":
            linear_out[source] = linear_out.get(source, 0) + 1
            linear_in[target] = linear_in.get(target, 0) + 1
            if linear_out[source] > 1:
                raise train.TrainManifestError(f"{where}: train {source!r} has multiple linear successors")
            if linear_in[target] > 1:
                raise train.TrainManifestError(f"{where}: train {target!r} has multiple linear predecessors")
        edges.append({"from": source, "to": target, "relation": relation, "boundary_sha": boundary})
        seen_pairs.add(pair)

    edge_keys = [(edge["from"], edge["to"], edge["relation"], edge["boundary_sha"]) for edge in edges]
    if edge_keys != sorted(edge_keys):
        raise train.TrainManifestError("catalog.edges: must be lexicographically sorted")

    adjacency: dict[str, set[str]] = {name: set() for name in by_name}
    for edge in edges:
        adjacency[edge["from"]].add(edge["to"])
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise train.TrainManifestError(f"catalog.edges: cycle detected through train {node!r}")
        if node in visited:
            return
        visiting.add(node)
        for nxt in sorted(adjacency[node]):
            visit(nxt)
        visiting.remove(node)
        visited.add(node)

    for name in sorted(by_name):
        visit(name)

    outgoing: dict[str, list[str]] = {}
    for edge in edges:
        outgoing.setdefault(edge["from"], []).append(edge["to"])
    edge_pairs = {(edge["from"], edge["to"]) for edge in edges}
    for source, siblings in outgoing.items():
        if len(siblings) < 2:
            continue
        for left in siblings:
            for right in siblings:
                if left >= right:
                    continue
                if (left, right) in edge_pairs or (right, left) in edge_pairs:
                    raise train.TrainManifestError(f"catalog.edges: sibling successors from {source!r} cannot also be directly linearized ({left!r}, {right!r}) in V1")

    normalized: dict[str, Any] = {"schema": SCHEMA, "program_id": program_id, "trains": trains, "edges": edges, "non_claims": non_claims}
    catalog_id = _compute_catalog_id_from_normalized(normalized)
    normalized["catalog_id"] = catalog_id
    if verify_declared_id and "catalog_id" in catalog:
        declared = _require_id(catalog["catalog_id"], where="catalog.catalog_id")
        if declared != catalog_id:
            raise train.TrainManifestError(f"catalog.catalog_id: expected {catalog_id}, got {declared}")
    return normalized


def compute_catalog_id(catalog: Any) -> str:
    normalized = normalize_catalog(catalog, verify_declared_id=False)
    return _compute_catalog_id_from_normalized(normalized)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise train.TrainManifestError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def load_catalog(path: Path, *, require_id: bool = False) -> dict[str, Any]:
    try:
        if path.stat().st_size > train.MAX_MANIFEST_BYTES:
            raise train.TrainManifestError(f"{path}: catalog exceeds {train.MAX_MANIFEST_BYTES} bytes")
        raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object_without_duplicate_keys)
    except train.TrainManifestError:
        raise
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise train.TrainManifestError(f"{path}: {error}") from error
    return normalize_catalog(raw, require_id=require_id)


def _run_git(repo: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=repo, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "git command failed"
        raise train.TrainManifestError(f"git {' '.join(args)}: {detail}")
    return result.stdout


def _require_git_repo(repo: Path) -> Path:
    repo = repo.resolve()
    result = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo, check=False, capture_output=True, text=True)
    if result.returncode != 0 or result.stdout.strip() != "true":
        raise train.TrainManifestError(f"repository is not a Git work tree: {repo}")
    return repo


def validate_git_bindings(catalog: Any, repo: Path) -> None:
    normalized = normalize_catalog(catalog)
    repo = _require_git_repo(repo)
    for entry in normalized["trains"]:
        spec = f"{entry['manifest_branch']}:{entry['manifest_path']}"
        raw_text = _run_git(repo, "show", spec)
        try:
            raw = json.loads(raw_text, object_pairs_hook=train._object_without_duplicate_keys)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise train.TrainManifestError(f"{entry['name']}: invalid manifest JSON at {spec}: {error}") from error
        manifest = train.normalize_manifest(raw, require_id=True)
        if manifest["train_id"] != entry["train_id"]:
            raise train.TrainManifestError(f"{entry['name']}: catalog train_id does not match {spec}")
        if manifest["base_subject"] != entry["base_subject"]:
            raise train.TrainManifestError(f"{entry['name']}: catalog base_subject does not match {spec}")
        if manifest["cumulative_tip_sha"] != entry["cumulative_tip_sha"]:
            raise train.TrainManifestError(f"{entry['name']}: catalog cumulative_tip_sha does not match {spec}")
        train.validate_git_chain(manifest, repo)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("catalog", type=Path, help="integration-train catalog JSON")
    parser.add_argument("--require-id", action="store_true", help="require the exact computed catalog_id")
    parser.add_argument("--verify-git", action="store_true", help="resolve every branch/path and verify train manifests and Git chains")
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Git work tree used by --verify-git")
    parser.add_argument("--print-normalized", action="store_true", help="print canonical normalized JSON including catalog_id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        normalized = load_catalog(args.catalog, require_id=args.require_id)
        if args.verify_git:
            validate_git_bindings(normalized, args.repo)
    except train.TrainManifestError as error:
        print(f"integration-train catalog invalid: {error}", file=sys.stderr)
        return 2
    if args.print_normalized:
        print(json.dumps(normalized, indent=2, ensure_ascii=False))
    else:
        print(normalized["catalog_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
