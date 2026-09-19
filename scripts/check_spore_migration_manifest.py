#!/usr/bin/env python3
"""Validate the pre-extraction Spore migration manifest.

This checker intentionally uses only the Python standard library so the
migration boundary can be checked without adding a package-manager dependency.
The JSON Schema is the normative structural description; this script adds the
cross-record authority/provenance invariants JSON Schema alone does not express
concisely.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA_ID = "spore-migration-manifest-v1"
DESTINATION_REPOSITORY = "Luminous-Dynamics/spore"
SHA1_RE = re.compile(r"^[0-9a-f]{40}$")
SPORE_AUTHORITY_CLASSES = {"recovery", "qualification"}


class ManifestError(ValueError):
    """A migration manifest violates a Spore migration invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _require_sha1(value: Any, label: str) -> None:
    _require(
        isinstance(value, str) and SHA1_RE.fullmatch(value) is not None,
        f"{label} must be a 40-character lowercase Git object SHA-1",
    )


def validate_manifest(manifest: dict[str, Any]) -> None:
    _require(manifest.get("schema") == SCHEMA_ID, f"schema must be {SCHEMA_ID!r}")
    _require(
        manifest.get("destination_repository") == DESTINATION_REPOSITORY,
        f"destination_repository must be {DESTINATION_REPOSITORY!r}",
    )
    _require(
        manifest.get("destination_repository_status") in {"not-created", "created"},
        "destination_repository_status must be 'not-created' or 'created'",
    )
    _require(
        manifest.get("qualification_transfer_policy") == "never-inherit",
        "qualification_transfer_policy must be 'never-inherit'",
    )

    lineages = manifest.get("lineages")
    artifacts = manifest.get("artifacts")
    _require(isinstance(lineages, list) and lineages, "lineages must be a non-empty array")
    _require(isinstance(artifacts, list) and artifacts, "artifacts must be a non-empty array")

    lineage_by_id: dict[str, dict[str, Any]] = {}
    for index, lineage in enumerate(lineages):
        _require(isinstance(lineage, dict), f"lineages[{index}] must be an object")
        lineage_id = lineage.get("id")
        _require(
            isinstance(lineage_id, str) and lineage_id,
            f"lineages[{index}].id must be a non-empty string",
        )
        _require(lineage_id not in lineage_by_id, f"duplicate lineage id: {lineage_id}")
        _require_sha1(lineage.get("source_commit"), f"lineage {lineage_id} source_commit")
        _require_sha1(lineage.get("source_tree"), f"lineage {lineage_id} source_tree")

        verification = lineage.get("commit_verification")
        _require(
            verification in {"verified", "unsigned", "unknown"},
            f"lineage {lineage_id} has invalid commit_verification",
        )

        boundary = lineage.get("qualification_boundary")
        _require(
            boundary
            in {
                "exact-committed-source",
                "transformed-candidate",
                "provenance-only",
                "not-assessed",
            },
            f"lineage {lineage_id} has invalid qualification_boundary",
        )

        mutation = lineage.get("source_mutation_in_qualification")
        _require(
            isinstance(mutation, bool),
            f"lineage {lineage_id} source_mutation_in_qualification must be boolean",
        )
        if mutation:
            _require(
                boundary == "transformed-candidate",
                f"lineage {lineage_id} mutates source during qualification and therefore "
                "must be classified transformed-candidate",
            )
        if boundary == "exact-committed-source":
            _require(
                not mutation,
                f"lineage {lineage_id} cannot be exact-committed-source while mutating "
                "source during qualification",
            )

        lineage_by_id[lineage_id] = lineage

    artifact_ids: set[str] = set()
    source_keys: set[tuple[str, str]] = set()
    destination_not_created = manifest["destination_repository_status"] == "not-created"

    for index, artifact in enumerate(artifacts):
        _require(isinstance(artifact, dict), f"artifacts[{index}] must be an object")
        artifact_id = artifact.get("id")
        _require(
            isinstance(artifact_id, str) and artifact_id,
            f"artifacts[{index}].id must be a non-empty string",
        )
        _require(artifact_id not in artifact_ids, f"duplicate artifact id: {artifact_id}")
        artifact_ids.add(artifact_id)

        lineage_id = artifact.get("lineage_id")
        _require(
            lineage_id in lineage_by_id,
            f"artifact {artifact_id} references unknown lineage {lineage_id!r}",
        )

        source_path = artifact.get("source_path")
        _require(
            isinstance(source_path, str) and source_path,
            f"artifact {artifact_id} source_path must be non-empty",
        )
        source_key = (lineage_id, source_path)
        _require(
            source_key not in source_keys,
            f"duplicate source path in lineage {lineage_id}: {source_path}",
        )
        source_keys.add(source_key)
        _require_sha1(artifact.get("source_blob_sha1"), f"artifact {artifact_id} source_blob_sha1")

        authority_class = artifact.get("authority_class")
        target_owner = artifact.get("target_owner")
        if authority_class in SPORE_AUTHORITY_CLASSES:
            _require(
                target_owner == "spore",
                f"artifact {artifact_id} carries {authority_class} authority and therefore "
                "must target owner 'spore'",
            )

        _require(
            artifact.get("destination_qualification") == "required",
            f"artifact {artifact_id} must require fresh destination qualification",
        )

        destination_path = artifact.get("destination_path")
        if destination_not_created:
            _require(
                destination_path is None,
                f"artifact {artifact_id} cannot claim a destination path before the "
                "destination repository exists",
            )

    _require(
        all(a.get("destination_qualification") == "required" for a in artifacts),
        "historical source evidence must never transfer destination qualification",
    )


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ManifestError("manifest root must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path("docs/architecture/spore-migration-manifest-v1.json"),
    )
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        validate_manifest(manifest)
    except (OSError, json.JSONDecodeError, ManifestError) as error:
        print(f"SPORE_MIGRATION_MANIFEST_V1: FAIL: {error}", file=sys.stderr)
        return 1

    print(
        "SPORE_MIGRATION_MANIFEST_V1: PASS "
        f"({len(manifest['lineages'])} lineages, {len(manifest['artifacts'])} artifacts)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
