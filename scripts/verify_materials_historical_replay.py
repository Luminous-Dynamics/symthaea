#!/usr/bin/env python3
"""Fail-closed verifier for the materials historical replay manifest.

This verifies only the manifest's internal historical/replay-preimage contract.
It does not inspect GitHub, run rustfmt, prove semantic equivalence, or qualify source.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA = "symthaea.materials-historical-replay-manifest.v1"
REPOSITORY = "Luminous-Dynamics/symthaea"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
ALLOWED_KINDS = {
    "CommonScientificKernel",
    "PlanningLayer",
    "ExecutionAdapter",
    "ExtractedGenericRepair",
}
ALLOWED_NEGATIVE_DISPOSITIONS = {
    "FormattingFailed",
    "CompileFailed",
    "TestsFailed",
    "ClippyFailed",
    "LockStale",
    "Cancelled",
    "InfrastructureUnavailable",
    "OutcomeUnknown",
}
EXPECTED_COMMON = ["MAT-006", "MAT-007", "MAT-008", "MAT-009", "MAT-010", "MAT-011"]
EXPECTED_PROVIDER = ["MAT-012", "MAT-013", "MAT-014", "MAT-015"]
EXPECTED_AUTHORITY = ["MAT-011B", "MAT-016A", "MAT-016B"]
EXPECTED_EXTRACTIONS = ["MAT-008B-0K"]
ZERO_K_PATH = "crates/domains/symthaea-materials/src/conditioned_property.rs"
ZERO_K_PRE = "8029f9583a7dcfc6c1f144fe68f6215abb1caa0b"
ZERO_K_POST = "8a3f6ff14ad1e1a689f593caa5ae5d25d5326953"


class ManifestError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def sha40(value: Any, field: str) -> str:
    require(isinstance(value, str) and HEX40.fullmatch(value) is not None, f"{field}: expected lowercase 40-hex SHA")
    return value


def reject_authority_shortcuts(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            require(lowered not in {"qualified", "passed", "is_qualified", "is_passed"}, f"{path}.{key}: authority shortcut field forbidden")
            reject_authority_shortcuts(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_authority_shortcuts(child, f"{path}[{index}]")


def validate_paths(edge: dict[str, Any]) -> set[str]:
    edge_id = edge["edge_id"]
    paths = edge.get("changed_paths")
    require(isinstance(paths, list) and paths, f"{edge_id}: changed_paths must be non-empty list")
    require(all(isinstance(path, str) and path and not path.startswith("/") for path in paths), f"{edge_id}: invalid repository-relative path")
    require(paths == sorted(set(paths)), f"{edge_id}: changed_paths must be sorted and unique")
    return set(paths)


def validate_blobs(edge: dict[str, Any], paths: set[str]) -> None:
    blobs = edge.get("historical_blobs", {})
    require(isinstance(blobs, dict), f"{edge['edge_id']}: historical_blobs must be object")
    for path, transition in blobs.items():
        require(path in paths, f"{edge['edge_id']}: blob transition for undeclared path {path}")
        require(isinstance(transition, dict), f"{edge['edge_id']}:{path}: transition must be object")
        for side in ("pre", "post"):
            value = transition.get(side)
            if value is not None:
                sha40(value, f"{edge['edge_id']}:{path}:{side}")
        require(transition.get("pre") is not None or transition.get("post") is not None, f"{edge['edge_id']}:{path}: both pre and post cannot be null")
        require(transition.get("pre") != transition.get("post"), f"{edge['edge_id']}:{path}: pre/post blobs must differ")


def validate_known_qualification(edge: dict[str, Any]) -> None:
    q = edge.get("known_qualification")
    if q is None:
        return
    require(isinstance(q, dict), f"{edge['edge_id']}: known_qualification must be object")
    run_id = q.get("run_id")
    require(isinstance(run_id, int) and run_id > 0, f"{edge['edge_id']}: invalid run_id")
    disposition = q.get("disposition")
    require(disposition in ALLOWED_NEGATIVE_DISPOSITIONS, f"{edge['edge_id']}: only explicit non-PASS dispositions belong in historical preimage manifest")
    require(q.get("later_executable_gates") != "Passed", f"{edge['edge_id']}: historical failure cannot claim later executable PASS")


def chain(edges: dict[str, dict[str, Any]], ids: list[str], expected_parent: str | None = None) -> None:
    previous_child = expected_parent
    for edge_id in ids:
        edge = edges[edge_id]
        if previous_child is not None:
            require(edge["parent_sha"] == previous_child, f"{edge_id}: lineage parent mismatch")
        previous_child = edge["child_sha"]


def validate_manifest(data: dict[str, Any]) -> None:
    reject_authority_shortcuts(data)
    require(data.get("schema") == SCHEMA, "unsupported schema")
    require(data.get("repository") == REPOSITORY, "repository mismatch")
    require(data.get("manifest_role") == "historical_preimage_only", "manifest role must remain historical_preimage_only")
    require(data.get("authority") == "none", "historical manifest must grant no authority")
    require(isinstance(data.get("claim_ceiling"), str) and data["claim_ceiling"].strip(), "claim_ceiling required")

    formatter = data.get("formatter_profile")
    require(isinstance(formatter, dict), "formatter_profile required")
    require(formatter.get("channel") == "1.96.0", "formatter channel drift")
    require(formatter.get("required_components") == ["clippy", "rustfmt"], "formatter component set/order drift")
    require(formatter.get("normalization_evidence") == "NOT_DERIVED", "v1 historical preimage manifest must not claim normalization evidence")

    policy = data.get("replay_policy")
    require(isinstance(policy, dict), "replay_policy required")
    require(policy.get("operator") == "normalize_parent_and_child_then_project_normalized_delta", "unexpected replay operator")
    require(policy.get("patch_fuzz_allowed") is False, "patch fuzz must be forbidden")
    require(policy.get("silent_three_way_resolution_allowed") is False, "silent three-way resolution must be forbidden")
    require(policy.get("manual_conflict_resolution_is_mechanical_replay") is False, "manual conflict resolution cannot be mechanical replay")
    require(policy.get("registered_failed_subjects_are_mutable_repair_targets") is False, "failed registered subjects must remain immutable")

    lineages = data.get("lineages")
    require(isinstance(lineages, dict), "lineages required")
    require(lineages.get("common_spine") == EXPECTED_COMMON, "common_spine inventory/order drift")
    require(lineages.get("provenance_provider_fork") == EXPECTED_PROVIDER, "provider fork inventory/order drift")
    require(lineages.get("authority_fork") == EXPECTED_AUTHORITY, "authority fork inventory/order drift")
    require(lineages.get("generic_extractions") == EXPECTED_EXTRACTIONS, "generic extraction inventory/order drift")

    raw_edges = data.get("edges")
    require(isinstance(raw_edges, list) and raw_edges, "edges must be non-empty list")
    edges: dict[str, dict[str, Any]] = {}
    child_shas: set[str] = set()
    for edge in raw_edges:
        require(isinstance(edge, dict), "each edge must be object")
        edge_id = edge.get("edge_id")
        require(isinstance(edge_id, str) and edge_id, "edge_id required")
        require(edge_id not in edges, f"duplicate edge_id {edge_id}")
        require(isinstance(edge.get("historical_pr"), int) and edge["historical_pr"] > 0, f"{edge_id}: invalid historical_pr")
        parent = sha40(edge.get("parent_sha"), f"{edge_id}.parent_sha")
        child = sha40(edge.get("child_sha"), f"{edge_id}.child_sha")
        require(parent != child, f"{edge_id}: parent == child")
        require(child not in child_shas, f"{edge_id}: duplicate historical child SHA")
        child_shas.add(child)
        require(edge.get("kind") in ALLOWED_KINDS, f"{edge_id}: unsupported kind")
        paths = validate_paths(edge)
        validate_blobs(edge, paths)
        validate_known_qualification(edge)
        edges[edge_id] = edge

    expected_ids = set(EXPECTED_COMMON + EXPECTED_PROVIDER + EXPECTED_AUTHORITY + EXPECTED_EXTRACTIONS)
    require(set(edges) == expected_ids, "edge inventory does not exactly match declared lineages")

    chain(edges, EXPECTED_COMMON)
    require(edges["MAT-012"]["parent_sha"] == edges["MAT-011"]["child_sha"], "MAT-012 must fork from exact MAT-011")
    chain(edges, EXPECTED_PROVIDER, expected_parent=edges["MAT-011"]["child_sha"])
    require(edges["MAT-011B"]["parent_sha"] == edges["MAT-011"]["child_sha"], "MAT-011B must fork from exact MAT-011")
    chain(edges, EXPECTED_AUTHORITY, expected_parent=edges["MAT-011"]["child_sha"])

    repair = edges["MAT-008B-0K"]
    require(repair["kind"] == "ExtractedGenericRepair", "MAT-008B-0K kind mismatch")
    require(repair["changed_paths"] == [ZERO_K_PATH], "MAT-008B-0K must change only conditioned_property.rs")
    transition = repair.get("historical_blobs", {}).get(ZERO_K_PATH)
    require(isinstance(transition, dict), "MAT-008B-0K blob transition required")
    require(transition.get("pre") == ZERO_K_PRE, "MAT-008B-0K historical pre-blob drift")
    require(transition.get("post") == ZERO_K_POST, "MAT-008B-0K historical post-blob drift")
    projection = repair.get("projection")
    require(isinstance(projection, dict), "MAT-008B-0K projection contract required")
    require(projection.get("after_edge") == "MAT-008", "MAT-008B-0K must project immediately after MAT-008")
    require(projection.get("forbidden_ancestry_import") == "MAG-001", "MAG-001 ancestry quarantine must be explicit")
    preserve = projection.get("preserve_through_edges")
    require(isinstance(preserve, list) and preserve == sorted(set(preserve), key=preserve.index), "preserve_through_edges must be unique")
    for edge_id in preserve:
        require(edge_id in edges, f"MAT-008B-0K preserve edge unknown: {edge_id}")
        overlap = set(edges[edge_id]["changed_paths"]) & {ZERO_K_PATH}
        require(not overlap, f"{edge_id}: historical scope overlaps admitted 0 K repair; mechanical replay must stop")

    # A provider crate addition must not mutate the common scientific crate in this historical edge.
    require(all(path.startswith("crates/domains/symthaea-materials-providers/") for path in edges["MAT-015"]["changed_paths"]), "MAT-015 scope escaped provider crate")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        data = json.loads(args.manifest.read_text(encoding="utf-8"))
        require(isinstance(data, dict), "manifest root must be object")
        validate_manifest(data)
    except (OSError, json.JSONDecodeError, ManifestError) as exc:
        print(f"REJECT: {exc}", file=sys.stderr)
        return 1
    print("PASS: historical replay manifest is internally consistent; authority=none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
