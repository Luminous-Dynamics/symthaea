#!/usr/bin/env python3
"""Build a deterministic, fail-closed review docket for the PITB family-coverage audit.

This tool organizes the two independent coverage-review passes required by the
protocol. It does NOT discover dependencies, adjudicate pairs, sign a receipt,
or certify that the holdout is clean.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import PurePosixPath
from typing import Any

UPSTREAM_REPOSITORY = "https://github.com/openai/math"
UPSTREAM_COMMIT = "1581d4cbdd69a0faf34b2089614fe2a814316a6a"
SPLIT_SEED = "pitb-v0-openai-math-2026-10-19"
SPLIT_ALGORITHM = (
    "sha256(seed || NUL || family_id), sort by digest then family ID, "
    "floor 70% train and floor 85% train+validation"
)
EXPECTED_PARTITION_SHA256 = (
    "e7ac417556cbd47cee1ddd51b98f9684fadfc8cd097587ae3beacc4dfb59f970"
)
EXPECTED_COUNTS = {"train": 260, "validation": 56, "test": 56}
REVIEW_PASSES = (
    {
        "pass_id": "independent_candidate_derivation",
        "purpose": (
            "Independently derive candidate cross-family relationships from "
            "the pinned source/catalogue, rather than reviewing another "
            "reviewer's candidate list as the sole source."
        ),
    },
    {
        "pass_id": "false_negative_challenge",
        "purpose": (
            "Perform a separate false-negative-oriented challenge pass over "
            "all families, including families with no discovered candidates."
        ),
    },
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def deterministic_splits(family_ids: list[str], seed: str) -> dict[str, str]:
    if len(family_ids) < 3 or len(set(family_ids)) != len(family_ids):
        raise ValueError("family IDs must be unique and contain at least three entries")
    ranked = sorted(
        (sha256(f"{seed}\0{family_id}".encode("utf-8")), family_id)
        for family_id in family_ids
    )
    train_end = len(ranked) * 70 // 100
    validation_end = len(ranked) * 85 // 100
    result: dict[str, str] = {}
    for index, (_, family_id) in enumerate(ranked):
        if index < train_end:
            split = "train"
        elif index < validation_end:
            split = "validation"
        else:
            split = "test"
        result[family_id] = split
    return result


def partition_digest(assignments: dict[str, str]) -> str:
    material = "".join(
        f"{family_id}\t{assignments[family_id]}\n"
        for family_id in sorted(assignments)
    )
    return sha256(material.encode("utf-8"))


def validate_manifest(
    manifest: dict[str, Any], *, require_pinned: bool
) -> tuple[list[dict[str, Any]], dict[str, str], str]:
    if manifest.get("schema_version") != "symthaea.proof-insight-transfer.splits.v0":
        raise ValueError("unsupported split manifest schema_version")
    upstream = manifest.get("upstream")
    if not isinstance(upstream, dict):
        raise ValueError("upstream provenance is missing")
    if require_pinned and (
        upstream.get("repository") != UPSTREAM_REPOSITORY
        or upstream.get("commit") != UPSTREAM_COMMIT
    ):
        raise ValueError("manifest does not match the registered OpenAI/math snapshot")

    plan = manifest.get("split_plan")
    if not isinstance(plan, dict):
        raise ValueError("split_plan is missing")
    if (
        plan.get("seed") != SPLIT_SEED
        or plan.get("algorithm") != SPLIT_ALGORITHM
        or plan.get("train_percent") != 70
        or plan.get("validation_percent") != 15
        or plan.get("test_percent") != 15
    ):
        raise ValueError("split plan differs from the registered PITB v0 algorithm")

    rows = manifest.get("families")
    if not isinstance(rows, list):
        raise ValueError("families must be an array")
    families: list[dict[str, Any]] = []
    ids: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("family rows must be objects")
        family_id = row.get("family_id")
        title = row.get("title")
        split = row.get("split")
        paths = row.get("linked_paths")
        if not isinstance(family_id, str) or not family_id.strip():
            raise ValueError("family_id must be a nonblank string")
        if family_id in seen:
            raise ValueError(f"duplicate family_id: {family_id}")
        seen.add(family_id)
        if not isinstance(title, str) or not title.strip():
            raise ValueError(f"{family_id}: title must be nonblank")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"{family_id}: unregistered split")
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"{family_id}: linked_paths must be a nonempty array")
        normalized_paths: list[str] = []
        for path in paths:
            if not isinstance(path, str) or not path.strip():
                raise ValueError(f"{family_id}: linked path must be a nonblank string")
            parsed = PurePosixPath(path)
            if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
                raise ValueError(f"{family_id}: unsafe repository-relative path: {path}")
            normalized_paths.append(path)
        families.append(
            {
                "family_id": family_id,
                "title": title,
                "split": split,
                "linked_paths": sorted(set(normalized_paths)),
            }
        )
        ids.append(family_id)

    expected = deterministic_splits(ids, SPLIT_SEED)
    supplied = {row["family_id"]: row["split"] for row in families}
    if supplied != expected:
        mismatches = sorted(
            family_id
            for family_id in set(expected) | set(supplied)
            if expected.get(family_id) != supplied.get(family_id)
        )
        raise ValueError(f"family assignments do not reproduce the registered split: {mismatches[:10]}")

    digest = partition_digest(expected)
    declared_digest = plan.get("partition_digest_sha256")
    if declared_digest != digest:
        raise ValueError("declared partition digest does not match recomputed assignments")
    counts = {name: sum(1 for split in supplied.values() if split == name)
              for name in ("train", "validation", "test")}
    if plan.get("counts") != counts:
        raise ValueError("declared split counts do not reconcile with family rows")

    if require_pinned and (
        len(families) != 372
        or counts != EXPECTED_COUNTS
        or digest != EXPECTED_PARTITION_SHA256
    ):
        raise ValueError("frozen PITB family count, split counts, or partition digest mismatch")

    families.sort(key=lambda row: row["family_id"])
    return families, expected, digest


def build_docket(
    manifest: dict[str, Any], manifest_bytes: bytes, *, require_pinned: bool = True
) -> dict[str, Any]:
    families, assignments, digest = validate_manifest(
        manifest, require_pinned=require_pinned
    )
    family_rows = []
    for family in families:
        family_rows.append(
            {
                **family,
                "review_passes": [
                    {
                        "pass_id": review_pass["pass_id"],
                        "status": "pending",
                        "reviewer_id": None,
                        "organization_id": None,
                        "review_log_sha256": None,
                        "inspected_artifact_paths": [],
                        "discovered_edge_ids": [],
                        "unresolved_candidates": [],
                        "limitations": [],
                    }
                    for review_pass in REVIEW_PASSES
                ],
            }
        )
    counts = {
        name: sum(1 for split in assignments.values() if split == name)
        for name in ("train", "validation", "test")
    }
    return {
        "schema_version": "pitb.coverage-review-docket.v0",
        "docket_kind": "review_assignment_template",
        "status": "pending_independent_review",
        "upstream": {
            "repository": manifest["upstream"]["repository"],
            "commit": manifest["upstream"]["commit"],
        },
        "source_artifacts": {
            "split_manifest_path": "data/benchmarks/proof-insight-transfer-v0/splits.json",
            "split_manifest_sha256": sha256(manifest_bytes),
            "partition_digest_sha256": digest,
        },
        "coverage_contract": {
            "family_count": len(family_rows),
            "independent_review_passes_per_family": len(REVIEW_PASSES),
            "required_family_review_rows": len(family_rows) * len(REVIEW_PASSES),
            "split_counts": counts,
            "required_passes": list(REVIEW_PASSES),
        },
        "families": family_rows,
        "coverage_status": "incomplete",
        "confirmatory_eligible": False,
        "limitations": [
            "This docket allocates review work; it does not discover or adjudicate relationships.",
            "Every review pass must inspect all families, including apparent singletons.",
            "Candidate and dismissed relationships must retain evidence, rationale, and search logs.",
            "Unresolved hard relationships remain in the conservative component graph.",
            "A completed docket is not a signed coverage receipt or proof of reviewer independence.",
        ],
    }


def run_self_tests() -> int:
    def sample_manifest(count: int = 20) -> dict[str, Any]:
        ids = [f"F{i:03d}" for i in range(count)]
        assignments = deterministic_splits(ids, SPLIT_SEED)
        counts = {name: sum(1 for s in assignments.values() if s == name)
                  for name in ("train", "validation", "test")}
        return {
            "schema_version": "symthaea.proof-insight-transfer.splits.v0",
            "upstream": {"repository": UPSTREAM_REPOSITORY, "commit": UPSTREAM_COMMIT},
            "split_plan": {
                "seed": SPLIT_SEED,
                "algorithm": SPLIT_ALGORITHM,
                "train_percent": 70,
                "validation_percent": 15,
                "test_percent": 15,
                "counts": counts,
                "partition_digest_sha256": partition_digest(assignments),
            },
            "families": [
                {
                    "family_id": family_id,
                    "title": f"Sample theorem family {family_id}",
                    "split": assignments[family_id],
                    "linked_paths": [f"lean/docs/{family_id}.md"],
                }
                for family_id in ids
            ],
        }

    checks = 0
    manifest = sample_manifest()
    raw = json.dumps(manifest, sort_keys=True).encode()
    docket = build_docket(manifest, raw, require_pinned=False)
    assert docket["coverage_status"] == "incomplete"
    assert docket["confirmatory_eligible"] is False
    assert docket["coverage_contract"]["family_count"] == 20
    assert docket["coverage_contract"]["required_family_review_rows"] == 40
    assert all(
        len(row["review_passes"]) == 2
        and all(item["status"] == "pending" for item in row["review_passes"])
        for row in docket["families"]
    )
    checks += 1

    second = build_docket(manifest, raw, require_pinned=False)
    assert json.dumps(docket, sort_keys=True) == json.dumps(second, sort_keys=True)
    checks += 1

    bad = json.loads(json.dumps(manifest))
    bad["families"][0]["split"] = "test" if bad["families"][0]["split"] != "test" else "train"
    try:
        build_docket(bad, raw, require_pinned=False)
        raise AssertionError("assignment mutation accepted")
    except ValueError:
        checks += 1

    bad = json.loads(json.dumps(manifest))
    bad["families"][1]["family_id"] = bad["families"][0]["family_id"]
    try:
        build_docket(bad, raw, require_pinned=False)
        raise AssertionError("duplicate family ID accepted")
    except ValueError:
        checks += 1

    bad = json.loads(json.dumps(manifest))
    bad["families"][0]["linked_paths"] = ["../../outside.pdf"]
    try:
        build_docket(bad, raw, require_pinned=False)
        raise AssertionError("path traversal accepted")
    except ValueError:
        checks += 1

    bad = json.loads(json.dumps(manifest))
    bad["split_plan"]["partition_digest_sha256"] = "0" * 64
    try:
        build_docket(bad, raw, require_pinned=False)
        raise AssertionError("stale partition digest accepted")
    except ValueError:
        checks += 1

    bad = json.loads(json.dumps(manifest))
    bad["split_plan"]["seed"] = "unregistered"
    try:
        build_docket(bad, raw, require_pinned=False)
        raise AssertionError("unregistered split seed accepted")
    except ValueError:
        checks += 1

    print(f"PITB_COVERAGE_DOCKET_SELF_TEST_PASS: checks={checks}")
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split_manifest", nargs="?")
    parser.add_argument("output", nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_tests()
        return 0
    if not args.split_manifest or not args.output:
        parser.error("split_manifest and output are required unless --self-test is used")

    manifest_path = PurePosixPath(args.split_manifest)
    del manifest_path  # Paths are opened by pathlib below; validation concerns source paths.
    from pathlib import Path

    source_path = Path(args.split_manifest)
    output_path = Path(args.output)
    raw = source_path.read_bytes()
    manifest = json.loads(raw)
    docket = build_docket(manifest, raw, require_pinned=True)
    encoded = json.dumps(docket, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output_path.write_text(encoded, encoding="utf-8")
    print(
        "PITB_COVERAGE_REVIEW_DOCKET_PASS: "
        f"families={docket['coverage_contract']['family_count']} "
        f"review_rows={docket['coverage_contract']['required_family_review_rows']} "
        "passes_per_family=2 status=pending_independent_review confirmatory=false"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"PITB_COVERAGE_REVIEW_DOCKET_ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
