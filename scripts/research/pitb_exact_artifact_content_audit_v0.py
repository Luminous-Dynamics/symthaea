#!/usr/bin/env python3
"""Research-only exact-byte cross-family artifact audit for PITB v0.

This scanner finds repeated raw artifact bytes across different frozen families.
A match is an unreviewed source-provenance candidate, never an adjudicated
mathematical dependency. A zero-match report says nothing about semantic or
proof-dependency independence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import stat
import subprocess
import sys
import tempfile
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

UPSTREAM_REPOSITORY = "openai/math"
UPSTREAM_COMMIT = "1581d4cbdd69a0faf34b2089614fe2a814316a6a"
SPLIT_SEED = "pitb-v0-openai-math-2026-10-19"
EXPECTED_PARTITION_DIGEST = "e7ac417556cbd47cee1ddd51b98f9684fadfc8cd097587ae3beacc4dfb59f970"
EXPECTED_COUNTS = {"train": 260, "validation": 56, "test": 56}
EXPECTED_FAMILY_COUNT = 372
EXPECTED_UNIQUE_ARTIFACT_PATH_COUNT = 961
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1\n"


class AuditError(Exception):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def safe_relative_path(raw: str) -> tuple[str, ...]:
    if not isinstance(raw, str) or not raw or raw.strip() != raw:
        raise AuditError("path is blank or has surrounding whitespace")
    if "\\" in raw or raw.startswith("/"):
        raise AuditError("path is not canonical repository-relative POSIX form")
    parts = tuple(raw.split("/"))
    if any(part in ("", ".", "..") for part in parts):
        raise AuditError("path contains an empty, dot, or parent component")
    if ":" in parts[0]:
        raise AuditError("path has a drive-qualified first component")
    return parts


def deterministic_assignments(family_ids: list[str]) -> dict[str, str]:
    if len(family_ids) != len(set(family_ids)) or any(not x.strip() for x in family_ids):
        raise AuditError("family IDs must be nonblank and unique")
    ranked = sorted(
        (sha256(f"{SPLIT_SEED}\0{family_id}".encode("utf-8")), family_id)
        for family_id in family_ids
    )
    train_end = len(ranked) * 70 // 100
    validation_end = len(ranked) * 85 // 100
    result: dict[str, str] = {}
    for index, (_, family_id) in enumerate(ranked):
        result[family_id] = (
            "train" if index < train_end else
            "validation" if index < validation_end else "test"
        )
    return result


def partition_digest(assignments: dict[str, str]) -> str:
    material = "".join(f"{family_id}\t{assignments[family_id]}\n" for family_id in sorted(assignments))
    return sha256(material.encode("utf-8"))


def load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, str], dict[str, list[str]], str]:
    raw = path.read_bytes()
    try:
        doc = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"manifest is not valid UTF-8 JSON: {exc}") from exc

    if doc.get("schema_version") != "symthaea.proof-insight-transfer.splits.v0":
        raise AuditError("unexpected split-manifest schema version")
    if doc.get("upstream", {}).get("repository") not in (
        "https://github.com/openai/math",
        UPSTREAM_REPOSITORY,
    ):
        raise AuditError("manifest identifies an unexpected upstream repository")
    if doc.get("upstream", {}).get("commit") != UPSTREAM_COMMIT:
        raise AuditError("manifest does not pin the required upstream commit")
    plan = doc.get("split_plan", {})
    if plan.get("seed") != SPLIT_SEED:
        raise AuditError("split seed differs from the frozen PITB v0 seed")
    expected_algorithm = (
        "sha256(seed || NUL || family_id), sort by digest then family ID, "
        "floor 70% train and floor 85% train+validation"
    )
    if plan.get("algorithm") != expected_algorithm:
        raise AuditError("split algorithm differs from the frozen PITB v0 algorithm")
    if (plan.get("train_percent"), plan.get("validation_percent"), plan.get("test_percent")) != (70, 15, 15):
        raise AuditError("split percentages differ from the frozen PITB v0 partition")

    claims = doc.get("catalogue_claims", {})
    if (
        claims.get("families") != EXPECTED_FAMILY_COUNT
        or claims.get("parsed_family_headers") != EXPECTED_FAMILY_COUNT
        or claims.get("linked_path_count") != EXPECTED_UNIQUE_ARTIFACT_PATH_COUNT
    ):
        raise AuditError("manifest catalogue claims differ from the frozen 372-family / 961-path census")

    rows = doc.get("families")
    if not isinstance(rows, list) or len(rows) != EXPECTED_FAMILY_COUNT:
        raise AuditError(f"expected exactly {EXPECTED_FAMILY_COUNT} frozen families")

    declared: dict[str, str] = {}
    family_paths: dict[str, list[str]] = {}
    path_owner: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise AuditError("family row is not an object")
        family_id = row.get("family_id")
        split = row.get("split")
        paths = row.get("linked_paths")
        if not isinstance(family_id, str) or not family_id or split not in EXPECTED_COUNTS:
            raise AuditError("family row has an invalid ID or split")
        if family_id in declared:
            raise AuditError(f"duplicate family ID: {family_id}")
        if not isinstance(paths, list) or not paths:
            raise AuditError(f"family {family_id} must have a non-empty linked_paths list")
        declared[family_id] = split
        normalized_paths: list[str] = []
        for raw_path in paths:
            parts = safe_relative_path(raw_path)
            normalized = "/".join(parts)
            if normalized in path_owner:
                raise AuditError(
                    f"linked path appears more than once: {normalized} "
                    f"(families {path_owner[normalized]} and {family_id})"
                )
            path_owner[normalized] = family_id
            normalized_paths.append(normalized)
        family_paths[family_id] = sorted(normalized_paths)

    if len(path_owner) != EXPECTED_UNIQUE_ARTIFACT_PATH_COUNT:
        raise AuditError(
            f"expected exactly {EXPECTED_UNIQUE_ARTIFACT_PATH_COUNT} unique linked artifact paths; "
            f"found {len(path_owner)}"
        )

    recomputed = deterministic_assignments(list(declared))
    if declared != recomputed:
        raise AuditError("family assignments do not reproduce from the pinned split rule")
    actual_counts = {name: sum(value == name for value in declared.values()) for name in EXPECTED_COUNTS}
    if actual_counts != EXPECTED_COUNTS or plan.get("counts") != EXPECTED_COUNTS:
        raise AuditError(f"split counts differ from the frozen values: {actual_counts}")
    digest = partition_digest(declared)
    if digest != EXPECTED_PARTITION_DIGEST or plan.get("partition_digest_sha256") != digest:
        raise AuditError("partition digest differs from the frozen PITB v0 value")

    return doc, declared, family_paths, sha256(raw)


def verify_source_checkout(root: Path) -> str:
    root = root.resolve(strict=True)
    if not root.is_dir():
        raise AuditError("source root is not a directory")

    def git(*args: str) -> str:
        proc = subprocess.run(
            ["git", "-C", str(root), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        if proc.returncode:
            raise AuditError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
        return proc.stdout.strip()

    actual = git("rev-parse", "HEAD")
    if actual != UPSTREAM_COMMIT:
        raise AuditError(f"upstream checkout is {actual}, expected {UPSTREAM_COMMIT}")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise AuditError("upstream tracked working tree is dirty; refusing to hash a modified snapshot")
    toplevel = Path(git("rev-parse", "--show-toplevel")).resolve(strict=True)
    if toplevel != root:
        raise AuditError("source root must be the upstream repository's top-level checkout")
    return actual


def read_artifact(root: Path, path: str) -> tuple[str, bytes | None, str | None]:
    parts = safe_relative_path(path)
    current = root
    try:
        for part in parts:
            current = current / part
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode):
                return "symlink_rejected", None, "path component is a symlink"
        final_info = current.lstat()
    except FileNotFoundError:
        return "missing", None, "linked path does not exist in pinned checkout"
    except OSError as exc:
        return "unreadable", None, f"filesystem inspection failed: {exc}"

    if not stat.S_ISREG(final_info.st_mode):
        return "not_regular_file", None, "linked path is not a regular file"
    try:
        data = current.read_bytes()
    except OSError as exc:
        return "unreadable", None, f"file read failed: {exc}"
    if data.startswith(LFS_POINTER_PREFIX):
        return "lfs_pointer_unmaterialized", None, "Git-LFS pointer was found instead of materialized artifact bytes"
    return "scanned", data, None


def build_report(
    manifest_sha256: str,
    assignments: dict[str, str],
    family_paths: dict[str, list[str]],
    source_commit: str,
    source_root: Path,
) -> dict[str, Any]:
    path_families = {
        path: family_id
        for family_id, paths in family_paths.items()
        for path in paths
    }
    artifact_inventory: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    by_digest: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_bytes = 0

    for path, family_id in sorted(path_families.items()):
        status, data, reason = read_artifact(source_root, path)
        if data is None:
            failures.append({"family_id": family_id, "path": path, "status": status, "reason": reason or "unknown"})
            artifact_inventory.append({"family_id": family_id, "path": path, "status": status})
            continue
        digest = sha256(data)
        row = {"family_id": family_id, "path": path, "status": "scanned", "size_bytes": len(data), "sha256": digest}
        artifact_inventory.append(row)
        by_digest[digest].append(row)
        total_bytes += len(data)

    duplicate_groups = []
    candidate_map: dict[tuple[str, str, str], dict[str, Any]] = {}
    for digest, rows in sorted(by_digest.items()):
        family_ids = sorted({row["family_id"] for row in rows})
        if len(family_ids) < 2:
            continue
        paths_by_family = {
            family_id: sorted(row["path"] for row in rows if row["family_id"] == family_id)
            for family_id in family_ids
        }
        duplicate_groups.append({
            "sha256": digest,
            "size_bytes": rows[0]["size_bytes"],
            "family_ids": family_ids,
            "paths_by_family": paths_by_family,
            "candidate_status": "unreviewed",
            "candidate_relation": "exact_byte_identity_only",
        })
        for source_family, target_family in combinations(family_ids, 2):
            key = (source_family, target_family, digest)
            evidence = {
                "source_family_id": source_family,
                "target_family_id": target_family,
                "sha256": digest,
                "size_bytes": rows[0]["size_bytes"],
                "source_paths": paths_by_family[source_family],
                "target_paths": paths_by_family[target_family],
            }
            candidate_map[key] = {
                **evidence,
                "source_split": assignments[source_family],
                "target_split": assignments[target_family],
                "candidate_relation": "exact_byte_identity_only",
                "candidate_status": "unreviewed",
                "evidence_sha256": sha256(canonical_json(evidence)),
            }

    candidates = sorted(
        candidate_map.values(),
        key=lambda row: (row["source_family_id"], row["target_family_id"], row["sha256"]),
    )
    cross_split = sum(row["source_split"] != row["target_split"] for row in candidates)
    return {
        "schema_version": "pitb.exact-artifact-content-audit.v0",
        "status": "complete" if not failures else "incomplete",
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": source_commit,
        "frozen_partition_digest_sha256": EXPECTED_PARTITION_DIGEST,
        "manifest_sha256": manifest_sha256,
        "expected_family_count": EXPECTED_FAMILY_COUNT,
        "scanned_family_count": len(assignments),
        "expected_unique_artifact_path_count": EXPECTED_UNIQUE_ARTIFACT_PATH_COUNT,
        "scanned_artifact_count": len(artifact_inventory) - len(failures),
        "failed_artifact_count": len(failures),
        "scanned_bytes_total": total_bytes,
        "artifact_inventory": artifact_inventory,
        "incomplete_artifacts": failures,
        "duplicate_groups": duplicate_groups,
        "candidate_pairs": candidates,
        "candidate_pair_count": len(candidates),
        "cross_split_candidate_pair_count": cross_split,
        "method": {
            "digest": "SHA-256 over exact materialized file bytes",
            "candidate_rule": "same exact digest at distinct linked paths owned by at least two frozen families",
            "review_status": "all candidates remain unreviewed",
            "limitation": (
                "No match is not evidence of independence. Exact byte identity does not establish "
                "mathematical dependency, theorem equivalence, or contamination; every candidate "
                "requires independent review. This scan does not inspect PDF text or Lean semantics."
            ),
        },
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="pitb-content-audit-") as temp:
        root = Path(temp)
        (root / "a.bin").write_bytes(b"same bytes")
        (root / "b.bin").write_bytes(b"same bytes")
        (root / "c.bin").write_bytes(b"different bytes")
        assignments = {"001": "train", "002": "test", "003": "validation"}
        paths = {"001": ["a.bin"], "002": ["b.bin"], "003": ["c.bin"]}
        report = build_report("0" * 64, assignments, paths, UPSTREAM_COMMIT, root)
        assert report["status"] == "complete"
        assert report["scanned_artifact_count"] == 3
        assert report["candidate_pair_count"] == 1
        candidate = report["candidate_pairs"][0]
        assert (candidate["source_family_id"], candidate["target_family_id"]) == ("001", "002")
        assert candidate["source_split"] == "train" and candidate["target_split"] == "test"
        assert candidate["candidate_status"] == "unreviewed"
        assert candidate["candidate_relation"] == "exact_byte_identity_only"

        no_match = build_report("0" * 64, assignments, {"001": ["a.bin"], "002": ["c.bin"], "003": []}, UPSTREAM_COMMIT, root)
        assert no_match["candidate_pair_count"] == 0
        assert "No match is not evidence of independence" in no_match["method"]["limitation"]

        try:
            safe_relative_path("../escape.pdf")
        except AuditError:
            pass
        else:
            raise AssertionError("parent traversal was accepted")

        (root / "lfs.bin").write_bytes(LFS_POINTER_PREFIX + b"oid sha256:" + b"0" * 64 + b"\nsize 10\n")
        assert read_artifact(root, "lfs.bin")[0] == "lfs_pointer_unmaterialized"

        try:
            (root / "link.bin").symlink_to(root / "a.bin")
        except (OSError, NotImplementedError):
            pass
        else:
            assert read_artifact(root, "link.bin")[0] == "symlink_rejected"

    print("PITB_EXACT_ARTIFACT_CONTENT_AUDIT_SELF_TEST_PASS: exact-match, no-match, traversal, symlink, and Git-LFS controls")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.manifest is None or args.source_root is None or args.out is None:
        parser.error("--manifest, --source-root, and --out are required unless --self-test is used")

    try:
        _, assignments, family_paths, manifest_sha256 = load_manifest(args.manifest)
        source_commit = verify_source_checkout(args.source_root)
        report = build_report(manifest_sha256, assignments, family_paths, source_commit, args.source_root.resolve(strict=True))
    except (AuditError, OSError, subprocess.SubprocessError) as exc:
        print(f"PITB_EXACT_ARTIFACT_CONTENT_AUDIT_REJECTED: {exc}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "PITB_EXACT_ARTIFACT_CONTENT_AUDIT: "
        f"status={report['status']} scanned={report['scanned_artifact_count']}/"
        f"{report['expected_unique_artifact_path_count']} candidates={report['candidate_pair_count']} "
        f"cross_split={report['cross_split_candidate_pair_count']}"
    )
    return 0 if report["status"] == "complete" else 3


if __name__ == "__main__":
    raise SystemExit(main())
