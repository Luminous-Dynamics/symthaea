#!/usr/bin/env python3
"""Seal exact Spore visual-review artifacts with reproducible SHA-256 evidence.

The galleries are review evidence for the real CPU framebuffer renderer, so CI
records exactly which bytes were produced. The seal contains no user data and
never participates in boot authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

EVIDENCE_FILES = {"EVIDENCE.sha256", "evidence-manifest.json"}
ALLOWED_SUFFIXES = {".ppm", ".png", ".json", ".html"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def source_commit() -> str:
    # On pull_request workflows GITHUB_SHA is normally the synthetic merge
    # commit. Preserve the PR head separately so visual evidence can be tied to
    # the actual patch lineage as well as the tree that CI evaluated.
    explicit = os.environ.get("SPORE_SOURCE_COMMIT", "").strip()
    if explicit:
        return explicit
    github_sha = os.environ.get("GITHUB_SHA", "").strip()
    if github_sha:
        return github_sha
    return git_head()


def evaluated_commit() -> str:
    github_sha = os.environ.get("GITHUB_SHA", "").strip()
    return github_sha or git_head()


def seal(root: Path, source: str, evaluated: str) -> dict:
    if not root.is_dir():
        raise SystemExit(f"visual evidence root does not exist: {root}")

    records: list[dict] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in EVIDENCE_FILES:
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        size = path.stat().st_size
        relative = path.relative_to(root).as_posix()
        records.append({"path": relative, "sha256": sha256(path), "bytes": size})
        total_bytes += size

    if not records:
        raise SystemExit(f"visual evidence root has no review artifacts: {root}")

    manifest = {
        "schema": "spore-visual-evidence-v1",
        "source_commit": source,
        "evaluated_commit": evaluated,
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID", "unknown"),
        "workflow_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "unknown"),
        "root": root.name,
        "file_count": len(records),
        "total_bytes": total_bytes,
        "hash": "sha256",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "files": records,
    }

    manifest_path = root / "evidence-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    sums = "".join(f"{record['sha256']}  {record['path']}\n" for record in records)
    (root / "EVIDENCE.sha256").write_text(sums)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Seal exact Spore renderer galleries with SHA-256 evidence"
    )
    parser.add_argument("roots", nargs="+", type=Path)
    args = parser.parse_args()

    source = source_commit()
    evaluated = evaluated_commit()
    summaries = []
    for root in args.roots:
        manifest = seal(root, source, evaluated)
        summaries.append(
            f"{manifest['root']}: {manifest['file_count']} files, "
            f"{manifest['total_bytes']} bytes"
        )
    print(
        f"sealed {len(summaries)} visual evidence roots "
        f"source={source} evaluated={evaluated}"
    )
    for summary in summaries:
        print(f"  {summary}")


if __name__ == "__main__":
    main()
