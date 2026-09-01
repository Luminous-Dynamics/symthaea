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


def source_commit() -> str:
    github_sha = os.environ.get("GITHUB_SHA", "").strip()
    if github_sha:
        return github_sha
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def seal(root: Path, commit: str) -> dict:
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
        "source_commit": commit,
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

    commit = source_commit()
    summaries = []
    for root in args.roots:
        manifest = seal(root, commit)
        summaries.append(
            f"{manifest['root']}: {manifest['file_count']} files, "
            f"{manifest['total_bytes']} bytes"
        )
    print(f"sealed {len(summaries)} visual evidence roots at {commit}")
    for summary in summaries:
        print(f"  {summary}")


if __name__ == "__main__":
    main()
