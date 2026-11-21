"""Build a content-addressed execution receipt for RCAF simulation outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_version(command: list[str]) -> str:
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"
    return (result.stdout or result.stderr).strip()


def tracked_rcaf_files(root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files", "-z", "research/rcaf"],
            check=True,
            cwd=root,
            capture_output=True,
            timeout=15,
        )
        names = [
            name.decode("utf-8")
            for name in result.stdout.split(b"\0")
            if name
        ]
    except (OSError, subprocess.SubprocessError):
        names = []
    files = [root / name for name in names]
    if not files:
        files = [path for path in (root / "research/rcaf").rglob("*") if path.is_file()]
    return sorted(path for path in files if path.is_file())


def build_receipt(root: Path, reports: list[Path]) -> dict[str, Any]:
    source_files = []
    tree = hashlib.sha256()
    for path in tracked_rcaf_files(root):
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        source_files.append({"path": relative, "sha256": digest, "bytes": path.stat().st_size})
        tree.update(relative.encode("utf-8"))
        tree.update(b"\0")
        tree.update(bytes.fromhex(digest))

    report_entries = []
    for path in reports:
        if not path.is_file():
            raise FileNotFoundError(f"simulation report does not exist: {path}")
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"simulation output is not valid JSON: {path}: {exc}") from exc
        nested_reports = []
        for collection_key in ("results", "reports"):
            collection = parsed.get(collection_key, [])
            if isinstance(collection, list):
                nested_reports.extend(item for item in collection if isinstance(item, dict))
        estimator_ids = {
            item["estimator"]
            for item in [parsed, *nested_reports]
            if isinstance(item.get("estimator"), str)
        }
        report_entries.append({
            "filename": path.name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "schema_version": parsed.get("schema_version"),
            "estimator_ids": sorted(estimator_ids),
        })

    revision = os.environ.get("GITHUB_SHA", "")
    if revision and (len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision.lower())):
        raise ValueError("GITHUB_SHA is not a 40-character Git commit SHA")

    return {
        "schema_version": "rcaf.execution-receipt/0.1.0",
        "source_commit": revision or None,
        "workflow": {
            "name": os.environ.get("GITHUB_WORKFLOW"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        },
        "environment": {
            "os": platform.platform(),
            "python": sys.version,
            "rustc": command_version(["rustc", "--version", "--verbose"]),
            "cargo": command_version(["cargo", "--version"]),
        },
        "source_tree_sha256": tree.hexdigest(),
        "source_files": source_files,
        "simulation_reports": report_entries,
        "claim_limits": [
            "receipt_hashes the exact checked-out RCAF source tree and simulation outputs",
            "does_not_prove independent replay",
            "does_not_validate_biological_signal_processing",
            "does_not_establish_consciousness_or_clinical_validity",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reports", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)
    try:
        root = args.root.resolve()
        reports = [path.resolve() for path in args.reports]
        receipt = build_receipt(root, reports)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print("PASS: content-addressed RCAF simulation execution receipt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
