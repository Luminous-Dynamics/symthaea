#!/usr/bin/env python3
"""Collect external benchmark results into a single JSON report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_manifest(root: Path) -> dict:
    manifest_path = root / "benchmarks" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect external benchmark results")
    parser.add_argument(
        "--output",
        default="results/aggregate.json",
        help="Output JSON path (relative to benchmarks/external)",
    )
    args = parser.parse_args()

    root = repo_root()
    manifest = load_manifest(root)

    results = []
    missing = []

    for bench in manifest.get("benchmarks", []):
        result_rel = bench.get("results")
        if not result_rel:
            continue
        result_path = (root / result_rel).resolve()
        if result_path.exists():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                results.append(payload)
            except json.JSONDecodeError:
                missing.append({
                    "id": bench.get("id"),
                    "path": str(result_path),
                    "reason": "invalid_json",
                })
        else:
            missing.append({
                "id": bench.get("id"),
                "path": str(result_path),
                "reason": "missing",
            })

    output = {
        "run_id": datetime.now(timezone.utc).isoformat(),
        "manifest_version": manifest.get("version"),
        "benchmarks_total": len(manifest.get("benchmarks", [])),
        "benchmarks_collected": len(results),
        "benchmarks_missing": len(missing),
        "results": results,
        "missing": missing,
    }

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote aggregate results to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
