#!/usr/bin/env python3

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
"""Run external Symthaea benchmarks via the unified manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_manifest(root: Path) -> dict:
    manifest_path = root / "benchmarks" / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(root: Path, rel: str) -> Path:
    return (root / rel).resolve()


def list_benchmarks(manifest: dict) -> None:
    for bench in manifest.get("benchmarks", []):
        print(f"{bench['id']}: {bench['name']}")


def select_benchmarks(manifest: dict, ids: list[str]) -> list[dict]:
    benches = manifest.get("benchmarks", [])
    if not ids:
        return benches
    by_id = {b["id"]: b for b in benches}
    selected = []
    for bid in ids:
        if bid not in by_id:
            raise KeyError(f"Unknown benchmark id: {bid}")
        selected.append(by_id[bid])
    return selected


def check_ready(root: Path, bench: dict) -> bool:
    ready_rel = bench.get("ready")
    if not ready_rel:
        return True
    ready_path = resolve_path(root, ready_rel)
    return ready_path.exists()


def run_bench(root: Path, bench: dict, dry_run: bool) -> int:
    run_rel = bench.get("run")
    if not run_rel:
        print(f"No run script for {bench['id']}")
        return 2
    run_path = resolve_path(root, run_rel)
    if not run_path.exists():
        print(f"Run script missing: {run_path}")
        return 2
    if dry_run:
        print(f"DRY RUN: {run_path}")
        return 0
    result = subprocess.run([str(run_path)], check=False)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Run external Symthaea benchmarks")
    parser.add_argument("--list", action="store_true", help="List available benchmarks")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--bench", action="append", default=[], help="Benchmark id (repeatable)")
    parser.add_argument("--check", action="store_true", help="Check dataset readiness")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    parser.add_argument(
        "--skip-ready-check",
        action="store_true",
        help="Run even if READY marker is missing",
    )

    args = parser.parse_args()

    root = repo_root()
    manifest = load_manifest(root)

    if args.list:
        list_benchmarks(manifest)
        return 0

    bench_ids = [] if args.all else args.bench
    benches = select_benchmarks(manifest, bench_ids)

    if not benches:
        print("No benchmarks selected. Use --list or --bench.")
        return 2

    if args.check:
        ok = True
        for bench in benches:
            ready = check_ready(root, bench)
            status = "READY" if ready else "MISSING"
            print(f"{bench['id']}: {status}")
            ok = ok and ready
        return 0 if ok else 3

    exit_code = 0
    for bench in benches:
        if not args.skip_ready_check and not check_ready(root, bench):
            print(f"{bench['id']}: data not ready. Run fetch.sh or follow README.")
            exit_code = 3
            continue
        code = run_bench(root, bench, args.dry_run)
        if code != 0:
            exit_code = code
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
