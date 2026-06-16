#!/usr/bin/env python3
"""Check rustfmt for changed Rust files without inheriting legacy drift."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_PREFIXES = (
    "patches/",
    "vendor/",
    "target/",
    "isolated/",
)

SMOKE_FILES = [
    "crates/domains/symthaea-broca/src/wasm_architect.rs",
    "crates/symthaea-core/src/api/holon.rs",
    "crates/symthaea-core/src/bin/symthaea-holon.rs",
    "crates/domains/symthaea-embeddings/src/qwen3/safetensors_loader.rs",
    "crates/domains/symthaea-soma/src/engine.rs",
    "crates/domains/symthaea-soma/src/native_ffi.rs",
    "crates/domains/symthaea-swarm/src/lib.rs",
    "crates/core/symthaea-core/src/dynamics/temporal_stability.rs",
    "crates/core/symthaea-core/src/hdc/topology_analysis.rs",
]


def run(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def best_range() -> list[str] | None:
    event = os.environ.get("GITHUB_EVENT_NAME")
    sha = os.environ.get("GITHUB_SHA", "HEAD")

    if not event:
        return None

    if event == "pull_request" and os.environ.get("GITHUB_BASE_REF"):
        base_ref = os.environ["GITHUB_BASE_REF"]
        run(["git", "fetch", "--no-tags", "origin", base_ref], check=False)
        merge_base = run(["git", "merge-base", f"origin/{base_ref}", sha], check=False)
        if merge_base.returncode == 0:
            return [merge_base.stdout.strip(), sha]

    before = os.environ.get("GITHUB_EVENT_BEFORE") or os.environ.get("GITHUB_BEFORE")
    if event == "push" and before and not set(before) <= {"0"}:
        return [before, sha]

    head_parent = run(["git", "rev-parse", "--verify", "HEAD^"], check=False)
    if head_parent.returncode == 0:
        return [head_parent.stdout.strip(), "HEAD"]

    return None


def changed_files() -> list[str]:
    prefix_result = run(["git", "rev-parse", "--show-prefix"], check=False)
    prefix = prefix_result.stdout.strip() if prefix_result.returncode == 0 else ""
    relative = [f"--relative={prefix}"] if prefix else []

    comparison = best_range()
    if comparison:
        diff = run(["git", "diff", *relative, "--name-only", "--diff-filter=ACMRT", *comparison])
    else:
        diff = run(["git", "diff", *relative, "--name-only", "--diff-filter=ACMRT", "HEAD", "--"])

    files = []
    for line in diff.stdout.splitlines():
        path = line.strip()
        if not path.endswith(".rs"):
            continue
        if path.startswith(EXCLUDED_PREFIXES):
            continue
        if (ROOT / path).is_file():
            files.append(path)
    return sorted(set(files))


def main() -> int:
    files = changed_files()
    if not files:
        files = [path for path in SMOKE_FILES if (ROOT / path).is_file()]

    if not files:
        print("No Rust files selected for rustfmt check.")
        return 0

    print("Checking rustfmt for:")
    for path in files:
        print(f"  {path}")

    cmd = ["rustfmt", "--edition", "2024", "--check", *files]
    result = subprocess.run(cmd, cwd=ROOT)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
