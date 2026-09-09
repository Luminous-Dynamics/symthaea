#!/usr/bin/env python3
"""Audit explicit Cargo test registration for inference qualification workflows.

The root package deliberately sets ``autotests = false``. Consequently every
integration-test binary named by an inference GitHub Actions workflow must be
registered explicitly as a ``[[test]]`` target in the root Cargo.toml.

The expected set is derived from workflow commands, not filenames or workflow
names. This matters because one workflow may run multiple binaries and the
transport target is named ``openai_compatible_transport``.

By default this checker is read-only and fail-closed. ``--fix`` appends only
missing, workflow-proven targets after validating that their source files exist
and that the existing manifest has no duplicate target names or accidentally
registered legacy catch-all.
"""

from __future__ import annotations

import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
MANIFEST = ROOT / "Cargo.toml"
TEST_DIR = ROOT / "tests"

TEST_ARG = re.compile(r"(?:^|\s)--test\s+([A-Za-z0-9_-]+)")
LEGACY_CATCH_ALL = "inference_integration"
MARKER = "# Inference qualification targets derived from .github/workflows/inference-*.yml."


def workflow_targets() -> dict[str, set[str]]:
    targets: dict[str, set[str]] = {}
    for workflow in sorted(WORKFLOW_DIR.glob("inference-*.yml")):
        names = set(TEST_ARG.findall(workflow.read_text(encoding="utf-8")))
        if names:
            targets[workflow.name] = names
    return targets


def manifest_test_names() -> tuple[set[str], set[str]]:
    with MANIFEST.open("rb") as handle:
        document = tomllib.load(handle)

    names: set[str] = set()
    duplicates: set[str] = set()
    for target in document.get("test", []):
        name = target.get("name")
        if not isinstance(name, str):
            continue
        if name in names:
            duplicates.add(name)
        names.add(name)
    return names, duplicates


def structural_errors(
    expected: set[str], registered: set[str], duplicates: set[str]
) -> list[str]:
    errors: list[str] = []
    if not expected:
        errors.append("no --test targets found in inference workflows")

    for target in sorted(expected):
        source = TEST_DIR / f"{target}.rs"
        if not source.is_file():
            errors.append(
                f"workflow target {target!r} has no source file at "
                f"{source.relative_to(ROOT)}"
            )

    for target in sorted(duplicates):
        errors.append(f"duplicate explicit [[test]] target in Cargo.toml: {target!r}")

    if LEGACY_CATCH_ALL in registered and LEGACY_CATCH_ALL not in expected:
        errors.append(
            f"legacy catch-all {LEGACY_CATCH_ALL!r} is registered but no inference "
            "workflow invokes it; keep the unqualified surface quarantined"
        )
    return errors


def append_missing_targets(missing: set[str]) -> None:
    if not missing:
        return

    block = ["", MARKER, "# Keep this list explicit because the root package sets autotests = false."]
    for target in sorted(missing):
        block.extend(
            [
                "",
                "[[test]]",
                f'name = "{target}"',
                f'path = "tests/{target}.rs"',
            ]
        )
    block.append("")

    with MANIFEST.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(block))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fix",
        action="store_true",
        help="append missing workflow-proven [[test]] registrations",
    )
    args = parser.parse_args()

    by_workflow = workflow_targets()
    expected = set().union(*by_workflow.values()) if by_workflow else set()
    registered, duplicates = manifest_test_names()

    errors = structural_errors(expected, registered, duplicates)
    if errors:
        print("registration audit cannot safely proceed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    missing = expected - registered
    if args.fix and missing:
        append_missing_targets(missing)
        registered, duplicates = manifest_test_names()
        errors = structural_errors(expected, registered, duplicates)
        if errors:
            print("post-fix structural audit FAILED:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 1
        missing = expected - registered

    print(
        f"inference workflows: {len(by_workflow)}; "
        f"distinct workflow test targets: {len(expected)}"
    )
    for workflow, targets in sorted(by_workflow.items()):
        print(f"  {workflow}: {', '.join(sorted(targets))}")

    if missing:
        print("\nregistration audit FAILED:", file=sys.stderr)
        for target in sorted(missing):
            print(
                f"  - workflow target {target!r} is not explicitly registered in Cargo.toml",
                file=sys.stderr,
            )
        return 1

    print("\nregistration audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
