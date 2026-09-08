#!/usr/bin/env python3
"""Pure verifier for registered GitHub Actions workflow-definition identity."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

SCHEMA = "symthaea-ci-workflow-definition-identity-v1"
BLOB_RE = re.compile(r"^[0-9a-f]{40}$")
WORKFLOW_RE = re.compile(r"^\.github/workflows/[a-z0-9][a-z0-9-]*\.yml$")


class WorkflowIdentityError(ValueError):
    pass


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def _workflow_path(value: Any) -> str:
    if not isinstance(value, str) or not WORKFLOW_RE.fullmatch(value):
        raise WorkflowIdentityError("workflow path must be canonical .github/workflows/*.yml")
    pure = PurePosixPath(value)
    if value != "/".join(pure.parts) or any(part in {"", ".", ".."} for part in pure.parts):
        raise WorkflowIdentityError("workflow path must use exact canonical POSIX spelling")
    return value


def _blob(value: Any) -> str:
    if not isinstance(value, str) or not BLOB_RE.fullmatch(value):
        raise WorkflowIdentityError("expected workflow Git blob must be 40 lowercase hex")
    return value


def _observe(root_value: Path, workflow: str, label: str) -> str:
    try:
        root = root_value.resolve(strict=True)
    except OSError as exc:
        raise WorkflowIdentityError(f"{label} root unavailable") from exc
    target = root.joinpath(*PurePosixPath(workflow).parts)
    if target.is_symlink():
        raise WorkflowIdentityError(f"{label} workflow symlink forbidden")
    try:
        resolved = target.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise WorkflowIdentityError(f"{label} workflow unavailable/outside root") from exc
    if not resolved.is_file():
        raise WorkflowIdentityError(f"{label} workflow must be a regular file")
    return git_blob_sha(resolved.read_bytes())


def verify_workflow_definition(
    *,
    workflow: Any,
    expected_git_blob: Any,
    head_root: Path,
    executing_root: Path,
) -> dict[str, Any]:
    path = _workflow_path(workflow)
    expected = _blob(expected_git_blob)
    head_blob = _observe(head_root, path, "head")
    executing_blob = _observe(executing_root, path, "executing")

    if head_blob != expected:
        raise WorkflowIdentityError("head workflow Git blob does not match registered blob")
    if executing_blob != expected:
        raise WorkflowIdentityError("executing workflow Git blob does not match registered blob")
    if head_blob != executing_blob:
        raise WorkflowIdentityError("head and executing workflow blobs differ")

    return {
        "schema": SCHEMA,
        "workflow": path,
        "expected_workflow_git_blob": expected,
        "head_workflow_git_blob": head_blob,
        "executing_workflow_git_blob": executing_blob,
        "authority": {
            "workflow_definition_identity_verified": True,
            "focused_theorem_passed": False,
            "merge_compatibility_passed": False,
            "scientific_execution_qualified": False,
            "transform_executed": False,
            "fmq010_established": False,
            "neural_alignment_established": False,
            "consciousness_evidence": False,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--expected-git-blob", required=True)
    parser.add_argument("--head-root", required=True, type=Path)
    parser.add_argument("--executing-root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify_workflow_definition(
            workflow=args.workflow,
            expected_git_blob=args.expected_git_blob,
            head_root=args.head_root,
            executing_root=args.executing_root,
        )
    except (OSError, WorkflowIdentityError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
