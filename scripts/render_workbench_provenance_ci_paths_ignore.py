#!/usr/bin/env python3
"""Render the exact v1 Workbench-provenance paths-ignore change for ci.yml.

The renderer is intentionally tied to the reviewed main-branch ci.yml blob. It
refuses to patch any other input, making concurrent workflow drift a review gate
rather than silently applying the optimization to changed CI semantics.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

EXPECTED_INPUT_GIT_BLOB = "a48366076b30eb8e12d22c927a3b8bf333181409"
PULL_REQUEST_STANZA = "  pull_request:\n"
PATCHED_PULL_REQUEST_PREFIX = "  pull_request:\n    paths-ignore:\n"

EXEMPT_PATHS = (
    ".github/workflows/workbench-execution-capsule-profile.yml",
    ".github/workflows/workbench-invocation-isolation-profile.yml",
    ".github/workflows/workbench-nix-closure-capture-verifier.yml",
    ".github/workflows/workbench-nix-closure-capture.yml",
    ".github/workflows/workbench-nix-closure-identity.yml",
    ".github/workflows/workbench-root-nar-membership.yml",
    "data/neuroscience/workbench_execution_capsule_profile_v1.json",
    "data/neuroscience/workbench_invocation_isolation_profile_v1.json",
    "docs/neuroscience/WORKBENCH_EXECUTION_CAPSULE_PROFILE_V1.md",
    "docs/neuroscience/WORKBENCH_INVOCATION_ISOLATION_PROFILE_V1.md",
    "docs/neuroscience/WORKBENCH_NIX_CLOSURE_CAPTURE_V1.md",
    "docs/neuroscience/WORKBENCH_NIX_CLOSURE_CAPTURE_VERIFIER_V1.md",
    "docs/neuroscience/WORKBENCH_NIX_CLOSURE_IDENTITY_V1.md",
    "docs/neuroscience/WORKBENCH_ROOT_NAR_MEMBERSHIP_V1.md",
    "scripts/check_workbench_invocation_isolation_profile_static.py",
    "scripts/test_verify_workbench_execution_capsule_profile.py",
    "scripts/test_verify_workbench_invocation_isolation_profile.py",
    "scripts/test_verify_workbench_nix_closure_capture.py",
    "scripts/test_workbench_nix_closure_capture.py",
    "scripts/test_workbench_nix_closure_identity.py",
    "scripts/test_workbench_root_nar_membership.py",
    "scripts/test_workbench_root_nar_target_spelling.py",
    "scripts/verify_workbench_execution_capsule_profile.py",
    "scripts/verify_workbench_invocation_isolation_profile.py",
    "scripts/verify_workbench_nix_closure_capture.py",
    "scripts/workbench_nix_closure_capture.py",
    "scripts/workbench_nix_closure_identity.py",
    "scripts/workbench_root_nar_membership.py",
)


class RenderError(ValueError):
    pass


def git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()


def rendered_stanza() -> str:
    lines = ["  pull_request:", "    paths-ignore:"]
    lines.extend(f"      - '{path}'" for path in EXEMPT_PATHS)
    return "\n".join(lines) + "\n"


def render_bytes(source: bytes, *, require_reviewed_input: bool = True) -> bytes:
    if require_reviewed_input and git_blob_sha(source) != EXPECTED_INPUT_GIT_BLOB:
        raise RenderError("ci.yml: input Git blob is not the reviewed v1 source")
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RenderError("ci.yml: UTF-8 required") from exc
    if PATCHED_PULL_REQUEST_PREFIX in text:
        raise RenderError("ci.yml: pull_request stanza already has paths-ignore")
    if text.count(PULL_REQUEST_STANZA) != 1:
        raise RenderError("ci.yml: expected exactly one bare pull_request stanza")
    rendered = text.replace(PULL_REQUEST_STANZA, rendered_stanza(), 1)
    if rendered == text:
        raise RenderError("ci.yml: renderer made no change")
    return rendered.encode("utf-8")


def verify_render(source: bytes, candidate: bytes, *, require_reviewed_input: bool = True) -> None:
    expected = render_bytes(source, require_reviewed_input=require_reviewed_input)
    if candidate != expected:
        raise RenderError("ci.yml: candidate differs from exact v1 transformation")


def write_exclusive(path: Path, data: bytes) -> None:
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError as exc:
        raise RenderError("output exists; overwrite forbidden") from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        source = args.input.read_bytes()
        rendered = render_bytes(source)
        write_exclusive(args.output, rendered)
        verify_render(source, args.output.read_bytes())
    except (OSError, RenderError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"input_git_blob={git_blob_sha(source)}")
    print(f"output_git_blob={git_blob_sha(rendered)}")
    print(f"exempt_path_count={len(EXEMPT_PATHS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
