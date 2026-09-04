#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Prove that focused-only validation is owned by trusted-base implementations.

Execute this script from the trusted BASE checkout. The PR head is inspected only
through Git objects. It intentionally fails until the trusted focused workflow
actually invokes the structural coverage bundle.

Before focused-only routing can be considered, the PR head must preserve:
- the exact trusted focused workflow blob;
- the exact trusted structural-coverage bundle blob;
- an executable audited qualification script;
- an explicit focused-workflow invocation of the structural bundle.

Any validator/workflow edit therefore falls back to full CI until that new
implementation is reviewed on the base branch and becomes the new trust root.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

SCHEMA = "spore-focused-validator-authority-v1"
FOCUSED_WORKFLOW = ".github/workflows/spore-boot-stack.yml"
STRUCTURAL_BUNDLE = "scripts/check-spore-focused-structural-coverage.sh"
QUALIFICATION_SCRIPT = "scripts/check-spore-boot-stack.sh"
STRUCTURAL_INVOCATION = "bash scripts/check-spore-focused-structural-coverage.sh"
_OBJECT_ID = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


class AuthorityError(ValueError):
    pass


def load_builder():
    path = Path(__file__).resolve().with_name("build-spore-routing-authorization.py")
    spec = importlib.util.spec_from_file_location("trusted_spore_authorization_builder", path)
    if spec is None or spec.loader is None:
        raise AuthorityError(f"cannot load trusted authorization builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def trusted_repo() -> Path:
    return Path(__file__).resolve().parent.parent


def trusted_object(builder, path: str) -> str:
    raw = builder.run_git(trusted_repo(), "rev-parse", f"HEAD:{path}")
    value = raw.decode("ascii").strip()
    if not _OBJECT_ID.fullmatch(value):
        raise AuthorityError(f"trusted object id is malformed for {path}")
    return value


def prove(source_root: Path, head: str) -> dict[str, object]:
    builder = load_builder()
    builder.validate_object_id(head, "head")
    actual_head = builder.run_git(source_root, "rev-parse", "HEAD").decode("ascii").strip()
    if actual_head != head:
        raise AuthorityError(f"source HEAD {actual_head} != requested head {head}")

    focused = builder.regular_blob_bytes(source_root, head, FOCUSED_WORKFLOW)
    structural = builder.regular_blob_bytes(source_root, head, STRUCTURAL_BUNDLE)
    qualification = builder.regular_blob_bytes(source_root, head, QUALIFICATION_SCRIPT)
    assert focused is not None and structural is not None and qualification is not None

    focused_raw, focused_mode, focused_object = focused
    structural_raw, structural_mode, structural_object = structural
    qualification_raw, qualification_mode, qualification_object = qualification

    expected_focused = trusted_object(builder, FOCUSED_WORKFLOW)
    expected_structural = trusted_object(builder, STRUCTURAL_BUNDLE)
    if focused_object != expected_focused:
        raise AuthorityError("PR head changed the focused workflow validation authority")
    if structural_object != expected_structural:
        raise AuthorityError("PR head changed the focused structural validation authority")
    if qualification_mode != "100755":
        raise AuthorityError(
            f"qualification script must retain executable Git mode 100755, got {qualification_mode}"
        )

    try:
        focused_text = focused_raw.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise AuthorityError("focused workflow is not UTF-8") from error
    count = focused_text.count(STRUCTURAL_INVOCATION)
    if count != 1:
        raise AuthorityError(
            "trusted focused workflow must invoke the structural coverage bundle exactly once; "
            f"found {count} invocations"
        )

    # The structural bundle may be 100644 because the focused workflow invokes it
    # through `bash`; it still must be an ordinary blob, which the builder already
    # enforced. Qualification is stricter because #406 explicitly requires `-x`.
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "source_commit": head,
        "focused_workflow_object": focused_object,
        "focused_workflow_mode": focused_mode,
        "trusted_focused_workflow_object": expected_focused,
        "structural_bundle_object": structural_object,
        "structural_bundle_mode": structural_mode,
        "trusted_structural_bundle_object": expected_structural,
        "qualification_script_object": qualification_object,
        "qualification_script_mode": qualification_mode,
        "structural_invocation_count": count,
        "structural_invocation": STRUCTURAL_INVOCATION,
    }


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--head")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    try:
        if args.source_root is None or args.head is None or args.receipt is None:
            raise AuthorityError("--source-root, --head, and --receipt are required")
        value = prove(args.source_root, args.head)
        write_json(args.receipt, value)
        print(json.dumps(value, sort_keys=True, indent=2))
        return 0
    except (OSError, UnicodeError, AuthorityError, AssertionError) as error:
        print(f"spore-focused-validator-authority: FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
