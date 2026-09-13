#!/usr/bin/env python3
"""Independently verify a qualification environment selection against an exact Git subject.

`qualification_environment_selection_v1` can validate the internal shape and content identity of
an imported selection, but an imported record must not gain authority merely because it contains
three hash-shaped selector blobs. This verifier re-resolves every frozen selector path from the
exact subject commit in a local Git object database and requires exact blob equality.

Positive theorem:

    valid exact subject + valid whole-tree input closure + valid environment selection
    + local Git reconstruction of every selector path/blob
        -> EnvironmentSelectionGitBindingVerifiedOnly

This does not prove Nix evaluation, realization, execution, provider authenticity, repository
remote ownership, or recipe execution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import integration_train_manifest as train
import qualification_environment_selection_v1 as selection_mod
import qualification_input_closure_v1 as closure_mod
import qualification_subject as subject_mod

POSITIVE_STATE = "EnvironmentSelectionGitBindingVerifiedOnly"


def verify_selection_git_binding(
    selection: Any,
    subject: Any,
    closure: Any,
    repo: Path,
) -> dict[str, Any]:
    normalized_subject = subject_mod.normalize_subject(subject, require_id=True)
    normalized_closure = closure_mod.validate_git_binding(closure, normalized_subject, repo)
    normalized_selection = selection_mod.validate_against_closure(selection, normalized_closure)

    expected_by_path = {
        item["path"]: item["git_blob"] for item in normalized_selection["selector_blobs"]
    }
    if tuple(sorted(expected_by_path)) != tuple(sorted(selection_mod.SELECTOR_PATHS)):
        # Defensive check: normalize_selection already proves this exact set.
        raise train.TrainManifestError(
            "qualification environment selection verifier: selector path set drift"
        )

    verified: list[dict[str, str]] = []
    for path in selection_mod.SELECTOR_PATHS:
        actual_blob = subject_mod._run_git(
            repo,
            "rev-parse",
            f"{normalized_subject['source_commit']}:{path}",
        )
        subject_mod._run_git(repo, "cat-file", "-e", f"{actual_blob}^{{blob}}")
        expected_blob = expected_by_path[path]
        if actual_blob != expected_blob:
            raise train.TrainManifestError(
                f"qualification environment selection verifier: {path} blob mismatch: "
                f"expected {expected_blob}, reconstructed {actual_blob}"
            )
        verified.append({"path": path, "git_blob": actual_blob})

    return {
        "state": POSITIVE_STATE,
        "qualification_subject_id": normalized_subject["subject_id"],
        "input_closure_id": normalized_closure["input_closure_id"],
        "environment_selection_id": normalized_selection["environment_selection_id"],
        "verified_selector_blobs": verified,
        "non_claims": [
            "does not authenticate the declared repository namespace",
            "does not prove Nix evaluated the selected flake output",
            "does not prove any Nix store closure was realized",
            "does not prove observed qualification tools came from the selection",
            "does not prove the qualification recipe executed",
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selection", type=Path)
    parser.add_argument("subject", type=Path)
    parser.add_argument("closure", type=Path)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        selection = selection_mod.load_selection(args.selection, require_id=True)
        subject = subject_mod.load_subject(args.subject, require_id=True)
        closure = closure_mod.load_closure(args.closure, require_id=True)
        witness = verify_selection_git_binding(selection, subject, closure, args.repo)
    except train.TrainManifestError as error:
        print(f"qualification environment selection Git binding invalid: {error}")
        return 2
    print(witness["environment_selection_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
