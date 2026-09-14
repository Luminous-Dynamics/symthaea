#!/usr/bin/env python3
"""Conservative exact qualification input closure V1.

V1 intentionally chooses soundness over routing efficiency:

    exact Git object format + exact repository root tree
        -> QualificationInputClosureIdV1

The closure covers every *tracked repository-controlled input* in the exact Git tree. Therefore
an unrelated tracked repository change also changes the closure and requires requalification.
A later narrow closure is allowed only under a separately qualified completeness algorithm; it
must use a distinct kind/schema and must not reinterpret this V1 identity.

This object does not authenticate repository origin, realize registry/network dependencies,
select the qualification environment, or prove that an executor actually checked out the tree.
Those are separate theorems.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import integration_train_manifest as train
import qualification_framing_v1 as framing
import qualification_subject as subject_mod

SCHEMA = "symthaea.qualification-input-closure.v1"
ID_DOMAIN = "symthaea.qualification-input-closure-id.v1"
KIND = "WholeRepositoryGitTree"
COVERAGE = "AllTrackedRepositoryControlledInputs"

NON_CLAIMS = [
    "does not authenticate the declared repository namespace",
    "does not prove external dependency bytes were realized",
    "does not prove the executor checked out this tree",
    "does not select or qualify an execution environment",
]


def _fields(value: dict[str, Any]) -> list[tuple[str, bytes]]:
    return [
        ("kind", framing.encode_enum(value["kind"])),
        ("coverage", framing.encode_enum(value["coverage"])),
        ("object_format", framing.encode_enum(value["object_format"])),
        ("source_tree", framing.encode_text(value["source_tree"])),
        (
            "non_claims",
            framing.encode_set(framing.encode_text(item) for item in value["non_claims"]),
        ),
    ]


def build_from_subject(subject: Any) -> dict[str, Any]:
    """Build the whole-tree closure from a validated qualification subject preimage."""

    normalized_subject = subject_mod.normalize_subject(subject, verify_declared_id=True)
    normalized: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": KIND,
        "coverage": COVERAGE,
        "object_format": normalized_subject["object_format"],
        "source_tree": normalized_subject["source_tree"],
        "non_claims": list(NON_CLAIMS),
    }
    normalized["input_closure_id"] = framing.semantic_sha256_id(ID_DOMAIN, _fields(normalized))
    return normalized


def normalize_closure(
    value: Any, *, verify_declared_id: bool = True, require_id: bool = False
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise train.TrainManifestError("qualification input closure: expected object")
    subject_mod._require_exact_keys(
        value,
        {"schema", "kind", "coverage", "object_format", "source_tree", "non_claims"},
        {"input_closure_id"},
        where="qualification input closure",
    )
    if value["schema"] != SCHEMA:
        raise train.TrainManifestError(
            f"qualification input closure.schema: expected {SCHEMA!r}"
        )
    if value["kind"] != KIND or value["coverage"] != COVERAGE:
        raise train.TrainManifestError(
            "qualification input closure: unsupported V1 kind/coverage semantics"
        )
    if require_id and "input_closure_id" not in value:
        raise train.TrainManifestError("qualification input closure.input_closure_id: required")

    object_format = subject_mod._require_object_format(
        value["object_format"], where="qualification input closure.object_format"
    )
    non_claims = train._require_sorted_unique_strings(
        value["non_claims"], where="qualification input closure.non_claims"
    )
    if non_claims != NON_CLAIMS:
        raise train.TrainManifestError(
            "qualification input closure.non_claims: V1 theorem boundary must equal the frozen set"
        )

    normalized: dict[str, Any] = {
        "schema": SCHEMA,
        "kind": KIND,
        "coverage": COVERAGE,
        "object_format": object_format,
        "source_tree": subject_mod._require_object_id(
            value["source_tree"],
            where="qualification input closure.source_tree",
            object_format=object_format,
        ),
        "non_claims": list(NON_CLAIMS),
    }
    expected_id = framing.semantic_sha256_id(ID_DOMAIN, _fields(normalized))
    normalized["input_closure_id"] = expected_id
    if verify_declared_id and "input_closure_id" in value:
        declared = value["input_closure_id"]
        if (
            not isinstance(declared, str)
            or not declared.startswith("sha256:")
            or len(declared) != 71
            or any(char not in "0123456789abcdef" for char in declared[7:])
        ):
            raise train.TrainManifestError(
                "qualification input closure.input_closure_id: expected sha256:<64 lowercase hex>"
            )
        if declared != expected_id:
            raise train.TrainManifestError(
                "qualification input closure.input_closure_id: closure semantics changed"
            )
    return normalized


def validate_against_subject(closure: Any, subject: Any) -> dict[str, Any]:
    normalized = normalize_closure(closure, require_id=True)
    normalized_subject = subject_mod.normalize_subject(subject, require_id=True)
    if normalized["object_format"] != normalized_subject["object_format"]:
        raise train.TrainManifestError(
            "qualification input closure: Git object format does not match subject"
        )
    if normalized["source_tree"] != normalized_subject["source_tree"]:
        raise train.TrainManifestError(
            "qualification input closure: source tree does not match subject"
        )
    return normalized


def validate_git_binding(closure: Any, subject: Any, repo: Path) -> dict[str, Any]:
    """Additionally prove the subject commit exists locally and resolves to this exact tree."""

    normalized = validate_against_subject(closure, subject)
    subject_mod.validate_git_binding(subject, repo)
    return normalized


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in pairs:
        if key in result:
            raise train.TrainManifestError(f"duplicate JSON object key: {key!r}")
        result[key] = item
    return result


def load_closure(path: Path, *, require_id: bool = False) -> dict[str, Any]:
    try:
        if path.stat().st_size > train.MAX_MANIFEST_BYTES:
            raise train.TrainManifestError(
                f"{path}: input closure exceeds {train.MAX_MANIFEST_BYTES} bytes"
            )
        raw = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_object_without_duplicate_keys
        )
    except train.TrainManifestError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise train.TrainManifestError(f"{path}: {error}") from error
    return normalize_closure(raw, require_id=require_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("subject", type=Path)

    verify = subparsers.add_parser("verify")
    verify.add_argument("closure", type=Path)
    verify.add_argument("subject", type=Path)
    verify.add_argument("--repo", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        subject = subject_mod.load_subject(args.subject, require_id=True)
        if args.command == "build":
            print(json.dumps(build_from_subject(subject), sort_keys=True, indent=2))
            return 0

        closure = load_closure(args.closure, require_id=True)
        if args.repo is None:
            validate_against_subject(closure, subject)
        else:
            validate_git_binding(closure, subject, args.repo)
    except train.TrainManifestError as error:
        print(f"qualification input closure invalid: {error}")
        return 2
    print(closure["input_closure_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
