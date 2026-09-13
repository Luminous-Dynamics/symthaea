#!/usr/bin/env python3
"""Validate V2 qualification-admission requests bound to immutable profile semantics.

V1 admission requests remain historical artifacts. V2 replaces the mutable profile-name string
with an exact profile reference whose semantic `profile_id` is independently verifiable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import integration_train_catalog as catalog
import integration_train_manifest as train
import qualification_admission as admission_v1
import qualification_profile as profile_mod

SCHEMA = "symthaea.qualification-admission-request.v2"
DOMAIN = b"symthaea.qualification-admission-request.v2\0"
SUBJECT_DOMAIN = b"symthaea.qualification-admission-subject.v2\0"


def _canonical_payload_from_normalized(request: dict[str, Any]) -> bytes:
    payload = {key: value for key, value in request.items() if key != "admission_id"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _compute_admission_id_from_normalized(request: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        DOMAIN + _canonical_payload_from_normalized(request)
    ).hexdigest()


def _subject_payload(normalized_request: dict[str, Any]) -> dict[str, Any]:
    profile = normalized_request["qualification_profile"]
    return {
        "program_id": normalized_request["program_id"],
        "catalog_id": normalized_request["catalog_id"],
        "target_train": normalized_request["target_train"],
        # The display name and locator are deliberately excluded. Only the immutable theorem
        # identity participates in work-subject deduplication.
        "qualification_profile_id": profile["profile_id"],
    }


def compute_admission_subject_id(request: Any) -> str:
    normalized = normalize_request(request)
    payload = json.dumps(
        _subject_payload(normalized),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(SUBJECT_DOMAIN + payload).hexdigest()


def normalize_request(
    request: Any, *, verify_declared_id: bool = True, require_id: bool = False
) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise train.TrainManifestError("admission v2: expected object")
    admission_v1._require_exact_keys(
        request,
        {
            "schema",
            "program_id",
            "catalog_id",
            "catalog_branch",
            "catalog_path",
            "target_train",
            "qualification_profile",
            "reason",
            "evidence_refs",
            "non_claims",
        },
        {"admission_id"},
        where="admission v2",
    )
    if request["schema"] != SCHEMA:
        raise train.TrainManifestError(f"admission v2.schema: expected {SCHEMA!r}")
    if require_id and "admission_id" not in request:
        raise train.TrainManifestError("admission v2.admission_id: required but absent")

    raw_target = request["target_train"]
    if not isinstance(raw_target, dict):
        raise train.TrainManifestError("admission v2.target_train: expected object")
    admission_v1._require_exact_keys(
        raw_target,
        {"name", "train_id", "subject_sha"},
        set(),
        where="admission v2.target_train",
    )

    raw_profile = request["qualification_profile"]
    if not isinstance(raw_profile, dict):
        raise train.TrainManifestError("admission v2.qualification_profile: expected object")
    admission_v1._require_exact_keys(
        raw_profile,
        {"name", "profile_id", "profile_branch", "profile_path"},
        set(),
        where="admission v2.qualification_profile",
    )

    normalized = {
        "schema": SCHEMA,
        "program_id": admission_v1._require_string(
            request["program_id"], where="admission v2.program_id"
        ),
        "catalog_id": admission_v1._require_id(
            request["catalog_id"], where="admission v2.catalog_id"
        ),
        "catalog_branch": catalog._require_branch(
            request["catalog_branch"], where="admission v2.catalog_branch"
        ),
        "catalog_path": catalog._require_manifest_path(
            request["catalog_path"], where="admission v2.catalog_path"
        ),
        "target_train": {
            "name": admission_v1._require_string(
                raw_target["name"], where="admission v2.target_train.name"
            ),
            "train_id": admission_v1._require_id(
                raw_target["train_id"], where="admission v2.target_train.train_id"
            ),
            "subject_sha": train._require_sha(
                raw_target["subject_sha"], where="admission v2.target_train.subject_sha"
            ),
        },
        "qualification_profile": {
            "name": admission_v1._require_string(
                raw_profile["name"], where="admission v2.qualification_profile.name"
            ),
            "profile_id": profile_mod._require_profile_id(
                raw_profile["profile_id"], where="admission v2.qualification_profile.profile_id"
            ),
            "profile_branch": catalog._require_branch(
                raw_profile["profile_branch"],
                where="admission v2.qualification_profile.profile_branch",
            ),
            "profile_path": catalog._require_manifest_path(
                raw_profile["profile_path"],
                where="admission v2.qualification_profile.profile_path",
            ),
        },
        "reason": admission_v1._require_string(
            request["reason"], where="admission v2.reason"
        ),
        "evidence_refs": train._require_sorted_unique_strings(
            request["evidence_refs"], where="admission v2.evidence_refs"
        ),
        "non_claims": train._require_sorted_unique_strings(
            request["non_claims"], where="admission v2.non_claims"
        ),
    }
    if not normalized["non_claims"]:
        raise train.TrainManifestError(
            "admission v2.non_claims: must contain at least one explicit non-claim"
        )

    admission_id = _compute_admission_id_from_normalized(normalized)
    normalized["admission_id"] = admission_id
    if verify_declared_id and "admission_id" in request:
        declared = admission_v1._require_id(
            request["admission_id"], where="admission v2.admission_id"
        )
        if declared != admission_id:
            raise train.TrainManifestError(
                f"admission v2.admission_id: expected {admission_id}, got {declared}"
            )
    return normalized


def validate_catalog_record(request: Any, catalog_value: Any) -> None:
    # Reuse the V1 catalog theorem after converting only the profile field to its V1 display form.
    normalized = normalize_request(request)
    v1_shape = {
        "schema": admission_v1.SCHEMA,
        "program_id": normalized["program_id"],
        "catalog_id": normalized["catalog_id"],
        "catalog_branch": normalized["catalog_branch"],
        "catalog_path": normalized["catalog_path"],
        "target_train": normalized["target_train"],
        "qualification_profile": normalized["qualification_profile"]["name"],
        "reason": normalized["reason"],
        "evidence_refs": normalized["evidence_refs"],
        "non_claims": normalized["non_claims"],
    }
    admission_v1.validate_catalog_record(v1_shape, catalog_value)


def validate_profile_record(request: Any, profile_value: Any) -> None:
    normalized = normalize_request(request)
    normalized_profile = profile_mod.normalize_profile(profile_value, require_id=True)
    declared = normalized["qualification_profile"]
    if normalized_profile["profile_name"] != declared["name"]:
        raise train.TrainManifestError(
            "admission v2.qualification_profile.name: does not match referenced profile"
        )
    if normalized_profile["profile_id"] != declared["profile_id"]:
        raise train.TrainManifestError(
            "admission v2.qualification_profile.profile_id: does not match referenced profile bytes"
        )


def validate_bindings(request: Any, repo: Path) -> None:
    normalized = normalize_request(request)
    repo = admission_v1._require_git_repo(repo)

    catalog_spec = f"{normalized['catalog_branch']}:{normalized['catalog_path']}"
    raw_catalog_text = admission_v1._run_git(repo, "show", catalog_spec)
    try:
        raw_catalog = json.loads(
            raw_catalog_text, object_pairs_hook=admission_v1._object_without_duplicate_keys
        )
    except json.JSONDecodeError as error:
        raise train.TrainManifestError(
            f"invalid catalog JSON at {catalog_spec}: {error}"
        ) from error
    validate_catalog_record(normalized, raw_catalog)
    catalog.validate_git_bindings(raw_catalog, repo)

    declared_profile = normalized["qualification_profile"]
    profile_spec = f"{declared_profile['profile_branch']}:{declared_profile['profile_path']}"
    raw_profile_text = admission_v1._run_git(repo, "show", profile_spec)
    try:
        raw_profile = json.loads(
            raw_profile_text, object_pairs_hook=admission_v1._object_without_duplicate_keys
        )
    except json.JSONDecodeError as error:
        raise train.TrainManifestError(
            f"invalid qualification profile JSON at {profile_spec}: {error}"
        ) from error
    validate_profile_record(normalized, raw_profile)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    return admission_v1._object_without_duplicate_keys(pairs)


def load_request(path: Path, *, require_id: bool = False) -> dict[str, Any]:
    try:
        if path.stat().st_size > train.MAX_MANIFEST_BYTES:
            raise train.TrainManifestError(
                f"{path}: admission v2 request exceeds {train.MAX_MANIFEST_BYTES} bytes"
            )
        raw = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_object_without_duplicate_keys
        )
    except train.TrainManifestError:
        raise
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise train.TrainManifestError(f"{path}: {error}") from error
    return normalize_request(raw, require_id=require_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path, help="qualification admission V2 JSON")
    parser.add_argument("--require-id", action="store_true", help="require exact computed admission_id")
    parser.add_argument(
        "--verify-bindings",
        action="store_true",
        help="resolve and verify both catalog and qualification profile",
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd(), help="Git repository used by --verify-bindings")
    parser.add_argument("--print-normalized", action="store_true", help="print canonical normalized JSON")
    parser.add_argument("--print-subject-id", action="store_true", help="print semantic AdmissionSubjectId instead of AdmissionRequestId")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        normalized = load_request(args.request, require_id=args.require_id)
        if args.verify_bindings:
            validate_bindings(normalized, args.repo)
    except train.TrainManifestError as error:
        print(f"qualification admission v2 invalid: {error}", file=sys.stderr)
        return 2
    if args.print_normalized:
        print(json.dumps(normalized, indent=2, ensure_ascii=False))
    elif args.print_subject_id:
        print(compute_admission_subject_id(normalized))
    else:
        print(normalized["admission_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
