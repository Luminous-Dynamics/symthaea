#!/usr/bin/env python3
"""Framed V2 identities for generic qualification admission.

V2 deliberately requires the full qualification-profile preimage. A V3 admission request
contains the exact subject bytes but only a profile ID/name/locator, so the profile theorem
must be independently resolved before V2 work identity can exist.

This module does not reinterpret V3 admission IDs. It validates the complete V3 request,
validates the resolved V1 profile record, derives framed V2 root IDs from their semantic
preimages, and only then derives V2 admission identities.

The two V2 identities remain distinct:

    QualificationAdmissionSubjectIdV2
        = exact qualification subject semantics + exact qualification profile semantics

    QualificationAdmissionRequestIdV2
        = work identity + request/provenance metadata

Changing request rationale or profile locator therefore does not redefine the semantic work
subject, while it does create a distinct request/provenance identity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import integration_train_manifest as train
import qualification_admission_v3 as admission_v3
import qualification_framing_v1 as framing
import qualification_profile as profile_mod
import qualification_semantic_ids_v2 as semantic_v2

SCHEMA = "symthaea.qualification-admission-identities.v2"
SUBJECT_ID_DOMAIN = "symthaea.qualification-admission-subject-id.v2"
REQUEST_ID_DOMAIN = "symthaea.qualification-admission-request-id.v2"


def _encoded_text_set(values: list[str]) -> bytes:
    return framing.encode_set(framing.encode_text(value) for value in values)


def _resolve(admission: Any, profile_value: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    # require_ids=True prevents a malformed/incomplete V3 request from being silently upgraded.
    request = admission_v3.normalize_request(admission, require_ids=True)
    profile = profile_mod.normalize_profile(profile_value, require_id=True)

    expected = request["qualification_profile"]
    if profile["profile_id"] != expected["profile_id"]:
        raise train.TrainManifestError(
            "admission v2 migration: resolved profile bytes do not match V3 profile_id"
        )
    if profile["profile_name"] != expected["name"]:
        raise train.TrainManifestError(
            "admission v2 migration: resolved profile name does not match V3 locator"
        )
    return request, profile


def derive_admission_identities_v2(admission: Any, profile_value: Any) -> dict[str, str]:
    """Validate all V1/V3 preimages and derive framed V2 work/request identities."""

    request, profile = _resolve(admission, profile_value)
    qualification_subject_id_v2 = semantic_v2.compute_subject_id_v2(request["subject"])
    qualification_profile_id_v2 = semantic_v2.compute_profile_id_v2(profile)
    admission_subject_id_v2 = framing.semantic_sha256_id(
        SUBJECT_ID_DOMAIN,
        [
            ("qualification_subject_id_v2", framing.encode_text(qualification_subject_id_v2)),
            ("qualification_profile_id_v2", framing.encode_text(qualification_profile_id_v2)),
        ],
    )

    locator = request["qualification_profile"]
    admission_request_id_v2 = framing.semantic_sha256_id(
        REQUEST_ID_DOMAIN,
        [
            ("admission_subject_id_v2", framing.encode_text(admission_subject_id_v2)),
            ("profile_branch", framing.encode_text(locator["profile_branch"])),
            ("profile_path", framing.encode_text(locator["profile_path"])),
            ("reason", framing.encode_text(request["reason"])),
            ("evidence_refs", _encoded_text_set(request["evidence_refs"])),
            ("non_claims", _encoded_text_set(request["non_claims"])),
        ],
    )

    return {
        "schema": SCHEMA,
        "qualification_subject_id_v2": qualification_subject_id_v2,
        "qualification_profile_id_v2": qualification_profile_id_v2,
        "admission_subject_id_v2": admission_subject_id_v2,
        "admission_request_id_v2": admission_request_id_v2,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("admission", type=Path, help="complete V3 admission JSON")
    parser.add_argument("profile", type=Path, help="resolved complete V1 qualification profile JSON")
    return parser


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise train.TrainManifestError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _load_json(path: Path) -> Any:
    try:
        if path.stat().st_size > train.MAX_MANIFEST_BYTES:
            raise train.TrainManifestError(
                f"{path}: migration input exceeds {train.MAX_MANIFEST_BYTES} bytes"
            )
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except train.TrainManifestError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise train.TrainManifestError(f"{path}: {error}") from error


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        admission = _load_json(args.admission)
        # Reuse the profile module's bounded, duplicate-key-rejecting loader rather than
        # defining a second profile-file parsing contract in the migration bridge.
        profile = profile_mod.load_profile(args.profile, require_id=True)
        result = derive_admission_identities_v2(admission, profile)
    except train.TrainManifestError as error:
        print(f"qualification admission v2 migration invalid: {error}")
        return 2
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
