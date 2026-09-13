#!/usr/bin/env python3
"""Generic V3 qualification admission: exact subject + immutable profile."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import integration_train_catalog as catalog
import integration_train_manifest as train
import qualification_profile as profile_mod
import qualification_subject as subject_mod

SCHEMA = "symthaea.qualification-admission-request.v3"
DOMAIN = b"symthaea.qualification-admission-request.v3\0"
SUBJECT_DOMAIN = b"symthaea.qualification-admission-subject.v3\0"


def _canonical_payload(request: dict[str, Any]) -> bytes:
    payload = {
        key: value
        for key, value in request.items()
        if key not in {"admission_id", "admission_subject_id"}
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _request_id(request: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(DOMAIN + _canonical_payload(request)).hexdigest()


def _work_payload(request: dict[str, Any]) -> dict[str, str]:
    return {
        "qualification_subject_id": request["subject"]["subject_id"],
        "qualification_profile_id": request["qualification_profile"]["profile_id"],
    }


def _work_id(request: dict[str, Any]) -> str:
    raw = json.dumps(_work_payload(request), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(SUBJECT_DOMAIN + raw).hexdigest()


def normalize_request(
    request: Any,
    *,
    verify_declared_ids: bool = True,
    require_ids: bool = False,
) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise train.TrainManifestError("admission v3: expected object")
    subject_mod._require_exact_keys(
        request,
        {"schema", "subject", "qualification_profile", "reason", "evidence_refs", "non_claims"},
        {"admission_id", "admission_subject_id"},
        where="admission v3",
    )
    if request["schema"] != SCHEMA:
        raise train.TrainManifestError(f"admission v3.schema: expected {SCHEMA!r}")
    if require_ids:
        for field in ("admission_id", "admission_subject_id"):
            if field not in request:
                raise train.TrainManifestError(f"admission v3.{field}: required but absent")

    subject = subject_mod.normalize_subject(request["subject"], require_id=True)
    raw_profile = request["qualification_profile"]
    if not isinstance(raw_profile, dict):
        raise train.TrainManifestError("admission v3.qualification_profile: expected object")
    subject_mod._require_exact_keys(
        raw_profile,
        {"name", "profile_id", "profile_branch", "profile_path"},
        set(),
        where="admission v3.qualification_profile",
    )
    profile = {
        "name": subject_mod._require_string(raw_profile["name"], where="admission v3.qualification_profile.name"),
        "profile_id": profile_mod._require_profile_id(raw_profile["profile_id"], where="admission v3.qualification_profile.profile_id"),
        "profile_branch": catalog._require_branch(raw_profile["profile_branch"], where="admission v3.qualification_profile.profile_branch"),
        "profile_path": catalog._require_manifest_path(raw_profile["profile_path"], where="admission v3.qualification_profile.profile_path"),
    }
    normalized: dict[str, Any] = {
        "schema": SCHEMA,
        "subject": subject,
        "qualification_profile": profile,
        "reason": subject_mod._require_string(request["reason"], where="admission v3.reason"),
        "evidence_refs": train._require_sorted_unique_strings(request["evidence_refs"], where="admission v3.evidence_refs"),
        "non_claims": train._require_sorted_unique_strings(request["non_claims"], where="admission v3.non_claims"),
    }
    if not normalized["non_claims"]:
        raise train.TrainManifestError("admission v3.non_claims: must contain at least one explicit non-claim")

    admission_id = _request_id(normalized)
    admission_subject_id = _work_id(normalized)
    normalized["admission_id"] = admission_id
    normalized["admission_subject_id"] = admission_subject_id

    if verify_declared_ids:
        if "admission_id" in request and request["admission_id"] != admission_id:
            raise train.TrainManifestError(f"admission v3.admission_id: expected {admission_id}, got {request['admission_id']}")
        if "admission_subject_id" in request and request["admission_subject_id"] != admission_subject_id:
            raise train.TrainManifestError("admission v3.admission_subject_id: subject/profile semantics changed")
    return normalized


def compute_admission_id(request: Any) -> str:
    return normalize_request(request, verify_declared_ids=False)["admission_id"]


def compute_admission_subject_id(request: Any) -> str:
    return normalize_request(request, verify_declared_ids=False)["admission_subject_id"]


def validate_profile_record(request: Any, profile_value: Any) -> None:
    normalized = normalize_request(request, require_ids=True)
    profile = profile_mod.normalize_profile(profile_value, require_id=True)
    expected = normalized["qualification_profile"]
    if profile["profile_id"] != expected["profile_id"]:
        raise train.TrainManifestError("admission v3 profile id does not match resolved profile bytes")
    if profile["profile_name"] != expected["name"]:
        raise train.TrainManifestError("admission v3 profile name does not match resolved profile")


def work_payload(request: Any) -> dict[str, str]:
    return _work_payload(normalize_request(request))
