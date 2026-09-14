#!/usr/bin/env python3
"""Qualification attempt V4 with content-addressed evidence descriptors.

V4 preserves the V3 attempt-registration identity and changes only the observation/history layer.
A provider locator says where execution was observed. `evidence_content_ids` name exact content by
canonical digest/blob descriptors. A positive `Passed` observation must name at least one evidence
content descriptor; non-PASS observations may legitimately have no artifact content at all.

Descriptor validity alone does NOT establish that referenced bytes exist, that the digest was
recomputed over those bytes, who produced them, whether the producer is trusted, whether the
content is correct, or whether it is sufficient for qualification.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import integration_train_manifest as train
import qualification_admission_v3 as admission_mod
import qualification_attempt_v3 as v3
import qualification_evidence_id as evidence_id_mod
import qualification_profile as profile_mod
import qualification_subject as subject_mod

REGISTRATION_SCHEMA = v3.REGISTRATION_SCHEMA
OBSERVATION_SCHEMA = "symthaea.qualification-attempt-observation.v4"
HISTORY_SCHEMA = "symthaea.qualification-attempt-history.v4"

OBSERVATION_DOMAIN = b"symthaea.qualification-attempt-observation.v4\0"
HISTORY_DOMAIN = b"symthaea.qualification-attempt-history.v4\0"

PRESTART_TERMINALS = set(v3.PRESTART_TERMINALS)
POSTSTART_TERMINALS = set(v3.POSTSTART_TERMINALS)
ALL_TERMINALS = PRESTART_TERMINALS | POSTSTART_TERMINALS

normalize_registration = v3.normalize_registration
validate_registration_against_admission = v3.validate_registration_against_admission


def _canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _id(domain: bytes, value: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(domain + _canon(value)).hexdigest()


def _require_id(value: Any, *, where: str) -> str:
    return v3._require_id(value, where=where)


def normalize_observation(
    value: Any, *, verify_declared_id: bool = True, require_id: bool = False
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise train.TrainManifestError("attempt observation V4: expected object")
    subject_mod._require_exact_keys(
        value,
        {
            "schema",
            "attempt_registration_id",
            "execution_started",
            "terminal_disposition",
            "provider_ref",
            "evidence_content_ids",
            "non_claims",
        },
        {"attempt_observation_id"},
        where="attempt observation V4",
    )
    if value["schema"] != OBSERVATION_SCHEMA:
        raise train.TrainManifestError(
            f"attempt observation V4.schema: expected {OBSERVATION_SCHEMA!r}"
        )
    if require_id and "attempt_observation_id" not in value:
        raise train.TrainManifestError("attempt observation V4.attempt_observation_id: required")

    execution_started = v3._require_bool(
        value["execution_started"], where="attempt observation V4.execution_started"
    )
    terminal = subject_mod._require_string(
        value["terminal_disposition"], where="attempt observation V4.terminal_disposition"
    )
    if terminal not in ALL_TERMINALS:
        raise train.TrainManifestError(
            "attempt observation V4.terminal_disposition: unsupported disposition"
        )
    if execution_started and terminal in PRESTART_TERMINALS:
        raise train.TrainManifestError(
            "attempt observation V4: pre-start terminal is incompatible with execution_started=true"
        )
    if not execution_started and terminal in POSTSTART_TERMINALS:
        raise train.TrainManifestError(
            "attempt observation V4: post-start terminal is incompatible with execution_started=false"
        )

    evidence_content_ids = evidence_id_mod.require_sorted_unique_evidence_content_ids(
        value["evidence_content_ids"], where="attempt observation V4.evidence_content_ids"
    )
    if terminal == "Passed" and not evidence_content_ids:
        raise train.TrainManifestError(
            "attempt observation V4: Passed requires at least one evidence content-id descriptor"
        )

    normalized: dict[str, Any] = {
        "schema": OBSERVATION_SCHEMA,
        "attempt_registration_id": _require_id(
            value["attempt_registration_id"],
            where="attempt observation V4.attempt_registration_id",
        ),
        "execution_started": execution_started,
        "terminal_disposition": terminal,
        "provider_ref": v3._require_provider_ref(value["provider_ref"]),
        "evidence_content_ids": evidence_content_ids,
        "non_claims": train._require_sorted_unique_strings(
            value["non_claims"], where="attempt observation V4.non_claims"
        ),
    }
    if not normalized["non_claims"]:
        raise train.TrainManifestError("attempt observation V4.non_claims: must not be empty")

    payload = dict(normalized)
    attempt_observation_id = _id(OBSERVATION_DOMAIN, payload)
    normalized["attempt_observation_id"] = attempt_observation_id
    if verify_declared_id and "attempt_observation_id" in value:
        declared = _require_id(
            value["attempt_observation_id"],
            where="attempt observation V4.attempt_observation_id",
        )
        if declared != attempt_observation_id:
            raise train.TrainManifestError(
                "attempt observation V4.attempt_observation_id: observation semantics changed"
            )
    return normalized


def validate_observation_against_registration(
    observation: Any, registration: Any
) -> dict[str, Any]:
    obs = normalize_observation(observation, require_id=True)
    reg = normalize_registration(registration, require_id=True)
    if obs["attempt_registration_id"] != reg["attempt_registration_id"]:
        raise train.TrainManifestError(
            "attempt observation V4: does not bind the exact attempt registration"
        )
    return obs


def build_history(
    *,
    admission: Any,
    profile: Any,
    registrations: list[Any],
    observations: list[Any],
) -> dict[str, Any]:
    adm = admission_mod.normalize_request(admission, require_ids=True)
    prof = profile_mod.normalize_profile(profile, require_id=True)
    admission_mod.validate_profile_record(adm, prof)
    if not registrations:
        raise train.TrainManifestError("attempt history V4: expected at least one registration")

    regs = [validate_registration_against_admission(item, adm, prof) for item in registrations]
    reg_by_id = {item["attempt_registration_id"]: item for item in regs}
    if len(reg_by_id) != len(regs):
        raise train.TrainManifestError("attempt history V4: duplicate attempt registration id")

    sequence_keys = [(item["recipe_id"], item["attempt_sequence"]) for item in regs]
    if len(sequence_keys) != len(set(sequence_keys)):
        raise train.TrainManifestError(
            "attempt history V4: duplicate attempt_sequence for the same recipe"
        )

    obs = [normalize_observation(item, require_id=True) for item in observations]
    obs_by_registration: dict[str, dict[str, Any]] = {}
    for item in obs:
        registration_id = item["attempt_registration_id"]
        if registration_id not in reg_by_id:
            raise train.TrainManifestError(
                "attempt history V4: orphan observation has no registered attempt"
            )
        if registration_id in obs_by_registration:
            raise train.TrainManifestError(
                "attempt history V4: a registration may have at most one terminal observation"
            )
        obs_by_registration[registration_id] = item

    represented_recipes = sorted({item["recipe_id"] for item in regs})
    required_recipes = prof["required_recipe_ids"]
    extra = sorted(set(represented_recipes) - set(required_recipes))
    if extra:
        raise train.TrainManifestError(
            "attempt history V4: contains recipe registrations outside the exact profile"
        )

    normalized = {
        "schema": HISTORY_SCHEMA,
        "admission_subject_id": adm["admission_subject_id"],
        "qualification_subject_id": adm["subject"]["subject_id"],
        "qualification_profile_id": prof["profile_id"],
        "required_recipe_ids": list(required_recipes),
        "represented_recipe_ids": represented_recipes,
        "attempt_registration_ids": sorted(reg_by_id),
        "terminal_observation_ids": sorted(item["attempt_observation_id"] for item in obs),
        "non_claims": [
            "content-id descriptor validity does not establish referenced bytes exist or match the digest",
            "content identity does not establish evidence authenticity, correctness, trust, or sufficiency",
            "a non-PASS observation with no evidence content does not prove no diagnostics or provider evidence exist",
            "does not erase failed or infrastructure attempts when a later retry passes",
            "does not establish externally anchored chronology between attempt_sequence values",
            "does not establish qualification merely because every recipe has an attempt",
            "does not prove required cross-cutting rules",
        ],
    }
    normalized["attempt_history_id"] = _id(HISTORY_DOMAIN, normalized)
    return normalized


def successful_attempts_by_recipe(
    *, registrations: list[Any], observations: list[Any]
) -> dict[str, list[str]]:
    regs = [normalize_registration(item, require_id=True) for item in registrations]
    obs = [normalize_observation(item, require_id=True) for item in observations]
    reg_by_id = {item["attempt_registration_id"]: item for item in regs}
    out: dict[str, list[str]] = {}
    for item in obs:
        if item["terminal_disposition"] != "Passed":
            continue
        reg = reg_by_id.get(item["attempt_registration_id"])
        if reg is None:
            raise train.TrainManifestError("successful attempt lookup V4: orphan observation")
        out.setdefault(reg["recipe_id"], []).append(item["attempt_observation_id"])
    for recipe_id in out:
        out[recipe_id].sort()
    return out
