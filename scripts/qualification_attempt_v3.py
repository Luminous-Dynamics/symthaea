#!/usr/bin/env python3
"""Generic content-addressed qualification attempt registration and terminal evidence.

V3 deliberately separates:

- qualification work identity: exact source subject + immutable qualification profile;
- attempt registration identity: one declared retry/attempt for one exact recipe;
- provider provenance: useful locator metadata, never the semantic work identity;
- terminal observation: what happened to that registered attempt;
- profile attempt history: append-only collection of immutable observations.

This module does not prove externally anchored chronology. `attempt_sequence` is a declared,
subject-scoped retry discriminator until a stronger event-anchoring theorem is composed.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import integration_train_manifest as train
import qualification_admission_v3 as admission_mod
import qualification_profile as profile_mod
import qualification_subject as subject_mod

REGISTRATION_SCHEMA = "symthaea.qualification-attempt-registration.v3"
OBSERVATION_SCHEMA = "symthaea.qualification-attempt-observation.v3"
HISTORY_SCHEMA = "symthaea.qualification-attempt-history.v3"

REGISTRATION_DOMAIN = b"symthaea.qualification-attempt-registration.v3\0"
OBSERVATION_DOMAIN = b"symthaea.qualification-attempt-observation.v3\0"
HISTORY_DOMAIN = b"symthaea.qualification-attempt-history.v3\0"

_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

PRESTART_TERMINALS = {
    "CancelledBeforeStart",
    "InfrastructureUnavailableBeforeStart",
    "OutcomeUnknownBeforeStart",
}
POSTSTART_TERMINALS = {
    "Passed",
    "RecipeFailed",
    "CancelledAfterStart",
    "InfrastructureUnavailableAfterStart",
    "OutcomeUnknownAfterStart",
}
ALL_TERMINALS = PRESTART_TERMINALS | POSTSTART_TERMINALS


def _canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _id(domain: bytes, value: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(domain + _canon(value)).hexdigest()


def _require_id(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise train.TrainManifestError(f"{where}: expected sha256:<64 lowercase hex>")
    return value


def _require_positive_int(value: Any, *, where: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise train.TrainManifestError(f"{where}: expected positive integer")
    return value


def _require_bool(value: Any, *, where: str) -> bool:
    if not isinstance(value, bool):
        raise train.TrainManifestError(f"{where}: expected boolean")
    return value


def _require_provider_ref(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise train.TrainManifestError("attempt observation.provider_ref: expected object")
    subject_mod._require_exact_keys(
        value,
        {"provider", "attempt_ref"},
        set(),
        where="attempt observation.provider_ref",
    )
    return {
        "provider": subject_mod._require_string(
            value["provider"], where="attempt observation.provider_ref.provider"
        ),
        "attempt_ref": subject_mod._require_string(
            value["attempt_ref"], where="attempt observation.provider_ref.attempt_ref"
        ),
    }


def normalize_registration(
    value: Any, *, verify_declared_id: bool = True, require_id: bool = False
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise train.TrainManifestError("attempt registration: expected object")
    subject_mod._require_exact_keys(
        value,
        {
            "schema",
            "admission_subject_id",
            "qualification_subject_id",
            "qualification_profile_id",
            "recipe_id",
            "input_closure_id",
            "qualification_environment_id",
            "attempt_sequence",
            "non_claims",
        },
        {"attempt_registration_id"},
        where="attempt registration",
    )
    if value["schema"] != REGISTRATION_SCHEMA:
        raise train.TrainManifestError(
            f"attempt registration.schema: expected {REGISTRATION_SCHEMA!r}"
        )
    if require_id and "attempt_registration_id" not in value:
        raise train.TrainManifestError("attempt registration.attempt_registration_id: required")

    normalized: dict[str, Any] = {
        "schema": REGISTRATION_SCHEMA,
        "admission_subject_id": _require_id(
            value["admission_subject_id"], where="attempt registration.admission_subject_id"
        ),
        "qualification_subject_id": _require_id(
            value["qualification_subject_id"],
            where="attempt registration.qualification_subject_id",
        ),
        "qualification_profile_id": profile_mod._require_profile_id(
            value["qualification_profile_id"],
            where="attempt registration.qualification_profile_id",
        ),
        "recipe_id": profile_mod._require_recipe_id(
            value["recipe_id"], where="attempt registration.recipe_id"
        ),
        "input_closure_id": _require_id(
            value["input_closure_id"], where="attempt registration.input_closure_id"
        ),
        "qualification_environment_id": _require_id(
            value["qualification_environment_id"],
            where="attempt registration.qualification_environment_id",
        ),
        "attempt_sequence": _require_positive_int(
            value["attempt_sequence"], where="attempt registration.attempt_sequence"
        ),
        "non_claims": train._require_sorted_unique_strings(
            value["non_claims"], where="attempt registration.non_claims"
        ),
    }
    if not normalized["non_claims"]:
        raise train.TrainManifestError("attempt registration.non_claims: must not be empty")

    payload = dict(normalized)
    attempt_registration_id = _id(REGISTRATION_DOMAIN, payload)
    normalized["attempt_registration_id"] = attempt_registration_id
    if verify_declared_id and "attempt_registration_id" in value:
        declared = _require_id(
            value["attempt_registration_id"],
            where="attempt registration.attempt_registration_id",
        )
        if declared != attempt_registration_id:
            raise train.TrainManifestError(
                "attempt registration.attempt_registration_id: registration semantics changed"
            )
    return normalized


def validate_registration_against_admission(
    registration: Any, admission: Any, profile: Any
) -> dict[str, Any]:
    reg = normalize_registration(registration, require_id=True)
    adm = admission_mod.normalize_request(admission, require_ids=True)
    prof = profile_mod.normalize_profile(profile, require_id=True)
    admission_mod.validate_profile_record(adm, prof)

    if reg["admission_subject_id"] != adm["admission_subject_id"]:
        raise train.TrainManifestError(
            "attempt registration: admission_subject_id does not match admission"
        )
    if reg["qualification_subject_id"] != adm["subject"]["subject_id"]:
        raise train.TrainManifestError(
            "attempt registration: qualification_subject_id does not match admission subject"
        )
    if reg["qualification_profile_id"] != adm["qualification_profile"]["profile_id"]:
        raise train.TrainManifestError(
            "attempt registration: qualification_profile_id does not match admission"
        )
    if reg["recipe_id"] not in prof["required_recipe_ids"]:
        raise train.TrainManifestError(
            "attempt registration: recipe is not required by the exact qualification profile"
        )
    return reg


def normalize_observation(
    value: Any, *, verify_declared_id: bool = True, require_id: bool = False
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise train.TrainManifestError("attempt observation: expected object")
    subject_mod._require_exact_keys(
        value,
        {
            "schema",
            "attempt_registration_id",
            "execution_started",
            "terminal_disposition",
            "provider_ref",
            "evidence_refs",
            "non_claims",
        },
        {"attempt_observation_id"},
        where="attempt observation",
    )
    if value["schema"] != OBSERVATION_SCHEMA:
        raise train.TrainManifestError(
            f"attempt observation.schema: expected {OBSERVATION_SCHEMA!r}"
        )
    if require_id and "attempt_observation_id" not in value:
        raise train.TrainManifestError("attempt observation.attempt_observation_id: required")

    execution_started = _require_bool(
        value["execution_started"], where="attempt observation.execution_started"
    )
    terminal = subject_mod._require_string(
        value["terminal_disposition"], where="attempt observation.terminal_disposition"
    )
    if terminal not in ALL_TERMINALS:
        raise train.TrainManifestError(
            "attempt observation.terminal_disposition: unsupported disposition"
        )
    if execution_started and terminal in PRESTART_TERMINALS:
        raise train.TrainManifestError(
            "attempt observation: pre-start terminal is incompatible with execution_started=true"
        )
    if not execution_started and terminal in POSTSTART_TERMINALS:
        raise train.TrainManifestError(
            "attempt observation: post-start terminal is incompatible with execution_started=false"
        )

    evidence_refs = train._require_sorted_unique_strings(
        value["evidence_refs"], where="attempt observation.evidence_refs"
    )
    if terminal == "Passed" and not evidence_refs:
        raise train.TrainManifestError(
            "attempt observation: Passed requires at least one execution evidence reference"
        )
    if terminal != "Passed" and not evidence_refs:
        raise train.TrainManifestError(
            "attempt observation: non-Passed terminal requires failure/infrastructure evidence"
        )

    normalized: dict[str, Any] = {
        "schema": OBSERVATION_SCHEMA,
        "attempt_registration_id": _require_id(
            value["attempt_registration_id"],
            where="attempt observation.attempt_registration_id",
        ),
        "execution_started": execution_started,
        "terminal_disposition": terminal,
        "provider_ref": _require_provider_ref(value["provider_ref"]),
        "evidence_refs": evidence_refs,
        "non_claims": train._require_sorted_unique_strings(
            value["non_claims"], where="attempt observation.non_claims"
        ),
    }
    if not normalized["non_claims"]:
        raise train.TrainManifestError("attempt observation.non_claims: must not be empty")

    payload = dict(normalized)
    attempt_observation_id = _id(OBSERVATION_DOMAIN, payload)
    normalized["attempt_observation_id"] = attempt_observation_id
    if verify_declared_id and "attempt_observation_id" in value:
        declared = _require_id(
            value["attempt_observation_id"],
            where="attempt observation.attempt_observation_id",
        )
        if declared != attempt_observation_id:
            raise train.TrainManifestError(
                "attempt observation.attempt_observation_id: observation semantics changed"
            )
    return normalized


def validate_observation_against_registration(
    observation: Any, registration: Any
) -> dict[str, Any]:
    obs = normalize_observation(observation, require_id=True)
    reg = normalize_registration(registration, require_id=True)
    if obs["attempt_registration_id"] != reg["attempt_registration_id"]:
        raise train.TrainManifestError(
            "attempt observation: does not bind the exact attempt registration"
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
        raise train.TrainManifestError("attempt history: expected at least one registration")

    regs = [validate_registration_against_admission(item, adm, prof) for item in registrations]
    reg_by_id = {item["attempt_registration_id"]: item for item in regs}
    if len(reg_by_id) != len(regs):
        raise train.TrainManifestError("attempt history: duplicate attempt registration id")

    sequence_keys = [(item["recipe_id"], item["attempt_sequence"]) for item in regs]
    if len(sequence_keys) != len(set(sequence_keys)):
        raise train.TrainManifestError(
            "attempt history: duplicate attempt_sequence for the same recipe"
        )

    obs = [normalize_observation(item, require_id=True) for item in observations]
    obs_by_registration: dict[str, dict[str, Any]] = {}
    for item in obs:
        registration_id = item["attempt_registration_id"]
        if registration_id not in reg_by_id:
            raise train.TrainManifestError(
                "attempt history: orphan observation has no registered attempt"
            )
        if registration_id in obs_by_registration:
            raise train.TrainManifestError(
                "attempt history: a registration may have at most one terminal observation"
            )
        obs_by_registration[registration_id] = item

    represented_recipes = sorted({item["recipe_id"] for item in regs})
    required_recipes = prof["required_recipe_ids"]
    extra = sorted(set(represented_recipes) - set(required_recipes))
    if extra:
        raise train.TrainManifestError(
            "attempt history: contains recipe registrations outside the exact profile"
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
            "does not establish externally anchored chronology between attempt_sequence values",
            "does not establish qualification merely because every recipe has an attempt",
            "does not prove required cross-cutting rules",
            "does not erase failed or infrastructure attempts when a later retry passes",
        ],
    }
    history_id = _id(HISTORY_DOMAIN, normalized)
    normalized["attempt_history_id"] = history_id
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
            raise train.TrainManifestError("successful attempt lookup: orphan observation")
        out.setdefault(reg["recipe_id"], []).append(item["attempt_observation_id"])
    for recipe_id in out:
        out[recipe_id].sort()
    return out
