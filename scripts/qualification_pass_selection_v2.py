#!/usr/bin/env python3
"""PASS-selection V2 requiring content-addressed evidence descriptors for all selected support.

V2 composes attempt-observation V4. Provider locators remain provenance only; recipe and
cross-cutting support must name exact content with canonical digest/blob descriptors. Descriptor
syntax alone does NOT establish that referenced bytes exist or match the digest, producer
authenticity, evidence correctness, sufficiency, current admission, or merge authority.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import integration_train_manifest as train
import qualification_admission_v3 as admission_mod
import qualification_attempt_subject as attempt_subject_mod
import qualification_attempt_v4 as attempts
import qualification_evidence_id as evidence_id_mod
import qualification_profile as profile_mod
import qualification_subject as subject_mod

SCHEMA = "symthaea.qualification-pass-selection.v2"
DOMAIN = b"symthaea.qualification-pass-selection.v2\0"


def _canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _selection_id(value: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(DOMAIN + _canon(value)).hexdigest()


def _normalize_cross_cutting_evidence(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise train.TrainManifestError("pass selection V2.cross_cutting_evidence: expected array")
    out: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        where = f"pass selection V2.cross_cutting_evidence[{index}]"
        if not isinstance(item, dict):
            raise train.TrainManifestError(f"{where}: expected object")
        subject_mod._require_exact_keys(
            item, {"rule", "evidence_content_ids"}, set(), where=where
        )
        content_ids = evidence_id_mod.require_sorted_unique_evidence_content_ids(
            item["evidence_content_ids"], where=f"{where}.evidence_content_ids"
        )
        if not content_ids:
            raise train.TrainManifestError(f"{where}.evidence_content_ids: must not be empty")
        out.append(
            {
                "rule": subject_mod._require_string(item["rule"], where=f"{where}.rule"),
                "evidence_content_ids": content_ids,
            }
        )
    rules = [item["rule"] for item in out]
    if rules != sorted(rules) or len(rules) != len(set(rules)):
        raise train.TrainManifestError(
            "pass selection V2.cross_cutting_evidence: rules must be sorted unique"
        )
    return out


def build_pass_selection(
    *,
    admission: Any,
    profile: Any,
    registrations: list[Any],
    observations: list[Any],
    selected_observation_ids_by_recipe: dict[str, str],
    cross_cutting_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    adm = admission_mod.normalize_request(admission, require_ids=True)
    prof = profile_mod.normalize_profile(profile, require_id=True)
    admission_mod.validate_profile_record(adm, prof)
    history = attempts.build_history(
        admission=adm,
        profile=prof,
        registrations=registrations,
        observations=observations,
    )
    regs = [attempts.validate_registration_against_admission(item, adm, prof) for item in registrations]
    obs = [attempts.normalize_observation(item, require_id=True) for item in observations]
    reg_by_id = {item["attempt_registration_id"]: item for item in regs}
    obs_by_id = {item["attempt_observation_id"]: item for item in obs}

    if not isinstance(selected_observation_ids_by_recipe, dict):
        raise train.TrainManifestError(
            "pass selection V2.selected_observation_ids_by_recipe: expected object"
        )
    required_recipes = prof["required_recipe_ids"]
    if sorted(selected_observation_ids_by_recipe) != required_recipes:
        raise train.TrainManifestError(
            "pass selection V2: selected recipe set must exactly equal profile required_recipe_ids"
        )

    selected: list[dict[str, Any]] = []
    closure_ids: set[str] = set()
    environment_ids: set[str] = set()
    for recipe_id in required_recipes:
        observation_id = attempts._require_id(
            selected_observation_ids_by_recipe[recipe_id],
            where=f"pass selection V2.selected[{recipe_id}]",
        )
        observation = obs_by_id.get(observation_id)
        if observation is None:
            raise train.TrainManifestError(
                f"pass selection V2: selected observation for recipe {recipe_id} is absent from history"
            )
        if not observation["execution_started"] or observation["terminal_disposition"] != "Passed":
            raise train.TrainManifestError(
                f"pass selection V2: recipe {recipe_id} must select an executed Passed observation"
            )
        registration = reg_by_id.get(observation["attempt_registration_id"])
        if registration is None or registration["recipe_id"] != recipe_id:
            raise train.TrainManifestError(
                f"pass selection V2: observation selected under wrong or orphan recipe {recipe_id}"
            )
        closure_ids.add(registration["input_closure_id"])
        environment_ids.add(registration["qualification_environment_id"])
        selected.append(
            {
                "recipe_id": recipe_id,
                "attempt_subject_id": attempt_subject_mod.compute_attempt_subject_id(registration),
                "attempt_registration_id": registration["attempt_registration_id"],
                "attempt_observation_id": observation_id,
                "evidence_content_ids": list(observation["evidence_content_ids"]),
            }
        )

    if len(closure_ids) != 1:
        raise train.TrainManifestError(
            "pass selection V2: V1 attempt substrate requires one common input_closure_id across selected recipes"
        )
    if len(environment_ids) != 1:
        raise train.TrainManifestError(
            "pass selection V2: V1 attempt substrate requires one common qualification_environment_id across selected recipes"
        )

    cross = _normalize_cross_cutting_evidence(cross_cutting_evidence)
    if [item["rule"] for item in cross] != prof["required_cross_cutting_rules"]:
        raise train.TrainManifestError(
            "pass selection V2: cross-cutting evidence must exactly cover profile required_cross_cutting_rules"
        )

    normalized = {
        "schema": SCHEMA,
        "admission_subject_id": adm["admission_subject_id"],
        "qualification_subject_id": adm["subject"]["subject_id"],
        "qualification_profile_id": prof["profile_id"],
        "input_closure_id": next(iter(closure_ids)),
        "qualification_environment_id": next(iter(environment_ids)),
        "attempt_history_id": history["attempt_history_id"],
        "selected_recipe_attempts": selected,
        "cross_cutting_evidence": cross,
        "non_claims": [
            "content-id descriptor validity does not establish referenced bytes exist or match the digest",
            "content-addressed evidence identity does not establish authenticity, correctness, trust, or sufficiency",
            "does not define the permanent portable qualification receipt framing",
            "does not establish current admission or merge authority",
            "does not establish provider authenticity or detached attestation",
            "does not erase or rewrite non-selected failed attempts from attempt history",
        ],
    }
    normalized["pass_selection_id"] = _selection_id(normalized)
    return normalized
