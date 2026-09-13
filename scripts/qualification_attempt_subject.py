#!/usr/bin/env python3
"""Derive the exact theorem identity for one qualification recipe attempt family.

The attempt theorem is narrower than profile-level admission and broader than one retry:

QualificationSubjectId
+ QualificationProfileId
+ QualificationRecipeId
+ InputClosureId
+ QualificationEnvironmentId
-> QualificationAttemptSubjectId

Provider identity, retry sequence, rationale, and timestamps are deliberately excluded.
Changing source, profile, recipe, closure, or environment creates a different theorem subject.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import integration_train_manifest as train
import qualification_attempt_v3 as attempts

DOMAIN = b"symthaea.qualification-attempt-subject.v1\0"


def _canonical(value: dict[str, str]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def work_payload(registration: Any) -> dict[str, str]:
    reg = attempts.normalize_registration(registration, require_id=True)
    return {
        "qualification_subject_id": reg["qualification_subject_id"],
        "qualification_profile_id": reg["qualification_profile_id"],
        "recipe_id": reg["recipe_id"],
        "input_closure_id": reg["input_closure_id"],
        "qualification_environment_id": reg["qualification_environment_id"],
    }


def compute_attempt_subject_id(registration: Any) -> str:
    payload = work_payload(registration)
    return "sha256:" + hashlib.sha256(DOMAIN + _canonical(payload)).hexdigest()


def validate_same_attempt_subject(left: Any, right: Any) -> str:
    left_id = compute_attempt_subject_id(left)
    right_id = compute_attempt_subject_id(right)
    if left_id != right_id:
        raise train.TrainManifestError(
            "qualification attempt theorem changed: source/profile/recipe/input-closure/environment mismatch"
        )
    return left_id
