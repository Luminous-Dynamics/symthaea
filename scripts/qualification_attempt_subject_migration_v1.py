#!/usr/bin/env python3
"""Migrate one selected historical attempt onto normative qualification semantic identities.

This is deliberately fail-closed. Subject/profile/recipe identities may migrate through exact
preimages, but input-closure and environment coordinates are accepted only when the immutable
historical attempt registration already committed the exact independently reconstructed normative
IDs. A mismatching old coordinate is not "close enough" and cannot be relabeled.

Positive output is an attempt-subject migration witness, not trusted PASS or occurrence authority.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import integration_train_manifest as train
import qualification_attempt_subject as attempt_subject_v1
import qualification_attempt_v4 as attempts
import qualification_environment_realization_v1 as realization_mod
import qualification_framing_v1 as framing
import qualification_input_closure_v1 as closure_mod
import qualification_pass_selection_replay_v1 as replay_mod
import qualification_pass_selection_semantic_projection_v2 as projection_v2

SCHEMA = "symthaea.qualification-attempt-subject-migration.v1"
ATTEMPT_SUBJECT_DOMAIN = "symthaea.qualification-attempt-subject-id.v2"
MIGRATION_ID_DOMAIN = "symthaea.qualification-attempt-subject-migration-id.v1"
POSITIVE_STATE = "AttemptSubjectMigrationVerifiedOnly"

NON_CLAIMS = sorted([
    "does not establish that provider-declared Passed observations are trustworthy",
    "does not establish evidence correctness, scientific validity, or externally anchored chronology",
    "does not authenticate the execution provider",
    "does not establish detached attestation, current admission, merge, deployment, or action authority",
])


def _attempt_subject_fields(value: dict[str, str]) -> list[tuple[str, bytes]]:
    return [
        ("qualification_subject_id_v2", framing.encode_text(value["qualification_subject_id_v2"])),
        ("qualification_profile_id_v2", framing.encode_text(value["qualification_profile_id_v2"])),
        ("qualification_recipe_id", framing.encode_text(value["qualification_recipe_id"])),
        ("qualification_input_closure_id_v1", framing.encode_text(value["qualification_input_closure_id_v1"])),
        ("qualification_environment_id_v1", framing.encode_text(value["qualification_environment_id_v1"])),
    ]


def _migration_fields(value: dict[str, Any]) -> list[tuple[str, bytes]]:
    return [
        ("state", framing.encode_enum(value["state"])),
        ("source_attempt_registration_id", framing.encode_text(value["source_attempt_registration_id"])),
        ("source_attempt_subject_id_v1", framing.encode_text(value["source_attempt_subject_id_v1"])),
        ("source_pass_selection_id", framing.encode_text(value["source_pass_selection_id"])),
        ("pass_selection_replay_id", framing.encode_text(value["pass_selection_replay_id"])),
        ("semantic_projection_id", framing.encode_text(value["semantic_projection_id"])),
        ("qualification_environment_id_v1", framing.encode_text(value["qualification_environment_id_v1"])),
        ("qualification_attempt_subject_id_v2", framing.encode_text(value["qualification_attempt_subject_id_v2"])),
        ("non_claims", framing.encode_set(framing.encode_text(x) for x in value["non_claims"])),
    ]


def migrate_selected_attempt(
    *, legacy_recipe_ref: str, selection: Any, admission: Any, profile: Any,
    registrations: list[Any], observations: list[Any], subject: Any, recipes: list[Any],
    environment_selection: Any, environment_resolution: Any, repo: Path,
    tool_requirements: Any, capture: Any, capture_root: Path,
) -> dict[str, Any]:
    replay = replay_mod.verify_replay(
        selection=selection, admission=admission, profile=profile,
        registrations=registrations, observations=observations,
    )
    projection = projection_v2.project_selection(
        selection=selection, subject=subject, profile=profile, recipes=recipes,
    )
    if replay["pass_selection_id"] != projection["source_pass_selection_id"]:
        raise train.TrainManifestError("attempt subject migration: replay/projection source selection mismatch")

    normative_closure = closure_mod.build_from_subject(subject)
    if normative_closure["input_closure_id"] != projection["qualification_input_closure_id_v1"]:
        raise train.TrainManifestError("attempt subject migration: semantic projection/input-closure reconstruction mismatch")

    realization = realization_mod.verify_realization_capture(
        resolution=environment_resolution, selection=environment_selection, subject=subject,
        closure=normative_closure, repo=repo, tool_requirements=tool_requirements,
        profile=profile, recipes=recipes, capture=capture, capture_root=capture_root,
    )

    selected_items = [item for item in selection["selected_recipe_attempts"] if item.get("recipe_id") == legacy_recipe_ref]
    if len(selected_items) != 1:
        raise train.TrainManifestError("attempt subject migration: selected legacy recipe must occur exactly once")
    selected = selected_items[0]
    registration_id = selected["attempt_registration_id"]

    normalized_regs = [attempts.normalize_registration(item, require_id=True) for item in registrations]
    matching_regs = [item for item in normalized_regs if item["attempt_registration_id"] == registration_id]
    if len(matching_regs) != 1:
        raise train.TrainManifestError("attempt subject migration: selected registration must resolve exactly once")
    registration = matching_regs[0]

    if registration["recipe_id"] != legacy_recipe_ref:
        raise train.TrainManifestError("attempt subject migration: selected registration recipe mismatch")
    if registration["input_closure_id"] != normative_closure["input_closure_id"]:
        raise train.TrainManifestError("attempt subject migration: historical registration did not commit the normative input closure")
    if selection["input_closure_id"] != normative_closure["input_closure_id"]:
        raise train.TrainManifestError("attempt subject migration: historical selection did not commit the normative input closure")

    normative_environment_id = realization["qualification_environment_id"]
    if registration["qualification_environment_id"] != normative_environment_id:
        raise train.TrainManifestError("attempt subject migration: historical registration did not commit the verified realized environment")
    if selection["qualification_environment_id"] != normative_environment_id:
        raise train.TrainManifestError("attempt subject migration: historical selection did not commit the verified realized environment")

    mappings = [item for item in projection["recipe_mappings"] if item["legacy_recipe_ref"] == legacy_recipe_ref]
    if len(mappings) != 1:
        raise train.TrainManifestError("attempt subject migration: semantic recipe mapping must resolve exactly once")
    semantic_recipe_id = mappings[0]["recipe_id"]

    semantic_fields = {
        "qualification_subject_id_v2": projection["qualification_subject_id_v2"],
        "qualification_profile_id_v2": projection["qualification_profile_id_v2"],
        "qualification_recipe_id": semantic_recipe_id,
        "qualification_input_closure_id_v1": normative_closure["input_closure_id"],
        "qualification_environment_id_v1": normative_environment_id,
    }
    attempt_subject_id_v2 = framing.semantic_sha256_id(ATTEMPT_SUBJECT_DOMAIN, _attempt_subject_fields(semantic_fields))

    witness = {
        "schema": SCHEMA,
        "state": POSITIVE_STATE,
        "source_attempt_registration_id": registration["attempt_registration_id"],
        "source_attempt_subject_id_v1": attempt_subject_v1.compute_attempt_subject_id(registration),
        "source_pass_selection_id": projection["source_pass_selection_id"],
        "pass_selection_replay_id": replay["replay_id"],
        "semantic_projection_id": projection["projection_id"],
        "qualification_environment_id_v1": normative_environment_id,
        "qualification_attempt_subject_id_v2": attempt_subject_id_v2,
        "semantic_fields": semantic_fields,
        "non_claims": list(NON_CLAIMS),
    }
    witness["migration_id"] = framing.semantic_sha256_id(MIGRATION_ID_DOMAIN, _migration_fields(witness))
    return witness
