#!/usr/bin/env python3
"""Project a historical PassSelectionV2 onto normative V2 subject/profile/recipe semantics."""

from __future__ import annotations
import re
from typing import Any
import integration_train_manifest as train
import qualification_framing_v1 as framing
import qualification_semantic_ids_v2 as semantic_v2
import qualification_subject as subject_mod
import qualification_pass_selection_v2 as selection_v2

SCHEMA = "symthaea.qualification-pass-selection-semantic-projection.v1"
SOURCE_SCHEMA = "symthaea.qualification-pass-selection.v2"
ID_DOMAIN = "symthaea.qualification-pass-selection-semantic-projection-id.v1"
_SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
NON_CLAIMS = sorted([
    "does not reinterpret or replace the historical PassSelectionV2 identity",
    "does not migrate AttemptHistoryId, InputClosureId, QualificationEnvironmentId, or attempt-subject identities",
    "does not establish that provider-declared Passed observations are trustworthy",
    "does not establish evidence correctness, sufficiency, provider authenticity, current admission, or merge authority",
])

class SemanticProjectionError(train.TrainManifestError):
    pass

def _require_sha256_id(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or _SHA256_ID.fullmatch(value) is None:
        raise SemanticProjectionError(f"{where}: expected sha256:<64 lowercase hex>")
    return value

def _require_exact_keys(value: Any, required: set[str], *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SemanticProjectionError(f"{where}: expected object")
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required)
    if missing:
        raise SemanticProjectionError(f"{where}: missing fields: {missing}")
    if unknown:
        raise SemanticProjectionError(f"{where}: unknown fields: {unknown}")
    return value

def _frame_mapping(item: dict[str, str]) -> bytes:
    return framing.frame_record(
        "symthaea.qualification-pass-selection-recipe-mapping.v1",
        [
            ("legacy_recipe_ref", framing.encode_text(item["legacy_recipe_ref"])),
            ("recipe_id", framing.encode_text(item["recipe_id"])),
            ("attempt_subject_id", framing.encode_text(item["attempt_subject_id"])),
        ],
    )

def _projection_fields(value: dict[str, Any]) -> list[tuple[str, bytes]]:
    return [
        ("source_pass_selection_id", framing.encode_text(value["source_pass_selection_id"])),
        ("qualification_subject_id_v2", framing.encode_text(value["qualification_subject_id_v2"])),
        ("qualification_profile_id_v2", framing.encode_text(value["qualification_profile_id_v2"])),
        ("input_closure_id_source", framing.encode_text(value["input_closure_id_source"])),
        ("qualification_environment_id_source", framing.encode_text(value["qualification_environment_id_source"])),
        ("recipe_mappings", framing.encode_set(_frame_mapping(item) for item in value["recipe_mappings"])),
        ("non_claims", framing.encode_set(framing.encode_text(x) for x in value["non_claims"])),
    ]

def project_selection(*, selection: Any, subject: Any, profile: Any, recipes: list[Any]) -> dict[str, Any]:
    selection = _require_exact_keys(
        selection,
        {
            "schema", "admission_subject_id", "qualification_subject_id",
            "qualification_profile_id", "input_closure_id",
            "qualification_environment_id", "attempt_history_id",
            "selected_recipe_attempts", "cross_cutting_evidence",
            "non_claims", "pass_selection_id",
        },
        where="pass selection V2",
    )
    if selection["schema"] != SOURCE_SCHEMA:
        raise SemanticProjectionError(f"pass selection V2.schema: expected {SOURCE_SCHEMA!r}")
    declared_selection_id = _require_sha256_id(
        selection["pass_selection_id"], where="pass selection V2.pass_selection_id"
    )
    source_payload = dict(selection)
    source_payload.pop("pass_selection_id")
    if selection_v2._selection_id(source_payload) != declared_selection_id:
        raise SemanticProjectionError(
            "pass selection V2.pass_selection_id does not match exact historical selection bytes"
        )

    normalized_subject = subject_mod.normalize_subject(subject, verify_declared_id=True)
    if selection["qualification_subject_id"] != normalized_subject["subject_id"]:
        raise SemanticProjectionError(
            "pass selection V2 qualification_subject_id does not match exact subject preimage"
        )

    normalized_profile, resolved_recipes = semantic_v2.resolve_profile_recipes(profile, recipes)
    if selection["qualification_profile_id"] != normalized_profile["profile_id"]:
        raise SemanticProjectionError(
            "pass selection V2 qualification_profile_id does not match exact V1 profile preimage"
        )

    expected_legacy_refs = list(normalized_profile["required_recipe_ids"])
    by_legacy = {recipe["legacy_recipe_ref"]: recipe for recipe in resolved_recipes}
    selected = selection["selected_recipe_attempts"]
    if not isinstance(selected, list) or not selected:
        raise SemanticProjectionError("pass selection V2.selected_recipe_attempts: expected non-empty array")

    observed_legacy_refs = []
    mappings = []
    for index, item in enumerate(selected):
        item = _require_exact_keys(
            item,
            {"recipe_id", "attempt_subject_id", "attempt_registration_id",
             "attempt_observation_id", "evidence_content_ids"},
            where=f"pass selection V2.selected_recipe_attempts[{index}]",
        )
        legacy_ref = item["recipe_id"]
        if not isinstance(legacy_ref, str):
            raise SemanticProjectionError(
                f"pass selection V2.selected_recipe_attempts[{index}].recipe_id: expected string"
            )
        recipe = by_legacy.get(legacy_ref)
        if recipe is None:
            raise SemanticProjectionError(
                f"pass selection V2 selected legacy recipe is not resolved by exact recipe preimages: {legacy_ref}"
            )
        observed_legacy_refs.append(legacy_ref)
        mappings.append({
            "legacy_recipe_ref": legacy_ref,
            "recipe_id": _require_sha256_id(recipe["recipe_id"], where=f"resolved recipe {legacy_ref}.recipe_id"),
            "attempt_subject_id": _require_sha256_id(
                item["attempt_subject_id"],
                where=f"pass selection V2.selected_recipe_attempts[{index}].attempt_subject_id",
            ),
        })

    if observed_legacy_refs != expected_legacy_refs:
        raise SemanticProjectionError(
            "pass selection V2 selected recipe order/set does not equal exact V1 profile required_recipe_ids"
        )

    semantic_ids = [item["recipe_id"] for item in mappings]
    if len(semantic_ids) != len(set(semantic_ids)):
        raise SemanticProjectionError(
            "distinct legacy recipe refs resolve to duplicate semantic QualificationRecipeId"
        )

    normalized = {
        "schema": SCHEMA,
        "source_pass_selection_id": declared_selection_id,
        "qualification_subject_id_v2": semantic_v2.compute_subject_id_v2(normalized_subject),
        "qualification_profile_id_v2": semantic_v2.compute_profile_id_v2(
            normalized_profile, resolved_recipes
        ),
        "input_closure_id_source": _require_sha256_id(
            selection["input_closure_id"], where="pass selection V2.input_closure_id"
        ),
        "qualification_environment_id_source": _require_sha256_id(
            selection["qualification_environment_id"],
            where="pass selection V2.qualification_environment_id",
        ),
        "recipe_mappings": mappings,
        "non_claims": list(NON_CLAIMS),
    }
    normalized["projection_id"] = framing.semantic_sha256_id(
        ID_DOMAIN, _projection_fields(normalized)
    )
    return normalized
