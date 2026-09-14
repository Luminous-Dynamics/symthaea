#!/usr/bin/env python3
"""Build a byte-closed support set for a qualification PASS selection.

This is a deliberately pre-receipt object. It proves only that every evidence-content descriptor
selected by PassSelectionV2 has exactly one supplied byte payload whose content identity verifies,
with no missing or extra payloads. It does NOT establish that a selected attempt really passed,
that a cross-cutting semantic rule is satisfied, that the bytes came from the claimed provider/run,
or that any producer/witness is trusted.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import integration_train_manifest as train
import qualification_evidence_binding as binding_mod
import qualification_pass_selection_v2 as selection_mod

SCHEMA = "symthaea.qualification-support-closure.v1"
DOMAIN = b"symthaea.qualification-support-closure.v1\0"


def _canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _closure_id(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "support_closure_id"}
    return "sha256:" + hashlib.sha256(DOMAIN + _canon(payload)).hexdigest()


def _selected_content_ids(selection: dict[str, Any]) -> list[str]:
    selected: set[str] = set()
    for attempt in selection["selected_recipe_attempts"]:
        selected.update(attempt["evidence_content_ids"])
    for cross in selection["cross_cutting_evidence"]:
        selected.update(cross["evidence_content_ids"])
    return sorted(selected)


def build_support_closure(
    *,
    admission: Any,
    profile: Any,
    registrations: list[Any],
    observations: list[Any],
    selected_observation_ids_by_recipe: dict[str, str],
    cross_cutting_evidence: list[dict[str, Any]],
    evidence_bytes_by_content_id: dict[str, bytes],
) -> dict[str, Any]:
    selection = selection_mod.build_pass_selection(
        admission=admission,
        profile=profile,
        registrations=registrations,
        observations=observations,
        selected_observation_ids_by_recipe=selected_observation_ids_by_recipe,
        cross_cutting_evidence=cross_cutting_evidence,
    )
    selected_content_ids = _selected_content_ids(selection)

    if not isinstance(evidence_bytes_by_content_id, dict):
        raise train.TrainManifestError(
            "qualification support closure.evidence_bytes_by_content_id: expected object"
        )
    supplied_ids = sorted(evidence_bytes_by_content_id)
    missing = sorted(set(selected_content_ids) - set(supplied_ids))
    extra = sorted(set(supplied_ids) - set(selected_content_ids))
    if missing or extra:
        raise train.TrainManifestError(
            "qualification support closure: evidence byte set must exactly equal selected content ids "
            f"(missing={missing}, extra={extra})"
        )

    bindings: list[dict[str, Any]] = []
    for content_id in selected_content_ids:
        result = binding_mod.verify_bytes(content_id, evidence_bytes_by_content_id[content_id])
        bindings.append(
            {
                "content_id": result["content_id"],
                "binding_id": result["binding_id"],
                "byte_length": result["byte_length"],
                "verification_method": result["verification_method"],
            }
        )

    normalized = {
        "schema": SCHEMA,
        "pass_selection_id": selection["pass_selection_id"],
        "qualification_subject_id": selection["qualification_subject_id"],
        "qualification_profile_id": selection["qualification_profile_id"],
        "attempt_history_id": selection["attempt_history_id"],
        "selected_content_ids": selected_content_ids,
        "byte_bindings": bindings,
        "cross_cutting_rules": [
            item["rule"] for item in selection["cross_cutting_evidence"]
        ],
        "non_claims": [
            "byte closure does not establish that a selected attempt terminal disposition is trustworthy",
            "byte closure does not establish that any cross-cutting semantic rule is satisfied",
            "byte closure does not establish trustworthy acquisition or provider/run provenance",
            "byte closure does not establish producer or witness authenticity",
            "byte closure does not establish evidence correctness or sufficiency",
            "byte closure does not establish qualification receipt authority, current admission, or merge authority",
        ],
    }
    normalized["support_closure_id"] = _closure_id(normalized)
    return normalized
