#!/usr/bin/env python3
"""Gate a byte-closed qualification selection for first-generation receipt witnessing.

V1 is intentionally conservative: generic string-named cross-cutting rules are not eligible for
portable positive-receipt construction because matching evidence bytes do not establish the rule's
semantics. Claim-bearing obligations must be modeled as executable content-addressed recipes (or a
future typed verified-obligation contract) before this gate admits the candidate for independent
witness evaluation.

Passing this gate is NOT qualification PASS. It means only that the candidate has an exact subject,
profile, attempt history, selected recipe support, byte closure, and no unsupported generic
cross-cutting obligations. Independent witness authority remains mandatory.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import integration_train_manifest as train
import qualification_profile as profile_mod
import qualification_support_closure as closure_mod

SCHEMA = "symthaea.qualification-receipt-candidate-gate.v1"
DOMAIN = b"symthaea.qualification-receipt-candidate-gate.v1\0"
DISPOSITION = "WitnessRequired"


def _canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _candidate_id(value: dict[str, Any]) -> str:
    payload = {key: item for key, item in value.items() if key != "receipt_candidate_id"}
    return "sha256:" + hashlib.sha256(DOMAIN + _canon(payload)).hexdigest()


def build_receipt_candidate_gate(
    *,
    admission: Any,
    profile: Any,
    registrations: list[Any],
    observations: list[Any],
    selected_observation_ids_by_recipe: dict[str, str],
    cross_cutting_evidence: list[dict[str, Any]],
    evidence_bytes_by_content_id: dict[str, bytes],
) -> dict[str, Any]:
    prof = profile_mod.normalize_profile(profile, require_id=True)
    if prof["required_cross_cutting_rules"]:
        raise train.TrainManifestError(
            "qualification receipt candidate V1: generic required_cross_cutting_rules are not "
            "receipt-eligible; encode claim-bearing obligations as required_recipe_ids or wait "
            "for a typed verified-obligation contract"
        )
    if cross_cutting_evidence:
        raise train.TrainManifestError(
            "qualification receipt candidate V1: cross_cutting_evidence must be empty when the "
            "profile has no generic cross-cutting rules"
        )

    closure = closure_mod.build_support_closure(
        admission=admission,
        profile=prof,
        registrations=registrations,
        observations=observations,
        selected_observation_ids_by_recipe=selected_observation_ids_by_recipe,
        cross_cutting_evidence=[],
        evidence_bytes_by_content_id=evidence_bytes_by_content_id,
    )

    normalized = {
        "schema": SCHEMA,
        "qualification_subject_id": closure["qualification_subject_id"],
        "qualification_profile_id": closure["qualification_profile_id"],
        "attempt_history_id": closure["attempt_history_id"],
        "pass_selection_id": closure["pass_selection_id"],
        "support_closure_id": closure["support_closure_id"],
        "required_recipe_ids": list(prof["required_recipe_ids"]),
        "disposition": DISPOSITION,
        "non_claims": [
            "WitnessRequired is not qualification PASS or receipt authority",
            "candidate eligibility does not establish that provider-declared Passed dispositions are trustworthy",
            "candidate eligibility does not establish provider/run provenance or producer authenticity",
            "candidate eligibility does not establish evidence correctness or scientific validity",
            "candidate eligibility does not establish current admission or merge authority",
            "generic cross-cutting rule semantics are deliberately unsupported in V1",
        ],
    }
    normalized["receipt_candidate_id"] = _candidate_id(normalized)
    return normalized
