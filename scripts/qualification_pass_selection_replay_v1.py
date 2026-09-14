#!/usr/bin/env python3
"""Replay PassSelectionV2 from exact source preimages before semantic migration.

A self-consistent PassSelectionId proves only that one selection object hashes to itself. It does
not prove that the object could have been produced by the normative PassSelectionV2 builder from
the claimed admission, profile, attempt history, and evidence coverage.

This verifier rebuilds the selection from those source preimages and requires exact object
equality. Positive output remains structural replay evidence, not trusted PASS or witness authority.
"""

from __future__ import annotations

from typing import Any

import integration_train_manifest as train
import qualification_framing_v1 as framing
import qualification_pass_selection_v2 as selection_v2

SCHEMA = "symthaea.qualification-pass-selection-replay.v1"
ID_DOMAIN = "symthaea.qualification-pass-selection-replay-id.v1"
POSITIVE_STATE = "PassSelectionReplayVerifiedOnly"

NON_CLAIMS = sorted([
    "does not establish provider authenticity or trustworthy Passed dispositions",
    "does not establish evidence correctness or scientific validity",
    "does not establish detached attestation, current admission, merge, deployment, or execution authority",
    "does not migrate historical qualification identities to normative semantic V2 identities",
])


def _fields(value: dict[str, Any]) -> list[tuple[str, bytes]]:
    return [
        ("state", framing.encode_enum(value["state"])),
        ("pass_selection_id", framing.encode_text(value["pass_selection_id"])),
        ("qualification_subject_id", framing.encode_text(value["qualification_subject_id"])),
        ("qualification_profile_id", framing.encode_text(value["qualification_profile_id"])),
        ("attempt_history_id", framing.encode_text(value["attempt_history_id"])),
        ("non_claims", framing.encode_set(framing.encode_text(x) for x in value["non_claims"])),
    ]


def verify_replay(
    *,
    selection: Any,
    admission: Any,
    profile: Any,
    registrations: list[Any],
    observations: list[Any],
) -> dict[str, Any]:
    if not isinstance(selection, dict):
        raise train.TrainManifestError("pass selection replay: selection must be object")

    try:
        selected_attempts = selection["selected_recipe_attempts"]
        cross_cutting = selection["cross_cutting_evidence"]
    except KeyError as error:
        raise train.TrainManifestError(
            f"pass selection replay: source selection missing {error.args[0]!r}"
        ) from error

    if not isinstance(selected_attempts, list) or not selected_attempts:
        raise train.TrainManifestError(
            "pass selection replay: selected_recipe_attempts must be non-empty array"
        )

    selected_ids: dict[str, str] = {}
    for index, item in enumerate(selected_attempts):
        if not isinstance(item, dict):
            raise train.TrainManifestError(
                f"pass selection replay: selected_recipe_attempts[{index}] must be object"
            )
        recipe_id = item.get("recipe_id")
        observation_id = item.get("attempt_observation_id")
        if not isinstance(recipe_id, str) or not isinstance(observation_id, str):
            raise train.TrainManifestError(
                f"pass selection replay: selected_recipe_attempts[{index}] lacks string recipe/observation IDs"
            )
        if recipe_id in selected_ids:
            raise train.TrainManifestError(
                f"pass selection replay: duplicate selected recipe identity {recipe_id}"
            )
        selected_ids[recipe_id] = observation_id

    rebuilt = selection_v2.build_pass_selection(
        admission=admission,
        profile=profile,
        registrations=registrations,
        observations=observations,
        selected_observation_ids_by_recipe=selected_ids,
        cross_cutting_evidence=cross_cutting,
    )

    if selection != rebuilt:
        raise train.TrainManifestError(
            "pass selection replay: supplied PassSelectionV2 does not exactly equal deterministic rebuild"
        )

    witness = {
        "schema": SCHEMA,
        "state": POSITIVE_STATE,
        "pass_selection_id": rebuilt["pass_selection_id"],
        "qualification_subject_id": rebuilt["qualification_subject_id"],
        "qualification_profile_id": rebuilt["qualification_profile_id"],
        "attempt_history_id": rebuilt["attempt_history_id"],
        "non_claims": list(NON_CLAIMS),
    }
    witness["replay_id"] = framing.semantic_sha256_id(ID_DOMAIN, _fields(witness))
    return witness
