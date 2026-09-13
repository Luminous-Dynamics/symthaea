#!/usr/bin/env python3
"""Framed V2 qualification profile identity.

V2 preserves V1 profile semantics while replacing serializer-dependent JSON hashing with an
explicit language-independent framed preimage and typed ID prefix.
"""

from __future__ import annotations

import re
from typing import Any

import integration_train_manifest as train
import qualification_framing as framing
import qualification_profile as v1

SCHEMA = "symthaea.qualification-profile.v2"
DOMAIN = "qualification-profile-v2"
ID_PREFIX = "qprofile-v2-sha256"
_ID_RE = re.compile(r"^qprofile-v2-sha256:[0-9a-f]{64}$")


def _preimage(normalized: dict[str, Any]) -> bytes:
    return framing.record(
        DOMAIN,
        [
            ("schema", framing.text(normalized["schema"])),
            ("profile_name", framing.text(normalized["profile_name"])),
            ("required_recipe_ids", framing.text_list(normalized["required_recipe_ids"])),
            ("owned_surface_rules", framing.text_list(normalized["owned_surface_rules"])),
            (
                "required_cross_cutting_rules",
                framing.text_list(normalized["required_cross_cutting_rules"]),
            ),
            ("fallback_policy", framing.text(normalized["fallback_policy"])),
            ("non_claims", framing.text_list(normalized["non_claims"])),
        ],
    )


def normalize_profile(value: Any, *, verify_declared_id: bool = True, require_id: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise train.TrainManifestError("qualification profile v2: expected object")
    v1._require_exact_keys(
        value,
        {
            "schema",
            "profile_name",
            "required_recipe_ids",
            "owned_surface_rules",
            "required_cross_cutting_rules",
            "fallback_policy",
            "non_claims",
        },
        {"profile_id"},
        where="qualification profile v2",
    )
    if value["schema"] != SCHEMA:
        raise train.TrainManifestError(f"qualification profile v2.schema: expected {SCHEMA!r}")
    if require_id and "profile_id" not in value:
        raise train.TrainManifestError("qualification profile v2.profile_id: required but absent")

    surrogate = dict(value)
    surrogate["schema"] = v1.SCHEMA
    surrogate.pop("profile_id", None)
    checked = v1.normalize_profile(surrogate, verify_declared_id=False)
    checked.pop("profile_id", None)
    checked["schema"] = SCHEMA

    profile_id = framing.sha256_hex_id(ID_PREFIX, _preimage(checked))
    checked["profile_id"] = profile_id
    if verify_declared_id and "profile_id" in value:
        declared = value["profile_id"]
        if not isinstance(declared, str) or _ID_RE.fullmatch(declared) is None:
            raise train.TrainManifestError(
                "qualification profile v2.profile_id: expected qprofile-v2-sha256:<64 lowercase hex>"
            )
        if declared != profile_id:
            raise train.TrainManifestError(
                f"qualification profile v2.profile_id: expected {profile_id}, got {declared}"
            )
    return checked


def compute_profile_id(value: Any) -> str:
    return normalize_profile(value, verify_declared_id=False)["profile_id"]


def framed_preimage(value: Any) -> bytes:
    normalized = normalize_profile(value, verify_declared_id=False)
    normalized.pop("profile_id", None)
    return _preimage(normalized)
