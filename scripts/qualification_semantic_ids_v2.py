#!/usr/bin/env python3
"""Framed V2 semantic identities for generic qualification subject/profile roots.

This is a migration bridge, not an in-place reinterpretation of existing identities.
The existing V1 transport schemas and their JSON-derived IDs remain distinct historical /
prototype objects. V2 IDs are computed from normalized semantic fields using Qualification
Framing V1.

The profile V2 identity is deliberately preimage-complete with respect to recipes: callers must
supply the exact recipe manifests corresponding to every legacy recipe reference in the V1
profile. A hash-shaped recipe reference alone cannot establish QualificationProfileIdV2.

Only the two root identities migrate here:

    exact Git qualification subject -> QualificationSubjectIdV2
    qualification theorem + resolved recipes -> QualificationProfileIdV2

Admission, routing, attempt, history, pass-selection and receipt identities remain on their
existing schemas until their V2 migration is specified separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import integration_train_manifest as train
import qualification_framing_v1 as framing
import qualification_profile as profile_mod
import qualification_recipe_v1 as recipe_mod
import qualification_subject as subject_mod

SUBJECT_ID_DOMAIN = "symthaea.qualification-subject-id.v2"
PROFILE_ID_DOMAIN = "symthaea.qualification-profile-id.v2"


def _subject_fields(subject: Any) -> list[tuple[str, bytes]]:
    # Verify an existing declared V1 ID when present. Migration must not launder a malformed
    # V1 artifact into a valid V2 identity merely because its semantic fields parse.
    normalized = subject_mod.normalize_subject(subject, verify_declared_id=True)
    return [
        ("kind", framing.encode_enum(normalized["kind"])),
        ("repository", framing.encode_text(normalized["repository"])),
        ("object_format", framing.encode_enum(normalized["object_format"])),
        ("source_commit", framing.encode_text(normalized["source_commit"])),
        ("source_tree", framing.encode_text(normalized["source_tree"])),
    ]


def subject_frame_v2(subject: Any) -> bytes:
    """Return the normative framed bytes for QualificationSubjectIdV2."""

    return framing.frame_record(SUBJECT_ID_DOMAIN, _subject_fields(subject))


def compute_subject_id_v2(subject: Any) -> str:
    """Compute a framed V2 subject ID without redefining the V1 subject ID."""

    return framing.semantic_sha256_id(SUBJECT_ID_DOMAIN, _subject_fields(subject))


def _encoded_text_set(values: list[str]) -> bytes:
    return framing.encode_set(framing.encode_text(value) for value in values)


def resolve_profile_recipes(profile: Any, recipes: list[Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one V1 profile and the complete exact recipe-preimage set it references."""

    normalized_profile = profile_mod.normalize_profile(profile, verify_declared_id=True)
    if not isinstance(recipes, list) or not recipes:
        raise train.TrainManifestError(
            "qualification profile v2: exact recipe preimages are required"
        )

    normalized_recipes = [
        recipe_mod.normalize_recipe(recipe, require_id=True) for recipe in recipes
    ]
    by_legacy_ref: dict[str, dict[str, Any]] = {}
    for recipe in normalized_recipes:
        legacy_ref = recipe["legacy_recipe_ref"]
        if legacy_ref in by_legacy_ref:
            raise train.TrainManifestError(
                f"qualification profile v2: duplicate recipe preimage for {legacy_ref}"
            )
        by_legacy_ref[legacy_ref] = recipe

    expected = normalized_profile["required_recipe_ids"]
    actual = sorted(by_legacy_ref)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise train.TrainManifestError(
            f"qualification profile v2: recipe preimage coverage mismatch: missing={missing}, extra={extra}"
        )

    ordered = [by_legacy_ref[legacy_ref] for legacy_ref in expected]
    return normalized_profile, ordered


def _profile_fields(profile: Any, recipes: list[Any]) -> list[tuple[str, bytes]]:
    normalized, resolved_recipes = resolve_profile_recipes(profile, recipes)
    resolved_recipe_ids = [recipe["recipe_id"] for recipe in resolved_recipes]
    return [
        ("profile_name", framing.encode_text(normalized["profile_name"])),
        ("resolved_recipe_ids", _encoded_text_set(resolved_recipe_ids)),
        ("owned_surface_rules", _encoded_text_set(normalized["owned_surface_rules"])),
        (
            "required_cross_cutting_rules",
            _encoded_text_set(normalized["required_cross_cutting_rules"]),
        ),
        ("fallback_policy", framing.encode_enum(normalized["fallback_policy"])),
        ("non_claims", _encoded_text_set(normalized["non_claims"])),
    ]


def profile_frame_v2(profile: Any, recipes: list[Any]) -> bytes:
    """Return normative framed bytes for a recipe-preimage-complete profile V2 identity."""

    return framing.frame_record(PROFILE_ID_DOMAIN, _profile_fields(profile, recipes))


def compute_profile_id_v2(profile: Any, recipes: list[Any]) -> str:
    """Compute a framed V2 profile ID after resolving every exact recipe preimage."""

    return framing.semantic_sha256_id(PROFILE_ID_DOMAIN, _profile_fields(profile, recipes))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="kind", required=True)

    subject_parser = subparsers.add_parser("subject")
    subject_parser.add_argument("path", type=Path)
    subject_parser.add_argument("--print-frame-hex", action="store_true")

    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("path", type=Path)
    profile_parser.add_argument(
        "--recipe",
        type=Path,
        action="append",
        required=True,
        help="exact recipe manifest; repeat once for every V1 required_recipe_id",
    )
    profile_parser.add_argument("--print-frame-hex", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.kind == "subject":
        value = subject_mod.load_subject(args.path)
        if args.print_frame_hex:
            print(subject_frame_v2(value).hex())
        else:
            print(compute_subject_id_v2(value))
        return 0

    value = profile_mod.load_profile(args.path)
    recipes = [recipe_mod.load_recipe(path, require_id=True) for path in args.recipe]
    if args.print_frame_hex:
        print(profile_frame_v2(value, recipes).hex())
    else:
        print(compute_profile_id_v2(value, recipes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
