#!/usr/bin/env python3
"""Framed V2 semantic identities for generic qualification subject/profile roots.

This is a migration bridge, not an in-place reinterpretation of existing identities.
The existing V1 transport schemas and their JSON-derived IDs remain distinct historical /
prototype objects. V2 IDs are computed from the normalized semantic fields using
Qualification Framing V1.

Only the two root identities migrate here:

    exact Git qualification subject -> QualificationSubjectIdV2
    qualification theorem profile   -> QualificationProfileIdV2

Admission, routing, attempt, history, pass-selection and receipt identities remain on their
existing schemas until their V2 migration is specified separately.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import qualification_framing_v1 as framing
import qualification_profile as profile_mod
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


def _profile_fields(profile: Any) -> list[tuple[str, bytes]]:
    # As with subjects, preserve V1 integrity during migration when a V1 ID is supplied.
    normalized = profile_mod.normalize_profile(profile, verify_declared_id=True)
    return [
        ("profile_name", framing.encode_text(normalized["profile_name"])),
        ("required_recipe_ids", _encoded_text_set(normalized["required_recipe_ids"])),
        ("owned_surface_rules", _encoded_text_set(normalized["owned_surface_rules"])),
        (
            "required_cross_cutting_rules",
            _encoded_text_set(normalized["required_cross_cutting_rules"]),
        ),
        ("fallback_policy", framing.encode_enum(normalized["fallback_policy"])),
        ("non_claims", _encoded_text_set(normalized["non_claims"])),
    ]


def profile_frame_v2(profile: Any) -> bytes:
    """Return the normative framed bytes for QualificationProfileIdV2."""

    return framing.frame_record(PROFILE_ID_DOMAIN, _profile_fields(profile))


def compute_profile_id_v2(profile: Any) -> str:
    """Compute a framed V2 profile ID without redefining the V1 profile ID."""

    return framing.semantic_sha256_id(PROFILE_ID_DOMAIN, _profile_fields(profile))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("subject", "profile"))
    parser.add_argument("path", type=Path)
    parser.add_argument(
        "--print-frame-hex",
        action="store_true",
        help="print the normative framed bytes instead of the semantic ID",
    )
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
    if args.print_frame_hex:
        print(profile_frame_v2(value).hex())
    else:
        print(compute_profile_id_v2(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
