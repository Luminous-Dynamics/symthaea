#!/usr/bin/env python3
"""Validate content-addressed qualification profile manifests.

A profile defines the semantic qualification theorem selected by routing/admission.
Presentation names and Git locators are deliberately excluded from identity except for the
human-readable `profile_name` itself, which is part of the declared semantics but is never used
alone as proof of theorem equivalence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import integration_train_manifest as train

SCHEMA = "symthaea.qualification-profile.v1"
DOMAIN = b"symthaea.qualification-profile.v1\0"
_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_RECIPE_RE = re.compile(r"^(?:sha256:[0-9a-f]{64}|git-blob-sha1:[0-9a-f]{40})$")
FALLBACK_POLICIES = {
    "FULL_CI_REQUIRED_ON_UNKNOWN_OR_AMBIGUOUS",
    "REFUSE_QUALIFICATION_ON_UNKNOWN_OR_AMBIGUOUS",
}


def _require_exact_keys(
    value: dict[str, Any], required: set[str], optional: set[str], *, where: str
) -> None:
    keys = set(value)
    missing = sorted(required - keys)
    unknown = sorted(keys - required - optional)
    if missing:
        raise train.TrainManifestError(f"{where}: missing fields: {', '.join(missing)}")
    if unknown:
        raise train.TrainManifestError(f"{where}: unknown fields: {', '.join(unknown)}")


def _require_string(value: Any, *, where: str) -> str:
    if not isinstance(value, str):
        raise train.TrainManifestError(f"{where}: expected string")
    if value != value.strip():
        raise train.TrainManifestError(f"{where}: leading/trailing whitespace is not canonical")
    if not value:
        raise train.TrainManifestError(f"{where}: must not be empty")
    if unicodedata.normalize("NFC", value) != value:
        raise train.TrainManifestError(f"{where}: text must use Unicode NFC normalization")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise train.TrainManifestError(f"{where}: control characters are not canonical")
    if len(value.encode("utf-8")) > train.MAX_TEXT_BYTES:
        raise train.TrainManifestError(f"{where}: exceeds {train.MAX_TEXT_BYTES} UTF-8 bytes")
    return value


def _require_profile_id(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise train.TrainManifestError(f"{where}: expected sha256:<64 lowercase hex>")
    return value


def _require_recipe_id(value: Any, *, where: str) -> str:
    value = _require_string(value, where=where)
    if _RECIPE_RE.fullmatch(value) is None:
        raise train.TrainManifestError(
            f"{where}: expected sha256:<64 lowercase hex> or git-blob-sha1:<40 lowercase hex>"
        )
    return value


def _require_sorted_unique_strings(value: Any, *, where: str) -> list[str]:
    if not isinstance(value, list):
        raise train.TrainManifestError(f"{where}: expected array")
    normalized = [_require_string(item, where=f"{where}[{index}]") for index, item in enumerate(value)]
    if normalized != sorted(normalized):
        raise train.TrainManifestError(f"{where}: values must be lexicographically sorted")
    if len(set(normalized)) != len(normalized):
        raise train.TrainManifestError(f"{where}: values must be unique")
    return normalized


def _require_sorted_unique_recipe_ids(value: Any, *, where: str) -> list[str]:
    if not isinstance(value, list):
        raise train.TrainManifestError(f"{where}: expected array")
    normalized = [
        _require_recipe_id(item, where=f"{where}[{index}]") for index, item in enumerate(value)
    ]
    if normalized != sorted(normalized):
        raise train.TrainManifestError(f"{where}: recipe ids must be lexicographically sorted")
    if len(set(normalized)) != len(normalized):
        raise train.TrainManifestError(f"{where}: recipe ids must be unique")
    return normalized


def _canonical_payload_from_normalized(profile: dict[str, Any]) -> bytes:
    payload = {key: value for key, value in profile.items() if key != "profile_id"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def _compute_profile_id_from_normalized(profile: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(
        DOMAIN + _canonical_payload_from_normalized(profile)
    ).hexdigest()


def normalize_profile(
    profile: Any, *, verify_declared_id: bool = True, require_id: bool = False
) -> dict[str, Any]:
    if not isinstance(profile, dict):
        raise train.TrainManifestError("qualification profile: expected object")
    _require_exact_keys(
        profile,
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
        where="qualification profile",
    )
    if profile["schema"] != SCHEMA:
        raise train.TrainManifestError(f"qualification profile.schema: expected {SCHEMA!r}")
    if require_id and "profile_id" not in profile:
        raise train.TrainManifestError("qualification profile.profile_id: required but absent")

    fallback_policy = _require_string(
        profile["fallback_policy"], where="qualification profile.fallback_policy"
    )
    if fallback_policy not in FALLBACK_POLICIES:
        raise train.TrainManifestError(
            "qualification profile.fallback_policy: unsupported fail-closed policy"
        )

    normalized = {
        "schema": SCHEMA,
        "profile_name": _require_string(
            profile["profile_name"], where="qualification profile.profile_name"
        ),
        "required_recipe_ids": _require_sorted_unique_recipe_ids(
            profile["required_recipe_ids"], where="qualification profile.required_recipe_ids"
        ),
        "owned_surface_rules": _require_sorted_unique_strings(
            profile["owned_surface_rules"], where="qualification profile.owned_surface_rules"
        ),
        "required_cross_cutting_rules": _require_sorted_unique_strings(
            profile["required_cross_cutting_rules"],
            where="qualification profile.required_cross_cutting_rules",
        ),
        "fallback_policy": fallback_policy,
        "non_claims": _require_sorted_unique_strings(
            profile["non_claims"], where="qualification profile.non_claims"
        ),
    }
    if not normalized["required_recipe_ids"]:
        raise train.TrainManifestError(
            "qualification profile.required_recipe_ids: must contain at least one recipe identity"
        )
    if not normalized["owned_surface_rules"]:
        raise train.TrainManifestError(
            "qualification profile.owned_surface_rules: must contain at least one rule"
        )
    if not normalized["non_claims"]:
        raise train.TrainManifestError(
            "qualification profile.non_claims: must contain at least one explicit non-claim"
        )

    profile_id = _compute_profile_id_from_normalized(normalized)
    normalized["profile_id"] = profile_id
    if verify_declared_id and "profile_id" in profile:
        declared = _require_profile_id(
            profile["profile_id"], where="qualification profile.profile_id"
        )
        if declared != profile_id:
            raise train.TrainManifestError(
                f"qualification profile.profile_id: expected {profile_id}, got {declared}"
            )
    return normalized


def compute_profile_id(profile: Any) -> str:
    normalized = normalize_profile(profile, verify_declared_id=False)
    return _compute_profile_id_from_normalized(normalized)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise train.TrainManifestError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def load_profile(path: Path, *, require_id: bool = False) -> dict[str, Any]:
    try:
        if path.stat().st_size > train.MAX_MANIFEST_BYTES:
            raise train.TrainManifestError(
                f"{path}: qualification profile exceeds {train.MAX_MANIFEST_BYTES} bytes"
            )
        raw = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_object_without_duplicate_keys
        )
    except train.TrainManifestError:
        raise
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise train.TrainManifestError(f"{path}: {error}") from error
    return normalize_profile(raw, require_id=require_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path, help="qualification profile JSON")
    parser.add_argument("--require-id", action="store_true", help="require exact declared profile_id")
    parser.add_argument(
        "--print-normalized", action="store_true", help="print canonical normalized JSON"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        normalized = load_profile(args.profile, require_id=args.require_id)
    except train.TrainManifestError as error:
        print(f"qualification profile invalid: {error}", file=sys.stderr)
        return 2
    if args.print_normalized:
        print(json.dumps(normalized, indent=2, ensure_ascii=False))
    else:
        print(normalized["profile_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
