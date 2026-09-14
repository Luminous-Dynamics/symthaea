#!/usr/bin/env python3
"""Exact, framed qualification recipe preimages.

A recipe is the semantic command contract that a qualification profile requires. It is not an
execution receipt and it does not select an environment or input closure.

V1 is intentionally restrictive:

- ordered, fail-fast command steps only;
- direct argv execution only (no shell command strings);
- canonical repository-relative working directories;
- explicit non-secret per-step environment overrides;
- closed stdin;
- explicit acceptable exit-code set;
- all other environment semantics belong to the separately qualified environment preimage.

Repository scripts are allowed as argv entries, but their bytes are NOT proven by the recipe
alone; they must be present in the exact input closure used by the attempt.

`legacy_recipe_ref` bridges existing profile references to this stronger preimage. It is a
migration coordinate, not proof that the manifest is semantically equivalent by itself.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import integration_train_manifest as train
import qualification_framing_v1 as framing
import qualification_profile as profile_mod
import qualification_subject as subject_mod

SCHEMA = "symthaea.qualification-recipe.v1"
ID_DOMAIN = "symthaea.qualification-recipe-id.v1"
EXECUTION_MODEL = "OrderedFailFastArgv"
ENVIRONMENT_CONTRACT = "ExactQualificationEnvironmentRequired"
STDIN_POLICY = "Closed"
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class QualificationRecipeError(train.TrainManifestError):
    pass


def _require_text(value: Any, *, where: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise QualificationRecipeError(f"{where}: expected string")
    if unicodedata.normalize("NFC", value) != value:
        raise QualificationRecipeError(f"{where}: text must use Unicode NFC normalization")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise QualificationRecipeError(f"{where}: control characters are not canonical")
    if not allow_empty and not value:
        raise QualificationRecipeError(f"{where}: must not be empty")
    if len(value.encode("utf-8")) > train.MAX_TEXT_BYTES:
        raise QualificationRecipeError(f"{where}: exceeds {train.MAX_TEXT_BYTES} UTF-8 bytes")
    return value


def _require_repo_dir(value: Any, *, where: str) -> str:
    value = _require_text(value, where=where)
    if value == ".":
        return value
    if value.startswith("/") or value.endswith("/") or "\\" in value:
        raise QualificationRecipeError(f"{where}: expected canonical repository-relative POSIX path")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise QualificationRecipeError(f"{where}: path traversal/non-canonical segment")
    return value


def _normalize_env(value: Any, *, where: str) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise QualificationRecipeError(f"{where}: expected array")
    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise QualificationRecipeError(f"{where}[{index}]: expected object")
        subject_mod._require_exact_keys(
            item, {"name", "value"}, set(), where=f"{where}[{index}]"
        )
        name = _require_text(item["name"], where=f"{where}[{index}].name")
        if _ENV_NAME.fullmatch(name) is None:
            raise QualificationRecipeError(f"{where}[{index}].name: invalid environment variable name")
        if name in seen:
            raise QualificationRecipeError(f"{where}: duplicate environment override {name!r}")
        seen.add(name)
        normalized.append(
            {
                "name": name,
                "value": _require_text(
                    item["value"], where=f"{where}[{index}].value", allow_empty=True
                ),
            }
        )
    normalized.sort(key=lambda item: item["name"])
    return normalized


def _normalize_exit_codes(value: Any, *, where: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise QualificationRecipeError(f"{where}: expected non-empty array")
    out: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool) or not 0 <= item <= 255:
            raise QualificationRecipeError(f"{where}[{index}]: expected integer in [0, 255]")
        out.append(item)
    if out != sorted(set(out)):
        raise QualificationRecipeError(f"{where}: exit codes must be sorted and unique")
    return out


def _normalize_step(value: Any, *, index: int) -> dict[str, Any]:
    where = f"qualification recipe.steps[{index}]"
    if not isinstance(value, dict):
        raise QualificationRecipeError(f"{where}: expected object")
    subject_mod._require_exact_keys(
        value,
        {
            "step_name",
            "working_directory",
            "argv",
            "environment_overrides",
            "stdin_policy",
            "acceptable_exit_codes",
        },
        set(),
        where=where,
    )
    argv = value["argv"]
    if not isinstance(argv, list) or not argv:
        raise QualificationRecipeError(f"{where}.argv: expected non-empty array")
    normalized_argv = [
        _require_text(item, where=f"{where}.argv[{arg_index}]")
        for arg_index, item in enumerate(argv)
    ]
    stdin_policy = _require_text(value["stdin_policy"], where=f"{where}.stdin_policy")
    if stdin_policy != STDIN_POLICY:
        raise QualificationRecipeError(
            f"{where}.stdin_policy: V1 requires {STDIN_POLICY!r}"
        )
    return {
        "step_name": _require_text(value["step_name"], where=f"{where}.step_name"),
        "working_directory": _require_repo_dir(
            value["working_directory"], where=f"{where}.working_directory"
        ),
        "argv": normalized_argv,
        "environment_overrides": _normalize_env(
            value["environment_overrides"], where=f"{where}.environment_overrides"
        ),
        "stdin_policy": STDIN_POLICY,
        "acceptable_exit_codes": _normalize_exit_codes(
            value["acceptable_exit_codes"], where=f"{where}.acceptable_exit_codes"
        ),
    }


def _frame_step(step: dict[str, Any]) -> bytes:
    env_frames = [
        framing.frame_record(
            "symthaea.qualification-recipe-env.v1",
            [
                ("name", framing.encode_text(item["name"])),
                ("value", framing.encode_text(item["value"])),
            ],
        )
        for item in step["environment_overrides"]
    ]
    return framing.frame_record(
        "symthaea.qualification-recipe-step.v1",
        [
            ("step_name", framing.encode_text(step["step_name"])),
            ("working_directory", framing.encode_text(step["working_directory"])),
            ("argv", framing.encode_list([framing.encode_text(arg) for arg in step["argv"]])),
            ("environment_overrides", framing.encode_list(env_frames)),
            ("stdin_policy", framing.encode_enum(step["stdin_policy"])),
            (
                "acceptable_exit_codes",
                framing.encode_set(
                    framing.encode_u64(code) for code in step["acceptable_exit_codes"]
                ),
            ),
        ],
    )


def _fields(recipe: dict[str, Any]) -> list[tuple[str, bytes]]:
    return [
        ("recipe_name", framing.encode_text(recipe["recipe_name"])),
        ("legacy_recipe_ref", framing.encode_text(recipe["legacy_recipe_ref"])),
        ("execution_model", framing.encode_enum(recipe["execution_model"])),
        ("environment_contract", framing.encode_enum(recipe["environment_contract"])),
        ("steps", framing.encode_list([_frame_step(step) for step in recipe["steps"]])),
        (
            "non_claims",
            framing.encode_set(framing.encode_text(value) for value in recipe["non_claims"]),
        ),
    ]


def normalize_recipe(
    value: Any, *, verify_declared_id: bool = True, require_id: bool = False
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise QualificationRecipeError("qualification recipe: expected object")
    subject_mod._require_exact_keys(
        value,
        {
            "schema",
            "recipe_name",
            "legacy_recipe_ref",
            "execution_model",
            "environment_contract",
            "steps",
            "non_claims",
        },
        {"recipe_id"},
        where="qualification recipe",
    )
    if value["schema"] != SCHEMA:
        raise QualificationRecipeError(f"qualification recipe.schema: expected {SCHEMA!r}")
    if require_id and "recipe_id" not in value:
        raise QualificationRecipeError("qualification recipe.recipe_id: required but absent")

    execution_model = _require_text(
        value["execution_model"], where="qualification recipe.execution_model"
    )
    if execution_model != EXECUTION_MODEL:
        raise QualificationRecipeError(
            f"qualification recipe.execution_model: V1 requires {EXECUTION_MODEL!r}"
        )
    environment_contract = _require_text(
        value["environment_contract"], where="qualification recipe.environment_contract"
    )
    if environment_contract != ENVIRONMENT_CONTRACT:
        raise QualificationRecipeError(
            "qualification recipe.environment_contract: unsupported environment contract"
        )

    steps = value["steps"]
    if not isinstance(steps, list) or not steps:
        raise QualificationRecipeError("qualification recipe.steps: expected non-empty array")
    normalized_steps = [_normalize_step(step, index=index) for index, step in enumerate(steps)]
    step_names = [step["step_name"] for step in normalized_steps]
    if len(step_names) != len(set(step_names)):
        raise QualificationRecipeError("qualification recipe.steps: duplicate step_name")

    non_claims = train._require_sorted_unique_strings(
        value["non_claims"], where="qualification recipe.non_claims"
    )
    if not non_claims:
        raise QualificationRecipeError("qualification recipe.non_claims: must not be empty")

    normalized = {
        "schema": SCHEMA,
        "recipe_name": _require_text(value["recipe_name"], where="qualification recipe.recipe_name"),
        "legacy_recipe_ref": profile_mod._require_recipe_id(
            value["legacy_recipe_ref"], where="qualification recipe.legacy_recipe_ref"
        ),
        "execution_model": EXECUTION_MODEL,
        "environment_contract": ENVIRONMENT_CONTRACT,
        "steps": normalized_steps,
        "non_claims": non_claims,
    }
    recipe_id = framing.semantic_sha256_id(ID_DOMAIN, _fields(normalized))
    normalized["recipe_id"] = recipe_id
    if verify_declared_id and "recipe_id" in value:
        declared = profile_mod._require_profile_id(
            value["recipe_id"], where="qualification recipe.recipe_id"
        )
        if declared != recipe_id:
            raise QualificationRecipeError(
                f"qualification recipe.recipe_id: expected {recipe_id}, got {declared}"
            )
    return normalized


def compute_recipe_id(value: Any) -> str:
    return normalize_recipe(value, verify_declared_id=False)["recipe_id"]


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QualificationRecipeError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def load_recipe(path: Path, *, require_id: bool = False) -> dict[str, Any]:
    try:
        if path.stat().st_size > train.MAX_MANIFEST_BYTES:
            raise QualificationRecipeError(
                f"{path}: recipe exceeds {train.MAX_MANIFEST_BYTES} bytes"
            )
        raw = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_object_without_duplicate_keys
        )
    except QualificationRecipeError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualificationRecipeError(f"{path}: {error}") from error
    return normalize_recipe(raw, require_id=require_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--require-id", action="store_true")
    parser.add_argument("--print-normalized", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        normalized = load_recipe(args.recipe, require_id=args.require_id)
    except QualificationRecipeError as error:
        print(f"qualification recipe invalid: {error}")
        return 2
    if args.print_normalized:
        print(json.dumps(normalized, sort_keys=True, indent=2))
    else:
        print(normalized["recipe_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
