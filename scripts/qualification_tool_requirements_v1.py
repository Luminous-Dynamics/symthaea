#!/usr/bin/env python3
"""Content-addressed tool requirements for one recipe-preimage-complete qualification profile.

This binds a reviewed tool-observation contract to the V2 profile and exact resolved recipe set.
It does not infer every transitive subprocess from Cargo/shell behavior; the later realization
verifier must observe every declared required tool from the captured realized closure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import integration_train_manifest as train
import qualification_framing_v1 as framing
import qualification_recipe_v1 as recipe_mod
import qualification_semantic_ids_v2 as semantic_v2
import qualification_subject as subject_mod

SCHEMA = "symthaea.qualification-tool-requirements.v1"
ID_DOMAIN = "symthaea.qualification-tool-requirements-id.v1"

NON_CLAIMS = sorted([
    "does not infer hidden/transitive subprocesses invoked by qualification tools",
    "does not prove recipe commands executed",
    "does not prove the declared tool set is minimal",
    "does not prove the required tools are present in any realized environment",
])


def _normalize_tool(value: Any, *, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise train.TrainManifestError(f"{where}: expected object")
    subject_mod._require_exact_keys(
        value,
        {"tool_name", "executable_basename", "version_argv_tail", "required_for_recipe_ids"},
        set(),
        where=where,
    )
    tool_name = subject_mod._require_string(value["tool_name"], where=f"{where}.tool_name")
    executable_basename = subject_mod._require_string(
        value["executable_basename"], where=f"{where}.executable_basename"
    )
    if "/" in executable_basename:
        raise train.TrainManifestError(f"{where}.executable_basename: basename must not contain '/'")

    argv_tail = value["version_argv_tail"]
    if not isinstance(argv_tail, list) or not argv_tail:
        raise train.TrainManifestError(f"{where}.version_argv_tail: expected non-empty array")
    argv_tail = [
        subject_mod._require_string(item, where=f"{where}.version_argv_tail[{index}]")
        for index, item in enumerate(argv_tail)
    ]

    recipe_ids = value["required_for_recipe_ids"]
    if not isinstance(recipe_ids, list) or not recipe_ids:
        raise train.TrainManifestError(f"{where}.required_for_recipe_ids: expected non-empty array")
    recipe_ids = [
        recipe_mod._require_id(item, where=f"{where}.required_for_recipe_ids[{index}]")
        for index, item in enumerate(recipe_ids)
    ]
    if recipe_ids != sorted(set(recipe_ids)):
        raise train.TrainManifestError(
            f"{where}.required_for_recipe_ids: must be sorted and unique"
        )
    return {
        "tool_name": tool_name,
        "executable_basename": executable_basename,
        "version_argv_tail": argv_tail,
        "required_for_recipe_ids": recipe_ids,
    }


def _tool_frame(tool: dict[str, Any]) -> bytes:
    return framing.frame_record(
        "symthaea.qualification-tool-requirement.v1",
        [
            ("tool_name", framing.encode_text(tool["tool_name"])),
            ("executable_basename", framing.encode_text(tool["executable_basename"])),
            (
                "version_argv_tail",
                framing.encode_list(
                    [framing.encode_text(item) for item in tool["version_argv_tail"]]
                ),
            ),
            (
                "required_for_recipe_ids",
                framing.encode_set(
                    framing.encode_text(item) for item in tool["required_for_recipe_ids"]
                ),
            ),
        ],
    )


def _fields(value: dict[str, Any]) -> list[tuple[str, bytes]]:
    return [
        (
            "qualification_profile_id_v2",
            framing.encode_text(value["qualification_profile_id_v2"]),
        ),
        (
            "resolved_recipe_ids",
            framing.encode_set(
                framing.encode_text(item) for item in value["resolved_recipe_ids"]
            ),
        ),
        ("tools", framing.encode_set(_tool_frame(tool) for tool in value["tools"])),
        (
            "non_claims",
            framing.encode_set(framing.encode_text(item) for item in value["non_claims"]),
        ),
    ]


def normalize_requirements(
    value: Any,
    *,
    profile: Any,
    recipes: list[Any],
    verify_declared_id: bool = True,
    require_id: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise train.TrainManifestError("qualification tool requirements: expected object")
    subject_mod._require_exact_keys(
        value,
        {"schema", "qualification_profile_id_v2", "resolved_recipe_ids", "tools", "non_claims"},
        {"tool_requirements_id"},
        where="qualification tool requirements",
    )
    if value["schema"] != SCHEMA:
        raise train.TrainManifestError(
            f"qualification tool requirements.schema: expected {SCHEMA!r}"
        )
    if require_id and "tool_requirements_id" not in value:
        raise train.TrainManifestError(
            "qualification tool requirements.tool_requirements_id: required"
        )

    normalized_profile, resolved_recipes = semantic_v2.resolve_profile_recipes(profile, recipes)
    profile_id_v2 = semantic_v2.compute_profile_id_v2(normalized_profile, resolved_recipes)
    resolved_recipe_ids = sorted(recipe["recipe_id"] for recipe in resolved_recipes)

    declared_profile_id = subject_mod._require_string(
        value["qualification_profile_id_v2"],
        where="qualification tool requirements.qualification_profile_id_v2",
    )
    if declared_profile_id != profile_id_v2:
        raise train.TrainManifestError(
            "qualification tool requirements: V2 profile identity does not match exact profile/recipe preimages"
        )

    declared_recipes = value["resolved_recipe_ids"]
    if not isinstance(declared_recipes, list):
        raise train.TrainManifestError(
            "qualification tool requirements.resolved_recipe_ids: expected array"
        )
    declared_recipes = [
        recipe_mod._require_id(
            item, where=f"qualification tool requirements.resolved_recipe_ids[{index}]"
        )
        for index, item in enumerate(declared_recipes)
    ]
    if declared_recipes != resolved_recipe_ids:
        raise train.TrainManifestError(
            "qualification tool requirements: resolved recipe set does not equal exact V2 profile recipe set"
        )

    tools_raw = value["tools"]
    if not isinstance(tools_raw, list) or not tools_raw:
        raise train.TrainManifestError(
            "qualification tool requirements.tools: expected non-empty array"
        )
    tools = [
        _normalize_tool(item, where=f"qualification tool requirements.tools[{index}]")
        for index, item in enumerate(tools_raw)
    ]
    tools.sort(key=lambda item: item["tool_name"])
    names = [tool["tool_name"] for tool in tools]
    if len(names) != len(set(names)):
        raise train.TrainManifestError(
            "qualification tool requirements.tools: duplicate tool_name"
        )

    known_recipe_ids = set(resolved_recipe_ids)
    covered: set[str] = set()
    for tool in tools:
        unknown = sorted(set(tool["required_for_recipe_ids"]) - known_recipe_ids)
        if unknown:
            raise train.TrainManifestError(
                f"qualification tool requirements: tool {tool['tool_name']} references recipes outside exact profile: {unknown}"
            )
        covered.update(tool["required_for_recipe_ids"])
    if covered != known_recipe_ids:
        missing = sorted(known_recipe_ids - covered)
        raise train.TrainManifestError(
            "qualification tool requirements: every profile recipe must be covered by at least one "
            f"required tool: missing={missing}"
        )

    # Automatic floor: every bare executable directly named by a recipe must be represented by a
    # tool requirement for that exact recipe. Repository-relative argv entries containing '/' are
    # source/input-closure concerns; auxiliary subprocesses remain explicitly reviewed profile
    # semantics rather than guessed here.
    tools_by_recipe: dict[str, set[str]] = {recipe_id: set() for recipe_id in resolved_recipe_ids}
    for tool in tools:
        for recipe_id in tool["required_for_recipe_ids"]:
            tools_by_recipe[recipe_id].add(tool["executable_basename"])
    for recipe in resolved_recipes:
        directly_invoked = sorted(
            {
                step["argv"][0]
                for step in recipe["steps"]
                if "/" not in step["argv"][0]
            }
        )
        missing_direct = sorted(set(directly_invoked) - tools_by_recipe[recipe["recipe_id"]])
        if missing_direct:
            raise train.TrainManifestError(
                "qualification tool requirements: direct recipe executables are not covered by "
                f"tool requirements for {recipe['recipe_id']}: {missing_direct}"
            )

    non_claims = train._require_sorted_unique_strings(
        value["non_claims"], where="qualification tool requirements.non_claims"
    )
    if non_claims != NON_CLAIMS:
        raise train.TrainManifestError(
            "qualification tool requirements.non_claims: V1 theorem boundary must equal frozen set"
        )

    normalized: dict[str, Any] = {
        "schema": SCHEMA,
        "qualification_profile_id_v2": profile_id_v2,
        "resolved_recipe_ids": resolved_recipe_ids,
        "tools": tools,
        "non_claims": list(NON_CLAIMS),
    }
    expected_id = framing.semantic_sha256_id(ID_DOMAIN, _fields(normalized))
    normalized["tool_requirements_id"] = expected_id
    if verify_declared_id and "tool_requirements_id" in value:
        declared = value["tool_requirements_id"]
        if declared != expected_id:
            raise train.TrainManifestError(
                "qualification tool requirements.tool_requirements_id: requirements semantics changed"
            )
    return normalized


def load_requirements(
    path: Path, *, profile: Any, recipes: list[Any], require_id: bool = False
) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise train.TrainManifestError(f"{path}: {error}") from error
    return normalize_requirements(
        raw, profile=profile, recipes=recipes, require_id=require_id
    )
