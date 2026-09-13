import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


train = load("integration_train_manifest")
framing = load("qualification_framing_v1")
profile_mod = load("qualification_profile")
recipe_mod = load("qualification_recipe_v1")
semantic_v2 = load("qualification_semantic_ids_v2")
requirements = load("qualification_tool_requirements_v1")

LEGACY_RECIPE = "git-blob-sha1:" + "a" * 40


def recipe(argv0="cargo"):
    raw = {
        "schema": recipe_mod.SCHEMA,
        "recipe_name": "focused-check",
        "legacy_recipe_ref": LEGACY_RECIPE,
        "execution_model": recipe_mod.EXECUTION_MODEL,
        "environment_contract": recipe_mod.ENVIRONMENT_CONTRACT,
        "steps": [
            {
                "step_name": "check",
                "working_directory": ".",
                "argv": [argv0, "check", "--locked"],
                "environment_overrides": [],
                "stdin_policy": recipe_mod.STDIN_POLICY,
                "acceptable_exit_codes": [0],
            }
        ],
        "non_claims": ["does not prove scientific validity"],
    }
    return recipe_mod.normalize_recipe(raw, verify_declared_id=False)


def profile():
    raw = {
        "schema": profile_mod.SCHEMA,
        "profile_name": "test.focused.v1",
        "required_recipe_ids": [LEGACY_RECIPE],
        "owned_surface_rules": ["crates/core/test/**"],
        "required_cross_cutting_rules": ["Cargo.lock exact"],
        "fallback_policy": "FULL_CI_REQUIRED_ON_UNKNOWN_OR_AMBIGUOUS",
        "non_claims": ["does not replace full CI"],
    }
    return profile_mod.normalize_profile(raw, verify_declared_id=False)


def raw_requirements(rec, *, executable="cargo", tool_name="cargo"):
    prof = profile()
    return {
        "schema": requirements.SCHEMA,
        "qualification_profile_id_v2": semantic_v2.compute_profile_id_v2(prof, [rec]),
        "resolved_recipe_ids": [rec["recipe_id"]],
        "tools": [
            {
                "tool_name": tool_name,
                "executable_basename": executable,
                "version_argv_tail": ["--version"],
                "required_for_recipe_ids": [rec["recipe_id"]],
            }
        ],
        "non_claims": list(requirements.NON_CLAIMS),
    }


def test_direct_recipe_executable_must_be_covered():
    rec = recipe("cargo")
    raw = raw_requirements(rec, executable="rustc", tool_name="rustc")
    with pytest.raises(train.TrainManifestError, match="direct recipe executables"):
        requirements.normalize_requirements(raw, profile=profile(), recipes=[rec])


def test_exact_direct_recipe_tool_is_accepted():
    rec = recipe("cargo")
    value = requirements.normalize_requirements(
        raw_requirements(rec), profile=profile(), recipes=[rec], verify_declared_id=False
    )
    assert value["tools"][0]["executable_basename"] == "cargo"
    assert value["tool_requirements_id"].startswith("sha256:")


def test_repository_relative_command_is_input_closure_not_bare_tool_requirement():
    rec = recipe("scripts/check-focused.sh")
    raw = raw_requirements(rec, executable="python3", tool_name="python")
    value = requirements.normalize_requirements(
        raw, profile=profile(), recipes=[rec], verify_declared_id=False
    )
    assert value["tools"][0]["tool_name"] == "python"


def test_tool_requirement_change_changes_identity():
    rec = recipe("cargo")
    first = requirements.normalize_requirements(
        raw_requirements(rec), profile=profile(), recipes=[rec], verify_declared_id=False
    )
    changed_raw = raw_requirements(rec)
    changed_raw["tools"][0]["version_argv_tail"] = ["--version", "--verbose"]
    second = requirements.normalize_requirements(
        changed_raw, profile=profile(), recipes=[rec], verify_declared_id=False
    )
    assert first["tool_requirements_id"] != second["tool_requirements_id"]


def test_profile_recipe_substitution_fails():
    rec = recipe("cargo")
    raw = raw_requirements(rec)
    changed = recipe("rustc")
    with pytest.raises(train.TrainManifestError):
        requirements.normalize_requirements(raw, profile=profile(), recipes=[changed])
