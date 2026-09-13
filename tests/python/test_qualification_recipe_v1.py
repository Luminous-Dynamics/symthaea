import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


train = load("integration_train_manifest")
profile = load("qualification_profile")
subject = load("qualification_subject")
framing = load("qualification_framing_v1")
recipe = load("qualification_recipe_v1")


def format_recipe():
    return {
        "schema": recipe.SCHEMA,
        "recipe_name": "research.format.v1",
        "legacy_recipe_ref": "git-blob-sha1:" + "1" * 40,
        "execution_model": recipe.EXECUTION_MODEL,
        "environment_contract": recipe.ENVIRONMENT_CONTRACT,
        "steps": [
            {
                "step_name": "rustfmt",
                "working_directory": ".",
                "argv": ["cargo", "fmt", "--all", "--", "--check"],
                "environment_overrides": [],
                "stdin_policy": recipe.STDIN_POLICY,
                "acceptable_exit_codes": [0],
            }
        ],
        "non_claims": ["does not prove semantic correctness"],
    }


def test_format_recipe_golden_identity():
    assert recipe.compute_recipe_id(format_recipe()) == (
        "sha256:c55be666f8951c96d27695f87ce6dd65dd36b98f0970620a47c49e3de93f5a4e"
    )


def test_test_recipe_golden_identity():
    raw = {
        "schema": recipe.SCHEMA,
        "recipe_name": "research.test.v1",
        "legacy_recipe_ref": "sha256:" + "2" * 64,
        "execution_model": recipe.EXECUTION_MODEL,
        "environment_contract": recipe.ENVIRONMENT_CONTRACT,
        "steps": [
            {
                "step_name": "tests",
                "working_directory": ".",
                "argv": ["cargo", "test", "--locked", "-p", "symthaea-research-protocol"],
                "environment_overrides": [
                    {"name": "CARGO_TERM_COLOR", "value": "never"}
                ],
                "stdin_policy": recipe.STDIN_POLICY,
                "acceptable_exit_codes": [0],
            }
        ],
        "non_claims": ["does not prove scientific validity"],
    }
    assert recipe.compute_recipe_id(raw) == (
        "sha256:caca3e9da04f348183b13b2fb3dd1d0d0b4772339a86a1e01760935c2ffeefc2"
    )


def test_argv_order_and_flags_are_semantic():
    baseline = format_recipe()
    changed = format_recipe()
    changed["steps"][0]["argv"] = ["cargo", "fmt", "--", "--check", "--all"]
    assert recipe.compute_recipe_id(baseline) != recipe.compute_recipe_id(changed)


def test_environment_override_order_is_canonical_but_values_are_semantic():
    first = format_recipe()
    first["steps"][0]["environment_overrides"] = [
        {"name": "B", "value": "2"},
        {"name": "A", "value": "1"},
    ]
    second = format_recipe()
    second["steps"][0]["environment_overrides"] = [
        {"name": "A", "value": "1"},
        {"name": "B", "value": "2"},
    ]
    assert recipe.compute_recipe_id(first) == recipe.compute_recipe_id(second)

    changed = format_recipe()
    changed["steps"][0]["environment_overrides"] = [
        {"name": "A", "value": "different"},
        {"name": "B", "value": "2"},
    ]
    assert recipe.compute_recipe_id(first) != recipe.compute_recipe_id(changed)


def test_duplicate_environment_names_fail_closed():
    raw = format_recipe()
    raw["steps"][0]["environment_overrides"] = [
        {"name": "A", "value": "1"},
        {"name": "A", "value": "2"},
    ]
    with pytest.raises(train.TrainManifestError, match="duplicate environment override"):
        recipe.normalize_recipe(raw)


def test_noncanonical_working_directory_fails_closed():
    raw = format_recipe()
    raw["steps"][0]["working_directory"] = "crates/core/../domains"
    with pytest.raises(train.TrainManifestError, match="traversal"):
        recipe.normalize_recipe(raw)


def test_duplicate_step_names_fail_closed():
    raw = format_recipe()
    raw["steps"].append(dict(raw["steps"][0]))
    with pytest.raises(train.TrainManifestError, match="duplicate step_name"):
        recipe.normalize_recipe(raw)


def test_declared_recipe_id_mismatch_fails_closed():
    raw = format_recipe()
    raw["recipe_id"] = "sha256:" + "0" * 64
    with pytest.raises(train.TrainManifestError, match="recipe_id"):
        recipe.normalize_recipe(raw, require_id=True)
