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
load("integration_train_catalog")
profile_mod = load("qualification_profile")
subject_mod = load("qualification_subject")
load("qualification_admission_v3")
load("qualification_attempt_v3")
load("qualification_evidence_id")
load("qualification_attempt_v4")
load("qualification_attempt_subject")
selection_v2 = load("qualification_pass_selection_v2")
load("qualification_framing_v1")
recipe_mod = load("qualification_recipe_v1")
semantic_v2 = load("qualification_semantic_ids_v2")
closure_v1 = load("qualification_input_closure_v1")
projection = load("qualification_pass_selection_semantic_projection_v2")

LEGACY = "git-blob-sha1:" + "1" * 40


def subject():
    raw = {
        "schema": subject_mod.SCHEMA,
        "kind": "GitCommit",
        "repository": "Luminous-Dynamics/symthaea",
        "object_format": "sha1",
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
    }
    raw["subject_id"] = subject_mod.compute_subject_id(raw)
    return raw


def profile():
    raw = {
        "schema": profile_mod.SCHEMA,
        "profile_name": "receipt.semantic-migration.v1",
        "required_recipe_ids": [LEGACY],
        "owned_surface_rules": ["owns:test"],
        "required_cross_cutting_rules": [],
        "fallback_policy": "FULL_CI_REQUIRED_ON_UNKNOWN_OR_AMBIGUOUS",
        "non_claims": ["does not authorize merge"],
    }
    raw["profile_id"] = profile_mod.compute_profile_id(raw)
    return raw


def recipe(extra_arg=None):
    argv = ["cargo", "test", "--locked", "-p", "example"]
    if extra_arg is not None:
        argv.append(extra_arg)
    raw = {
        "schema": recipe_mod.SCHEMA,
        "recipe_name": "receipt.semantic-migration.recipe.v1",
        "legacy_recipe_ref": LEGACY,
        "execution_model": recipe_mod.EXECUTION_MODEL,
        "environment_contract": recipe_mod.ENVIRONMENT_CONTRACT,
        "steps": [{
            "step_name": "tests",
            "working_directory": ".",
            "argv": argv,
            "environment_overrides": [],
            "stdin_policy": recipe_mod.STDIN_POLICY,
            "acceptable_exit_codes": [0],
        }],
        "non_claims": ["does not prove semantic correctness"],
    }
    raw["recipe_id"] = recipe_mod.compute_recipe_id(raw)
    return raw


def sha(char):
    return "sha256:" + char * 64


def selection(input_closure_id=None):
    raw = {
        "schema": selection_v2.SCHEMA,
        "admission_subject_id": sha("1"),
        "qualification_subject_id": subject()["subject_id"],
        "qualification_profile_id": profile()["profile_id"],
        "input_closure_id": sha("2") if input_closure_id is None else input_closure_id,
        "qualification_environment_id": sha("3"),
        "attempt_history_id": sha("4"),
        "selected_recipe_attempts": [{
            "recipe_id": LEGACY,
            "attempt_subject_id": sha("5"),
            "attempt_registration_id": sha("6"),
            "attempt_observation_id": sha("7"),
            "evidence_content_ids": [sha("8")],
        }],
        "cross_cutting_evidence": [],
        "non_claims": ["historical selection fixture"],
    }
    raw["pass_selection_id"] = selection_v2._selection_id(raw)
    return raw


def project(recipes=None, source_selection=None):
    return projection.project_selection(
        selection=selection() if source_selection is None else source_selection,
        subject=subject(),
        profile=profile(),
        recipes=[recipe()] if recipes is None else recipes,
    )


def test_projection_binds_normative_input_closure():
    result = project()
    rec = recipe()
    assert result["qualification_subject_id_v2"] == semantic_v2.compute_subject_id_v2(subject())
    assert result["qualification_profile_id_v2"] == semantic_v2.compute_profile_id_v2(
        profile(), [rec]
    )
    assert result["qualification_input_closure_id_v1"] == closure_v1.build_from_subject(
        subject()
    )["input_closure_id"]
    assert result["input_closure_id_source"] == sha("2")
    assert result["recipe_mappings"][0]["recipe_id"] == rec["recipe_id"]


def test_historical_input_closure_remains_distinct_from_normative_reconstruction():
    first = project()
    second = project(source_selection=selection(input_closure_id=sha("9")))
    assert first["qualification_input_closure_id_v1"] == second[
        "qualification_input_closure_id_v1"
    ]
    assert first["input_closure_id_source"] != second["input_closure_id_source"]
    assert first["source_pass_selection_id"] != second["source_pass_selection_id"]
    assert first["projection_id"] != second["projection_id"]


def test_recipe_semantic_change_changes_projection_without_reinterpreting_source_selection():
    first = project()
    second = project([recipe("--verbose")])
    assert first["source_pass_selection_id"] == second["source_pass_selection_id"]
    assert first["recipe_mappings"][0]["recipe_id"] != second["recipe_mappings"][0]["recipe_id"]
    assert first["qualification_profile_id_v2"] != second["qualification_profile_id_v2"]
    assert first["projection_id"] != second["projection_id"]


def test_bad_source_pass_selection_id_cannot_be_laundered():
    bad = selection()
    bad["pass_selection_id"] = sha("f")
    with pytest.raises(train.TrainManifestError, match="does not match exact historical selection bytes"):
        projection.project_selection(
            selection=bad, subject=subject(), profile=profile(), recipes=[recipe()]
        )


def test_missing_recipe_preimage_fails_closed():
    with pytest.raises(train.TrainManifestError, match="recipe preimages are required"):
        project([])
