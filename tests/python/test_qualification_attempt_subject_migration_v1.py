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
admission_mod = load("qualification_admission_v3")
load("qualification_attempt_v3")
load("qualification_evidence_id")
attempts_v4 = load("qualification_attempt_v4")
load("qualification_attempt_subject")
selection_v2 = load("qualification_pass_selection_v2")
load("qualification_framing_v1")
recipe_mod = load("qualification_recipe_v1")
load("qualification_semantic_ids_v2")
closure_mod = load("qualification_input_closure_v1")
load("qualification_pass_selection_semantic_projection_v2")
load("qualification_pass_selection_replay_v1")
realization_mod = load("qualification_environment_realization_v1")
migration = load("qualification_attempt_subject_migration_v1")

LEGACY = "git-blob-sha1:" + "1" * 40


def sha(char):
    return "sha256:" + char * 64


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
        "profile_name": "attempt.semantic-migration.v1",
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
        "recipe_name": "attempt.semantic-migration.recipe.v1",
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


def fixture(*, closure_id=None, environment_id=None, attempt_sequence=1):
    subj = subject()
    prof = profile()
    closure_id = closure_mod.build_from_subject(subj)["input_closure_id"] if closure_id is None else closure_id
    environment_id = sha("e") if environment_id is None else environment_id

    adm = {
        "schema": admission_mod.SCHEMA,
        "subject": subj,
        "qualification_profile": {
            "name": prof["profile_name"],
            "profile_id": prof["profile_id"],
            "profile_branch": "receipt/profile",
            "profile_path": "docs/profile.json",
        },
        "reason": "migrate selected attempt subject",
        "evidence_refs": ["issue:905"],
        "non_claims": ["does not establish PASS"],
    }
    normalized_adm = admission_mod.normalize_request(adm, verify_declared_ids=False)
    adm["admission_id"] = normalized_adm["admission_id"]
    adm["admission_subject_id"] = normalized_adm["admission_subject_id"]

    reg = {
        "schema": attempts_v4.REGISTRATION_SCHEMA,
        "admission_subject_id": adm["admission_subject_id"],
        "qualification_subject_id": subj["subject_id"],
        "qualification_profile_id": prof["profile_id"],
        "recipe_id": LEGACY,
        "input_closure_id": closure_id,
        "qualification_environment_id": environment_id,
        "attempt_sequence": attempt_sequence,
        "non_claims": ["attempt sequence is not external chronology"],
    }
    reg["attempt_registration_id"] = attempts_v4.normalize_registration(reg, verify_declared_id=False)["attempt_registration_id"]

    obs = {
        "schema": attempts_v4.OBSERVATION_SCHEMA,
        "attempt_registration_id": reg["attempt_registration_id"],
        "execution_started": True,
        "terminal_disposition": "Passed",
        "provider_ref": {"provider": "github-actions", "attempt_ref": f"run-{attempt_sequence}"},
        "evidence_content_ids": [sha("8")],
        "non_claims": ["provider-declared PASS is not trusted witness authority"],
    }
    obs["attempt_observation_id"] = attempts_v4.normalize_observation(obs, verify_declared_id=False)["attempt_observation_id"]

    selection = selection_v2.build_pass_selection(
        admission=adm,
        profile=prof,
        registrations=[reg],
        observations=[obs],
        selected_observation_ids_by_recipe={LEGACY: obs["attempt_observation_id"]},
        cross_cutting_evidence=[],
    )
    return {"subject": subj, "profile": prof, "admission": adm, "registration": reg, "observation": obs, "selection": selection, "environment_id": environment_id}


def migrate(monkeypatch, tmp_path, fx, recipes=None, realized_environment_id=None):
    realized_environment_id = fx["environment_id"] if realized_environment_id is None else realized_environment_id
    monkeypatch.setattr(realization_mod, "verify_realization_capture", lambda **kwargs: {"qualification_environment_id": realized_environment_id})
    return migration.migrate_selected_attempt(
        legacy_recipe_ref=LEGACY,
        selection=fx["selection"],
        admission=fx["admission"],
        profile=fx["profile"],
        registrations=[fx["registration"]],
        observations=[fx["observation"]],
        subject=fx["subject"],
        recipes=[recipe()] if recipes is None else recipes,
        environment_selection={},
        environment_resolution={},
        repo=tmp_path,
        tool_requirements={},
        capture={},
        capture_root=tmp_path,
    )


def test_positive_migration_requires_exact_normative_coordinates(monkeypatch, tmp_path):
    fx = fixture()
    result = migrate(monkeypatch, tmp_path, fx)
    assert result["state"] == migration.POSITIVE_STATE
    assert result["qualification_environment_id_v1"] == fx["environment_id"]
    assert result["qualification_attempt_subject_id_v2"].startswith("sha256:")
    assert result["migration_id"].startswith("sha256:")


def test_historical_closure_mismatch_fails_closed(monkeypatch, tmp_path):
    fx = fixture(closure_id=sha("9"))
    with pytest.raises(train.TrainManifestError, match="normative input closure"):
        migrate(monkeypatch, tmp_path, fx)


def test_historical_environment_mismatch_fails_closed(monkeypatch, tmp_path):
    fx = fixture(environment_id=sha("9"))
    with pytest.raises(train.TrainManifestError, match="verified realized environment"):
        migrate(monkeypatch, tmp_path, fx, realized_environment_id=sha("e"))


def test_retry_changes_migration_evidence_not_attempt_theorem(monkeypatch, tmp_path):
    first = migrate(monkeypatch, tmp_path, fixture(attempt_sequence=1))
    second = migrate(monkeypatch, tmp_path, fixture(attempt_sequence=2))
    assert first["qualification_attempt_subject_id_v2"] == second["qualification_attempt_subject_id_v2"]
    assert first["source_attempt_registration_id"] != second["source_attempt_registration_id"]
    assert first["migration_id"] != second["migration_id"]


def test_recipe_semantic_change_changes_normative_attempt_subject(monkeypatch, tmp_path):
    fx = fixture()
    first = migrate(monkeypatch, tmp_path, fx)
    second = migrate(monkeypatch, tmp_path, fx, recipes=[recipe("--verbose")])
    assert first["source_attempt_subject_id_v1"] == second["source_attempt_subject_id_v1"]
    assert first["qualification_attempt_subject_id_v2"] != second["qualification_attempt_subject_id_v2"]


def test_provider_pass_still_not_trust_authority(monkeypatch, tmp_path):
    result = migrate(monkeypatch, tmp_path, fixture())
    assert "does not establish that provider-declared Passed observations are trustworthy" in result["non_claims"]
