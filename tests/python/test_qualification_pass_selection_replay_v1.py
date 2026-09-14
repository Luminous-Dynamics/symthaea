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
replay = load("qualification_pass_selection_replay_v1")

RECIPE = "git-blob-sha1:" + "1" * 40


def sha(char):
    return "sha256:" + char * 64


def fixture():
    subj = {
        "schema": subject_mod.SCHEMA,
        "kind": "GitCommit",
        "repository": "Luminous-Dynamics/symthaea",
        "object_format": "sha1",
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
    }
    subj["subject_id"] = subject_mod.compute_subject_id(subj)

    prof = {
        "schema": profile_mod.SCHEMA,
        "profile_name": "selection.replay.v1",
        "required_recipe_ids": [RECIPE],
        "owned_surface_rules": ["owns:test"],
        "required_cross_cutting_rules": [],
        "fallback_policy": "FULL_CI_REQUIRED_ON_UNKNOWN_OR_AMBIGUOUS",
        "non_claims": ["does not authorize merge"],
    }
    prof["profile_id"] = profile_mod.compute_profile_id(prof)

    adm = {
        "schema": admission_mod.SCHEMA,
        "subject": subj,
        "qualification_profile": {
            "name": prof["profile_name"],
            "profile_id": prof["profile_id"],
            "profile_branch": "receipt/profile",
            "profile_path": "docs/profile.json",
        },
        "reason": "replay exact pass selection",
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
        "recipe_id": RECIPE,
        "input_closure_id": sha("2"),
        "qualification_environment_id": sha("3"),
        "attempt_sequence": 1,
        "non_claims": ["attempt sequence is not external chronology"],
    }
    reg["attempt_registration_id"] = attempts_v4.normalize_registration(
        reg, verify_declared_id=False
    )["attempt_registration_id"]

    obs = {
        "schema": attempts_v4.OBSERVATION_SCHEMA,
        "attempt_registration_id": reg["attempt_registration_id"],
        "execution_started": True,
        "terminal_disposition": "Passed",
        "provider_ref": {"provider": "github-actions", "attempt_ref": "candidate-run"},
        "evidence_content_ids": [sha("8")],
        "non_claims": ["provider-declared PASS is not trusted witness authority"],
    }
    obs["attempt_observation_id"] = attempts_v4.normalize_observation(
        obs, verify_declared_id=False
    )["attempt_observation_id"]

    selection = selection_v2.build_pass_selection(
        admission=adm,
        profile=prof,
        registrations=[reg],
        observations=[obs],
        selected_observation_ids_by_recipe={RECIPE: obs["attempt_observation_id"]},
        cross_cutting_evidence=[],
    )
    return adm, prof, reg, obs, selection


def verify(parts):
    adm, prof, reg, obs, selection = parts
    return replay.verify_replay(
        selection=selection,
        admission=adm,
        profile=prof,
        registrations=[reg],
        observations=[obs],
    )


def test_exact_rebuild_is_accepted():
    result = verify(fixture())
    assert result["state"] == replay.POSITIVE_STATE
    assert result["pass_selection_id"].startswith("sha256:")
    assert result["replay_id"].startswith("sha256:")


def test_self_hashed_forged_nonclaims_fail_replay():
    parts = list(fixture())
    forged = dict(parts[-1])
    forged["non_claims"] = ["forged selection semantics"]
    payload = dict(forged)
    payload.pop("pass_selection_id")
    forged["pass_selection_id"] = selection_v2._selection_id(payload)
    parts[-1] = forged
    with pytest.raises(train.TrainManifestError, match="does not exactly equal deterministic rebuild"):
        verify(tuple(parts))


def test_self_hashed_forged_input_closure_fails_replay():
    parts = list(fixture())
    forged = dict(parts[-1])
    forged["input_closure_id"] = sha("9")
    payload = dict(forged)
    payload.pop("pass_selection_id")
    forged["pass_selection_id"] = selection_v2._selection_id(payload)
    parts[-1] = forged
    with pytest.raises(train.TrainManifestError, match="does not exactly equal deterministic rebuild"):
        verify(tuple(parts))


def test_selected_observation_not_backed_by_history_fails():
    parts = list(fixture())
    forged = dict(parts[-1])
    selected = [dict(forged["selected_recipe_attempts"][0])]
    selected[0]["attempt_observation_id"] = sha("f")
    forged["selected_recipe_attempts"] = selected
    payload = dict(forged)
    payload.pop("pass_selection_id")
    forged["pass_selection_id"] = selection_v2._selection_id(payload)
    parts[-1] = forged
    with pytest.raises(train.TrainManifestError):
        verify(tuple(parts))


def test_provider_declared_pass_remains_below_trust_authority():
    result = verify(fixture())
    assert (
        "does not establish provider authenticity or trustworthy Passed dispositions"
        in result["non_claims"]
    )
