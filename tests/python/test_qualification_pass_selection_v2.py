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
load("integration_train_catalog")
profile_mod = load("qualification_profile")
subject_mod = load("qualification_subject")
admission_mod = load("qualification_admission_v3")
attempts_v3 = load("qualification_attempt_v3")
load("qualification_evidence_id")
attempts_v4 = load("qualification_attempt_v4")
load("qualification_attempt_subject")
selection_v2 = load("qualification_pass_selection_v2")


def ident(char):
    return "sha256:" + char * 64


def blob(char):
    return "git-blob-sha1:" + char * 40


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
        "profile_name": "pass-selection-content-evidence.v1",
        "required_recipe_ids": [blob("1")],
        "owned_surface_rules": ["owns:test"],
        "required_cross_cutting_rules": ["workspace-lock-current"],
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
            "profile_branch": "evidence/profile",
            "profile_path": "docs/profile.json",
        },
        "reason": "qualify",
        "evidence_refs": ["issue:986"],
        "non_claims": ["does not establish PASS"],
    }
    normalized_adm = admission_mod.normalize_request(adm, verify_declared_ids=False)
    adm["admission_id"] = normalized_adm["admission_id"]
    adm["admission_subject_id"] = normalized_adm["admission_subject_id"]

    def reg(sequence=1):
        raw = {
            "schema": attempts_v4.REGISTRATION_SCHEMA,
            "admission_subject_id": adm["admission_subject_id"],
            "qualification_subject_id": subj["subject_id"],
            "qualification_profile_id": prof["profile_id"],
            "recipe_id": prof["required_recipe_ids"][0],
            "input_closure_id": ident("c"),
            "qualification_environment_id": ident("d"),
            "attempt_sequence": sequence,
            "non_claims": ["attempt sequence is not external chronology"],
        }
        normalized = attempts_v4.normalize_registration(raw, verify_declared_id=False)
        raw["attempt_registration_id"] = normalized["attempt_registration_id"]
        return raw

    def obs(registration, terminal="Passed", evidence=None, ref="run-1"):
        raw = {
            "schema": attempts_v4.OBSERVATION_SCHEMA,
            "attempt_registration_id": registration["attempt_registration_id"],
            "execution_started": True,
            "terminal_disposition": terminal,
            "provider_ref": {"provider": "github-actions", "attempt_ref": ref},
            "evidence_content_ids": sorted(evidence or [ident("e")]),
            "non_claims": ["content-id descriptor is not producer authenticity"],
        }
        normalized = attempts_v4.normalize_observation(raw, verify_declared_id=False)
        raw["attempt_observation_id"] = normalized["attempt_observation_id"]
        return raw

    return prof, adm, reg, obs


def build(prof, adm, registrations, observations, selected, cross_id):
    return selection_v2.build_pass_selection(
        admission=adm,
        profile=prof,
        registrations=registrations,
        observations=observations,
        selected_observation_ids_by_recipe={prof["required_recipe_ids"][0]: selected},
        cross_cutting_evidence=[
            {"rule": "workspace-lock-current", "evidence_content_ids": [cross_id]}
        ],
    )


def test_v2_pass_selection_binds_recipe_and_cross_cutting_content_descriptors():
    prof, adm, reg, obs = fixture()
    registration = reg()
    passed = obs(registration, evidence=[ident("1")])
    selection = build(prof, adm, [registration], [passed], passed["attempt_observation_id"], ident("2"))
    assert selection["selected_recipe_attempts"][0]["evidence_content_ids"] == [ident("1")]
    assert selection["cross_cutting_evidence"][0]["evidence_content_ids"] == [ident("2")]


def test_v2_rejects_label_like_cross_cutting_evidence():
    prof, adm, reg, obs = fixture()
    registration = reg()
    passed = obs(registration)
    with pytest.raises(train.TrainManifestError, match="expected sha256"):
        build(prof, adm, [registration], [passed], passed["attempt_observation_id"], "evidence:lock")


def test_descriptor_validity_is_not_byte_verification():
    prof, adm, reg, obs = fixture()
    registration = reg()
    passed = obs(registration, evidence=[ident("a")])
    selection = build(prof, adm, [registration], [passed], passed["attempt_observation_id"], ident("b"))
    assert (
        "content-id descriptor validity does not establish referenced bytes exist or match the digest"
        in selection["non_claims"]
    )


def test_cross_cutting_content_descriptor_change_changes_pass_selection_identity():
    prof, adm, reg, obs = fixture()
    registration = reg()
    passed = obs(registration)
    first = build(prof, adm, [registration], [passed], passed["attempt_observation_id"], ident("1"))
    second = build(prof, adm, [registration], [passed], passed["attempt_observation_id"], ident("2"))
    assert first["pass_selection_id"] != second["pass_selection_id"]
    assert first["selected_recipe_attempts"] == second["selected_recipe_attempts"]


def test_retry_history_remains_bound_even_when_selected_pass_is_same():
    prof, adm, reg, obs = fixture()
    passed_reg = reg(sequence=1)
    passed = obs(passed_reg, evidence=[ident("a")], ref="pass")
    clean = build(prof, adm, [passed_reg], [passed], passed["attempt_observation_id"], ident("c"))
    failed_reg = reg(sequence=2)
    failed = obs(failed_reg, terminal="RecipeFailed", evidence=[ident("f")], ref="fail")
    with_failure = build(
        prof,
        adm,
        [passed_reg, failed_reg],
        [passed, failed],
        passed["attempt_observation_id"],
        ident("c"),
    )
    assert clean["selected_recipe_attempts"] == with_failure["selected_recipe_attempts"]
    assert clean["attempt_history_id"] != with_failure["attempt_history_id"]
    assert clean["pass_selection_id"] != with_failure["pass_selection_id"]


def test_content_identity_is_explicitly_not_authenticity_or_sufficiency():
    prof, adm, reg, obs = fixture()
    registration = reg()
    passed = obs(registration)
    selection = build(prof, adm, [registration], [passed], passed["attempt_observation_id"], ident("a"))
    assert (
        "content-addressed evidence identity does not establish authenticity, correctness, trust, or sufficiency"
        in selection["non_claims"]
    )


def test_v3_observation_cannot_be_selected_by_v2():
    prof, adm, reg, _ = fixture()
    registration = reg()
    old = {
        "schema": attempts_v3.OBSERVATION_SCHEMA,
        "attempt_registration_id": registration["attempt_registration_id"],
        "execution_started": True,
        "terminal_disposition": "Passed",
        "provider_ref": {"provider": "github-actions", "attempt_ref": "old"},
        "evidence_refs": ["artifact:receipt"],
        "non_claims": ["historical V3 semantics"],
    }
    old_normalized = attempts_v3.normalize_observation(old, verify_declared_id=False)
    old["attempt_observation_id"] = old_normalized["attempt_observation_id"]
    with pytest.raises(train.TrainManifestError, match="missing fields|unknown fields"):
        selection_v2.build_pass_selection(
            admission=adm,
            profile=prof,
            registrations=[registration],
            observations=[old],
            selected_observation_ids_by_recipe={prof["required_recipe_ids"][0]: old["attempt_observation_id"]},
            cross_cutting_evidence=[
                {"rule": "workspace-lock-current", "evidence_content_ids": [ident("c")]}
            ],
        )
