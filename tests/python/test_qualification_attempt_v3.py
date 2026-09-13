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
catalog = load("integration_train_catalog")
profile_mod = load("qualification_profile")
subject_mod = load("qualification_subject")
admission_mod = load("qualification_admission_v3")
attempts = load("qualification_attempt_v3")


def ident(char):
    return "sha256:" + char * 64


def blob(char):
    return "git-blob-sha1:" + char * 40


def subject(commit="a", tree="b"):
    raw = {
        "schema": subject_mod.SCHEMA,
        "kind": "GitCommit",
        "repository": "Luminous-Dynamics/symthaea",
        "object_format": "sha1",
        "source_commit": commit * 40,
        "source_tree": tree * 40,
    }
    raw["subject_id"] = subject_mod.compute_subject_id(raw)
    return raw


def profile(recipes=None, cross_cutting=None, name="research.integrity.focused.v1"):
    recipes = recipes or [blob("1"), blob("2")]
    raw = {
        "schema": profile_mod.SCHEMA,
        "profile_name": name,
        "required_recipe_ids": sorted(recipes),
        "owned_surface_rules": ["owns:crates/core/symthaea-research-*"],
        "required_cross_cutting_rules": sorted(cross_cutting or ["workspace-lock-current"]),
        "fallback_policy": "FULL_CI_REQUIRED_ON_UNKNOWN_OR_AMBIGUOUS",
        "non_claims": ["does not authorize merge"],
    }
    raw["profile_id"] = profile_mod.compute_profile_id(raw)
    return raw


def admission(prof=None, subj=None):
    prof = prof or profile()
    subj = subj or subject()
    raw = {
        "schema": admission_mod.SCHEMA,
        "subject": subj,
        "qualification_profile": {
            "name": prof["profile_name"],
            "profile_id": prof["profile_id"],
            "profile_branch": "evidence/research-profile",
            "profile_path": "docs/research-profile.json",
        },
        "reason": "explicit qualification promotion",
        "evidence_refs": ["issue:986"],
        "non_claims": ["does not imply qualification passed"],
    }
    normalized = admission_mod.normalize_request(raw, verify_declared_ids=False)
    raw["admission_id"] = normalized["admission_id"]
    raw["admission_subject_id"] = normalized["admission_subject_id"]
    return raw


def registration(adm=None, prof=None, recipe=None, sequence=1, closure="c", env="d"):
    prof = prof or profile()
    adm = adm or admission(prof)
    recipe = recipe or prof["required_recipe_ids"][0]
    raw = {
        "schema": attempts.REGISTRATION_SCHEMA,
        "admission_subject_id": adm["admission_subject_id"],
        "qualification_subject_id": adm["subject"]["subject_id"],
        "qualification_profile_id": prof["profile_id"],
        "recipe_id": recipe,
        "input_closure_id": ident(closure),
        "qualification_environment_id": ident(env),
        "attempt_sequence": sequence,
        "non_claims": ["attempt sequence is not externally anchored chronology"],
    }
    normalized = attempts.normalize_registration(raw, verify_declared_id=False)
    raw["attempt_registration_id"] = normalized["attempt_registration_id"]
    return raw


def observation(reg, terminal="Passed", started=True, provider_attempt="run-1"):
    raw = {
        "schema": attempts.OBSERVATION_SCHEMA,
        "attempt_registration_id": reg["attempt_registration_id"],
        "execution_started": started,
        "terminal_disposition": terminal,
        "provider_ref": {
            "provider": "github-actions",
            "attempt_ref": provider_attempt,
        },
        "evidence_refs": ["artifact:receipt"],
        "non_claims": ["provider metadata is not semantic work identity"],
    }
    normalized = attempts.normalize_observation(raw, verify_declared_id=False)
    raw["attempt_observation_id"] = normalized["attempt_observation_id"]
    return raw


def test_attempt_registration_binds_exact_generic_subject_profile_and_recipe():
    prof = profile()
    adm = admission(prof)
    reg = registration(adm, prof)
    normalized = attempts.validate_registration_against_admission(reg, adm, prof)
    assert normalized["qualification_subject_id"] == adm["subject"]["subject_id"]
    assert normalized["qualification_profile_id"] == prof["profile_id"]
    assert normalized["recipe_id"] in prof["required_recipe_ids"]


def test_profile_forbids_unrequired_recipe_attempt():
    prof = profile()
    adm = admission(prof)
    reg = registration(adm, prof, recipe=blob("9"))
    with pytest.raises(train.TrainManifestError, match="not required"):
        attempts.validate_registration_against_admission(reg, adm, prof)


def test_retry_sequence_creates_distinct_attempt_without_changing_work_subject():
    prof = profile()
    adm = admission(prof)
    first = registration(adm, prof, sequence=1)
    second = registration(adm, prof, sequence=2)
    assert first["attempt_registration_id"] != second["attempt_registration_id"]
    assert first["admission_subject_id"] == second["admission_subject_id"]
    assert first["qualification_subject_id"] == second["qualification_subject_id"]
    assert first["qualification_profile_id"] == second["qualification_profile_id"]


def test_provider_locator_changes_observation_not_registration_identity():
    reg = registration()
    first = observation(reg, provider_attempt="run-1")
    second = observation(reg, provider_attempt="other-provider-attempt")
    assert first["attempt_observation_id"] != second["attempt_observation_id"]
    assert first["attempt_registration_id"] == second["attempt_registration_id"]


def test_prestart_and_poststart_dispositions_cannot_be_substituted():
    reg = registration()
    with pytest.raises(train.TrainManifestError, match="pre-start terminal"):
        observation(reg, "CancelledBeforeStart", started=True)
    with pytest.raises(train.TrainManifestError, match="post-start terminal"):
        observation(reg, "RecipeFailed", started=False)


def test_prestart_infrastructure_is_not_recipe_failure():
    reg = registration()
    obs = observation(reg, "InfrastructureUnavailableBeforeStart", started=False)
    normalized = attempts.validate_observation_against_registration(obs, reg)
    assert normalized["execution_started"] is False
    assert normalized["terminal_disposition"] == "InfrastructureUnavailableBeforeStart"


def test_later_pass_preserves_earlier_failed_attempt_in_history():
    prof = profile(recipes=[blob("1")])
    adm = admission(prof)
    first = registration(adm, prof, sequence=1)
    second = registration(adm, prof, sequence=2)
    failed = observation(first, "RecipeFailed", started=True, provider_attempt="run-fail")
    passed = observation(second, "Passed", started=True, provider_attempt="run-pass")
    history = attempts.build_history(
        admission=adm,
        profile=prof,
        registrations=[first, second],
        observations=[failed, passed],
    )
    assert history["attempt_registration_ids"] == sorted(
        [first["attempt_registration_id"], second["attempt_registration_id"]]
    )
    assert history["terminal_observation_ids"] == sorted(
        [failed["attempt_observation_id"], passed["attempt_observation_id"]]
    )
    assert attempts.successful_attempts_by_recipe(
        registrations=[first, second], observations=[failed, passed]
    )[blob("1")] == [passed["attempt_observation_id"]]


def test_partial_history_is_allowed_but_does_not_claim_profile_qualification():
    prof = profile(recipes=[blob("1"), blob("2")])
    adm = admission(prof)
    reg = registration(adm, prof, recipe=blob("1"))
    obs = observation(reg)
    history = attempts.build_history(
        admission=adm, profile=prof, registrations=[reg], observations=[obs]
    )
    assert history["required_recipe_ids"] == [blob("1"), blob("2")]
    assert history["represented_recipe_ids"] == [blob("1")]
    assert "does not establish qualification merely because every recipe has an attempt" in history["non_claims"]


def test_cross_cutting_rule_is_not_laundered_into_recipe_execution():
    prof = profile(recipes=[blob("1")], cross_cutting=["workspace-lock-current"])
    adm = admission(prof)
    reg = registration(adm, prof)
    history = attempts.build_history(
        admission=adm,
        profile=prof,
        registrations=[reg],
        observations=[observation(reg)],
    )
    assert prof["required_cross_cutting_rules"] == ["workspace-lock-current"]
    assert "workspace-lock-current" not in history["represented_recipe_ids"]
    assert "does not prove required cross-cutting rules" in history["non_claims"]


def test_orphan_and_duplicate_terminal_observations_fail_closed():
    prof = profile(recipes=[blob("1")])
    adm = admission(prof)
    reg = registration(adm, prof)
    other = registration(adm, prof, sequence=2)
    orphan = observation(other)
    with pytest.raises(train.TrainManifestError, match="orphan"):
        attempts.build_history(
            admission=adm, profile=prof, registrations=[reg], observations=[orphan]
        )
    first = observation(reg, provider_attempt="run-1")
    second = observation(reg, provider_attempt="run-2")
    with pytest.raises(train.TrainManifestError, match="at most one"):
        attempts.build_history(
            admission=adm,
            profile=prof,
            registrations=[reg],
            observations=[first, second],
        )


def test_source_change_cannot_reuse_old_attempt_registration():
    prof = profile(recipes=[blob("1")])
    old_adm = admission(prof, subject(commit="a", tree="b"))
    new_adm = admission(prof, subject(commit="9", tree="8"))
    reg = registration(old_adm, prof)
    with pytest.raises(train.TrainManifestError, match="admission_subject_id"):
        attempts.validate_registration_against_admission(reg, new_adm, prof)
