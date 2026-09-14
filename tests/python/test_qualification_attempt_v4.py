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
evidence_id_mod = load("qualification_evidence_id")
attempts_v4 = load("qualification_attempt_v4")


def ident(char):
    return "sha256:" + char * 64


def blob(char):
    return "git-blob-sha1:" + char * 40


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
        "profile_name": "evidence-content-identity.v1",
        "required_recipe_ids": [blob("1")],
        "owned_surface_rules": ["owns:test"],
        "required_cross_cutting_rules": ["workspace-lock-current"],
        "fallback_policy": "FULL_CI_REQUIRED_ON_UNKNOWN_OR_AMBIGUOUS",
        "non_claims": ["does not authorize merge"],
    }
    raw["profile_id"] = profile_mod.compute_profile_id(raw)
    return raw


def admission(prof):
    subj = subject()
    raw = {
        "schema": admission_mod.SCHEMA,
        "subject": subj,
        "qualification_profile": {
            "name": prof["profile_name"],
            "profile_id": prof["profile_id"],
            "profile_branch": "evidence/profile",
            "profile_path": "docs/profile.json",
        },
        "reason": "qualify exact evidence semantics",
        "evidence_refs": ["issue:986"],
        "non_claims": ["does not establish PASS"],
    }
    normalized = admission_mod.normalize_request(raw, verify_declared_ids=False)
    raw["admission_id"] = normalized["admission_id"]
    raw["admission_subject_id"] = normalized["admission_subject_id"]
    return raw


def registration(adm, prof, sequence=1):
    raw = {
        "schema": attempts_v4.REGISTRATION_SCHEMA,
        "admission_subject_id": adm["admission_subject_id"],
        "qualification_subject_id": adm["subject"]["subject_id"],
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


def observation(reg, *, terminal="Passed", started=True, evidence=None, provider_ref="run-1"):
    if evidence is None:
        evidence = [ident("e")] if terminal == "Passed" else []
    raw = {
        "schema": attempts_v4.OBSERVATION_SCHEMA,
        "attempt_registration_id": reg["attempt_registration_id"],
        "execution_started": started,
        "terminal_disposition": terminal,
        "provider_ref": {"provider": "github-actions", "attempt_ref": provider_ref},
        "evidence_content_ids": sorted(evidence),
        "non_claims": ["content-id descriptor is not producer authenticity"],
    }
    normalized = attempts_v4.normalize_observation(raw, verify_declared_id=False)
    raw["attempt_observation_id"] = normalized["attempt_observation_id"]
    return raw


def test_v4_preserves_v3_attempt_registration_identity():
    prof = profile()
    adm = admission(prof)
    reg = registration(adm, prof)
    assert attempts_v4.normalize_registration(reg, require_id=True) == attempts_v3.normalize_registration(
        reg, require_id=True
    )


def test_passed_requires_content_descriptor():
    prof = profile()
    adm = admission(prof)
    reg = registration(adm, prof)
    with pytest.raises(train.TrainManifestError, match="Passed requires at least one"):
        observation(reg, terminal="Passed", evidence=[])


def test_prestart_infrastructure_unavailable_may_have_no_content_artifact():
    prof = profile()
    adm = admission(prof)
    reg = registration(adm, prof)
    obs = observation(
        reg,
        terminal="InfrastructureUnavailableBeforeStart",
        started=False,
        evidence=[],
        provider_ref="queued-without-runner",
    )
    assert obs["evidence_content_ids"] == []
    assert obs["execution_started"] is False


def test_poststart_nonpass_may_have_no_content_artifact():
    prof = profile()
    adm = admission(prof)
    reg = registration(adm, prof)
    obs = observation(reg, terminal="RecipeFailed", started=True, evidence=[])
    assert obs["evidence_content_ids"] == []


def test_v4_rejects_label_like_execution_evidence():
    prof = profile()
    adm = admission(prof)
    reg = registration(adm, prof)
    with pytest.raises(train.TrainManifestError, match="expected sha256"):
        observation(reg, evidence=["artifact:receipt"])


def test_valid_digest_descriptor_is_not_byte_verification():
    descriptor = ident("a")
    assert evidence_id_mod.require_evidence_content_id(descriptor, where="fixture") == descriptor
    prof = profile()
    adm = admission(prof)
    reg = registration(adm, prof)
    obs = observation(reg, evidence=[descriptor])
    history = attempts_v4.build_history(
        admission=adm, profile=prof, registrations=[reg], observations=[obs]
    )
    assert (
        "content-id descriptor validity does not establish referenced bytes exist or match the digest"
        in history["non_claims"]
    )


def test_v3_history_is_not_silently_reinterpreted_as_v4():
    prof = profile()
    adm = admission(prof)
    reg = registration(adm, prof)
    old = {
        "schema": attempts_v3.OBSERVATION_SCHEMA,
        "attempt_registration_id": reg["attempt_registration_id"],
        "execution_started": True,
        "terminal_disposition": "Passed",
        "provider_ref": {"provider": "github-actions", "attempt_ref": "old-run"},
        "evidence_refs": ["artifact:receipt"],
        "non_claims": ["historical V3 semantics"],
    }
    assert attempts_v3.normalize_observation(old)["terminal_disposition"] == "Passed"
    with pytest.raises(train.TrainManifestError, match="unknown fields|missing fields"):
        attempts_v4.normalize_observation(old)


def test_evidence_content_descriptor_changes_observation_identity():
    prof = profile()
    adm = admission(prof)
    reg = registration(adm, prof)
    first = observation(reg, evidence=[ident("1")])
    second = observation(reg, evidence=[ident("2")])
    assert first["attempt_observation_id"] != second["attempt_observation_id"]


def test_provider_locator_remains_separate_provenance():
    prof = profile()
    adm = admission(prof)
    reg = registration(adm, prof)
    first = observation(reg, provider_ref="run-1")
    second = observation(reg, provider_ref="run-2")
    assert first["evidence_content_ids"] == second["evidence_content_ids"]
    assert first["attempt_observation_id"] != second["attempt_observation_id"]


def test_v4_history_retains_artifactless_failed_retry_and_later_pass():
    prof = profile()
    adm = admission(prof)
    failed_reg = registration(adm, prof, sequence=1)
    passed_reg = registration(adm, prof, sequence=2)
    failed = observation(failed_reg, terminal="RecipeFailed", evidence=[])
    passed = observation(passed_reg, evidence=[ident("a")])
    history = attempts_v4.build_history(
        admission=adm,
        profile=prof,
        registrations=[failed_reg, passed_reg],
        observations=[failed, passed],
    )
    assert history["terminal_observation_ids"] == sorted(
        [failed["attempt_observation_id"], passed["attempt_observation_id"]]
    )
    assert (
        "a non-PASS observation with no evidence content does not prove no diagnostics or provider evidence exist"
        in history["non_claims"]
    )
