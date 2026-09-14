import hashlib
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
load("qualification_attempt_v3")
load("qualification_evidence_id")
attempts_v4 = load("qualification_attempt_v4")
load("qualification_attempt_subject")
load("qualification_pass_selection_v2")
load("qualification_evidence_binding")
load("qualification_support_closure")
gate_mod = load("qualification_receipt_candidate_gate")


def sha256_id(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def git_blob_id(char: str) -> str:
    return "git-blob-sha1:" + char * 40


def fixture(recipe_bytes=b"qualified recipe support\n"):
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
        "profile_name": "receipt-candidate-v1",
        "required_recipe_ids": [git_blob_id("1")],
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
        "reason": "qualify receipt candidate gate",
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
        "recipe_id": prof["required_recipe_ids"][0],
        "input_closure_id": sha256_id(b"input-closure"),
        "qualification_environment_id": sha256_id(b"environment"),
        "attempt_sequence": 1,
        "non_claims": ["attempt sequence is not external chronology"],
    }
    normalized_reg = attempts_v4.normalize_registration(reg, verify_declared_id=False)
    reg["attempt_registration_id"] = normalized_reg["attempt_registration_id"]

    content_id = sha256_id(recipe_bytes)
    obs = {
        "schema": attempts_v4.OBSERVATION_SCHEMA,
        "attempt_registration_id": reg["attempt_registration_id"],
        "execution_started": True,
        "terminal_disposition": "Passed",
        "provider_ref": {"provider": "github-actions", "attempt_ref": "candidate-run"},
        "evidence_content_ids": [content_id],
        "non_claims": ["provider-declared PASS is not trusted witness authority"],
    }
    normalized_obs = attempts_v4.normalize_observation(obs, verify_declared_id=False)
    obs["attempt_observation_id"] = normalized_obs["attempt_observation_id"]

    selected = {prof["required_recipe_ids"][0]: obs["attempt_observation_id"]}
    evidence = {content_id: recipe_bytes}
    return prof, adm, reg, obs, selected, evidence


def build(parts):
    prof, adm, reg, obs, selected, evidence = parts
    return gate_mod.build_receipt_candidate_gate(
        admission=adm,
        profile=prof,
        registrations=[reg],
        observations=[obs],
        selected_observation_ids_by_recipe=selected,
        cross_cutting_evidence=[],
        evidence_bytes_by_content_id=evidence,
    )


def test_v1_emits_witness_required_never_pass():
    result = build(fixture())
    assert result["disposition"] == "WitnessRequired"
    assert result["disposition"] != "Passed"
    assert "WitnessRequired is not qualification PASS or receipt authority" in result["non_claims"]


def test_generic_cross_cutting_rule_is_not_receipt_eligible():
    prof, adm, reg, obs, selected, evidence = fixture()
    prof = dict(prof)
    prof["required_cross_cutting_rules"] = ["workspace-lock-current"]
    prof.pop("profile_id")
    prof["profile_id"] = profile_mod.compute_profile_id(prof)
    with pytest.raises(train.TrainManifestError, match="generic required_cross_cutting_rules"):
        gate_mod.build_receipt_candidate_gate(
            admission=adm,
            profile=prof,
            registrations=[reg],
            observations=[obs],
            selected_observation_ids_by_recipe=selected,
            cross_cutting_evidence=[],
            evidence_bytes_by_content_id=evidence,
        )


def test_cross_cutting_evidence_cannot_be_smuggled_into_empty_profile():
    prof, adm, reg, obs, selected, evidence = fixture()
    with pytest.raises(train.TrainManifestError, match="cross_cutting_evidence must be empty"):
        gate_mod.build_receipt_candidate_gate(
            admission=adm,
            profile=prof,
            registrations=[reg],
            observations=[obs],
            selected_observation_ids_by_recipe=selected,
            cross_cutting_evidence=[
                {"rule": "workspace-lock-current", "evidence_content_ids": [sha256_id(b"lock")]}
            ],
            evidence_bytes_by_content_id=evidence,
        )


def test_missing_recipe_evidence_bytes_still_fail_closed():
    parts = list(fixture())
    parts[-1] = {}
    with pytest.raises(train.TrainManifestError, match="must exactly equal selected content ids"):
        build(tuple(parts))


def test_exact_evidence_change_changes_receipt_candidate_identity():
    first = build(fixture(recipe_bytes=b"support-a\n"))
    second = build(fixture(recipe_bytes=b"support-b\n"))
    assert first["receipt_candidate_id"] != second["receipt_candidate_id"]
    assert first["qualification_subject_id"] == second["qualification_subject_id"]


def test_provider_declared_pass_remains_untrusted():
    result = build(fixture())
    assert (
        "candidate eligibility does not establish that provider-declared Passed dispositions are trustworthy"
        in result["non_claims"]
    )
    assert "candidate eligibility does not establish provider/run provenance or producer authenticity" in result[
        "non_claims"
    ]
