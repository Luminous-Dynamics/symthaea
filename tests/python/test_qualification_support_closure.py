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
closure_mod = load("qualification_support_closure")


def sha256_id(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def blob_id(char: str) -> str:
    return "git-blob-sha1:" + char * 40


def fixture(recipe_bytes=b"recipe receipt\n", cross_bytes=b"lock evidence\n"):
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
        "profile_name": "support-closure.v1",
        "required_recipe_ids": [blob_id("1")],
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
            "profile_branch": "support/profile",
            "profile_path": "docs/profile.json",
        },
        "reason": "qualify support closure",
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
        "input_closure_id": sha256_id(b"closure"),
        "qualification_environment_id": sha256_id(b"environment"),
        "attempt_sequence": 1,
        "non_claims": ["attempt sequence is not external chronology"],
    }
    normalized_reg = attempts_v4.normalize_registration(reg, verify_declared_id=False)
    reg["attempt_registration_id"] = normalized_reg["attempt_registration_id"]

    recipe_id = sha256_id(recipe_bytes)
    obs = {
        "schema": attempts_v4.OBSERVATION_SCHEMA,
        "attempt_registration_id": reg["attempt_registration_id"],
        "execution_started": True,
        "terminal_disposition": "Passed",
        "provider_ref": {"provider": "github-actions", "attempt_ref": "candidate-run"},
        "evidence_content_ids": [recipe_id],
        "non_claims": ["provider-declared PASS is not trusted witness authority"],
    }
    normalized_obs = attempts_v4.normalize_observation(obs, verify_declared_id=False)
    obs["attempt_observation_id"] = normalized_obs["attempt_observation_id"]

    cross_id = sha256_id(cross_bytes)
    selected = {prof["required_recipe_ids"][0]: obs["attempt_observation_id"]}
    cross = [{"rule": "workspace-lock-current", "evidence_content_ids": [cross_id]}]
    evidence = {recipe_id: recipe_bytes, cross_id: cross_bytes}
    return prof, adm, reg, obs, selected, cross, evidence


def build(parts):
    prof, adm, reg, obs, selected, cross, evidence = parts
    return closure_mod.build_support_closure(
        admission=adm,
        profile=prof,
        registrations=[reg],
        observations=[obs],
        selected_observation_ids_by_recipe=selected,
        cross_cutting_evidence=cross,
        evidence_bytes_by_content_id=evidence,
    )


def test_exact_selected_support_is_byte_closed():
    parts = fixture()
    result = build(parts)
    assert result["selected_content_ids"] == sorted(parts[-1])
    assert [item["content_id"] for item in result["byte_bindings"]] == sorted(parts[-1])
    assert result["cross_cutting_rules"] == ["workspace-lock-current"]


def test_missing_selected_bytes_fail_closed():
    parts = list(fixture())
    evidence = dict(parts[-1])
    evidence.pop(next(iter(evidence)))
    parts[-1] = evidence
    with pytest.raises(train.TrainManifestError, match="must exactly equal selected content ids"):
        build(tuple(parts))


def test_extra_unselected_bytes_fail_closed():
    parts = list(fixture())
    evidence = dict(parts[-1])
    evidence[sha256_id(b"unselected")] = b"unselected"
    parts[-1] = evidence
    with pytest.raises(train.TrainManifestError, match="must exactly equal selected content ids"):
        build(tuple(parts))


def test_wrong_bytes_for_selected_descriptor_fail_digest_check():
    parts = list(fixture())
    evidence = dict(parts[-1])
    first = next(iter(evidence))
    evidence[first] = b"wrong bytes"
    parts[-1] = evidence
    with pytest.raises(train.TrainManifestError, match="digest mismatch"):
        build(tuple(parts))


def test_content_change_changes_support_closure_identity():
    first = build(fixture(recipe_bytes=b"recipe-a\n"))
    second = build(fixture(recipe_bytes=b"recipe-b\n"))
    assert first["support_closure_id"] != second["support_closure_id"]


def test_same_content_used_by_two_obligations_is_bound_once():
    shared = b"shared evidence\n"
    parts = list(fixture(recipe_bytes=shared, cross_bytes=shared))
    result = build(tuple(parts))
    assert len(result["selected_content_ids"]) == 1
    assert len(result["byte_bindings"]) == 1


def test_byte_closure_does_not_upgrade_pass_or_rule_semantics():
    result = build(fixture())
    assert (
        "byte closure does not establish that a selected attempt terminal disposition is trustworthy"
        in result["non_claims"]
    )
    assert (
        "byte closure does not establish that any cross-cutting semantic rule is satisfied"
        in result["non_claims"]
    )
    assert "byte closure does not establish producer or witness authenticity" in result["non_claims"]
