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
attempt_subject = load("qualification_attempt_subject")


def ident(c):
    return "sha256:" + c * 64


def blob(c):
    return "git-blob-sha1:" + c * 40


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


def profile(recipe=blob("1")):
    raw = {
        "schema": profile_mod.SCHEMA,
        "profile_name": "research.integrity.focused.v1",
        "required_recipe_ids": [recipe],
        "owned_surface_rules": ["owns:research-integrity"],
        "required_cross_cutting_rules": ["workspace-lock-current"],
        "fallback_policy": "FULL_CI_REQUIRED_ON_UNKNOWN_OR_AMBIGUOUS",
        "non_claims": ["does not authorize merge"],
    }
    raw["profile_id"] = profile_mod.compute_profile_id(raw)
    return raw


def admission(prof, subj=None):
    subj = subj or subject()
    raw = {
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
    out = admission_mod.normalize_request(raw, verify_declared_ids=False)
    raw["admission_id"] = out["admission_id"]
    raw["admission_subject_id"] = out["admission_subject_id"]
    return raw


def registration(*, prof=None, subj=None, recipe=None, closure="c", env="d", sequence=1):
    prof = prof or profile(recipe or blob("1"))
    adm = admission(prof, subj)
    raw = {
        "schema": attempts.REGISTRATION_SCHEMA,
        "admission_subject_id": adm["admission_subject_id"],
        "qualification_subject_id": adm["subject"]["subject_id"],
        "qualification_profile_id": prof["profile_id"],
        "recipe_id": recipe or prof["required_recipe_ids"][0],
        "input_closure_id": ident(closure),
        "qualification_environment_id": ident(env),
        "attempt_sequence": sequence,
        "non_claims": ["attempt sequence is not external chronology"],
    }
    out = attempts.normalize_registration(raw, verify_declared_id=False)
    raw["attempt_registration_id"] = out["attempt_registration_id"]
    return raw


def test_retry_sequence_does_not_change_attempt_theorem():
    first = registration(sequence=1)
    second = registration(sequence=2)
    assert first["attempt_registration_id"] != second["attempt_registration_id"]
    assert attempt_subject.compute_attempt_subject_id(first) == attempt_subject.compute_attempt_subject_id(second)


@pytest.mark.parametrize(
    "changed",
    [
        {"closure": "9"},
        {"env": "9"},
        {"recipe": blob("9"), "prof": profile(blob("9"))},
        {"subj": subject(commit="9", tree="8")},
    ],
)
def test_source_recipe_closure_or_environment_change_changes_attempt_theorem(changed):
    base = registration()
    other = registration(**changed)
    assert attempt_subject.compute_attempt_subject_id(base) != attempt_subject.compute_attempt_subject_id(other)
    with pytest.raises(train.TrainManifestError, match="theorem changed"):
        attempt_subject.validate_same_attempt_subject(base, other)


def test_profile_semantics_change_changes_attempt_theorem_even_with_same_display_name():
    p1 = profile(blob("1"))
    p2 = {
        **p1,
        "required_cross_cutting_rules": ["workspace-lock-current", "strict-clippy-required"],
    }
    p2.pop("profile_id")
    p2["profile_id"] = profile_mod.compute_profile_id(p2)
    first = registration(prof=p1)
    second = registration(prof=p2, recipe=blob("1"))
    assert attempt_subject.compute_attempt_subject_id(first) != attempt_subject.compute_attempt_subject_id(second)
