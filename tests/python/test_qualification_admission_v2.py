import importlib.util
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


integration_train = load_module("integration_train_manifest", SCRIPTS / "integration_train_manifest.py")

import sys
sys.modules["integration_train_manifest"] = integration_train
integration_catalog = load_module("integration_train_catalog", SCRIPTS / "integration_train_catalog.py")
sys.modules["integration_train_catalog"] = integration_catalog
qualification_admission = load_module("qualification_admission", SCRIPTS / "qualification_admission.py")
sys.modules["qualification_admission"] = qualification_admission
qualification_profile = load_module("qualification_profile", SCRIPTS / "qualification_profile.py")
sys.modules["qualification_profile"] = qualification_profile
admission_v2 = load_module("qualification_admission_v2", SCRIPTS / "qualification_admission_v2.py")


def sha(char):
    return char * 40


def ident(char):
    return "sha256:" + char * 64


def profile(recipe="git-blob-sha1:" + "a" * 40):
    raw = {
        "schema": qualification_profile.SCHEMA,
        "profile_name": "research.integrity.focused.v1",
        "required_recipe_ids": [recipe],
        "owned_surface_rules": [
            "crates/core/symthaea-research-analysis/**",
            "crates/core/symthaea-research-protocol/**",
            "crates/core/symthaea-research-replication/**",
            "crates/core/symthaea-research-result/**",
        ],
        "required_cross_cutting_rules": [
            "Cargo.lock changes require exact regenerated workspace lock",
            "workflow/toolchain/dependency changes create a new qualification subject",
        ],
        "fallback_policy": "FULL_CI_REQUIRED_ON_UNKNOWN_OR_AMBIGUOUS",
        "non_claims": [
            "does not establish scientific validity",
            "does not replace full repository integration qualification",
        ],
    }
    normalized = qualification_profile.normalize_profile(raw, verify_declared_id=False)
    raw["profile_id"] = normalized["profile_id"]
    return raw


def request(profile_value=None):
    profile_value = profile_value or profile()
    raw = {
        "schema": admission_v2.SCHEMA,
        "program_id": "research-integrity.v2",
        "catalog_id": ident("b"),
        "catalog_branch": "evidence/catalog-v2",
        "catalog_path": "docs/catalog.v2.json",
        "target_train": {
            "name": "RI-1",
            "train_id": ident("c"),
            "subject_sha": sha("d"),
        },
        "qualification_profile": {
            "name": profile_value["profile_name"],
            "profile_id": profile_value["profile_id"],
            "profile_branch": "evidence/qualification-profile-v1",
            "profile_path": "docs/qualification-profiles/research-integrity-focused.v1.json",
        },
        "reason": "qualify the exact research-integrity software-contract subject",
        "evidence_refs": ["issue:1940", "issue:986"],
        "non_claims": [
            "does not establish scientific validity",
            "does not authorize merge",
        ],
    }
    normalized = admission_v2.normalize_request(raw, verify_declared_id=False)
    raw["admission_id"] = normalized["admission_id"]
    return raw


def test_same_profile_id_different_locator_same_semantic_subject():
    first = request()
    second = request()
    second["qualification_profile"]["profile_branch"] = "mirror/qualification-profile-v1"
    second["qualification_profile"]["profile_path"] = "other/profile.json"
    second["admission_id"] = admission_v2.normalize_request(
        second, verify_declared_id=False
    )["admission_id"]

    assert first["admission_id"] != second["admission_id"]
    assert admission_v2.compute_admission_subject_id(first) == admission_v2.compute_admission_subject_id(second)


def test_same_profile_name_changed_profile_bytes_changes_semantic_subject():
    first_profile = profile(recipe="git-blob-sha1:" + "a" * 40)
    second_profile = profile(recipe="git-blob-sha1:" + "b" * 40)
    assert first_profile["profile_name"] == second_profile["profile_name"]
    assert first_profile["profile_id"] != second_profile["profile_id"]

    assert admission_v2.compute_admission_subject_id(request(first_profile)) != admission_v2.compute_admission_subject_id(request(second_profile))


def test_declared_profile_id_must_match_referenced_profile_bytes():
    referenced = profile(recipe="git-blob-sha1:" + "a" * 40)
    request_value = request(referenced)
    other = profile(recipe="git-blob-sha1:" + "b" * 40)
    with pytest.raises(integration_train.TrainManifestError, match="does not match referenced profile bytes"):
        admission_v2.validate_profile_record(request_value, other)


def test_profile_name_must_match_referenced_profile():
    referenced = profile()
    request_value = request(referenced)
    other = dict(referenced)
    other["profile_name"] = "different.profile.name"
    other["profile_id"] = qualification_profile.normalize_profile(
        other, verify_declared_id=False
    )["profile_id"]
    with pytest.raises(integration_train.TrainManifestError, match="name: does not match"):
        admission_v2.validate_profile_record(request_value, other)


def test_v1_and_v2_schemas_are_not_silently_reinterpreted():
    v2 = request()
    assert v2["schema"] == admission_v2.SCHEMA
    with pytest.raises(integration_train.TrainManifestError, match="expected"):
        qualification_admission.normalize_request(v2)


def test_declared_admission_id_mismatch_fails_closed():
    raw = request()
    raw["admission_id"] = ident("f")
    with pytest.raises(integration_train.TrainManifestError, match="expected sha256"):
        admission_v2.normalize_request(raw, require_id=True)
