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
profile = load("qualification_profile")
subject = load("qualification_subject")
framing = load("qualification_framing_v1")
semantic = load("qualification_semantic_ids_v2")


def v1_subject(commit="a", tree="b", repository="Luminous-Dynamics/symthaea"):
    raw = {
        "schema": subject.SCHEMA,
        "kind": "GitCommit",
        "repository": repository,
        "object_format": "sha1",
        "source_commit": commit * 40,
        "source_tree": tree * 40,
    }
    normalized = subject.normalize_subject(raw, verify_declared_id=False)
    raw["subject_id"] = normalized["subject_id"]
    return raw


def v1_profile():
    raw = {
        "schema": profile.SCHEMA,
        "profile_name": "research.integrity.focused.v1",
        "required_recipe_ids": [
            "git-blob-sha1:" + "1" * 40,
            "sha256:" + "2" * 64,
        ],
        "owned_surface_rules": ["rules/a", "rules/b"],
        "required_cross_cutting_rules": ["governance", "source-integrity"],
        "fallback_policy": "FULL_CI_REQUIRED_ON_UNKNOWN_OR_AMBIGUOUS",
        "non_claims": ["does not authorize merge"],
    }
    normalized = profile.normalize_profile(raw, verify_declared_id=False)
    raw["profile_id"] = normalized["profile_id"]
    return raw


def test_subject_v2_golden_identity_and_frame():
    value = v1_subject()
    assert semantic.compute_subject_id_v2(value) == (
        "sha256:a9df781b45453a8e147478d4341b1f2fbb92abf347d34a4d2dcbd460a5cd871f"
    )
    assert semantic.subject_frame_v2(value).hex() == (
        "53594d5146524d310001002473796d74686165612e7175616c696669636174696f6e2d7375626a6563742d69642e7632"
        "00000005"
        "00046b696e640000000000000012450000000000000009476974436f6d6d6974"
        "000a7265706f7369746f7279000000000000002354000000000000001a4c756d696e6f75732d44796e616d6963732f73796d7468616561"
        "000d6f626a6563745f666f726d6174000000000000000d45000000000000000473686131"
        "000d736f757263655f636f6d6d6974000000000000003154000000000000002861616161616161616161616161616161616161616161616161616161616161616161616161616161"
        "000b736f757263655f7472656500000000000000315400000000000000286262626262626262626262626262626262626262626262626262626262626262626262626262626262"
    )


def test_profile_v2_golden_identity():
    assert semantic.compute_profile_id_v2(v1_profile()) == (
        "sha256:6e0cbabf84cf2f1c0e297bc4fce214007a83bb51443956fb4148c89e2755d15e"
    )


def test_v2_does_not_reinterpret_v1_id_namespace():
    subject_value = v1_subject()
    profile_value = v1_profile()
    assert semantic.compute_subject_id_v2(subject_value) != subject_value["subject_id"]
    assert semantic.compute_profile_id_v2(profile_value) != profile_value["profile_id"]


def test_subject_semantic_changes_change_v2_identity():
    baseline = semantic.compute_subject_id_v2(v1_subject())
    assert semantic.compute_subject_id_v2(v1_subject(commit="9")) != baseline
    assert semantic.compute_subject_id_v2(v1_subject(tree="8")) != baseline
    # Repository remains a declared namespace coordinate, not authenticated origin proof.
    assert semantic.compute_subject_id_v2(
        v1_subject(repository="Luminous-Dynamics/other")
    ) != baseline


def test_bad_declared_v1_subject_id_cannot_be_laundered_into_v2():
    value = v1_subject()
    value["subject_id"] = "sha256:" + "0" * 64
    with pytest.raises(train.TrainManifestError, match="subject_id"):
        semantic.compute_subject_id_v2(value)


def test_bad_declared_v1_profile_id_cannot_be_laundered_into_v2():
    value = v1_profile()
    value["profile_id"] = "sha256:" + "0" * 64
    with pytest.raises(train.TrainManifestError, match="profile_id"):
        semantic.compute_profile_id_v2(value)


def test_profile_semantic_changes_change_v2_identity():
    baseline = semantic.compute_profile_id_v2(v1_profile())

    changed = v1_profile()
    changed.pop("profile_id")
    changed["non_claims"] = ["does not authorize deployment"]
    assert semantic.compute_profile_id_v2(changed) != baseline

    changed = v1_profile()
    changed.pop("profile_id")
    changed["required_recipe_ids"] = ["git-blob-sha1:" + "1" * 40]
    assert semantic.compute_profile_id_v2(changed) != baseline
