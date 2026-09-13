import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


train = load("integration_train_manifest")
framing = load("qualification_framing")
subject_v1 = load("qualification_subject")
profile_v1 = load("qualification_profile")
subject_v2 = load("qualification_subject_v2")
profile_v2 = load("qualification_profile_v2")


def subject():
    return {
        "schema": subject_v2.SCHEMA,
        "kind": "GitCommit",
        "repository": "Luminous-Dynamics/symthaea",
        "object_format": "sha1",
        "source_commit": "a" * 40,
        "source_tree": "b" * 40,
    }


def profile():
    return {
        "schema": profile_v2.SCHEMA,
        "profile_name": "research.integrity.focused.v1",
        "required_recipe_ids": ["git-blob-sha1:" + "a" * 40],
        "owned_surface_rules": sorted(
            [
                "crates/core/symthaea-research-analysis/**",
                "crates/core/symthaea-research-protocol/**",
                "crates/core/symthaea-research-replication/**",
                "crates/core/symthaea-research-result/**",
            ]
        ),
        "required_cross_cutting_rules": sorted(
            [
                "Cargo.lock changes require exact regenerated workspace lock",
                "workflow/toolchain/dependency changes create a new qualification subject",
            ]
        ),
        "fallback_policy": "FULL_CI_REQUIRED_ON_UNKNOWN_OR_AMBIGUOUS",
        "non_claims": sorted(
            [
                "does not establish scientific validity",
                "does not replace full repository integration qualification",
            ]
        ),
    }


def test_subject_v2_golden_vector():
    assert subject_v2.compute_subject_id(subject()) == (
        "qsubject-v2-sha256:65d71abf25d9af3f5c043c90f251c06213eb6c0ea0ce29157e395132c316bc60"
    )


def test_profile_v2_golden_vector():
    assert profile_v2.compute_profile_id(profile()) == (
        "qprofile-v2-sha256:07a1095a4cb9bf60220a6702983fd330a20bc558870646593039f8d1bc7a0fed"
    )


def test_v1_ids_cannot_be_substituted_for_v2_ids():
    raw = subject()
    v1_raw = dict(raw)
    v1_raw["schema"] = subject_v1.SCHEMA
    v1_id = subject_v1.compute_subject_id(v1_raw)
    raw["subject_id"] = v1_id
    with pytest.raises(train.TrainManifestError, match="qsubject-v2-sha256"):
        subject_v2.normalize_subject(raw, require_id=True)

    raw_profile = profile()
    v1_profile = dict(raw_profile)
    v1_profile["schema"] = profile_v1.SCHEMA
    v1_profile_id = profile_v1.compute_profile_id(v1_profile)
    raw_profile["profile_id"] = v1_profile_id
    with pytest.raises(train.TrainManifestError, match="qprofile-v2-sha256"):
        profile_v2.normalize_profile(raw_profile, require_id=True)


def test_framing_is_not_concatenation_ambiguous():
    left = framing.record("demo", [("items", framing.text_list(["ab", "c"]))])
    right = framing.record("demo", [("items", framing.text_list(["a", "bc"]))])
    assert left != right
    assert framing.sha256_hex_id("demo", left) != framing.sha256_hex_id("demo", right)


def test_field_order_is_explicit_and_identity_bearing():
    first = framing.record("demo", [("a", framing.text("1")), ("b", framing.text("2"))])
    second = framing.record("demo", [("b", framing.text("2")), ("a", framing.text("1"))])
    assert first != second


def test_duplicate_field_names_are_rejected():
    with pytest.raises(framing.FramingError, match="unique"):
        framing.record("demo", [("a", b"1"), ("a", b"2")])


def test_v2_semantic_change_changes_identity():
    first = subject_v2.compute_subject_id(subject())
    changed = subject()
    changed["source_tree"] = "c" * 40
    assert first != subject_v2.compute_subject_id(changed)

    first_profile = profile_v2.compute_profile_id(profile())
    changed_profile = profile()
    changed_profile["fallback_policy"] = "REFUSE_QUALIFICATION_ON_UNKNOWN_OR_AMBIGUOUS"
    assert first_profile != profile_v2.compute_profile_id(changed_profile)
