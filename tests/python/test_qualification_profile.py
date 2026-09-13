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
qualification_profile = load_module("qualification_profile", SCRIPTS / "qualification_profile.py")


def profile(name="research.integrity.focused.v1", recipe="git-blob-sha1:" + "a" * 40):
    return {
        "schema": qualification_profile.SCHEMA,
        "profile_name": name,
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


def normalized(raw):
    result = qualification_profile.normalize_profile(raw, verify_declared_id=False)
    raw = dict(raw)
    raw["profile_id"] = result["profile_id"]
    return qualification_profile.normalize_profile(raw, require_id=True)


def test_same_name_changed_recipe_changes_profile_identity():
    first = normalized(profile(recipe="git-blob-sha1:" + "a" * 40))
    second = normalized(profile(recipe="git-blob-sha1:" + "b" * 40))
    assert first["profile_name"] == second["profile_name"]
    assert first["profile_id"] != second["profile_id"]


def test_rule_change_changes_profile_identity():
    first = normalized(profile())
    changed = profile()
    changed["required_cross_cutting_rules"] = sorted(
        changed["required_cross_cutting_rules"]
        + ["unknown routing state requires full CI"]
    )
    second = normalized(changed)
    assert first["profile_id"] != second["profile_id"]


def test_declared_profile_id_mismatch_fails_closed():
    raw = profile()
    raw["profile_id"] = "sha256:" + "f" * 64
    with pytest.raises(integration_train.TrainManifestError, match="expected sha256"):
        qualification_profile.normalize_profile(raw, require_id=True)


def test_unsorted_or_duplicate_semantics_rejected():
    raw = profile()
    raw["owned_surface_rules"] = list(reversed(raw["owned_surface_rules"]))
    with pytest.raises(integration_train.TrainManifestError, match="lexicographically sorted"):
        qualification_profile.normalize_profile(raw)

    raw = profile()
    raw["required_recipe_ids"] = [
        "git-blob-sha1:" + "a" * 40,
        "git-blob-sha1:" + "a" * 40,
    ]
    with pytest.raises(integration_train.TrainManifestError, match="unique"):
        qualification_profile.normalize_profile(raw)


def test_unknown_fallback_policy_rejected():
    raw = profile()
    raw["fallback_policy"] = "SKIP_FULL_CI_ON_UNKNOWN"
    with pytest.raises(integration_train.TrainManifestError, match="unsupported fail-closed policy"):
        qualification_profile.normalize_profile(raw)


def test_locator_is_not_part_of_profile_semantics():
    # The manifest intentionally contains no branch/path fields. A caller may locate the same
    # bytes in multiple Git refs without changing the semantic profile identity.
    first = normalized(profile())
    second = normalized(profile())
    assert first["profile_id"] == second["profile_id"]
