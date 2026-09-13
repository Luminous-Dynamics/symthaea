import importlib.util
import json
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
profile = load("qualification_profile")
subject = load("qualification_subject")
admission_v3 = load("qualification_admission_v3")
framing = load("qualification_framing_v1")
recipe = load("qualification_recipe_v1")
semantic_v2 = load("qualification_semantic_ids_v2")
admission_v2 = load("qualification_admission_ids_v2")


def v1_subject(commit="a", tree="b"):
    raw = {
        "schema": subject.SCHEMA,
        "kind": "GitCommit",
        "repository": "Luminous-Dynamics/symthaea",
        "object_format": "sha1",
        "source_commit": commit * 40,
        "source_tree": tree * 40,
    }
    normalized = subject.normalize_subject(raw, verify_declared_id=False)
    raw["subject_id"] = normalized["subject_id"]
    return raw


def v1_profile(non_claim="does not authorize merge"):
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
        "non_claims": [non_claim],
    }
    normalized = profile.normalize_profile(raw, verify_declared_id=False)
    raw["profile_id"] = normalized["profile_id"]
    return raw


def recipe_preimages():
    values = [
        {
            "schema": recipe.SCHEMA,
            "recipe_name": "research.format.v1",
            "legacy_recipe_ref": "git-blob-sha1:" + "1" * 40,
            "execution_model": recipe.EXECUTION_MODEL,
            "environment_contract": recipe.ENVIRONMENT_CONTRACT,
            "steps": [
                {
                    "step_name": "rustfmt",
                    "working_directory": ".",
                    "argv": ["cargo", "fmt", "--all", "--", "--check"],
                    "environment_overrides": [],
                    "stdin_policy": recipe.STDIN_POLICY,
                    "acceptable_exit_codes": [0],
                }
            ],
            "non_claims": ["does not prove semantic correctness"],
        },
        {
            "schema": recipe.SCHEMA,
            "recipe_name": "research.test.v1",
            "legacy_recipe_ref": "sha256:" + "2" * 64,
            "execution_model": recipe.EXECUTION_MODEL,
            "environment_contract": recipe.ENVIRONMENT_CONTRACT,
            "steps": [
                {
                    "step_name": "tests",
                    "working_directory": ".",
                    "argv": ["cargo", "test", "--locked", "-p", "symthaea-research-protocol"],
                    "environment_overrides": [
                        {"name": "CARGO_TERM_COLOR", "value": "never"}
                    ],
                    "stdin_policy": recipe.STDIN_POLICY,
                    "acceptable_exit_codes": [0],
                }
            ],
            "non_claims": ["does not prove scientific validity"],
        },
    ]
    for value in values:
        value["recipe_id"] = recipe.compute_recipe_id(value)
    return values


def v3_admission(
    profile_value=None,
    *,
    branch="evidence/profile-v1",
    path="docs/profile.json",
    reason="qualify",
    evidence_refs=None,
):
    resolved = profile_value or v1_profile()
    raw = {
        "schema": admission_v3.SCHEMA,
        "subject": v1_subject(),
        "qualification_profile": {
            "name": resolved["profile_name"],
            "profile_id": resolved["profile_id"],
            "profile_branch": branch,
            "profile_path": path,
        },
        "reason": reason,
        "evidence_refs": evidence_refs or ["issue:986"],
        "non_claims": ["does not authorize merge"],
    }
    normalized = admission_v3.normalize_request(raw, verify_declared_ids=False)
    raw["admission_id"] = normalized["admission_id"]
    raw["admission_subject_id"] = normalized["admission_subject_id"]
    return raw


def test_admission_v2_golden_identities():
    p = v1_profile()
    recipes = recipe_preimages()
    result = admission_v2.derive_admission_identities_v2(v3_admission(p), p, recipes)
    assert result == {
        "schema": admission_v2.SCHEMA,
        "qualification_subject_id_v2": "sha256:a9df781b45453a8e147478d4341b1f2fbb92abf347d34a4d2dcbd460a5cd871f",
        "qualification_profile_id_v2": "sha256:6d712ce2517000d4e1d76b6b11d20349a59efcd10eb2e0fa7e8d599b8fdd1572",
        "admission_subject_id_v2": "sha256:23f1dd6b4fc1aee8cef47953bf0512a7600b614b904025422209f8195370de49",
        "admission_request_id_v2": "sha256:d353eeef51767f741e0bd95ec98a84a4a1d515d2b0e9a3d473770b327e1157ca",
    }


def test_request_provenance_changes_but_v2_work_identity_does_not():
    p = v1_profile()
    recipes = recipe_preimages()
    first = admission_v2.derive_admission_identities_v2(v3_admission(p), p, recipes)
    second = admission_v2.derive_admission_identities_v2(
        v3_admission(
            p,
            branch="evidence/other-v1",
            path="docs/other-profile.json",
            reason="same theorem, different request provenance",
            evidence_refs=["issue:2318"],
        ),
        p,
        recipes,
    )
    assert first["admission_subject_id_v2"] == second["admission_subject_id_v2"]
    assert first["admission_request_id_v2"] != second["admission_request_id_v2"]


def test_profile_preimage_mismatch_fails_closed():
    p = v1_profile()
    admission = v3_admission(p)
    different = v1_profile(non_claim="does not authorize deployment")
    with pytest.raises(train.TrainManifestError, match="profile bytes"):
        admission_v2.derive_admission_identities_v2(
            admission, different, recipe_preimages()
        )


def test_missing_recipe_preimage_fails_closed():
    p = v1_profile()
    with pytest.raises(train.TrainManifestError, match="coverage mismatch"):
        admission_v2.derive_admission_identities_v2(
            v3_admission(p), p, recipe_preimages()[:1]
        )


def test_bad_declared_v3_admission_id_cannot_be_laundered_into_v2():
    p = v1_profile()
    admission = v3_admission(p)
    admission["admission_id"] = "sha256:" + "0" * 64
    with pytest.raises(train.TrainManifestError, match="admission_id"):
        admission_v2.derive_admission_identities_v2(admission, p, recipe_preimages())


def test_changed_profile_or_recipe_semantics_change_v2_work_identity():
    baseline_profile = v1_profile()
    baseline_recipes = recipe_preimages()
    baseline = admission_v2.derive_admission_identities_v2(
        v3_admission(baseline_profile), baseline_profile, baseline_recipes
    )

    changed_profile = v1_profile(non_claim="does not authorize deployment")
    changed = admission_v2.derive_admission_identities_v2(
        v3_admission(changed_profile), changed_profile, recipe_preimages()
    )
    assert baseline["qualification_profile_id_v2"] != changed["qualification_profile_id_v2"]
    assert baseline["admission_subject_id_v2"] != changed["admission_subject_id_v2"]

    changed_recipes = recipe_preimages()
    changed_recipes[0].pop("recipe_id")
    changed_recipes[0]["steps"][0]["argv"].append("--verbose")
    changed_recipes[0]["recipe_id"] = recipe.compute_recipe_id(changed_recipes[0])
    changed_recipe_result = admission_v2.derive_admission_identities_v2(
        v3_admission(baseline_profile), baseline_profile, changed_recipes
    )
    assert baseline["qualification_profile_id_v2"] != changed_recipe_result["qualification_profile_id_v2"]
    assert baseline["admission_subject_id_v2"] != changed_recipe_result["admission_subject_id_v2"]


def test_v2_work_identity_is_not_v3_json_identity():
    p = v1_profile()
    admission = v3_admission(p)
    result = admission_v2.derive_admission_identities_v2(
        admission, p, recipe_preimages()
    )
    assert result["admission_subject_id_v2"] != admission["admission_subject_id"]
    assert result["admission_request_id_v2"] != admission["admission_id"]


def test_migration_file_loader_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "ambiguous.json"
    path.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
    with pytest.raises(train.TrainManifestError, match="duplicate JSON object key"):
        admission_v2._load_json(path)


def test_migration_file_loader_rejects_oversized_input(tmp_path, monkeypatch):
    path = tmp_path / "oversized.json"
    path.write_text(json.dumps({"padding": "x" * 128}), encoding="utf-8")
    monkeypatch.setattr(train, "MAX_MANIFEST_BYTES", 32)
    with pytest.raises(train.TrainManifestError, match="exceeds 32 bytes"):
        admission_v2._load_json(path)
