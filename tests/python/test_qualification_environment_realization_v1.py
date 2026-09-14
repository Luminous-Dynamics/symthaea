import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


train = load("integration_train_manifest")
framing = load("qualification_framing_v1")
profile_mod = load("qualification_profile")
recipe_mod = load("qualification_recipe_v1")
semantic_v2 = load("qualification_semantic_ids_v2")
requirements_mod = load("qualification_tool_requirements_v1")
resolution_mod = load("qualification_environment_resolution_v1")
selection_verifier = load("qualification_environment_selection_verifier_v1")
realization = load("qualification_environment_realization_v1")

LEGACY_RECIPE = "git-blob-sha1:" + "a" * 40
ROOT = "/nix/store/" + "a" * 32 + "-env"
TOOL_ROOT = "/nix/store/" + "b" * 32 + "-cargo"


def recipe():
    raw = {
        "schema": recipe_mod.SCHEMA,
        "recipe_name": "focused-check",
        "legacy_recipe_ref": LEGACY_RECIPE,
        "execution_model": recipe_mod.EXECUTION_MODEL,
        "environment_contract": recipe_mod.ENVIRONMENT_CONTRACT,
        "steps": [
            {
                "step_name": "check",
                "working_directory": ".",
                "argv": ["cargo", "check", "--locked"],
                "environment_overrides": [],
                "stdin_policy": recipe_mod.STDIN_POLICY,
                "acceptable_exit_codes": [0],
            }
        ],
        "non_claims": ["does not prove scientific validity"],
    }
    normalized = recipe_mod.normalize_recipe(raw, verify_declared_id=False)
    return normalized


def profile():
    raw = {
        "schema": profile_mod.SCHEMA,
        "profile_name": "test.focused.v1",
        "required_recipe_ids": [LEGACY_RECIPE],
        "owned_surface_rules": ["crates/core/test/**"],
        "required_cross_cutting_rules": ["Cargo.lock exact"],
        "fallback_policy": "FULL_CI_REQUIRED_ON_UNKNOWN_OR_AMBIGUOUS",
        "non_claims": ["does not replace full CI"],
    }
    return profile_mod.normalize_profile(raw, verify_declared_id=False)


def requirements():
    rec = recipe()
    prof = profile()
    raw = {
        "schema": requirements_mod.SCHEMA,
        "qualification_profile_id_v2": semantic_v2.compute_profile_id_v2(prof, [rec]),
        "resolved_recipe_ids": [rec["recipe_id"]],
        "tools": [
            {
                "tool_name": "cargo",
                "executable_basename": "cargo",
                "version_argv_tail": ["--version", "--verbose"],
                "required_for_recipe_ids": [rec["recipe_id"]],
            }
        ],
        "non_claims": list(requirements_mod.NON_CLAIMS),
    }
    return requirements_mod.normalize_requirements(
        raw, profile=prof, recipes=[rec], verify_declared_id=False
    )


def resolution():
    raw = {
        "schema": resolution_mod.SCHEMA,
        "state": resolution_mod.POSITIVE_STATE,
        "environment_selection_id": "sha256:" + "1" * 64,
        "qualification_subject_id": "sha256:" + "2" * 64,
        "input_closure_id": "sha256:" + "3" * 64,
        "target_system": "x86_64-linux",
        "flake_output": "devShells.x86_64-linux.resource-qualification",
        "derivation_path": "/nix/store/" + "c" * 32 + "-env.drv",
        "output_paths": [ROOT],
        "nix_version": "nix (Nix) 2.31.0",
        "non_claims": list(resolution_mod.NON_CLAIMS),
    }
    return resolution_mod.normalize_resolution(raw, verify_declared_id=False)


def write_capture_files(root: Path, *, suffix: str = ""):
    root.mkdir(parents=True, exist_ok=True)
    names = {
        "root_nar": f"root{suffix}.nar",
        "tool_nar": f"tool{suffix}.nar",
        "exe": f"cargo{suffix}.bin",
        "version": f"cargo{suffix}.version",
    }
    (root / names["root_nar"]).write_bytes(b"root nar bytes")
    (root / names["tool_nar"]).write_bytes(b"tool nar bytes")
    (root / names["exe"]).write_bytes(b"cargo executable bytes")
    (root / names["version"]).write_bytes(b"cargo 1.96.0\n")
    return names


def capture(names, *, evidence="capture:one", extra_objects=None):
    objects = [
        {"store_path": ROOT, "nar_file": names["root_nar"], "references": [TOOL_ROOT]},
        {"store_path": TOOL_ROOT, "nar_file": names["tool_nar"], "references": []},
    ]
    if extra_objects:
        objects.extend(extra_objects)
    return {
        "schema": realization.CAPTURE_SCHEMA,
        "environment_resolution_id": resolution()["environment_resolution_id"],
        "store_objects": objects,
        "tools": [
            {
                "tool_name": "cargo",
                "executable_path": TOOL_ROOT + "/bin/cargo",
                "executable_file": names["exe"],
                "version_argv": [TOOL_ROOT + "/bin/cargo", "--version", "--verbose"],
                "version_output_file": names["version"],
            }
        ],
        "evidence_refs": [evidence],
    }


def install_upstream(monkeypatch):
    monkeypatch.setattr(
        selection_verifier,
        "verify_selection_git_binding",
        lambda selection, subject, closure, repo: {
            "state": selection_verifier.POSITIVE_STATE,
            "environment_selection_id": resolution()["environment_selection_id"],
        },
    )
    monkeypatch.setattr(
        resolution_mod,
        "validate_against_context",
        lambda value, selection, subject, closure: resolution_mod.normalize_resolution(
            value, require_id=True
        ),
    )


def verify(monkeypatch, tmp_path, cap):
    install_upstream(monkeypatch)
    return realization.verify_realization_capture(
        resolution=resolution(),
        selection={},
        subject={},
        closure={},
        repo=tmp_path,
        tool_requirements=requirements(),
        profile=profile(),
        recipes=[recipe()],
        capture=cap,
        capture_root=tmp_path,
    )


def test_positive_reconstruction(monkeypatch, tmp_path):
    names = write_capture_files(tmp_path)
    witness = verify(monkeypatch, tmp_path, capture(names))
    assert witness["state"] == realization.POSITIVE_STATE
    assert witness["output_paths"] == [ROOT]
    assert len(witness["store_objects"]) == 2
    assert witness["tools"][0]["tool_name"] == "cargo"
    assert witness["qualification_environment_id"].startswith("sha256:")


def test_capture_locator_and_occurrence_refs_are_not_environment_identity(monkeypatch, tmp_path):
    first_names = write_capture_files(tmp_path, suffix="-a")
    first = verify(monkeypatch, tmp_path, capture(first_names, evidence="capture:first"))
    second_names = write_capture_files(tmp_path, suffix="-b")
    second = verify(monkeypatch, tmp_path, capture(second_names, evidence="capture:second"))
    assert first["qualification_environment_id"] == second["qualification_environment_id"]
    assert first["capture_evidence_refs"] != second["capture_evidence_refs"]


def test_missing_reference_fails_closed(monkeypatch, tmp_path):
    names = write_capture_files(tmp_path)
    cap = capture(names)
    cap["store_objects"][0]["references"] = ["/nix/store/" + "d" * 32 + "-missing"]
    with pytest.raises(train.TrainManifestError, match="absent from captured closure"):
        verify(monkeypatch, tmp_path, cap)


def test_unreachable_extra_store_object_fails_closed(monkeypatch, tmp_path):
    names = write_capture_files(tmp_path)
    extra_file = tmp_path / "extra.nar"
    extra_file.write_bytes(b"extra nar bytes")
    extra = {
        "store_path": "/nix/store/" + "d" * 32 + "-extra",
        "nar_file": "extra.nar",
        "references": [],
    }
    with pytest.raises(train.TrainManifestError, match="unreachable extra"):
        verify(monkeypatch, tmp_path, capture(names, extra_objects=[extra]))


def test_tool_argv_substitution_fails_closed(monkeypatch, tmp_path):
    names = write_capture_files(tmp_path)
    cap = capture(names)
    cap["tools"][0]["version_argv"] = [TOOL_ROOT + "/bin/cargo", "--version"]
    with pytest.raises(train.TrainManifestError, match="version argv"):
        verify(monkeypatch, tmp_path, cap)


def test_extra_or_missing_tool_fails_closed(monkeypatch, tmp_path):
    names = write_capture_files(tmp_path)
    cap = capture(names)
    cap["tools"] = []
    with pytest.raises(train.TrainManifestError, match="non-empty array"):
        verify(monkeypatch, tmp_path, cap)


def test_executable_byte_drift_changes_environment_identity(monkeypatch, tmp_path):
    names = write_capture_files(tmp_path)
    first = verify(monkeypatch, tmp_path, capture(names))
    (tmp_path / names["exe"]).write_bytes(b"different cargo executable bytes")
    second = verify(monkeypatch, tmp_path, capture(names))
    assert first["qualification_environment_id"] != second["qualification_environment_id"]


def test_version_output_drift_changes_environment_identity(monkeypatch, tmp_path):
    names = write_capture_files(tmp_path)
    first = verify(monkeypatch, tmp_path, capture(names))
    (tmp_path / names["version"]).write_bytes(b"cargo 1.97.0\n")
    second = verify(monkeypatch, tmp_path, capture(names))
    assert first["qualification_environment_id"] != second["qualification_environment_id"]


def test_empty_nar_preimage_fails(monkeypatch, tmp_path):
    names = write_capture_files(tmp_path)
    (tmp_path / names["tool_nar"]).write_bytes(b"")
    with pytest.raises(train.TrainManifestError, match="NAR preimage is empty"):
        verify(monkeypatch, tmp_path, capture(names))
