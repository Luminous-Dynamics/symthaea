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
subject = load("qualification_subject")
closure = load("qualification_input_closure_v1")
selection = load("qualification_environment_selection_v1")
selection_verifier = load("qualification_environment_selection_verifier_v1")
resolution = load("qualification_environment_resolution_v1")

COMMIT = "a" * 40
TREE = "b" * 40


def subject_value():
    raw = {
        "schema": subject.SCHEMA,
        "kind": "GitCommit",
        "repository": "Luminous-Dynamics/symthaea",
        "object_format": "sha1",
        "source_commit": COMMIT,
        "source_tree": TREE,
    }
    raw["subject_id"] = subject.compute_subject_id(raw)
    return raw


def closure_value():
    return closure.build_from_subject(subject_value())


def selection_value():
    raw = {
        "schema": selection.SCHEMA,
        "kind": selection.KIND,
        "selection_strength": selection.SELECTION_STRENGTH,
        "input_closure_id": closure_value()["input_closure_id"],
        "object_format": "sha1",
        "target_system": "x86_64-linux",
        "flake_output": "devShells.x86_64-linux.resource-qualification",
        "selector_blobs": [
            {"path": "flake.lock", "git_blob": "1" * 40},
            {"path": "flake.nix", "git_blob": "2" * 40},
            {"path": "rust-toolchain.toml", "git_blob": "3" * 40},
        ],
        "non_claims": list(selection.NON_CLAIMS),
    }
    return selection.normalize_selection(raw, verify_declared_id=False)


def install_selection_witness(monkeypatch):
    def fake_verify(sel, subj, clos, repo):
        assert sel["environment_selection_id"] == selection_value()["environment_selection_id"]
        assert subj["source_commit"] == COMMIT
        assert clos["input_closure_id"] == closure_value()["input_closure_id"]
        return {
            "state": selection_verifier.POSITIVE_STATE,
            "environment_selection_id": sel["environment_selection_id"],
        }

    monkeypatch.setattr(selection_verifier, "verify_selection_git_binding", fake_verify)


def install_nix(monkeypatch, *, drv="/nix/store/" + "d" * 32 + "-env.drv", output="/nix/store/" + "e" * 32 + "-env", version="nix (Nix) 2.31.0"):
    calls = []

    def fake_run(repo, argv):
        calls.append(argv)
        if argv[:3] == ["nix", "eval", "--raw"]:
            assert f"rev={COMMIT}" in argv[3]
            assert argv[3].endswith("#devShells.x86_64-linux.resource-qualification.drvPath")
            return drv
        if argv[:4] == ["nix", "build", "--no-link", "--print-out-paths"]:
            assert f"rev={COMMIT}" in argv[4]
            assert argv[4].endswith("#devShells.x86_64-linux.resource-qualification")
            return output
        if argv == ["nix", "--version"]:
            return version
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(resolution, "_run", fake_run)
    return calls


def test_resolution_uses_exact_git_commit_and_nix_outputs(monkeypatch, tmp_path):
    install_selection_witness(monkeypatch)
    calls = install_nix(monkeypatch)
    result = resolution.resolve_environment_selection(
        selection_value(), subject_value(), closure_value(), tmp_path
    )
    assert result["state"] == resolution.POSITIVE_STATE
    assert result["derivation_path"].endswith("-env.drv")
    assert result["output_paths"][0].endswith("-env")
    assert len(calls) == 3


def test_empty_nix_output_fails_closed(monkeypatch, tmp_path):
    install_selection_witness(monkeypatch)
    install_nix(monkeypatch, output="")
    with pytest.raises(train.TrainManifestError, match="no realized output"):
        resolution.resolve_environment_selection(
            selection_value(), subject_value(), closure_value(), tmp_path
        )


def test_malformed_store_path_fails_closed(monkeypatch, tmp_path):
    install_selection_witness(monkeypatch)
    install_nix(monkeypatch, drv="/tmp/not-a-nix-derivation")
    with pytest.raises(train.TrainManifestError, match="/nix/store"):
        resolution.resolve_environment_selection(
            selection_value(), subject_value(), closure_value(), tmp_path
        )


def test_nix_version_is_resolution_identity_bearing(monkeypatch, tmp_path):
    install_selection_witness(monkeypatch)
    install_nix(monkeypatch, version="nix (Nix) 2.31.0")
    first = resolution.resolve_environment_selection(
        selection_value(), subject_value(), closure_value(), tmp_path
    )
    install_nix(monkeypatch, version="nix (Nix) 2.32.0")
    second = resolution.resolve_environment_selection(
        selection_value(), subject_value(), closure_value(), tmp_path
    )
    assert first["environment_resolution_id"] != second["environment_resolution_id"]


def test_flake_output_is_resolution_identity_bearing(monkeypatch, tmp_path):
    install_selection_witness(monkeypatch)
    install_nix(monkeypatch)
    first = resolution.resolve_environment_selection(
        selection_value(), subject_value(), closure_value(), tmp_path
    )

    changed = selection_value()
    changed = dict(changed)
    changed["flake_output"] = "devShells.x86_64-linux.other"
    changed.pop("environment_selection_id")
    changed = selection.normalize_selection(changed, verify_declared_id=False)

    def fake_verify(sel, subj, clos, repo):
        return {
            "state": selection_verifier.POSITIVE_STATE,
            "environment_selection_id": sel["environment_selection_id"],
        }
    monkeypatch.setattr(selection_verifier, "verify_selection_git_binding", fake_verify)

    def fake_run(repo, argv):
        if argv[:3] == ["nix", "eval", "--raw"]:
            return "/nix/store/" + "f" * 32 + "-other.drv"
        if argv[:4] == ["nix", "build", "--no-link", "--print-out-paths"]:
            return "/nix/store/" + "0" * 32 + "-other"
        return "nix (Nix) 2.31.0"
    monkeypatch.setattr(resolution, "_run", fake_run)

    second = resolution.resolve_environment_selection(
        changed, subject_value(), closure_value(), tmp_path
    )
    assert first["environment_resolution_id"] != second["environment_resolution_id"]
