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
verifier = load("qualification_environment_selection_verifier_v1")

COMMIT = "a" * 40
TREE = "b" * 40
BLOBS = {
    "flake.lock": "1" * 40,
    "flake.nix": "2" * 40,
    "rust-toolchain.toml": "3" * 40,
}


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


def selection_value(blobs=None):
    blobs = BLOBS if blobs is None else blobs
    raw = {
        "schema": selection.SCHEMA,
        "kind": selection.KIND,
        "selection_strength": selection.SELECTION_STRENGTH,
        "input_closure_id": closure_value()["input_closure_id"],
        "object_format": "sha1",
        "target_system": "x86_64-linux",
        "flake_output": "devShells.x86_64-linux.resource-qualification",
        "selector_blobs": [
            {"path": path, "git_blob": blobs[path]}
            for path in selection.SELECTOR_PATHS
        ],
        "non_claims": list(selection.NON_CLAIMS),
    }
    return selection.normalize_selection(raw, verify_declared_id=False)


def install_fake_git(monkeypatch, actual_blobs=None, tree=TREE):
    actual_blobs = BLOBS if actual_blobs is None else actual_blobs

    def fake_run_git(repo, *args):
        if args == ("rev-parse", "--is-inside-work-tree"):
            return "true"
        if args == ("rev-parse", "--show-object-format"):
            return "sha1"
        if args == ("cat-file", "-e", f"{COMMIT}^{{commit}}"):
            return ""
        if args == ("rev-parse", f"{COMMIT}^{{tree}}"):
            return tree
        for path, blob in actual_blobs.items():
            if args == ("rev-parse", f"{COMMIT}:{path}"):
                return blob
            if args == ("cat-file", "-e", f"{blob}^{{blob}}"):
                return ""
        raise AssertionError(f"unexpected git call: {args}")

    monkeypatch.setattr(subject, "_run_git", fake_run_git)


def test_exact_selection_is_git_bound(monkeypatch, tmp_path):
    install_fake_git(monkeypatch)
    witness = verifier.verify_selection_git_binding(
        selection_value(), subject_value(), closure_value(), tmp_path
    )
    assert witness["state"] == verifier.POSITIVE_STATE
    assert witness["environment_selection_id"] == selection_value()["environment_selection_id"]
    assert witness["verified_selector_blobs"] == [
        {"path": path, "git_blob": BLOBS[path]} for path in selection.SELECTOR_PATHS
    ]


def test_internally_valid_but_forged_selector_blob_fails(monkeypatch, tmp_path):
    forged = dict(BLOBS)
    forged["flake.lock"] = "9" * 40
    # The forged object is internally valid and receives its own selection identity.
    candidate = selection_value(forged)
    selection.normalize_selection(candidate, require_id=True)

    install_fake_git(monkeypatch, actual_blobs=BLOBS)
    with pytest.raises(train.TrainManifestError, match="flake.lock blob mismatch"):
        verifier.verify_selection_git_binding(
            candidate, subject_value(), closure_value(), tmp_path
        )


def test_subject_tree_mismatch_fails_before_selector_authority(monkeypatch, tmp_path):
    install_fake_git(monkeypatch, tree="c" * 40)
    with pytest.raises(train.TrainManifestError, match="source_tree"):
        verifier.verify_selection_git_binding(
            selection_value(), subject_value(), closure_value(), tmp_path
        )


def test_selection_for_different_closure_fails(monkeypatch, tmp_path):
    install_fake_git(monkeypatch)
    candidate = selection_value()
    candidate = dict(candidate)
    candidate["input_closure_id"] = "sha256:" + "f" * 64
    candidate.pop("environment_selection_id")
    candidate = selection.normalize_selection(candidate, verify_declared_id=False)
    with pytest.raises(train.TrainManifestError, match="input closure"):
        verifier.verify_selection_git_binding(
            candidate, subject_value(), closure_value(), tmp_path
        )
