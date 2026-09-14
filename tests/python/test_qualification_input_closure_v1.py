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
subject = load("qualification_subject")
framing = load("qualification_framing_v1")
closure = load("qualification_input_closure_v1")


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


def test_whole_tree_closure_golden_identity():
    value = closure.build_from_subject(v1_subject())
    assert value["input_closure_id"] == (
        "sha256:abb53ba87c73aa475ddd5c9db11144beddefda3897a1a71cd9aba6a24d9591c6"
    )
    assert closure.normalize_closure(value, require_id=True) == value


def test_same_tree_different_commit_preserves_input_closure():
    first = closure.build_from_subject(v1_subject(commit="a", tree="b"))
    second = closure.build_from_subject(v1_subject(commit="9", tree="b"))
    assert first["input_closure_id"] == second["input_closure_id"]
    assert v1_subject(commit="a", tree="b")["subject_id"] != v1_subject(
        commit="9", tree="b"
    )["subject_id"]


def test_changed_tree_changes_input_closure():
    first = closure.build_from_subject(v1_subject(tree="b"))
    second = closure.build_from_subject(v1_subject(tree="8"))
    assert first["input_closure_id"] != second["input_closure_id"]


def test_closure_must_match_exact_subject_tree():
    value = closure.build_from_subject(v1_subject(tree="b"))
    with pytest.raises(train.TrainManifestError, match="source tree does not match"):
        closure.validate_against_subject(value, v1_subject(tree="8"))


def test_tampered_declared_closure_id_fails_closed():
    value = closure.build_from_subject(v1_subject())
    value["input_closure_id"] = "sha256:" + "0" * 64
    with pytest.raises(train.TrainManifestError, match="closure semantics changed"):
        closure.normalize_closure(value, require_id=True)


def test_nonclaims_are_frozen_theorem_boundary():
    value = closure.build_from_subject(v1_subject())
    value["non_claims"] = value["non_claims"][:-1]
    with pytest.raises(train.TrainManifestError, match="frozen set"):
        closure.normalize_closure(value)
