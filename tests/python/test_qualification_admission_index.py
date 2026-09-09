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
admission_index = load_module("qualification_admission_index", SCRIPTS / "qualification_admission_index.py")


def sha(char):
    return char * 40


def ident(char):
    return "sha256:" + char * 64


def request(reason="first rationale", evidence=None, profile="resources.stack.v1"):
    raw = {
        "schema": qualification_admission.SCHEMA,
        "program_id": "resources.planning-research.v1",
        "catalog_id": ident("a"),
        "catalog_branch": "evidence/catalog-v1",
        "catalog_path": "docs/catalog.v1.json",
        "target_train": {
            "name": "R3",
            "train_id": ident("b"),
            "subject_sha": sha("c"),
        },
        "qualification_profile": profile,
        "reason": reason,
        "evidence_refs": evidence or ["issue:986"],
        "non_claims": [
            "does not establish hosted qualification",
            "does not state a qualification outcome",
        ],
    }
    normalized = qualification_admission.normalize_request(raw, verify_declared_id=False)
    raw["admission_id"] = normalized["admission_id"]
    return raw


def test_same_work_subject_coalesces_distinct_request_provenance():
    first = request("first rationale", ["issue:986"])
    second = request("second rationale", ["issue:987"])
    assert first["admission_id"] != second["admission_id"]
    assert admission_index.compute_admission_subject_id(first) == admission_index.compute_admission_subject_id(second)

    index = admission_index.build_index([second, first])
    assert len(index["subjects"]) == 1
    assert index["subjects"][0]["request_ids"] == sorted([first["admission_id"], second["admission_id"]])


def test_profile_change_creates_distinct_subject():
    first = request(profile="resources.stack.v1")
    second = request(profile="resources.stack.v2")
    assert admission_index.compute_admission_subject_id(first) != admission_index.compute_admission_subject_id(second)
    assert len(admission_index.build_index([first, second])["subjects"]) == 2


def test_duplicate_request_identity_fails_closed():
    first = request()
    with pytest.raises(integration_train.TrainManifestError, match="duplicate admission_id"):
        admission_index.build_index([first, first])


def test_index_identity_is_order_independent_for_request_set():
    first = request("first rationale", ["issue:986"])
    second = request("second rationale", ["issue:987"])
    left = admission_index.build_index([first, second])
    right = admission_index.build_index([second, first])
    assert left["index_id"] == right["index_id"]
    assert left == right


def test_subject_identity_ignores_locator_and_rationale_but_not_catalog_content():
    first = request()
    second = request("different reason", ["issue:999"])
    second["catalog_branch"] = "mirror/catalog-v1"
    second["catalog_path"] = "other/catalog.json"
    second["admission_id"] = qualification_admission.normalize_request(second, verify_declared_id=False)["admission_id"]
    assert admission_index.compute_admission_subject_id(first) == admission_index.compute_admission_subject_id(second)

    third = request()
    third["catalog_id"] = ident("d")
    third["admission_id"] = qualification_admission.normalize_request(third, verify_declared_id=False)["admission_id"]
    assert admission_index.compute_admission_subject_id(first) != admission_index.compute_admission_subject_id(third)
