import importlib.util
import json
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
MANIFEST_SCRIPT = SCRIPTS / "integration_train_manifest.py"
CATALOG_SCRIPT = SCRIPTS / "integration_train_catalog.py"
ADMISSION_SCRIPT = SCRIPTS / "qualification_admission.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


integration_train = load_module("integration_train_manifest", MANIFEST_SCRIPT)

import sys
sys.modules["integration_train_manifest"] = integration_train
integration_catalog = load_module("integration_train_catalog", CATALOG_SCRIPT)
sys.modules["integration_train_catalog"] = integration_catalog
admission = load_module("qualification_admission", ADMISSION_SCRIPT)


def sha(char):
    return char * 40


def ident(char):
    return "sha256:" + char * 64


def valid_catalog():
    raw = {
        "schema": integration_catalog.SCHEMA,
        "program_id": "resources.planning-research.v1",
        "trains": [
            {
                "name": "R3",
                "train_id": ident("a"),
                "manifest_branch": "evidence/r3",
                "manifest_path": "docs/r3.json",
                "base_subject": sha("1"),
                "cumulative_tip_sha": sha("2"),
                "role": "qualification boundary",
                "status": "SourceManifestOnly",
            }
        ],
        "edges": [],
        "non_claims": ["does not establish hosted qualification"],
    }
    normalized = integration_catalog.normalize_catalog(raw, verify_declared_id=False)
    raw["catalog_id"] = normalized["catalog_id"]
    return raw


def valid_request():
    cat = valid_catalog()
    return {
        "schema": admission.SCHEMA,
        "program_id": cat["program_id"],
        "catalog_id": cat["catalog_id"],
        "catalog_branch": "evidence/catalog-v1",
        "catalog_path": "docs/catalog.v1.json",
        "target_train": {
            "name": "R3",
            "train_id": cat["trains"][0]["train_id"],
            "subject_sha": cat["trains"][0]["cumulative_tip_sha"],
        },
        "qualification_profile": "resources.stack-qualification.v1",
        "reason": "promote the cumulative resource stack to heavyweight qualification",
        "evidence_refs": ["issue:986", "pr:833"],
        "non_claims": [
            "does not authorize merge",
            "does not establish hosted qualification",
            "does not state a qualification outcome",
        ],
    }


def test_admission_identity_is_deterministic_and_semantic():
    request = valid_request()
    first = admission.compute_admission_id(request)
    reordered = json.loads(json.dumps(request, sort_keys=True))
    assert admission.compute_admission_id(reordered) == first

    changed = valid_request()
    changed["qualification_profile"] = "resources.other.v1"
    assert admission.compute_admission_id(changed) != first


def test_declared_id_must_match():
    request = valid_request()
    request["admission_id"] = ident("0")
    with pytest.raises(integration_train.TrainManifestError, match="admission_id"):
        admission.normalize_request(request)


def test_catalog_record_binding_is_exact():
    request = valid_request()
    cat = valid_catalog()
    admission.validate_catalog_record(request, cat)

    wrong = valid_request()
    wrong["target_train"]["train_id"] = ident("b")
    with pytest.raises(integration_train.TrainManifestError, match="train_id"):
        admission.validate_catalog_record(wrong, cat)


def test_subject_must_equal_catalog_train_tip():
    request = valid_request()
    cat = valid_catalog()
    request["target_train"]["subject_sha"] = sha("3")
    with pytest.raises(integration_train.TrainManifestError, match="subject_sha"):
        admission.validate_catalog_record(request, cat)


def test_set_like_fields_and_locations_are_canonical():
    request = valid_request()
    request["evidence_refs"] = ["pr:833", "issue:986"]
    with pytest.raises(integration_train.TrainManifestError, match="lexicographically sorted"):
        admission.normalize_request(request)

    request = valid_request()
    request["catalog_path"] = "../catalog.json"
    with pytest.raises(integration_train.TrainManifestError, match="non-canonical"):
        admission.normalize_request(request)


def test_duplicate_json_keys_fail_closed(tmp_path):
    path = tmp_path / "admission.json"
    path.write_text('{"schema":"a","schema":"b"}')
    with pytest.raises(integration_train.TrainManifestError, match="duplicate JSON object key"):
        admission.load_request(path)
