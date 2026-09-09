import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


train = load_module("integration_train_manifest", SCRIPTS / "integration_train_manifest.py")
sys.modules["integration_train_manifest"] = train
attempts = load_module("qualification_attempt_ledger", SCRIPTS / "qualification_attempt_ledger.py")


def ident(char):
    return "sha256:" + char * 64


def sha(char):
    return char * 40


def record(subject="a", disposition="FormattingFailed", checkout_kind="synthetic_merge_candidate"):
    theorem_results = [
        {"name": "cargo_check", "outcome": "NotExecuted"},
        {"name": "cargo_lock_freshness", "outcome": "Passed"},
        {"name": "clippy", "outcome": "NotExecuted"},
        {"name": "doc_tests", "outcome": "NotExecuted"},
        {"name": "namespace_bijection", "outcome": "Passed"},
        {"name": "rustfmt", "outcome": "Failed"},
        {"name": "tests", "outcome": "NotExecuted"},
    ]
    raw = {
        "schema": attempts.RECORD_SCHEMA,
        "subject_id": ident(subject),
        "qualification_profile": "resources.stack-qualification.v1",
        "provider": "github-actions",
        "provider_attempt_ref": "workflow-run:34215218821",
        "requested_subject_sha": sha("b"),
        "actual_checkout": {"sha": sha("c"), "kind": checkout_kind},
        "theorem_results": theorem_results,
        "terminal_disposition": disposition,
        "evidence_refs": ["pr:833", "workflow-run:34215218821"],
        "non_claims": [
            "does not establish chronological ordering",
            "does not establish compilation",
        ],
    }
    normalized = attempts.normalize_record(raw)
    raw["attempt_id"] = normalized["attempt_id"]
    return raw


def test_run_can_bind_requested_head_and_distinct_synthetic_checkout():
    value = attempts.normalize_record(record())
    assert value["requested_subject_sha"] != value["actual_checkout"]["sha"]
    assert value["actual_checkout"]["kind"] == "synthetic_merge_candidate"


def test_pass_requires_every_theorem_to_pass():
    raw = record()
    raw["terminal_disposition"] = "Passed"
    raw.pop("attempt_id")
    with pytest.raises(train.TrainManifestError, match="Passed requires"):
        attempts.normalize_record(raw)


def test_source_failure_requires_exactly_one_failed_theorem():
    raw = record()
    raw["theorem_results"][0]["outcome"] = "Failed"
    raw.pop("attempt_id")
    with pytest.raises(train.TrainManifestError, match="exactly one Failed"):
        attempts.normalize_record(raw)


def test_attempt_set_is_order_independent_but_subject_scoped():
    first = record()
    second = record()
    second["provider_attempt_ref"] = "workflow-run:34215218822"
    second.pop("attempt_id")
    second["attempt_id"] = attempts.normalize_record(second)["attempt_id"]
    left = attempts.build_attempt_set([first, second])
    right = attempts.build_attempt_set([second, first])
    assert left == right

    other = record(subject="d")
    with pytest.raises(train.TrainManifestError, match="mixed qualification subjects"):
        attempts.build_attempt_set([first, other])


def test_duplicate_attempt_fails_closed():
    first = record()
    with pytest.raises(train.TrainManifestError, match="duplicate attempt identity"):
        attempts.build_attempt_set([first, first])


def test_source_changing_repair_is_not_same_subject():
    failed = record(subject="a")
    repaired = record(subject="d")
    assert failed["subject_id"] != repaired["subject_id"]
    with pytest.raises(train.TrainManifestError, match="mixed qualification subjects"):
        attempts.build_attempt_set([failed, repaired])
