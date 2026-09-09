import importlib.util
import sys
from pathlib import Path

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
attempts = load_module("qualification_attempt_set", SCRIPTS / "qualification_attempt_set.py")


def ident(char):
    return "sha256:" + char * 64


def sha(char):
    return char * 40


def observation(obs="a", subject="b", checkout="c", terminal="FormattingFailed", run=1):
    return {
        "schema": attempts.OBSERVATION_SCHEMA,
        "provider": "github-actions",
        "repository": "Luminous-Dynamics/symthaea",
        "workflow": {
            "name": "Resource Stack Qualification",
            "run_id": run,
            "run_number": run,
            "job_id": run + 100,
            "pull_request": 833,
        },
        "source_subject": {
            "pr_head_sha": sha("d"),
            "pr_base_sha": sha("e"),
            "checked_out_sha": sha(checkout),
        },
        "admission_correlation": {
            "admission_id": ident("f"),
            "subject_id": ident(subject),
            "predates_admission_request": False,
            "correlation_only": False,
        },
        "execution": {
            "started_at": "2026-09-09T10:00:00Z",
            "failed_at": "2026-09-09T10:01:00Z",
            "terminal_disposition": terminal,
        },
        "theorem_progress": [
            {"name": "CargoLockFreshness", "disposition": "Passed"},
            {"name": "Rustfmt", "disposition": "Passed" if terminal == "Passed" else "Failed"},
        ],
        "non_claims": ["does not establish later theorems"],
        "observation_id": ident(obs),
    }


def test_same_subject_forms_order_independent_set():
    first = observation(obs="1", run=10)
    second = observation(obs="2", checkout="9", run=11)
    left = attempts.build_attempt_set([first, second])
    right = attempts.build_attempt_set([second, first])
    assert left == right


def test_different_subjects_fail_closed():
    with pytest.raises(train.TrainManifestError, match="different admission subjects"):
        attempts.build_attempt_set([observation(obs="1", subject="a"), observation(obs="2", subject="b")])


def test_duplicate_observation_identity_fails_closed():
    first = observation(obs="1")
    with pytest.raises(train.TrainManifestError, match="duplicate observation_id"):
        attempts.build_attempt_set([first, first])


def test_different_checked_out_sha_remains_visible_without_equivalence_claim():
    result = attempts.build_attempt_set([
        observation(obs="1", checkout="a"),
        observation(obs="2", checkout="b"),
    ])
    assert {entry["checked_out_sha"] for entry in result["attempt_observations"]} == {sha("a"), sha("b")}
    assert "does not establish identical checked-out bytes across attempts" in result["non_claims"]


def test_source_changing_repair_is_not_same_subject_attempt():
    with pytest.raises(train.TrainManifestError):
        attempts.build_attempt_set([
            observation(obs="1", subject="a"),
            observation(obs="2", subject="b"),
        ])


def test_terminal_disposition_fails_closed():
    raw = observation()
    raw["execution"]["terminal_disposition"] = "MaybePassed"
    with pytest.raises(train.TrainManifestError, match="unsupported"):
        attempts.build_attempt_set([raw])
