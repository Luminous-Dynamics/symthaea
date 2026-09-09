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


def observation(subject="b", checkout="c", terminal="FormattingFailed", run=1):
    raw = {
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
        "execution_environment": {
            "runner_os": "ubuntu-24.04.4",
            "runner_image": "ubuntu-24.04",
            "runner_image_version": "20260831.293.1",
            "rustc": "1.96.0 (example)",
            "target": "x86_64-unknown-linux-gnu",
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
        "failure": {
            "package": "symthaea-resource-accounting",
            "path": "crates/domains/symthaea-resource-accounting/src/lib.rs",
        },
        "non_claims": ["does not establish later theorems"],
    }
    raw["observation_id"] = attempts.compute_observation_id(raw)
    return raw


def reidentify(raw):
    raw["observation_id"] = attempts.compute_observation_id(raw)
    return raw


def test_same_subject_forms_order_independent_set():
    first = observation(run=10)
    second = observation(checkout="9", run=11)
    left = attempts.build_attempt_set([first, second])
    right = attempts.build_attempt_set([second, first])
    assert left == right


def test_different_subjects_fail_closed():
    with pytest.raises(train.TrainManifestError, match="different admission subjects"):
        attempts.build_attempt_set([observation(subject="a", run=1), observation(subject="b", run=2)])


def test_duplicate_observation_identity_fails_closed():
    first = observation()
    with pytest.raises(train.TrainManifestError, match="duplicate observation_id"):
        attempts.build_attempt_set([first, first])


def test_duplicate_provider_attempt_identity_fails_closed_even_with_distinct_content():
    first = observation(run=10)
    second = observation(run=10)
    second["source_subject"]["checked_out_sha"] = sha("9")
    reidentify(second)
    with pytest.raises(train.TrainManifestError, match="duplicate provider run/job identity"):
        attempts.build_attempt_set([first, second])


def test_different_checked_out_sha_remains_visible_without_equivalence_claim():
    result = attempts.build_attempt_set([
        observation(checkout="a", run=10),
        observation(checkout="b", run=11),
    ])
    assert {entry["checked_out_sha"] for entry in result["attempt_observations"]} == {sha("a"), sha("b")}
    assert "does not establish identical checked-out bytes across attempts" in result["non_claims"]


def test_source_changing_repair_is_not_same_subject_attempt():
    with pytest.raises(train.TrainManifestError):
        attempts.build_attempt_set([
            observation(subject="a", run=1),
            observation(subject="b", run=2),
        ])


def test_terminal_disposition_fails_closed():
    raw = observation()
    raw["execution"]["terminal_disposition"] = "MaybePassed"
    reidentify(raw)
    with pytest.raises(train.TrainManifestError, match="unsupported"):
        attempts.build_attempt_set([raw])


def test_observation_identity_tampering_fails_closed():
    raw = observation()
    raw["failure"]["path"] = "changed/path.rs"
    with pytest.raises(train.TrainManifestError, match="does not match canonical observation contents"):
        attempts.build_attempt_set([raw])


def test_source_failure_requires_exactly_one_failed_theorem():
    raw = observation()
    raw["theorem_progress"] = [
        {"name": "CargoLockFreshness", "disposition": "Passed"},
        {"name": "Rustfmt", "disposition": "Failed"},
        {"name": "CargoCheck", "disposition": "Failed"},
    ]
    reidentify(raw)
    with pytest.raises(train.TrainManifestError, match="exactly one Failed"):
        attempts.build_attempt_set([raw])


def test_theorem_after_source_failure_must_be_not_executed():
    raw = observation()
    raw["theorem_progress"] = [
        {"name": "CargoLockFreshness", "disposition": "Passed"},
        {"name": "Rustfmt", "disposition": "Failed"},
        {"name": "CargoCheck", "disposition": "Passed"},
    ]
    reidentify(raw)
    with pytest.raises(train.TrainManifestError, match="after a source failure"):
        attempts.build_attempt_set([raw])


def test_passed_terminal_requires_every_theorem_passed():
    raw = observation(terminal="Passed")
    raw["theorem_progress"].append({"name": "CargoCheck", "disposition": "NotExecuted"})
    reidentify(raw)
    with pytest.raises(train.TrainManifestError, match="requires every theorem to be Passed"):
        attempts.build_attempt_set([raw])


def test_infrastructure_terminal_allows_passed_prefix_then_not_executed_suffix():
    raw = observation(terminal="FormattingFailed")
    raw["execution"]["terminal_disposition"] = "InfrastructureUnavailable"
    raw["theorem_progress"] = [
        {"name": "CargoLockFreshness", "disposition": "Passed"},
        {"name": "Rustfmt", "disposition": "Passed"},
        {"name": "CargoCheck", "disposition": "NotExecuted"},
        {"name": "Clippy", "disposition": "NotExecuted"},
    ]
    reidentify(raw)
    result = attempts.build_attempt_set([raw])
    assert result["attempt_observations"][0]["terminal_disposition"] == "InfrastructureUnavailable"


def test_infrastructure_terminal_cannot_resume_after_not_executed():
    raw = observation(terminal="FormattingFailed")
    raw["execution"]["terminal_disposition"] = "OutcomeUnknown"
    raw["theorem_progress"] = [
        {"name": "CargoLockFreshness", "disposition": "Passed"},
        {"name": "Rustfmt", "disposition": "NotExecuted"},
        {"name": "CargoCheck", "disposition": "Passed"},
    ]
    reidentify(raw)
    with pytest.raises(train.TrainManifestError, match="cannot resume"):
        attempts.build_attempt_set([raw])
