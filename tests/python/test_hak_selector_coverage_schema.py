import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((ROOT / "docs/architecture/hak/selector-coverage-v1.schema.json").read_text())
VALIDATOR = Draft202012Validator(SCHEMA)


def base_record(result):
    return {
        "schema_version": "hak.selector-coverage.v1",
        "canonicalization_profile": "hak.canonical-json.v1",
        "digest_domain": "hak.selector-coverage.v1",
        "source_receipt": {
            "schema_version": "hak.normalization-execution-receipt.v1",
            "receipt_digest": "sha256:" + "1" * 64,
            "execution_status": "Succeeded",
        },
        "resource_kind": "WorkflowJobStepsObservation",
        "policy": {
            "policy_id": "github-actions-source-observation-normalization-v1",
            "policy_digest": "sha256:" + "2" * 64,
        },
        "coverage_semantics": {
            "model": "HAKSelectorPathTerminalContextV1",
            "nonfailed_conservation": "applicable == matches + missing",
            "empty_wildcard_semantics": "zero applicable contexts -> NotApplicable",
            "historical_receipt_status_preserved": True,
        },
        "optional_results": [result],
        "coverage_digest": "sha256:" + "3" * 64,
    }


def validates(record):
    VALIDATOR.validate(record)


def test_schema_is_valid_draft_2020_12():
    Draft202012Validator.check_schema(SCHEMA)


@pytest.mark.parametrize("result", [
    {"path": "steps[*].name", "source_status": "Present", "coverage_state": "Present", "applicable": 2, "matches": 2, "missing": 0},
    {"path": "steps[*].name", "source_status": "Present", "coverage_state": "PartiallyPresent", "applicable": 2, "matches": 1, "missing": 1},
    {"path": "steps[*].name", "source_status": "Absent", "coverage_state": "Absent", "applicable": 2, "matches": 0, "missing": 2},
    {"path": "steps[*].name", "source_status": "Absent", "coverage_state": "NotApplicable", "applicable": 0, "matches": 0, "missing": 0},
    {"path": "steps[*].name", "source_status": "Failed", "coverage_state": "Failed", "applicable": None, "matches": 0, "missing": 0},
])
def test_schema_accepts_each_coverage_state_shape(result):
    validates(base_record(result))


def test_schema_rejects_partial_as_present_when_missing_nonzero():
    record = base_record({"path": "x", "source_status": "Present", "coverage_state": "Present", "applicable": 2, "matches": 1, "missing": 1})
    with pytest.raises(ValidationError):
        validates(record)


def test_schema_rejects_not_applicable_with_nonzero_counts():
    record = base_record({"path": "x", "source_status": "Absent", "coverage_state": "NotApplicable", "applicable": 1, "matches": 0, "missing": 1})
    with pytest.raises(ValidationError):
        validates(record)


def test_schema_rejects_failed_numeric_applicability():
    record = base_record({"path": "x", "source_status": "Failed", "coverage_state": "Failed", "applicable": 0, "matches": 0, "missing": 0})
    with pytest.raises(ValidationError):
        validates(record)


def test_schema_rejects_unknown_record_field():
    record = base_record({"path": "x", "source_status": "Absent", "coverage_state": "NotApplicable", "applicable": 0, "matches": 0, "missing": 0})
    record["unexpected"] = True
    with pytest.raises(ValidationError):
        validates(record)
