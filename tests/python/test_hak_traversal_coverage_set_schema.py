import copy
import json
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/python"
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(TESTS))
sys.path.insert(0, str(SCRIPTS))

import test_hak_traversal_coverage_set as fixtures

SCHEMA_PATH = ROOT / "docs/architecture/hak/traversal-coverage-set-v1.schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text())
VALIDATOR = Draft202012Validator(SCHEMA)


def valid_record():
    return fixtures.derive()[3]


def test_schema_itself_is_valid_draft_2020_12():
    Draft202012Validator.check_schema(SCHEMA)


def test_valid_runtime_record_matches_schema():
    VALIDATOR.validate(valid_record())


def test_unknown_root_field_is_rejected():
    record = valid_record()
    record["oracle"] = True
    with pytest.raises(ValidationError):
        VALIDATOR.validate(record)


def test_population_qualification_is_fixed_to_not_established():
    record = valid_record()
    record["population_qualification"] = "Verified"
    with pytest.raises(ValidationError):
        VALIDATOR.validate(record)


def test_normalization_input_replay_is_fixed_to_not_established():
    record = valid_record()
    record["normalization_input_replay"] = "Verified"
    with pytest.raises(ValidationError):
        VALIDATOR.validate(record)


def test_normalization_policy_content_verification_is_fixed_to_not_established():
    record = valid_record()
    record["normalization_policy_content_verification"] = "Verified"
    with pytest.raises(ValidationError):
        VALIDATOR.validate(record)


def test_source_assurance_cannot_be_promoted():
    record = valid_record()
    record["source_assurance"]["raw_response_content_verification"] = "Verified"
    with pytest.raises(ValidationError):
        VALIDATOR.validate(record)


def test_unassessed_page_cannot_carry_assessed_fields():
    declaration, traversal, witnesses = fixtures.build_world()
    witnesses[1] = {"page_id": "page-2", "disposition": "Unassessed", "reason": "not assessed"}
    record = fixtures.derive((declaration, traversal, witnesses))[3]
    record["pages"][1]["coverage_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError):
        VALIDATOR.validate(record)


def test_assessed_page_requires_profile_binding_digest():
    record = valid_record()
    del record["pages"][0]["profile_binding_digest"]
    with pytest.raises(ValidationError):
        VALIDATOR.validate(record)


def test_aggregate_state_vocabulary_is_closed():
    record = valid_record()
    record["aggregate"]["coverage_state"] = "PopulationPresent"
    with pytest.raises(ValidationError):
        VALIDATOR.validate(record)
