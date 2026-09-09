import copy
import json
import sys
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, ValidationError

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import hak_provider_response_envelope as hak
import test_hak_provider_response_envelope as fixtures

ENVELOPE_SCHEMA = json.loads(
    (ROOT / "docs/architecture/hak/provider-response-envelope-v1.schema.json").read_text()
)
PROJECTION_SCHEMA = json.loads(
    (ROOT / "docs/architecture/hak/link-pagination-projection-receipt-v1.schema.json").read_text()
)
ENV_VALIDATOR = Draft202012Validator(ENVELOPE_SCHEMA)
PROJECTION_VALIDATOR = Draft202012Validator(PROJECTION_SCHEMA)


def valid_envelope():
    return fixtures.make_env([
        {"name": "Link", "value": '<https://api.github.com/x?page=2>; rel="next"'}
    ])


def valid_projection():
    return fixtures.project(valid_envelope())


def test_both_schemas_are_valid_draft_2020_12():
    Draft202012Validator.check_schema(ENVELOPE_SCHEMA)
    Draft202012Validator.check_schema(PROJECTION_SCHEMA)


def test_runtime_envelope_matches_schema():
    ENV_VALIDATOR.validate(valid_envelope())


def test_runtime_projection_matches_schema():
    PROJECTION_VALIDATOR.validate(valid_projection())


def test_envelope_unknown_root_field_is_rejected():
    value = valid_envelope()
    value["oracle"] = True
    with pytest.raises(ValidationError):
        ENV_VALIDATOR.validate(value)


def test_envelope_provider_authentication_is_fixed_not_established():
    value = valid_envelope()
    value["provider_authentication"] = "Verified"
    with pytest.raises(ValidationError):
        ENV_VALIDATOR.validate(value)


def test_envelope_raw_http_wire_representation_is_fixed_not_retained():
    value = valid_envelope()
    value["retention"]["raw_http_wire_representation"] = "Retained"
    with pytest.raises(ValidationError):
        ENV_VALIDATOR.validate(value)


def test_projection_unknown_root_field_is_rejected():
    value = valid_projection()
    value["oracle"] = True
    with pytest.raises(ValidationError):
        PROJECTION_VALIDATOR.validate(value)


def test_projection_policy_digest_is_exact_strengthened_policy():
    value = valid_projection()
    value["policy"]["policy_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValidationError):
        PROJECTION_VALIDATOR.validate(value)


def test_projection_provider_authentication_is_fixed_not_established():
    value = valid_projection()
    value["provider_authentication"] = "Verified"
    with pytest.raises(ValidationError):
        PROJECTION_VALIDATOR.validate(value)


def test_projection_http_wire_verification_is_fixed_not_established():
    value = valid_projection()
    value["http_wire_verification"] = "Verified"
    with pytest.raises(ValidationError):
        PROJECTION_VALIDATOR.validate(value)


def test_projection_exhaustion_verification_is_fixed_not_established():
    value = valid_projection()
    value["exhaustion_verification"] = "Verified"
    with pytest.raises(ValidationError):
        PROJECTION_VALIDATOR.validate(value)


def test_absent_next_requires_null_targets():
    value = fixtures.project(fixtures.make_env())
    value["next_relation"]["target"] = "https://api.github.com/x?page=2"
    with pytest.raises(ValidationError):
        PROJECTION_VALIDATOR.validate(value)


def test_present_next_requires_nonempty_target():
    value = valid_projection()
    value["next_relation"]["target"] = None
    with pytest.raises(ValidationError):
        PROJECTION_VALIDATOR.validate(value)
