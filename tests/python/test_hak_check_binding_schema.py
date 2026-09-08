import json
from copy import deepcopy
from pathlib import Path

import pytest

jsonschema = pytest.importorskip("jsonschema")

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "docs/architecture/hak/check-evidence-binding-policy-v1.schema.json"
POLICY_PATH = ROOT / "docs/architecture/hak/policies/hak010-precommitted-check-binding-policy-v1.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(doc):
    jsonschema.Draft202012Validator(load(SCHEMA_PATH)).validate(doc)


def test_static_hak010_policy_matches_schema():
    validate(load(POLICY_PATH))


def test_schema_rejects_malformed_policy_digest():
    doc = load(POLICY_PATH)
    doc["policy_digest"] = "sha256:not-a-digest"
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)


def test_schema_rejects_nonpositive_step_number():
    doc = load(POLICY_PATH)
    doc["bindings"][0]["step_number"] = 0
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)


def test_schema_rejects_unknown_obligation_kind():
    doc = load(POLICY_PATH)
    doc["bindings"][0]["obligation_kind"] = "AdHocCheck"
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)


def test_schema_rejects_nonterminal_accepted_conclusion():
    doc = load(POLICY_PATH)
    doc["bindings"][0]["accepted_step_conclusions"] = ["queued"]
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)


def test_schema_rejects_unexpected_selector_fields():
    doc = load(POLICY_PATH)
    doc["bindings"][0]["regex"] = ".*"
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)
