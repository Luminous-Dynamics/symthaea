import importlib.util
import json
import sys
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

FIXTURES_PATH = ROOT / "tests/python/test_hak_interpretation_lint.py"
spec = importlib.util.spec_from_file_location("hak_interpretation_fixtures", FIXTURES_PATH)
assert spec and spec.loader
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)

CONFORMANCE_SCHEMA_PATH = ROOT / "docs/architecture/hak/plan-conformance-v1.schema.json"
INTERPRETATION_SCHEMA_PATH = ROOT / "docs/architecture/hak/evidence-interpretation-v1.schema.json"


def schema(path: Path):
    doc = json.loads(path.read_text())
    jsonschema.Draft202012Validator.check_schema(doc)
    return doc


def validate(doc, schema_doc):
    jsonschema.Draft202012Validator(schema_doc).validate(doc)


def test_hak008_schemas_are_valid_draft_2020_12():
    schema(CONFORMANCE_SCHEMA_PATH)
    schema(INTERPRETATION_SCHEMA_PATH)


def test_conformance_schema_accepts_current_runtime_shape():
    plan = fixtures.load_plan()
    rec = fixtures.receipt(plan)
    validate(fixtures.conformance(plan, rec), schema(CONFORMANCE_SCHEMA_PATH))


def test_interpretation_schema_accepts_current_runtime_shape():
    plan = fixtures.load_plan()
    rec = fixtures.receipt(plan)
    conf = fixtures.conformance(plan, rec)
    validate(fixtures.interpretation(plan, rec, conf), schema(INTERPRETATION_SCHEMA_PATH))


def test_conformance_schema_requires_exact_plan_digest_binding():
    plan = fixtures.load_plan()
    rec = fixtures.receipt(plan)
    doc = fixtures.conformance(plan, rec)
    del doc["plan"]["plan_digest"]
    with pytest.raises(jsonschema.ValidationError):
        validate(doc, schema(CONFORMANCE_SCHEMA_PATH))


def test_interpretation_schema_requires_plan_and_evidence_digest_bindings():
    plan = fixtures.load_plan()
    rec = fixtures.receipt(plan)
    conf = fixtures.conformance(plan, rec)
    doc = fixtures.interpretation(plan, rec, conf)
    del doc["plan_digest"]
    with pytest.raises(jsonschema.ValidationError):
        validate(doc, schema(INTERPRETATION_SCHEMA_PATH))

    doc = fixtures.interpretation(plan, rec, conf)
    del doc["receipt_bindings"][0]["receipt_digest"]
    with pytest.raises(jsonschema.ValidationError):
        validate(doc, schema(INTERPRETATION_SCHEMA_PATH))

    doc = fixtures.interpretation(plan, rec, conf)
    del doc["conformance_bindings"][0]["conformance_digest"]
    with pytest.raises(jsonschema.ValidationError):
        validate(doc, schema(INTERPRETATION_SCHEMA_PATH))


def test_schema_rejects_model_assistance_as_authority_kind():
    plan = fixtures.load_plan()
    rec = fixtures.receipt(plan)
    conf = fixtures.conformance(plan, rec)
    doc = fixtures.interpretation(plan, rec, conf)
    doc["interpreter"]["kind"] = "ModelAssistedReview"
    doc["interpreter"]["model_assisted"] = True
    doc["interpreter"]["model_ref"] = "model:test"
    with pytest.raises(jsonschema.ValidationError):
        validate(doc, schema(INTERPRETATION_SCHEMA_PATH))


def test_schema_requires_model_ref_when_model_assisted():
    plan = fixtures.load_plan()
    rec = fixtures.receipt(plan)
    conf = fixtures.conformance(plan, rec)
    doc = fixtures.interpretation(plan, rec, conf)
    doc["interpreter"]["kind"] = "HumanReviewer"
    doc["interpreter"]["model_assisted"] = True
    doc["interpreter"]["model_ref"] = None
    with pytest.raises(jsonschema.ValidationError):
        validate(doc, schema(INTERPRETATION_SCHEMA_PATH))


def test_schema_rejects_supported_tier_on_nonqualified_claim():
    plan = fixtures.load_plan()
    rec = fixtures.receipt(plan)
    conf = fixtures.conformance(plan, rec)
    doc = fixtures.interpretation(plan, rec, conf, status="InsufficientEvidence", tier=None)
    doc["claims"][0]["supported_tier"] = "E5"
    with pytest.raises(jsonschema.ValidationError):
        validate(doc, schema(INTERPRETATION_SCHEMA_PATH))
