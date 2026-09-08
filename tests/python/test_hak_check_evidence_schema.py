import importlib.util
import json
import sys
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

fixture_spec = importlib.util.spec_from_file_location(
    "hak_check_evidence_fixtures", ROOT / "tests/python/test_hak_check_evidence_lint.py"
)
assert fixture_spec and fixture_spec.loader
fixtures = importlib.util.module_from_spec(fixture_spec)
fixture_spec.loader.exec_module(fixtures)

SCHEMA_PATH = ROOT / "docs/architecture/hak/provider-bound-check-evidence-v1.schema.json"


def schema():
    doc = json.loads(SCHEMA_PATH.read_text())
    jsonschema.Draft202012Validator.check_schema(doc)
    return doc


def validate(doc):
    jsonschema.Draft202012Validator(schema()).validate(doc)


def test_schema_is_valid_draft_2020_12():
    schema()


def test_schema_accepts_current_provider_bound_shape():
    plan = fixtures.fixtures.load_plan()
    receipt = fixtures.fixtures.receipt(plan)
    validate(fixtures.record(plan, receipt))


def test_schema_rejects_non_provider_bound_assurance_class():
    plan = fixtures.fixtures.load_plan()
    receipt = fixtures.fixtures.receipt(plan)
    doc = fixtures.record(plan, receipt)
    doc["assurance_class"] = "CryptographicallyAttested"
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)


def test_schema_requires_exact_receipt_digest_binding():
    plan = fixtures.fixtures.load_plan()
    receipt = fixtures.fixtures.receipt(plan)
    doc = fixtures.record(plan, receipt)
    del doc["qualification_receipt"]["receipt_digest"]
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)


def test_schema_requires_terminal_provider_step():
    plan = fixtures.fixtures.load_plan()
    receipt = fixtures.fixtures.receipt(plan)
    doc = fixtures.record(plan, receipt)
    doc["provider_binding"]["step_status"] = "in_progress"
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)
