import json
from pathlib import Path

import jsonschema
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "docs/architecture/hak/real-provider-evidence-capsule-v1.schema.json"
CAPSULE_PATH = ROOT / "docs/architecture/hak/evidence/real/hak007-run-34225891059.capsule.json"


def load(path):
    return json.loads(path.read_text())


def test_real_provider_capsule_schema_is_valid():
    schema = load(SCHEMA_PATH)
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(load(CAPSULE_PATH), schema)


def test_schema_rejects_provider_check_evidence_when_steps_are_empty():
    schema = load(SCHEMA_PATH)
    doc = load(CAPSULE_PATH)
    doc["provider_check_evidence"] = ["fabricated:evidence"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)


def test_schema_rejects_malformed_capsule_digest():
    schema = load(SCHEMA_PATH)
    doc = load(CAPSULE_PATH)
    doc["capsule_digest"] = "sha256:not-a-digest"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(doc, schema)
