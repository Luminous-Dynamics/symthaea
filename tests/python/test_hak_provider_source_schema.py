import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
OBS_SCHEMA_PATH = ROOT / "docs/architecture/hak/provider-source-observation-v1.schema.json"
PROJECTION_SCHEMA_PATH = ROOT / "docs/architecture/hak/provider-projection-map-v1.schema.json"
OBS_PATHS = [
    ROOT / "docs/architecture/hak/evidence/source/hak007-run-34225891059.workflow-run.observation.json",
    ROOT / "docs/architecture/hak/evidence/source/hak007-run-34225891059.jobs.observation.json",
    ROOT / "docs/architecture/hak/evidence/source/hak007-run-34225891059.job-102059802264.steps.observation.json",
]
PROJECTION_PATH = ROOT / "docs/architecture/hak/evidence/source/hak007-run-34225891059.provider-projection.json"


def load(path):
    return json.loads(path.read_text())


def test_provider_source_observation_schema_is_valid_and_accepts_real_observations():
    schema = load(OBS_SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema)
    for path in OBS_PATHS:
        validator.validate(load(path))


def test_provider_projection_schema_is_valid_and_accepts_real_projection():
    schema = load(PROJECTION_SCHEMA_PATH)
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(load(PROJECTION_PATH))
