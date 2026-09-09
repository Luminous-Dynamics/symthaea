import json
from copy import deepcopy
from pathlib import Path
import pytest
from jsonschema import Draft202012Validator, ValidationError

ROOT=Path(__file__).resolve().parents[2]
POLICY_SCHEMA=json.loads((ROOT/"docs/architecture/hak/normalization-policy-v1.schema.json").read_text())
BINDING_SCHEMA=json.loads((ROOT/"docs/architecture/hak/normalization-observation-binding-v1.schema.json").read_text())
POLICY=json.loads((ROOT/"docs/architecture/hak/policies/normalization/github-actions-source-observation-normalization-v1.json").read_text())
BINDING=json.loads((ROOT/"docs/architecture/hak/evidence/source/hak007-run-34225891059.workflow-run.normalization-binding.json").read_text())

def test_schemas_are_valid():
    Draft202012Validator.check_schema(POLICY_SCHEMA); Draft202012Validator.check_schema(BINDING_SCHEMA)
def test_representative_policy_matches_schema(): Draft202012Validator(POLICY_SCHEMA).validate(POLICY)
def test_representative_binding_matches_schema(): Draft202012Validator(BINDING_SCHEMA).validate(BINDING)
def test_schema_rejects_missing_policy_digest():
    doc=deepcopy(POLICY); doc.pop("policy_digest")
    with pytest.raises(ValidationError): Draft202012Validator(POLICY_SCHEMA).validate(doc)
def test_schema_rejects_missing_selector_grammar():
    doc=deepcopy(POLICY); doc.pop("selector_grammar")
    with pytest.raises(ValidationError): Draft202012Validator(POLICY_SCHEMA).validate(doc)
def test_schema_rejects_unsupported_selector_grammar_version():
    doc=deepcopy(POLICY); doc["selector_grammar"]["version"]=2
    with pytest.raises(ValidationError): Draft202012Validator(POLICY_SCHEMA).validate(doc)
def test_schema_rejects_missing_path_semantics():
    doc=deepcopy(POLICY); doc.pop("path_semantics")
    with pytest.raises(ValidationError): Draft202012Validator(POLICY_SCHEMA).validate(doc)
def test_schema_rejects_missing_required_containers():
    doc=deepcopy(POLICY); doc["resource_profiles"][0].pop("required_containers")
    with pytest.raises(ValidationError): Draft202012Validator(POLICY_SCHEMA).validate(doc)
def test_schema_rejects_invalid_container_type():
    doc=deepcopy(POLICY); doc["resource_profiles"][0]["required_containers"][0]["container_type"]="scalar"
    with pytest.raises(ValidationError): Draft202012Validator(POLICY_SCHEMA).validate(doc)
def test_schema_rejects_unknown_binding_status():
    doc=deepcopy(BINDING); doc["binding_status"]="Precommitted"
    with pytest.raises(ValidationError): Draft202012Validator(BINDING_SCHEMA).validate(doc)
