import json
from copy import deepcopy
from pathlib import Path
import pytest
from jsonschema import Draft202012Validator, ValidationError

ROOT=Path(__file__).resolve().parents[2]
SCHEMA=json.loads((ROOT/"docs/architecture/hak/normalization-execution-receipt-v1.schema.json").read_text())

def representative():
    return {"schema_version":"hak.normalization-execution-receipt.v1","execution_status":"Succeeded","resource_kind":"WorkflowJobsObservation","raw_source":{"source_ref":"x","raw_response_digest":"sha256:"+"1"*64},"policy":{"policy_id":"p","policy_digest":"sha256:"+"2"*64,"artifact_ref":"git:r@"+"a"*40+":p","selector_grammar":{"id":"hak.selector-path","version":1}},"interpreter":{"id":"hak-normalization-executor","version":1,"artifact_ref":"git:r@"+"b"*40+":i","content_digest":"sha256:"+"3"*64},"container_results":[],"required_selector_results":[],"optional_selector_results":[],"errors":[],"selected_payload":{},"selected_result_digest":"sha256:"+"4"*64,"observation_issuance_permitted":True,"receipt_digest":"sha256:"+"5"*64}

def test_schema_is_valid(): Draft202012Validator.check_schema(SCHEMA)
def test_representative_matches_schema(): Draft202012Validator(SCHEMA).validate(representative())
def test_schema_rejects_unknown_execution_status():
    d=representative(); d["execution_status"]="Green"
    with pytest.raises(ValidationError): Draft202012Validator(SCHEMA).validate(d)
def test_schema_rejects_unsupported_selector_grammar():
    d=representative(); d["policy"]["selector_grammar"]["version"]=2
    with pytest.raises(ValidationError): Draft202012Validator(SCHEMA).validate(d)
def test_schema_rejects_non_git_interpreter_ref():
    d=representative(); d["interpreter"]["artifact_ref"]="latest"
    with pytest.raises(ValidationError): Draft202012Validator(SCHEMA).validate(d)
def test_schema_accepts_vacuously_satisfied_required_selector():
    d=representative(); d["required_selector_results"]=[{"path":"jobs[*].id","status":"VacuouslySatisfied","matches":0,"missing":0}]
    Draft202012Validator(SCHEMA).validate(d)
def test_schema_rejects_absent_required_selector_status():
    d=representative(); d["required_selector_results"]=[{"path":"jobs[*].id","status":"Absent","matches":0,"missing":0}]
    with pytest.raises(ValidationError): Draft202012Validator(SCHEMA).validate(d)
def test_schema_rejects_vacuously_satisfied_optional_selector():
    d=representative(); d["optional_selector_results"]=[{"path":"steps[*].name","status":"VacuouslySatisfied","matches":0,"missing":0}]
    with pytest.raises(ValidationError): Draft202012Validator(SCHEMA).validate(d)
