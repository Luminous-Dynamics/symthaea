import importlib.util, json, sys
from copy import deepcopy
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=ROOT/"scripts"; sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location("hak_normalization_execute",SCRIPTS/"hak_normalization_execute.py")
assert spec and spec.loader
hak=importlib.util.module_from_spec(spec); spec.loader.exec_module(hak)
POLICY=json.loads((ROOT/"docs/architecture/hak/policies/normalization/github-actions-source-observation-normalization-v1.json").read_text())
POLICY_REF="git:Luminous-Dynamics/symthaea@877ae00faed562ebdc4a18834cd6c30c9649fa96:docs/architecture/hak/policies/normalization/github-actions-source-observation-normalization-v1.json"
INTERPRETER_REF="git:Luminous-Dynamics/symthaea@1111111111111111111111111111111111111111:scripts/hak_normalization_execute.py"
INTERPRETER_BYTES=hak.current_interpreter_bytes()

def run(raw_obj,kind="WorkflowJobsObservation"):
    raw=json.dumps(raw_obj,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
    receipt,code=hak.execute_normalization(raw,deepcopy(POLICY),resource_kind=kind,raw_source_ref="github-actions:test-resource",policy_artifact_ref=POLICY_REF,interpreter_ref=INTERPRETER_REF,interpreter_bytes=INTERPRETER_BYTES)
    hak.validate_receipt_against_inputs(receipt,raw,POLICY,INTERPRETER_BYTES,policy_artifact_ref=POLICY_REF,interpreter_ref=INTERPRETER_REF,raw_source_ref="github-actions:test-resource",resource_kind=kind)
    return raw,receipt,code

def raw_run(raw,kind="WorkflowJobsObservation",source="x"):
    r,c=hak.execute_normalization(raw,deepcopy(POLICY),resource_kind=kind,raw_source_ref=source,policy_artifact_ref=POLICY_REF,interpreter_ref=INTERPRETER_REF,interpreter_bytes=INTERPRETER_BYTES)
    hak.validate_receipt_against_inputs(r,raw,POLICY,INTERPRETER_BYTES,policy_artifact_ref=POLICY_REF,interpreter_ref=INTERPRETER_REF,raw_source_ref=source,resource_kind=kind)
    return r,c

def test_success_omits_unknown_fields_and_preserves_order():
    _,r,c=run({"total_count":2,"jobs":[{"id":2,"run_id":1,"run_attempt":1,"name":"b","status":"completed","conclusion":"success","created_at":"c","started_at":"s","completed_at":"e","secret":"omit"},{"id":1,"run_id":1,"run_attempt":1,"name":"a","status":"completed","conclusion":None,"created_at":"c","started_at":"s","completed_at":"e"}],"unknown":"omit"})
    assert c==0 and r["execution_status"]=="Succeeded" and r["observation_issuance_permitted"] is True
    assert [x["id"] for x in r["selected_payload"]["jobs"]]==[2,1]
    assert "secret" not in r["selected_payload"]["jobs"][0] and "unknown" not in r["selected_payload"]
    assert all(x["status"]=="Present" for x in r["required_selector_results"])

def test_empty_required_wildcard_array_is_vacuously_satisfied_not_present():
    _,r,c=run({"total_count":0,"jobs":[]})
    assert c==0 and r["execution_status"]=="Succeeded" and r["selected_payload"]=={"jobs":[],"total_count":0}
    wildcard=[x for x in r["required_selector_results"] if "[*]" in x["path"]]
    assert wildcard and all(x["status"]=="VacuouslySatisfied" and x["matches"]==0 and x["missing"]==0 for x in wildcard)

def test_existing_element_missing_required_descendant_is_partial():
    _,r,c=run({"total_count":1,"jobs":[{"id":1,"run_id":1,"run_attempt":1}]})
    assert c==2 and r["execution_status"]=="Partial" and r["observation_issuance_permitted"] is False

def test_missing_required_container_is_non_success():
    _,r,c=run({"total_count":0}); assert c==2 and r["execution_status"]=="Partial"
def test_wrong_container_type_is_schema_drift_not_success():
    _,r,c=run({"total_count":0,"jobs":{}}); assert c==2 and r["execution_status"]=="Partial"
def test_optional_missing_descendant_is_allowed_and_shape_preserved():
    _,r,c=run({"total_count":1,"jobs":[{"id":1,"run_id":1,"run_attempt":1,"name":"a","status":"queued","created_at":"c","started_at":None,"completed_at":None}]}); assert c==0

def test_required_null_value_counts_as_present():
    _,r,c=run({"total_count":1,"jobs":[{"id":1,"run_id":1,"run_attempt":1,"name":None,"status":"queued","created_at":"c","started_at":None,"completed_at":None}]})
    assert c==0 and next(x for x in r["required_selector_results"] if x["path"]=="jobs[*].name")["status"]=="Present"

def test_invalid_json_produces_valid_failed_receipt():
    r,c=raw_run(b"{bad json"); assert c==2 and r["execution_status"]=="Failed"
def test_non_object_json_root_produces_valid_failed_receipt():
    r,c=raw_run(b"[]"); assert c==2 and r["execution_status"]=="Failed"
def test_duplicate_raw_json_key_produces_failed_receipt():
    r,c=raw_run(b'{"total_count":0,"total_count":1,"jobs":[]}')
    assert c==2 and r["execution_status"]=="Failed" and any("duplicate JSON object key" in e for e in r["errors"])
def test_nonstandard_nan_constant_produces_failed_receipt():
    r,c=raw_run(b'{"total_count":NaN,"jobs":[]}')
    assert c==2 and r["execution_status"]=="Failed" and any("non-standard JSON numeric constant" in e for e in r["errors"])

def test_raw_source_tampering_rejected_by_input_join():
    raw,r,_=run({"total_count":0,"jobs":[]})
    with pytest.raises(hak.NormalizationExecutionError): hak.validate_receipt_against_inputs(r,raw+b" ",POLICY,INTERPRETER_BYTES)
def test_policy_substitution_rejected_by_input_join():
    raw,r,_=run({"total_count":0,"jobs":[]}); other=deepcopy(POLICY); other["policy_id"]="other"
    with pytest.raises(Exception): hak.validate_receipt_against_inputs(r,raw,other,INTERPRETER_BYTES)
def test_interpreter_substitution_rejected_by_input_join():
    raw,r,_=run({"total_count":0,"jobs":[]})
    with pytest.raises(hak.NormalizationExecutionError): hak.validate_receipt_against_inputs(r,raw,POLICY,b"different")
def test_executor_rejects_bytes_that_do_not_match_import_snapshot():
    with pytest.raises(hak.NormalizationExecutionError): hak.execute_normalization(b"{}",deepcopy(POLICY),resource_kind="WorkflowJobsObservation",raw_source_ref="x",policy_artifact_ref=POLICY_REF,interpreter_ref=INTERPRETER_REF,interpreter_bytes=b"different")
def test_selected_payload_tampering_rejected_even_if_receipt_redigested():
    _,r,_=run({"total_count":0,"jobs":[]}); r["selected_payload"]["extra"]=1; r["receipt_digest"]=hak.compute_receipt_digest(r)
    with pytest.raises(hak.NormalizationExecutionError): hak.validate_receipt(r)
def test_coherently_redigested_selected_result_rejected_by_deterministic_replay():
    raw,r,_=run({"total_count":0,"jobs":[]}); r["selected_payload"]["extra"]=1; r["selected_result_digest"]=hak.compute_selected_result_digest(r["selected_payload"]); r["receipt_digest"]=hak.compute_receipt_digest(r); hak.validate_receipt(r)
    with pytest.raises(hak.NormalizationExecutionError): hak.validate_receipt_against_inputs(r,raw,POLICY,INTERPRETER_BYTES,policy_artifact_ref=POLICY_REF,interpreter_ref=INTERPRETER_REF,raw_source_ref="github-actions:test-resource",resource_kind="WorkflowJobsObservation")
def test_receipt_digest_tampering_rejected():
    _,r,_=run({"total_count":0,"jobs":[]}); r["receipt_digest"]="sha256:"+"0"*64
    with pytest.raises(hak.NormalizationExecutionError): hak.validate_receipt(r)
def test_partial_receipt_cannot_claim_observation_issuance():
    _,r,_=run({"total_count":1,"jobs":[{"id":1}]}); r["observation_issuance_permitted"]=True; r["receipt_digest"]=hak.compute_receipt_digest(r)
    with pytest.raises(hak.NormalizationExecutionError): hak.validate_receipt(r)
def test_unsupported_policy_grammar_rejected_before_execution():
    p=deepcopy(POLICY); p["selector_grammar"]["version"]=2; p["policy_digest"]=__import__("hak_normalization_policy_lint").compute_policy_digest(p)
    with pytest.raises(Exception): hak.execute_normalization(b"{}",p,resource_kind="WorkflowJobsObservation",raw_source_ref="x",policy_artifact_ref=POLICY_REF,interpreter_ref=INTERPRETER_REF,interpreter_bytes=INTERPRETER_BYTES)
def test_empty_required_wildcard_cannot_be_redigested_as_direct_presence():
    _,r,_=run({"total_count":0,"jobs":[]}); target=next(x for x in r["required_selector_results"] if "[*]" in x["path"]); target["status"]="Present"; r["receipt_digest"]=hak.compute_receipt_digest(r)
    with pytest.raises(hak.NormalizationExecutionError): hak.validate_receipt(r)
def test_optional_selector_cannot_claim_vacuous_satisfaction():
    _,r,_=run({"steps":[]},kind="WorkflowJobStepsObservation"); r["optional_selector_results"][0]["status"]="VacuouslySatisfied"; r["receipt_digest"]=hak.compute_receipt_digest(r)
    with pytest.raises(hak.NormalizationExecutionError): hak.validate_receipt(r)
