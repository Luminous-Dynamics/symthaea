import importlib.util, json, sys
from copy import deepcopy
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[2]
SCRIPTS=ROOT/"scripts"; sys.path.insert(0,str(SCRIPTS))
spec=importlib.util.spec_from_file_location("hak_normalization_policy_lint",SCRIPTS/"hak_normalization_policy_lint.py")
assert spec and spec.loader
hak=importlib.util.module_from_spec(spec); spec.loader.exec_module(hak)
POLICY=ROOT/"docs/architecture/hak/policies/normalization/github-actions-source-observation-normalization-v1.json"
OBS=ROOT/"docs/architecture/hak/evidence/source/hak007-run-34225891059.workflow-run.observation.json"
BIND=ROOT/"docs/architecture/hak/evidence/source/hak007-run-34225891059.workflow-run.normalization-binding.json"

def load(p): return json.loads(p.read_text())
def redigest_policy(d): d["policy_digest"]=hak.compute_policy_digest(d)
def redigest_binding(d): d["binding_digest"]=hak.compute_binding_digest(d)

def test_policy_valid(): hak.validate_policy(load(POLICY))
def test_historical_binding_valid(): hak.validate_binding(load(OBS),load(BIND),load(POLICY))

def test_steps_container_is_presence_shape_not_whole_subtree_selection():
    p=load(POLICY)
    profile=next(x for x in p["resource_profiles"] if x["resource_kind"]=="WorkflowJobStepsObservation")
    assert profile["required_container_paths"]==["steps"]
    assert "steps" not in profile["required_paths"]
    assert "steps" not in profile["optional_paths"]
    assert p["path_semantics"]["required_container_paths"]=="RequireAndSelectContainerShapeOnly"
    hak.validate_policy(p)

def test_symbolic_profile_ref_is_not_policy_identity():
    b=load(BIND); b["binding_status"]="BoundToPolicy"; b["policy_identity"]=b["profile_ref"]; redigest_binding(b)
    with pytest.raises(hak.NormalizationPolicyLintError): hak.validate_binding(load(OBS),b,load(POLICY))

def test_policy_digest_tampering_rejected():
    p=load(POLICY); p["omission_semantics"]+=" changed"
    with pytest.raises(hak.NormalizationPolicyLintError): hak.validate_policy(p)

def test_duplicate_resource_profile_rejected():
    p=load(POLICY); p["resource_profiles"].append(deepcopy(p["resource_profiles"][0])); redigest_policy(p)
    with pytest.raises(hak.NormalizationPolicyLintError): hak.validate_policy(p)

def test_required_optional_overlap_rejected():
    p=load(POLICY); p["resource_profiles"][0]["optional_paths"].append(p["resource_profiles"][0]["required_paths"][0]); redigest_policy(p)
    with pytest.raises(hak.NormalizationPolicyLintError): hak.validate_policy(p)

def test_required_container_exact_overlap_with_selected_path_rejected():
    p=load(POLICY); p["resource_profiles"][0]["required_container_paths"].append(p["resource_profiles"][0]["required_paths"][0]); redigest_policy(p)
    with pytest.raises(hak.NormalizationPolicyLintError): hak.validate_policy(p)

def test_container_prefix_of_selected_descendant_is_allowed():
    p=load(POLICY)
    profile=next(x for x in p["resource_profiles"] if x["resource_kind"]=="WorkflowJobStepsObservation")
    assert "steps" in profile["required_container_paths"]
    assert "steps[*].name" in profile["optional_paths"]
    hak.validate_policy(p)

def test_missing_required_container_path_semantics_rejected():
    p=load(POLICY); p["path_semantics"].pop("required_container_paths"); redigest_policy(p)
    with pytest.raises(hak.NormalizationPolicyLintError): hak.validate_policy(p)

def test_duplicate_selected_path_rejected():
    p=load(POLICY); p["resource_profiles"][0]["required_paths"].append(p["resource_profiles"][0]["required_paths"][0]); redigest_policy(p)
    with pytest.raises(hak.NormalizationPolicyLintError): hak.validate_policy(p)

def test_missing_omission_semantics_rejected():
    p=load(POLICY); p["omission_semantics"]=""; redigest_policy(p)
    with pytest.raises(hak.NormalizationPolicyLintError): hak.validate_policy(p)

def test_historical_binding_cannot_upgrade_without_policy_identity():
    b=load(BIND); b["binding_status"]="BoundToPolicy"; redigest_binding(b)
    with pytest.raises(hak.NormalizationPolicyLintError): hak.validate_binding(load(OBS),b,load(POLICY))

def test_historical_binding_cannot_claim_precommit_without_temporal_evidence():
    b=load(BIND); b["commitment_relation"]="PreObservationEstablished"; redigest_binding(b)
    with pytest.raises(hak.NormalizationPolicyLintError): hak.validate_binding(load(OBS),b,load(POLICY))

def test_historical_binding_cannot_claim_omitted_field_completeness():
    b=load(BIND); b["omitted_field_completeness"]="EstablishedWithinPolicyScope"; redigest_binding(b)
    with pytest.raises(hak.NormalizationPolicyLintError): hak.validate_binding(load(OBS),b,load(POLICY))

def test_historical_binding_cannot_claim_replayability_without_raw_response():
    b=load(BIND); b["normalization_replayability"]="ReplayableFromRetainedRawResponse"; redigest_binding(b)
    with pytest.raises(hak.NormalizationPolicyLintError): hak.validate_binding(load(OBS),b,load(POLICY))

def test_historical_binding_profile_mismatch_rejected():
    b=load(BIND); b["profile_ref"]="hak:other"; redigest_binding(b)
    with pytest.raises(hak.NormalizationPolicyLintError): hak.validate_binding(load(OBS),b,load(POLICY))

def test_binding_digest_tampering_rejected():
    b=load(BIND); b["reason"]+=" changed"
    with pytest.raises(hak.NormalizationPolicyLintError): hak.validate_binding(load(OBS),b,load(POLICY))
