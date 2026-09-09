#!/usr/bin/env python3
"""Audit-only HAK-013 normalization policy validator."""
from __future__ import annotations
import argparse, hashlib, json, re
from pathlib import Path
from typing import Any

POLICY_SCHEMA="hak.normalization-policy.v1"
BINDING_SCHEMA="hak.normalization-observation-binding.v1"
RESOURCE_KINDS={"WorkflowRunObservation","WorkflowJobsObservation","WorkflowJobStepsObservation"}
PATH_SEMANTICS={"required_containers":"RequireTypedContainerAndSelectShapeOnly","required_paths":"SelectValueAndRequirePresence","optional_paths":"SelectValueIfPresent"}
SELECTOR_GRAMMAR={"id":"hak.selector-path","version":1}
CONTAINER_TYPES={"array","object"}
DIGEST=re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_REF=re.compile(r"^git:[^@]+@[0-9a-f]{40}:.+$")

class NormalizationPolicyLintError(ValueError): pass

def _require(ok: bool,msg: str)->None:
    if not ok: raise NormalizationPolicyLintError(msg)
def _text(v: Any,name: str)->str:
    _require(isinstance(v,str) and v.strip(),f"{name} must be non-empty"); return v
def _array(v: Any,name: str)->list[Any]:
    _require(isinstance(v,list),f"{name} must be an array"); return v
def _obj(v: Any,name: str)->dict[str,Any]:
    _require(isinstance(v,dict),f"{name} must be an object"); return v
def _digest(domain: str,doc: dict[str,Any],field: str)->str:
    payload={k:v for k,v in doc.items() if k!=field}
    encoded=json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    return "sha256:"+hashlib.sha256(domain.encode()+b"\0"+encoded).hexdigest()
def compute_policy_digest(doc): return _digest(POLICY_SCHEMA,doc,"policy_digest")
def compute_binding_digest(doc): return _digest(BINDING_SCHEMA,doc,"binding_digest")

def validate_policy(doc: dict[str,Any])->None:
    _require(doc.get("schema_version")==POLICY_SCHEMA,f"schema_version must be {POLICY_SCHEMA}")
    for f in ("policy_id","provider","api_family","profile_ref"): _text(doc.get(f),f)
    _require(doc.get("inclusion_rule")=="ExplicitAllowList","inclusion_rule must be ExplicitAllowList")
    _require(doc.get("unknown_field_policy")=="OmitOutsidePolicyScope","unknown_field_policy must be OmitOutsidePolicyScope")
    transform=_obj(doc.get("transform"),"transform")
    _require(transform.get("method")=="FieldSelectionNoSemanticTransform","transform.method must be FieldSelectionNoSemanticTransform")
    _require(isinstance(transform.get("version"),int) and transform["version"]>0,"transform.version must be positive")
    grammar=_obj(doc.get("selector_grammar"),"selector_grammar")
    _require(grammar==SELECTOR_GRAMMAR,"selector_grammar must be hak.selector-path v1")
    semantics=_obj(doc.get("path_semantics"),"path_semantics")
    _require(set(semantics)==set(PATH_SEMANTICS),"path_semantics must define exactly the HAK-013 path categories")
    for f,e in PATH_SEMANTICS.items(): _require(semantics.get(f)==e,f"path_semantics.{f} must be {e}")
    seen=set(); profiles=_array(doc.get("resource_profiles"),"resource_profiles"); _require(bool(profiles),"resource_profiles must be non-empty")
    for i,raw in enumerate(profiles):
        item=_obj(raw,f"resource_profiles[{i}]"); kind=item.get("resource_kind")
        _require(kind in RESOURCE_KINDS,f"resource_profiles[{i}].resource_kind invalid")
        _require(kind not in seen,f"duplicate resource profile: {kind}"); seen.add(kind)
        containers=_array(item.get("required_containers"),f"resource_profiles[{i}].required_containers"); cpaths=[]
        for j,rc in enumerate(containers):
            c=_obj(rc,f"resource_profiles[{i}].required_containers[{j}]")
            _require(set(c)=={"path","container_type"},f"resource_profiles[{i}].required_containers[{j}] must contain exactly path/container_type")
            cpaths.append(_text(c.get("path"),f"resource_profiles[{i}].required_containers[{j}].path"))
            _require(c.get("container_type") in CONTAINER_TYPES,f"resource_profiles[{i}].required_containers[{j}].container_type invalid")
        _require(len(cpaths)==len(set(cpaths)),f"resource_profiles[{i}].required_containers contains duplicate paths")
        required=_array(item.get("required_paths"),f"resource_profiles[{i}].required_paths")
        optional=_array(item.get("optional_paths"),f"resource_profiles[{i}].optional_paths")
        for label,paths in (("required_paths",required),("optional_paths",optional)):
            for j,p in enumerate(paths): _text(p,f"resource_profiles[{i}].{label}[{j}]")
            _require(len(paths)==len(set(paths)),f"resource_profiles[{i}].{label} contains duplicates")
        _require(bool(cpaths or required or optional),f"resource_profiles[{i}] must select or require at least one path")
        _require(set(required).isdisjoint(optional),f"resource_profiles[{i}] required_paths/optional_paths overlap")
        for label,paths in (("required_paths",required),("optional_paths",optional)):
            overlap=set(cpaths).intersection(paths); _require(not overlap,f"resource_profiles[{i}] required_containers/{label} paths overlap: {sorted(overlap)}")
    _text(doc.get("omission_semantics"),"omission_semantics"); _text(doc.get("redaction_semantics"),"redaction_semantics")
    s=doc.get("supersedes_policy_id"); _require(s is None or (isinstance(s,str) and s.strip()),"supersedes_policy_id must be null or non-empty")
    d=doc.get("policy_digest"); _require(isinstance(d,str) and DIGEST.fullmatch(d),"policy_digest must be sha256:<64 hex>")
    _require(d==compute_policy_digest(doc),"policy_digest does not match canonical policy content")

def validate_binding(observation,binding,policy=None)->None:
    _require(binding.get("schema_version")==BINDING_SCHEMA,f"schema_version must be {BINDING_SCHEMA}"); _text(binding.get("binding_id"),"binding_id")
    _require(binding.get("observation_id")==observation.get("observation_id"),"binding observation_id must match observation")
    profile=_obj(observation.get("normalization"),"observation.normalization").get("profile_ref")
    _require(binding.get("profile_ref")==profile,"binding profile_ref must match observation normalization.profile_ref")
    status=binding.get("binding_status"); _require(status in {"BoundToPolicy","RetrospectiveUnbound"},"binding_status invalid")
    identity=binding.get("policy_identity"); relation=binding.get("commitment_relation")
    _require(relation in {"PreObservationEstablished","PostObservationEstablished","NotEstablished"},"commitment_relation invalid")
    temporal=binding.get("temporal_evidence_ref")
    if relation=="NotEstablished": _require(temporal is None,"NotEstablished commitment relation requires null temporal_evidence_ref")
    else: _text(temporal,"temporal_evidence_ref")
    if status=="RetrospectiveUnbound":
        _require(identity is None,"RetrospectiveUnbound must not carry policy_identity"); _require(relation=="NotEstablished","RetrospectiveUnbound cannot claim pre/post-observation policy timing")
        _require(binding.get("omitted_field_completeness")=="NotEstablished","RetrospectiveUnbound cannot establish omitted-field completeness")
    else:
        ident=_obj(identity,"policy_identity"); _require(policy is not None,"BoundToPolicy requires loaded policy"); validate_policy(policy)
        _require(ident.get("policy_id")==policy.get("policy_id"),"policy_identity.policy_id mismatch"); _require(ident.get("policy_digest")==policy.get("policy_digest"),"policy_identity.policy_digest mismatch")
        _require(GIT_REF.fullmatch(_text(ident.get("artifact_ref"),"policy_identity.artifact_ref")) is not None,"policy_identity.artifact_ref must be exact git ref")
        _require(binding.get("profile_ref")==policy.get("profile_ref"),"bound policy profile_ref mismatch")
    raw=binding.get("raw_response_retained"); _require(isinstance(raw,bool),"raw_response_retained must be boolean")
    replay=binding.get("normalization_replayability"); completeness=binding.get("omitted_field_completeness")
    if raw: _require(replay=="ReplayableFromRetainedRawResponse","retained raw response requires replayable normalization")
    else:
        _require(replay=="NotReplayableWithoutRawResponse","missing raw response cannot claim replayability"); _require(completeness=="NotEstablished","missing raw response cannot establish omitted-field completeness")
    if completeness=="EstablishedWithinPolicyScope":
        _require(status=="BoundToPolicy","omitted-field completeness requires content-bound policy"); _require(raw is True,"omitted-field completeness requires retained raw response")
    _text(binding.get("reason"),"reason"); d=binding.get("binding_digest")
    _require(isinstance(d,str) and DIGEST.fullmatch(d),"binding_digest must be sha256:<64 hex>"); _require(d==compute_binding_digest(binding),"binding_digest does not match canonical binding content")

def _load(path: Path):
    doc=json.loads(path.read_text()); _require(isinstance(doc,dict),f"{path} root must be an object"); return doc

def main(argv=None)->int:
    p=argparse.ArgumentParser(); p.add_argument("--policy",type=Path,required=True); p.add_argument("--observation",type=Path); p.add_argument("--binding",type=Path); args=p.parse_args(argv)
    try:
        policy=_load(args.policy); validate_policy(policy)
        if args.observation or args.binding:
            _require(args.observation is not None and args.binding is not None,"--observation and --binding must be supplied together")
            validate_binding(_load(args.observation),_load(args.binding),policy)
    except (OSError,json.JSONDecodeError,NormalizationPolicyLintError) as exc:
        print(f"FAIL: {exc}"); return 1
    print("OK   HAK normalization policy/binding"); return 0
if __name__=="__main__": raise SystemExit(main())
