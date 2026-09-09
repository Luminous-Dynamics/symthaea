#!/usr/bin/env python3
"""HAK-014 deterministic normalization-policy executor.

Audit/evidence tooling only. Applies HAK-013 normalization policy semantics to
retained JSON bytes and emits a normalization execution receipt. It does not
authenticate the provider, issue ProviderSourceObservation authority, interpret
semantic truth, or grant runtime authority.
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path
from typing import Any

import hak_normalization_policy_lint as policy_lint

RECEIPT_SCHEMA="hak.normalization-execution-receipt.v1"
GRAMMAR={"id":"hak.selector-path","version":1}
INTERPRETER={"id":"hak-normalization-executor","version":1}
DIGEST_RE=re.compile(r"^sha256:[0-9a-f]{64}$")
GIT_REF_RE=re.compile(r"^git:[^@]+@[0-9a-f]{40}:.+$")
TOKEN_RE=re.compile(r"^[A-Za-z0-9_]+(?:\[\*\])?$")
TERMINAL={"Succeeded","Partial","Failed"}
_INTERPRETER_SOURCE_SNAPSHOT=Path(__file__).read_bytes()

class NormalizationExecutionError(ValueError): pass
class _Missing: pass
_MISSING=_Missing()

def _require(ok: bool,msg: str)->None:
    if not ok: raise NormalizationExecutionError(msg)
def _canonical_digest(domain: str,value: Any)->str:
    encoded=json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
    return "sha256:"+hashlib.sha256(domain.encode()+b"\0"+encoded).hexdigest()
def compute_selected_result_digest(payload: Any)->str:
    return _canonical_digest("hak.normalized-selected-result.v1",payload)
def compute_receipt_digest(receipt: dict[str,Any])->str:
    return _canonical_digest(RECEIPT_SCHEMA,{k:v for k,v in receipt.items() if k!="receipt_digest"})
def sha256_bytes(data: bytes)->str:
    return "sha256:"+hashlib.sha256(data).hexdigest()
def current_interpreter_bytes()->bytes:
    return _INTERPRETER_SOURCE_SNAPSHOT
def _strict_object(pairs: list[tuple[str,Any]])->dict[str,Any]:
    out={}
    for key,value in pairs:
        if key in out: raise NormalizationExecutionError(f"duplicate JSON object key: {key}")
        out[key]=value
    return out
def _reject_json_constant(value: str)->None:
    raise NormalizationExecutionError(f"non-standard JSON numeric constant: {value}")
def _strict_json_loads(raw_bytes: bytes)->Any:
    return json.loads(raw_bytes,object_pairs_hook=_strict_object,parse_constant=_reject_json_constant)
def copy_json(v: Any)->Any:
    if isinstance(v,dict): return {k:copy_json(x) for k,x in v.items()}
    if isinstance(v,list): return [copy_json(x) for x in v]
    return v
def _container_kind(v: Any)->str:
    if isinstance(v,list): return "array"
    if isinstance(v,dict): return "object"
    if v is None: return "null"
    if isinstance(v,bool): return "boolean"
    if isinstance(v,(int,float)): return "number"
    if isinstance(v,str): return "string"
    return type(v).__name__
def _parse_selector(path: str)->list[tuple[str,bool]]:
    _require(isinstance(path,str) and path,"selector path must be non-empty")
    out=[]
    for part in path.split("."):
        _require(TOKEN_RE.fullmatch(part) is not None,f"unsupported selector token: {part}")
        wildcard=part.endswith("[*]"); out.append((part[:-3] if wildcard else part,wildcard))
    return out
def _shape_only(v: Any)->Any:
    if isinstance(v,list): return [_shape_only(x) if isinstance(x,(list,dict)) else None for x in v]
    if isinstance(v,dict): return {}
    raise NormalizationExecutionError("shape-only selection requires array/object")
def _merge(dst: Any,src: Any)->Any:
    if src is _MISSING: return dst
    if dst is _MISSING: return copy_json(src)
    if isinstance(dst,dict) and isinstance(src,dict):
        out=copy_json(dst)
        for k,v in src.items(): out[k]=_merge(out.get(k,_MISSING),v)
        return out
    if isinstance(dst,list) and isinstance(src,list):
        _require(len(dst)==len(src),"selector merge array cardinality mismatch")
        return [_merge(a,b) for a,b in zip(dst,src)]
    return copy_json(src)

def _select(node: Any,tokens: list[tuple[str,bool]],required: bool)->tuple[Any,dict[str,int]]:
    """hak.selector-path v1.

    Empty wildcard arrays are vacuously satisfied for required descendants.
    Existing wildcard elements must each satisfy required descendants. Optional
    missing descendants preserve array cardinality as empty objects.
    """
    if not tokens: return copy_json(node),{"matches":1,"missing":0}
    name,wildcard=tokens[0]; rest=tokens[1:]
    if not isinstance(node,dict):
        raise NormalizationExecutionError(f"selector expected object before field {name}, got {_container_kind(node)}")
    if name not in node:
        if required: raise NormalizationExecutionError(f"required selector field missing: {name}")
        return _MISSING,{"matches":0,"missing":1}
    value=node[name]
    if wildcard:
        if not isinstance(value,list):
            raise NormalizationExecutionError(f"selector wildcard {name} expected array, got {_container_kind(value)}")
        if not value: return {name:[]},{"matches":0,"missing":0}
        items=[]; matches=missing=0
        for idx,item in enumerate(value):
            try: child,c=_select(item,rest,required)
            except NormalizationExecutionError as exc:
                raise NormalizationExecutionError(f"{name}[{idx}]: {exc}") from exc
            items.append({} if child is _MISSING else child); matches+=c["matches"]; missing+=c["missing"]
        return {name:items},{"matches":matches,"missing":missing}
    child,c=_select(value,rest,required)
    return (_MISSING,c) if child is _MISSING else ({name:child},c)

def _required_selector_status(tokens: list[tuple[str,bool]], counts: dict[str,int])->str:
    if any(wildcard for _,wildcard in tokens) and counts["matches"]==0 and counts["missing"]==0:
        return "VacuouslySatisfied"
    return "Present"

def _find_profile(policy: dict[str,Any],kind: str)->dict[str,Any]:
    matches=[p for p in policy.get("resource_profiles",[]) if p.get("resource_kind")==kind]
    _require(len(matches)==1,f"policy must contain exactly one profile for {kind}")
    return matches[0]

def _base_receipt(raw_bytes: bytes,policy: dict[str,Any],resource_kind: str,raw_source_ref: str,
                  policy_artifact_ref: str,interpreter_ref: str,interpreter_bytes: bytes)->dict[str,Any]:
    return {"schema_version":RECEIPT_SCHEMA,"execution_status":"Failed","resource_kind":resource_kind,
      "raw_source":{"source_ref":raw_source_ref,"raw_response_digest":sha256_bytes(raw_bytes)},
      "policy":{"policy_id":policy.get("policy_id"),"policy_digest":policy.get("policy_digest"),"artifact_ref":policy_artifact_ref,"selector_grammar":copy_json(policy.get("selector_grammar"))},
      "interpreter":{**INTERPRETER,"artifact_ref":interpreter_ref,"content_digest":sha256_bytes(interpreter_bytes)},
      "container_results":[],"required_selector_results":[],"optional_selector_results":[],"errors":[],
      "selected_payload":{},"selected_result_digest":compute_selected_result_digest({}),"observation_issuance_permitted":False}

def execute_normalization(raw_bytes: bytes,policy: dict[str,Any],*,resource_kind: str,raw_source_ref: str,
                          policy_artifact_ref: str,interpreter_ref: str,interpreter_bytes: bytes)->tuple[dict[str,Any],int]:
    policy_lint.validate_policy(policy)
    _require(policy.get("selector_grammar")==GRAMMAR,"policy selector_grammar is not hak.selector-path v1")
    _require(interpreter_bytes==current_interpreter_bytes(),"interpreter_bytes must match the import-time HAK-014 source snapshot")
    _require(isinstance(raw_source_ref,str) and raw_source_ref,"raw_source_ref must be non-empty")
    _require(GIT_REF_RE.fullmatch(policy_artifact_ref) is not None,"policy_artifact_ref must be exact git ref")
    _require(GIT_REF_RE.fullmatch(interpreter_ref) is not None,"interpreter_ref must be exact git ref")
    profile=_find_profile(policy,resource_kind)
    receipt=_base_receipt(raw_bytes,policy,resource_kind,raw_source_ref,policy_artifact_ref,interpreter_ref,interpreter_bytes)
    try:
        raw=_strict_json_loads(raw_bytes); _require(isinstance(raw,dict),f"raw provider JSON root must be an object, got {_container_kind(raw)}")
    except (json.JSONDecodeError,UnicodeDecodeError,NormalizationExecutionError) as exc:
        receipt["errors"]=[f"raw parse/root failure: {exc}"]; receipt["receipt_digest"]=compute_receipt_digest(receipt); return receipt,2

    selected: Any={}; errors:list[str]=[]
    for spec in profile.get("required_containers",[]):
        path=spec["path"]; expected=spec["container_type"]; observed=None
        try:
            tokens=_parse_selector(path); _require(not any(w for _,w in tokens),"required container paths may not contain wildcards in v1")
            node: Any=raw
            for name,_ in tokens:
                if not isinstance(node,dict) or name not in node: raise NormalizationExecutionError(f"required container missing: {path}")
                node=node[name]
            observed=_container_kind(node); _require(observed==expected,f"required container {path} expected {expected}, got {observed}")
            projection={}; cursor=projection
            for i,(name,_) in enumerate(tokens):
                if i==len(tokens)-1: cursor[name]=_shape_only(node)
                else: cursor[name]={}; cursor=cursor[name]
            selected=_merge(selected,projection); receipt["container_results"].append({"path":path,"expected_type":expected,"observed_type":observed,"status":"Present"})
        except NormalizationExecutionError as exc:
            errors.append(str(exc)); receipt["container_results"].append({"path":path,"expected_type":expected,"observed_type":observed,"status":"Failed"})

    for path in profile.get("required_paths",[]):
        try:
            tokens=_parse_selector(path)
            projection,c=_select(raw,tokens,True); selected=_merge(selected,projection)
            receipt["required_selector_results"].append({"path":path,"status":_required_selector_status(tokens,c),**c})
        except NormalizationExecutionError as exc:
            errors.append(f"{path}: {exc}"); receipt["required_selector_results"].append({"path":path,"status":"Failed","matches":0,"missing":1})

    for path in profile.get("optional_paths",[]):
        try:
            projection,c=_select(raw,_parse_selector(path),False)
            if projection is not _MISSING: selected=_merge(selected,projection)
            receipt["optional_selector_results"].append({"path":path,"status":"Present" if c["matches"] else "Absent",**c})
        except NormalizationExecutionError as exc:
            errors.append(f"{path}: {exc}"); receipt["optional_selector_results"].append({"path":path,"status":"Failed","matches":0,"missing":0})

    status="Succeeded" if not errors else ("Partial" if bool(selected) else "Failed")
    receipt["execution_status"]=status; receipt["errors"]=errors; receipt["selected_payload"]=selected
    receipt["selected_result_digest"]=compute_selected_result_digest(selected); receipt["observation_issuance_permitted"]=status=="Succeeded"
    receipt["receipt_digest"]=compute_receipt_digest(receipt); return receipt,0 if status=="Succeeded" else 2

def _validate_selector_result(result: Any,*,required: bool)->None:
    _require(isinstance(result,dict),"selector result must be object")
    _require(isinstance(result.get("path"),str) and result["path"],"selector result path invalid")
    matches=result.get("matches"); missing=result.get("missing")
    _require(isinstance(matches,int) and matches>=0,"selector result matches invalid")
    _require(isinstance(missing,int) and missing>=0,"selector result missing invalid")
    status=result.get("status")
    if required:
        _require(status in {"Present","VacuouslySatisfied","Failed"},"required selector status invalid")
        if status=="Present": _require(matches>0,"required Present selector must contain direct matches")
        elif status=="VacuouslySatisfied": _require(matches==0 and missing==0,"VacuouslySatisfied requires zero matches and zero missing evidence")
        else: _require(missing>0,"Failed required selector must record missing evidence")
    else:
        _require(status in {"Present","Absent","Failed"},"optional selector status invalid")
        if status=="Present": _require(matches>0,"optional Present selector must contain direct matches")
        elif status=="Absent": _require(matches==0,"Absent optional selector cannot contain matches")

def validate_receipt(receipt: dict[str,Any])->None:
    _require(receipt.get("schema_version")==RECEIPT_SCHEMA,"receipt schema_version invalid")
    status=receipt.get("execution_status"); _require(status in TERMINAL,"receipt execution_status invalid")
    errors=receipt.get("errors"); _require(isinstance(errors,list) and all(isinstance(e,str) and e for e in errors),"receipt errors invalid")
    _require((status=="Succeeded")==(len(errors)==0),"Succeeded requires no errors and non-success requires errors")
    policy=receipt.get("policy"); _require(isinstance(policy,dict),"receipt policy must be object"); _require(policy.get("selector_grammar")==GRAMMAR,"receipt selector grammar mismatch")
    _require(DIGEST_RE.fullmatch(str(policy.get("policy_digest",""))) is not None,"receipt policy_digest invalid"); _require(GIT_REF_RE.fullmatch(str(policy.get("artifact_ref",""))) is not None,"receipt policy artifact_ref invalid")
    interp=receipt.get("interpreter"); _require(isinstance(interp,dict),"receipt interpreter must be object"); _require(interp.get("id")==INTERPRETER["id"] and interp.get("version")==1,"receipt interpreter identity invalid")
    _require(GIT_REF_RE.fullmatch(str(interp.get("artifact_ref",""))) is not None,"receipt interpreter artifact_ref invalid"); _require(DIGEST_RE.fullmatch(str(interp.get("content_digest",""))) is not None,"receipt interpreter content_digest invalid")
    raw=receipt.get("raw_source"); _require(isinstance(raw,dict),"receipt raw_source must be object"); _require(isinstance(raw.get("source_ref"),str) and raw["source_ref"],"receipt source_ref invalid")
    _require(DIGEST_RE.fullmatch(str(raw.get("raw_response_digest",""))) is not None,"receipt raw_response_digest invalid")
    required_results=receipt.get("required_selector_results"); optional_results=receipt.get("optional_selector_results")
    _require(isinstance(required_results,list),"required_selector_results must be array")
    _require(isinstance(optional_results,list),"optional_selector_results must be array")
    for result in required_results: _validate_selector_result(result,required=True)
    for result in optional_results: _validate_selector_result(result,required=False)
    _require(receipt.get("selected_result_digest")==compute_selected_result_digest(receipt.get("selected_payload")),"selected_result_digest mismatch")
    _require(receipt.get("receipt_digest")==compute_receipt_digest(receipt),"receipt_digest mismatch")
    permitted=receipt.get("observation_issuance_permitted"); _require(isinstance(permitted,bool),"observation_issuance_permitted must be boolean"); _require(permitted==(status=="Succeeded"),"observation issuance must require successful normalization")

def validate_receipt_against_inputs(receipt: dict[str,Any],raw_bytes: bytes,policy: dict[str,Any],interpreter_bytes: bytes,*,policy_artifact_ref: str|None=None,interpreter_ref: str|None=None,raw_source_ref: str|None=None,resource_kind: str|None=None)->None:
    validate_receipt(receipt); policy_lint.validate_policy(policy)
    _require(interpreter_bytes==current_interpreter_bytes(),"interpreter_bytes do not match import-time HAK-014 source snapshot")
    _require(receipt["raw_source"]["raw_response_digest"]==sha256_bytes(raw_bytes),"raw response digest mismatch")
    _require(receipt["policy"]["policy_id"]==policy.get("policy_id"),"receipt policy_id mismatch"); _require(receipt["policy"]["policy_digest"]==policy.get("policy_digest"),"receipt policy_digest does not match loaded policy")
    _require(receipt["policy"]["selector_grammar"]==policy.get("selector_grammar"),"receipt selector grammar does not match loaded policy")
    _require(receipt["interpreter"]["content_digest"]==sha256_bytes(interpreter_bytes),"interpreter content digest mismatch")
    expected_policy_ref=policy_artifact_ref or receipt["policy"]["artifact_ref"]
    expected_interpreter_ref=interpreter_ref or receipt["interpreter"]["artifact_ref"]
    expected_source_ref=raw_source_ref or receipt["raw_source"]["source_ref"]
    expected_kind=resource_kind or receipt["resource_kind"]
    if policy_artifact_ref is not None: _require(receipt["policy"]["artifact_ref"]==policy_artifact_ref,"receipt policy artifact_ref mismatch")
    if interpreter_ref is not None: _require(receipt["interpreter"]["artifact_ref"]==interpreter_ref,"receipt interpreter artifact_ref mismatch")
    if raw_source_ref is not None: _require(receipt["raw_source"]["source_ref"]==raw_source_ref,"receipt raw source_ref mismatch")
    if resource_kind is not None: _require(receipt["resource_kind"]==resource_kind,"receipt resource_kind mismatch")
    replayed,_=execute_normalization(raw_bytes,policy,resource_kind=expected_kind,raw_source_ref=expected_source_ref,policy_artifact_ref=expected_policy_ref,interpreter_ref=expected_interpreter_ref,interpreter_bytes=interpreter_bytes)
    _require(receipt==replayed,"receipt does not match deterministic normalization replay for the supplied inputs")

def main(argv=None)->int:
    p=argparse.ArgumentParser(); p.add_argument("--raw",type=Path,required=True); p.add_argument("--policy",type=Path,required=True); p.add_argument("--resource-kind",required=True)
    p.add_argument("--raw-source-ref",required=True); p.add_argument("--policy-ref",required=True); p.add_argument("--interpreter-ref",required=True); p.add_argument("--output",type=Path); args=p.parse_args(argv)
    try:
        raw_bytes=args.raw.read_bytes(); policy=json.loads(args.policy.read_text()); interpreter_bytes=current_interpreter_bytes()
        receipt,code=execute_normalization(raw_bytes,policy,resource_kind=args.resource_kind,raw_source_ref=args.raw_source_ref,policy_artifact_ref=args.policy_ref,interpreter_ref=args.interpreter_ref,interpreter_bytes=interpreter_bytes)
        validate_receipt_against_inputs(receipt,raw_bytes,policy,interpreter_bytes,policy_artifact_ref=args.policy_ref,interpreter_ref=args.interpreter_ref,raw_source_ref=args.raw_source_ref,resource_kind=args.resource_kind)
        text=json.dumps(receipt,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
        if args.output: args.output.write_text(text+"\n")
        else: print(text)
        return code
    except (OSError,json.JSONDecodeError,UnicodeDecodeError,NormalizationExecutionError,policy_lint.NormalizationPolicyLintError,ValueError) as exc:
        print(f"FAIL: {exc}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
