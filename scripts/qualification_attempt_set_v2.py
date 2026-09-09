#!/usr/bin/env python3
"""Validate exact V2 resource qualification attempt observations and subject-scoped sets."""

from __future__ import annotations
import argparse, hashlib, json, re, sys, unicodedata
from pathlib import Path
from typing import Any
import integration_train_manifest as train

OBSERVATION_SCHEMA="symthaea.resource-qualification-attempt-observation.v2"
SET_SCHEMA="symthaea.resource-qualification-attempt-set.v2"
PROFILE="resources.stack-qualification.v1"
OBS_DOMAIN=b"symthaea.resource-qualification-attempt-observation.v2\0"
SET_DOMAIN=b"symthaea.resource-qualification-attempt-set.v2\0"
ID_RE=re.compile(r"^sha256:[0-9a-f]{64}$")
SHA_RE=re.compile(r"^[0-9a-f]{40}$")
UTC_RE=re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
THEOREMS=("CargoLockFreshness","ResourceNamespaceBijection","ResourcePackageDiscovery","Rustfmt","CargoCheck","Clippy","Tests","DocTests")
DISPOSITIONS={"Passed","Failed","NotExecuted","OutcomeUnknown"}
TERMINALS={"Passed","LockStale","NamespaceInvalid","FormattingFailed","CompileFailed","ClippyFailed","TestsFailed","DocTestsFailed","InfrastructureUnavailable","Cancelled","OutcomeUnknown"}
FAILURE_STAGE={"LockStale":"CargoLockFreshness","NamespaceInvalid":"ResourceNamespaceBijection","FormattingFailed":"Rustfmt","CompileFailed":"CargoCheck","ClippyFailed":"Clippy","TestsFailed":"Tests","DocTestsFailed":"DocTests"}

def _canon(x): return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def _exact(d,req,opt,where):
    missing=sorted(req-set(d)); unknown=sorted(set(d)-req-opt)
    if missing: raise train.TrainManifestError(f"{where}: missing fields: {', '.join(missing)}")
    if unknown: raise train.TrainManifestError(f"{where}: unknown fields: {', '.join(unknown)}")
def _s(x,w):
    if not isinstance(x,str) or not x or x!=x.strip(): raise train.TrainManifestError(f"{w}: expected canonical non-empty string")
    if unicodedata.normalize("NFC",x)!=x: raise train.TrainManifestError(f"{w}: text must use Unicode NFC")
    if any(ord(c)<32 or ord(c)==127 for c in x): raise train.TrainManifestError(f"{w}: control character")
    return x
def _id(x,w):
    if not isinstance(x,str) or ID_RE.fullmatch(x) is None: raise train.TrainManifestError(f"{w}: expected sha256 id")
    return x
def _sha(x,w):
    if not isinstance(x,str) or SHA_RE.fullmatch(x) is None: raise train.TrainManifestError(f"{w}: expected Git SHA")
    return x
def _pos(x,w):
    if not isinstance(x,int) or isinstance(x,bool) or x<=0: raise train.TrainManifestError(f"{w}: expected positive integer")
    return x
def _utc(x,w):
    x=_s(x,w)
    if UTC_RE.fullmatch(x) is None: raise train.TrainManifestError(f"{w}: expected UTC second timestamp")
    return x
def _sorted_strings(x,w):
    if not isinstance(x,list): raise train.TrainManifestError(f"{w}: expected list")
    y=[_s(v,f"{w}[]") for v in x]
    if y!=sorted(y) or len(y)!=len(set(y)): raise train.TrainManifestError(f"{w}: must be sorted unique")
    return y

def _progress(raw):
    if not isinstance(raw,list) or len(raw)!=len(THEOREMS): raise train.TrainManifestError("observation.theorem_progress: wrong length")
    out=[]
    for i,name in enumerate(THEOREMS):
        item=raw[i]
        if not isinstance(item,dict): raise train.TrainManifestError(f"observation.theorem_progress[{i}]: expected object")
        _exact(item,{"name","disposition"},set(),f"observation.theorem_progress[{i}]")
        got=_s(item["name"],f"observation.theorem_progress[{i}].name")
        if got!=name: raise train.TrainManifestError(f"observation.theorem_progress[{i}].name: expected {name!r}")
        disp=_s(item["disposition"],f"observation.theorem_progress[{i}].disposition")
        if disp not in DISPOSITIONS: raise train.TrainManifestError(f"observation.theorem_progress[{i}].disposition: unsupported")
        out.append({"name":got,"disposition":disp})
    return out

def _consistent(terminal,progress):
    ds=[x["disposition"] for x in progress]
    if terminal=="Passed":
        if ds!=["Passed"]*len(THEOREMS): raise train.TrainManifestError("observation: Passed requires every theorem Passed")
        return
    stage=FAILURE_STAGE.get(terminal)
    if stage:
        i=THEOREMS.index(stage); expected=["Passed"]*i+["Failed"]+["NotExecuted"]*(len(THEOREMS)-i-1)
        if ds!=expected: raise train.TrainManifestError(f"observation: {terminal} inconsistent theorem progress")
        return
    phase=0; unknown=0
    for d in ds:
        if phase==0 and d=="Passed": continue
        if phase==0 and d=="OutcomeUnknown": unknown+=1; phase=1; continue
        if d=="NotExecuted": phase=1; continue
        raise train.TrainManifestError(f"observation: {terminal} inconsistent theorem progress")
    if unknown>1: raise train.TrainManifestError("observation: at most one OutcomeUnknown theorem")

def normalize(raw:Any,require_id=True):
    if not isinstance(raw,dict): raise train.TrainManifestError("observation: expected object")
    _exact(raw,{"schema","qualification_profile","provider","repository","provider_attempt","subject","execution_environment","execution","theorem_progress","failure_evidence_refs","non_claims"},{"observation_id"},"observation")
    if raw["schema"]!=OBSERVATION_SCHEMA: raise train.TrainManifestError("observation.schema: wrong schema")
    if raw["qualification_profile"]!=PROFILE: raise train.TrainManifestError("observation.qualification_profile: wrong profile")
    pa,sub,env,exe=raw["provider_attempt"],raw["subject"],raw["execution_environment"],raw["execution"]
    for n,v in (("provider_attempt",pa),("subject",sub),("execution_environment",env),("execution",exe)):
        if not isinstance(v,dict): raise train.TrainManifestError(f"observation.{n}: expected object")
    _exact(pa,{"workflow_name","run_id","run_number","job_id","pull_request"},set(),"observation.provider_attempt")
    _exact(sub,{"admission_subject_id","admission_request_id","requested_head_sha","requested_base_sha","checked_out_sha"},set(),"observation.subject")
    _exact(env,{"runner_os","runner_image","runner_image_version","rustc","target"},set(),"observation.execution_environment")
    _exact(exe,{"started_at","completed_at","terminal_disposition"},set(),"observation.execution")
    terminal=_s(exe["terminal_disposition"],"observation.execution.terminal_disposition")
    if terminal not in TERMINALS: raise train.TrainManifestError("observation.execution.terminal_disposition: unsupported")
    progress=_progress(raw["theorem_progress"]); _consistent(terminal,progress)
    out={"schema":OBSERVATION_SCHEMA,"qualification_profile":PROFILE,
         "provider":_s(raw["provider"],"observation.provider"),"repository":_s(raw["repository"],"observation.repository"),
         "provider_attempt":{"workflow_name":_s(pa["workflow_name"],"observation.provider_attempt.workflow_name"),"run_id":_pos(pa["run_id"],"observation.provider_attempt.run_id"),"run_number":_pos(pa["run_number"],"observation.provider_attempt.run_number"),"job_id":_pos(pa["job_id"],"observation.provider_attempt.job_id"),"pull_request":_pos(pa["pull_request"],"observation.provider_attempt.pull_request")},
         "subject":{"admission_subject_id":_id(sub["admission_subject_id"],"observation.subject.admission_subject_id"),"admission_request_id":_id(sub["admission_request_id"],"observation.subject.admission_request_id"),"requested_head_sha":_sha(sub["requested_head_sha"],"observation.subject.requested_head_sha"),"requested_base_sha":_sha(sub["requested_base_sha"],"observation.subject.requested_base_sha"),"checked_out_sha":_sha(sub["checked_out_sha"],"observation.subject.checked_out_sha")},
         "execution_environment":{"runner_os":_s(env["runner_os"],"observation.execution_environment.runner_os"),"runner_image":_s(env["runner_image"],"observation.execution_environment.runner_image"),"runner_image_version":_s(env["runner_image_version"],"observation.execution_environment.runner_image_version"),"rustc":_s(env["rustc"],"observation.execution_environment.rustc"),"target":_s(env["target"],"observation.execution_environment.target")},
         "execution":{"started_at":_utc(exe["started_at"],"observation.execution.started_at"),"completed_at":_utc(exe["completed_at"],"observation.execution.completed_at"),"terminal_disposition":terminal},
         "theorem_progress":progress,"failure_evidence_refs":_sorted_strings(raw["failure_evidence_refs"],"observation.failure_evidence_refs"),"non_claims":_sorted_strings(raw["non_claims"],"observation.non_claims")}
    if out["execution"]["completed_at"]<out["execution"]["started_at"]: raise train.TrainManifestError("observation.execution: completed_at precedes started_at")
    if terminal=="Passed" and out["failure_evidence_refs"]: raise train.TrainManifestError("observation: Passed cannot carry failure evidence")
    if terminal!="Passed" and not out["failure_evidence_refs"]: raise train.TrainManifestError("observation: non-Passed requires failure evidence")
    if not out["non_claims"]: raise train.TrainManifestError("observation.non_claims: must not be empty")
    oid="sha256:"+hashlib.sha256(OBS_DOMAIN+_canon(out)).hexdigest(); out["observation_id"]=oid
    if require_id and _id(raw.get("observation_id"),"observation.observation_id")!=oid: raise train.TrainManifestError(f"observation.observation_id: expected {oid}")
    return out

def compute_observation_id(raw): return normalize(raw,False)["observation_id"]

def build_attempt_set(observations):
    if not observations: raise train.TrainManifestError("attempt set: expected observations")
    vals=[normalize(x,True) for x in observations]
    subjects={x["subject"]["admission_subject_id"] for x in vals}
    if len(subjects)!=1: raise train.TrainManifestError("attempt set: different admission subjects")
    ids=[x["observation_id"] for x in vals]
    if len(ids)!=len(set(ids)): raise train.TrainManifestError("attempt set: duplicate observation_id")
    refs=[(x["provider"],x["provider_attempt"]["run_id"],x["provider_attempt"]["job_id"]) for x in vals]
    if len(refs)!=len(set(refs)): raise train.TrainManifestError("attempt set: duplicate provider run/job identity")
    out={"schema":SET_SCHEMA,"qualification_profile":PROFILE,"admission_subject_id":next(iter(subjects)),"observation_ids":sorted(ids),
         "non_claims":["does not establish chronological ordering between attempts","does not establish identical checked-out bytes across attempts","does not include source-changing repair links","does not establish qualification unless an included observation is Passed"]}
    out["set_id"]="sha256:"+hashlib.sha256(SET_DOMAIN+_canon(out)).hexdigest()
    return out

def _pairs(pairs):
    d={}
    for k,v in pairs:
        if k in d: raise train.TrainManifestError(f"duplicate JSON key: {k!r}")
        d[k]=v
    return d

def load(path):
    try:
        if path.stat().st_size>train.MAX_MANIFEST_BYTES: raise train.TrainManifestError(f"{path}: too large")
        raw=json.loads(path.read_text(encoding="utf-8"),object_pairs_hook=_pairs)
    except train.TrainManifestError: raise
    except (OSError,UnicodeDecodeError,json.JSONDecodeError) as e: raise train.TrainManifestError(f"{path}: {e}") from e
    return normalize(raw,True)
def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("observations",nargs="+",type=Path); p.add_argument("--print-normalized",action="store_true"); a=p.parse_args(argv)
    try: result=build_attempt_set([load(x) for x in a.observations])
    except train.TrainManifestError as e: print(f"qualification attempt set invalid: {e}",file=sys.stderr); return 2
    print(json.dumps(result,indent=2,ensure_ascii=False) if a.print_normalized else result["set_id"]); return 0
if __name__=="__main__": raise SystemExit(main())
