#!/usr/bin/env python3
"""Validate Symthaea mathematical retrieval index manifests v1."""

import argparse, copy, json, math, sys
from pathlib import Path

VERSION="math-retrieval-index-v1"; AUTH="MeasurementOnly"
ROOT={"version","index_id","authority","candidate_universe","representation","index","retrieval_output"}
CU={"corpus_snapshot_sha256","knowledge_boundary_sha256","candidate_eligibility_policy_sha256","candidate_set_sha256","candidate_count","exclude_query_source_item","exclude_solution_artifacts","source_identity_kind","dedup_policy","canonical_candidate_order"}
REP={"channel","representation_family","representation_sha256","normalization_contract_sha256","normalization_implementation_sha256","item_serialization_sha256","serialization_encoding","max_serialized_item_bytes","oversize_policy","truncation_policy_sha256","control_transform","control_seed","control_artifact_sha256"}
IX={"search_mode","index_build_policy_sha256","index_artifact_sha256","index_seed","top_k_supported","scoring_metric","scoring_policy_sha256","score_precision_policy_sha256","query_normalization_policy_sha256","tie_break","deterministic","approximation_validation_policy_sha256","minimum_exact_recall_at_k"}
OUT={"order_policy","source_provenance_required","item_byte_accounting","partial_item_policy"}
FAMS={"None","Lexical","CanonicalSparse","HDC"}; CHANNELS={"None","Syntax","ExactNormalForm"}
CONTROLS={"None","RandomRetrieval","ShuffledHdcVectors","PermutedChallengeAssociations"}
METRICS={"Cosine","HammingSimilarity","Jaccard","BM25","ExactMatch","RandomDeterministic","Custom"}

class V(ValueError): pass
def closed(o,k,w):
    x=set(o)-k
    if x: raise V(f"{w}: unknown fields {sorted(x)}")
def req(o,k,w):
    x=k-set(o)
    if x: raise V(f"{w}: missing fields {sorted(x)}")
def text(x,w):
    if not isinstance(x,str) or not x.strip(): raise V(f"{w}: non-empty string required")
    return x
def sha(x,w):
    x=text(x,w)
    if len(x)!=71 or not x.startswith("sha256:") or any(c not in "0123456789abcdef" for c in x[7:]): raise V(f"{w}: invalid sha256")
def integer(x,w,m=0):
    if not isinstance(x,int) or isinstance(x,bool) or x<m: raise V(f"{w}: integer >= {m} required")
def dg(s): return "sha256:"+(s.encode().hex()+"0"*64)[:64]

def validate(d):
    if not isinstance(d,dict): raise V("root: object required")
    closed(d,ROOT,"root"); req(d,ROOT,"root")
    if d["version"]!=VERSION or d["authority"]!=AUTH: raise V("root: identity/authority invariant failed")
    text(d["index_id"],"index_id")
    c=d["candidate_universe"]
    if not isinstance(c,dict): raise V("candidate_universe: object required")
    closed(c,CU,"candidate_universe"); req(c,CU,"candidate_universe")
    for k in ("corpus_snapshot_sha256","knowledge_boundary_sha256","candidate_eligibility_policy_sha256","candidate_set_sha256"): sha(c[k],f"candidate_universe.{k}")
    integer(c["candidate_count"],"candidate_universe.candidate_count",1)
    if c["exclude_query_source_item"] is not True or c["exclude_solution_artifacts"] is not True: raise V("candidate_universe: self/solution exclusion required")
    if c["source_identity_kind"]!="SourceObjectDigest" or c["dedup_policy"]!="OneEntryPerSourceObject" or c["canonical_candidate_order"]!="SourceObjectDigestAscending": raise V("candidate_universe: identity/dedup/order invariant failed")
    r=d["representation"]
    if not isinstance(r,dict): raise V("representation: object required")
    closed(r,REP,"representation")
    base={"channel","representation_family","representation_sha256","item_serialization_sha256","serialization_encoding","max_serialized_item_bytes","oversize_policy","control_transform"}
    req(r,base,"representation")
    if r["channel"] not in CHANNELS or r["representation_family"] not in FAMS or r["control_transform"] not in CONTROLS: raise V("representation: unsupported enum")
    for k in ("representation_sha256","item_serialization_sha256"): sha(r[k],f"representation.{k}")
    if r["serialization_encoding"]!="UTF-8": raise V("representation: UTF-8 required")
    integer(r["max_serialized_item_bytes"],"representation.max_serialized_item_bytes",1)
    if r["oversize_policy"] not in {"RejectItem","DeterministicTruncate"}: raise V("representation: unsupported oversize policy")
    if r["oversize_policy"]=="DeterministicTruncate":
        if "truncation_policy_sha256" not in r: raise V("representation: truncation policy digest required")
        sha(r["truncation_policy_sha256"],"representation.truncation_policy_sha256")
    elif "truncation_policy_sha256" in r: raise V("representation: reject policy must not carry truncation policy")
    ch=r["channel"]; fam=r["representation_family"]
    if fam=="None" and ch!="None": raise V("representation: None family requires None channel")
    if ch=="None" and fam!="None": raise V("representation: None channel requires None family")
    if fam=="Lexical" and ch!="Syntax": raise V("representation: lexical requires syntax")
    if ch=="ExactNormalForm":
        for k in ("normalization_contract_sha256","normalization_implementation_sha256"):
            if k not in r: raise V("representation: exact normal form requires normalizer binding")
            sha(r[k],f"representation.{k}")
    elif "normalization_contract_sha256" in r or "normalization_implementation_sha256" in r:
        raise V("representation: syntax/random index must not carry normalizer binding")
    ctrl=r["control_transform"]
    if ctrl=="None":
        if "control_seed" in r or "control_artifact_sha256" in r: raise V("representation: non-control index must not carry control artifact")
    else:
        if "control_seed" not in r or "control_artifact_sha256" not in r: raise V("representation: control seed/artifact required")
        integer(r["control_seed"],"representation.control_seed"); sha(r["control_artifact_sha256"],"representation.control_artifact_sha256")
    if ctrl=="RandomRetrieval" and not(ch=="None" and fam=="None"): raise V("representation: random retrieval must not consume representation")
    if ctrl=="ShuffledHdcVectors" and fam!="HDC": raise V("representation: shuffled vectors require HDC family")
    i=d["index"]
    if not isinstance(i,dict): raise V("index: object required")
    closed(i,IX,"index")
    basei={"search_mode","index_build_policy_sha256","index_artifact_sha256","index_seed","top_k_supported","scoring_metric","scoring_policy_sha256","score_precision_policy_sha256","query_normalization_policy_sha256","tie_break","deterministic"}
    req(i,basei,"index")
    for k in ("index_build_policy_sha256","index_artifact_sha256","scoring_policy_sha256","score_precision_policy_sha256","query_normalization_policy_sha256"): sha(i[k],f"index.{k}")
    integer(i["index_seed"],"index.index_seed"); integer(i["top_k_supported"],"index.top_k_supported",1)
    if i["top_k_supported"]>c["candidate_count"]: raise V("index: top_k exceeds candidate count")
    if i["scoring_metric"] not in METRICS: raise V("index: unsupported scoring metric")
    if i["tie_break"]!="SourceObjectDigestAscending" or i["deterministic"] is not True: raise V("index: deterministic tie break required")
    if i["search_mode"]=="ApproximateDeterministic":
        for k in ("approximation_validation_policy_sha256","minimum_exact_recall_at_k"):
            if k not in i: raise V("index: approximate search requires validation policy + recall floor")
        sha(i["approximation_validation_policy_sha256"],"index.approximation_validation_policy_sha256")
        q=i["minimum_exact_recall_at_k"]
        if isinstance(q,bool) or not isinstance(q,(int,float)) or not math.isfinite(q) or not 0<q<=1: raise V("index: exact-recall floor must be in (0,1]")
    elif i["search_mode"]=="ExactDeterministic":
        if "approximation_validation_policy_sha256" in i or "minimum_exact_recall_at_k" in i: raise V("index: exact search must not carry ANN validation")
    else: raise V("index: unsupported search mode")
    if ctrl=="RandomRetrieval" and i["scoring_metric"]!="RandomDeterministic": raise V("index: random control requires RandomDeterministic scoring")
    o=d["retrieval_output"]
    if not isinstance(o,dict): raise V("retrieval_output: object required")
    closed(o,OUT,"retrieval_output"); req(o,OUT,"retrieval_output")
    if o!={"order_policy":"ScoreDescendingThenSourceObjectDigest","source_provenance_required":True,"item_byte_accounting":"CanonicalUtf8Bytes","partial_item_policy":"RejectWholeItem"}: raise V("retrieval_output: canonical output policy required")

def fixture():
    return {"version":VERSION,"index_id":"fixture","authority":AUTH,"candidate_universe":{"corpus_snapshot_sha256":dg("corpus"),"knowledge_boundary_sha256":dg("knowledge"),"candidate_eligibility_policy_sha256":dg("eligibility"),"candidate_set_sha256":dg("candidates"),"candidate_count":100,"exclude_query_source_item":True,"exclude_solution_artifacts":True,"source_identity_kind":"SourceObjectDigest","dedup_policy":"OneEntryPerSourceObject","canonical_candidate_order":"SourceObjectDigestAscending"},"representation":{"channel":"Syntax","representation_family":"CanonicalSparse","representation_sha256":dg("rep"),"item_serialization_sha256":dg("serialization"),"serialization_encoding":"UTF-8","max_serialized_item_bytes":8192,"oversize_policy":"RejectItem","control_transform":"None"},"index":{"search_mode":"ExactDeterministic","index_build_policy_sha256":dg("build"),"index_artifact_sha256":dg("artifact"),"index_seed":0,"top_k_supported":16,"scoring_metric":"Cosine","scoring_policy_sha256":dg("score"),"score_precision_policy_sha256":dg("precision"),"query_normalization_policy_sha256":dg("query"),"tie_break":"SourceObjectDigestAscending","deterministic":True},"retrieval_output":{"order_policy":"ScoreDescendingThenSourceObjectDigest","source_provenance_required":True,"item_byte_accounting":"CanonicalUtf8Bytes","partial_item_policy":"RejectWholeItem"}}

def self_test():
    v=fixture(); validate(v)
    attacks=[lambda d:d["candidate_universe"].__setitem__("exclude_solution_artifacts",False),lambda d:d["candidate_universe"].__setitem__("dedup_policy","KeepAll"),lambda d:d["representation"].__setitem__("normalization_implementation_sha256",dg("leak")),lambda d:d["index"].__setitem__("tie_break","InsertionOrder"),lambda d:d["index"].__setitem__("top_k_supported",1000),lambda d:d["representation"].__setitem__("oversize_policy","DeterministicTruncate"),lambda d:d["index"].__setitem__("search_mode","ApproximateDeterministic")]
    for f in attacks:
        x=copy.deepcopy(v); f(x)
        try: validate(x)
        except V: continue
        raise AssertionError("adversarial self-test unexpectedly passed")
    n=copy.deepcopy(v); n["representation"].update({"channel":"ExactNormalForm","normalization_contract_sha256":dg("nc"),"normalization_implementation_sha256":dg("ni")}); validate(n)
    a=copy.deepcopy(v); a["index"].update({"search_mode":"ApproximateDeterministic","approximation_validation_policy_sha256":dg("ann"),"minimum_exact_recall_at_k":0.98}); validate(a)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("path",nargs="?",type=Path); ap.add_argument("--self-test",action="store_true"); z=ap.parse_args()
    if z.self_test: self_test(); print("math-retrieval index v1 self-test: PASS"); return 0
    if z.path is None: ap.error("path required unless --self-test")
    try: validate(json.loads(z.path.read_text()))
    except (OSError,json.JSONDecodeError,V) as e: print(f"INVALID: {e}",file=sys.stderr); return 1
    print("VALID"); return 0
if __name__=="__main__": raise SystemExit(main())
