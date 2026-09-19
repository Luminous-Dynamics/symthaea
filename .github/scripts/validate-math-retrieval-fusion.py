#!/usr/bin/env python3
"""Validate Symthaea mathematical retrieval fusion policies v1."""

import argparse, copy, json, sys
from pathlib import Path

VERSION="math-retrieval-fusion-v1"; AUTH="MeasurementOnly"
ROOT={"version","fusion_id","authority","inputs","fusion","global_budget","output"}
IN={"channel","index_manifest_sha256","max_input_items","max_input_bytes","rank_weight"}
FU={"method","rrf_k","tie_break","dedup_key","duplicate_merge_policy","channel_underfill_policy","unused_quota_reallocation","deterministic"}
BU={"max_output_items","max_output_bytes","max_output_item_bytes","byte_accounting","partial_item_policy","packing_policy"}
OUT={"payload_kind","payload_serialization_sha256","source_provenance_required","channel_provenance_required","duplicate_source_counts_once"}

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
def integer(x,w,m=1):
    if not isinstance(x,int) or isinstance(x,bool) or x<m: raise V(f"{w}: integer >= {m} required")
def dg(s): return "sha256:"+(s.encode().hex()+"0"*64)[:64]

def validate(d):
    if not isinstance(d,dict): raise V("root: object required")
    closed(d,ROOT,"root"); req(d,ROOT,"root")
    if d["version"]!=VERSION or d["authority"]!=AUTH: raise V("root: identity/authority invariant failed")
    text(d["fusion_id"],"fusion_id")
    xs=d["inputs"]
    if not isinstance(xs,list) or len(xs)!=2: raise V("inputs: exactly two channels required")
    seen=set(); item_sum=0; byte_sum=0
    for j,x in enumerate(xs):
        w=f"inputs[{j}]"
        if not isinstance(x,dict): raise V(f"{w}: object required")
        closed(x,IN,w); req(x,IN,w)
        if x["channel"] not in {"Syntax","ExactNormalForm"} or x["channel"] in seen: raise V(f"{w}: unique Syntax + ExactNormalForm required")
        seen.add(x["channel"]); sha(x["index_manifest_sha256"],f"{w}.index_manifest_sha256")
        integer(x["max_input_items"],f"{w}.max_input_items"); integer(x["max_input_bytes"],f"{w}.max_input_bytes")
        if x["rank_weight"]!=1: raise V(f"{w}: v1 requires equal rank_weight=1")
        item_sum+=x["max_input_items"]; byte_sum+=x["max_input_bytes"]
    if seen!={"Syntax","ExactNormalForm"}: raise V("inputs: Syntax and ExactNormalForm required")
    f=d["fusion"]
    if not isinstance(f,dict): raise V("fusion: object required")
    closed(f,FU,"fusion")
    base={"method","tie_break","dedup_key","duplicate_merge_policy","channel_underfill_policy","unused_quota_reallocation","deterministic"}
    req(f,base,"fusion")
    if f["method"] not in {"ReciprocalRankFusion","DeterministicInterleave"}: raise V("fusion: unsupported method")
    if f["method"]=="ReciprocalRankFusion":
        if "rrf_k" not in f: raise V("fusion: RRF requires rrf_k")
        integer(f["rrf_k"],"fusion.rrf_k")
    elif "rrf_k" in f: raise V("fusion: interleave must not carry rrf_k")
    if f["tie_break"]!="SourceObjectDigestAscending" or f["dedup_key"]!="SourceObjectDigest" or f["duplicate_merge_policy"]!="OneOutputUnionChannelProvenance": raise V("fusion: deterministic source dedup/tie policy required")
    if f["channel_underfill_policy"]!="NoReallocation" or f["unused_quota_reallocation"] is not False: raise V("fusion: adaptive quota refill forbidden in v1")
    if f["deterministic"] is not True: raise V("fusion: deterministic=true required")
    b=d["global_budget"]
    if not isinstance(b,dict): raise V("global_budget: object required")
    closed(b,BU,"global_budget"); req(b,BU,"global_budget")
    for k in ("max_output_items","max_output_bytes","max_output_item_bytes"): integer(b[k],f"global_budget.{k}")
    if b["max_output_item_bytes"]>b["max_output_bytes"]: raise V("global_budget: one item cannot exceed total output bytes")
    if item_sum>b["max_output_items"]: raise V("global_budget: aggregate channel item quotas exceed global output item ceiling")
    if byte_sum>b["max_output_bytes"]: raise V("global_budget: aggregate channel byte quotas exceed global output byte ceiling")
    if b["byte_accounting"]!="CanonicalUtf8Bytes" or b["partial_item_policy"]!="RejectWholeItem" or b["packing_policy"]!="FusedRankOrderWholeItems": raise V("global_budget: canonical whole-item packing required")
    o=d["output"]
    if not isinstance(o,dict): raise V("output: object required")
    closed(o,OUT,"output"); req(o,OUT,"output")
    sha(o["payload_serialization_sha256"],"output.payload_serialization_sha256")
    if o["payload_kind"]!="CanonicalSourceObject" or o["source_provenance_required"] is not True or o["channel_provenance_required"] is not True or o["duplicate_source_counts_once"] is not True: raise V("output: canonical source payload + provenance/dedup required")

def fixture():
    return {"version":VERSION,"fusion_id":"fixture","authority":AUTH,"inputs":[{"channel":"Syntax","index_manifest_sha256":dg("s"),"max_input_items":4,"max_input_bytes":16000,"rank_weight":1},{"channel":"ExactNormalForm","index_manifest_sha256":dg("n"),"max_input_items":4,"max_input_bytes":16000,"rank_weight":1}],"fusion":{"method":"ReciprocalRankFusion","rrf_k":60,"tie_break":"SourceObjectDigestAscending","dedup_key":"SourceObjectDigest","duplicate_merge_policy":"OneOutputUnionChannelProvenance","channel_underfill_policy":"NoReallocation","unused_quota_reallocation":False,"deterministic":True},"global_budget":{"max_output_items":8,"max_output_bytes":32000,"max_output_item_bytes":8000,"byte_accounting":"CanonicalUtf8Bytes","partial_item_policy":"RejectWholeItem","packing_policy":"FusedRankOrderWholeItems"},"output":{"payload_kind":"CanonicalSourceObject","payload_serialization_sha256":dg("payload"),"source_provenance_required":True,"channel_provenance_required":True,"duplicate_source_counts_once":True}}

def self_test():
    v=fixture(); validate(v)
    attacks=[lambda d:d["inputs"][0].__setitem__("max_input_items",8),lambda d:d["inputs"][0].__setitem__("rank_weight",2),lambda d:d["fusion"].__setitem__("channel_underfill_policy","FillFromOtherChannel"),lambda d:d["fusion"].__setitem__("tie_break","InsertionOrder"),lambda d:d["output"].__setitem__("payload_kind","ChannelRepresentation"),lambda d:d["output"].__setitem__("duplicate_source_counts_once",False),lambda d:d["global_budget"].__setitem__("partial_item_policy","TruncateFinalItem")]
    for f in attacks:
        x=copy.deepcopy(v); f(x)
        try: validate(x)
        except V: continue
        raise AssertionError("adversarial self-test unexpectedly passed")
    z=copy.deepcopy(v); z["fusion"]={"method":"DeterministicInterleave","tie_break":"SourceObjectDigestAscending","dedup_key":"SourceObjectDigest","duplicate_merge_policy":"OneOutputUnionChannelProvenance","channel_underfill_policy":"NoReallocation","unused_quota_reallocation":False,"deterministic":True}; validate(z)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("path",nargs="?",type=Path); ap.add_argument("--self-test",action="store_true"); z=ap.parse_args()
    if z.self_test: self_test(); print("math-retrieval fusion v1 self-test: PASS"); return 0
    if z.path is None: ap.error("path required unless --self-test")
    try: validate(json.loads(z.path.read_text()))
    except (OSError,json.JSONDecodeError,V) as e: print(f"INVALID: {e}",file=sys.stderr); return 1
    print("VALID"); return 0
if __name__=="__main__": raise SystemExit(main())
