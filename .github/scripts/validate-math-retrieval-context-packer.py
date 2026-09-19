#!/usr/bin/env python3
"""Validate Symthaea mathematical retrieval context packers v1."""

import argparse, copy, json, sys
from pathlib import Path

VERSION="math-retrieval-context-packer-v1"; AUTH="MeasurementOnly"
ROOT={"version","packer_id","authority","source","visibility","packing","budget"}
SRC={"payload_kind","source_object_contract_sha256","source_fetch_policy_sha256","payload_serialization_sha256"}
VIS={"retrieval_score_visible_to_downstream","retrieval_rank_visible_to_downstream","retrieval_channel_visible_to_downstream","representation_bytes_visible_to_downstream","normal_form_visible_to_downstream","provenance_sidecar_required","provenance_sidecar_visible_to_downstream"}
PACK={"input_kind","dedup_key","order_policy","byte_accounting","partial_item_policy","nonfitting_item_policy","unused_budget_reallocation","deterministic"}
BUD={"max_output_items","max_output_bytes","max_output_item_bytes"}

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
def integer(x,w):
    if not isinstance(x,int) or isinstance(x,bool) or x<1: raise V(f"{w}: positive integer required")
def dg(s): return "sha256:"+(s.encode().hex()+"0"*64)[:64]

def validate(d):
    if not isinstance(d,dict): raise V("root: object required")
    closed(d,ROOT,"root"); req(d,ROOT,"root")
    if d["version"]!=VERSION or d["authority"]!=AUTH: raise V("root: identity/authority invariant failed")
    text(d["packer_id"],"packer_id")
    s=d["source"]
    if not isinstance(s,dict): raise V("source: object required")
    closed(s,SRC,"source"); req(s,SRC,"source")
    if s["payload_kind"]!="CanonicalSourceObject": raise V("source: canonical source-object payload required")
    for k in SRC-{"payload_kind"}: sha(s[k],f"source.{k}")
    v=d["visibility"]
    if not isinstance(v,dict): raise V("visibility: object required")
    closed(v,VIS,"visibility"); req(v,VIS,"visibility")
    for k in ("retrieval_score_visible_to_downstream","retrieval_rank_visible_to_downstream","retrieval_channel_visible_to_downstream","representation_bytes_visible_to_downstream","normal_form_visible_to_downstream","provenance_sidecar_visible_to_downstream"):
        if v[k] is not False: raise V(f"visibility.{k}: must be false")
    if v["provenance_sidecar_required"] is not True: raise V("visibility.provenance_sidecar_required: must be true")
    p=d["packing"]
    if not isinstance(p,dict): raise V("packing: object required")
    closed(p,PACK,"packing"); req(p,PACK,"packing")
    expected={"input_kind":"RankedSourceObjectDigests","dedup_key":"SourceObjectDigest","order_policy":"PreserveRetrievedRank","byte_accounting":"CanonicalUtf8Bytes","partial_item_policy":"RejectWholeItem","nonfitting_item_policy":"StopBeforeFirstNonFittingItem","unused_budget_reallocation":False,"deterministic":True}
    if p!=expected: raise V("packing: canonical deterministic whole-item policy required")
    b=d["budget"]
    if not isinstance(b,dict): raise V("budget: object required")
    closed(b,BUD,"budget"); req(b,BUD,"budget")
    for k in BUD: integer(b[k],f"budget.{k}")
    if b["max_output_item_bytes"]>b["max_output_bytes"]: raise V("budget: one item cannot exceed total output bytes")

def fixture():
    return {"version":VERSION,"packer_id":"fixture","authority":AUTH,"source":{"payload_kind":"CanonicalSourceObject","source_object_contract_sha256":dg("object"),"source_fetch_policy_sha256":dg("fetch"),"payload_serialization_sha256":dg("payload")},"visibility":{"retrieval_score_visible_to_downstream":False,"retrieval_rank_visible_to_downstream":False,"retrieval_channel_visible_to_downstream":False,"representation_bytes_visible_to_downstream":False,"normal_form_visible_to_downstream":False,"provenance_sidecar_required":True,"provenance_sidecar_visible_to_downstream":False},"packing":{"input_kind":"RankedSourceObjectDigests","dedup_key":"SourceObjectDigest","order_policy":"PreserveRetrievedRank","byte_accounting":"CanonicalUtf8Bytes","partial_item_policy":"RejectWholeItem","nonfitting_item_policy":"StopBeforeFirstNonFittingItem","unused_budget_reallocation":False,"deterministic":True},"budget":{"max_output_items":8,"max_output_bytes":32768,"max_output_item_bytes":8192}}

def self_test():
    v=fixture(); validate(v)
    attacks=[lambda d:d["visibility"].__setitem__("retrieval_score_visible_to_downstream",True),lambda d:d["visibility"].__setitem__("normal_form_visible_to_downstream",True),lambda d:d["visibility"].__setitem__("provenance_sidecar_visible_to_downstream",True),lambda d:d["packing"].__setitem__("nonfitting_item_policy","SkipAndContinue"),lambda d:d["packing"].__setitem__("unused_budget_reallocation",True),lambda d:d["budget"].__setitem__("max_output_item_bytes",999999)]
    for f in attacks:
        x=copy.deepcopy(v); f(x)
        try: validate(x)
        except V: continue
        raise AssertionError("adversarial self-test unexpectedly passed")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("path",nargs="?",type=Path); ap.add_argument("--self-test",action="store_true"); z=ap.parse_args()
    if z.self_test: self_test(); print("math-retrieval context packer v1 self-test: PASS"); return 0
    if z.path is None: ap.error("path required unless --self-test")
    try: validate(json.loads(z.path.read_text()))
    except (OSError,json.JSONDecodeError,V) as e: print(f"INVALID: {e}",file=sys.stderr); return 1
    print("VALID"); return 0
if __name__=="__main__": raise SystemExit(main())
