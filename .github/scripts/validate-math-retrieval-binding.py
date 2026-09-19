#!/usr/bin/env python3
"""Validate Symthaea mathematical retrieval bindings v1."""

import argparse, copy, json, sys
from pathlib import Path

VERSION="math-retrieval-binding-v1"; AUTH="MeasurementOnly"
ROOT={"version","binding_id","authority","arm_id","mode","context_packer_sha256","index_manifest_sha256","syntax_index_manifest_sha256","normal_form_index_manifest_sha256","fusion_policy_sha256"}

class V(ValueError): pass
def closed(o,k,w):
    x=set(o)-k
    if x: raise V(f"{w}: unknown fields {sorted(x)}")
def text(x,w):
    if not isinstance(x,str) or not x.strip(): raise V(f"{w}: non-empty string required")
    return x
def sha(x,w):
    x=text(x,w)
    if len(x)!=71 or not x.startswith("sha256:") or any(c not in "0123456789abcdef" for c in x[7:]): raise V(f"{w}: invalid sha256")
def dg(s): return "sha256:"+(s.encode().hex()+"0"*64)[:64]

def validate(d):
    if not isinstance(d,dict): raise V("root: object required")
    closed(d,ROOT,"root")
    for k in ("version","binding_id","authority","arm_id","mode","context_packer_sha256"):
        if k not in d: raise V(f"root: missing {k}")
    if d["version"]!=VERSION or d["authority"]!=AUTH: raise V("root: identity/authority invariant failed")
    text(d["binding_id"],"binding_id"); aid=text(d["arm_id"],"arm_id")
    if len(aid)>64 or not aid[0].isalpha() or any(not(c.isalnum() or c in "._-") for c in aid): raise V("arm_id: invalid identifier")
    sha(d["context_packer_sha256"],"context_packer_sha256")
    mode=d["mode"]
    if mode=="SingleIndex":
        if "index_manifest_sha256" not in d: raise V("SingleIndex: index_manifest_sha256 required")
        sha(d["index_manifest_sha256"],"index_manifest_sha256")
        for k in ("syntax_index_manifest_sha256","normal_form_index_manifest_sha256","fusion_policy_sha256"):
            if k in d: raise V(f"SingleIndex: {k} forbidden")
    elif mode=="Fusion":
        for k in ("syntax_index_manifest_sha256","normal_form_index_manifest_sha256","fusion_policy_sha256"):
            if k not in d: raise V(f"Fusion: {k} required")
            sha(d[k],k)
        if "index_manifest_sha256" in d: raise V("Fusion: singular index_manifest_sha256 forbidden")
        if d["syntax_index_manifest_sha256"]==d["normal_form_index_manifest_sha256"]: raise V("Fusion: syntax and normal-form index digests must differ")
    else:
        raise V("mode: unsupported")

def fixture(mode="SingleIndex"):
    d={"version":VERSION,"binding_id":"fixture","authority":AUTH,"arm_id":"S","mode":mode,"context_packer_sha256":dg("packer")}
    if mode=="SingleIndex": d["index_manifest_sha256"]=dg("index")
    else: d.update({"syntax_index_manifest_sha256":dg("syntax"),"normal_form_index_manifest_sha256":dg("normal"),"fusion_policy_sha256":dg("fusion")})
    return d

def self_test():
    for mode in ("SingleIndex","Fusion"): validate(fixture(mode))
    attacks=[]
    a=fixture(); attacks.append(lambda d:d.pop("index_manifest_sha256")); attacks.append(lambda d:d.__setitem__("fusion_policy_sha256",dg("leak")))
    for f in attacks:
        x=copy.deepcopy(a); f(x)
        try: validate(x)
        except V: continue
        raise AssertionError("single-index adversarial self-test unexpectedly passed")
    b=fixture("Fusion")
    attacks2=[lambda d:d.pop("fusion_policy_sha256"),lambda d:d.__setitem__("index_manifest_sha256",dg("ambiguous")),lambda d:d.__setitem__("normal_form_index_manifest_sha256",d["syntax_index_manifest_sha256"])]
    for f in attacks2:
        x=copy.deepcopy(b); f(x)
        try: validate(x)
        except V: continue
        raise AssertionError("fusion adversarial self-test unexpectedly passed")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("path",nargs="?",type=Path); ap.add_argument("--self-test",action="store_true"); z=ap.parse_args()
    if z.self_test: self_test(); print("math-retrieval binding v1 self-test: PASS"); return 0
    if z.path is None: ap.error("path required unless --self-test")
    try: validate(json.loads(z.path.read_text()))
    except (OSError,json.JSONDecodeError,V) as e: print(f"INVALID: {e}",file=sys.stderr); return 1
    print("VALID"); return 0
if __name__=="__main__": raise SystemExit(main())
