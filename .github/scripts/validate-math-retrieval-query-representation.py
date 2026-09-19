#!/usr/bin/env python3
"""Validate a content-addressed mathematical query representation receipt."""
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, json, sys
from pathlib import Path

VERSION="math-retrieval-query-representation-v1"
REPORT_VERSION="math-retrieval-query-representation-validation-report-v1"
AUTHORITY="MeasurementOnly"
REQUIRED={
    "version","receipt_id","authority","query_id","query_source_object_sha256",
    "source_object_contract_sha256","input_stage","input_object_sha256",
    "representation_sha256","item_serialization_sha256","wire_kind",
    "producer_implementation_sha256","toolchain_manifest_sha256",
    "query_wire_sha256","serialized_bytes",
}
OPTIONAL={"normalization_contract_sha256","normalization_implementation_sha256"}

class ValidationError(ValueError): pass

def dg(raw:bytes)->str:return "sha256:"+hashlib.sha256(raw).hexdigest()
def sh(x,w):
    if not isinstance(x,str) or len(x)!=71 or not x.startswith("sha256:") or any(c not in "0123456789abcdef" for c in x[7:]):
        raise ValidationError(f"{w}: invalid sha256")
    return x
def text(x,w):
    if not isinstance(x,str) or not x.strip(): raise ValidationError(f"{w}: non-empty string required")
    return x
def posint(x,w):
    if not isinstance(x,int) or isinstance(x,bool) or x<1: raise ValidationError(f"{w}: positive integer required")
    return x
def load_wire():
    p=Path(__file__).with_name("validate-math-retrieval-representation-wire.py")
    s=importlib.util.spec_from_file_location("sym_query_wire",p)
    if s is None or s.loader is None: raise ValidationError(f"cannot load wire validator: {p}")
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def validate_receipt(d):
    if not isinstance(d,dict): raise ValidationError("receipt: object required")
    extra=set(d)-REQUIRED-OPTIONAL; missing=REQUIRED-set(d)
    if extra: raise ValidationError(f"receipt: unknown fields {sorted(extra)}")
    if missing: raise ValidationError(f"receipt: missing fields {sorted(missing)}")
    if d["version"]!=VERSION or d["authority"]!=AUTHORITY: raise ValidationError("receipt: version/authority invariant failed")
    text(d["receipt_id"],"receipt.receipt_id");text(d["query_id"],"receipt.query_id")
    for k in (
        "query_source_object_sha256","source_object_contract_sha256","input_object_sha256",
        "representation_sha256","item_serialization_sha256","producer_implementation_sha256",
        "toolchain_manifest_sha256","query_wire_sha256"
    ): sh(d[k],f"receipt.{k}")
    stage=d["input_stage"]
    if stage not in {"SourceObject","ParsedFolFormulaExt","ExactNormalForm"}: raise ValidationError("receipt.input_stage unsupported")
    if d["wire_kind"] not in {"CanonicalSparseV1","BinaryHV16K"}: raise ValidationError("receipt.wire_kind unsupported")
    posint(d["serialized_bytes"],"receipt.serialized_bytes")
    norms={"normalization_contract_sha256","normalization_implementation_sha256"}
    if stage=="ExactNormalForm":
        if not norms<=set(d): raise ValidationError("ExactNormalForm input requires normalization identities")
        for k in norms: sh(d[k],f"receipt.{k}")
    elif norms&set(d):
        raise ValidationError("non-normal-form query must not carry normalization identities")
    return d

def validate_bound(receipt_doc,receipt_raw,wire_doc,wire_raw,wire_module=None):
    d=validate_receipt(receipt_doc)
    wire=wire_module or load_wire()
    try:
        if d["wire_kind"]=="CanonicalSparseV1": wire.validate_sparse(wire_doc,wire_raw)
        else: wire.validate_hdc(wire_doc,wire_raw)
    except Exception as e:
        if isinstance(e,ValidationError): raise
        raise ValidationError(f"query wire invalid: {e}") from e
    got=dg(wire_raw)
    if d["query_wire_sha256"]!=got: raise ValidationError("receipt.query_wire_sha256 differs from exact wire bytes")
    if d["serialized_bytes"]!=len(wire_raw): raise ValidationError("receipt.serialized_bytes differs from exact wire length")
    return {
        "version":REPORT_VERSION,"authority":AUTHORITY,"receipt_sha256":dg(receipt_raw),
        "query_id":d["query_id"],"query_source_object_sha256":d["query_source_object_sha256"],
        "input_stage":d["input_stage"],"input_object_sha256":d["input_object_sha256"],
        "representation_sha256":d["representation_sha256"],"item_serialization_sha256":d["item_serialization_sha256"],
        "wire_kind":d["wire_kind"],"query_wire_sha256":got,"serialized_bytes":len(wire_raw),
        "all_checks_passed":True,
    }

def _sha(label): return "sha256:"+hashlib.sha256(label.encode()).hexdigest()
class _FakeWire:
    @staticmethod
    def validate_sparse(d,raw):
        if d.get("version")!="math-canonical-sparse-wire-v1" or not raw.endswith(b"\n"): raise ValueError("bad sparse wire")
    @staticmethod
    def validate_hdc(d,raw):
        if d.get("version")!="math-binary-hdc-wire-v1" or not raw.endswith(b"\n"): raise ValueError("bad hdc wire")

def fixture():
    wire_doc={"version":"math-canonical-sparse-wire-v1","feature_order":"FeatureIdUtf8Ascending","features":[{"feature_id":"node:X","count":1}]}
    wire_raw=(json.dumps(wire_doc,sort_keys=True,separators=(",",":"))+"\n").encode()
    d={
        "version":VERSION,"receipt_id":"qrep-fixture","authority":AUTHORITY,"query_id":"q1",
        "query_source_object_sha256":_sha("source"),"source_object_contract_sha256":_sha("source-contract"),
        "input_stage":"ParsedFolFormulaExt","input_object_sha256":_sha("parsed"),
        "representation_sha256":_sha("representation"),"item_serialization_sha256":_sha("wire-policy"),
        "wire_kind":"CanonicalSparseV1","producer_implementation_sha256":_sha("producer"),
        "toolchain_manifest_sha256":_sha("toolchain"),"query_wire_sha256":dg(wire_raw),"serialized_bytes":len(wire_raw),
    }
    raw=(json.dumps(d,sort_keys=True,separators=(",",":"))+"\n").encode()
    return d,raw,wire_doc,wire_raw

def self_test():
    d,raw,w,wr=fixture(); validate_bound(d,raw,w,wr,_FakeWire)
    attacks=[
        ("wire-digest",lambda x:x.__setitem__("query_wire_sha256",_sha("wrong"))),
        ("wire-length",lambda x:x.__setitem__("serialized_bytes",1)),
        ("normalizer-leak",lambda x:x.__setitem__("normalization_contract_sha256",_sha("n"))),
        ("wire-kind",lambda x:x.__setitem__("wire_kind","Unknown")),
        ("producer",lambda x:x.__setitem__("producer_implementation_sha256","bad")),
    ]
    for name,f in attacks:
        x=copy.deepcopy(d);f(x);xr=(json.dumps(x,sort_keys=True,separators=(",",":"))+"\n").encode()
        try: validate_bound(x,xr,w,wr,_FakeWire)
        except (ValidationError,ValueError): continue
        raise AssertionError(f"attack unexpectedly passed: {name}")
    n=copy.deepcopy(d);n["input_stage"]="ExactNormalForm";n["normalization_contract_sha256"]=_sha("nc");n["normalization_implementation_sha256"]=_sha("ni")
    nr=(json.dumps(n,sort_keys=True,separators=(",",":"))+"\n").encode();validate_bound(n,nr,w,wr,_FakeWire)
    bad=copy.deepcopy(n);del bad["normalization_implementation_sha256"]
    try: validate_receipt(bad)
    except ValidationError: pass
    else: raise AssertionError("normal-form receipt without implementation passed")
    print("math retrieval query representation v1 self-test: PASS")

def main():
    p=argparse.ArgumentParser();p.add_argument("receipt",nargs="?",type=Path);p.add_argument("query_wire",nargs="?",type=Path);p.add_argument("--self-test",action="store_true");p.add_argument("--report",type=Path);a=p.parse_args()
    if a.self_test:self_test();return 0
    if a.receipt is None or a.query_wire is None:p.error("receipt and query_wire required unless --self-test")
    try:
        rr=a.receipt.read_bytes();rd=json.loads(rr.decode());wr=a.query_wire.read_bytes();wd=json.loads(wr.decode());rep=validate_bound(rd,rr,wd,wr)
    except (OSError,UnicodeDecodeError,json.JSONDecodeError,ValidationError) as e:
        print(f"INVALID: {e}",file=sys.stderr);return 1
    out=json.dumps(rep,sort_keys=True,separators=(",",":"))
    if a.report:a.report.write_text(out+"\n",encoding="utf-8")
    print(out);return 0
if __name__=="__main__":raise SystemExit(main())
