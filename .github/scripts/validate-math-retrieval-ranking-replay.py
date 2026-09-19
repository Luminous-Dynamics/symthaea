#!/usr/bin/env python3
"""Bind frozen evidence artifacts and independently replay exact retrieval ranking."""
from __future__ import annotations
import argparse, base64, importlib.util, json, sys
from pathlib import Path

def load(name,filename):
 p=Path(__file__).with_name(filename);s=importlib.util.spec_from_file_location(name,p)
 if s is None or s.loader is None:raise RuntimeError(f"cannot load {p}")
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
core=load("rank_core","math-retrieval-ranking-replay-core.py")
V=core.ValidationError

def validate_bound(receipt_doc,receipt_raw,policy_doc,policy_raw,qreceipt_doc,qreceipt_raw,qreport_doc,qreport_raw,query_doc,query_raw,build_report_doc,build_report_raw,index_doc,index_raw,artifact_doc,artifact_raw,trace_doc,trace_raw,trace_report_doc,trace_report_raw,mods=None):
 d=core.validate_receipt(receipt_doc);p=core.validate_policy(policy_doc)
 mods=mods or {
  "qrep":load("rank_qrep","validate-math-retrieval-query-representation.py"),
  "wire":load("rank_wire","validate-math-retrieval-representation-wire.py"),
  "index":load("rank_index","validate-math-retrieval-index.py"),
  "build":load("rank_build","validate-math-retrieval-index-build.py"),
 }
 raw_map={"scoring_policy_bundle_sha256":policy_raw,"query_representation_receipt_sha256":qreceipt_raw,"query_representation_report_sha256":qreport_raw,"index_build_report_sha256":build_report_raw,"index_manifest_sha256":index_raw,"index_artifact_sha256":artifact_raw,"trace_sha256":trace_raw,"trace_validation_report_sha256":trace_report_raw}
 for k,raw in raw_map.items():
  if d[k]!=core.dg(raw):raise V(f"receipt.{k} differs from exact bytes")
 qbound=mods["qrep"].validate_bound(qreceipt_doc,qreceipt_raw,query_doc,query_raw,mods["wire"])
 qr=core.validate_qreport(qreport_doc)
 if qr["receipt_sha256"]!=core.dg(qreceipt_raw) or qr["query_wire_sha256"]!=core.dg(query_raw):raise V("query report does not bind supplied query representation")
 for k in ("query_id","query_source_object_sha256","input_stage","input_object_sha256","representation_sha256","item_serialization_sha256","wire_kind","serialized_bytes"):
  if qr[k]!=qbound[k]:raise V(f"query report differs from revalidation: {k}")
 br=core.validate_build_report(build_report_doc)
 if br["index_manifest_sha256"]!=core.dg(index_raw) or br["index_artifact_sha256"]!=core.dg(artifact_raw):raise V("build report does not bind supplied index files")
 tr=core.validate_trace_report(trace_report_doc)
 if tr["trace_sha256"]!=core.dg(trace_raw):raise V("trace report does not bind supplied trace")
 mods["index"].validate(index_doc)
 artifact,items=mods["build"].validate_artifact(artifact_doc)
 if index_doc["index"]["index_artifact_sha256"]!=core.dg(artifact_raw):raise V("index manifest does not bind supplied artifact")
 if index_doc["index"]["search_mode"]!="ExactDeterministic":raise V("replay v1 requires ExactDeterministic index")
 if index_doc["index"]["scoring_policy_sha256"]!=core.component_digest(p["scoring"]):raise V("scoring_policy_sha256 differs from executable policy component")
 if index_doc["index"]["score_precision_policy_sha256"]!=core.component_digest(p["precision"]):raise V("score_precision_policy_sha256 differs from executable policy component")
 if index_doc["index"]["query_normalization_policy_sha256"]!=core.component_digest(p["query_normalization"]):raise V("query_normalization_policy_sha256 differs from executable policy component")
 rep=index_doc["representation"]
 if rep["representation_sha256"]!=qr["representation_sha256"] or rep["item_serialization_sha256"]!=qr["item_serialization_sha256"]:raise V("query/index representation identity mismatch")
 if p["wire_kind"]!=qr["wire_kind"]:raise V("scoring policy wire kind differs from query receipt")
 if p["wire_kind"]=="CanonicalSparseV1":
  if rep["representation_family"]!="CanonicalSparse" or index_doc["index"]["scoring_metric"]!="Cosine":raise V("sparse replay requires CanonicalSparse + Cosine")
 else:
  if rep["representation_family"]!="HDC" or index_doc["index"]["scoring_metric"]!="HammingSimilarity":raise V("HDC replay requires HDC + HammingSimilarity")
 if rep["channel"]=="ExactNormalForm":
  if qreceipt_doc["input_stage"]!="ExactNormalForm":raise V("normal-form index requires normal-form query representation")
  for k in ("normalization_contract_sha256","normalization_implementation_sha256"):
   if rep.get(k)!=qreceipt_doc.get(k):raise V(f"normal-form query/index mismatch: {k}")
 qid,qs,idx,art,k,runtime=core.select_trace(trace_doc,d["trace_selector"])
 if qid!=qreceipt_doc["query_id"] or qs!=qreceipt_doc["query_source_object_sha256"]:raise V("trace query identity differs from query receipt")
 if idx!=core.dg(index_raw) or art!=core.dg(artifact_raw):raise V("trace selected index differs from supplied exact index")
 if tr["trace_id"]!=trace_doc.get("trace_id"):raise V("trace report trace_id differs from trace")
 wire=mods["wire"]
 if p["wire_kind"]=="CanonicalSparseV1":
  wire.validate_sparse(query_doc,query_raw);scored=[]
  for item in items:
   source=item["source_object_sha256"]
   if source==qs:continue
   raw=base64.b64decode(item["payload_base64"],validate=True);doc=json.loads(raw.decode());wire.validate_sparse(doc,raw)
   dot,norm=core.sparse_witness(query_doc,doc);scored.append((source,dot,norm))
  scored=core.sort_sparse(scored);full=[x[0] for x in scored]
  entries=[{"rank":i+1,"source_object_sha256":x[0],"score_witness":{"kind":"SparseCosineExact","dot":x[1],"candidate_norm_sq":x[2]}} for i,x in enumerate(scored)]
 else:
  wire.validate_hdc(query_doc,query_raw);qb=base64.b64decode(query_doc["bytes_base64"],validate=True);scored=[]
  for item in items:
   source=item["source_object_sha256"]
   if source==qs:continue
   raw=base64.b64decode(item["payload_base64"],validate=True);doc=json.loads(raw.decode());wire.validate_hdc(doc,raw);cbits=base64.b64decode(doc["bytes_base64"],validate=True)
   scored.append((source,core.hamming(qb,cbits)))
  scored.sort(key=lambda x:(x[1],x[0]));full=[x[0] for x in scored]
  entries=[{"rank":i+1,"source_object_sha256":x[0],"score_witness":{"kind":"BinaryHammingExact","hamming_distance":x[1]}} for i,x in enumerate(scored)]
 expected=full[:min(k,len(full))]
 if len(runtime)!=min(k,len(full)):raise V("runtime returned fewer/more than exact min(requested_k, eligible_count)")
 if runtime!=expected:raise V("runtime top-k differs from independent exact scoring replay")
 witness={"version":core.WITNESS_VERSION,"authority":core.AUTHORITY,"metric":p["scoring"]["metric"],"query_wire_sha256":core.dg(query_raw),"entries":entries};wraw=core.cb(witness)
 report={"version":core.REPORT_VERSION,"authority":core.AUTHORITY,"receipt_sha256":core.dg(receipt_raw),"trace_sha256":core.dg(trace_raw),"trace_selector":d["trace_selector"],"metric":p["scoring"]["metric"],"query_representation_report_sha256":core.dg(qreport_raw),"index_build_report_sha256":core.dg(build_report_raw),"scoring_policy_bundle_sha256":core.dg(policy_raw),"requested_k":k,"eligible_count":len(full),"expected_top_k":expected,"runtime_top_k":runtime,"full_ranking_sha256":core.dg(core.cb(full)),"witness_sha256":core.dg(wraw),"all_checks_passed":True}
 return report,witness

def main():
 p=argparse.ArgumentParser()
 names=("receipt","policy","query_receipt","query_report","query_wire","build_report","index_manifest","index_artifact","trace","trace_report")
 for name in names:p.add_argument(name,nargs="?",type=Path)
 p.add_argument("--report",type=Path);p.add_argument("--witness",type=Path);a=p.parse_args()
 vals=[getattr(a,n) for n in names]
 if any(x is None for x in vals):p.error("all ten artifacts are required")
 try:
  docs=[];raws=[]
  for path in vals:
   raw=path.read_bytes();docs.append(json.loads(raw.decode()));raws.append(raw)
  rep,wit=validate_bound(docs[0],raws[0],docs[1],raws[1],docs[2],raws[2],docs[3],raws[3],docs[4],raws[4],docs[5],raws[5],docs[6],raws[6],docs[7],raws[7],docs[8],raws[8],docs[9],raws[9])
 except (OSError,UnicodeDecodeError,json.JSONDecodeError,V,ValueError) as e:
  print(f"INVALID: {e}",file=sys.stderr);return 1
 out=json.dumps(rep,sort_keys=True,separators=(",",":"))
 if a.report:a.report.write_text(out+"\n",encoding="utf-8")
 if a.witness:a.witness.write_bytes(core.cb(wit))
 print(out);return 0
if __name__=="__main__":raise SystemExit(main())
