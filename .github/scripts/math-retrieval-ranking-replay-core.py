#!/usr/bin/env python3
"""Pure deterministic scoring primitives and closed contracts for MATH-RET-RANK-REPLAY-001A."""
from __future__ import annotations
import functools, hashlib, json

RECEIPT_VERSION="math-retrieval-ranking-replay-receipt-v1"
POLICY_VERSION="math-retrieval-scoring-replay-policy-v1"
WITNESS_VERSION="math-retrieval-ranking-replay-witness-v1"
REPORT_VERSION="math-retrieval-ranking-replay-validation-report-v1"
AUTHORITY="MeasurementOnly"
RECEIPT_FIELDS={
 "version","receipt_id","authority","trace_selector","scoring_policy_bundle_sha256",
 "query_representation_receipt_sha256","query_representation_report_sha256",
 "index_build_report_sha256","index_manifest_sha256","index_artifact_sha256",
 "trace_sha256","trace_validation_report_sha256",
}
POLICY_FIELDS={"version","authority","wire_kind","scoring","precision","query_normalization"}

class ValidationError(ValueError): pass

def dg(b:bytes)->str:return "sha256:"+hashlib.sha256(b).hexdigest()
def cb(o)->bytes:return (json.dumps(o,sort_keys=True,separators=(",",":"))+"\n").encode()
def sh(x,w):
 if not isinstance(x,str) or len(x)!=71 or not x.startswith("sha256:") or any(c not in "0123456789abcdef" for c in x[7:]):raise ValidationError(f"{w}: invalid sha256")
 return x
def text(x,w):
 if not isinstance(x,str) or not x.strip():raise ValidationError(f"{w}: non-empty string required")
 return x
def posint(x,w):
 if not isinstance(x,int) or isinstance(x,bool) or x<1:raise ValidationError(f"{w}: positive integer required")
 return x
def closed(o,keys,w):
 if not isinstance(o,dict) or set(o)!=set(keys):raise ValidationError(f"{w}: exact fields required")
 return o

def validate_receipt(d):
 closed(d,RECEIPT_FIELDS,"receipt")
 if d["version"]!=RECEIPT_VERSION or d["authority"]!=AUTHORITY:raise ValidationError("receipt: version/authority invariant failed")
 text(d["receipt_id"],"receipt.receipt_id")
 if d["trace_selector"] not in {"SingleIndex","Syntax","ExactNormalForm"}:raise ValidationError("receipt.trace_selector unsupported")
 for k in RECEIPT_FIELDS-{"version","receipt_id","authority","trace_selector"}:sh(d[k],f"receipt.{k}")
 return d

def validate_policy(p):
 closed(p,POLICY_FIELDS,"policy")
 if p["version"]!=POLICY_VERSION or p["authority"]!=AUTHORITY:raise ValidationError("policy: version/authority invariant failed")
 wire=p["wire_kind"]
 if wire not in {"CanonicalSparseV1","BinaryHV16K"}:raise ValidationError("policy.wire_kind unsupported")
 s=p["scoring"];closed(s,{"version","metric","ordering","candidate_population","query_source_exclusion","threshold_policy","tie_break","top_k_policy"},"policy.scoring")
 if s["version"]!="math-retrieval-scoring-semantics-v1" or s["candidate_population"]!="AllEligibleFromExactIndex" or s["query_source_exclusion"]!="Required" or s["threshold_policy"]!="None" or s["tie_break"]!="SourceObjectDigestAscending" or s["top_k_policy"]!="MinRequestedKEligibleCount":raise ValidationError("policy.scoring invariant failed")
 pr=p["precision"];closed(pr,{"version","arithmetic","floating_point_ordering","score_display"},"policy.precision")
 if pr["version"]!="math-retrieval-score-precision-semantics-v1" or pr["floating_point_ordering"]!="Forbidden" or pr["score_display"]!="NonAuthoritative":raise ValidationError("policy.precision invariant failed")
 q=p["query_normalization"];closed(q,{"version","mode","transform"},"policy.query_normalization")
 if q!={"version":"math-retrieval-query-normalization-semantics-v1","mode":"CanonicalWireValidationOnly","transform":"None"}:raise ValidationError("policy.query_normalization invariant failed")
 if wire=="CanonicalSparseV1":
  if s["metric"]!="Cosine" or s["ordering"]!="SparseCosineExactOrder" or pr["arithmetic"]!="ExactIntegerCrossProduct":raise ValidationError("policy: sparse semantics mismatch")
 else:
  if s["metric"]!="HammingSimilarity" or s["ordering"]!="HammingDistanceAscending" or pr["arithmetic"]!="ExactIntegerPopcount":raise ValidationError("policy: HDC semantics mismatch")
 return p

def component_digest(o):return dg(cb(o))

def validate_qreport(r):
 keys={"version","authority","receipt_sha256","query_id","query_source_object_sha256","input_stage","input_object_sha256","representation_sha256","item_serialization_sha256","wire_kind","query_wire_sha256","serialized_bytes","all_checks_passed"}
 closed(r,keys,"query_report")
 if r["version"]!="math-retrieval-query-representation-validation-report-v1" or r["authority"]!=AUTHORITY or r["all_checks_passed"] is not True:raise ValidationError("query_report: PASS identity required")
 for k in {"receipt_sha256","query_source_object_sha256","input_object_sha256","representation_sha256","item_serialization_sha256","query_wire_sha256"}:sh(r[k],f"query_report.{k}")
 posint(r["serialized_bytes"],"query_report.serialized_bytes")
 return r

def validate_build_report(r):
 keys={"version","authority","receipt_sha256","coverage_sha256","candidate_set_sha256","index_manifest_sha256","index_artifact_sha256","input_set_sha256","item_count","target_id","all_payload_hashes_match","all_coverage_rows_match","all_checks_passed"}
 closed(r,keys,"build_report")
 if r["version"]!="math-retrieval-index-build-validation-report-v1" or r["authority"]!=AUTHORITY or any(r[k] is not True for k in ("all_payload_hashes_match","all_coverage_rows_match","all_checks_passed")):raise ValidationError("build_report: PASS identity required")
 for k in {"receipt_sha256","coverage_sha256","candidate_set_sha256","index_manifest_sha256","index_artifact_sha256","input_set_sha256"}:sh(r[k],f"build_report.{k}")
 posint(r["item_count"],"build_report.item_count");text(r["target_id"],"build_report.target_id")
 return r

def validate_trace_report(r):
 keys={"version","authority","trace_id","trace_sha256","graph_report_sha256","experiment_sha256","arm_id","experiment_seed","packed_output_items","packed_output_bytes","all_checks_passed"}
 closed(r,keys,"trace_report")
 if r["version"]!="math-retrieval-trace-validation-report-v1" or r["authority"]!=AUTHORITY or r["all_checks_passed"] is not True:raise ValidationError("trace_report: PASS identity required")
 for k in {"trace_sha256","graph_report_sha256","experiment_sha256"}:sh(r[k],f"trace_report.{k}")
 return r

def select_trace(trace,selector):
 q=trace.get("query")
 if not isinstance(q,dict):raise ValidationError("trace.query required")
 qid=text(q.get("query_id"),"trace.query.query_id");qs=sh(q.get("query_source_object_sha256"),"trace.query.query_source_object_sha256")
 r=trace.get("retrieval")
 if not isinstance(r,dict):raise ValidationError("trace.retrieval required")
 if selector=="SingleIndex":
  if r.get("mode")!="SingleIndex" or not isinstance(r.get("single"),dict):raise ValidationError("trace selector SingleIndex does not match trace")
  x=r["single"];idx=sh(x.get("index_manifest_sha256"),"trace.single.index_manifest_sha256");art=sh(x.get("index_artifact_sha256"),"trace.single.index_artifact_sha256");k=posint(x.get("requested_k"),"trace.single.requested_k");rank=x.get("ranked_source_object_digests")
 else:
  if r.get("mode")!="Fusion" or not isinstance(r.get("fusion"),dict):raise ValidationError("fusion trace required for channel selector")
  xs=r["fusion"].get("channels")
  if not isinstance(xs,list):raise ValidationError("trace fusion channels required")
  x=next((z for z in xs if isinstance(z,dict) and z.get("channel")==selector),None)
  if x is None:raise ValidationError("requested trace channel absent")
  idx=sh(x.get("index_manifest_sha256"),"trace.channel.index_manifest_sha256");art=sh(x.get("index_artifact_sha256"),"trace.channel.index_artifact_sha256");k=posint(x.get("requested_k"),"trace.channel.requested_k");rank=x.get("ranked_source_object_digests")
 if not isinstance(rank,list):raise ValidationError("runtime ranking list required")
 for i,d in enumerate(rank):sh(d,f"runtime_rank[{i}]")
 if len(rank)!=len(set(rank)):raise ValidationError("runtime ranking contains duplicate source")
 return qid,qs,idx,art,k,rank

def sparse_map(doc):return {x["feature_id"]:x["count"] for x in doc["features"]}
def sparse_witness(q,c):
 qm=sparse_map(q);cm=sparse_map(c);dot=sum(v*cm.get(k,0) for k,v in qm.items());norm=sum(v*v for v in cm.values());return dot,norm
def sparse_cmp(a,b):
 left=a[1]*a[1]*b[2];right=b[1]*b[1]*a[2]
 if left!=right:return -1 if left>right else 1
 return -1 if a[0]<b[0] else (1 if a[0]>b[0] else 0)
def sort_sparse(scored):
 out=list(scored);out.sort(key=functools.cmp_to_key(sparse_cmp));return out
def hamming(qb,cb):return sum((a^b).bit_count() for a,b in zip(qb,cb))

def policy_fixture(kind):
 sparse=kind=="CanonicalSparseV1"
 return {"version":POLICY_VERSION,"authority":AUTHORITY,"wire_kind":kind,
  "scoring":{"version":"math-retrieval-scoring-semantics-v1","metric":"Cosine" if sparse else "HammingSimilarity","ordering":"SparseCosineExactOrder" if sparse else "HammingDistanceAscending","candidate_population":"AllEligibleFromExactIndex","query_source_exclusion":"Required","threshold_policy":"None","tie_break":"SourceObjectDigestAscending","top_k_policy":"MinRequestedKEligibleCount"},
  "precision":{"version":"math-retrieval-score-precision-semantics-v1","arithmetic":"ExactIntegerCrossProduct" if sparse else "ExactIntegerPopcount","floating_point_ordering":"Forbidden","score_display":"NonAuthoritative"},
  "query_normalization":{"version":"math-retrieval-query-normalization-semantics-v1","mode":"CanonicalWireValidationOnly","transform":"None"}}
