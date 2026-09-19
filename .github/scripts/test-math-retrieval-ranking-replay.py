#!/usr/bin/env python3
"""Adversarial integration tests for exact ranking replay."""
from __future__ import annotations
import base64, copy, hashlib, importlib.util, json
from pathlib import Path

def load(name,filename):
 p=Path(__file__).with_name(filename);s=importlib.util.spec_from_file_location(name,p)
 if s is None or s.loader is None:raise RuntimeError(p)
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
core=load("rank_core_test","math-retrieval-ranking-replay-core.py")
val=load("rank_validator_test","validate-math-retrieval-ranking-replay.py")
def sha(s):return "sha256:"+hashlib.sha256(s.encode()).hexdigest()

class FW:
 @staticmethod
 def validate_sparse(d,raw):
  if d.get("version")!="math-canonical-sparse-wire-v1" or raw!=core.cb(d):raise ValueError("bad sparse")
 @staticmethod
 def validate_hdc(d,raw):
  if d.get("version")!="math-binary-hdc-wire-v1" or raw!=core.cb(d):raise ValueError("bad hdc")
class FQ:
 @staticmethod
 def validate_bound(rd,rr,wd,wr,wire):
  wire.validate_sparse(wd,wr) if rd["wire_kind"]=="CanonicalSparseV1" else wire.validate_hdc(wd,wr)
  return {"query_id":rd["query_id"],"query_source_object_sha256":rd["query_source_object_sha256"],"input_stage":rd["input_stage"],"input_object_sha256":rd["input_object_sha256"],"representation_sha256":rd["representation_sha256"],"item_serialization_sha256":rd["item_serialization_sha256"],"wire_kind":rd["wire_kind"],"query_wire_sha256":core.dg(wr),"serialized_bytes":len(wr),"all_checks_passed":True}
class FI:
 @staticmethod
 def validate(d):
  if d["index"]["search_mode"]!="ExactDeterministic":raise ValueError("bad index")
class FB:
 @staticmethod
 def validate_artifact(d):return d,d["items"]
MODS={"qrep":FQ,"wire":FW,"index":FI,"build":FB}

def fixture():
 policy=core.policy_fixture("CanonicalSparseV1");pr=core.cb(policy)
 qdoc={"version":"math-canonical-sparse-wire-v1","feature_order":"FeatureIdUtf8Ascending","features":[{"feature_id":"a","count":1},{"feature_id":"b","count":1}]};qr=core.cb(qdoc)
 qsrc=sha("query-source");rep_sha=sha("rep");ser_sha=sha("ser")
 qrec={"version":"math-retrieval-query-representation-v1","receipt_id":"qr","authority":core.AUTHORITY,"query_id":"Q","query_source_object_sha256":qsrc,"source_object_contract_sha256":sha("soc"),"input_stage":"ParsedFolFormulaExt","input_object_sha256":sha("parsed"),"representation_sha256":rep_sha,"item_serialization_sha256":ser_sha,"wire_kind":"CanonicalSparseV1","producer_implementation_sha256":sha("prod"),"toolchain_manifest_sha256":sha("tool"),"query_wire_sha256":core.dg(qr),"serialized_bytes":len(qr)};qrr=core.cb(qrec)
 qreport={"version":"math-retrieval-query-representation-validation-report-v1","authority":core.AUTHORITY,"receipt_sha256":core.dg(qrr),"query_id":"Q","query_source_object_sha256":qsrc,"input_stage":"ParsedFolFormulaExt","input_object_sha256":qrec["input_object_sha256"],"representation_sha256":rep_sha,"item_serialization_sha256":ser_sha,"wire_kind":"CanonicalSparseV1","query_wire_sha256":core.dg(qr),"serialized_bytes":len(qr),"all_checks_passed":True};qrepr=core.cb(qreport)
 docs=[(sha("s1"),{"version":"math-canonical-sparse-wire-v1","feature_order":"FeatureIdUtf8Ascending","features":[{"feature_id":"a","count":2},{"feature_id":"b","count":2}]}),(sha("s2"),{"version":"math-canonical-sparse-wire-v1","feature_order":"FeatureIdUtf8Ascending","features":[{"feature_id":"a","count":3}]}),(sha("s3"),{"version":"math-canonical-sparse-wire-v1","feature_order":"FeatureIdUtf8Ascending","features":[{"feature_id":"c","count":9}]})]
 items=[]
 for sid,doc in sorted(docs):
  raw=core.cb(doc);items.append({"source_object_sha256":sid,"representation_object_sha256":core.dg(raw),"serialized_bytes":len(raw),"payload_base64":base64.b64encode(raw).decode()})
 artifact={"version":"math-retrieval-exact-index-artifact-v1","index_id":"ix","authority":core.AUTHORITY,"candidate_set_sha256":sha("candidates"),"target_id":"S","representation_sha256":rep_sha,"item_serialization_sha256":ser_sha,"payload_encoding":"Base64","item_order":"SourceObjectDigestAscending","item_count":len(items),"items":items};ar=core.cb(artifact)
 index={"version":"math-retrieval-index-v1","index_id":"ix","authority":core.AUTHORITY,"candidate_universe":{"candidate_set_sha256":sha("candidates")},"representation":{"channel":"Syntax","representation_family":"CanonicalSparse","representation_sha256":rep_sha,"item_serialization_sha256":ser_sha},"index":{"search_mode":"ExactDeterministic","index_artifact_sha256":core.dg(ar),"scoring_metric":"Cosine","scoring_policy_sha256":core.component_digest(policy["scoring"]),"score_precision_policy_sha256":core.component_digest(policy["precision"]),"query_normalization_policy_sha256":core.component_digest(policy["query_normalization"])},"retrieval_output":{}};ir=core.cb(index)
 br={"version":"math-retrieval-index-build-validation-report-v1","authority":core.AUTHORITY,"receipt_sha256":sha("buildrec"),"coverage_sha256":sha("cov"),"candidate_set_sha256":sha("candidates"),"index_manifest_sha256":core.dg(ir),"index_artifact_sha256":core.dg(ar),"input_set_sha256":sha("inputs"),"item_count":3,"target_id":"S","all_payload_hashes_match":True,"all_coverage_rows_match":True,"all_checks_passed":True};brr=core.cb(br)
 expected=[sha("s1"),sha("s2"),sha("s3")]
 trace={"version":"math-retrieval-trace-v1","trace_id":"t","authority":core.AUTHORITY,"graph":{},"experiment_seed":1,"query":{"query_id":"Q","query_source_object_sha256":qsrc},"retrieval":{"mode":"SingleIndex","single":{"index_manifest_sha256":core.dg(ir),"index_artifact_sha256":core.dg(ar),"requested_k":3,"ranked_source_object_digests":expected}},"packing":{},"resources":{},"control_bindings":[]};tr=core.cb(trace)
 trep={"version":"math-retrieval-trace-validation-report-v1","authority":core.AUTHORITY,"trace_id":"t","trace_sha256":core.dg(tr),"graph_report_sha256":sha("graph"),"experiment_sha256":sha("exp"),"arm_id":"S","experiment_seed":1,"packed_output_items":0,"packed_output_bytes":0,"all_checks_passed":True};trepr=core.cb(trep)
 rec={"version":core.RECEIPT_VERSION,"receipt_id":"r","authority":core.AUTHORITY,"trace_selector":"SingleIndex","scoring_policy_bundle_sha256":core.dg(pr),"query_representation_receipt_sha256":core.dg(qrr),"query_representation_report_sha256":core.dg(qrepr),"index_build_report_sha256":core.dg(brr),"index_manifest_sha256":core.dg(ir),"index_artifact_sha256":core.dg(ar),"trace_sha256":core.dg(tr),"trace_validation_report_sha256":core.dg(trepr)};rr=core.cb(rec)
 return [rec,rr,policy,pr,qrec,qrr,qreport,qrepr,qdoc,qr,br,brr,index,ir,artifact,ar,trace,tr,trep,trepr]

def rebind_trace(vals,new_ranking):
 x=copy.deepcopy(vals);t=x[16];t["retrieval"]["single"]["ranked_source_object_digests"]=new_ranking;tr=core.cb(t);x[17]=tr
 rp=x[18];rp["trace_sha256"]=core.dg(tr);x[19]=core.cb(rp)
 rec=x[0];rec["trace_sha256"]=core.dg(tr);rec["trace_validation_report_sha256"]=core.dg(x[19]);x[1]=core.cb(rec);return x

def must_fail(vals,name):
 try:val.validate_bound(*vals,mods=MODS)
 except ValueError:return
 raise AssertionError(f"attack unexpectedly passed: {name}")

def main():
 # Pure core policy and arithmetic.
 for kind in ("CanonicalSparseV1","BinaryHV16K"):core.validate_policy(core.policy_fixture(kind))
 bad=core.policy_fixture("CanonicalSparseV1");bad["scoring"]["threshold_policy"]="SimilarityGt0.1"
 try:core.validate_policy(bad)
 except core.ValidationError:pass
 else:raise AssertionError("threshold policy attack passed")
 bad=core.policy_fixture("CanonicalSparseV1");bad["precision"]["floating_point_ordering"]="Allowed"
 try:core.validate_policy(bad)
 except core.ValidationError:pass
 else:raise AssertionError("floating ordering attack passed")
 if core.hamming(b"\x00\xff",b"\x00\xfe")!=1:raise AssertionError("Hamming popcount failed")

 vals=fixture();rep,wit=val.validate_bound(*vals,mods=MODS)
 expected=[sha("s1"),sha("s2"),sha("s3")]
 if rep["expected_top_k"]!=expected:raise AssertionError("exact sparse replay failed")
 must_fail(rebind_trace(fixture(),[sha("s2"),sha("s1"),sha("s3")]),"permuted-ranking")
 must_fail(rebind_trace(fixture(),[sha("s1")]),"threshold-like-early-truncation")
 # Component-digest substitution is rejected independently of file hashes.
 x=fixture();x[12]["index"]["scoring_policy_sha256"]=sha("wrong-policy");x[13]=core.cb(x[12])
 # Rebind build report, trace and receipt to the substituted manifest so only policy semantics differ.
 x[10]["index_manifest_sha256"]=core.dg(x[13]);x[11]=core.cb(x[10]);x[16]["retrieval"]["single"]["index_manifest_sha256"]=core.dg(x[13]);x[17]=core.cb(x[16]);x[18]["trace_sha256"]=core.dg(x[17]);x[19]=core.cb(x[18])
 x[0]["index_manifest_sha256"]=core.dg(x[13]);x[0]["index_build_report_sha256"]=core.dg(x[11]);x[0]["trace_sha256"]=core.dg(x[17]);x[0]["trace_validation_report_sha256"]=core.dg(x[19]);x[1]=core.cb(x[0])
 must_fail(x,"opaque-scoring-policy-substitution")
 print("math retrieval ranking replay v1 adversarial integration: PASS")
if __name__=="__main__":main()
