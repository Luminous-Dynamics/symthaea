#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path
DOMAIN="symthaea.se001q.github-provenance.v1"
def canonical(o): return json.dumps(o,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def sha_bytes(b): return "sha256:"+hashlib.sha256(b).hexdigest()
def die(m): raise SystemExit(m)
def load(path):
 p=Path(path); raw=p.read_bytes(); return json.loads(raw),raw
def reject(v,path="$"):
 if isinstance(v,dict):
  for k,c in v.items():
   q=f"{path}.{k}"
   if k=="repair_authority": die(f"forbidden repair_authority at {q}")
   if k=="repair_authority_claim" and c!="NONE": die(f"repair authority violation at {q}")
   if k=="qualification_claim" and c!="NONE": die(f"qualification authority violation at {q}")
   reject(c,q)
 elif isinstance(v,list):
  for i,c in enumerate(v): reject(c,f"{path}[{i}]")
def one(items,pred,what):
 xs=[x for x in items if pred(x)]
 if len(xs)!=1: die(f"expected exactly one {what}, found {len(xs)}")
 return xs[0]
def main():
 ap=argparse.ArgumentParser()
 for name in ("contract","run-json","jobs-json","artifacts-json","expected-run-id","expected-pr-number","expected-head-sha","expected-workflow-path","expected-job-name","expected-artifact-name","output"):
  ap.add_argument("--"+name,required=True)
 ns=ap.parse_args()
 c,cb=load(ns.contract); run,rb=load(ns.run_json); jobs,jb=load(ns.jobs_json); arts,ab=load(ns.artifacts_json)
 for o in (c,run,jobs,arts): reject(o)
 if c.get("schema")!="symthaea.se001q.github-provenance-contract.v1" or c.get("domain")!=DOMAIN: die("bad contract")
 if c.get("authority",{}).get("sufficient_for_repair_grant") is not False: die("contract authority violation")
 rid=int(ns.expected_run_id); prn=int(ns.expected_pr_number)
 if run.get("id")!=rid or run.get("event")!=c["required_event"] or run.get("status")!=c["required_run_status"] or run.get("conclusion")!=c["required_run_conclusion"]: die("run state mismatch")
 if run.get("head_sha")!=ns.expected_head_sha or run.get("path")!=ns.expected_workflow_path: die("run identity mismatch")
 if int(run.get("run_attempt",0))<1: die("bad run attempt")
 pr=one(run.get("pull_requests") or [],lambda p:p.get("number")==prn,"matching pull request")
 if (pr.get("head") or {}).get("sha")!=ns.expected_head_sha: die("PR head mismatch")
 job=one(jobs.get("jobs") or [],lambda j:j.get("name")==ns.expected_job_name and j.get("run_id")==rid,"matching job")
 if job.get("status")!=c["required_job_status"] or job.get("conclusion")!=c["required_job_conclusion"]: die("job state mismatch")
 steps=job.get("steps")
 if c.get("require_nonempty_job_steps") and not steps: die("job steps missing")
 art=one(arts.get("artifacts") or [],lambda a:a.get("name")==ns.expected_artifact_name,"matching artifact")
 if c.get("require_artifact_not_expired") and art.get("expired") is not False: die("artifact expired")
 if int(art.get("size_in_bytes",0))<=0: die("artifact empty")
 dig=art.get("digest")
 if not isinstance(dig,str) or not dig.startswith(c["require_artifact_digest_prefix"]): die("artifact digest invalid")
 wr=art.get("workflow_run")
 if isinstance(wr,dict) and wr.get("id") is not None and wr.get("id")!=rid: die("artifact run mismatch")
 snaps={"run_json_sha256":sha_bytes(rb),"jobs_json_sha256":sha_bytes(jb),"artifacts_json_sha256":sha_bytes(ab)}
 if c.get("require_distinct_api_snapshots") and len(set(snaps.values()))!=3: die("snapshot digest collision")
 ident={"domain":DOMAIN,"source":c["source"],"contract_sha256":sha_bytes(cb),"run":{"id":str(rid),"attempt":str(run.get("run_attempt")),"event":run.get("event"),"status":run.get("status"),"conclusion":run.get("conclusion"),"head_sha":run.get("head_sha"),"head_branch":run.get("head_branch"),"workflow_path":run.get("path"),"pr_number":str(prn)},"job":{"id":str(job.get("id")),"name":job.get("name"),"status":job.get("status"),"conclusion":job.get("conclusion"),"steps":[{"name":s.get("name"),"status":s.get("status"),"conclusion":s.get("conclusion"),"number":s.get("number")} for s in (steps or [])]},"artifact":{"id":str(art.get("id")),"name":art.get("name"),"size_in_bytes":art.get("size_in_bytes"),"digest":dig,"expired":art.get("expired")},"api_snapshots":snaps,"provenance_strength":"API_CONSISTENCY_WITNESS_NOT_GITHUB_SIGNATURE","authority":{"meaning":"execution provenance consistency only","sufficient_for_repair_grant":False,"qualification_claim":"NONE","repair_authority_claim":"NONE"}}
 out={"schema":DOMAIN,"provenance_witness_id":sha_bytes(canonical(ident)),"identity":ident}; reject(out)
 p=Path(ns.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"schema":DOMAIN,"result":"PASS","provenance_witness_id":out["provenance_witness_id"],"run_id":str(rid),"artifact_id":str(art.get("id")),"artifact_digest":dig,"sufficient_for_repair_grant":False,"qualification_claim":"NONE","repair_authority_claim":"NONE"},sort_keys=True))
if __name__=="__main__": main()
