#!/usr/bin/env python3
"""Static projected-BCE receipt checker plus tiny exhaustive oracle; not a formal checker."""
import copy,hashlib,itertools,json,pathlib
from fractions import Fraction
HERE=pathlib.Path(__file__).resolve().parent
CERT=HERE/"projected_bce_step_certificate.v1.json"; OUT=HERE/"projected_bce_step_results.v1.json"
def norm(u,cs):
 return {"universe":sorted(u),"clauses":[{"id":c["id"],"literals":sorted(c["literals"],key=lambda x:(abs(x),x<0))} for c in sorted(cs,key=lambda c:c["id"])]}
def digest(u,cs):
 b=json.dumps(norm(u,cs),sort_keys=True,separators=(",",":")).encode()
 return "sha256:"+hashlib.sha256(b).hexdigest()
def tw(xs):
 s=set(xs); return next((abs(x) for x in sorted(s,key=lambda z:(abs(z),z<0)) if -x in s),None)
def check(c):
 e=[]
 if c.get("format")!="symthaea.projected-bce-step-certificate" or c.get("version")!=1:return ["unsupported version"]
 p=c["problem"];u=p["variables"];v=p["visible"];h=p["hidden"]
 if len(u)!=len(set(u)) or len(v)!=len(set(v)) or len(h)!=len(set(h)) or set(v)&set(h) or set(v)|set(h)!=set(u):e.append("invalid partition")
 if c["claim"]!={"relation":"projected_support_equivalence","theorem_profile":"bce-on-quantified-variable-v1"}:e.append("unsupported/overclaimed relation")
 s,t=c["source"],c["target"]
 for label,f in (("source",s),("target",t)):
  ids=[x["id"] for x in f["clauses"]]
  if len(ids)!=len(set(ids)):e.append(label+" duplicate clause ID")
  for cl in f["clauses"]:
   if len(cl["literals"])!=len(set(cl["literals"])) or any(x==0 or abs(x)>len(u) for x in cl["literals"]):e.append(label+" malformed literals")
 if s["digest"]!=digest(u,s["clauses"]):e.append("source digest mismatch")
 if t["digest"]!=digest(u,t["clauses"]):e.append("target digest mismatch")
 tr=c["transformation"]
 if tr.get("operation")!="remove_blocked_clause":return e+["unsupported operation"]
 sb={x["id"]:x for x in s["clauses"]};tb={x["id"]:x for x in t["clauses"]};rid=tr["removed_clause_id"]
 if rid not in sb:return e+["missing removed clause"]
 C=sb[rid];lit=tr["blocking_literal"]
 if lit not in C["literals"]:e.append("blocker not in clause")
 if abs(lit)<1 or abs(lit)>len(u) or u[abs(lit)-1] not in h:e.append("blocker not hidden")
 if tw(C["literals"]) is not None:e.append("removed clause tautological")
 if tb!={k:x for k,x in sb.items() if k!=rid}:e.append("target is not exact single-clause deletion")
 opp={x["id"]:x for x in s["clauses"] if x["id"]!=rid and -lit in x["literals"]}
 supplied={x["opposing_clause_id"]:x for x in c["resolvent_checks"]}
 if len(supplied)!=len(c["resolvent_checks"]) or set(supplied)!=set(opp):e.append("opposing coverage incomplete/duplicated")
 for cid,D in opp.items():
  item=supplied.get(cid)
  if item is None:continue
  res=[x for x in C["literals"] if x!=lit]+[x for x in D["literals"] if x!=-lit]
  w=tw(res);canonical=sorted(set(res),key=lambda x:(abs(x),x<0))
  if w is None:e.append("non-tautological resolvent "+cid)
  if item.get("resolvent")!=canonical:e.append("resolvent mismatch "+cid)
  if item.get("tautology_variable")!=w:e.append("tautology witness mismatch "+cid)
 return e
def sat(cs,bits):return all(any((bits[abs(l)-1]==1)==(l>0) for l in c["literals"]) for c in cs)
def oracle(c,key):
 u=c["problem"]["variables"];vi=[u.index(x) for x in c["problem"]["visible"]];n=0;ss=set()
 for b in itertools.product((0,1),repeat=len(u)):
  if sat(c[key]["clauses"],b):n+=1;ss.add("".join(str(b[i]) for i in vi))
 return n,sorted(ss)
def main():
 c=json.loads(CERT.read_text());err=check(c)
 if err:raise SystemExit("FAIL valid receipt: "+"; ".join(err))
 sf,ss=oracle(c,"source");tf,ts=oracle(c,"target");ex=c["oracle_expectation"]
 assert (sf,tf)==(int(ex["source_full_model_count"]),int(ex["target_full_model_count"])) and sf!=tf
 assert ss==ts==ex["source_projected_support"]==ex["target_projected_support"]
 w=ex["pwmc_weights"]["a"]
 def val(s):
  z=sum((Fraction(w["true"] if b=="1" else w["false"]) for b in s),Fraction(0))
  return f"{z.numerator}/{z.denominator}"
 assert val(ss)==val(ts)==ex["pwmc_value"]
 print("PASS static hidden-variable BCE receipt; projected support/PWMC preserved; full models",sf,"->",tf)
 xs=[]
 def mut(label,fn):
  x=copy.deepcopy(c);fn(x);xs.append((label,x))
 mut("blocking variable visible",lambda x:(x["problem"].update(hidden=[],visible=["a","h"])))
 mut("duplicate visible variable",lambda x:x["problem"].update(visible=["a","a"]))
 mut("missing opposing coverage",lambda x:x.update(resolvent_checks=[]))
 mut("false tautology witness",lambda x:x["resolvent_checks"][0].update(tautology_variable=2))
 mut("undeclared target rewrite",lambda x:x["target"]["clauses"].append({"id":"c3","literals":[1]}))
 mut("full-model overclaim",lambda x:x["claim"].update(relation="full_model_equivalence"))
 mut("wrong blocker",lambda x:x["transformation"].update(blocking_literal=-2))
 mut("stale source digest",lambda x:x["source"].update(digest="sha256:"+"0"*64))
 mut("stale target digest",lambda x:x["target"].update(digest="sha256:"+"f"*64))
 for label,x in xs:
  e=check(x)
  if not e:raise AssertionError("false accept "+label)
  print("REJECT",label,"-",e[0])
 result={"format":"symthaea.projected-bce-step-results","version":1,
 "claim_ceiling":"syntactic_static_BCE_receipt_validation_plus_exhaustive_fixture_oracle_not_formal_proof",
 "certificate_check":"passed","source_full_models":sf,"target_full_models":tf,
 "source_projected_support":ss,"target_projected_support":ts,"pwmc_visible_weighted_value":val(ss),
 "negative_controls":len(xs),"negative_controls_rejected":len(xs),
 "source_digest":c["source"]["digest"],"target_digest":c["target"]["digest"],
 "certificate_sha256":"sha256:623eb9beed7063c5aa273019de6f93492589b4a5617cb40e18003bae49bdac6a",
 "checker_sha256":"sha256:df75fdee730c25a295ac00d1359bdd982f09cd1217e3990f6d927d726b495f2e"}
 OUT.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
 print("PASS",len(xs),"negative controls; reference checker only")
if __name__=="__main__":main()
