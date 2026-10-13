#!/usr/bin/env python3
"""Independent stdlib-only oracle for FEP-EFE-ACTION-001A."""
from __future__ import annotations
import argparse, copy, hashlib, json, subprocess
from decimal import Decimal, localcontext, ROUND_HALF_EVEN, DecimalException
from fractions import Fraction
from pathlib import Path

SCHEMA="fep-efe-action-001a-reference-v1"
PROFILE="DiagonalGaussianPredictedEntropyDeltaV1"
NOVELTY_PROFILE="ActionEventCountNoveltyV1"
EVENT_CLASS="CommittedAction"
LEGACY_PROFILE="CurrentActionInvariantCore"
DIGEST="3eeb5fe88a4e6893414fcfac9139912ba1312d219a656f99f417e2aeb6a53858"
PATH="docs/fep/data/fep_efe_action_001a_reference_v1.json"
IDS=[f"F{i:02d}" for i in range(1,26)]
PREC=50
TOL=Decimal("1e-12")
CLAIM="software reference semantics only; no FEP scientific validation, expected-information-gain or mutual-information claim, real-world information gain, optimality, engineering evidence, or execution authority"
TOP={"schema","issue","source_kind","reference_profile","novelty_profile","novelty_event_class","numeric_policy","claim_ceiling","vocabularies","cases"}
KINDS={"ScoreSet","HistoryPurity","HistoryUpdate","EnumerationInvariant","SetIdentity","InputValidation","Currentness","IdentityChange","ExternalConstraint","ClaimCeiling","LegacyProfile","EventClassBoundary"}
VOCAB={
 "input_validity":["Valid","RejectDuplicateActionIdentity","RejectMissingPredictedPrecision","RejectNonPositivePrecision","RejectDimensionMismatch"],
 "epistemic_discrimination":["Available","Tie","Unavailable"],
 "novelty_semantics":["CommittedActionCount"],"scoring_purity":["Pure"],
 "enumeration_invariance":["Required"],"tie_disposition":["Preserved"],
 "currentness":["ReviewRebindRequired"],"claim_authority":["SoftwareReferenceOnly","None"]}
NUM={"decimal_precision":50,"rounding":"ROUND_HALF_EVEN","comparison_tolerance":"1e-12","expected_values":"relations-preferred"}
EKEYS={"input_validity","epistemic_discrimination","epistemic_relation","pragmatic_relation","novelty_relation","total_relation","claim_authority","tie_disposition","disposition","scoring_purity","final_committed_count","novelty","novelty_fraction","novelty_semantics","enumeration_invariance","per_action_outputs","set_identity","currentness","result_identity","reason","total_score_identity","raw_component_identity","winner","engineering_authority","evidence_authority","upgrade","counted_events"}
DISP={"ComponentTradeoffPreserved","ZeroWeightDoesNotEraseRawComponent","BlockedByExternalConstraint","OnlyDeclaredEventClassChangesNovelty"}
EXPECTED_SHAPES = {
 "ScoreSet": {
  frozenset({"claim_authority","epistemic_discrimination","epistemic_relation","input_validity","total_relation"}),
  frozenset({"epistemic_discrimination","epistemic_relation","input_validity","tie_disposition"}),
  frozenset({"epistemic_relation","input_validity","total_relation"}),
  frozenset({"disposition","epistemic_relation","input_validity","pragmatic_relation","total_relation"}),
  frozenset({"disposition","epistemic_relation","input_validity","tie_disposition","total_relation"}),
  frozenset({"disposition","input_validity","novelty_relation","total_relation"}),
  frozenset({"input_validity","novelty_relation","novelty_semantics","total_relation"}),
  frozenset({"input_validity","tie_disposition","total_relation","winner"}),
 },
 "HistoryPurity": {frozenset({"final_committed_count","novelty","novelty_semantics","scoring_purity"})},
 "HistoryUpdate": {
  frozenset({"final_committed_count","novelty","novelty_semantics"}),
  frozenset({"final_committed_count","novelty_fraction","novelty_semantics"}),
 },
 "EnumerationInvariant": {frozenset({"enumeration_invariance","per_action_outputs"})},
 "SetIdentity": {frozenset({"enumeration_invariance","set_identity"})},
 "InputValidation": {frozenset({"input_validity"})},
 "Currentness": {frozenset({"currentness"})},
 "IdentityChange": {
  frozenset({"reason","result_identity"}),
  frozenset({"raw_component_identity","total_score_identity"}),
 },
 "ExternalConstraint": {frozenset({"claim_authority","disposition"})},
 "ClaimCeiling": {frozenset({"claim_authority","engineering_authority","evidence_authority"})},
 "LegacyProfile": {frozenset({"epistemic_discrimination","upgrade"})},
 "EventClassBoundary": {frozenset({"counted_events","disposition","novelty_semantics"})},
}
RAW={
 "ScoreSet":{"current_precision","weights","actions"},
 "HistoryPurity":{"action_id","candidate_score_calls","initial_committed_count","commit_events"},
 "HistoryUpdate":{"action_id","initial_committed_count","commit_events"},
 "EnumerationInvariant":{"current_precision","weights","actions","orders"},
 "SetIdentity":{"actions","orders"},
 "Currentness":{"result_generation","current_generation"},
 "ExternalConstraint":{"action_id","epistemic_gain","feasibility","execution_authority"},
 "ClaimCeiling":{"reference_profile_pass"},
 "LegacyProfile":{"profile","epistemic_values","actions"},
 "EventClassBoundary":{"novelty_event_class","candidate_score_calls","selection_events","commit_events","external_execution_events"}}

def fail(s): raise AssertionError(s)
def keys(x,w,n):
 if not isinstance(x,dict) or set(x)!=w: fail(f"{n} keys")
def dec(x):
 if isinstance(x,bool) or not isinstance(x,(str,int)): fail("bad decimal type")
 d=Decimal(str(x))
 if not d.is_finite(): fail("nonfinite decimal")
 return d
def nni(x,n="count"):
 if isinstance(x,bool) or not isinstance(x,int) or x<0: fail(f"bad {n}")
 return x
def prec(x,n):
 if not isinstance(x,list) or not x: fail(f"bad {n}")
 y=[dec(v) for v in x]
 if any(v<=0 for v in y): fail("precision must be > 0")
 return y

def action(a,score=True):
 keys(a,{"id","predicted_precision","pragmatic","committed_count"} if score else {"id"},"action")
 if not isinstance(a["id"],str) or not a["id"].strip(): fail("bad action id")
 if score:
  if a["predicted_precision"] is not None and not isinstance(a["predicted_precision"],list): fail("bad predicted precision")
  p=dec(a["pragmatic"])
  if p<0: fail("pragmatic divergence must be nonnegative")
  nni(a["committed_count"],"committed_count")

def shape(kind,r):
 if kind=="InputValidation":
  if set(r)=={"action_ids"}:
   ids=r["action_ids"]
   if not isinstance(ids,list) or not ids or not all(isinstance(x,str) and x.strip() for x in ids): fail("bad action_ids")
   return
  keys(r,{"current_precision","actions"},"input raw")
  prec(r["current_precision"],"current_precision")
  if not isinstance(r["actions"],list) or not r["actions"]: fail("missing actions")
  action_ids=[]
  for a in r["actions"]:
   keys(a,{"id","predicted_precision"},"validation action")
   if not isinstance(a["id"],str) or not a["id"].strip(): fail("bad validation id")
   action_ids.append(a["id"])
   if a["predicted_precision"] is not None and not isinstance(a["predicted_precision"],list): fail("bad validation precision")
  if len(action_ids)!=len(set(action_ids)): fail("duplicate validation action identity")
  return
 if kind=="IdentityChange":
  if set(r)=={"profile_before","profile_after"}:
   before,after=r["profile_before"],r["profile_after"]
   if not isinstance(before,dict) or not isinstance(after,dict) or not before or set(before)!=set(after): fail("bad profile identity shape")
   for action_id in before:
    if not isinstance(action_id,str) or not action_id.strip(): fail("bad profile action id")
    a=prec(before[action_id],"profile_before precision")
    b=prec(after[action_id],"profile_after precision")
    if len(a)!=len(b): fail("profile dimension change requires a new reference profile")
  elif set(r)=={"weights_before","weights_after"}:
   for name in ("weights_before","weights_after"):
    weights=r[name]
    keys(weights,{"pragmatic","epistemic","novelty"},name)
    for value in weights.values():
     if dec(value)<0: fail("weights must be nonnegative")
  else: fail("bad identity shape")
  return
 keys(r,RAW[kind],f"{kind} raw")
 if kind in {"ScoreSet","EnumerationInvariant"}:
  keys(r["weights"],{"pragmatic","epistemic","novelty"},"weights")
  for v in r["weights"].values():
   if dec(v)<0: fail("weights must be nonnegative")
  if not isinstance(r["actions"],list) or len(r["actions"])<2: fail("need actions")
  for a in r["actions"]: action(a)
  if kind=="EnumerationInvariant":
   ids=[a["id"] for a in r["actions"]]
   orders=r["orders"]
   if not isinstance(orders,list) or len(orders)<2: fail("need at least two enumeration orders")
   for order in orders:
    if not isinstance(order,list) or len(order)!=len(ids): fail("bad order length")
    if not all(isinstance(x,str) and x.strip() for x in order): fail("bad order action id")
    if len(order)!=len(set(order)) or set(order)!=set(ids): fail("order must be a permutation of action ids")
 elif kind=="SetIdentity":
  if not isinstance(r["actions"],list) or len(r["actions"])<2: fail("need set-identity actions")
  for a in r["actions"]: action(a,False)
  ids=[a["id"] for a in r["actions"]]
  if len(ids)!=len(set(ids)): fail("duplicate set-identity action")
  orders=r["orders"]
  if not isinstance(orders,list) or len(orders)<2: fail("need at least two set-identity orders")
  for order in orders:
   if not isinstance(order,list) or len(order)!=len(ids): fail("bad set-identity order length")
   if not all(isinstance(x,str) and x.strip() for x in order): fail("bad set-identity order id")
   if len(order)!=len(set(order)) or set(order)!=set(ids): fail("set-identity order must be a permutation")
 elif kind=="HistoryPurity":
  if not isinstance(r["action_id"],str) or not r["action_id"].strip(): fail("bad action_id")
  nni(r["candidate_score_calls"]); nni(r["initial_committed_count"]); nni(r["commit_events"])
 elif kind=="HistoryUpdate":
  if not isinstance(r["action_id"],str) or not r["action_id"].strip(): fail("bad action_id")
  nni(r["initial_committed_count"]); nni(r["commit_events"])
 elif kind=="Currentness":
  for key in ("result_generation","current_generation"):
   if not isinstance(r[key],str) or not r[key].strip(): fail("bad generation identity")
 elif kind=="ExternalConstraint":
  if not isinstance(r["action_id"],str) or not r["action_id"].strip(): fail("bad external-constraint action id")
  if not isinstance(r["epistemic_gain"],str) or not r["epistemic_gain"].strip(): fail("bad epistemic-gain descriptor")
  if r["feasibility"] not in {"Observable","NotObservable"}: fail("unknown feasibility value")
  if r["execution_authority"] not in {"Present","Absent"}: fail("unknown execution-authority value")
 elif kind=="ClaimCeiling":
  if not isinstance(r["reference_profile_pass"],bool): fail("reference_profile_pass must be boolean")
 elif kind=="LegacyProfile":
  if r["profile"] != LEGACY_PROFILE: fail("unexpected legacy profile identity")
  values=r["epistemic_values"]
  actions=r["actions"]
  if not isinstance(values,list) or not values: fail("missing legacy epistemic values")
  if not isinstance(actions,list) or len(actions)!=len(values): fail("legacy action/value count mismatch")
  if not all(isinstance(x,str) and x.strip() for x in actions) or len(actions)!=len(set(actions)): fail("bad legacy action identities")
  for value in values: dec(value)
 elif kind=="EventClassBoundary":
  if r["novelty_event_class"]!=EVENT_CLASS: fail("wrong event class")
  for k in ("candidate_score_calls","selection_events","commit_events","external_execution_events"): nni(r[k],k)

def entropy(p):
 with localcontext() as c:
  c.prec=PREC; c.rounding=ROUND_HALF_EVEN
  pi=Decimal("3.1415926535897932384626433832795028841971693993751")
  d=Decimal(len(p)); logs=sum((x.ln() for x in p),Decimal(0))
  return (d+d*(Decimal(2)*pi).ln()-logs)/Decimal(2)
def epi(cur,pred):
 if len(cur)!=len(pred): fail("dimension mismatch")
 with localcontext() as c:
  c.prec=PREC; c.rounding=ROUND_HALF_EVEN
  return entropy(pred)-entropy(cur)
def nov(count):
 with localcontext() as c:
  c.prec=PREC; c.rounding=ROUND_HALF_EVEN
  return Decimal(1)/Decimal(1+nni(count,"committed_count"))
def weighted_total(wp,p,we,e,wn,n):
 with localcontext() as c:
  c.prec=PREC; c.rounding=ROUND_HALF_EVEN
  return wp*p+we*e-wn*n
def cmp(a,b):
 with localcontext() as c:
  c.prec=PREC; c.rounding=ROUND_HALF_EVEN
  z=a-b
  return 0 if abs(z)<=TOL else (-1 if z<0 else 1)
def rel(ai,a,bi,b): return f"{ai}{'=' if cmp(a,b)==0 else '<' if cmp(a,b)<0 else '>'}{bi}"
def scores(r):
 cur=prec(r["current_precision"],"current_precision"); w=r["weights"]
 wp,we,wn=[dec(w[k]) for k in ("pragmatic","epistemic","novelty")]
 ids=[a["id"] for a in r["actions"]]
 if len(ids)!=len(set(ids)): fail("duplicate action")
 out={}
 for a in r["actions"]:
  if a["predicted_precision"] is None: fail("missing predicted precision")
  pp=prec(a["predicted_precision"],"predicted_precision")
  if len(pp)!=len(cur): fail("dimension mismatch")
  p=dec(a["pragmatic"]); e=epi(cur,pp); n=nov(a["committed_count"])
  out[a["id"]]={"pragmatic":p,"epistemic":e,"novelty":n,"total":weighted_total(wp,p,we,e,wn,n)}
 return out
def pref(r):
 if "<" in r:return r.split("<")[0]
 if ">" in r:return r.split(">")[1]
 return None

def derive(c):
 keys(c,{"id","kind","raw","expected"},"case")
 k=c["kind"]; r=c["raw"]
 if k not in KINDS or not isinstance(r,dict): fail("bad kind/raw")
 shape(k,r); o={}
 if k=="ScoreSet":
  s=scores(r); a,b=list(s)[:2]
  er=rel(a,s[a]["epistemic"],b,s[b]["epistemic"]); pr=rel(a,s[a]["pragmatic"],b,s[b]["pragmatic"]); nr=rel(a,s[a]["novelty"],b,s[b]["novelty"]); tr=rel(a,s[a]["total"],b,s[b]["total"])
  o.update(input_validity="Valid",epistemic_relation=er,pragmatic_relation=pr,novelty_relation=nr,total_relation=tr,epistemic_discrimination="Tie" if "=" in er else "Available",novelty_semantics="CommittedActionCount",claim_authority="SoftwareReferenceOnly")
  if "=" in tr:o.update(tie_disposition="Preserved",winner="NoneByIterationOrder")
  rr={"pragmatic":pr,"epistemic":er,"novelty":nr}
  if any(dec(r["weights"][x])==0 and "=" not in rr[x] for x in rr):o["disposition"]="ZeroWeightDoesNotEraseRawComponent"
  if pref(er) is not None and pref(pr) is not None and pref(er)!=pref(pr):o["disposition"]="ComponentTradeoffPreserved"
 elif k=="HistoryPurity":
  f=r["initial_committed_count"]+r["commit_events"]; o.update(scoring_purity="Pure",final_committed_count=f,novelty=format(nov(f),"f"),novelty_semantics="CommittedActionCount")
 elif k=="HistoryUpdate":
  f=r["initial_committed_count"]+r["commit_events"]; o.update(final_committed_count=f,novelty=format(nov(f),"f"),novelty_fraction=str(Fraction(1,1+f)),novelty_semantics="CommittedActionCount")
 elif k=="EnumerationInvariant":
  by_id={a["id"]:a for a in r["actions"]}
  outputs=[]
  for order in r["orders"]:
   ordered=dict(r)
   ordered["actions"]=[by_id[action_id] for action_id in order]
   result=scores(ordered)
   outputs.append({
    action_id:{component:result[action_id][component] for component in ("epistemic","pragmatic","novelty","total")}
    for action_id in sorted(result)
   })
  if any(output!=outputs[0] for output in outputs[1:]):
   fail("candidate scores depend on enumeration order")
  o.update(enumeration_invariance="Required",per_action_outputs="IdenticalAcrossOrders")
 elif k=="SetIdentity":
  ids=[a["id"] for a in r["actions"]]
  if len(ids)!=len(set(ids)):fail("duplicate set identity")
  o.update(set_identity="Same" if all(set(x)==set(ids) and len(x)==len(ids) for x in r["orders"]) else "Changed",enumeration_invariance="Required")
 elif k=="InputValidation":
  if "action_ids" in r:o["input_validity"]="RejectDuplicateActionIdentity" if len(r["action_ids"])!=len(set(r["action_ids"])) else "Valid"
  elif any(a["predicted_precision"] is None for a in r["actions"]):o["input_validity"]="RejectMissingPredictedPrecision"
  else:
   try:
    cur=prec(r["current_precision"],"current_precision"); ps=[prec(a["predicted_precision"],"predicted_precision") for a in r["actions"]]
    o["input_validity"]="RejectDimensionMismatch" if any(len(p)!=len(cur) for p in ps) else "Valid"
   except AssertionError as e:
    if "precision must be > 0" in str(e):o["input_validity"]="RejectNonPositivePrecision"
    else:raise
 elif k=="Currentness":o["currentness"]="ReviewRebindRequired" if r["result_generation"]!=r["current_generation"] else "Current"
 elif k=="IdentityChange":
  if "profile_before" in r:o.update(result_identity="Changed" if r["profile_before"]!=r["profile_after"] else "Same",reason="UncertaintyProfileChanged")
  else:o.update(total_score_identity="Changed" if r["weights_before"]!=r["weights_after"] else "Same",raw_component_identity="UnchangedIfInputsUnchanged")
 elif k=="ExternalConstraint":
  blocked=r["feasibility"]!="Observable" or r["execution_authority"]!="Present"; o.update(disposition="BlockedByExternalConstraint" if blocked else "EligibleByThisReference",claim_authority="None" if blocked else "SoftwareReferenceOnly")
 elif k=="ClaimCeiling":o.update(claim_authority="SoftwareReferenceOnly",engineering_authority="None",evidence_authority="None")
 elif k=="LegacyProfile":
  vals=[dec(v) for v in r["epistemic_values"]]; u=len(set(vals))<=1; o.update(epistemic_discrimination="Unavailable" if u else "Available",upgrade="Forbidden" if u else "ReviewRequired")
 elif k=="EventClassBoundary":o.update(novelty_semantics="CommittedActionCount",counted_events=r["commit_events"],disposition="OnlyDeclaredEventClassChangesNovelty")
 return o

def expected(c,g):
 e=c["expected"]
 if not isinstance(e,dict) or not e or set(e)-EKEYS:fail("bad expected shape")
 allowed=EXPECTED_SHAPES.get(c["kind"])
 if allowed is None or frozenset(e) not in allowed:fail("incomplete or unrecognized expected assertion shape")
 if "disposition" in e and e["disposition"] not in DISP:fail("bad disposition")
 for k,w in e.items():
  if k not in g:fail(f"{c['id']} missing {k}")
  a=g[k]
  if k=="novelty":
   if cmp(dec(a),dec(w))!=0:fail(f"{c['id']} novelty mismatch")
  elif a!=w:fail(f"{c['id']} {k}: {a!r} != {w!r}")

def validate(d):
 keys(d,TOP,"top")
 if not isinstance(d["cases"],list): fail("cases must be an array")
 if d["schema"]!=SCHEMA or d["issue"]!=6210 or d["source_kind"]!="synthetic-reference":fail("identity mismatch")
 if d["reference_profile"]!=PROFILE or d["novelty_profile"]!=NOVELTY_PROFILE or d["novelty_event_class"]!=EVENT_CLASS:fail("profile mismatch")
 if d["numeric_policy"]!=NUM or d["claim_ceiling"]!=CLAIM or d["vocabularies"]!=VOCAB:fail("contract mismatch")
 if [c.get("id") for c in d["cases"]]!=IDS:fail("case order mismatch")
 for c in d["cases"]:expected(c,derive(c))
def case(d,i):return next(c for c in d["cases"] if c["id"]==i)
def detected(m):
 try:validate(m);return False
 except (AssertionError,AttributeError,DecimalException,IndexError,KeyError,TypeError,ValueError):return True

def mutations(d):
 fs=[]
 def add(name,f):fs.append((name,f))
 add("schema",lambda m:m.__setitem__("schema","bad")); add("top-field",lambda m:m.__setitem__("x",1)); add("profile",lambda m:m.__setitem__("reference_profile","bad")); add("profile-promotion",lambda m:m.__setitem__("reference_profile","ExpectedInformationGainV1")); add("novelty-profile",lambda m:m.__setitem__("novelty_profile","ExecutedActionCountNoveltyV1")); add("event-class",lambda m:m.__setitem__("novelty_event_class","ExecutedAction")); add("numeric",lambda m:m["numeric_policy"].__setitem__("decimal_precision",34)); add("claim",lambda m:m.__setitem__("claim_ceiling","broad")); add("vocab",lambda m:m["vocabularies"].__setitem__("novelty_semantics",["ExecutedActionCount"])); add("remove-case",lambda m:m["cases"].pop()); add("reorder",lambda m:m["cases"].__setitem__(slice(0,2),[m["cases"][1],m["cases"][0]])); add("flatten-E",lambda m:case(m,"F01")["raw"]["actions"][0].__setitem__("predicted_precision",["1","1"])); add("reverse-E",lambda m:case(m,"F03")["raw"]["actions"][0].__setitem__("predicted_precision",["4","4"])); add("erase-N",lambda m:case(m,"F08")["raw"]["actions"][1].__setitem__("committed_count",0)); add("history-empty-action-id",lambda m:case(m,"F09")["raw"].__setitem__("action_id","")); add("history-whitespace-action-id",lambda m:case(m,"F10")["raw"].__setitem__("action_id","   ")); add("history-null-action-id",lambda m:case(m,"F11")["raw"].__setitem__("action_id",None)); add("history-wrong-type-action-id",lambda m:case(m,"F09")["raw"].__setitem__("action_id",{})); add("score-as-commit",lambda m:case(m,"F09")["raw"].__setitem__("commit_events",1)); add("erase-commit",lambda m:case(m,"F10")["raw"].__setitem__("commit_events",0)); add("erase-second-commit",lambda m:case(m,"F11")["raw"].__setitem__("commit_events",1)); add("enum-single-order",lambda m:case(m,"F12")["raw"].__setitem__("orders",[case(m,"F12")["raw"]["orders"][0]])); add("enum-missing-id",lambda m:case(m,"F12")["raw"]["orders"][1].pop()); add("enum-duplicate-id",lambda m:case(m,"F12")["raw"]["orders"][1].__setitem__(0,"B")); add("enum-non-string-id",lambda m:case(m,"F12")["raw"]["orders"][0].__setitem__(0,None)); add("boundary-class",lambda m:case(m,"F25")["raw"].__setitem__("novelty_event_class","SelectedAction")); add("boundary-commit",lambda m:case(m,"F25")["raw"].__setitem__("commit_events",1)); add("dup-id",lambda m:case(m,"F01")["raw"]["actions"][1].__setitem__("id","A")); add("missing-pred",lambda m:case(m,"F01")["raw"]["actions"][0].__setitem__("predicted_precision",None)); add("zero-precision",lambda m:case(m,"F01")["raw"]["actions"][0].__setitem__("predicted_precision",["0","2"])); add("dim",lambda m:case(m,"F01")["raw"]["actions"][0].__setitem__("predicted_precision",["2"])); add("zero-E-weight",lambda m:case(m,"F01")["raw"]["weights"].__setitem__("epistemic","0")); add("break-tie",lambda m:case(m,"F21")["raw"]["actions"][0].__setitem__("pragmatic","1")); add("currentness",lambda m:case(m,"F18")["raw"].__setitem__("current_generation","G1")); add("legacy-upgrade",lambda m:case(m,"F24")["raw"].__setitem__("epistemic_values",["0","1","0"])); add("legacy-unknown-profile",lambda m:case(m,"F24")["raw"].__setitem__("profile","AnotherProfile")); add("erase-blocker",lambda m:case(m,"F22")["raw"].update({"feasibility":"Observable","execution_authority":"Present"})); add("expected-delete-assertion",lambda m:case(m,"F01")["expected"].pop("epistemic_relation")); add("expected-delete-validity",lambda m:case(m,"F14")["expected"].pop("input_validity")); add("expected-add-unknown-assertion",lambda m:case(m,"F01")["expected"].__setitem__("future_claim","PASS")); add("expected-tamper",lambda m:case(m,"F01")["expected"].__setitem__("total_relation","A=B")); add("action-field",lambda m:case(m,"F01")["raw"]["actions"][0].__setitem__("mystery",1)); add("set-identity-single-order",lambda m:case(m,"F13")["raw"].__setitem__("orders",[case(m,"F13")["raw"]["orders"][0]])); add("set-identity-wrong-order-type",lambda m:case(m,"F13")["raw"]["orders"].__setitem__(0,None)); add("set-identity-duplicate-order-id",lambda m:case(m,"F13")["raw"]["orders"][1].__setitem__(0,"B")); add("currentness-empty-generation",lambda m:case(m,"F18")["raw"].__setitem__("current_generation","  ")); add("currentness-nonstring-generation",lambda m:case(m,"F18")["raw"].__setitem__("current_generation",None)); add("identity-empty-action-key",lambda m:case(m,"F19")["raw"]["profile_after"].__setitem__("",["2","1"])); add("identity-unknown-weight-key",lambda m:case(m,"F20")["raw"]["weights_after"].__setitem__("hidden","1")); add("constraint-empty-action-id",lambda m:case(m,"F22")["raw"].__setitem__("action_id","")); add("constraint-unknown-feasibility",lambda m:case(m,"F22")["raw"].__setitem__("feasibility","maybe")); add("constraint-unknown-authority",lambda m:case(m,"F22")["raw"].__setitem__("execution_authority","maybe")); add("claim-pass-nonboolean",lambda m:case(m,"F23")["raw"].__setitem__("reference_profile_pass","true")); add("legacy-action-count-mismatch",lambda m:case(m,"F24")["raw"]["actions"].pop()); add("legacy-nonfinite-epistemic",lambda m:case(m,"F24")["raw"]["epistemic_values"].__setitem__(0,"NaN")); add("negative-score-weight",lambda m:case(m,"F01")["raw"]["weights"].__setitem__("pragmatic","-1")); add("negative-pragmatic-divergence",lambda m:case(m,"F01")["raw"]["actions"][0].__setitem__("pragmatic","-0.1")); add("input-empty-action-id",lambda m:case(m,"F14")["raw"]["action_ids"].__setitem__(0,"")); add("input-empty-action-set",lambda m:case(m,"F14")["raw"].__setitem__("action_ids",[])); add("cases-object",lambda m:m.__setitem__("cases",{})); add("cases-null",lambda m:m.__setitem__("cases",None)); add("case-null-entry",lambda m:m["cases"].__setitem__(0,None)); add("input-duplicate-validation-actions",lambda m:case(m,"F15")["raw"]["actions"].append({"id":"A","predicted_precision":["1"]})); add("invalid-current-precision",lambda m:case(m,"F01")["raw"].__setitem__("current_precision",["0"])); add("nonfinite-weight",lambda m:case(m,"F01")["raw"]["weights"].__setitem__("pragmatic","NaN")); add("disposition",lambda m:case(m,"F04")["expected"].__setitem__("disposition","MagicPass"))
 n=0
 for name,f in fs:
  m=copy.deepcopy(d);f(m)
  if not detected(m):fail(f"mutation escaped: {name}")
  n+=1
 f09=case(d,"F09"); buggy=f09["raw"]["initial_committed_count"]+f09["raw"]["candidate_score_calls"]+f09["raw"]["commit_events"]
 if buggy==f09["expected"]["final_committed_count"]:fail("considered-as-commit mutant escaped")
 n+=1
 f25=case(d,"F25")
 if f25["raw"]["commit_events"]+f25["raw"]["external_execution_events"]==f25["expected"]["counted_events"]:fail("execution-as-commit mutant escaped")
 n+=1
 if f25["raw"]["commit_events"]+f25["raw"]["selection_events"]==f25["expected"]["counted_events"]:fail("selection-as-commit mutant escaped")
 return n

def main():
 p=argparse.ArgumentParser();p.add_argument("--source-ref");p.add_argument("--corpus-file");a=p.parse_args()
 raw=subprocess.check_output(["git","show",f"{a.source_ref}:{PATH}"]) if a.source_ref else Path(a.corpus_file or PATH).read_bytes()
 got=hashlib.sha256(raw).hexdigest()
 if got!=DIGEST:fail(f"digest {got} != {DIGEST}")
 d=json.loads(raw.decode());validate(d);n=mutations(d)
 print(f"PASS FEP-EFE-ACTION-001A1: {len(IDS)} cases; {n} hostile mutations; profile={PROFILE}; event_class={EVENT_CLASS}")
if __name__=="__main__":main()
