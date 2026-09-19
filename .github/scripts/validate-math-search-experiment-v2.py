#!/usr/bin/env python3
"""Semantic validator for Symthaea math-search experiment manifest v2."""

import argparse, copy, json, math, sys
from pathlib import Path

SCHEMA="symthaea.math-search-experiment.v2"; AUTH="MeasurementOnly"
STAGES={"Q0RepresentationRetrieval","Q1BlindSolvedTransfer","Q2CrossDomainTransfer","Q3FormalConjecturesRediscovery","Q4ResearchHeldOut"}
RETR={"StructuralNeighborRecallAtK","EquivalentFamilyRecallAtK","MeanReciprocalRank","NormalizedDiscountedCumulativeGain","PositiveRankMargin","FalseEquivalentNeighborRate","RetrievalLatency","RetrievalNormalizedCompute"}
SEARCH={"FormallySolvedRate","VerifiedUsefulLemmaRate","ValidCounterexamples","ProofCallsPerSolved","SearchNodesPerSolved","NormalizedComputePerSolved","TimeToFirstUsefulLemma","CrossDomainTransferRate","RepeatedFailureRate","FalsePruningRate","HarmfulTransferRate"}
ENDPOINTS=RETR|SEARCH
REQ_CONTROLS={"LexicalRetrieval","RandomRetrieval","ShuffledHdcVectors","PermutedChallengeAssociations"}
BASE_CHECKS={"RetrievalDiffersFromBaseline","RepresentationInterventionApplied","BudgetAccountingComplete"}
ROOT={"schema_version","experiment_id","authority","stage","frozen_before_evaluation","shared_contract","budget","statistical_plan_sha256","seeds","primary_endpoints","secondary_endpoints","manipulation_checks","strategy_controls","evolutionary_search","arms","planned_contrasts"}
SHARED={"challenge_set_sha256","corpus_snapshot_sha256","knowledge_boundary_sha256","source_object_contract_sha256","normalization_contract_sha256","normalization_implementation_sha256","toolchain_manifest_sha256","human_intervention_policy_sha256"}
BUDGET={"accounting_policy_sha256","retrieved_items_max","retrieved_item_bytes_max","retrieval_context_bytes_max","total_context_bytes_max","normalized_compute_units_max","wall_time_ms_max","proof_calls_max","solver_calls_max","search_nodes_max","candidate_count_max","retrieval_queries_max","unused_budget_reallocation"}
AR={"arm_id","arm_role","representation_channels","representation_family","representation_sha256","retriever_family","retriever_sha256","control_kind","negative_search_memory"}
AO={"index_manifest_sha256","fusion_policy_sha256","parent_arm_id"}
CK={"None","LexicalRetrieval","RandomRetrieval","ShuffledHdcVectors","PermutedChallengeAssociations"}
RF={"None","Lexical","CanonicalSparse","HDC","FusionConventional","FusionHDC"}
RT={"None","Conventional","HDC","Fusion","Control"}
ROLES={"Baseline","Treatment","Control","MemoryAugmented"}
DIFF={"RetrievalAugmentation","SyntaxVsNormalForm","HdcSpecificRepresentation","FusionBenefit","RetrieverMechanism","NegativeSearchMemory","Control"}
CON={"contrast_id","left_arm_id","right_arm_id","intended_difference","primary_endpoint","direction"}

class V(ValueError): pass
def closed(o,k,w):
    x=set(o)-k
    if x: raise V(f"{w}: unknown fields {sorted(x)}")
def req(o,k,w):
    x=k-set(o)
    if x: raise V(f"{w}: missing fields {sorted(x)}")
def text(x,w):
    if not isinstance(x,str) or not x.strip(): raise V(f"{w}: expected non-empty string")
    return x
def sha(x,w):
    x=text(x,w)
    if len(x)!=71 or not x.startswith("sha256:") or any(c not in "0123456789abcdef" for c in x[7:]): raise V(f"{w}: invalid sha256")
def ints(x,w,m=0):
    if not isinstance(x,int) or isinstance(x,bool) or x<m: raise V(f"{w}: expected integer >= {m}")
def uniq(x,w,empty=False):
    if not isinstance(x,list) or (not x and not empty): raise V(f"{w}: expected list")
    y=[text(v,f"{w}[{i}]") for i,v in enumerate(x)]
    if len(y)!=len(set(y)): raise V(f"{w}: duplicates")
    return y
def dg(s): return "sha256:"+(s.encode().hex()+"0"*64)[:64]

def budget(b):
    if not isinstance(b,dict): raise V("budget: object required")
    closed(b,BUDGET,"budget"); req(b,BUDGET,"budget"); sha(b["accounting_policy_sha256"],"budget.accounting_policy_sha256")
    for k in BUDGET-{"accounting_policy_sha256","normalized_compute_units_max","unused_budget_reallocation"}: ints(b[k],f"budget.{k}",1 if k in {"total_context_bytes_max","wall_time_ms_max"} else 0)
    n=b["normalized_compute_units_max"]
    if isinstance(n,bool) or not isinstance(n,(int,float)) or not math.isfinite(n) or n<=0: raise V("budget.normalized_compute_units_max: positive finite number required")
    if b["unused_budget_reallocation"] is not False: raise V("budget: unused budget reallocation forbidden")
    if b["retrieval_context_bytes_max"]>b["total_context_bytes_max"]: raise V("budget: retrieval context exceeds total context")
    if b["retrieved_items_max"] and b["retrieved_item_bytes_max"]>b["retrieval_context_bytes_max"]: raise V("budget: item max exceeds retrieval context")

def arm(a,w,stage):
    if not isinstance(a,dict): raise V(f"{w}: object required")
    closed(a,AR|AO,w); req(a,AR,w)
    aid=text(a["arm_id"],f"{w}.arm_id")
    if len(aid)>64 or not aid[0].isalpha() or any(not(c.isalnum() or c in "._-") for c in aid): raise V(f"{w}.arm_id invalid")
    if a["arm_role"] not in ROLES or a["representation_family"] not in RF or a["retriever_family"] not in RT or a["control_kind"] not in CK: raise V(f"{w}: unsupported enum")
    ch=set(uniq(a["representation_channels"],f"{w}.representation_channels",True))
    if not ch<={"Syntax","ExactNormalForm"} or len(ch)>2: raise V(f"{w}: bad channels")
    sha(a["representation_sha256"],f"{w}.representation_sha256"); sha(a["retriever_sha256"],f"{w}.retriever_sha256")
    for k in ("index_manifest_sha256","fusion_policy_sha256"):
        if k in a: sha(a[k],f"{w}.{k}")
    if not isinstance(a["negative_search_memory"],bool): raise V(f"{w}: negative_search_memory bool required")
    fam=a["representation_family"]; ret=a["retriever_family"]; role=a["arm_role"]; ctrl=a["control_kind"]
    if fam=="None" and ch: raise V(f"{w}: None representation must have no channels")
    if fam=="Lexical" and ch!={"Syntax"}: raise V(f"{w}: lexical is syntax-only")
    if fam in {"CanonicalSparse","HDC"} and len(ch)!=1: raise V(f"{w}: single-channel family needs one channel")
    if fam in {"FusionConventional","FusionHDC"} and ch!={"Syntax","ExactNormalForm"}: raise V(f"{w}: fusion family needs both channels")
    fusion=len(ch)==2
    if fusion and ("fusion_policy_sha256" not in a or ret!="Fusion"): raise V(f"{w}: fused arm needs fusion policy + Fusion retriever")
    if not fusion and "fusion_policy_sha256" in a: raise V(f"{w}: non-fusion arm cannot carry fusion policy")
    if ret=="None" and "index_manifest_sha256" in a: raise V(f"{w}: no-retrieval arm cannot carry index")
    if ret!="None" and "index_manifest_sha256" not in a: raise V(f"{w}: retrieval arm needs index")
    if fam=="HDC" and ret not in {"HDC","Control"}: raise V(f"{w}: HDC family needs HDC/control retriever")
    if fam in {"Lexical","CanonicalSparse"} and ret not in {"Conventional","Control"}: raise V(f"{w}: conventional family needs conventional/control retriever")
    if role=="Baseline":
        if ch or fam!="None" or ret!="None" or ctrl!="None" or a["negative_search_memory"] or "parent_arm_id" in a: raise V(f"{w}: invalid baseline")
    elif role=="Control":
        if ctrl=="None" or a["negative_search_memory"] or "parent_arm_id" in a: raise V(f"{w}: invalid control arm")
    elif role=="Treatment":
        if ctrl!="None" or a["negative_search_memory"] or "parent_arm_id" in a: raise V(f"{w}: invalid treatment arm")
    else:
        if not a["negative_search_memory"] or ctrl!="None" or "parent_arm_id" not in a: raise V(f"{w}: invalid memory arm")
        if stage=="Q0RepresentationRetrieval": raise V(f"{w}: memory forbidden at Q0")
    if ctrl=="LexicalRetrieval" and not(role=="Control" and fam=="Lexical" and ch=={"Syntax"}): raise V(f"{w}: bad lexical control")
    if ctrl=="RandomRetrieval" and not(role=="Control" and ret=="Control"): raise V(f"{w}: bad random control")
    if ctrl=="ShuffledHdcVectors" and not(role=="Control" and fam in {"HDC","FusionHDC"}): raise V(f"{w}: bad shuffled-HDC control")
    if ctrl=="PermutedChallengeAssociations" and not(role=="Control" and ret!="None"): raise V(f"{w}: bad permuted control")
    return a

def same(a,b):
    return all(a.get(k)==b.get(k) for k in ("representation_channels","representation_family","representation_sha256","retriever_family","retriever_sha256","index_manifest_sha256","fusion_policy_sha256","control_kind"))

def contrast(c,w,arms,primary):
    if not isinstance(c,dict): raise V(f"{w}: object required")
    closed(c,CON,w); req(c,CON,w)
    cid=text(c["contrast_id"],f"{w}.contrast_id"); l=text(c["left_arm_id"],f"{w}.left_arm_id"); r=text(c["right_arm_id"],f"{w}.right_arm_id")
    if l==r or l not in arms or r not in arms: raise V(f"{w}: bad arm refs")
    d=c["intended_difference"]
    if d not in DIFF or c["primary_endpoint"] not in primary or c["direction"] not in {"TwoSided","LeftGreater","RightGreater"}: raise V(f"{w}: invalid contrast metadata")
    a,b=arms[l],arms[r]; ac,bc=set(a["representation_channels"]),set(b["representation_channels"])
    if d=="RetrievalAugmentation" and {a["retriever_family"]=="None",b["retriever_family"]=="None"}!={True,False}: raise V(f"{w}: retrieval-vs-none required")
    if d=="SyntaxVsNormalForm" and {frozenset(ac),frozenset(bc)}!={frozenset({"Syntax"}),frozenset({"ExactNormalForm"})}: raise V(f"{w}: syntax-vs-normal required")
    if d=="HdcSpecificRepresentation":
        pair={a["representation_family"],b["representation_family"]}
        if ac!=bc or pair not in ({"HDC","CanonicalSparse"},{"HDC","Lexical"},{"FusionHDC","FusionConventional"}): raise V(f"{w}: HDC-vs-conventional with channels fixed required")
    if d=="FusionBenefit" and sorted((len(ac),len(bc)))!=[1,2]: raise V(f"{w}: single-vs-fusion required")
    if d=="RetrieverMechanism":
        if ac!=bc or a["representation_family"]!=b["representation_family"] or a["representation_sha256"]!=b["representation_sha256"] or (a["retriever_family"]==b["retriever_family"] and a["retriever_sha256"]==b["retriever_sha256"]): raise V(f"{w}: representation-fixed retriever change required")
    if d=="NegativeSearchMemory" and (a["negative_search_memory"]==b["negative_search_memory"] or not same(a,b)): raise V(f"{w}: memory-only difference required")
    if d=="Control" and a["arm_role"]!="Control" and b["arm_role"]!="Control": raise V(f"{w}: control arm required")
    return cid

def validate(d):
    if not isinstance(d,dict): raise V("root: object required")
    closed(d,ROOT,"root"); req(d,ROOT,"root")
    if d["schema_version"]!=SCHEMA or d["authority"]!=AUTH or d["stage"] not in STAGES or d["frozen_before_evaluation"] is not True or d["evolutionary_search"] is not False: raise V("root: identity/stage/freeze invariant failed")
    text(d["experiment_id"],"experiment_id"); sha(d["statistical_plan_sha256"],"statistical_plan_sha256")
    s=d["shared_contract"]
    if not isinstance(s,dict): raise V("shared_contract: object required")
    closed(s,SHARED,"shared_contract"); req(s,SHARED,"shared_contract")
    for k in SHARED: sha(s[k],f"shared_contract.{k}")
    budget(d["budget"])
    seeds=d["seeds"]
    if not isinstance(seeds,list) or len(seeds)<3 or any(not isinstance(x,int) or isinstance(x,bool) or x<0 for x in seeds) or len(seeds)!=len(set(seeds)): raise V("seeds: >=3 unique non-negative ints required")
    p=set(uniq(d["primary_endpoints"],"primary_endpoints"))
    if p-ENDPOINTS: raise V("primary_endpoints: unsupported")
    if p&set(uniq(d["secondary_endpoints"],"secondary_endpoints")): raise V("secondary endpoints duplicate primary")
    if d["stage"]=="Q0RepresentationRetrieval":
        if not p<=RETR: raise V("Q0 primary endpoints must be retrieval-native")
        needed=BASE_CHECKS|{"StructuralNeighborShift"}
    else:
        if not p&SEARCH: raise V("Q1+ requires a search primary endpoint")
        needed=BASE_CHECKS|{"StrategyDistributionShift","SearchTrajectoryShift"}
    if needed-set(uniq(d["manipulation_checks"],"manipulation_checks")): raise V("missing manipulation checks")
    if "MajorityStrategy" not in set(uniq(d["strategy_controls"],"strategy_controls")): raise V("MajorityStrategy control required")
    raw=d["arms"]
    if not isinstance(raw,list) or not 3<=len(raw)<=32: raise V("arms: 3..32 required")
    arms={}
    for i,x in enumerate(raw):
        a=arm(x,f"arms[{i}]",d["stage"]); aid=a["arm_id"]
        if aid in arms: raise V("duplicate arm_id")
        arms[aid]=a
    if sum(a["arm_role"]=="Baseline" for a in arms.values())!=1: raise V("exactly one baseline required")
    kinds={a["control_kind"] for a in arms.values() if a["arm_role"]=="Control"}
    if REQ_CONTROLS-kinds: raise V(f"missing control arms {sorted(REQ_CONTROLS-kinds)}")
    if any(a["retriever_family"]!="None" for a in arms.values()):
        b=d["budget"]
        if b["retrieved_items_max"]<=0 or b["retrieval_context_bytes_max"]<=0 or b["retrieval_queries_max"]<=0: raise V("positive retrieval ceilings required")
    for aid,a in arms.items():
        if a["arm_role"]=="MemoryAugmented":
            pid=a["parent_arm_id"]
            if pid not in arms or arms[pid]["negative_search_memory"] or not same(a,arms[pid]): raise V(f"{aid}: invalid memory parent")
    cs=d["planned_contrasts"]
    if not isinstance(cs,list) or not 1<=len(cs)<=64: raise V("planned_contrasts: 1..64 required")
    seen=set()
    for i,c in enumerate(cs):
        cid=contrast(c,f"planned_contrasts[{i}]",arms,p)
        if cid in seen: raise V("duplicate contrast_id")
        seen.add(cid)

def fixture():
    def a(i,role,ch,fam,ret,ctrl="None",fusion=False):
        x={"arm_id":i,"arm_role":role,"representation_channels":ch,"representation_family":fam,"representation_sha256":dg("r"+i),"retriever_family":ret,"retriever_sha256":dg("t"+i),"control_kind":ctrl,"negative_search_memory":False}
        if ret!="None": x["index_manifest_sha256"]=dg("i"+i)
        if fusion: x["fusion_policy_sha256"]=dg("f"+i)
        return x
    arms=[a("A","Baseline",[],"None","None"),a("L","Control",["Syntax"],"Lexical","Conventional","LexicalRetrieval"),a("R","Control",[],"None","Control","RandomRetrieval"),a("S","Treatment",["Syntax"],"CanonicalSparse","Conventional"),a("H","Treatment",["Syntax"],"HDC","HDC"),a("N","Treatment",["ExactNormalForm"],"CanonicalSparse","Conventional"),a("F","Treatment",["Syntax","ExactNormalForm"],"FusionConventional","Fusion",fusion=True),a("SHUF","Control",["Syntax"],"HDC","HDC","ShuffledHdcVectors"),a("PERM","Control",["Syntax"],"CanonicalSparse","Conventional","PermutedChallengeAssociations")]
    return {"schema_version":SCHEMA,"experiment_id":"fixture","authority":AUTH,"stage":"Q0RepresentationRetrieval","frozen_before_evaluation":True,"shared_contract":{k:dg(k) for k in SHARED},"budget":{"accounting_policy_sha256":dg("budget"),"retrieved_items_max":8,"retrieved_item_bytes_max":8192,"retrieval_context_bytes_max":32768,"total_context_bytes_max":65536,"normalized_compute_units_max":100.0,"wall_time_ms_max":60000,"proof_calls_max":0,"solver_calls_max":0,"search_nodes_max":0,"candidate_count_max":32,"retrieval_queries_max":8,"unused_budget_reallocation":False},"statistical_plan_sha256":dg("stats"),"seeds":[11,23,47,89],"primary_endpoints":["StructuralNeighborRecallAtK","MeanReciprocalRank"],"secondary_endpoints":["PositiveSimilarity"],"manipulation_checks":sorted(BASE_CHECKS|{"StructuralNeighborShift"}),"strategy_controls":["MajorityStrategy"],"evolutionary_search":False,"arms":arms,"planned_contrasts":[{"contrast_id":"rvn","left_arm_id":"S","right_arm_id":"A","intended_difference":"RetrievalAugmentation","primary_endpoint":"StructuralNeighborRecallAtK","direction":"TwoSided"},{"contrast_id":"svn","left_arm_id":"S","right_arm_id":"N","intended_difference":"SyntaxVsNormalForm","primary_endpoint":"StructuralNeighborRecallAtK","direction":"TwoSided"},{"contrast_id":"hvc","left_arm_id":"H","right_arm_id":"S","intended_difference":"HdcSpecificRepresentation","primary_endpoint":"MeanReciprocalRank","direction":"TwoSided"},{"contrast_id":"fvs","left_arm_id":"F","right_arm_id":"S","intended_difference":"FusionBenefit","primary_endpoint":"MeanReciprocalRank","direction":"TwoSided"},{"contrast_id":"shuf","left_arm_id":"H","right_arm_id":"SHUF","intended_difference":"Control","primary_endpoint":"MeanReciprocalRank","direction":"TwoSided"}]}

def self_test():
    v=fixture(); validate(v)
    attacks=[lambda d:d.__setitem__("frozen_before_evaluation",False),lambda d:d.__setitem__("evolutionary_search",True),lambda d:d["budget"].__setitem__("unused_budget_reallocation",True),lambda d:d["arms"][3].__setitem__("retrieved_items_max",999),lambda d:d["arms"][6].pop("fusion_policy_sha256"),lambda d:d["arms"][5].__setitem__("normalization_implementation_sha256",dg("evil")),lambda d:d.__setitem__("arms",[a for a in d["arms"] if a["control_kind"]!="RandomRetrieval"]),lambda d:d["planned_contrasts"][1].__setitem__("right_arm_id","H"),lambda d:d["planned_contrasts"][0].__setitem__("primary_endpoint","FalseEquivalentNeighborRate")]
    for f in attacks:
        d=copy.deepcopy(v); f(d)
        try: validate(d)
        except V: continue
        raise AssertionError("adversarial self-test unexpectedly passed")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("path",nargs="?",type=Path); ap.add_argument("--self-test",action="store_true"); z=ap.parse_args()
    if z.self_test: self_test(); print("math-search experiment v2 self-test: PASS"); return 0
    if z.path is None: ap.error("path required unless --self-test")
    try: validate(json.loads(z.path.read_text()))
    except (OSError,json.JSONDecodeError,V) as e: print(f"INVALID: {e}",file=sys.stderr); return 1
    print("VALID"); return 0
if __name__=="__main__": raise SystemExit(main())
