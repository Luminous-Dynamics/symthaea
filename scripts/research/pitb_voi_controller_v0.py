#!/usr/bin/env python3
"""Research-only sequential value-of-information controller for PITB.

Ranks an immediate continuation against a bounded diagnostic followed by a
conditional continuation. It does not execute proofs/probes or authenticate
receipts; its outputs are scheduling proposals, not proof evidence.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, sys
from fractions import Fraction
from pathlib import Path

VERSION = "pitb.voi-decision-request.v0"
RESOURCES = ("tokens", "model_calls", "tool_calls", "wall_ms", "lean_cpu_ms", "cost_microusd")
HYPOTHESES = {
    "missing_formalization_infrastructure", "missing_dependency_route",
    "type_or_instance_mismatch", "incorrect_decomposition_or_invariant",
    "environment_or_toolchain_mismatch", "target_semantics_mismatch",
    "search_policy_exhausted", "unknown",
}
CONTINUATIONS = {"continue_current_target", "retrieve_context", "change_decomposition", "typed_repair", "tactic_probe", "semantic_check", "restart_search"}
DIAGNOSTICS = {"dependency_route_probe", "type_instance_probe", "decomposition_probe", "environment_probe", "target_correspondence_probe", "counterexample_probe"}

class ContractError(ValueError): pass

def fail(message: str): raise ContractError(message)
def obj(x, path):
    if not isinstance(x, dict): fail(f"{path} must be an object")
    return x
def arr(x, path):
    if not isinstance(x, list): fail(f"{path} must be an array")
    return x
def keys(x, required, path):
    if set(x) != set(required): fail(f"{path} keys differ; missing={sorted(set(required)-set(x))}; extra={sorted(set(x)-set(required))}")
def string(x, path):
    if not isinstance(x, str) or not x.strip(): fail(f"{path} must be a nonempty string")
    return x
def integer(x, path, maximum=None):
    if type(x) is not int or x < 0 or (maximum is not None and x > maximum): fail(f"{path} must be a nonnegative integer within bounds")
    return x
def digest(x, path, n=64):
    x = string(x, path)
    if len(x) != n or any(c not in "0123456789abcdef" for c in x): fail(f"{path} must be lowercase hex digest length {n}")
    return x
def canonical_sha(x):
    b = json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(b).hexdigest()
def resources(x, path):
    o = obj(x, path); keys(o, RESOURCES, path)
    return {k: integer(o[k], f"{path}.{k}") for k in RESOURCES}
def cost_units(used, remaining, weights, full_cost):
    if any(used[k] > remaining[k] for k in RESOURCES): fail("action exceeds remaining task budget")
    share = sum((Fraction(weights[k], 1_000_000) * Fraction(used[k], remaining[k]) if remaining[k] else Fraction(0)) for k in RESOURCES)
    return (share * full_cost).__ceil__()

def validate(req):
    r = obj(req, "request")
    keys(r, {"schema_version","decision_id","evaluation_split","target","posterior","policy","remaining_budget","continuation_actions","diagnostic_actions"}, "request")
    if r["schema_version"] != VERSION: fail("unsupported schema_version")
    string(r["decision_id"], "decision_id")
    if r["evaluation_split"] not in {"test", "external_challenge"}: fail("evaluation_split must be test or external_challenge; validation is calibration-only")
    t = obj(r["target"], "target"); keys(t, {"task_id_sha256","repository_commit","statement_sha256","proof_state_sha256","environment_sha256"}, "target")
    for k in ("task_id_sha256","statement_sha256","proof_state_sha256","environment_sha256"): digest(t[k], f"target.{k}")
    digest(t["repository_commit"], "target.repository_commit", 40)
    p = obj(r["posterior"], "posterior"); keys(p, {"model_id","model_version","calibration_split","calibration_manifest_sha256","calibration_receipt_sha256","sample_count","hypotheses"}, "posterior")
    string(p["model_id"], "posterior.model_id"); string(p["model_version"], "posterior.model_version")
    if p["calibration_split"] != "validation": fail("calibration must use validation split")
    digest(p["calibration_manifest_sha256"], "posterior.calibration_manifest_sha256"); digest(p["calibration_receipt_sha256"], "posterior.calibration_receipt_sha256")
    if integer(p["sample_count"], "posterior.sample_count") < 200: fail("at least 200 validation observations required")
    hs = arr(p["hypotheses"], "posterior.hypotheses")
    if not 2 <= len(hs) <= 8: fail("posterior must contain 2..8 hypotheses")
    prior = {}
    for i, row in enumerate(hs):
        q = obj(row, f"posterior.hypotheses[{i}]"); keys(q, {"hypothesis_id","probability_ppm"}, "posterior hypothesis")
        h = string(q["hypothesis_id"], "hypothesis_id")
        if h not in HYPOTHESES or h in prior: fail("unknown or duplicate hypothesis")
        prior[h] = integer(q["probability_ppm"], f"{h}.probability_ppm", 1_000_000)
    if any(v == 0 for v in prior.values()) or sum(prior.values()) != 1_000_000: fail("posterior must be positive and sum to 1,000,000 ppm")
    pol = obj(r["policy"], "policy"); keys(pol, {"policy_id","policy_sha256","verified_success_value_units","minimum_net_voi_gain_units","full_budget_cost_units","shadow_weights_ppm"}, "policy")
    if pol["policy_id"] != "pitb-sequential-voi-v0": fail("unsupported policy")
    claimed = digest(pol["policy_sha256"], "policy.policy_sha256"); ph = dict(pol); ph.pop("policy_sha256")
    if canonical_sha(ph) != claimed: fail("policy_sha256 does not match canonical policy bytes")
    value = integer(pol["verified_success_value_units"], "verified_success_value_units")
    full = integer(pol["full_budget_cost_units"], "full_budget_cost_units")
    threshold = integer(pol["minimum_net_voi_gain_units"], "minimum_net_voi_gain_units")
    if not value or not full: fail("verified value and full budget cost must be positive")
    w = obj(pol["shadow_weights_ppm"], "shadow_weights_ppm"); keys(w, RESOURCES, "shadow_weights_ppm")
    weights = {k: integer(w[k], f"shadow_weights_ppm.{k}", 1_000_000) for k in RESOURCES}
    if sum(weights.values()) != 1_000_000: fail("resource weights must sum to 1,000,000 ppm")
    budget = resources(r["remaining_budget"], "remaining_budget")
    cont = arr(r["continuation_actions"], "continuation_actions"); diag = arr(r["diagnostic_actions"], "diagnostic_actions")
    if not 1 <= len(cont) <= 32 or len(diag) > 16: fail("continuation count must be 1..32 and diagnostics <=16")
    seen = set(); crows = []
    for i, a in enumerate(cont):
        path=f"continuation_actions[{i}]"; a=obj(a,path); keys(a,{"action_id","kind","prediction_evidence_sha256","expected_resources","success_by_hypothesis"},path)
        aid=string(a["action_id"],path+".action_id")
        if aid in seen: fail("action IDs must be globally unique")
        seen.add(aid)
        if a["kind"] not in CONTINUATIONS: fail("unknown continuation kind")
        ev=digest(a["prediction_evidence_sha256"],path+".prediction_evidence_sha256"); used=resources(a["expected_resources"],path+".expected_resources")
        if not any(used.values()): fail("action must consume at least one resource")
        c=cost_units(used,budget,weights,full); rows=arr(a["success_by_hypothesis"],path+".success_by_hypothesis")
        if len(rows)!=len(prior): fail("continuation must have one outcome estimate per hypothesis")
        succ={}
        for row in rows:
            q=obj(row,"success estimate"); keys(q,{"hypothesis_id","mean_probability_ppm","lcb_probability_ppm"},"success estimate")
            h=string(q["hypothesis_id"],"success.hypothesis_id")
            if h not in prior or h in succ: fail("unknown or duplicated conditional hypothesis")
            mean=integer(q["mean_probability_ppm"],"mean",1_000_000); lcb=integer(q["lcb_probability_ppm"],"lcb",1_000_000)
            if lcb>mean: fail("LCB cannot exceed mean")
            succ[h]={"mean":mean,"lcb":lcb}
        if set(succ)!=set(prior): fail("missing conditional success hypothesis")
        crows.append({"action_id":aid,"kind":a["kind"],"evidence_sha256":ev,"resources":used,"cost":c,"success":succ})
    drows=[]
    for i, a in enumerate(diag):
        path=f"diagnostic_actions[{i}]"; a=obj(a,path); keys(a,{"action_id","kind","prediction_evidence_sha256","expected_resources","outcomes"},path)
        aid=string(a["action_id"],path+".action_id")
        if aid in seen: fail("action IDs must be globally unique")
        seen.add(aid)
        if a["kind"] not in DIAGNOSTICS: fail("unknown diagnostic kind")
        ev=digest(a["prediction_evidence_sha256"],path+".prediction_evidence_sha256"); used=resources(a["expected_resources"],path+".expected_resources")
        if not any(used.values()): fail("diagnostic must consume at least one resource")
        c=cost_units(used,budget,weights,full); outcomes=arr(a["outcomes"],path+".outcomes")
        if not 2<=len(outcomes)<=16: fail("diagnostic requires 2..16 outcomes")
        likelihoods={}
        for out in outcomes:
            q=obj(out,"diagnostic outcome"); keys(q,{"outcome_id","likelihood_by_hypothesis"},"diagnostic outcome")
            oid=string(q["outcome_id"],"outcome_id")
            if oid in likelihoods: fail("duplicate diagnostic outcome")
            ls=arr(q["likelihood_by_hypothesis"],"likelihoods"); like={}
            for row in ls:
                z=obj(row,"likelihood row"); keys(z,{"hypothesis_id","probability_ppm"},"likelihood row")
                h=string(z["hypothesis_id"],"likelihood.hypothesis_id")
                if h not in prior or h in like: fail("unknown or duplicate likelihood hypothesis")
                like[h]=integer(z["probability_ppm"],"likelihood.probability_ppm",1_000_000)
            if set(like)!=set(prior): fail("incomplete likelihood row")
            likelihoods[oid]=like
        for h in prior:
            if sum(x[h] for x in likelihoods.values())!=1_000_000: fail(f"likelihoods for {h} must sum to one")
        drows.append({"action_id":aid,"kind":a["kind"],"evidence_sha256":ev,"resources":used,"cost":c,"outcomes":likelihoods})
    return {"decision_id":r["decision_id"],"target":t,"prior":{h:Fraction(v,1_000_000) for h,v in prior.items()},"value":value,"threshold":threshold,"budget":budget,"continuations":crows,"diagnostics":drows}

def best(actions, posterior, value):
    ranked=[]
    for a in actions:
        lcb=sum(posterior[h]*Fraction(a["success"][h]["lcb"],1_000_000) for h in posterior)
        mean=sum(posterior[h]*Fraction(a["success"][h]["mean"],1_000_000) for h in posterior)
        utility=lcb*value-a["cost"]
        if utility>0: ranked.append((utility,a["action_id"],a,mean,lcb))
    ranked.sort(key=lambda x:(-x[0],x[1]))
    return ranked[0] if ranked else None

def decide(v):
    direct=best(v["continuations"],v["prior"],v["value"])
    direct_value=direct[0] if direct else Fraction(0)
    candidates=[]
    for d in v["diagnostics"]:
        feasible=[a for a in v["continuations"] if all(d["resources"][k]+a["resources"][k]<=v["budget"][k] for k in RESOURCES)]
        expected=Fraction(0); reports=[]
        for oid, like in d["outcomes"].items():
            mass=sum(v["prior"][h]*Fraction(like[h],1_000_000) for h in v["prior"])
            post={h:(v["prior"][h]*Fraction(like[h],1_000_000)/mass if mass else Fraction(0)) for h in v["prior"]}
            follow=best(feasible,post,v["value"]) if mass else None
            val=follow[0] if follow else Fraction(0); expected+=mass*val
            reports.append({"outcome_id":oid,"probability_ppm":(mass*1_000_000).__floor__(),"posterior_ppm":{h:(p*1_000_000).__floor__() for h,p in post.items()},"best_continuation_action_id":follow[2]["action_id"] if follow else None,"best_continuation_expected_resources":follow[2]["resources"] if follow else None,"best_continuation_lcb_value_units":val.__floor__(),"best_continuation_mean_success_probability_ppm":(follow[3]*1_000_000).__floor__() if follow else 0})
        net=expected-d["cost"]; candidates.append((net-direct_value,d["action_id"],d,net,reports))
    candidates.sort(key=lambda x:(-x[0],x[1])); top=candidates[0] if candidates else None
    if top and top[0]>v["threshold"] and top[3]>0:
        decision="diagnose"; d=top[2]; selected={"action_id":d["action_id"],"kind":d["kind"],"expected_shadow_cost_units":d["cost"],"expected_resources":d["resources"],"expected_net_value_after_probe_units":top[3].__floor__()}; reports=top[4]; gain=top[0]
    elif direct:
        decision="execute"; a=direct[2]; selected={"action_id":a["action_id"],"kind":a["kind"],"expected_shadow_cost_units":a["cost"],"expected_lcb_success_probability_ppm":(direct[4]*1_000_000).__floor__(),"expected_net_value_units":direct[0].__floor__()}; reports=[]; gain=top[0] if top else Fraction(0)
    else:
        decision="stop"; selected=None; reports=[]; gain=top[0] if top else Fraction(0)
    return {"schema_version":"pitb.voi-decision-result.v0","decision_id":v["decision_id"],"target":v["target"],"decision":decision,"selected_action":selected,"best_direct_action_id":direct[2]["action_id"] if direct else None,"best_direct_net_value_units":direct_value.__floor__(),"best_direct_mean_success_probability_ppm":(direct[3]*1_000_000).__floor__() if direct else 0,"best_diagnostic_net_gain_over_direct_units":gain.__floor__(),"minimum_net_voi_gain_units":v["threshold"],"diagnostic_outcome_continuations":reports,"selection_rule":"select a diagnostic only when expected post-observation best continuation value minus diagnostic cost exceeds best immediate continuation value by the registered threshold; otherwise execute the best positive-value continuation; stop when neither has positive value","proof_or_probe_executed":False,"semantic_success_claimed":False,"evidence_authentication_claimed":False}

def self_test(path):
    req=json.loads(path.read_text(encoding="utf-8")); v=validate(req); result=decide(v)
    assert result["decision"]=="diagnose" and result["selected_action"]["action_id"]=="diagnose-route-vs-type", result
    follow={x["outcome_id"]:x["best_continuation_action_id"] for x in result["diagnostic_outcome_continuations"]}
    assert follow["type_mismatch"]=="change-abstraction" and follow["route_missing"]=="retrieve-route",follow
    def reject(label, candidate):
        try: validate(candidate)
        except ContractError: return
        raise AssertionError("accepted negative control: "+label)
    x=copy.deepcopy(req);x["posterior"]["hypotheses"][0]["probability_ppm"]+=1;reject("posterior drift",x)
    x=copy.deepcopy(req);x["evaluation_split"]="validation";reject("split leakage",x)
    x=copy.deepcopy(req);x["continuation_actions"][0]["success_by_hypothesis"].pop();reject("missing hypothesis row",x)
    x=copy.deepcopy(req);x["continuation_actions"][0]["expected_resources"]["tokens"]=100001;reject("budget overrun",x)
    x=copy.deepcopy(req);x["policy"]["minimum_net_voi_gain_units"]+=1;reject("policy digest drift",x)
    x=copy.deepcopy(req)
    for out in x["diagnostic_actions"][0]["outcomes"]:
        for row in out["likelihood_by_hypothesis"]: row["probability_ppm"]=500000
    y=decide(validate(x));assert y["decision"]=="execute" and y["selected_action"]["action_id"]=="retrieve-route",y
    x=copy.deepcopy(req)
    for a in x["continuation_actions"]: a["expected_resources"]["tokens"]=98000
    y=decide(validate(x));assert y["decision"]=="execute" and not y["diagnostic_outcome_continuations"],y
    x=copy.deepcopy(req);x["diagnostic_actions"]=[];y=decide(validate(x));assert y["decision"]=="execute",y
    x=copy.deepcopy(req);x["diagnostic_actions"]=[]
    for a in x["continuation_actions"]:
        for row in a["success_by_hypothesis"]: row["mean_probability_ppm"]=row["lcb_probability_ppm"]=0
    pol=x["policy"]; old=pol.pop("policy_sha256");pol["policy_sha256"]=canonical_sha(pol)
    y=decide(validate(x));assert y["decision"]=="stop",y
    print("PITB_VOI_SELF_TEST_PASS")
    print("- diagnostic chosen only when expected conditional continuation value exceeds immediate value")
    print("- outcome posteriors select the intended next action")
    print("- uninformative diagnostic, budget overrun, malformed posterior, split leakage and policy drift rejected")
    print("- execute and stop dispositions verified; no proof/probe execution or success claim")

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--input",type=Path);ap.add_argument("--output",type=Path);ap.add_argument("--self-test",action="store_true");ap.add_argument("--fixture",type=Path)
    a=ap.parse_args()
    if a.self_test:
        if a.input or a.output: ap.error("--self-test excludes --input/--output")
        fixture=a.fixture or Path(__file__).resolve().parents[2]/"data/benchmarks/pitb-voi-controller-v0/fixtures/diagnostic-v0.synthetic.json";self_test(fixture);return 0
    if not a.input or not a.output: ap.error("--input and --output are required")
    raw=a.input.read_bytes();result=decide(validate(json.loads(raw)));a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"PITB_VOI_DECISION_PASS: decision={result['decision']} input_sha256={hashlib.sha256(raw).hexdigest()}");return 0
if __name__=="__main__":
    try: raise SystemExit(main())
    except (ContractError,json.JSONDecodeError,OSError) as e: print(f"PITB_VOI_DECISION_FAIL: {e}",file=sys.stderr);raise SystemExit(2)
