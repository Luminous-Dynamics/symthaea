#!/usr/bin/env python3
"""Risk-aware memory escalation benchmark. Standard library only; synthetic research lane."""
from __future__ import annotations
import argparse, hashlib, json, math, random, statistics, sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROTOCOL = "BROCA-RISK-ESCALATION-V1"
SEED = 20261028
FEATURES = ["similarity", "support", "freshness", "provenance", "conflict", "scope_match",
            "coverage", "uncertainty", "revision_signal", "revocation_signal", "source_independence",
            "absence_evidence"]
CLASSES = ["fresh", "stale", "conflict", "absent", "scope_shift", "distractor", "revoked"]
CLASS_P = {
    "iid": [0.50, .13, .10, .11, .08, .06, .02],
    "high_similarity_distractor": [.35, .10, .10, .10, .08, .25, .02],
    "stale_revision": [.32, .28, .15, .10, .07, .05, .03],
    "scope_shift": [.30, .10, .10, .10, .25, .08, .07],
    "compound_shift": [.25, .20, .15, .12, .18, .07, .03],
}
MEANS = {
 "fresh":       [.82,.89,.94,.90,.06,.93,.82,.14,.04,.01,.78,.02],
 "stale":       [.91,.74,.18,.88,.47,.90,.78,.48,.80,.12,.56,.01],
 "conflict":    [.88,.70,.67,.77,.86,.87,.82,.78,.35,.03,.62,.01],
 "absent":      [.78,.24,.70,.70,.12,.85,.22,.72,.08,.04,.14,.10],
 "scope_shift": [.89,.69,.83,.85,.18,.16,.74,.64,.08,.02,.53,.01],
 "distractor":  [.97,.38,.91,.89,.08,.91,.17,.69,.03,.02,.13,.02],
 "revoked":     [.88,.73,.25,.91,.40,.93,.86,.50,.87,.91,.82,.01],
}
DEEP_MEANS = {
 "fresh":       [.86,.93,.95,.92,.04,.94,.90,.10,.04,.01,.82,.01],
 "stale":       [.88,.90,.91,.90,.12,.93,.90,.16,.96,.10,.83,.01],
 "conflict":    [.85,.86,.83,.84,.23,.90,.92,.28,.42,.03,.78,.01],
 "absent":      [.42,.08,.71,.75,.08,.89,.94,.80,.08,.03,.12,.94],
 "scope_shift": [.85,.87,.91,.88,.12,.91,.86,.22,.08,.02,.81,.01],
 "distractor":  [.91,.82,.88,.90,.12,.92,.82,.30,.06,.02,.68,.01],
 "revoked":     [.83,.69,.16,.93,.38,.94,.90,.56,.94,.98,.82,.02],
}
CHEAP_P = {"fresh":.94,"stale":.13,"conflict":.47,"absent":.025,"scope_shift":.16,"distractor":.10,"revoked":.015}
DEEP_P  = {"fresh":.97,"stale":.89,"conflict":.82,"absent":.01,"scope_shift":.76,"distractor":.68,"revoked":.015}
TOKEN_COST = {"answer": 450, "escalate": 1050, "abstain": 100}
TOKEN_UTILITY_COST = 0.00018
HARM_UTILITY = -2.0
GOOD_UTILITY = 1.0
ABSTAIN_GOOD = 0.25
ABSTAIN_BAD = -0.60
REGIMES = ["iid", "high_similarity_distractor", "stale_revision", "scope_shift", "compound_shift"]


def clamp(x: float) -> float: return max(0.0, min(1.0, x))
def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()
def canonical(obj: Any) -> str: return json.dumps(obj, sort_keys=True, separators=(",",":"), ensure_ascii=False)

class Case(dict):
    """Evaluator-only record. Policies receive only case['x']; truth never enters policy APIs."""
    pass


def noise(rng: random.Random, sigma: float) -> float:
    return rng.gauss(0.0, sigma)

def make_features(cls: str, regime: str, rng: random.Random, deep: bool=False) -> dict[str,float]:
    means = (DEEP_MEANS if deep else MEANS)[cls]
    # Compound shift obscures some of the signals that would normally reveal risk.
    sigma = .075 if regime in ("iid", "high_similarity_distractor", "stale_revision") else .10
    if regime == "compound_shift": sigma = .16
    vals = {key: clamp(means[i] + noise(rng, sigma)) for i,key in enumerate(FEATURES)}
    if regime in ("scope_shift", "compound_shift") and cls == "scope_shift":
        # Deliberate calibration shift: wrong-scope memories look more locally plausible.
        vals["scope_match"] = clamp((.56 if regime == "scope_shift" else .66) + noise(rng, .12 if regime=="scope_shift" else .18))
    if regime == "compound_shift" and cls in ("stale", "revoked"):
        vals["freshness"] = clamp(vals["freshness"] + .19)
        vals["revision_signal"] = clamp(vals["revision_signal"] - .20)
    if regime == "high_similarity_distractor" and cls == "distractor":
        vals["similarity"] = clamp(.97 + noise(rng,.025))
    return vals

def utility_for(action: str, should_answer: bool, correct: bool) -> tuple[float,bool,bool]:
    """Return (net utility, task success, harmful action). Labels are evaluator-only."""
    cost = TOKEN_COST[action] * TOKEN_UTILITY_COST
    if action == "abstain":
        ok = not should_answer
        return (ABSTAIN_GOOD if ok else ABSTAIN_BAD) - cost, ok, False
    # Answer is harmful when unsupported/invalid, or when the emitted answer is wrong.
    harmful = (not should_answer) or (not correct)
    ok = should_answer and correct
    return (GOOD_UTILITY if ok else HARM_UTILITY) - cost, ok, harmful

def evidence_score(x: dict[str,float]) -> float:
    # Hand-authored, interpretable score used only as a baseline.
    return clamp(.18*x["support"] + .12*x["freshness"] + .08*x["provenance"] +
                 .12*x["scope_match"] + .16*x["coverage"] + .08*x["source_independence"] +
                 .10*(1-x["conflict"]) + .06*(1-x["uncertainty"]) + .05*(1-x["revocation_signal"]) +
                 .05*(1-x["absence_evidence"]))

def direct_q(x: dict[str,float]) -> list[float]:
    return [1.0, *[x[k] for k in FEATURES],
            x["similarity"]*x["conflict"], x["support"]*(1-x["freshness"]),
            x["support"]*(1-x["scope_match"]), x["support"]*x["coverage"],
            x["freshness"]*x["revocation_signal"], x["similarity"]*(1-x["coverage"]),
            x["conflict"]*x["uncertainty"], x["scope_match"]*x["support"]]

def sigmoid(v: float) -> float:
    if v > 30: return 1.0
    if v < -30: return 0.0
    return 1.0/(1.0+math.exp(-v))

def solve_linear(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve a square system using rank-aware Gauss-Jordan elimination.

    Singular columns become explicit free variables (set to zero). The solver
    never destroys a row or fabricates a pivot. Inconsistent zero rows raise.
    Ridge fitting adds positive diagonal regularization, so its normal systems
    should have full rank; the free-variable path is principally defensive.
    """
    n=len(b)
    if len(a)!=n or any(len(row)!=n for row in a):
        raise ValueError("solve_linear requires a square matrix and matching RHS")
    m=[list(map(float,row)) + [float(b[i])] for i,row in enumerate(a)]
    pivot_row=0
    pivot_cols=[]
    eps=1e-12
    for col in range(n):
        if pivot_row>=n: break
        pivot=max(range(pivot_row,n),key=lambda r:abs(m[r][col]))
        if abs(m[pivot][col]) < eps:
            continue
        m[pivot_row],m[pivot]=m[pivot],m[pivot_row]
        div=m[pivot_row][col]
        for j in range(col,n+1): m[pivot_row][j]/=div
        for r in range(n):
            if r==pivot_row: continue
            factor=m[r][col]
            if factor:
                for j in range(col,n+1): m[r][j]-=factor*m[pivot_row][j]
        pivot_cols.append(col)
        pivot_row+=1
    for r in range(pivot_row,n):
        if all(abs(m[r][j])<eps for j in range(n)) and abs(m[r][n])>=eps:
            raise ValueError("linear system is inconsistent")
    solution=[0.0]*n
    for row,col in enumerate(pivot_cols): solution[col]=m[row][n]
    return solution

def fit_ridge(rows: list[list[float]], ys: list[float], lam: float=4.0) -> list[float]:
    n=len(rows[0]); xtx=[[0.0]*n for _ in range(n)]; xty=[0.0]*n
    for x,y in zip(rows,ys):
        for i,xi in enumerate(x):
            xty[i]+=xi*y
            for j,xj in enumerate(x): xtx[i][j]+=xi*xj
    for i in range(1,n): xtx[i][i]+=lam
    xtx[0][0]+=1e-8
    return solve_linear(xtx,xty)

def dot(a: list[float], b: list[float]) -> float: return sum(x*y for x,y in zip(a,b))


def make_case(rng: random.Random, idx: int, regime: str, split: str, group_size:int=5) -> Case:
    cls=rng.choices(CLASSES,weights=CLASS_P[regime],k=1)[0]
    x=make_features(cls,regime,rng,False); deep_x=make_features(cls,regime,rng,True)
    should_answer=cls not in ("absent","revoked")
    cheap_correct=rng.random() < CHEAP_P[cls]
    deep_correct=rng.random() < DEEP_P[cls]
    # Fixed second-stage controller uses only deep evidence, never the latent label.
    ds=evidence_score(deep_x)
    deep_action="abstain" if (deep_x["revocation_signal"]>=.72 or (deep_x["absence_evidence"]>=.70 and deep_x["support"]<.30)) else ("answer" if ds>=.57 and deep_x["conflict"]<.67 else "abstain")
    deep_gross_success = ((deep_action == "answer" and should_answer and deep_correct) or
                          (deep_action == "abstain" and not should_answer))
    deep_harm = bool(deep_action == "answer" and ((not should_answer) or (not deep_correct)))
    if deep_action == "abstain":
        deep_gross = ABSTAIN_GOOD if not should_answer else ABSTAIN_BAD
    else:
        deep_gross = GOOD_UTILITY if deep_action == "answer" and should_answer and deep_correct else HARM_UTILITY
    # Escalation cost replaces the ordinary answer/abstain cost: count the full
    # second-stage call once, rather than accidentally double-charging it.
    deep_util = deep_gross - TOKEN_COST["escalate"] * TOKEN_UTILITY_COST
    cheap_util,cheap_success,cheap_harm=utility_for("answer",should_answer,cheap_correct)
    abst_util,abst_success,abst_harm=utility_for("abstain",should_answer,False)
    return Case(case_id=f"{split}-{regime}-{idx:06d}",group_id=f"{split}-{regime}-g{idx//group_size:05d}",split=split,regime=regime,
                x=x,deep_x=deep_x,latent_class=cls,should_answer=should_answer,cheap_correct=cheap_correct,deep_correct=deep_correct,
                outcomes={"answer":{"utility":cheap_util,"success":cheap_success,"harm":cheap_harm},
                          "escalate":{"utility":deep_util,"success":deep_gross_success,"harm":deep_harm},
                          "abstain":{"utility":abst_util,"success":abst_success,"harm":abst_harm}},
                deep_action=deep_action)


def train_models(dev: list[Case]) -> dict[str,Any]:
    xx=[direct_q(c["x"]) for c in dev]
    utility={}; harm={}
    for action in ("answer","escalate","abstain"):
        utility[action]=fit_ridge(xx,[c["outcomes"][action]["utility"] for c in dev],lam=18.0)
        harm[action]=fit_ridge(xx,[float(c["outcomes"][action]["harm"]) for c in dev],lam=30.0)
    return {"utility":utility,"harm":harm}

def wilson_upper(k: int, n: int, z: float=1.96) -> float:
    """Two-sided 95% Wilson upper endpoint, used as a conservative calibration bound."""
    if n <= 0: return 1.0
    p=k/n; z2=z*z
    center=p+z2/(2*n)
    spread=z*math.sqrt((p*(1-p)/n)+(z2/(4*n*n)))
    return min(1.0,(center+spread)/(1+z2/n))

def calibrate_risk_thresholds(calibration: list[Case], models: dict[str,Any], risk_budget: float=.05, min_selected: int=50) -> dict[str,Any]:
    """Choose action-specific score cutoffs using a held-out calibration split only.

    This empirical control assumes calibration/test exchangeability. It is not a
    distribution-free guarantee under the deliberately shifted test regimes.
    """
    result={}
    for action in ("answer","escalate"):
        scored=[]
        for c in calibration:
            x=direct_q(c["x"])
            pred=clamp(dot(x,models["harm"][action]))
            scored.append((pred,int(c["outcomes"][action]["harm"]),float(c["outcomes"][action]["utility"])))
        scored.sort(key=lambda z:z[0])
        best=None; k=0
        for i,(pred,harm,utility) in enumerate(scored,1):
            k += harm
            if i < min_selected: continue
            # Evaluate only end-of-tie prefixes, so threshold == prediction is faithful.
            if i < len(scored) and abs(scored[i][0]-pred) < 1e-12: continue
            upper=wilson_upper(k,i)
            if upper <= risk_budget:
                candidate={"threshold":pred,"selected":i,"observed_harm":k,"observed_rate":k/i,"wilson_upper_95":upper,
                           "mean_net_utility":statistics.mean(v[2] for v in scored[:i])}
                if best is None or candidate["selected"] > best["selected"]: best=candidate
        if best is None: best={"threshold":-1.0,"selected":0,"observed_harm":0,"observed_rate":0.0,"wilson_upper_95":1.0,"mean_net_utility":None}
        result[action]=best
    result["risk_budget"]=risk_budget
    result["confidence_method"]="95% Wilson upper endpoint on held-out calibration cases; empirical, not distribution-free under shift"
    return result

def choose_policy_action(policy: str, x: dict[str,float], cfg: dict[str,Any], models: dict[str,Any]|None=None) -> str:
    """Policy API takes observable features only; do not pass a Case/truth object here."""
    if policy == "always_answer": return "answer"
    if policy == "always_abstain": return "abstain"
    if policy == "always_escalate": return "escalate"
    if policy == "similarity_threshold": return "answer" if x["similarity"] >= cfg["similarity_threshold"] else "abstain"
    if policy == "evidence_threshold": return "answer" if evidence_score(x) >= cfg["evidence_threshold"] else "abstain"
    q=direct_q(x)
    if policy == "learned_value":
        return max(("answer","escalate","abstain"),key=lambda a:dot(q,models["utility"][a]))
    if policy == "risk_budgeted":
        allowable=["abstain"]
        for a in ("answer","escalate"):
            # Point-estimated risk; useful comparator, not a statistical guarantee.
            if clamp(dot(q,models["harm"][a])) <= cfg["risk_budget"]: allowable.append(a)
        return max(allowable,key=lambda a:dot(q,models["utility"][a]))
    if policy.startswith("risk_control_"):
        allowable=["abstain"]
        cal_thresholds=cfg["calibrated_risk_thresholds_by_budget"][policy]
        for a in ("answer","escalate"):
            threshold=cal_thresholds[a]["threshold"]
            if threshold >= 0 and clamp(dot(q,models["harm"][a])) <= threshold: allowable.append(a)
        return max(allowable,key=lambda a:dot(q,models["utility"][a]))
    if policy == "sequential_rule":
        score=evidence_score(x)
        if score >= cfg["direct_threshold"]: return "answer"
        needs_deeper = (x["conflict"] >= cfg["conflict_trigger"] or
                        x["freshness"] <= cfg["freshness_trigger"] or
                        x["scope_match"] <= cfg["scope_trigger"] or
                        (x["similarity"] >= .78 and x["support"] < .60 and x["coverage"] < .48))
        if needs_deeper and x["revocation_signal"] < .78: return "escalate"
        return "abstain"
    raise ValueError(f"unknown policy: {policy}")

RISK_BUDGETS=[.01,.02,.035,.05,.075,.10]
RISK_POLICY_NAMES={b:f"risk_control_{int(round(b*1000)):02d}permille" for b in RISK_BUDGETS}
POLICIES=["always_answer","always_abstain","always_escalate","similarity_threshold","evidence_threshold","sequential_rule","learned_value","risk_budgeted"] + list(RISK_POLICY_NAMES.values())

def run_policy(case: Case, policy: str, cfg: dict[str,Any], models: dict[str,Any]|None) -> dict[str,Any]:
    action=choose_policy_action(policy,case["x"],cfg,models)
    o=case["outcomes"][action]
    token=TOKEN_COST[action]
    return {"protocol_id":PROTOCOL,"case_id":case["case_id"],"group_id":case["group_id"],"split":case["split"],
            "regime":case["regime"],"policy":policy,"action":action,"task_success":int(o["success"]),"harmful":int(o["harm"]),
            "net_utility":round(o["utility"],6),"tokens":token,"latent_class":case["latent_class"],
            "should_answer":int(case["should_answer"]),"x":case["x"],"deep_action":case["deep_action"] if action=="escalate" else None}

def grouped_bootstrap_delta(rows_a: list[dict[str,Any]], rows_b: list[dict[str,Any]], metric: str, reps:int=600, seed:int=SEED) -> tuple[float,float,float]:
    # Paired cluster bootstrap; both arms are matched by case and group.
    a={r["case_id"]:r for r in rows_a}; b={r["case_id"]:r for r in rows_b}
    ids=sorted(set(a)&set(b)); per_group=defaultdict(list)
    for i in ids: per_group[a[i]["group_id"]].append(a[i][metric]-b[i][metric])
    groups=sorted(per_group)
    # All benchmark groups have fixed size by construction; cluster-mean resampling
    # is algebraically equivalent and avoids repeatedly expanding each cluster.
    group_means=[statistics.mean(per_group[g]) for g in groups]
    obs=statistics.mean(group_means) if group_means else 0.0
    rng=random.Random(seed); samples=[]; n_groups=len(group_means)
    for _ in range(reps):
        samples.append(sum(group_means[rng.randrange(n_groups)] for _ in range(n_groups))/n_groups if n_groups else 0.0)
    samples.sort()
    return obs,samples[int(.025*reps)],samples[min(reps-1,int(.975*reps))]

def summarize(rows: list[dict[str,Any]]) -> dict[str,Any]:
    n=len(rows)
    return {"n":n,"task_success":round(sum(r["task_success"] for r in rows)/n,6) if n else None,
            "harmful_rate":round(sum(r["harmful"] for r in rows)/n,6) if n else None,
            "mean_tokens":round(sum(r["tokens"] for r in rows)/n,3) if n else None,
            "net_utility":round(statistics.mean(r["net_utility"] for r in rows),6) if n else None,
            "escalation_rate":round(sum(r["action"]=="escalate" for r in rows)/n,6) if n else None,
            "answer_rate":round(sum(r["action"]=="answer" for r in rows)/n,6) if n else None,
            "abstention_rate":round(sum(r["action"]=="abstain" for r in rows)/n,6) if n else None}

def tune_threshold(dev: list[Case], field: str, score_fn, cfg_key: str, max_tokens:float=700.0, max_harm:float=.055) -> float:
    candidates=[round(.20+i*.02,3) for i in range(40)]
    best=None
    for t in candidates:
        rr=[]
        for c in dev:
            act="answer" if score_fn(c["x"]) >= t else "abstain"
            o=c["outcomes"][act]
            rr.append({"action":act,"task_success":int(o["success"]),"harmful":int(o["harm"]),"tokens":TOKEN_COST[act],"net_utility":o["utility"]})
        s=summarize(rr)
        if s["mean_tokens"]>max_tokens or s["harmful_rate"]>max_harm: continue
        key=(s["net_utility"],s["task_success"],-s["harmful_rate"],-t)
        if best is None or key>best[0]: best=(key,t)
    return best[1] if best else .99

def tune_sequential(dev:list[Case]) -> dict[str,float]:
    best=None
    for dt in [.42,.46,.50,.54,.58,.62,.66,.70]:
      for ct in [.60,.68,.76,.84]:
       for ft in [.22,.32,.42,.52]:
        cfg={"direct_threshold":dt,"conflict_trigger":ct,"freshness_trigger":ft,"scope_trigger":.44}
        r=[]
        for c in dev:
            a=choose_policy_action("sequential_rule",c["x"],cfg)
            o=c["outcomes"][a]
            r.append({"action":a,"task_success":int(o["success"]),"harmful":int(o["harm"]),"tokens":TOKEN_COST[a],"net_utility":o["utility"]})
        s=summarize(r)
        if s["mean_tokens"]>780 or s["harmful_rate"]>.055: continue
        key=(s["net_utility"],s["task_success"],-s["harmful_rate"],-s["mean_tokens"])
        if best is None or key>best[0]: best=(key,cfg)
    return best[1] if best else {"direct_threshold":.99,"conflict_trigger":.7,"freshness_trigger":.3,"scope_trigger":.44}

def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--out",required=True); ap.add_argument("--seed",type=int,default=SEED); ap.add_argument("--dev-cases",type=int,default=3000); ap.add_argument("--test-cases-per-regime",type=int,default=2000)
    args=ap.parse_args(); out=Path(args.out)
    if out.exists() and any(out.iterdir()): print(f"Refusing to overwrite non-empty output directory: {out}",file=sys.stderr); return 2
    out.mkdir(parents=True,exist_ok=True)
    rng=random.Random(args.seed)
    dev=[make_case(rng,i,"iid","dev") for i in range(args.dev_cases)]
    # Entire groups are separated: train first, calibrate/tune second; test regimes are unseen.
    cut=max(1,(len(dev)//10)*5)
    train_dev=dev[:cut]; calibration=dev[cut:]
    models=train_models(train_dev)
    cfg={"risk_budget":.035}
    cfg["similarity_threshold"]=tune_threshold(calibration,"similarity",lambda x:x["similarity"],"similarity_threshold")
    cfg["evidence_threshold"]=tune_threshold(calibration,"evidence",evidence_score,"evidence_threshold")
    cfg.update(tune_sequential(calibration))
    cfg["calibrated_risk_thresholds_by_budget"]={RISK_POLICY_NAMES[b]:calibrate_risk_thresholds(calibration,models,risk_budget=b,min_selected=50) for b in RISK_BUDGETS}
    cases=[]
    for regime in REGIMES:
        cases.extend(make_case(rng,i,regime,"test") for i in range(args.test_cases_per_regime))
    model_counts={key:{a:len(v) for a,v in models[key].items()} for key in models}
    config_obj={"protocol_id":PROTOCOL,"seed":args.seed,"dev_cases":args.dev_cases,"training_cases":len(train_dev),"calibration_cases":len(calibration),
                "test_cases_per_regime":args.test_cases_per_regime,"regimes":REGIMES,"policies":POLICIES+["oracle_diagnostic"],
                "feature_names":FEATURES,"token_cost":TOKEN_COST,"utility_per_token":TOKEN_UTILITY_COST,"risk_thresholds":cfg,
                "rng":"Python standard-library random.Random","class_priors":CLASS_P,"cheap_correctness_probabilities":CHEAP_P,"deep_correctness_probabilities":DEEP_P}
    config_path=out/"config.json"
    config_path.write_text(json.dumps(config_obj,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    config_digest=sha256_file(config_path)
    allrows=[]
    for c in cases:
        for p in POLICIES:
            allrows.append(run_policy(c,p,cfg,models))
    # Oracle is an evaluator-only upper bound, explicitly not an eligible policy.
    for c in cases:
        acts=c["outcomes"]
        best=max(acts,key=lambda a:acts[a]["utility"])
        o=acts[best]
        allrows.append({"protocol_id":PROTOCOL,"case_id":c["case_id"],"group_id":c["group_id"],"split":"test","regime":c["regime"],"policy":"oracle_diagnostic","action":best,
                        "task_success":int(o["success"]),"harmful":int(o["harm"]),"net_utility":round(o["utility"],6),"tokens":TOKEN_COST[best],"latent_class":c["latent_class"],"should_answer":int(c["should_answer"]),"x":c["x"],"deep_action":c["deep_action"] if best=="escalate" else None})
    percase=out/"per_case_policy_outcomes.jsonl"
    with percase.open("w",encoding="utf-8") as f:
        for row in allrows: f.write(canonical(row)+"\n")
    rows_by=defaultdict(list)
    for row in allrows: rows_by[(row["regime"],row["policy"])].append(row)
    summ={}
    for regime in REGIMES:
        summ[regime]={}
        for p in POLICIES+["oracle_diagnostic"]:
            summ[regime][p]=summarize(rows_by[(regime,p)])
        baseline=rows_by[(regime,"always_answer")]
        for p in POLICIES+["oracle_diagnostic"]:
            if p=="always_answer": continue
            delta,lo,hi=grouped_bootstrap_delta(rows_by[(regime,p)],baseline,"net_utility",reps=600,seed=args.seed+REGIMES.index(regime))
            summ[regime][p]["net_utility_delta_vs_always_answer"]={"mean":round(delta,6),"ci95":[round(lo,6),round(hi,6)]}
    primary_pairs=[("risk_control_50permille","evidence_threshold"),("learned_value","evidence_threshold"),("risk_control_75permille","risk_control_50permille")]
    primary_contrasts={}
    for regime in REGIMES:
        primary_contrasts[regime]={}
        for pa,pb in primary_pairs:
            primary_contrasts[regime][f"{pa}_minus_{pb}"]={}
            for metric in ("task_success","harmful","tokens","net_utility"):
                delta,lo,hi=grouped_bootstrap_delta(rows_by[(regime,pa)],rows_by[(regime,pb)],metric,reps=600,seed=args.seed+11+REGIMES.index(regime))
                primary_contrasts[regime][f"{pa}_minus_{pb}"][metric]={"mean":round(delta,6),"ci95":[round(lo,6),round(hi,6)]}
    runmeta={"protocol_id":PROTOCOL,"classification":"synthetic research-only; not Broca or external benchmark evidence","seed":args.seed,"dev_cases":args.dev_cases,"test_cases_per_regime":args.test_cases_per_regime,"regimes":REGIMES,"policies":POLICIES+["oracle_diagnostic"],"policy_config":cfg,
             "training_policy":"Development is split by complete groups into model-train and policy-calibration partitions; thresholds and risk cutoffs use calibration only; test truth is not passed to policy functions.","training_cases":len(train_dev),"calibration_cases":len(calibration),"calibrated_risk_thresholds_by_budget":cfg["calibrated_risk_thresholds_by_budget"],"cost_model":{"tokens":TOKEN_COST,"utility_per_token":TOKEN_UTILITY_COST},"artifact":percase.name,"artifact_sha256":sha256_file(percase),"artifact_rows":len(allrows),"config_file":config_path.name,"config_sha256":config_digest,"expected_rows":(len(cases)*(len(POLICIES)+1)),"model_coeff_count":model_counts,"summaries":summ,"primary_contrasts":primary_contrasts}
    summary_path=out/"summary.json"; summary_path.write_text(json.dumps(runmeta,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    report=out/"report.md"
    lines=["# Risk-Aware Memory Escalation Benchmark", "", "**Classification:** synthetic research-only. Not an official benchmark, not a real model run, and not evidence of SOTA.", "", f"- Seed: `{args.seed}`; development cases: {args.dev_cases}; test cases/regime: {args.test_cases_per_regime}.", f"- Policies × regimes: {len(runmeta['policies'])} × {len(REGIMES)}; artifact rows: {len(allrows):,}.", f"- Outcome artifact SHA-256: `{runmeta['artifact_sha256']}`.", "- Similarity/evidence thresholds and sequential thresholds are tuned on development data only; ridge models also use development outcomes only.", "", "## Results by regime", "", "All percentages are synthetic. Utility includes declared token cost. Harm means an incorrect or invalid answer was emitted. Oracle is evaluator-only and ineligible as a deployable policy.", "", "| Regime | Policy | Task success | Harmful rate | Mean tokens | Net utility | Escalate | Abstain | Δ utility vs always-answer (95% grouped bootstrap CI) |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for reg in REGIMES:
        for p in runmeta["policies"]:
            s=summ[reg][p]; d=s.get("net_utility_delta_vs_always_answer")
            dtext="—" if not d else f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}]"
            lines.append(f"| {reg} | {p} | {s['task_success']*100:.2f}% | {s['harmful_rate']*100:.2f}% | {s['mean_tokens']:.1f} | {s['net_utility']:.4f} | {s['escalation_rate']*100:.1f}% | {s['abstention_rate']*100:.1f}% | {dtext} |")
    lines += ["", "## Calibrated risk-budget frontier", "", "Each `risk_control_*permille` policy uses a separate calibration threshold for an intended harmful-answer budget. Thresholds are selected on the held-out calibration split by requiring the 95% Wilson upper endpoint on observed calibration risk to stay at or below the budget. This is an empirical calibration constraint, **not** a finite-sample distribution-free guarantee and not a guarantee under shift.", "", "| Regime | Policy | Intended risk budget | Observed harmful rate | Task success | Mean tokens | Net utility |", "|---|---|---:|---:|---:|---:|---:|"] + [f"| {reg} | {p} | {next((b*100 for b,n in RISK_POLICY_NAMES.items() if n==p),0):.1f}% | {summ[reg][p]['harmful_rate']*100:.2f}% | {summ[reg][p]['task_success']*100:.2f}% | {summ[reg][p]['mean_tokens']:.1f} | {summ[reg][p]['net_utility']:.4f} |" for reg in REGIMES for p in RISK_POLICY_NAMES.values()]
    lines += ["", "## Paired primary contrasts", "", "Deltas use matched cases and 600 paired episode-group bootstrap resamples. Task-success and harmful deltas are absolute rates (multiply by 100 for percentage points); token delta is tokens/query; utility is the declared synthetic score.", "", "| Regime | Contrast (left − right) | Metric | Mean delta | 95% grouped-bootstrap CI |", "|---|---|---|---:|---|"]
    for reg in REGIMES:
        for pair,vals in primary_contrasts[reg].items():
            for metric,pretty in (("task_success","task success"),("harmful","harmful rate"),("tokens","tokens"),("net_utility","net utility")):
                v=vals[metric]; lines.append(f"| {reg} | {pair} | {pretty} | {v['mean']:+.4f} | [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] |")
    lines += ["", "## Policy operating points", "", "Model action rates are policy outputs, while latent classes and outcomes appear only in the evaluator artifact. The artifact intentionally retains labels for auditing; the policy function accepts only observable feature dictionaries.", "", "## Limitations and next gates", "", "1. Every outcome and feature comes from a hand-specified synthetic generator; no generalization to real agents follows.", "2. Risk predictions can be miscalibrated under the declared regime shifts; the report must be read by regime, not just pooled.", "3. Token costs are assumed units, not measured vendor charges or wall-clock latency.", "4. Next gate: implement the policy in the actual Rust-backed Broca adapter and execute matched official LongMemEval-V2 arms; add Mem2ActBench for action grounding and TimlMemBench/FAMA for governance and invalidation.", "5. The oracle is a diagnostic upper bound only. It sees evaluator-only outcomes and cannot be deployed.", ""]
    report.write_text("\n".join(lines),encoding="utf-8")
    receipt={"protocol_id":PROTOCOL,"source_file":"benchmark.py","source_sha256":sha256_file(Path(__file__)),"config_file":config_path.name,"config_sha256":config_digest,"summary_sha256":sha256_file(summary_path),"outcome_sha256":sha256_file(percase),"outcome_rows":len(allrows),"expected_rows":len(allrows),"test_case_count":len(cases),"regimes":REGIMES,"policy_config":cfg,"evidence_boundary":"Synthetic only; receipt validates byte integrity, not scientific correctness or external performance."}
    (out/"receipt.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(f"PASS: wrote {len(allrows):,} policy/case rows across {len(REGIMES)} held-out regimes")
    print(f"PASS: outcome_sha256={runmeta['artifact_sha256']}")
    print(f"PASS: summary={summary_path}")
    return 0

if __name__=="__main__": raise SystemExit(main())
