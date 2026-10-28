#!/usr/bin/env python3
"""Independent streaming verifier for a risk-aware escalation run."""
from __future__ import annotations
import argparse, hashlib, json, math, sys
from collections import defaultdict
from pathlib import Path

CORE_METRICS = ("task_success","harmful_rate","mean_tokens","net_utility","escalation_rate","answer_rate","abstention_rate")
ACTIONS={"answer","escalate","abstain"}

def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()

def close(a,b,tol=2e-5):
    return isinstance(a,(int,float)) and isinstance(b,(int,float)) and abs(a-b)<=tol

def verify(run_dir: Path) -> tuple[bool,list[str]]:
    errors=[]
    try:
        receipt=json.loads((run_dir/"receipt.json").read_text(encoding="utf-8"))
        summary=json.loads((run_dir/"summary.json").read_text(encoding="utf-8"))
    except Exception as e:
        return False,[f"cannot read receipt/summary: {e}"]
    if receipt.get("protocol_id") != "BROCA-RISK-ESCALATION-V1": errors.append("unknown protocol")
    if summary.get("protocol_id") != receipt.get("protocol_id"): errors.append("receipt-summary protocol mismatch")
    if receipt.get("summary_sha256") != sha(run_dir/"summary.json"): errors.append("summary SHA-256 mismatch")
    config_path=run_dir/receipt.get("config_file", "config.json")
    if not config_path.is_file(): errors.append("config artifact missing")
    elif receipt.get("config_sha256") != sha(config_path): errors.append("config SHA-256 mismatch")
    source_path=Path(__file__).with_name(receipt.get("source_file", "benchmark.py"))
    if not source_path.is_file(): errors.append("source file for recorded source digest missing")
    elif receipt.get("source_sha256") != sha(source_path): errors.append("benchmark source SHA-256 mismatch")
    artifact=run_dir/receipt.get("artifact", "per_case_policy_outcomes.jsonl")
    if not artifact.is_file(): errors.append("outcome artifact missing")
    else:
        actual=sha(artifact)
        if receipt.get("outcome_sha256") != actual: errors.append("outcome SHA-256 mismatch")
        if summary.get("artifact_sha256") != actual: errors.append("summary-outcome digest mismatch")
    expected=receipt.get("expected_rows")
    if not isinstance(expected,int) or expected<=0: errors.append("invalid expected row count")
    policies=summary.get("policies",[])
    regimes=summary.get("regimes",[])
    if not policies or not regimes: errors.append("empty policy/regime matrix")
    if len(set(policies))!=len(policies): errors.append("duplicate policy names")
    if len(set(regimes))!=len(regimes): errors.append("duplicate regimes")
    action_tokens=summary.get("cost_model",{}).get("tokens",{})
    cases_by_regime=defaultdict(set); arms_by_case=defaultdict(set); accum=defaultdict(lambda:{"n":0,"success":0,"harm":0,"tokens":0,"utility":0.0,"answer":0,"escalate":0,"abstain":0})
    rows=0
    if artifact.is_file():
      try:
        with artifact.open(encoding="utf-8") as f:
          for line_no,line in enumerate(f,1):
            if not line.strip(): errors.append(f"blank line at {line_no}"); continue
            try: r=json.loads(line)
            except Exception as e: errors.append(f"invalid JSON at line {line_no}: {e}"); continue
            rows+=1
            reg=r.get("regime"); case=r.get("case_id"); pol=r.get("policy"); act=r.get("action")
            if r.get("protocol_id")!=receipt.get("protocol_id"): errors.append(f"protocol mismatch at line {line_no}")
            if r.get("split")!="test": errors.append(f"non-test row at line {line_no}")
            if reg not in regimes: errors.append(f"unknown regime at line {line_no}"); continue
            if pol not in policies: errors.append(f"unknown policy at line {line_no}")
            if act not in ACTIONS: errors.append(f"unknown action at line {line_no}"); continue
            if not isinstance(case,str) or not case: errors.append(f"missing case_id at line {line_no}"); continue
            key=(reg,case)
            if pol in arms_by_case[key]: errors.append(f"duplicate case/policy pair at line {line_no}")
            arms_by_case[key].add(pol); cases_by_regime[reg].add(case)
            if r.get("tokens") != action_tokens.get(act): errors.append(f"token cost does not match action at line {line_no}")
            if r.get("task_success") not in (0,1) or r.get("harmful") not in (0,1): errors.append(f"non-binary outcome at line {line_no}")
            if not isinstance(r.get("net_utility"),(int,float)) or not math.isfinite(r["net_utility"]): errors.append(f"invalid utility at line {line_no}")
            if r.get("latent_class") not in ("fresh","stale","conflict","absent","scope_shift","distractor","revoked"): errors.append(f"invalid latent class at line {line_no}")
            x=r.get("x")
            if not isinstance(x,dict) or any(not isinstance(v,(int,float)) or v<0 or v>1 for v in x.values()): errors.append(f"invalid observable features at line {line_no}")
            if r.get("should_answer") not in (0,1): errors.append(f"invalid evaluator label at line {line_no}")
            a=accum[(reg,pol)]; a["n"]+=1; a["success"]+=r.get("task_success",0); a["harm"]+=r.get("harmful",0); a["tokens"]+=r.get("tokens",0); a["utility"]+=r.get("net_utility",0.0); a[act]+=1
      except Exception as e: errors.append(f"could not stream outcomes: {e}")
    if rows!=expected or rows!=receipt.get("outcome_rows"): errors.append(f"row count mismatch: {rows}, expected {expected}")
    expected_cases=summary.get("test_cases_per_regime",0)
    for reg in regimes:
        if len(cases_by_regime[reg])!=expected_cases: errors.append(f"case count mismatch for regime {reg}: {len(cases_by_regime[reg])}")
        for case in cases_by_regime[reg]:
            if arms_by_case[(reg,case)] != set(policies): errors.append(f"policy matrix incomplete for {reg}/{case}"); break
        for pol in policies:
            a=accum[(reg,pol)]; n=a["n"]
            if n!=expected_cases: errors.append(f"row count mismatch for {reg}/{pol}: {n}"); continue
            rec=summary.get("summaries",{}).get(reg,{}).get(pol)
            if not rec: errors.append(f"missing summary for {reg}/{pol}"); continue
            values={"n":n,"task_success":a["success"]/n,"harmful_rate":a["harm"]/n,"mean_tokens":a["tokens"]/n,"net_utility":a["utility"]/n,
                    "escalation_rate":a["escalate"]/n,"answer_rate":a["answer"]/n,"abstention_rate":a["abstain"]/n}
            for m in CORE_METRICS:
                if not close(rec.get(m),round(values[m],6)): errors.append(f"summary mismatch {reg}/{pol}/{m}")
    if errors: return False,errors
    return True,[f"verified {rows:,} rows",f"verified {len(regimes)} regimes × {len(policies)} policies",f"outcome SHA-256 {sha(artifact)}","summary aggregates independently recomputed"]

def main():
    p=argparse.ArgumentParser(); p.add_argument("run_dir"); args=p.parse_args()
    ok,lines=verify(Path(args.run_dir))
    for line in lines: print(("PASS: " if ok else "FAIL: ")+line)
    return 0 if ok else 1
if __name__=="__main__": raise SystemExit(main())
