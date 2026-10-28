#!/usr/bin/env python3
"""Export compact policy/regime and paired-contrast CSVs from a verified summary."""
import csv, json, sys
from pathlib import Path

if len(sys.argv) != 3:
    raise SystemExit("Usage: export_frontier.py SUMMARY.json OUTPUT_DIR")
src=Path(sys.argv[1]); out=Path(sys.argv[2]); out.mkdir(parents=True,exist_ok=True)
data=json.loads(src.read_text(encoding="utf-8"))
with (out/"policy_frontier.csv").open("w",newline="",encoding="utf-8") as f:
    fields=["regime","policy","n","task_success","harmful_rate","mean_tokens","net_utility","answer_rate","escalation_rate","abstention_rate","delta_utility_vs_always_answer","delta_utility_ci95_low","delta_utility_ci95_high"]
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for regime,policies in data["summaries"].items():
        for policy,metrics in policies.items():
            delta=metrics.get("net_utility_delta_vs_always_answer",{})
            w.writerow({"regime":regime,"policy":policy,"n":metrics["n"],"task_success":metrics["task_success"],"harmful_rate":metrics["harmful_rate"],"mean_tokens":metrics["mean_tokens"],"net_utility":metrics["net_utility"],"answer_rate":metrics["answer_rate"],"escalation_rate":metrics["escalation_rate"],"abstention_rate":metrics["abstention_rate"],"delta_utility_vs_always_answer":delta.get("mean"),"delta_utility_ci95_low":(delta.get("ci95") or [None,None])[0],"delta_utility_ci95_high":(delta.get("ci95") or [None,None])[1]})
with (out/"primary_paired_contrasts.csv").open("w",newline="",encoding="utf-8") as f:
    fields=["regime","contrast","metric","mean_delta","ci95_low","ci95_high"]
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for regime,pairs in data.get("primary_contrasts",{}).items():
        for pair,metrics in pairs.items():
            for metric,rec in metrics.items():
                w.writerow({"regime":regime,"contrast":pair,"metric":metric,"mean_delta":rec["mean"],"ci95_low":rec["ci95"][0],"ci95_high":rec["ci95"][1]})
print(f"PASS: exported {out/'policy_frontier.csv'} and {out/'primary_paired_contrasts.csv'}")
