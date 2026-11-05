#!/usr/bin/env python3
"""Validate CEB human-escalation pilot preregistration offline."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {path}: {error}") from error


def semantic_errors(payload: dict) -> list[str]:
    errors=[]
    families=payload["population"]["task_families"]
    ids=[row["family_id"] for row in families]
    if len(ids)!=len(set(ids)):
        errors.append("duplicate blocker-family quota")
    if sum(row["planned_pairs"] for row in families)!=payload["population"]["planned_pair_count"]:
        errors.append("blocker-family quotas do not sum to planned pair count")
    if payload["population"]["total_planned_variants"] != payload["population"]["planned_pair_count"]*2:
        errors.append("planned variant count must equal twice the pair count")
    custody=payload["task_custody"]
    if custody["development_pairs"]+custody["heldout_pairs"] != payload["population"]["planned_pair_count"]:
        errors.append("development and holdout quotas do not sum to pair count")
    if payload["population"]["actual_pair_count"] != 0:
        errors.append("V0 design-only preregistration must contain zero admitted task pairs")
    if payload["status"] != "design_only_not_executed":
        errors.append("V0 must remain design_only_not_executed")
    if payload["execution_gates"]["candidate_controlled_execution"] != "blocked_until_issue_10064_qualified":
        errors.append("candidate execution gate was weakened")
    if payload["execution_gates"]["independent_regrade_claim"] != "blocked_until_issue_10369_qualified":
        errors.append("independent regrade trust gate was weakened")
    if payload["help_protocol"]["no_help_control"]["status"] != "not_run_blocked":
        errors.append("no-help necessity control cannot be reported as run")
    if payload["help_protocol"]["full_information_control"]["status"] != "not_run_blocked":
        errors.append("full-information sufficiency control cannot be reported as run")
    if payload["help_protocol"]["answer_authority"] == "llm_only_prohibited":
        errors.append("LLM-only answer authority is prohibited")
    if payload["statistics"]["independent_runs_for_pass_at_3"] != 3:
        errors.append("Pass@3 must use three independent runs")
    if payload["counterfactual_pair_contract"]["one_changed_fact_only"] is not True:
        errors.append("counterfactual pair must differ by exactly one fact")
    if payload["task_qualification"]["llm_only_oracle"] is not False:
        errors.append("LLM-only oracle is prohibited")
    metric_ids=[m["metric_id"] for m in payload["metrics"]]
    required={"question_precision","blocker_recall","selective_escalation_f1","help_enabled_pass_at_3","paired_act_ask_accuracy","premature_action_rate","duplicate_question_rate","answer_integration_success","no_help_pass_rate","full_information_pass_rate"}
    if not required.issubset(metric_ids):
        errors.append("human escalation pilot is missing one or more required metrics")
    return errors


def validate(payload:dict,schema:dict) -> list[str]:
    validator=Draft202012Validator(schema,format_checker=FormatChecker())
    schema_errors=sorted(validator.iter_errors(payload),key=lambda e:list(e.absolute_path))
    if schema_errors:
        return [f"schema {'/'.join(map(str,e.absolute_path))}: {e.message}" for e in schema_errors]
    return semantic_errors(payload)


def self_test(payload:dict,schema:dict):
    if validate(payload,schema):
        raise SystemExit("valid human-escalation pilot design unexpectedly failed")
    def must_reject(mutant:dict,label:str):
        if not validate(mutant,schema):
            raise SystemExit(f"human-escalation negative control unexpectedly passed: {label}")
    mutant=copy.deepcopy(payload)
    mutant["population"]["task_families"][0]["planned_pairs"]=7
    must_reject(mutant,"family quota drift")
    mutant=copy.deepcopy(payload)
    mutant["task_custody"]["heldout_pairs"]=7
    must_reject(mutant,"holdout partition shrink")
    mutant=copy.deepcopy(payload)
    mutant["execution_gates"]["candidate_controlled_execution"]="allowed"
    must_reject(mutant,"runner isolation gate weakened")
    mutant=copy.deepcopy(payload)
    mutant["help_protocol"]["answer_authority"]="llm_only_prohibited"
    must_reject(mutant,"LLM-only help authority")
    mutant=copy.deepcopy(payload)
    mutant["help_protocol"]["no_help_control"]["status"]="passed"
    must_reject(mutant,"unrun no-help control reported as passed")
    mutant=copy.deepcopy(payload)
    mutant["statistics"]["independent_runs_for_pass_at_3"]=1
    must_reject(mutant,"Pass@3 downgraded to one run")
    mutant=copy.deepcopy(payload)
    mutant["counterfactual_pair_contract"]["one_changed_fact_only"]=False
    must_reject(mutant,"counterfactual pair changes multiple facts")
    mutant=copy.deepcopy(payload)
    mutant["population"]["actual_pair_count"]=1
    must_reject(mutant,"design-only task pairs falsely admitted")
    print("CEB_HUMAN_ESCALATION_SELF_TEST_PASS")


def main()->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root",type=Path,default=Path.cwd())
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    base=args.repo_root.resolve()/"benchmarks"/"ceb"
    try:
        payload=load(base/"human-escalation-pilot.v0.json")
        schema=load(base/"human-escalation-pilot.schema.json")
        Draft202012Validator.check_schema(schema)
    except Exception as error:
        print(f"CEB_HUMAN_ESCALATION_ERROR: {error}",file=sys.stderr)
        return 2
    errors=validate(payload,schema)
    if errors:
        for error in errors:
            print(f"CEB_HUMAN_ESCALATION_FAIL: {error}",file=sys.stderr)
        return 1
    if args.self_test:
        self_test(payload,schema)
    print(
        "CEB_HUMAN_ESCALATION_PASS: "
        f"planned_pairs={payload['population']['planned_pair_count']} "
        f"actual_pairs={payload['population']['actual_pair_count']} "
        f"families={len(payload['population']['task_families'])} "
        f"development={payload['task_custody']['development_pairs']} "
        f"holdout={payload['task_custody']['heldout_pairs']}"
    )
    return 0


if __name__=="__main__":
    raise SystemExit(main())
