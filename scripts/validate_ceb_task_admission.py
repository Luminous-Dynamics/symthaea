#!/usr/bin/env python3
"""Validate CEB task-admission registry and individual task manifests offline."""
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


def validate_task(task: dict, schema: dict, format_checker: FormatChecker) -> list[str]:
    validator = Draft202012Validator(schema, format_checker=format_checker)
    errors = sorted(validator.iter_errors(task), key=lambda error: list(error.absolute_path))
    if errors:
        return [
            f"{'/'.join(map(str, error.absolute_path)) or '<task>'}: {error.message}"
            for error in errors
        ]
    semantic: list[str] = []
    reviewers = task["oracle"]["reviewers"]
    reviewer_ids = [row["reviewer_id"] for row in reviewers]
    if len(set(reviewer_ids)) != len(reviewer_ids):
        semantic.append("oracle reviewer IDs must be distinct")
    if len([row for row in reviewers if row["role"] == "gold_reviewer"]) < 2:
        semantic.append("oracle requires two distinct gold reviewers")
    if task["oracle"]["disagreement_status"] == "unresolved" and task["task_status"] != "proposed":
        semantic.append("unresolved oracle disagreement cannot be statically reviewed or admitted")
    if task["oracle"]["disagreement_status"] == "resolved_by_adjudicator":
        adjudicator = task["oracle"]["adjudicator_id"]
        if not adjudicator or adjudicator in reviewer_ids:
            semantic.append("third adjudicator must be identified separately from the two gold reviewers")
        if not task["oracle"]["adjudication_record_sha256"]:
            semantic.append("adjudicator decision must have a record digest")
        if task["oracle"]["adjudication_decision"] not in ("accept", "reject"):
            semantic.append("resolved disagreement requires an explicit accept/reject adjudicator decision")
        if task["task_status"] in ("static_reviewed", "execution_qualified", "admitted") and task["oracle"]["adjudication_decision"] != "accept":
            semantic.append("only an accepting adjudication can qualify a task")
    if task["oracle"]["disagreement_status"] == "none":
        if task["oracle"]["adjudicator_id"] is not None or task["oracle"]["adjudication_record_sha256"] is not None or task["oracle"]["adjudication_decision"] != "not_needed":
            semantic.append("adjudicator fields must be empty when gold reviewers agree")
    if task["oracle"]["disagreement_status"] == "unresolved":
        if task["task_status"] != "proposed" or task["oracle"]["adjudication_decision"] != "unresolved":
            semantic.append("unresolved disagreement remains proposed and cannot be qualified")
    if task["oracle"]["disagreement_status"] == "none" and task["task_status"] in ("static_reviewed", "execution_qualified", "admitted"):
        if any(row["decision"] != "accept" for row in reviewers):
            semantic.append("without adjudication, both gold reviewers must accept")
    if task["oracle"]["disagreement_status"] == "resolved_by_adjudicator":
        if task["task_status"] in ("static_reviewed", "execution_qualified", "admitted") and task["oracle"]["adjudication_decision"] != "accept":
            semantic.append("adjudicator must accept a task before it can qualify")
        if task["oracle"]["adjudication_decision"] == "reject" and task["task_status"] != "rejected":
            semantic.append("rejected adjudication cannot be advanced to a qualifying task state")
        if all(row["decision"] == "accept" for row in reviewers):
            semantic.append("adjudication record is unexpected when both gold reviewers already agree")
    if not task["oracle"]["blinded_to_candidate_output"]:
        semantic.append("gold review must be blinded to candidate output")
    if task["repository"]["license"]["review_status"] != "compatible" and task["task_status"] != "proposed":
        semantic.append("static review/admission requires compatible license review")
    if task["task_status"] in ("static_reviewed", "execution_qualified", "admitted"):
        if not task["repository"]["license"]["license_review_record_sha256"] or not task["repository"]["license"]["license_reviewer_id"]:
            semantic.append("compatible license decision requires a named reviewer and review-record digest")
    if task["source"]["solution_leakage_review"]["status"] != "complete" and task["task_status"] != "proposed":
        semantic.append("static review/admission requires a complete solution-leakage review")
    if task["source"]["solution_leakage_review"]["status"] in ("suspected_leakage", "confirmed_leakage"):
        semantic.append("suspected or confirmed solution leakage blocks admission")
    if task["task_spec"]["blind_solve_review"]["status"] != "passed" and task["task_status"] != "proposed":
        semantic.append("static review/admission requires a passed blind-solve instruction review")
    if task["task_status"] in ("static_reviewed", "execution_qualified", "admitted"):
        if task["task_spec"]["blind_solve_review"]["reviewer_id"] in reviewer_ids:
            semantic.append("blind-solve reviewer must be distinct from gold reviewers")
        if task["source"]["solution_leakage_review"]["reviewer_id"] in reviewer_ids:
            semantic.append("solution-leakage reviewer must be distinct from gold reviewers")
        if task["source"]["fix_visibility"] == "unknown":
            semantic.append("unknown fix visibility blocks static task qualification")
    if not task["task_spec"]["blind_solve_review"]["instruction_only_sufficient"] and task["task_status"] != "proposed":
        semantic.append("task instructions must be sufficient without access to the fix diff")
    if task["custody"]["agent_access_before_freeze"]:
        semantic.append("task material cannot be exposed to the agent before the task freeze")
    if task["custody"]["partition"] == "holdout":
        if task["source"]["fix_visibility"] not in ("private", "not_published"):
            semantic.append("holdout requires private or not-published fix visibility; unknown is not sealed")
        if task["source"]["fix_visibility"] == "public":
            semantic.append("publicly visible fix cannot be admitted as sealed holdout")
        if not task["custody"]["holdout_material_outside_agent_workspace"]:
            semantic.append("holdout material must remain outside candidate workspace")
        if task["custody"]["public_fix_exposure_accepted"]:
            semantic.append("public fix exposure cannot be accepted for sealed holdout")
    if task["custody"]["partition"] == "development" and task["custody"]["public_fix_exposure_accepted"] is False and task["source"]["fix_visibility"] == "public":
        semantic.append("development task must explicitly disclose accepted public fix exposure")
    if task["family_id"] == "performance-correctness" and task["environment"]["workload_sha256"] is None:
        semantic.append("performance task requires a pinned workload digest")
    if task["acceptance_contract"]["security_review_status"] == "failed":
        semantic.append("failed security review blocks task admission")
    if task["acceptance_contract"]["security_review_status"] == "passed" and task["task_status"] in ("static_reviewed", "execution_qualified", "admitted"):
        if not task["acceptance_contract"]["security_review_record_sha256"] or not task["acceptance_contract"]["security_reviewer_id"]:
            semantic.append("passed security review requires reviewer identity and record digest")
    if task["acceptance_contract"]["destructive_side_effects"]:
        semantic.append("tasks with destructive external side effects are excluded from the pilot")
    if task["execution_qualification"]["candidate_code_executed"]:
        if task["execution_qualification"]["isolation_status"] != "qualified":
            semantic.append("candidate code cannot execute before isolation is qualified under #10064")
        if not task["execution_qualification"]["isolation_profile_sha256"]:
            semantic.append("executed candidate must bind the qualified isolation profile digest")
    if task["execution_qualification"]["isolation_status"] == "qualified":
        if not task["execution_qualification"]["isolation_profile_sha256"]:
            semantic.append("qualified isolation requires a policy/profile digest")
    if task["execution_qualification"]["isolation_status"] != "qualified" and not task["execution_qualification"]["candidate_code_executed"]:
        if task["oracle"]["reference_patch_control"] != "not_run_blocked" or task["oracle"]["empty_patch_control"] != "not_run_blocked":
            semantic.append("reference/empty-patch controls must remain not_run_blocked until isolation is qualified")
    if task["task_status"] in ("static_reviewed", "execution_qualified", "admitted"):
        if any(row["decision"] != "accept" for row in reviewers if row["role"] == "gold_reviewer"):
            semantic.append("static review requires two accepting gold-reviewer decisions")
    if task["execution_qualification"]["trust_status"] == "qualified" and not task["execution_qualification"]["trust_policy_sha256"]:
        semantic.append("qualified trust status requires a trust-policy digest")
    if task["execution_qualification"]["independent_regrade_claimed"]:
        if task["execution_qualification"]["trust_status"] != "qualified":
            semantic.append("independent regrade claim requires trust qualification under #10369")
    controls = task["oracle"]
    if task["task_status"] in ("execution_qualified", "admitted"):
        if controls["reference_patch_control"] != "pass":
            semantic.append("two-sided oracle gate requires reference-patch control pass")
        if controls["empty_patch_control"] != "fail":
            semantic.append("two-sided oracle gate requires empty-patch control fail")
        if task["execution_qualification"]["isolation_status"] != "qualified":
            semantic.append("execution-qualified task requires qualified isolation")
    if task["task_status"] == "admitted":
        if task["execution_qualification"]["trust_status"] != "qualified":
            semantic.append("admitted task requires qualified trust and regrade custody")
        if task["execution_qualification"]["independent_regrade_claimed"] is not False:
            semantic.append("V0 does not permit an independent regrade claim")
    if task["task_status"] == "rejected" and not task["claim_boundary"].strip():
        semantic.append("rejected task requires a reason/claim boundary")
    return semantic


def validate_registry(registry: dict, registry_schema: dict, task_schema: dict, format_checker: FormatChecker) -> list[str]:
    errors = []
    validator = Draft202012Validator(registry_schema, format_checker=format_checker)
    schema_errors = sorted(validator.iter_errors(registry), key=lambda error: list(error.absolute_path))
    if schema_errors:
        return [f"registry {'/'.join(map(str, error.absolute_path))}: {error.message}" for error in schema_errors]
    family_ids = [row["family_id"] for row in registry["family_targets"]]
    if len(family_ids) != len(set(family_ids)):
        errors.append("duplicate pilot family quota")
    if sum(row["target_tasks"] for row in registry["family_targets"]) != registry["target_task_count"]:
        errors.append("family quotas do not sum to 48")
    if registry["actual_task_count"] != len(registry["tasks"]):
        errors.append("actual_task_count does not equal task record count")
    task_ids = [task.get("task_id") for task in registry["tasks"]]
    if len(task_ids) != len(set(task_ids)):
        errors.append("duplicate task_id")
    quota_by_family = {row["family_id"]: row["target_tasks"] for row in registry["family_targets"]}
    counts_by_family: dict[str, int] = {}
    for task in registry["tasks"]:
        family = task.get("family_id")
        counts_by_family[family] = counts_by_family.get(family, 0) + 1
    for family, count in counts_by_family.items():
        if family not in quota_by_family or count > quota_by_family[family]:
            errors.append(f"task family {family} exceeds or lacks a declared quota")
    if registry["status"] == "intake_contract_only_no_tasks_admitted" and registry["actual_task_count"] != 0:
        errors.append("intake-only V0 cannot claim actual task records")
    if registry["actual_task_count"] == 0 and registry["status"] != "intake_contract_only_no_tasks_admitted":
        errors.append("zero task records must remain explicitly intake-only")
    for task in registry["tasks"]:
        for error in validate_task(task, task_schema, format_checker):
            errors.append(f"{task.get('task_id','<unknown>')}: {error}")
    return errors


def valid_fixture() -> dict:
    digest = "a" * 64
    return {
      "schema_version":0,"task_id":"ceb-rust-fixture-001","task_status":"proposed","family_id":"rust-types-traits",
      "repository":{"repo_full_name":"example/project","url":"https://github.com/example/project","source_revision":"b"*40,
       "license":{"spdx_expression":"Apache-2.0","license_file_path":"LICENSE","license_file_sha256":digest,"review_status":"compatible","license_review_record_sha256":digest,"license_reviewer_id":"license-reviewer-01"}},
      "source":{"source_kind":"issue","source_url":"https://github.com/example/project/issues/1","source_id":"1","opened_at":"2026-01-01T00:00:00Z","fix_commit_sha":"c"*40,"fix_visibility":"public",
       "solution_leakage_review":{"status":"complete","reviewed_on":"2026-11-05","search_protocol_id":"CEB-LEAKAGE-SEARCH-001","search_record_sha256":digest,"searched_surfaces":["source-host","commit-history","benchmark-datasets"],"reviewer_id":"leakage-reviewer-01"}},
      "task_spec":{"prompt_sha256":digest,"acceptance_spec_sha256":"d"*64,"authoring_protocol_version":"0.1.0",
       "blind_solve_review":{"status":"passed","reviewer_id":"blind-reviewer-01","review_record_sha256":digest,"instruction_only_sufficient":True}},
      "environment":{"rust_toolchain":"1.96.0","cargo_lock_sha256":digest,"target_triple":"x86_64-unknown-linux-gnu","feature_set":["default"],"environment_image_digest":"sha256:"+"e"*64,"resource_policy_sha256":digest,"network_policy":"disabled","replay_manifest_sha256":digest,"workload_sha256":None},
      "acceptance_contract":{"contract_sha256":digest,"fail_to_pass_check_ids":["issue-acceptance-001"],"pass_to_pass_check_ids":["existing-regression-001"],"critical_invariants":["trait contract remains sound"],"security_review_status":"passed","destructive_side_effects":False,"security_review_record_sha256":digest,"security_reviewer_id":"security-reviewer-01"},
      "oracle":{"reviewers":[{"reviewer_id":"gold-reviewer-01","role":"gold_reviewer","review_record_sha256":digest,"decision":"accept"},{"reviewer_id":"gold-reviewer-02","role":"gold_reviewer","review_record_sha256":digest,"decision":"accept"}],"blinded_to_candidate_output":True,"llm_only_oracle":False,"disagreement_status":"none","adjudicator_id":None,"adjudication_record_sha256":None,"adjudication_decision":"not_needed","gold_patch_sha256":digest,"semantically_equivalent_repairs_allowed":True,"reference_patch_control":"not_run_blocked","empty_patch_control":"not_run_blocked","oracle_record_sha256":digest},
      "custody":{"partition":"development","custody_domain_id":"dev-custody-01","custodian_id":"custodian-01","custody_manifest_sha256":digest,"holdout_material_outside_agent_workspace":False,"agent_access_before_freeze":False,"public_fix_exposure_accepted":True},
      "execution_qualification":{"candidate_code_executed":False,"isolation_status":"blocked_issue_10064","isolation_profile_sha256":None,"trust_status":"blocked_issue_10369","trust_policy_sha256":None,"independent_regrade_claimed":False},
      "dispositions":{"timeouts_counted":True,"abstentions_counted":True,"escalations_counted":True,"ungraded_counted":True,"denominator_policy_sha256":digest},
      "claim_boundary":"Synthetic validator fixture only; not a real benchmark task and not admitted."
    }


def self_test(registry: dict, registry_schema: dict, task_schema: dict, format_checker: FormatChecker) -> None:
    fixture = valid_fixture()
    if validate_task(fixture, task_schema, format_checker):
        raise SystemExit("valid synthetic task-manifest fixture unexpectedly failed")

    def must_reject(mutant: dict, label: str) -> None:
        errors = validate_task(mutant, task_schema, format_checker)
        if not errors:
            raise SystemExit(f"task-admission negative control unexpectedly passed: {label}")

    mutant = copy.deepcopy(fixture)
    mutant["repository"]["source_revision"] = "main"
    must_reject(mutant, "mutable source ref")
    mutant = copy.deepcopy(fixture)
    mutant["repository"]["license"]["license_file_sha256"] = "missing"
    must_reject(mutant, "missing license digest")
    mutant = copy.deepcopy(fixture)
    mutant["oracle"]["reviewers"] = mutant["oracle"]["reviewers"][:1]
    must_reject(mutant, "single gold reviewer")
    mutant = copy.deepcopy(fixture)
    mutant["oracle"]["reviewers"][1]["reviewer_id"] = mutant["oracle"]["reviewers"][0]["reviewer_id"]
    must_reject(mutant, "duplicate gold reviewer identity")
    mutant = copy.deepcopy(fixture)
    mutant["oracle"]["llm_only_oracle"] = True
    must_reject(mutant, "LLM-only oracle")
    mutant = copy.deepcopy(fixture)
    mutant["oracle"]["blinded_to_candidate_output"] = False
    must_reject(mutant, "unblinded gold review")
    mutant = copy.deepcopy(fixture)
    mutant["custody"]["partition"] = "holdout"
    mutant["custody"]["holdout_material_outside_agent_workspace"] = True
    mutant["source"]["fix_visibility"] = "public"
    must_reject(mutant, "public fix mislabeled as sealed holdout")
    mutant = copy.deepcopy(fixture)
    mutant["task_status"] = "execution_qualified"
    mutant["oracle"]["reference_patch_control"] = "pass"
    mutant["oracle"]["empty_patch_control"] = "pass"
    mutant["execution_qualification"]["isolation_status"] = "blocked_issue_10064"
    must_reject(mutant, "execution before OS isolation qualification")

    mutant = copy.deepcopy(fixture)
    mutant["task_status"] = "execution_qualified"
    mutant["oracle"]["reference_patch_control"] = "pass"
    mutant["oracle"]["empty_patch_control"] = "pass"
    mutant["execution_qualification"]["isolation_status"] = "qualified"
    mutant["execution_qualification"]["isolation_profile_sha256"] = "f" * 64
    must_reject(mutant, "empty patch incorrectly passes the two-sided oracle gate")

    mutant = copy.deepcopy(fixture)
    mutant["custody"]["partition"] = "holdout"
    mutant["custody"]["holdout_material_outside_agent_workspace"] = True
    mutant["source"]["fix_visibility"] = "unknown"
    must_reject(mutant, "unknown fix visibility treated as sealed holdout")
    mutant = copy.deepcopy(fixture)
    mutant["execution_qualification"]["candidate_code_executed"] = True
    must_reject(mutant, "candidate code executed without isolation")
    mutant = copy.deepcopy(fixture)
    mutant["execution_qualification"]["independent_regrade_claimed"] = True
    must_reject(mutant, "independent regrade claimed before trust qualification")
    mutant = copy.deepcopy(fixture)
    mutant["dispositions"]["timeouts_counted"] = False
    must_reject(mutant, "timeout silently dropped from denominator")
    mutant = copy.deepcopy(fixture)
    mutant["family_id"] = "performance-correctness"
    must_reject(mutant, "performance task with no workload digest")
    print("CEB_TASK_MANIFEST_SELF_TEST_PASS")

    registry_mutant = copy.deepcopy(registry)
    registry_mutant["family_targets"][0]["target_tasks"] = 9
    errors = validate_registry(registry_mutant, registry_schema, task_schema, format_checker)
    if not errors:
        raise SystemExit("task admission quota negative control unexpectedly passed")
    registry_mutant = copy.deepcopy(registry)
    registry_mutant["actual_task_count"] = 1
    errors = validate_registry(registry_mutant, registry_schema, task_schema, format_checker)
    if not errors:
        raise SystemExit("empty task registry count negative control unexpectedly passed")
    print("CEB_TASK_REGISTRY_SELF_TEST_PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    base = args.repo_root.resolve() / "benchmarks" / "ceb"
    try:
        registry = load(base / "task-admission-registry.v0.json")
        registry_schema = load(base / "task-admission-registry.schema.json")
        task_schema = load(base / "task-admission.schema.json")
        Draft202012Validator.check_schema(registry_schema)
        Draft202012Validator.check_schema(task_schema)
    except Exception as error:
        print(f"CEB_TASK_ADMISSION_ERROR: {error}", file=sys.stderr)
        return 2
    format_checker = FormatChecker()
    errors = validate_registry(registry, registry_schema, task_schema, format_checker)
    if errors:
        for error in errors:
            print(f"CEB_TASK_ADMISSION_FAIL: {error}", file=sys.stderr)
        return 1
    if args.self_test:
        self_test(registry, registry_schema, task_schema, format_checker)
    print(
        "CEB_TASK_ADMISSION_PASS: "
        f"families={len(registry['family_targets'])} "
        f"target_slots={registry['target_task_count']} "
        f"actual_tasks={registry['actual_task_count']} admitted=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
