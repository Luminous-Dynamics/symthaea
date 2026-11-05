#!/usr/bin/env python3
"""Validate CEB release versioning, saturation and resource-calibration policy offline."""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load {path}: {error}") from error


def semver(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        raise ValueError(f"invalid semantic version {value!r}")
    return tuple(int(part) for part in match.groups())


def lifecycle_errors(entry: dict) -> list[str]:
    errors = []
    excluded = entry["task_status"] != "active"
    if excluded and entry["included_in_current_aggregate"]:
        errors.append("non-active task must not be in current aggregate")
    if not excluded and not entry["included_in_current_aggregate"]:
        errors.append("active task must remain in current aggregate")
    if not entry["included_in_longitudinal_panel"]:
        errors.append("all active, superseded, invalid and saturated tasks must remain in longitudinal panel")
    if not entry["historical_denominator_preserved"]:
        errors.append("task lifecycle transitions must preserve historical denominators")
    return errors


def saturation_errors(record: dict, registered_families: list[str], minimum_trials: int) -> list[str]:
    errors = []
    results = record["configuration_results"]
    observed = {row["model_family_id"] for row in results}
    expected = set(registered_families)
    if not expected:
        errors.append("saturation cannot be assessed without a preregistered current-generation model-family registry")
    if observed != expected:
        errors.append(f"saturation cohort mismatch; missing={sorted(expected-observed)}, extra={sorted(observed-expected)}")
    for row in results:
        if row["trials"] < minimum_trials:
            errors.append(f"{row['model_family_id']}/{row['configuration_id']}: fewer than {minimum_trials} independent trials")
        if len(row["trial_result_digests"]) != row["trials"]:
            errors.append(f"{row['model_family_id']}/{row['configuration_id']}: trial result digest count mismatch")
        if row["successes"] > row["trials"]:
            errors.append(f"{row['model_family_id']}/{row['configuration_id']}: successes exceed trials")
    configuration_keys = [(row["model_family_id"], row["configuration_id"]) for row in results]
    if len(configuration_keys) != len(set(configuration_keys)):
        errors.append("duplicate model-family/configuration saturation cohort")
    all_saturated = bool(results) and observed == expected and all(
        row["trials"] >= minimum_trials
        and row["successes"] == row["trials"]
        and len(row["trial_result_digests"]) == row["trials"]
        for row in results
    )
    if record["decision"] == "confirmed_saturated" and not all_saturated:
        errors.append("confirmed saturation requires every registered family/configuration to succeed on every trial")
    if record["decision"] == "saturated_candidate" and all_saturated:
        errors.append("fully satisfied saturation evidence must be confirmed_saturated, not candidate")
    if record["decision"] == "not_saturated" and all_saturated:
        errors.append("complete 5/5 success across all registered cohorts cannot be labeled not_saturated")
    return errors


def release_semantic_errors(release: dict) -> list[str]:
    errors = []
    action = release["release_action"]
    changes = release["change_records"]
    previous = release["previous_version"]
    current = semver(release["current_version"])
    classes = {row["change_class"] for row in changes}
    digests = release["artifact_digests"]

    for change in changes:
        if change["change_class"] == "agent_visible" and change["evaluation_inputs_unchanged"]:
            errors.append(f"{change['change_id']}: agent-visible change cannot claim evaluation inputs are unchanged")
        if change["change_class"] == "agent_visible" and not change["affected_task_ids"]:
            errors.append(f"{change['change_id']}: agent-visible change requires an explicit affected-task set")
        if change["change_class"] == "grader_only" and not change["evaluation_inputs_unchanged"]:
            errors.append(f"{change['change_id']}: grader-only change must preserve agent-visible evaluation inputs")
        if change["change_class"] == "metadata_only" and (
            not change["evaluation_inputs_unchanged"] or change["affected_task_ids"]
        ):
            errors.append(f"{change['change_id']}: metadata-only change cannot alter inputs or task membership")
        if change["change_class"] == "grader_only" and not change["affected_task_ids"]:
            errors.append(f"{change['change_id']}: grader changes require an explicit affected-task set")

    if release["previous_release_id"] is None:
        if previous is not None or action != "initial_design_no_qualified_results":
            errors.append("initial release must have no parent and use initial_design_no_qualified_results")
        if release["status"] != "design_only_no_qualified_results":
            errors.append("initial release cannot claim to be qualified or released")
        if changes or release["task_lifecycle_entries"] or release["saturation_observations"]:
            errors.append("initial design-only release cannot contain unqualified changes, lifecycle decisions or saturation observations")
        if release["resource_calibration"]["status"] != "not_run_blocked" or release["resource_calibration"]["actual_reference_task_count"] != 0 or release["resource_calibration"]["minimum_trials_observed_per_reference_task"] != 0:
            errors.append("initial design-only release cannot claim resource calibration")
        if any(value is not None for value in digests.values()) or release["previous_evaluation_input_sha256"] is not None:
            errors.append("initial design-only release cannot invent qualified artifact digests")
    else:
        if previous is None:
            errors.append("non-initial release requires previous_version")
        elif not changes:
            errors.append("non-initial release requires at least one classified change record")
        else:
            old = semver(previous)
            if "agent_visible" in classes:
                expected_version = (old[0] + 1, 0, 0)
                expected_action = "full_rerun"
            elif "grader_only" in classes:
                expected_version = (old[0], old[1] + 1, 0)
                all_raw = all(row["raw_execution_artifacts_available"] for row in changes if row["change_class"] == "grader_only")
                expected_action = "regrade_affected" if all_raw else "affected_task_rerun"
            else:
                expected_version = (old[0], old[1], old[2] + 1)
                expected_action = "reuse"
                if any(not row["evaluation_inputs_unchanged"] for row in changes):
                    errors.append("metadata-only change cannot alter evaluation inputs")
                if release["artifact_digests"]["evaluation_input_sha256"] is None or release["previous_evaluation_input_sha256"] is None:
                    errors.append("metadata reuse requires previous and current evaluation-input digests")
            if current != expected_version:
                errors.append(f"version bump mismatch; expected {expected_version[0]}.{expected_version[1]}.{expected_version[2]}")
            if action != expected_action:
                errors.append(f"release action mismatch; expected {expected_action}")
            if expected_action == "reuse":
                if release["artifact_digests"]["evaluation_input_sha256"] != release["previous_evaluation_input_sha256"]:
                    errors.append("result reuse requires identical previous/current evaluation-input digests")

    if release["status"] == "design_only_no_qualified_results" and (
        release["task_lifecycle_entries"] or release["saturation_observations"]
    ):
        errors.append("design-only release cannot claim task lifecycle or saturation results")
    for record in release["saturation_observations"]:
        if record["decision"] == "confirmed_saturated":
            entries = [e for e in release["task_lifecycle_entries"] if e["task_id"] == record["task_id"]]
            if not any(e["task_status"] == "saturated_retained" for e in entries):
                errors.append(f"{record['task_id']}: confirmed saturated task needs a saturated_retained lifecycle entry")

    if release["status"] == "released":
        required_digests = [
            digests["task_set_sha256"], digests["prompt_set_sha256"], digests["source_set_sha256"],
            digests["environment_set_sha256"], digests["agent_tool_policy_sha256"], digests["resource_policy_sha256"],
            digests["grader_sha256"], digests["rubric_sha256"], digests["reference_control_set_sha256"],
            digests["metadata_sha256"], digests["evaluation_input_sha256"]
        ]
        if any(value is None for value in required_digests):
            errors.append("released benchmark requires all task, environment, policy, grader and metadata digests")
        if release["resource_calibration"]["status"] != "qualified":
            errors.append("released benchmark requires qualified resource calibration")
    if release["resource_calibration"]["status"] == "qualified":
        calibration = release["resource_calibration"]
        if calibration["actual_reference_task_count"] < calibration["planned_reference_task_count"]:
            errors.append("qualified resource calibration has too few representative reference tasks")
        if calibration["minimum_trials_observed_per_reference_task"] < calibration["planned_independent_runs_per_reference_task"]:
            errors.append("qualified resource calibration has fewer than the planned repeated runs per reference task")
        if calibration["observed_infrastructure_failure_rate"] is None or calibration["observed_infrastructure_failure_rate"] > calibration["infrastructure_failure_rate_target"]:
            errors.append("qualified resource calibration exceeds its infrastructure-failure target")
        if not calibration["resource_policy_sha256"] or not calibration["calibration_record_sha256"] or not calibration["agent_timeout_policy_sha256"]:
            errors.append("qualified resource calibration requires policy, record and agent-timeout digests")

    for entry in release["task_lifecycle_entries"]:
        errors.extend(f"{entry['task_id']}: {error}" for error in lifecycle_errors(entry))
    for record in release["saturation_observations"]:
        errors.extend(
            f"{record['task_id']}: {error}"
            for error in saturation_errors(
                record,
                release["saturation_policy"]["model_family_registry"],
                release["saturation_policy"]["minimum_trials_per_configuration"]
            )
        )
    if not release["historical_policy"]["immutable_release_records"] or not release["historical_policy"]["no_silent_task_deletion"]:
        errors.append("historical release and task records must be immutable; silent deletion is prohibited")
    return errors


def validate_release(release: dict, schema: dict) -> list[str]:
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    schema_errors = sorted(validator.iter_errors(release), key=lambda error: list(error.absolute_path))
    if schema_errors:
        return [
            f"schema {'/'.join(map(str, error.absolute_path)) or '<root>'}: {error.message}"
            for error in schema_errors
        ]
    return release_semantic_errors(release)


def saturation_fixture():
    return {
        "task_id":"task-001","release_id":"ceb-v0-test","model_generation":"frontier-2026q4","observation_date":"2026-11-05",
        "configuration_results":[
            {"model_family_id":"family-a","configuration_id":"harness-a","trials":5,"successes":5,"trial_result_digests":["a"*64]*5},
            {"model_family_id":"family-b","configuration_id":"harness-b","trials":5,"successes":5,"trial_result_digests":["b"*64]*5}
        ],
        "decision":"confirmed_saturated"
    }


def self_test(release: dict, schema: dict) -> None:
    errors = validate_release(release, schema)
    if errors:
        raise SystemExit("valid initial release fixture failed: " + "; ".join(errors))

    def must_reject(mutant: dict, label: str):
        if not validate_release(mutant, schema):
            raise SystemExit(f"release lifecycle negative control unexpectedly passed: {label}")

    mutant = copy.deepcopy(release)
    mutant["release_action"] = "reuse"
    must_reject(mutant, "initial design misreported as result reuse")

    mutant = copy.deepcopy(release)
    mutant["resource_calibration"]["status"] = "qualified"
    mutant["resource_calibration"]["actual_reference_task_count"] = 0
    mutant["resource_calibration"]["minimum_trials_observed_per_reference_task"] = 0
    must_reject(mutant, "resource calibration qualified without trials")

    mutant = copy.deepcopy(release)
    mutant["saturation_policy"]["model_family_registry"] = ["family-a","family-b"]
    record = saturation_fixture()
    record["configuration_results"][1]["successes"] = 4
    mutant["saturation_observations"] = [record]
    must_reject(mutant, "task marked saturated with a 4/5 family")

    mutant = copy.deepcopy(release)
    mutant["saturation_policy"]["model_family_registry"] = ["family-a","family-b"]
    record = saturation_fixture()
    record["configuration_results"].pop()
    mutant["saturation_observations"] = [record]
    must_reject(mutant, "saturation without all registered model families")

    mutant = copy.deepcopy(release)
    mutant["task_lifecycle_entries"] = [{
      "task_id":"task-001","task_status":"saturated_retained","previous_status":"active",
      "effective_release_id":"ceb-v0-design","reason":"solved 5/5 by all cohorts","decision_record_sha256":"c"*64,
      "included_in_current_aggregate":False,"included_in_longitudinal_panel":False,"historical_denominator_preserved":False
    }]
    must_reject(mutant, "saturated task silently removed from historical denominator")

    # A synthetic future release with an agent-visible change must make a major bump and rerun.
    mutant = copy.deepcopy(release)
    mutant.update({
      "status":"release_candidate","release_id":"ceb-v1-test","current_version":"0.1.1",
      "previous_release_id":"ceb-v0-design","previous_version":"0.1.0","release_action":"reuse",
      "change_records":[{
        "change_id":"prompt-change-001","change_class":"agent_visible","paths":["benchmarks/ceb/tasks/task-001.json"],
        "old_sha256":"a"*64,"new_sha256":"b"*64,"affected_task_ids":["task-001"],
        "raw_execution_artifacts_available":True,"evaluation_inputs_unchanged":False
      }]
    })
    must_reject(mutant, "agent-visible prompt change under-bumped to patch/reuse")

    # A verifier-only change requires minor version; regrade only if raw artifacts exist.
    mutant = copy.deepcopy(release)
    mutant.update({
      "status":"release_candidate","release_id":"ceb-v0.2-test","current_version":"0.2.0",
      "previous_release_id":"ceb-v0-design","previous_version":"0.1.0","release_action":"regrade_affected",
      "change_records":[{
        "change_id":"verifier-change-001","change_class":"grader_only","paths":["benchmarks/ceb/verifiers/check.json"],
        "old_sha256":"a"*64,"new_sha256":"b"*64,"affected_task_ids":["task-001"],
        "raw_execution_artifacts_available":True,"evaluation_inputs_unchanged":True
      }]
    })
    if validate_release(mutant,schema):
        raise SystemExit("valid verifier-only regrade fixture unexpectedly failed")

    # Agent-visible changes cannot be marked unaffected.
    mutant=copy.deepcopy(release)
    mutant.update({
      "status":"release_candidate","release_id":"ceb-v1-test","current_version":"1.0.0",
      "previous_release_id":"ceb-v0-design","previous_version":"0.1.0","release_action":"full_rerun",
      "change_records":[{
        "change_id":"environment-change-001","change_class":"agent_visible","paths":["benchmarks/ceb/environments/image.json"],
        "old_sha256":"a"*64,"new_sha256":"b"*64,"affected_task_ids":["task-001"],
        "raw_execution_artifacts_available":False,"evaluation_inputs_unchanged":False
      }]
    })
    if validate_release(mutant,schema):
        raise SystemExit("valid major/full-rerun release fixture unexpectedly failed")

    # Metadata-only release may reuse results when evaluation inputs are identical.
    reuse=copy.deepcopy(release)
    reuse.update({
      "status":"release_candidate","release_id":"ceb-v0.1.1-reuse-test","current_version":"0.1.1",
      "previous_release_id":"ceb-v0-design","previous_version":"0.1.0","release_action":"reuse",
      "previous_evaluation_input_sha256":"f"*64,
      "change_records":[{
        "change_id":"docs-change-positive-001","change_class":"metadata_only","paths":["docs/CEB.md"],
        "old_sha256":"a"*64,"new_sha256":"b"*64,"affected_task_ids":[],
        "raw_execution_artifacts_available":False,"evaluation_inputs_unchanged":True
      }]
    })
    reuse["artifact_digests"]["evaluation_input_sha256"]="f"*64
    if validate_release(reuse,schema):
        raise SystemExit("valid metadata-only result reuse unexpectedly failed")

    # Metadata changes can reuse results only if evaluation-input digests remain identical.
    mutant=copy.deepcopy(release)
    mutant.update({
      "status":"release_candidate","release_id":"ceb-v0.1.1-test","current_version":"0.1.1",
      "previous_release_id":"ceb-v0-design","previous_version":"0.1.0","release_action":"reuse",
      "change_records":[{
        "change_id":"docs-change-001","change_class":"metadata_only","paths":["docs/CEB.md"],
        "old_sha256":"a"*64,"new_sha256":"b"*64,"affected_task_ids":[],
        "raw_execution_artifacts_available":False,"evaluation_inputs_unchanged":True
      }],
      "previous_evaluation_input_sha256":"e"*64
    })
    mutant["artifact_digests"]["evaluation_input_sha256"]="d"*64
    must_reject(mutant, "metadata-only result reuse with changed evaluation digest")

    # Verifier changes without raw artifacts must rerun affected tasks, not regrade.
    mutant=copy.deepcopy(release)
    mutant.update({
      "status":"release_candidate","release_id":"ceb-v0.2-test","current_version":"0.2.0",
      "previous_release_id":"ceb-v0-design","previous_version":"0.1.0","release_action":"regrade_affected",
      "change_records":[{
        "change_id":"verifier-change-no-raw-001","change_class":"grader_only","paths":["benchmarks/ceb/verifier.json"],
        "old_sha256":"a"*64,"new_sha256":"b"*64,"affected_task_ids":["task-001"],
        "raw_execution_artifacts_available":False,"evaluation_inputs_unchanged":True
      }]
    })
    must_reject(mutant, "regrade requested without raw execution artifacts")

    # The positive saturation fixture and its 4/5 negative counterpart define the boundary.
    if saturation_errors(saturation_fixture(),["family-a","family-b"],5):
        raise SystemExit("5/5 saturation fixture unexpectedly failed")
    partial=saturation_fixture()
    partial["configuration_results"][1]["successes"]=4
    if not saturation_errors(partial,["family-a","family-b"],5):
        raise SystemExit("4/5 family fixture incorrectly met saturation")
    incomplete=saturation_fixture()
    incomplete["configuration_results"].pop()
    if not saturation_errors(incomplete,["family-a","family-b"],5):
        raise SystemExit("missing model-family fixture incorrectly met saturation")
    print("CEB_SATURATION_POLICY_SELF_TEST_PASS")
    print("CEB_RELEASE_LIFECYCLE_SELF_TEST_PASS")


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root",type=Path,default=Path.cwd())
    parser.add_argument("--self-test",action="store_true")
    args=parser.parse_args()
    base=args.repo_root.resolve()/"benchmarks"/"ceb"
    try:
        release=load(base/"release-lifecycle.v0.json")
        schema=load(base/"release-lifecycle.schema.json")
        Draft202012Validator.check_schema(schema)
    except Exception as error:
        print(f"CEB_RELEASE_LIFECYCLE_ERROR: {error}",file=sys.stderr)
        return 2
    errors=validate_release(release,schema)
    if errors:
        for error in errors:
            print(f"CEB_RELEASE_LIFECYCLE_FAIL: {error}",file=sys.stderr)
        return 1
    if args.self_test:
        self_test(release,schema)
    print(
        "CEB_RELEASE_LIFECYCLE_PASS: "
        f"version={release['current_version']} "
        f"changes={len(release['change_records'])} "
        f"task_lifecycle_entries={len(release['task_lifecycle_entries'])} "
        f"saturation_observations={len(release['saturation_observations'])} "
        f"reference_tasks={release['resource_calibration']['actual_reference_task_count']} "
        f"status={release['status']}"
    )
    return 0


if __name__=="__main__":
    raise SystemExit(main())
