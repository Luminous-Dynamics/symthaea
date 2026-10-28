#!/usr/bin/env python3
"""Pinned, analysis-only cohort rollup for VLEB task-admission reports.

The cohort manifest and each audit report must be pinned independently. This tool
pools counts instead of averaging per-task percentages and never grants score
eligibility. It reports integrity failures separately from candidate outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath
from typing import Any

try:
    from jsonschema import Draft202012Validator, FormatChecker
except ImportError as exc:
    raise SystemExit("jsonschema==4.26.0 is required; refusing to skip schema validation") from exc

HEX = re.compile(r"^[a-f0-9]{64}$")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key {key!r}")
        value[key] = item
    return value


def reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {value!r} rejected")


def strict_json_loads(raw: bytes | str) -> Any:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="strict")
    return json.loads(raw, object_pairs_hook=reject_duplicate_keys, parse_constant=reject_constant)


def safe_file(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise ValueError("path must be a non-empty canonical POSIX relative path")
    rel = PurePosixPath(relative)
    if rel.is_absolute() or ".." in rel.parts or "." in rel.parts or str(rel) != relative:
        raise ValueError("absolute, traversal, or non-canonical path rejected")
    resolved_root = root.resolve(strict=True)
    cursor = resolved_root
    for component in rel.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise ValueError("symlink evidence path rejected")
    resolved = cursor.resolve(strict=True)
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise ValueError("audit report must be a regular file under the artifact root")
    return resolved


def empty_summary(expected_count: int | None, issues: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "schema_version": "vleb-admission-cohort-summary-v1",
        "cohort_integrity_valid": False,
        "cohort_complete": False,
        "pooled_metrics_population": "verified_reports_only",
        "pooled_metrics_complete": False,
        "analysis_only": True,
        "score_eligible": False,
        "expected_task_count": expected_count,
        "verified_report_count": 0,
        "audit_eligible_task_count": 0,
        "audit_eligible_task_pct": None,
        "requirement_source_item_count": 0,
        "requirement_source_traced_count": 0,
        "requirement_source_coverage_pct": None,
        "required_test_id_count": 0,
        "traced_required_test_id_count": 0,
        "required_test_trace_coverage_pct": None,
        "traceability_complete_task_count": 0,
        "traceability_complete_task_pct": None,
        "baseline_concrete_execution_count": 0,
        "baseline_expected_status_count": 0,
        "baseline_observed_status_count": 0,
        "baseline_matching_status_count": 0,
        "per_test_baseline_oracle_alignment_pct": None,
        "baseline_complete_task_count": 0,
        "baseline_complete_task_pct": None,
        "full_admission_pass_task_count": 0,
        "full_admission_pass_task_pct": None,
        "issues": issues,
    }


def _nonnegative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _pct(numerator: int, denominator: int) -> float | None:
    return 100.0 * numerator / denominator if denominator else None


def _validate_report(
    report: Any,
    row: dict[str, Any],
    *,
    expected_protocol_sha256: str,
    expected_auditor_sha256: str,
) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    issues: list[dict[str, str]] = []
    if not isinstance(report, dict):
        return None, [{"code":"audit_report_invalid","message":"audit report root must be an object"}]

    required_top = ("task_id","eligible","verdict","receipt_sha256","task_contract_sha256",
                    "protocol_sha256","candidate_patch_sha256","auditor_sha256",
                    "requirement_traceability","per_test_baseline_oracle_alignment",
                    "run_id","attempt_number","expected_revision","observed_revision",
                    "expected_image_id","runtime_image_id","network_mode")
    missing = [key for key in required_top if key not in report]
    if missing:
        issues.append({"code":"audit_report_fields_missing","message":f"audit report is missing fields: {missing}"})
        return None, issues

    if report["task_id"] != row["task_id"]:
        issues.append({"code":"task_identity_mismatch","message":"audit report task_id differs from cohort row"})
    if report["run_id"] != row["run_id"] or type(report["attempt_number"]) is not int or report["attempt_number"] != row["attempt_number"]:
        issues.append({"code":"execution_identity_mismatch","message":"audit report run ID/attempt differs from evaluator-pinned cohort row"})
    if report["expected_revision"] != row["expected_revision"] or report["observed_revision"] != row["expected_revision"]:
        issues.append({"code":"checkout_revision_mismatch","message":"audit report expected/observed checkout revision differs from evaluator-pinned revision"})
    if report["expected_image_id"] != row["expected_image_id"] or report["runtime_image_id"] != row["expected_image_id"]:
        issues.append({"code":"runtime_image_mismatch","message":"audit report expected/observed runtime image differs from evaluator-pinned image"})
    if report["network_mode"] != row["expected_network_mode"]:
        issues.append({"code":"runtime_network_mode_mismatch","message":"audit report runtime network mode differs from evaluator-pinned mode"})
    for key in ("receipt_sha256","task_contract_sha256","protocol_sha256","candidate_patch_sha256","auditor_sha256"):
        if report.get(key) != row.get(key):
            issues.append({"code":f"{key}_mismatch","message":f"audit report {key} differs from cohort-row pin"})
    if report.get("protocol_sha256") != expected_protocol_sha256:
        issues.append({"code":"protocol_pin_mismatch","message":"audit report protocol differs from evaluator-supplied pin"})
    if report.get("auditor_sha256") != expected_auditor_sha256:
        issues.append({"code":"auditor_pin_mismatch","message":"audit report auditor differs from evaluator-supplied pin"})
    if not isinstance(report.get("eligible"), bool) or report.get("verdict") not in ("pass","fail","unresolved"):
        issues.append({"code":"audit_verdict_invalid","message":"eligible must be boolean and verdict must be pass/fail/unresolved"})
    elif (report["eligible"] and report["verdict"] != "pass") or (not report["eligible"] and report["verdict"] == "pass"):
        issues.append({"code":"audit_verdict_inconsistent","message":"eligible boolean and verdict disagree"})

    trace = report.get("requirement_traceability")
    if not isinstance(trace, dict):
        issues.append({"code":"traceability_summary_invalid","message":"requirement_traceability must be an object"})
    else:
        fields = ("declared_source_item_count","traced_source_item_count","required_test_count","traced_required_test_count")
        if any(not _nonnegative_int(trace.get(key)) for key in fields) or not isinstance(trace.get("complete"), bool):
            issues.append({"code":"traceability_summary_invalid","message":"traceability counts must be non-negative integers and complete must be boolean"})
        elif (
            trace["traced_source_item_count"] > trace["declared_source_item_count"]
            or trace["traced_required_test_count"] > trace["required_test_count"]
            or (trace["complete"] and (
                trace["traced_source_item_count"] != trace["declared_source_item_count"]
                or trace["traced_required_test_count"] != trace["required_test_count"]
            ))
        ):
            issues.append({"code":"traceability_counts_inconsistent","message":"traceability summary counts are internally inconsistent"})

    baseline = report.get("per_test_baseline_oracle_alignment")
    if not isinstance(baseline, dict):
        issues.append({"code":"baseline_summary_invalid","message":"per_test_baseline_oracle_alignment must be an object"})
    else:
        fields = ("concrete_execution_count","expected_status_count","observed_status_count","matching_status_count")
        if any(not _nonnegative_int(baseline.get(key)) for key in fields) or not isinstance(baseline.get("complete"), bool):
            issues.append({"code":"baseline_summary_invalid","message":"baseline summary counts must be non-negative integers and complete must be boolean"})
        elif (
            baseline["expected_status_count"] > baseline["concrete_execution_count"]
            or baseline["observed_status_count"] > baseline["concrete_execution_count"]
            or baseline["matching_status_count"] > min(baseline["expected_status_count"],baseline["observed_status_count"])
            or (baseline["complete"] and not (
                baseline["expected_status_count"] == baseline["concrete_execution_count"]
                and baseline["observed_status_count"] == baseline["concrete_execution_count"]
                and baseline["matching_status_count"] == baseline["concrete_execution_count"]
            ))
        ):
            issues.append({"code":"baseline_counts_inconsistent","message":"baseline alignment summary counts are internally inconsistent"})
        pct = baseline.get("alignment_pct")
        expected_pct = _pct(baseline.get("matching_status_count",0),baseline.get("concrete_execution_count",0))
        if expected_pct is None:
            if pct is not None:
                issues.append({"code":"baseline_pct_inconsistent","message":"alignment_pct must be null when there are no mapped concrete tests"})
        elif not isinstance(pct,(int,float)) or isinstance(pct,bool) or abs(float(pct)-expected_pct)>1e-8:
            issues.append({"code":"baseline_pct_inconsistent","message":"alignment_pct does not equal the reported matching/denominator counts"})

    trace_value = report.get("requirement_traceability")
    baseline_value = report.get("per_test_baseline_oracle_alignment")
    if report.get("eligible") is True and isinstance(trace_value,dict) and isinstance(baseline_value,dict):
        if trace_value.get("complete") is not True or baseline_value.get("complete") is not True:
            issues.append({"code":"audit_eligibility_inconsistent","message":"eligible audit report must have complete requirement traceability and complete baseline alignment"})
    if issues:
        return None, issues
    return report, []


def summarize_cohort(
    manifest_path: Path,
    schema_path: Path,
    artifact_root: Path,
    *,
    expected_cohort_sha256: str,
    expected_schema_sha256: str,
    expected_protocol_sha256: str,
    expected_auditor_sha256: str,
    expected_summarizer_sha256: str,
) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    expected_count: int | None = None
    pins = {
        "cohort": expected_cohort_sha256,
        "schema": expected_schema_sha256,
        "protocol": expected_protocol_sha256,
        "auditor": expected_auditor_sha256,
        "summarizer": expected_summarizer_sha256,
    }
    for label, value in pins.items():
        if not isinstance(value,str) or not HEX.fullmatch(value):
            issues.append({"code":"invalid_external_pin","message":f"external {label} pin must be lowercase SHA-256"})
    if issues:
        return empty_summary(None,issues)

    try:
        observed_summarizer_sha = digest(Path(__file__).read_bytes())
        if observed_summarizer_sha != expected_summarizer_sha256:
            return empty_summary(None,[{"code":"summarizer_pin_mismatch","message":"running cohort summarizer differs from evaluator-supplied SHA-256"}])
        schema_raw = schema_path.read_bytes()
        if digest(schema_raw) != expected_schema_sha256:
            return empty_summary(None,[{"code":"schema_pin_mismatch","message":"cohort schema differs from evaluator-supplied SHA-256"}])
        cohort_raw = manifest_path.read_bytes()
    except OSError as exc:
        return empty_summary(None,[{"code":"trusted_input_unavailable","message":str(exc)}])

    observed_cohort_sha = digest(cohort_raw)
    if observed_cohort_sha != expected_cohort_sha256:
        return empty_summary(None,[{"code":"cohort_pin_mismatch","message":"cohort manifest differs from evaluator-supplied SHA-256"}])
    try:
        manifest = strict_json_loads(cohort_raw)
        schema = strict_json_loads(schema_raw)
    except (UnicodeError,ValueError,json.JSONDecodeError) as exc:
        return empty_summary(None,[{"code":"strict_json_invalid","message":str(exc)}])
    try:
        Draft202012Validator.check_schema(schema)
        schema_errors = list(Draft202012Validator(schema,format_checker=FormatChecker()).iter_errors(manifest))
    except Exception as exc:
        return empty_summary(None,[{"code":"schema_validator_error","message":str(exc)}])
    for error in schema_errors:
        path = ".".join(map(str,error.absolute_path)) or "$"
        issues.append({"code":"cohort_schema_invalid","message":f"{path}: {error.message}"})
    if issues:
        return empty_summary(None,issues)
    if not isinstance(manifest,dict):
        return empty_summary(None,[{"code":"cohort_root_invalid","message":"cohort manifest root must be an object"}])

    expected_count = len(manifest["task_ids"])
    if manifest["protocol_sha256"] != expected_protocol_sha256:
        issues.append({"code":"cohort_protocol_pin_mismatch","message":"cohort manifest protocol differs from evaluator-supplied pin"})
    rows = manifest["task_results"]
    expected_task_ids = set(manifest["task_ids"])
    task_row_ids = [row["task_id"] for row in rows]
    if task_row_ids != manifest["task_ids"]:
        issues.append({"code":"task_inventory_mismatch","message":"task_results must match the locked task_ids inventory in order"})
    if len(task_row_ids) != len(set(task_row_ids)):
        issues.append({"code":"duplicate_task_result","message":"cohort manifest has duplicate task result rows"})

    valid_reports: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in rows:
        task_id = row["task_id"]
        if task_id not in expected_task_ids:
            issues.append({"code":"unknown_task_result","message":f"{task_id}: task result is outside the locked task inventory"})
            continue
        if task_id in seen_ids:
            continue
        seen_ids.add(task_id)
        if row["protocol_sha256"] != expected_protocol_sha256:
            issues.append({"code":"cohort_row_protocol_mismatch","message":f"{task_id}: row protocol differs from evaluator-supplied pin"})
        if row["auditor_sha256"] != expected_auditor_sha256:
            issues.append({"code":"cohort_row_auditor_mismatch","message":f"{task_id}: row auditor differs from evaluator-supplied pin"})
        try:
            report_path = safe_file(artifact_root,row["audit_report_path"])
            report_raw = report_path.read_bytes()
        except (OSError,ValueError) as exc:
            issues.append({"code":"audit_report_unavailable","message":f"{task_id}: {exc}"})
            continue
        if digest(report_raw) != row["audit_report_sha256"]:
            issues.append({"code":"audit_report_digest_mismatch","message":f"{task_id}: audit report bytes differ from row digest"})
            continue
        try:
            report_value = strict_json_loads(report_raw)
        except (UnicodeError,ValueError,json.JSONDecodeError) as exc:
            issues.append({"code":"audit_report_json_invalid","message":f"{task_id}: {exc}"})
            continue
        report, report_issues = _validate_report(
            report_value,row,
            expected_protocol_sha256=expected_protocol_sha256,
            expected_auditor_sha256=expected_auditor_sha256,
        )
        if report_issues:
            for issue in report_issues:
                issues.append({"code":issue["code"],"message":f"{task_id}: {issue['message']}"})
            continue
        valid_reports.append(report)

    source_total = source_traced = test_total = test_traced = 0
    baseline_total = baseline_expected = baseline_observed = baseline_matches = 0
    audit_eligible_count = trace_complete_count = baseline_complete_count = full_pass_count = 0
    for report in valid_reports:
        trace = report["requirement_traceability"]
        baseline = report["per_test_baseline_oracle_alignment"]
        source_total += trace["declared_source_item_count"]
        source_traced += trace["traced_source_item_count"]
        test_total += trace["required_test_count"]
        test_traced += trace["traced_required_test_count"]
        baseline_total += baseline["concrete_execution_count"]
        baseline_expected += baseline["expected_status_count"]
        baseline_observed += baseline["observed_status_count"]
        baseline_matches += baseline["matching_status_count"]
        trace_complete = trace["complete"] is True
        baseline_complete = baseline["complete"] is True
        trace_complete_count += int(trace_complete)
        baseline_complete_count += int(baseline_complete)
        audit_eligible = report["eligible"] is True
        audit_eligible_count += int(audit_eligible)
        full_pass_count += int(audit_eligible and trace_complete and baseline_complete)

    valid_report_count = len(valid_reports)
    complete = not issues and valid_report_count == expected_count
    summary = empty_summary(expected_count,issues)
    summary.update({
        "cohort_id":manifest["cohort_id"],
        "cohort_sha256":observed_cohort_sha,
        "schema_sha256":expected_schema_sha256,
        "protocol_sha256":expected_protocol_sha256,
        "auditor_sha256":expected_auditor_sha256,
        "summarizer_sha256":observed_summarizer_sha,
        "cohort_integrity_valid":complete,
        "cohort_complete":complete,
        "pooled_metrics_population":"all_locked_tasks" if complete else "verified_reports_only",
        "pooled_metrics_complete":complete,
        "verified_report_count":valid_report_count,
        "audit_eligible_task_count":audit_eligible_count,
        "audit_eligible_task_pct":_pct(audit_eligible_count,expected_count),
        "requirement_source_item_count":source_total,
        "requirement_source_traced_count":source_traced,
        "requirement_source_coverage_pct":_pct(source_traced,source_total),
        "required_test_id_count":test_total,
        "traced_required_test_id_count":test_traced,
        "required_test_trace_coverage_pct":_pct(test_traced,test_total),
        "traceability_complete_task_count":trace_complete_count,
        "traceability_complete_task_pct":_pct(trace_complete_count,expected_count),
        "baseline_concrete_execution_count":baseline_total,
        "baseline_expected_status_count":baseline_expected,
        "baseline_observed_status_count":baseline_observed,
        "baseline_matching_status_count":baseline_matches,
        "per_test_baseline_oracle_alignment_pct":_pct(baseline_matches,baseline_total),
        "baseline_complete_task_count":baseline_complete_count,
        "baseline_complete_task_pct":_pct(baseline_complete_count,expected_count),
        "full_admission_pass_task_count":full_pass_count,
        "full_admission_pass_task_pct":_pct(full_pass_count,expected_count),
        "analysis_only":True,
        "score_eligible":False,
        "issues":issues,
    })
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--schema",type=Path,default=Path(__file__).with_name("admission-cohort.schema.json"))
    parser.add_argument("--artifact-root",type=Path,required=True)
    parser.add_argument("--expected-cohort-sha256",required=True)
    parser.add_argument("--expected-schema-sha256",required=True)
    parser.add_argument("--expected-protocol-sha256",required=True)
    parser.add_argument("--expected-auditor-sha256",required=True)
    parser.add_argument("--expected-summarizer-sha256",required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    result=summarize_cohort(
        args.manifest,args.schema,args.artifact_root,
        expected_cohort_sha256=args.expected_cohort_sha256,
        expected_schema_sha256=args.expected_schema_sha256,
        expected_protocol_sha256=args.expected_protocol_sha256,
        expected_auditor_sha256=args.expected_auditor_sha256,
        expected_summarizer_sha256=args.expected_summarizer_sha256,
    )
    rendered=json.dumps(result,indent=2,sort_keys=True)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(rendered,encoding="utf-8")
    print(rendered,end="")
    return 0 if result.get("cohort_integrity_valid") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
