#!/usr/bin/env python3
"""Mutation-test VLEB's own cohort-integrity guardrails.

A sentinel guard mutation is killed only if the unmodified adversarial suite
passes first, the mutated source compiles, the suite actually runs, and the
suite then fails. Import/syntax failures do not count as killed mutants.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

RAN_TESTS = re.compile(r"Ran \d+ tests?\b")
MUTATIONS: tuple[dict[str, str], ...] = (
    {
        "id": "cohort_manifest_digest",
        "description": "bypass evaluator-supplied cohort-manifest digest verification",
        "anchor": "if observed_cohort_sha != expected_cohort_sha256:",
        "replacement": "if False:",
    },
    {
        "id": "audit_report_digest",
        "description": "bypass per-report byte digest verification",
        "anchor": 'if digest(report_raw) != row["audit_report_sha256"]:',
        "replacement": "if False:",
    },
    {
        "id": "locked_task_inventory",
        "description": "bypass exact ordered task inventory verification",
        "anchor": 'if task_row_ids != manifest["task_ids"]:',
        "replacement": "if False:",
    },
    {
        "id": "unexpected_task_rejection",
        "description": "accept reports for task IDs outside the locked inventory",
        "anchor": "if task_id not in expected_task_ids:",
        "replacement": "if False:",
    },
    {
        "id": "run_attempt_binding",
        "description": "bypass run-ID and attempt-number binding",
        "anchor": 'if report["run_id"] != row["run_id"] or type(report["attempt_number"]) is not int or report["attempt_number"] != row["attempt_number"]:',
        "replacement": "if False:",
    },
    {
        "id": "checkout_revision_binding",
        "description": "bypass expected and observed checkout revision binding",
        "anchor": 'if report["expected_revision"] != row["expected_revision"] or report["observed_revision"] != row["expected_revision"]:',
        "replacement": "if False:",
    },
    {
        "id": "runtime_image_binding",
        "description": "bypass immutable runtime image binding",
        "anchor": 'if report["expected_image_id"] != row["expected_image_id"] or report["runtime_image_id"] != row["expected_image_id"]:',
        "replacement": "if False:",
    },
    {
        "id": "network_mode_binding",
        "description": "bypass observed network-mode binding",
        "anchor": 'if report["network_mode"] != row["expected_network_mode"]:',
        "replacement": "if False:",
    },
    {
        "id": "strict_duplicate_key_parser",
        "description": "replace strict cohort JSON parsing with default permissive parsing",
        "anchor": "manifest = strict_json_loads(cohort_raw)",
        "replacement": "manifest = json.loads(cohort_raw)",
    },
    {
        "id": "eligibility_consistency",
        "description": "allow eligible reports with incomplete admission summaries",
        "anchor": 'if report.get("eligible") is True and isinstance(trace_value,dict) and isinstance(baseline_value,dict):',
        "replacement": "if False:",
    },
    {
        "id": "score_eligibility_leak",
        "description": "make the rollup claim score eligibility",
        "anchor": '"analysis_only":True,\n        "score_eligible":False,',
        "replacement": '"analysis_only":True,\n        "score_eligible":True,',
    },
)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def run_suite(root: Path, *, timeout_seconds: int) -> dict[str, Any]:
    suite = root / "tools" / "vleb" / "test_summarize_admission_cohort.py"
    try:
        completed = subprocess.run(
            [sys.executable, str(suite)],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=timeout_seconds,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return {
            "returncode": None,
            "timed_out": True,
            "tests_started": False,
            "output_sha256": digest(str(output).encode()),
            "output_tail": str(output)[-2500:],
        }
    output = completed.stdout
    return {
        "returncode": completed.returncode,
        "timed_out": False,
        "tests_started": RAN_TESTS.search(output) is not None,
        "output_sha256": digest(output.encode("utf-8")),
        "output_tail": output[-2500:] if completed.returncode != 0 else None,
    }


def prepare_tree(
    destination: Path,
    source_path: Path,
    suite_path: Path,
    schema_path: Path,
    source_override: str | None = None,
) -> Path:
    tools_dir = destination / "tools" / "vleb"
    tools_dir.mkdir(parents=True, exist_ok=True)
    (destination / "tools").mkdir(parents=True, exist_ok=True)
    source = source_override if source_override is not None else source_path.read_text(encoding="utf-8")
    (tools_dir / source_path.name).write_text(source, encoding="utf-8")
    shutil.copyfile(suite_path, tools_dir / suite_path.name)
    shutil.copyfile(schema_path, tools_dir / schema_path.name)
    return destination


def run_campaign(
    source_path: Path,
    suite_path: Path,
    schema_path: Path,
    *,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    source_raw = source_path.read_bytes()
    suite_raw = suite_path.read_bytes()
    schema_raw = schema_path.read_bytes()
    source_text = source_raw.decode("utf-8")
    baseline: dict[str, Any]
    results: list[dict[str, Any]] = []

    with tempfile.TemporaryDirectory(prefix="vleb-auditor-mutation-") as temporary:
        root = Path(temporary) / "baseline"
        prepare_tree(root, source_path, suite_path, schema_path)
        baseline = run_suite(root, timeout_seconds=timeout_seconds)

    baseline_ok = (
        baseline["returncode"] == 0
        and baseline["tests_started"]
        and not baseline["timed_out"]
    )
    if not baseline_ok:
        return {
            "schema_version": "vleb-auditor-mutation-campaign-v1",
            "baseline_passed": False,
            "baseline": baseline,
            "source_sha256": digest(source_raw),
            "suite_sha256": digest(suite_raw),
            "schema_sha256": digest(schema_raw),
            "mutation_count": len(MUTATIONS),
            "killed_count": 0,
            "survived_count": 0,
            "invalid_count": 0,
            "mutation_kill_rate_pct": None,
            "analysis_only": True,
            "score_eligible": False,
            "mutations": [],
            "issues": [{"code":"baseline_suite_failed","message":"unmodified cohort adversarial suite must pass before mutation results are meaningful"}],
        }

    with tempfile.TemporaryDirectory(prefix="vleb-auditor-mutation-") as temporary:
        root_parent = Path(temporary)
        for mutation in MUTATIONS:
            count = source_text.count(mutation["anchor"])
            result: dict[str, Any] = {
                "id": mutation["id"],
                "description": mutation["description"],
                "anchor_count": count,
                "status": "invalid",
            }
            if count != 1:
                result["reason"] = "mutation anchor must occur exactly once"
                results.append(result)
                continue
            mutated_source = source_text.replace(mutation["anchor"], mutation["replacement"], 1)
            try:
                compile(mutated_source, str(source_path), "exec")
            except SyntaxError as exc:
                result["reason"] = f"mutated source does not compile: {exc.msg}"
                result["mutated_source_sha256"] = digest(mutated_source.encode())
                results.append(result)
                continue

            copy_root = root_parent / mutation["id"]
            prepare_tree(copy_root, source_path, suite_path, schema_path, mutated_source)
            execution = run_suite(copy_root, timeout_seconds=timeout_seconds)
            result["mutated_source_sha256"] = digest(mutated_source.encode())
            result["tests_started"] = execution["tests_started"]
            result["test_output_sha256"] = execution["output_sha256"]
            result["timed_out"] = execution["timed_out"]
            result["test_returncode"] = execution["returncode"]
            if execution["timed_out"]:
                result["reason"] = "mutated suite timed out; not counted as killed"
                result["status"] = "invalid"
            elif not execution["tests_started"]:
                result["reason"] = "test suite failed before reporting a test run; not counted as killed"
                result["status"] = "invalid"
                result["output_tail"] = execution.get("output_tail")
            elif execution["returncode"] != 0:
                result["status"] = "killed"
            else:
                result["status"] = "survived"
                result["output_tail"] = execution.get("output_tail")
            results.append(result)

    killed = sum(row["status"] == "killed" for row in results)
    survived = sum(row["status"] == "survived" for row in results)
    invalid = sum(row["status"] == "invalid" for row in results)
    count = len(results)
    issues = []
    if survived:
        issues.append({"code":"auditor_mutations_survived","message":f"{survived} sentinel guard mutation(s) survived"})
    if invalid:
        issues.append({"code":"auditor_mutations_invalid","message":f"{invalid} sentinel mutation(s) were invalid or failed to execute cleanly"})
    if killed != count:
        issues.append({"code":"auditor_mutation_campaign_incomplete","message":"every reviewed sentinel guard mutation must be killed by the adversarial test suite"})
    return {
        "schema_version": "vleb-auditor-mutation-campaign-v1",
        "baseline_passed": True,
        "baseline": baseline,
        "source_sha256": digest(source_raw),
        "suite_sha256": digest(suite_raw),
        "schema_sha256": digest(schema_raw),
        "mutation_count": count,
        "killed_count": killed,
        "survived_count": survived,
        "invalid_count": invalid,
        "mutation_kill_rate_pct": (100.0*killed/count) if count else None,
        "analysis_only": True,
        "score_eligible": False,
        "mutations": results,
        "issues": issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    tools_dir = Path(__file__).resolve().parent
    parser.add_argument("--source", type=Path, default=tools_dir / "summarize_admission_cohort.py")
    parser.add_argument("--tests", type=Path, default=tools_dir / "test_summarize_admission_cohort.py")
    parser.add_argument("--schema", type=Path, default=tools_dir / "admission-cohort.schema.json")
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.timeout_seconds < 1 or args.timeout_seconds > 900:
        parser.error("--timeout-seconds must be between 1 and 900")
    result = run_campaign(args.source, args.tests, args.schema, timeout_seconds=args.timeout_seconds)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result.get("baseline_passed") is True and result.get("mutation_kill_rate_pct") == 100.0 and result.get("survived_count") == 0 and result.get("invalid_count") == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
