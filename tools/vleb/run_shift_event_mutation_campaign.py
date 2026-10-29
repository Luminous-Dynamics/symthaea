#!/usr/bin/env python3
"""Mutation-test VLEB's shift-event evidence auditor against its named regression tests.

A sentinel is killed only when the original suite passes, the mutated auditor
compiles, the suite actually runs, and the specifically assigned test fails.
Invalid runs and surviving sentinels both fail the campaign.
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

RAN_TESTS = re.compile(r"Ran [1-9]\\d* tests?\\b")
FAILURE_NAMES = re.compile(r"^(?:FAIL|ERROR): (test_[A-Za-z0-9_]+) \\(", re.MULTILINE)
MUTATIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "manifest_protocol_pin_bypass", "family": "trusted_input_pins",
        "description": "bypass the external protocol pin",
        "expected_failure_test": "test_protocol_pin_mismatch_fails_closed",
        "replacements": [{"anchor": 'if manifest["protocol_sha256"] != expected_protocol_sha256:', "replacement": "if False:"}],
    },
    {
        "id": "input_manifest_pin_bypass", "family": "trusted_input_pins",
        "description": "skip the externally pinned raw manifest digest",
        "expected_failure_test": "test_cli_wrong_manifest_pin_fails_closed",
        "replacements": [{"anchor": "if observed_input_sha256 != args.expected_input_sha256:", "replacement": "if False:"}],
    },
    {
        "id": "auditor_source_pin_bypass", "family": "trusted_input_pins",
        "description": "skip the expected verifier source digest check",
        "expected_failure_test": "test_cli_wrong_verifier_pin_fails_closed",
        "replacements": [{"anchor": "if observed_verifier_sha256 != expected_verifier_sha256:", "replacement": "if False:"}],
    },
    {
        "id": "scenario_digest_binding_bypass", "family": "scenario_identity",
        "description": "accept a receipt bound to a different scenario digest",
        "expected_failure_test": "test_scenario_hash_mismatch_is_rejected",
        "replacements": [{"anchor": 'if receipt["scenario_sha256"] != scenario_hashes[scenario_id]:', "replacement": "if False:"}],
    },
    {
        "id": "task_contract_binding_bypass", "family": "scenario_identity",
        "description": "skip condition-specific before/after task contract binding",
        "expected_failure_test": "test_task_contract_digest_mismatch_is_rejected",
        "replacements": [{"anchor": 'if receipt["task_contract_sha256"] != (', "replacement": "if False and ("}],
    },
    {
        "id": "starting_state_pin_bypass", "family": "starting_state_identity",
        "description": "skip receipt-to-scenario initial state/checkpoint/revision/image checks",
        "expected_failure_test": "test_initial_checkpoint_must_match_locked_scenario",
        "replacements": [{"anchor": "if receipt[key] != expected:", "replacement": "if False:"}],
    },
    {
        "id": "event_payload_binding_bypass", "family": "shift_event_integrity",
        "description": "accept an observed event with a different change payload",
        "expected_failure_test": "test_shift_payload_must_match_locked_scenario",
        "replacements": [{"anchor": 'if event["payload_sha256"] != scenario["shift_payload_sha256"]:', "replacement": "if False:"}],
    },
    {
        "id": "event_change_family_bypass", "family": "shift_event_integrity",
        "description": "accept an event classified as a different change family",
        "expected_failure_test": "test_event_change_family_must_match_locked_scenario",
        "replacements": [{"anchor": 'if event["change_family"] != scenario["change_family"]:', "replacement": "if False:"}],
    },
    {
        "id": "runner_event_origin_bypass", "family": "shift_event_integrity",
        "description": "skip the declared applied/trusted-runner guard",
        "expected_failure_test": "test_event_must_be_runner_sourced_and_applied",
        "replacements": [{"anchor": 'if event["applied"] is not True or event["event_source"] != "trusted_runner":', "replacement": "if False:"}],
    },
    {
        "id": "stale_inventory_bypass", "family": "evidence_freshness",
        "description": "allow incomplete stale-artifact inventory accounting",
        "expected_failure_test": "test_stale_evidence_inventory_must_be_complete_and_event_bound",
        "replacements": [
            {"anchor": 'if len(stale_disposition) != len(scenario["stale_artifact_sha256s"]):', "replacement": "if False:"},
            {"anchor": 'if seen_stale != set(scenario["stale_artifact_sha256s"]):', "replacement": "if False:"},
        ],
    },
    {
        "id": "final_workspace_revalidation_bypass", "family": "evidence_freshness",
        "description": "accept checks that reference a different workspace state",
        "expected_failure_test": "test_all_required_checks_must_be_present_and_bind_final_state",
        "replacements": [{"anchor": 'if check["input_workspace_sha256"] != receipt["final_workspace_sha256"]:', "replacement": "if False:"}],
    },
    {
        "id": "passing_outcome_guard_bypass", "family": "evidence_freshness",
        "description": "allow a receipt to pass despite failed or unresolved required checks",
        "expected_failure_test": "test_pass_cannot_hide_a_failed_required_check",
        "replacements": [{"anchor": 'if receipt["outcome"] == "pass" and any(check["status"] != "pass" for check in checks):', "replacement": "if False:"}],
    },
    {
        "id": "retry_substitution_bypass", "family": "attempt_integrity",
        "description": "allow a retry to replace the preregistered primary attempt",
        "expected_failure_test": "test_retry_substitution_is_rejected",
        "replacements": [{"anchor": "if type(receipt[\"attempt_number\"]) is not int or receipt[\"attempt_number\"] != 1:", "replacement": "if False:"}],
    },
)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def replace_reviewed_anchors(source: str, mutation: dict[str, Any]) -> str:
    mutated = source
    replacements = mutation.get("replacements")
    if not isinstance(replacements, list) or not replacements:
        raise ValueError("mutation must contain at least one replacement")
    for replacement in replacements:
        anchor, new = replacement.get("anchor"), replacement.get("replacement")
        if not isinstance(anchor, str) or not anchor or not isinstance(new, str):
            raise ValueError("each replacement needs a non-empty anchor and a string replacement")
        count = mutated.count(anchor)
        if count != 1:
            raise ValueError(f"mutation {mutation.get('id')} anchor count is {count}, expected exactly one")
        mutated = mutated.replace(anchor, new, 1)
    return mutated


def run_suite(root: Path, timeout_seconds: int) -> dict[str, Any]:
    suite = root / "test_audit_shift_event_evidence.py"
    try:
        completed = subprocess.run(
            [sys.executable, str(suite)], cwd=root, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, check=False, timeout=timeout_seconds,
            text=True, encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        return {
            "returncode": None, "timed_out": True, "tests_started": False,
            "failure_test_names": [], "output_sha256": digest(str(output).encode("utf-8")),
            "output_tail": str(output)[-2500:],
        }
    output = completed.stdout
    return {
        "returncode": completed.returncode,
        "timed_out": False,
        "tests_started": RAN_TESTS.search(output) is not None,
        "failure_test_names": sorted(set(FAILURE_NAMES.findall(output))),
        "output_sha256": digest(output.encode("utf-8")),
        "output_tail": output[-3000:] if completed.returncode else None,
    }


def mutation_campaign(
    auditor_path: Path,
    suite_path: Path,
    *,
    timeout_seconds: int = 90,
) -> dict[str, Any]:
    source_raw = auditor_path.read_bytes()
    suite_raw = suite_path.read_bytes()
    source = source_raw.decode("utf-8", errors="strict")
    source_sha = digest(source_raw)
    suite_sha = digest(suite_raw)
    with tempfile.TemporaryDirectory(prefix="vleb-shift-audit-baseline-") as temporary:
        root = Path(temporary)
        (root / auditor_path.name).write_bytes(source_raw)
        (root / suite_path.name).write_bytes(suite_raw)
        baseline = run_suite(root, timeout_seconds)

    baseline_passed = (
        baseline["returncode"] == 0 and baseline["tests_started"] and not baseline["timed_out"]
    )
    if not baseline_passed:
        return {
            "schema_version": "vleb-shift-event-mutation-campaign-v1",
            "campaign_passed": False, "baseline_passed": False,
            "auditor_source_sha256": source_sha, "test_suite_sha256": suite_sha,
            "baseline": baseline, "expected_mutations": len(MUTATIONS),
            "killed": 0, "survived": 0, "invalid": len(MUTATIONS), "not_run": 0,
            "results": [
                {"id": mutation["id"], "family": mutation["family"], "status": "not_run",
                 "reason": "unmodified regression suite did not pass first"}
                for mutation in MUTATIONS
            ],
            "analysis_only": True, "score_eligible": False,
        }

    results: list[dict[str, Any]] = []
    for mutation in MUTATIONS:
        item: dict[str, Any] = {
            "id": mutation["id"], "family": mutation["family"],
            "description": mutation["description"],
            "expected_failure_test": mutation["expected_failure_test"],
            "anchor_sha256": digest("\n".join(row["anchor"] for row in mutation["replacements"]).encode()),
        }
        try:
            mutated_source = replace_reviewed_anchors(source, mutation)
        except ValueError as exc:
            item.update({"status": "invalid", "reason": str(exc)})
            results.append(item)
            continue

        with tempfile.TemporaryDirectory(prefix="vleb-shift-audit-mutant-") as temporary:
            root = Path(temporary)
            mutant_path = root / auditor_path.name
            suite_dest = root / suite_path.name
            mutant_path.write_text(mutated_source, encoding="utf-8")
            suite_dest.write_bytes(suite_raw)
            compiled = subprocess.run(
                [sys.executable, "-m", "py_compile", str(mutant_path)],
                cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                check=False, timeout=timeout_seconds, text=True, encoding="utf-8", errors="replace",
            )
            if compiled.returncode != 0:
                item.update({
                    "status": "invalid", "reason": "mutated auditor does not compile",
                    "compile_output_sha256": digest(compiled.stdout.encode()),
                })
                results.append(item)
                continue
            observed = run_suite(root, timeout_seconds)
        item["observed"] = {k: v for k, v in observed.items() if k != "output_tail"}
        if observed["timed_out"] or not observed["tests_started"]:
            item.update({"status": "invalid", "reason": "regression suite timed out or did not start"})
        elif mutation["expected_failure_test"] in observed["failure_test_names"]:
            item["status"] = "killed"
        elif observed["returncode"] == 0:
            item.update({"status": "survived", "reason": "designated regression test passed under mutation"})
        else:
            item.update({"status": "invalid", "reason": "suite failed without failing the designated regression test"})
            item["unexpected_failure_test_names"] = observed["failure_test_names"]
        results.append(item)

    killed = sum(row["status"] == "killed" for row in results)
    survived = sum(row["status"] == "survived" for row in results)
    invalid = sum(row["status"] == "invalid" for row in results)
    not_run = sum(row["status"] == "not_run" for row in results)
    family_summary: dict[str, dict[str, int]] = {}
    for mutation in MUTATIONS:
        family = mutation["family"]
        summary = family_summary.setdefault(family, {"expected": 0, "killed": 0, "survived": 0, "invalid": 0, "not_run": 0})
        summary["expected"] += 1
    for row in results:
        family_summary[row["family"]][row["status"]] += 1
    return {
        "schema_version": "vleb-shift-event-mutation-campaign-v1",
        "campaign_passed": baseline_passed and killed == len(MUTATIONS) and survived == 0 and invalid == 0 and not_run == 0,
        "baseline_passed": baseline_passed,
        "auditor_source_sha256": source_sha,
        "test_suite_sha256": suite_sha,
        "baseline": {k: v for k, v in baseline.items() if k != "output_tail"},
        "expected_mutations": len(MUTATIONS),
        "killed": killed, "survived": survived, "invalid": invalid, "not_run": not_run,
        "overall_kill_rate_pct": round(100.0 * killed / len(MUTATIONS), 6),
        "family_summary": family_summary,
        "results": results,
        "analysis_only": True, "score_eligible": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auditor", type=Path, default=Path(__file__).with_name("audit_shift_event_evidence.py"))
    parser.add_argument("--tests", type=Path, default=Path(__file__).with_name("test_audit_shift_event_evidence.py"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=90)
    args = parser.parse_args(argv)
    try:
        report = mutation_campaign(args.auditor, args.tests, timeout_seconds=args.timeout_seconds)
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError) as exc:
        report = {
            "schema_version": "vleb-shift-event-mutation-campaign-v1",
            "campaign_passed": False, "baseline_passed": False,
            "error": str(exc), "analysis_only": True, "score_eligible": False,
        }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report.get("campaign_passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
