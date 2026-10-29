#!/usr/bin/env python3
"""Mutation-test the VLEB two-sided oracle-control verifier against its own tests.

A sentinel is killed only when the unmodified suite passes, the mutated module
compiles, the suite starts, and the specific test protecting that guard fails.
Syntax/import failures, timeouts, unrelated failures and surviving sentinels are
invalid or survived rather than evidence of adequate test sensitivity.
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

RAN_TESTS = re.compile(r"Ran [1-9]\d* tests?\b")
MUTATIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "manifest_pin_bypass",
        "family": "trusted_input_pins",
        "description": "bypass evaluator-pinned oracle manifest digest",
        "expected_failure_test": "test_wrong_manifest_pin_is_rejected",
        "replacements": [
            {"anchor": "if digest(manifest_raw) != expected_manifest_sha256:", "replacement": "if False:"},
        ],
    },
    {
        "id": "reference_patch_pin_bypass",
        "family": "trusted_input_pins",
        "description": "bypass evaluator-pinned reference patch digest",
        "expected_failure_test": "test_wrong_reference_patch_pin_is_rejected",
        "replacements": [
            {"anchor": "if digest(reference_patch_raw) != expected_reference_patch_sha256:", "replacement": "if False:"},
        ],
    },
    {
        "id": "test_bundle_pin_bypass",
        "family": "trusted_input_pins",
        "description": "bypass evaluator-pinned independent test bundle digest",
        "expected_failure_test": "test_wrong_test_bundle_pin_is_rejected",
        "replacements": [
            {"anchor": "if digest(test_bundle_raw) != expected_test_bundle_sha256:", "replacement": "if False:"},
        ],
    },
    {
        "id": "verifier_source_pin_bypass",
        "family": "trusted_input_pins",
        "description": "bypass evaluator-pinned verifier source digest",
        "expected_failure_test": "test_wrong_verifier_source_pin_is_rejected",
        "replacements": [
            {"anchor": "if verifier_sha != expected_verifier_sha256:", "replacement": "if False:"},
        ],
    },
    {
        "id": "dirty_source_guard_bypass",
        "family": "source_isolation",
        "description": "allow an unclean source checkout to enter oracle replay",
        "expected_failure_test": "test_dirty_source_worktree_is_rejected",
        "replacements": [
            {"anchor": "if status:", "replacement": "if False:"},
        ],
    },
    {
        "id": "ambient_environment_inheritance",
        "family": "execution_identity",
        "description": "replace explicit command environment with inherited runner environment",
        "expected_failure_test": "test_actual_empty_patch_fails_and_reference_patch_passes",
        "replacements": [
            {
                "anchor": 'env=command_environment(command["environment"]),',
                "replacement": "env=__import__('os').environ.copy(),",
            },
        ],
    },
    {
        "id": "test_start_marker_bypass",
        "family": "outcome_classification",
        "description": "allow a nonzero test result to count without a recognized test-start marker",
        "expected_failure_test": "test_invalid_test_command_markers_do_not_count_as_failure",
        "replacements": [
            {
                "anchor": "elif not start_seen or pass_seen == fail_seen:",
                "replacement": "elif pass_seen == fail_seen:",
            },
        ],
    },
    {
        "id": "reference_pass_gate_bypass",
        "family": "two_sided_gate",
        "description": "accept a reference arm whose preregistered test command failed",
        "expected_failure_test": "test_reference_that_does_not_fix_behavior_fails_gate",
        "replacements": [
            {
                "anchor": 'reference_all_passed = (\n            len(reference_results) == len(commands)\n            and all(row["observed_test_outcome"] == "passed" for row in reference_results)\n        )',
                "replacement": "reference_all_passed = True",
            },
        ],
    },
    {
        "id": "empty_baseline_failure_gate_bypass",
        "family": "two_sided_gate",
        "description": "allow a task whose test suite also passes on the empty baseline",
        "expected_failure_test": "test_oracle_that_passes_both_empty_and_reference_arms_is_rejected",
        "replacements": [
            {
                "anchor": "controls_passed = baseline_failed and baseline_complete and reference_all_passed",
                "replacement": "controls_passed = baseline_complete and reference_all_passed",
            },
            {"anchor": "if not baseline_failed:", "replacement": "if False:"},
        ],
    },
    {
        "id": "test_bundle_byte_integrity_bypass",
        "family": "post_run_integrity",
        "description": "ignore a changed test-bundle file after execution",
        "expected_failure_test": "test_test_bundle_mutation_during_execution_invalidates_gate",
        "replacements": [
            {"anchor": 'if observed != item["sha256"]:', "replacement": "if False:"},
        ],
    },
    {
        "id": "tracked_reference_tree_integrity_bypass",
        "family": "post_run_integrity",
        "description": "report a reference checkout as intact after tracked source mutation",
        "expected_failure_test": "test_tracked_source_mutation_during_reference_tests_invalidates_gate",
        "replacements": [
            {"anchor": "reference_delta_before == reference_delta_after", "replacement": "True"},
        ],
    },
    {
        "id": "untracked_reference_inventory_bypass",
        "family": "post_run_integrity",
        "description": "report a reference checkout as intact after undeclared untracked output",
        "expected_failure_test": "test_undeclared_untracked_file_during_test_invalidates_gate",
        "replacements": [
            {
                "anchor": "reference_untracked_before == reference_untracked_after == reference_bundle_paths",
                "replacement": "True",
            },
        ],
    },
    {
        "id": "zip_directory_symlink_bypass",
        "family": "archive_safety",
        "description": "allow a ZIP directory entry to traverse a symlink parent",
        "expected_failure_test": "test_zip_directory_entry_cannot_follow_symlink_parent",
        "replacements": [
            {
                "anchor": 'if cursor.is_symlink():\n                            raise ControlError(f"symlink directory in test bundle path rejected: {name!r}")',
                "replacement": 'if False:\n                            raise ControlError(f"symlink directory in test bundle path rejected: {name!r}")',
            },
        ],
    },
    {
        "id": "source_overwrite_guard_bypass",
        "family": "archive_safety",
        "description": "allow the independent test ZIP to overwrite a source file",
        "expected_failure_test": "test_zip_rejects_source_overwrite",
        "replacements": [
            {
                "anchor": "if destination.exists() or destination.is_symlink():",
                "replacement": "if False:",
            },
        ],
    },
    {
        "id": "secret_environment_name_guard_bypass",
        "family": "execution_identity",
        "description": "allow secret-bearing environment variable names into command configuration",
        "expected_failure_test": "test_manifest_rejects_secret_environment_variables",
        "replacements": [
            {"anchor": "if SECRET_ENV_NAME.search(key):", "replacement": "if False:"},
        ],
    },
)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def replace_reviewed_anchors(source: str, mutation: dict[str, Any]) -> str:
    mutated = source
    replacements = mutation.get("replacements")
    if not isinstance(replacements, list) or not replacements:
        raise ValueError("mutation must contain at least one reviewed replacement")
    for replacement in replacements:
        anchor = replacement.get("anchor")
        new_value = replacement.get("replacement")
        if not isinstance(anchor, str) or not anchor or not isinstance(new_value, str):
            raise ValueError("mutation replacement requires non-empty anchor and string replacement")
        count = mutated.count(anchor)
        if count != 1:
            raise ValueError(f"mutation {mutation.get('id')} anchor count is {count}, expected exactly one")
        mutated = mutated.replace(anchor, new_value, 1)
    return mutated


def run_suite(root: Path, *, timeout_seconds: int) -> dict[str, Any]:
    suite = root / "test_verify_oracle_controls.py"
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
            "failure_test_names": [],
            "output_sha256": digest(str(output).encode("utf-8")),
            "output_tail": str(output)[-3000:],
        }
    output = completed.stdout
    failure_names = sorted(set(re.findall(r"^(?:FAIL|ERROR): (test_[A-Za-z0-9_]+) \\(", output, re.MULTILINE)))
    return {
        "returncode": completed.returncode,
        "timed_out": False,
        "tests_started": RAN_TESTS.search(output) is not None,
        "failure_test_names": failure_names,
        "output_sha256": digest(output.encode("utf-8")),
        "output_tail": output[-4000:] if completed.returncode != 0 else None,
    }


def run_mutation_campaign(
    verifier_path: Path,
    suite_path: Path,
    *,
    timeout_seconds: int = 180,
) -> dict[str, Any]:
    original_raw = verifier_path.read_bytes()
    suite_raw = suite_path.read_bytes()
    original = original_raw.decode("utf-8")
    baseline_source_sha = digest(original_raw)
    suite_sha = digest(suite_raw)

    with tempfile.TemporaryDirectory(prefix="vleb-oracle-mutation-baseline-") as temporary:
        baseline_root = Path(temporary)
        (baseline_root / verifier_path.name).write_bytes(original_raw)
        (baseline_root / suite_path.name).write_bytes(suite_raw)
        baseline = run_suite(baseline_root, timeout_seconds=timeout_seconds)

    baseline_passed = baseline["returncode"] == 0 and baseline["tests_started"] and not baseline["timed_out"]
    if not baseline_passed:
        return {
            "schema_version": "vleb-oracle-control-mutation-campaign-v1",
            "baseline_passed": False,
            "baseline": baseline,
            "verifier_source_sha256": baseline_source_sha,
            "test_suite_sha256": suite_sha,
            "mutation_count": len(MUTATIONS),
            "killed_count": 0,
            "survived_count": 0,
            "invalid_count": 0,
            "mutation_kill_rate_pct": None,
            "mutations": [],
            "analysis_only": True,
            "score_eligible": False,
            "issues": ["unmodified oracle-control adversarial suite must pass before mutations are meaningful"],
        }

    results: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="vleb-oracle-mutation-campaign-") as temporary:
        parent = Path(temporary)
        for mutation in MUTATIONS:
            result: dict[str, Any] = {
                "id": mutation["id"],
                "family": mutation["family"],
                "description": mutation["description"],
                "expected_failure_test": mutation["expected_failure_test"],
                "status": "invalid",
                "anchor_counts": [],
            }
            try:
                mutated_text = replace_reviewed_anchors(original, mutation)
                for replacement in mutation["replacements"]:
                    anchor = replacement["anchor"]
                    result["anchor_counts"].append({
                        "anchor_sha256": digest(anchor.encode("utf-8")),
                        "count": original.count(anchor),
                    })
            except (KeyError, TypeError, ValueError) as exc:
                result["reason"] = str(exc)
                results.append(result)
                continue

            mutant_root = parent / mutation["id"]
            mutant_root.mkdir(parents=True, exist_ok=False)
            mutant_verifier = mutant_root / verifier_path.name
            mutant_suite = mutant_root / suite_path.name
            mutant_raw = mutated_text.encode("utf-8")
            mutant_verifier.write_bytes(mutant_raw)
            mutant_suite.write_bytes(suite_raw)
            result["mutated_verifier_sha256"] = digest(mutant_raw)

            compile_path = mutant_root / "mutant.pyc"
            compiled = subprocess.run(
                [sys.executable, "-m", "py_compile", "-q", str(mutant_verifier)],
                cwd=mutant_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=30,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            result["compile_returncode"] = compiled.returncode
            result["compile_output_sha256"] = digest(compiled.stdout.encode("utf-8"))
            if compiled.returncode != 0:
                result["reason"] = "mutated verifier did not compile"
                result["compile_output_tail"] = compiled.stdout[-2000:]
                results.append(result)
                continue

            suite = run_suite(mutant_root, timeout_seconds=timeout_seconds)
            result["suite"] = suite
            targeted_failure_seen = mutation["expected_failure_test"] in suite.get("failure_test_names", [])
            if suite["timed_out"] or not suite["tests_started"]:
                result["status"] = "invalid"
                result["reason"] = "mutated suite timed out or did not demonstrably start"
            elif suite["returncode"] == 0:
                result["status"] = "survived"
                result["reason"] = "mutated implementation passed the full adversarial test suite"
            elif not targeted_failure_seen:
                result["status"] = "invalid"
                result["reason"] = "suite failed, but the test specifically assigned to this guard did not fail"
            else:
                result["status"] = "killed"
                result["reason"] = "the designated guard test failed against a compiling mutation"

            results.append(result)

    families: dict[str, Any] = {}
    for family in sorted({row["family"] for row in MUTATIONS}):
        expected = [row for row in MUTATIONS if row["family"] == family]
        observed = [row for row in results if row.get("family") == family]
        killed = sum(row.get("status") == "killed" for row in observed)
        survived = sum(row.get("status") == "survived" for row in observed)
        invalid = sum(row.get("status") == "invalid" for row in observed)
        families[family] = {
            "expected_count": len(expected),
            "executed_count": len(observed),
            "killed_count": killed,
            "survived_count": survived,
            "invalid_count": invalid,
            "not_run_count": max(0, len(expected) - len(observed)),
            "kill_rate_pct": round(100.0 * killed / len(expected), 6) if len(observed) == len(expected) else None,
            "complete": len(observed) == len(expected) and killed == len(expected) and survived == 0 and invalid == 0,
        }

    killed = sum(row.get("status") == "killed" for row in results)
    survived = sum(row.get("status") == "survived" for row in results)
    invalid = sum(row.get("status") == "invalid" for row in results)
    return {
        "schema_version": "vleb-oracle-control-mutation-campaign-v1",
        "baseline_passed": True,
        "baseline": baseline,
        "verifier_source_sha256": baseline_source_sha,
        "test_suite_sha256": suite_sha,
        "mutation_count": len(MUTATIONS),
        "killed_count": killed,
        "survived_count": survived,
        "invalid_count": invalid,
        "mutation_kill_rate_pct": round(100.0 * killed / len(MUTATIONS), 6),
        "family_summary": families,
        "mutations": results,
        "analysis_only": True,
        "score_eligible": False,
        "complete": killed == len(MUTATIONS) and survived == 0 and invalid == 0 and all(row["complete"] for row in families.values()),
        "issues": [
            {"code": row["status"] + "_mutation", "mutation_id": row["id"], "reason": row.get("reason", "")}
            for row in results if row.get("status") != "killed"
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier", type=Path, default=Path(__file__).with_name("verify_oracle_controls.py"))
    parser.add_argument("--test-suite", type=Path, default=Path(__file__).with_name("test_verify_oracle_controls.py"))
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not 1 <= args.timeout_seconds <= 900:
        parser.error("--timeout-seconds must be in [1, 900]")

    try:
        result = run_mutation_campaign(args.verifier, args.test_suite, timeout_seconds=args.timeout_seconds)
    except (OSError, UnicodeError, ValueError, subprocess.SubprocessError) as exc:
        result = {
            "schema_version": "vleb-oracle-control-mutation-campaign-v1",
            "baseline_passed": False,
            "analysis_only": True,
            "score_eligible": False,
            "complete": False,
            "issues": [{"code": "mutation_campaign_rejected", "reason": str(exc)}],
        }

    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result.get("complete") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
