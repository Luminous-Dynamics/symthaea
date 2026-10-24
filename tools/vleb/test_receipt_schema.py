#!/usr/bin/env python3
"""CI-only JSON Schema conformance tests for VLEB evidence receipts.

This checks the receipt schema's structural pass gates. It is not the independent
runtime auditor: artifact hashes, checkout identity, exact test inventory, and
task/protocol bindings still have to be recomputed by an auditor.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Callable

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "tools/vleb/evidence-receipt.schema.json"
DIGEST = "a" * 64


def valid_pass_receipt() -> dict[str, object]:
    """Small positive control exercising every required pass gate."""
    digest = lambda char: char * 64
    return {
        "schema_version": "vleb-evidence-v1",
        "run_id": "schema-test-run",
        "task_id": "vleb-ledger-reconcile-0001",
        "task_contract_sha256": digest("a"),
        "protocol_sha256": digest("b"),
        "candidate_patch_sha256": digest("c"),
        "attempt_number": 1,
        "agent": {
            "model_id": "schema-test-agent",
            "model_version": "test-1",
            "scaffold_sha256": digest("d"),
            "tool_policy": {
                "filesystem_scope": "task_checkout",
                "container_network": "none",
                "browser_allowed": False,
                "external_retrieval_allowed": False,
            },
        },
        "environment": {
            "platform": "linux",
            "toolchains": [
                {
                    "name": "rustc",
                    "version": "1.96.0",
                    "identity_sha256": digest("e"),
                }
            ],
            "runtime_identity": {
                "container_or_host_id": "schema-test-container",
                "configured_image": "vleb-fixture-image",
                "immutable_image_id": "sha256:" + digest("f"),
                "network_mode": "none",
                "network_mode_observed": True,
            },
            "checkout_identity": {
                "repository": "Luminous-Dynamics/symthaea",
                "expected_revision": "abc123",
                "observed_revision": "abc123",
                "clean_worktree": True,
                "candidate_row_sha256": digest("1"),
            },
            "sanitization": {
                "git_metadata_exit_code": 0,
                "hooks_disabled": True,
                "unexpected_refs_checked": True,
                "unreachable_objects_checked": True,
                "ignored_cache_policy": "removed",
            },
            "trust_boundary": "independently_inspected",
        },
        "commands": [
            {
                "command_id": "reference-tests",
                "argv": ["cargo", "test", "--locked"],
                "working_directory": "/workspace",
                "exit_code": 0,
                "timed_out": False,
                "stdout_sha256": digest("2"),
                "stderr_sha256": digest("3"),
                "duration_ms": 1000,
            }
        ],
        "required_test_ids": ["ledger.normal_exact_output"],
        "test_results": [
            {
                "test_id": "ledger.normal_exact_output",
                "status": "passed",
                "log_sha256": digest("4"),
                "exit_code": 0,
                "duration_ms": 10,
                "notes": "schema conformance positive control",
            }
        ],
        "mutation_results": [
            {
                "mutation_id": "VLEB-LDG-001",
                "mutation_class": "duplicate_processing",
                "compilable": True,
                "equivalence_status": "non_equivalent",
                "observed_verdict": "rejected",
                "evidence_sha256": digest("5"),
                "review_notes": "synthetic schema test only",
            }
        ],
        "differential_checks": [
            {
                "check_id": "stdout-equality",
                "observable": "stdout_bytes",
                "status": "equal",
                "expected_sha256": digest("6"),
                "observed_sha256": digest("6"),
                "mismatch_count": 0,
                "details_sha256": digest("7"),
            }
        ],
        "verifier": {
            "verifier_id": "vleb-schema-test",
            "verifier_version": "test-1",
            "verifier_binary_sha256": digest("8"),
            "receipt_recomputed": True,
            "required_test_coverage": 1,
            "environment_match": "match",
            "verdict": "pass",
            "reason_codes": [],
            "auditor_output_sha256": digest("9"),
        },
        "outcome": "pass",
        "unresolved_reasons": [],
        "abstention_reason": None,
        "timing": {
            "started_at": "2026-10-24T10:00:00Z",
            "ended_at": "2026-10-24T10:00:01Z",
            "wall_clock_ms": 1000,
        },
        "cost": {
            "model_cost_usd": 0,
            "tool_cost_usd": 0,
            "compute_cost_usd": 0,
            "currency": "USD",
            "cost_accounting_complete": True,
        },
    }


def main() -> int:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())

    positive = valid_pass_receipt()
    errors = list(validator.iter_errors(positive))
    if errors:
        print("FAIL: valid positive control was rejected:", file=sys.stderr)
        for error in errors:
            print(f"  {list(error.absolute_path)}: {error.message}", file=sys.stderr)
        return 1

    controls: list[tuple[str, Callable[[dict[str, object]], None]]] = [
        (
            "runtime self-report cannot pass",
            lambda r: r["environment"].__setitem__("trust_boundary", "runtime_reported"),
        ),
        (
            "missing stdout hash cannot pass",
            lambda r: r["commands"][0].__setitem__("stdout_sha256", None),
        ),
        (
            "skipped required test cannot pass",
            lambda r: r["test_results"][0].__setitem__("status", "skipped"),
        ),
        (
            "unreviewed mutation cannot pass",
            lambda r: r["mutation_results"][0].__setitem__("equivalence_status", "unreviewed"),
        ),
        (
            "non-compilable semantic mutant cannot pass",
            lambda r: r["mutation_results"][0].__setitem__("compilable", False),
        ),
        (
            "different differential result cannot pass",
            lambda r: r["differential_checks"][0].__setitem__("status", "different"),
        ),
        (
            "incomplete cost accounting cannot pass",
            lambda r: r["cost"].__setitem__("cost_accounting_complete", False),
        ),
        (
            "missing auditor-output digest cannot pass",
            lambda r: r["verifier"].pop("auditor_output_sha256"),
        ),
        (
            "missing immutable runtime identity cannot pass",
            lambda r: (
                r["environment"]["runtime_identity"].__setitem__("immutable_image_id", None),
                r["environment"]["runtime_identity"].__setitem__("host_attestation_uri", None),
            ),
        ),
    ]

    for name, mutate in controls:
        candidate = copy.deepcopy(positive)
        mutate(candidate)
        if not list(validator.iter_errors(candidate)):
            print(f"FAIL: invalid receipt was accepted: {name}", file=sys.stderr)
            return 1

    print(
        f"VLEB receipt schema valid; positive control accepted; "
        f"{len(controls)} fail-closed negative controls rejected."
    )
    print("These are schema tests, not evidence that any benchmark run passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
