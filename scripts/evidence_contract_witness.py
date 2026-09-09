#!/usr/bin/env python3
"""Strict data-only verifier for focused evidence-contract candidate receipts.

This is qualification/witness infrastructure, not scientific evidence. It never
executes candidate code and never treats candidate-controlled receipt fields as
trusted provider or recipe identity.

The trusted phase must independently supply exact subject/run/artifact identity,
candidate workflow + qualifier hashes fetched as data, the expected authority
target set, and either an exact predecessor recipe/root snapshot or explicit
bootstrap mode. Historical witness identity is separate from current admission.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA = "symthaea-evidence-contract-qualification-v1"
WITNESS_SCHEMA = "symthaea.evidence-contract-witness.v1"
RECIPE_SCHEMA = "symthaea.evidence-contract-recipe.v1"
ROOT_SCHEMA = "symthaea.evidence-contract-trust-root-snapshot.v1"

RECIPE_DOMAIN = b"symthaea.evidence-contract-recipe.v1\0"
ROOT_DOMAIN = b"symthaea.evidence-contract-trust-root-snapshot.v1\0"
WITNESS_DOMAIN = b"symthaea.evidence-contract-witness.v1\0"

_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SHA256_ID_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_DECIMAL_RE = re.compile(r"^(0|[1-9][0-9]*)$")
_TARGET_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
_KEY_RE = re.compile(r"^[a-z0-9_]+$")

RECEIPT_KEYS = (
    "schema", "status", "exit_code", "terminal_stage", "scope",
    "scientific_authority", "full_repository_ci", "baseline_generation",
    "environment_authority", "receipt_attestation", "provider_metadata_basis",
    "qualified_sha", "expected_sha", "committed_tree", "source_state",
    "execution_provider", "runner_label", "runner_os", "runner_arch",
    "runner_image_os", "runner_image_version", "os_release_sha256",
    "kernel_release", "rustc_release", "rustc_commit_hash", "rustc_host",
    "cargo_version", "checkout_action_sha", "upload_artifact_action_sha",
    "cargo_lock_sha256", "workspace_manifest_sha256", "rust_toolchain_sha256",
    "qualifier_script_sha256", "workflow_sha256", "authority_integration_targets",
    "github_event_name", "github_repository", "github_workflow_ref", "github_job",
    "github_run_id", "github_run_attempt",
)
RECEIPT_KEY_SET = frozenset(RECEIPT_KEYS)


class WitnessError(ValueError):
    """Fail-closed receipt/witness validation error."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _domain_id(domain: bytes, value: Any) -> str:
    return "sha256:" + hashlib.sha256(domain + _canonical_json(value)).hexdigest()


def _require_scalar(value: str, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise WitnessError(f"{where}: non-empty string required")
    if value != value.strip():
        raise WitnessError(f"{where}: surrounding whitespace is non-canonical")
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise WitnessError(f"{where}: ASCII control characters are forbidden")
    if any(ch in "\u0085\u2028\u2029" for ch in value):
        raise WitnessError(f"{where}: Unicode line separators are forbidden")
    return value


def _require_sha40(value: str, *, where: str) -> str:
    if _SHA40_RE.fullmatch(value) is None:
        raise WitnessError(f"{where}: expected 40 lowercase hex")
    return value


def _require_sha256(value: str, *, where: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise WitnessError(f"{where}: expected 64 lowercase hex")
    return value


def _require_sha256_id(value: str, *, where: str) -> str:
    if _SHA256_ID_RE.fullmatch(value) is None:
        raise WitnessError(f"{where}: expected sha256:<64 lowercase hex>")
    return value


def _require_decimal(value: str, *, where: str) -> int:
    if _DECIMAL_RE.fullmatch(value) is None:
        raise WitnessError(f"{where}: expected canonical non-negative decimal")
    return int(value)


def _parse_targets(value: str, *, where: str) -> tuple[str, ...]:
    if value == "none":
        return ()
    parts = value.split(",")
    if any(not part or _TARGET_RE.fullmatch(part) is None for part in parts):
        raise WitnessError(f"{where}: invalid target list")
    if parts != sorted(set(parts)):
        raise WitnessError(f"{where}: targets must be sorted and unique")
    return tuple(parts)


def parse_receipt_bytes(data: bytes) -> dict[str, str]:
    """Parse hostile V1 TSV bytes with exact framing, order, and closed schema."""
    if not data:
        raise WitnessError("receipt: empty")
    if b"\x00" in data or b"\r" in data:
        raise WitnessError("receipt: NUL/CR bytes are forbidden")
    if not data.endswith(b"\n"):
        raise WitnessError("receipt: canonical V1 must end with LF")

    raw_lines = data[:-1].split(b"\n")
    parsed: dict[str, str] = {}
    observed_order: list[str] = []
    for line_no, raw_line in enumerate(raw_lines, start=1):
        if not raw_line:
            raise WitnessError(f"receipt line {line_no}: blank line forbidden")
        if raw_line.count(b"\t") != 1:
            raise WitnessError(f"receipt line {line_no}: expected exactly one TSV separator")
        raw_key, raw_value = raw_line.split(b"\t", 1)
        try:
            key = raw_key.decode("ascii", errors="strict")
            value = raw_value.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise WitnessError(f"receipt line {line_no}: invalid encoding: {exc}") from exc
        if _KEY_RE.fullmatch(key) is None:
            raise WitnessError(f"receipt line {line_no}: invalid key {key!r}")
        if key in parsed:
            raise WitnessError(f"receipt: duplicate key {key!r}")
        parsed[key] = _require_scalar(value, where=f"receipt.{key}")
        observed_order.append(key)

    keys = set(parsed)
    missing = sorted(RECEIPT_KEY_SET - keys)
    unknown = sorted(keys - RECEIPT_KEY_SET)
    if missing:
        raise WitnessError(f"receipt: missing keys: {', '.join(missing)}")
    if unknown:
        raise WitnessError(f"receipt: unknown keys: {', '.join(unknown)}")
    if tuple(observed_order) != RECEIPT_KEYS:
        raise WitnessError("receipt: V1 fields are not in canonical order")

    _validate_receipt_semantics(parsed)
    return parsed


def _validate_receipt_semantics(receipt: dict[str, str]) -> None:
    fixed = {
        "schema": SCHEMA,
        "scope": "software-contract-only",
        "scientific_authority": "none",
        "full_repository_ci": "independent",
        "baseline_generation": "forbidden",
        "environment_authority": "observed-not-capsule-qualified",
        "receipt_attestation": "none",
        "provider_metadata_basis": "ambient-runtime",
    }
    for key, expected in fixed.items():
        if receipt[key] != expected:
            raise WitnessError(f"receipt.{key}: expected {expected!r}")

    if receipt["status"] not in {"PASS", "FAIL"}:
        raise WitnessError("receipt.status: expected PASS or FAIL")
    exit_code = _require_decimal(receipt["exit_code"], where="receipt.exit_code")

    _require_sha40(receipt["qualified_sha"], where="receipt.qualified_sha")
    _require_sha40(receipt["expected_sha"], where="receipt.expected_sha")
    _require_sha40(receipt["committed_tree"], where="receipt.committed_tree")
    if receipt["qualified_sha"] != receipt["expected_sha"]:
        raise WitnessError("receipt: qualified_sha != expected_sha")

    for key in (
        "os_release_sha256", "cargo_lock_sha256", "workspace_manifest_sha256",
        "rust_toolchain_sha256", "qualifier_script_sha256", "workflow_sha256",
    ):
        _require_sha256(receipt[key], where=f"receipt.{key}")
    for key in ("checkout_action_sha", "upload_artifact_action_sha", "rustc_commit_hash"):
        _require_sha40(receipt[key], where=f"receipt.{key}")

    _parse_targets(receipt["authority_integration_targets"], where="receipt.authority_integration_targets")
    if receipt["execution_provider"] != "github-actions":
        raise WitnessError("receipt.execution_provider: witness V1 requires github-actions")
    if receipt["github_event_name"] not in {"pull_request", "workflow_dispatch"}:
        raise WitnessError("receipt.github_event_name: unsupported GitHub event")
    _require_decimal(receipt["github_run_id"], where="receipt.github_run_id")
    if _require_decimal(receipt["github_run_attempt"], where="receipt.github_run_attempt") < 1:
        raise WitnessError("receipt.github_run_attempt: must be >= 1")
    for key in (
        "runner_label", "runner_os", "runner_arch", "runner_image_os",
        "runner_image_version", "github_repository", "github_workflow_ref", "github_job",
    ):
        if receipt[key] in {"unknown", "unavailable", "not-applicable"}:
            raise WitnessError(f"receipt.{key}: GitHub metadata unavailable")

    if receipt["status"] == "PASS":
        if exit_code != 0:
            raise WitnessError("receipt: PASS requires exit_code=0")
        if receipt["terminal_stage"] != "none":
            raise WitnessError("receipt: PASS requires terminal_stage=none")
        if receipt["source_state"] != "clean-exact-checkout-postflight":
            raise WitnessError("receipt: PASS requires clean-exact-checkout-postflight")
    else:
        if exit_code == 0:
            raise WitnessError("receipt: FAIL requires nonzero exit_code")
        if receipt["terminal_stage"] == "none":
            raise WitnessError("receipt: FAIL requires terminal failure stage")


def compute_recipe_id(qualifier_script_sha256: str, workflow_sha256: str) -> str:
    qualifier = _require_sha256(qualifier_script_sha256, where="recipe.qualifier_script_sha256")
    workflow = _require_sha256(workflow_sha256, where="recipe.workflow_sha256")
    return _domain_id(RECIPE_DOMAIN, {
        "schema": RECIPE_SCHEMA,
        "qualifier_script_sha256": f"sha256:{qualifier}",
        "workflow_sha256": f"sha256:{workflow}",
    })


def compute_trust_root_snapshot_id(
    repository_id: int,
    predecessor_sha: str,
    predecessor_tree: str,
    trusted_recipe_id: str,
) -> str:
    if repository_id < 1:
        raise WitnessError("trust_root.repository_id: positive integer required")
    _require_sha40(predecessor_sha, where="trust_root.predecessor_sha")
    _require_sha40(predecessor_tree, where="trust_root.predecessor_tree")
    _require_sha256_id(trusted_recipe_id, where="trust_root.trusted_recipe_id")
    return _domain_id(ROOT_DOMAIN, {
        "schema": ROOT_SCHEMA,
        "repository_id": repository_id,
        "predecessor_sha": predecessor_sha,
        "predecessor_tree": predecessor_tree,
        "trusted_recipe_id": trusted_recipe_id,
    })


def evaluate_witness(
    *,
    receipt_bytes: bytes,
    expected_subject_sha: str,
    expected_subject_tree: str,
    expected_run_id: int,
    expected_run_attempt: int,
    expected_run_conclusion: str,
    expected_workflow_id: int,
    expected_artifact_id: int,
    expected_artifact_name: str,
    expected_artifact_archive_sha256: str,
    expected_repository: str,
    expected_repository_id: int,
    expected_candidate_qualifier_sha256: str,
    expected_candidate_workflow_sha256: str,
    expected_authority_integration_targets: str,
    trusted_predecessor_sha: str | None = None,
    trusted_predecessor_tree: str | None = None,
    trusted_qualifier_sha256: str | None = None,
    trusted_workflow_sha256: str | None = None,
    bootstrap_no_predecessor: bool = False,
) -> dict[str, Any]:
    receipt = parse_receipt_bytes(receipt_bytes)

    expected_subject_sha = _require_sha40(expected_subject_sha, where="expected_subject_sha")
    expected_subject_tree = _require_sha40(expected_subject_tree, where="expected_subject_tree")
    if min(expected_run_id, expected_run_attempt, expected_workflow_id, expected_artifact_id, expected_repository_id) < 1:
        raise WitnessError("run/attempt/workflow/artifact/repository IDs must be positive integers")
    expected_repository = _require_scalar(expected_repository, where="expected_repository")
    expected_artifact_name = _require_scalar(expected_artifact_name, where="expected_artifact_name")
    canonical_artifact_name = f"evidence-contract-qualification-{expected_run_id}-{expected_run_attempt}"
    if expected_artifact_name != canonical_artifact_name:
        raise WitnessError("expected_artifact_name: does not bind exact run/attempt")
    expected_artifact_archive_sha256 = _require_sha256(
        expected_artifact_archive_sha256, where="expected_artifact_archive_sha256"
    )
    if expected_run_conclusion not in {
        "success", "failure", "cancelled", "timed_out", "action_required", "neutral", "skipped",
    }:
        raise WitnessError("expected_run_conclusion: unsupported provider conclusion")

    expected_targets = _parse_targets(expected_authority_integration_targets, where="expected_authority_integration_targets")
    receipt_targets = _parse_targets(receipt["authority_integration_targets"], where="receipt.authority_integration_targets")
    if receipt["qualified_sha"] != expected_subject_sha:
        raise WitnessError("witness: receipt subject SHA != trusted provider subject SHA")
    if receipt["committed_tree"] != expected_subject_tree:
        raise WitnessError("witness: receipt tree != independently expected subject tree")
    if int(receipt["github_run_id"]) != expected_run_id:
        raise WitnessError("witness: receipt run ID != triggering run")
    if int(receipt["github_run_attempt"]) != expected_run_attempt:
        raise WitnessError("witness: receipt run attempt != triggering run attempt")
    if receipt["github_repository"] != expected_repository:
        raise WitnessError("witness: receipt repository != expected repository")
    if receipt_targets != expected_targets:
        raise WitnessError("witness: receipt authority target list != independently expected list")

    candidate_qualifier = _require_sha256(expected_candidate_qualifier_sha256, where="expected_candidate_qualifier_sha256")
    candidate_workflow = _require_sha256(expected_candidate_workflow_sha256, where="expected_candidate_workflow_sha256")
    if receipt["qualifier_script_sha256"] != candidate_qualifier:
        raise WitnessError("witness: self-reported qualifier hash != independently observed candidate bytes")
    if receipt["workflow_sha256"] != candidate_workflow:
        raise WitnessError("witness: self-reported workflow hash != independently observed candidate bytes")

    if receipt["status"] == "PASS" and expected_run_conclusion != "success":
        raise WitnessError("witness: candidate PASS conflicts with provider run conclusion")
    if receipt["status"] == "FAIL" and expected_run_conclusion == "success":
        raise WitnessError("witness: candidate FAIL conflicts with provider success conclusion")

    receipt_id = "sha256:" + hashlib.sha256(receipt_bytes).hexdigest()
    candidate_recipe_id = compute_recipe_id(candidate_qualifier, candidate_workflow)

    trusted_recipe_id: str | None = None
    trusted_root_snapshot_id: str | None = None
    trusted_values = (
        trusted_predecessor_sha, trusted_predecessor_tree,
        trusted_qualifier_sha256, trusted_workflow_sha256,
    )
    if bootstrap_no_predecessor:
        if any(value is not None for value in trusted_values):
            raise WitnessError("bootstrap_no_predecessor cannot include trusted predecessor inputs")
        recipe_relation = "BootstrapNoPredecessor"
    else:
        if any(value is None for value in trusted_values):
            raise WitnessError("all trusted predecessor recipe/root inputs are required")
        assert trusted_predecessor_sha is not None
        assert trusted_predecessor_tree is not None
        assert trusted_qualifier_sha256 is not None
        assert trusted_workflow_sha256 is not None
        trusted_recipe_id = compute_recipe_id(trusted_qualifier_sha256, trusted_workflow_sha256)
        trusted_root_snapshot_id = compute_trust_root_snapshot_id(
            expected_repository_id, trusted_predecessor_sha, trusted_predecessor_tree, trusted_recipe_id
        )
        recipe_relation = "BaseIdentical" if candidate_recipe_id == trusted_recipe_id else "RecipeChanged"

    if receipt["status"] == "FAIL":
        disposition = "CandidateFailed" if expected_run_conclusion == "failure" else "ProviderNonSuccess"
    elif recipe_relation == "BootstrapNoPredecessor":
        disposition = "BootstrapNoPredecessor"
    elif recipe_relation == "RecipeChanged":
        disposition = "RecipeChangedConformanceOnly"
    else:
        disposition = "FocusedSoftwareContractWitnessed"

    witness: dict[str, Any] = {
        "schema": WITNESS_SCHEMA,
        "repository": expected_repository,
        "repository_id": expected_repository_id,
        "subject_sha": expected_subject_sha,
        "subject_tree": expected_subject_tree,
        "candidate_run_id": expected_run_id,
        "candidate_run_attempt": expected_run_attempt,
        "provider_run_conclusion": expected_run_conclusion,
        "provider_workflow_id": expected_workflow_id,
        "provider_artifact_id": expected_artifact_id,
        "provider_artifact_name": expected_artifact_name,
        "provider_artifact_archive_sha256": f"sha256:{expected_artifact_archive_sha256}",
        "candidate_receipt_id": receipt_id,
        "candidate_recipe_id": candidate_recipe_id,
        "trusted_recipe_id": trusted_recipe_id,
        "trusted_root_snapshot_id": trusted_root_snapshot_id,
        "recipe_relation": recipe_relation,
        "candidate_status": receipt["status"],
        "candidate_terminal_stage": receipt["terminal_stage"],
        "authority_integration_targets": list(expected_targets),
        "witness_disposition": disposition,
        "authority_scope": "software-contract-only",
        "scientific_authority": "none",
        "current_admission": "not-evaluated",
        "receipt_attestation": "none",
    }
    witness["witness_id"] = _domain_id(WITNESS_DOMAIN, witness)
    return witness


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--expected-subject-sha", required=True)
    parser.add_argument("--expected-subject-tree", required=True)
    parser.add_argument("--expected-run-id", required=True, type=int)
    parser.add_argument("--expected-run-attempt", required=True, type=int)
    parser.add_argument("--expected-run-conclusion", required=True)
    parser.add_argument("--expected-workflow-id", required=True, type=int)
    parser.add_argument("--expected-artifact-id", required=True, type=int)
    parser.add_argument("--expected-artifact-name", required=True)
    parser.add_argument("--expected-artifact-archive-sha256", required=True)
    parser.add_argument("--expected-repository", default="Luminous-Dynamics/symthaea")
    parser.add_argument("--expected-repository-id", required=True, type=int)
    parser.add_argument("--expected-candidate-qualifier-sha256", required=True)
    parser.add_argument("--expected-candidate-workflow-sha256", required=True)
    parser.add_argument("--expected-authority-integration-targets", default="none")
    parser.add_argument("--trusted-predecessor-sha")
    parser.add_argument("--trusted-predecessor-tree")
    parser.add_argument("--trusted-qualifier-sha256")
    parser.add_argument("--trusted-workflow-sha256")
    parser.add_argument("--bootstrap-no-predecessor", action="store_true")
    parser.add_argument("--conformance", action="store_true", help="exit zero for valid non-authorizing dispositions")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = evaluate_witness(
            receipt_bytes=args.receipt.read_bytes(),
            expected_subject_sha=args.expected_subject_sha,
            expected_subject_tree=args.expected_subject_tree,
            expected_run_id=args.expected_run_id,
            expected_run_attempt=args.expected_run_attempt,
            expected_run_conclusion=args.expected_run_conclusion,
            expected_workflow_id=args.expected_workflow_id,
            expected_artifact_id=args.expected_artifact_id,
            expected_artifact_name=args.expected_artifact_name,
            expected_artifact_archive_sha256=args.expected_artifact_archive_sha256,
            expected_repository=args.expected_repository,
            expected_repository_id=args.expected_repository_id,
            expected_candidate_qualifier_sha256=args.expected_candidate_qualifier_sha256,
            expected_candidate_workflow_sha256=args.expected_candidate_workflow_sha256,
            expected_authority_integration_targets=args.expected_authority_integration_targets,
            trusted_predecessor_sha=args.trusted_predecessor_sha,
            trusted_predecessor_tree=args.trusted_predecessor_tree,
            trusted_qualifier_sha256=args.trusted_qualifier_sha256,
            trusted_workflow_sha256=args.trusted_workflow_sha256,
            bootstrap_no_predecessor=args.bootstrap_no_predecessor,
        )
    except (OSError, WitnessError) as exc:
        print(f"evidence-contract witness invalid: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    if result["witness_disposition"] == "FocusedSoftwareContractWitnessed":
        return 0
    return 0 if args.conformance else 3


if __name__ == "__main__":
    raise SystemExit(main())
