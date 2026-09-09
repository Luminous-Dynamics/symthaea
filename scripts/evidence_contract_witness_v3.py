#!/usr/bin/env python3
"""V3 base-owned witness requiring TrustedMainRootReceiptV2.

V1 remains the low-level hostile candidate/provider/recipe verifier. V2 proved
that an independently expected root-receipt ID must be supplied. V3 tightens
the accepted root format: ordinary post-bootstrap positive witnessing requires
the independently-selected TrustedMainRootReceiptV2 produced by #1240.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import evidence_contract_witness as v1

WITNESS_SCHEMA = "symthaea.evidence-contract-witness.v3"
WITNESS_DOMAIN = b"symthaea.evidence-contract-witness.v3\0"
ROOT_RECEIPT_SCHEMA = "symthaea.github-trusted-main-root-receipt.v2"
ROOT_RECEIPT_DOMAIN = b"symthaea.github-trusted-main-root-receipt.v2\0"
SELECTION_SCHEMA = "symthaea.github-trusted-main-evidence-selection.v1"
SELECTION_DOMAIN = b"symthaea.github-trusted-main-evidence-selection.v1\0"
MAX_JSON_BYTES = 1_000_000

ROOT_RECEIPT_KEYS = frozenset({
    "schema", "repository", "repository_id", "target_ref", "root_sha",
    "root_tree", "root_subject_id", "protection_policy_id",
    "protection_verification_id", "effective_rules_verification_id",
    "enforcement_evidence_id", "trusted_evidence_selection_id", "ruleset_id",
    "bypass_policy_id", "review_assurance", "root_harness_identity",
    "v1_root_receipt_id", "admission_disposition", "evidence_selection_basis",
    "historical_scope", "current_admission", "receipt_attestation",
    "self_hosted_runner_activation", "scientific_authority",
    "bootstrap_authority", "root_receipt_id",
})


class WitnessV3Error(ValueError):
    """Fail-closed V3 admitted-root selection/binding error."""


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise WitnessV3Error(f"duplicate JSON object key: {key!r}")
        out[key] = value
    return out


def _load_json(path: Path) -> Any:
    try:
        if path.stat().st_size > MAX_JSON_BYTES:
            raise WitnessV3Error(f"{path}: exceeds {MAX_JSON_BYTES} bytes")
        return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicates)
    except WitnessV3Error:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WitnessV3Error(f"{path}: {exc}") from exc


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _domain_id(domain: bytes, value: Any) -> str:
    return "sha256:" + hashlib.sha256(domain + _canonical(value)).hexdigest()


def normalize_root_receipt_v2(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise WitnessV3Error("trusted root receipt: object required")
    missing = sorted(ROOT_RECEIPT_KEYS - set(raw))
    unknown = sorted(set(raw) - ROOT_RECEIPT_KEYS)
    if missing:
        raise WitnessV3Error(f"trusted root receipt: missing fields: {', '.join(missing)}")
    if unknown:
        raise WitnessV3Error(f"trusted root receipt: unknown fields: {', '.join(unknown)}")

    out = dict(raw)
    if out["schema"] != ROOT_RECEIPT_SCHEMA:
        raise WitnessV3Error(f"trusted root receipt.schema: expected {ROOT_RECEIPT_SCHEMA!r}")
    for field in (
        "root_subject_id", "protection_policy_id", "protection_verification_id",
        "effective_rules_verification_id", "enforcement_evidence_id",
        "trusted_evidence_selection_id", "bypass_policy_id", "v1_root_receipt_id",
        "root_receipt_id",
    ):
        try:
            v1._require_sha256_id(out[field], where=f"trusted_root_receipt.{field}")
        except v1.WitnessError as exc:
            raise WitnessV3Error(str(exc)) from exc
    try:
        v1._require_sha40(out["root_sha"], where="trusted_root_receipt.root_sha")
        v1._require_sha40(out["root_tree"], where="trusted_root_receipt.root_tree")
    except v1.WitnessError as exc:
        raise WitnessV3Error(str(exc)) from exc
    if isinstance(out["repository_id"], bool) or not isinstance(out["repository_id"], int) or out["repository_id"] < 1:
        raise WitnessV3Error("trusted root receipt.repository_id: positive integer required")
    if isinstance(out["ruleset_id"], bool) or not isinstance(out["ruleset_id"], int) or out["ruleset_id"] < 1:
        raise WitnessV3Error("trusted root receipt.ruleset_id: positive integer required")
    for field in ("repository", "target_ref", "review_assurance", "root_harness_identity"):
        try:
            v1._require_scalar(out[field], where=f"trusted_root_receipt.{field}")
        except v1.WitnessError as exc:
            raise WitnessV3Error(str(exc)) from exc

    fixed = {
        "admission_disposition": "AdmittedHistoricalRoot",
        "evidence_selection_basis": "trusted-phase-independent-expected-ids",
        "historical_scope": "exact-root-only",
        "current_admission": "not-evaluated",
        "receipt_attestation": "none",
        "self_hosted_runner_activation": "not-authorized",
        "scientific_authority": "none",
        "bootstrap_authority": "none",
    }
    for field, expected in fixed.items():
        if out[field] != expected:
            raise WitnessV3Error(f"trusted root receipt.{field}: expected {expected!r}")
    if out["review_assurance"] not in {
        "pr-mediated-no-required-approval", "pr-mediated-required-approval"
    }:
        raise WitnessV3Error("trusted root receipt.review_assurance: unsupported value")
    if out["root_harness_identity"] == "none-pre-bootstrap":
        raise WitnessV3Error("trusted root receipt: ordinary witness requires post-bootstrap harness identity")
    try:
        v1._require_sha256_id(out["root_harness_identity"], where="trusted_root_receipt.root_harness_identity")
    except v1.WitnessError as exc:
        raise WitnessV3Error(str(exc)) from exc

    expected_selection_id = _domain_id(SELECTION_DOMAIN, {
        "schema": SELECTION_SCHEMA,
        "protection_policy_id": out["protection_policy_id"],
        "protection_verification_id": out["protection_verification_id"],
        "effective_rules_verification_id": out["effective_rules_verification_id"],
        "root_subject_id": out["root_subject_id"],
        "enforcement_evidence_id": out["enforcement_evidence_id"],
    })
    if out["trusted_evidence_selection_id"] != expected_selection_id:
        raise WitnessV3Error("trusted root receipt.trusted_evidence_selection_id: content ID mismatch")

    observed_id = out["root_receipt_id"]
    payload = dict(out)
    payload.pop("root_receipt_id")
    expected_id = _domain_id(ROOT_RECEIPT_DOMAIN, payload)
    if observed_id != expected_id:
        raise WitnessV3Error("trusted root receipt.root_receipt_id: content ID mismatch")
    return out


def evaluate_witness_v3(
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
    trusted_root_receipt_v2: Any | None = None,
    expected_trusted_root_receipt_id: str | None = None,
    trusted_predecessor_sha: str | None = None,
    trusted_predecessor_tree: str | None = None,
    trusted_qualifier_sha256: str | None = None,
    trusted_workflow_sha256: str | None = None,
    bootstrap_no_predecessor: bool = False,
) -> dict[str, Any]:
    if bootstrap_no_predecessor:
        if trusted_root_receipt_v2 is not None or expected_trusted_root_receipt_id is not None:
            raise WitnessV3Error("bootstrap cannot include a trusted root receipt or expected root receipt ID")
        if any(value is not None for value in (
            trusted_predecessor_sha, trusted_predecessor_tree,
            trusted_qualifier_sha256, trusted_workflow_sha256,
        )):
            raise WitnessV3Error("bootstrap cannot include predecessor root/recipe inputs")
        base = v1.evaluate_witness(
            receipt_bytes=receipt_bytes,
            expected_subject_sha=expected_subject_sha,
            expected_subject_tree=expected_subject_tree,
            expected_run_id=expected_run_id,
            expected_run_attempt=expected_run_attempt,
            expected_run_conclusion=expected_run_conclusion,
            expected_workflow_id=expected_workflow_id,
            expected_artifact_id=expected_artifact_id,
            expected_artifact_name=expected_artifact_name,
            expected_artifact_archive_sha256=expected_artifact_archive_sha256,
            expected_repository=expected_repository,
            expected_repository_id=expected_repository_id,
            expected_candidate_qualifier_sha256=expected_candidate_qualifier_sha256,
            expected_candidate_workflow_sha256=expected_candidate_workflow_sha256,
            expected_authority_integration_targets=expected_authority_integration_targets,
            bootstrap_no_predecessor=True,
        )
        trusted_root_id = None
        selection_id = None
        relation = "BootstrapNoPredecessor"
    else:
        if trusted_root_receipt_v2 is None or expected_trusted_root_receipt_id is None:
            raise WitnessV3Error("ordinary witness requires TrustedMainRootReceiptV2 and independently expected root receipt ID")
        root = normalize_root_receipt_v2(trusted_root_receipt_v2)
        try:
            expected_root_id = v1._require_sha256_id(
                expected_trusted_root_receipt_id, where="expected_trusted_root_receipt_id"
            )
        except v1.WitnessError as exc:
            raise WitnessV3Error(str(exc)) from exc
        if root["root_receipt_id"] != expected_root_id:
            raise WitnessV3Error("trusted root receipt: content ID != independently expected root receipt ID")
        if any(value is None for value in (
            trusted_predecessor_sha, trusted_predecessor_tree,
            trusted_qualifier_sha256, trusted_workflow_sha256,
        )):
            raise WitnessV3Error("ordinary witness requires independently observed predecessor root and recipe bytes")
        assert trusted_predecessor_sha is not None
        assert trusted_predecessor_tree is not None
        assert trusted_qualifier_sha256 is not None
        assert trusted_workflow_sha256 is not None
        if root["repository"] != expected_repository or root["repository_id"] != expected_repository_id:
            raise WitnessV3Error("trusted root receipt: repository identity mismatch")
        if root["target_ref"] != "refs/heads/main":
            raise WitnessV3Error("trusted root receipt: expected refs/heads/main")
        if root["root_sha"] != trusted_predecessor_sha or root["root_tree"] != trusted_predecessor_tree:
            raise WitnessV3Error("trusted root receipt: predecessor SHA/tree mismatch")
        trusted_recipe_id = v1.compute_recipe_id(trusted_qualifier_sha256, trusted_workflow_sha256)
        if root["root_harness_identity"] != trusted_recipe_id:
            raise WitnessV3Error("trusted root receipt: harness identity != independently observed trusted recipe")

        base = v1.evaluate_witness(
            receipt_bytes=receipt_bytes,
            expected_subject_sha=expected_subject_sha,
            expected_subject_tree=expected_subject_tree,
            expected_run_id=expected_run_id,
            expected_run_attempt=expected_run_attempt,
            expected_run_conclusion=expected_run_conclusion,
            expected_workflow_id=expected_workflow_id,
            expected_artifact_id=expected_artifact_id,
            expected_artifact_name=expected_artifact_name,
            expected_artifact_archive_sha256=expected_artifact_archive_sha256,
            expected_repository=expected_repository,
            expected_repository_id=expected_repository_id,
            expected_candidate_qualifier_sha256=expected_candidate_qualifier_sha256,
            expected_candidate_workflow_sha256=expected_candidate_workflow_sha256,
            expected_authority_integration_targets=expected_authority_integration_targets,
            trusted_predecessor_sha=trusted_predecessor_sha,
            trusted_predecessor_tree=trusted_predecessor_tree,
            trusted_qualifier_sha256=trusted_qualifier_sha256,
            trusted_workflow_sha256=trusted_workflow_sha256,
        )
        trusted_root_id = root["root_receipt_id"]
        selection_id = root["trusted_evidence_selection_id"]
        relation = "IndependentlySelectedAdmittedRootBound"

    upgraded = dict(base)
    legacy_id = upgraded.pop("witness_id")
    upgraded["schema"] = WITNESS_SCHEMA
    upgraded["legacy_v1_witness_id"] = legacy_id
    upgraded["trusted_root_receipt_id"] = trusted_root_id
    upgraded["trusted_root_receipt_schema"] = ROOT_RECEIPT_SCHEMA if trusted_root_id is not None else None
    upgraded["trusted_evidence_selection_id"] = selection_id
    upgraded["trusted_root_receipt_relation"] = relation
    upgraded["witness_id"] = _domain_id(WITNESS_DOMAIN, upgraded)
    return upgraded


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
    parser.add_argument("--trusted-root-receipt-v2", type=Path)
    parser.add_argument("--expected-trusted-root-receipt-id")
    parser.add_argument("--trusted-predecessor-sha")
    parser.add_argument("--trusted-predecessor-tree")
    parser.add_argument("--trusted-qualifier-sha256")
    parser.add_argument("--trusted-workflow-sha256")
    parser.add_argument("--bootstrap-no-predecessor", action="store_true")
    parser.add_argument("--conformance", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = evaluate_witness_v3(
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
            trusted_root_receipt_v2=_load_json(args.trusted_root_receipt_v2) if args.trusted_root_receipt_v2 else None,
            expected_trusted_root_receipt_id=args.expected_trusted_root_receipt_id,
            trusted_predecessor_sha=args.trusted_predecessor_sha,
            trusted_predecessor_tree=args.trusted_predecessor_tree,
            trusted_qualifier_sha256=args.trusted_qualifier_sha256,
            trusted_workflow_sha256=args.trusted_workflow_sha256,
            bootstrap_no_predecessor=args.bootstrap_no_predecessor,
        )
    except (OSError, v1.WitnessError, WitnessV3Error) as exc:
        print(f"evidence-contract witness v3 invalid: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2, sort_keys=True))
    if result["witness_disposition"] == "FocusedSoftwareContractWitnessed":
        return 0
    return 0 if args.conformance else 3


if __name__ == "__main__":
    raise SystemExit(main())
