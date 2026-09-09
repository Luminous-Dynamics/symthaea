#!/usr/bin/env python3
"""V2 trusted-main root admission bound to independently selected evidence IDs.

V1 remains the strict content/cross-binding verifier and historical conformance
composer. V2 closes the remaining self-selection gap: a self-consistent bundle
cannot nominate its own protection/root/enforcement evidence and then call that
selection trusted.

The trusted phase must independently supply the expected identities for the
reviewed P0 policy, structural ruleset verification, effective-rule projection,
exact root subject, and enforcement evidence. V2 compares the re-derived IDs
from the supplied bytes against those trusted selections before issuing an
admitted historical-root receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import trusted_main_root_receipt as v1

ROOT_RECEIPT_SCHEMA = "symthaea.github-trusted-main-root-receipt.v2"
ROOT_RECEIPT_DOMAIN = b"symthaea.github-trusted-main-root-receipt.v2\0"
SELECTION_SCHEMA = "symthaea.github-trusted-main-evidence-selection.v1"
SELECTION_DOMAIN = b"symthaea.github-trusted-main-evidence-selection.v1\0"


class RootReceiptV2Error(ValueError):
    """Fail-closed V2 trusted evidence-selection error."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _content_id(domain: bytes, value: Any) -> str:
    return "sha256:" + hashlib.sha256(domain + _canonical(value)).hexdigest()


def _expected_id(value: str, *, where: str) -> str:
    try:
        return v1._sha256_id(value, where=where)
    except v1.RootReceiptError as exc:
        raise RootReceiptV2Error(str(exc)) from exc


def _selection_id(*, policy_id: str, structural_id: str, effective_id: str, root_subject_id: str, enforcement_id: str) -> str:
    return _content_id(SELECTION_DOMAIN, {
        "schema": SELECTION_SCHEMA,
        "protection_policy_id": policy_id,
        "protection_verification_id": structural_id,
        "effective_rules_verification_id": effective_id,
        "root_subject_id": root_subject_id,
        "enforcement_evidence_id": enforcement_id,
    })


def compose_root_receipt_v2(
    *,
    policy: Any,
    structural_verification: Any,
    effective_rules_verification: Any,
    root_subject: Any,
    enforcement_evidence: Any,
    root_harness_identity: str,
    expected_protection_policy_id: str,
    expected_protection_verification_id: str,
    expected_effective_rules_verification_id: str,
    expected_root_subject_id: str,
    expected_enforcement_evidence_id: str,
) -> dict[str, Any]:
    """Compose an admitted historical root only from independently selected IDs."""
    expected_policy_id = _expected_id(expected_protection_policy_id, where="expected_protection_policy_id")
    expected_structural_id = _expected_id(
        expected_protection_verification_id, where="expected_protection_verification_id"
    )
    expected_effective_id = _expected_id(
        expected_effective_rules_verification_id, where="expected_effective_rules_verification_id"
    )
    expected_subject_id = _expected_id(expected_root_subject_id, where="expected_root_subject_id")
    expected_enforcement_id = _expected_id(
        expected_enforcement_evidence_id, where="expected_enforcement_evidence_id"
    )

    # V1 revalidates closed schemas, content IDs, repository/ref identity,
    # ruleset identity, root SHA binding, and enforcement semantics.
    v1_receipt = v1.compose_root_receipt(
        policy=policy,
        structural_verification=structural_verification,
        effective_rules_verification=effective_rules_verification,
        root_subject=root_subject,
        enforcement_evidence=enforcement_evidence,
        root_harness_identity=root_harness_identity,
    )
    if v1_receipt["admission_disposition"] != "AdmittedHistoricalRoot":
        raise RootReceiptV2Error(
            "V2 admission requires V1 AdmittedHistoricalRoot; "
            f"observed {v1_receipt['admission_disposition']!r}"
        )

    actual = {
        "protection_policy_id": v1_receipt["protection_policy_id"],
        "protection_verification_id": v1_receipt["protection_verification_id"],
        "effective_rules_verification_id": v1_receipt["effective_rules_verification_id"],
        "root_subject_id": v1_receipt["root_subject_id"],
        "enforcement_evidence_id": v1_receipt["enforcement_evidence_id"],
    }
    expected = {
        "protection_policy_id": expected_policy_id,
        "protection_verification_id": expected_structural_id,
        "effective_rules_verification_id": expected_effective_id,
        "root_subject_id": expected_subject_id,
        "enforcement_evidence_id": expected_enforcement_id,
    }
    for field in (
        "protection_policy_id",
        "protection_verification_id",
        "effective_rules_verification_id",
        "root_subject_id",
        "enforcement_evidence_id",
    ):
        if actual[field] != expected[field]:
            raise RootReceiptV2Error(f"{field}: supplied evidence bytes do not match independently expected identity")

    selection_id = _selection_id(
        policy_id=expected_policy_id,
        structural_id=expected_structural_id,
        effective_id=expected_effective_id,
        root_subject_id=expected_subject_id,
        enforcement_id=expected_enforcement_id,
    )

    receipt: dict[str, Any] = {
        "schema": ROOT_RECEIPT_SCHEMA,
        "repository": v1_receipt["repository"],
        "repository_id": v1_receipt["repository_id"],
        "target_ref": v1_receipt["target_ref"],
        "root_sha": v1_receipt["root_sha"],
        "root_tree": v1_receipt["root_tree"],
        "root_subject_id": expected_subject_id,
        "protection_policy_id": expected_policy_id,
        "protection_verification_id": expected_structural_id,
        "effective_rules_verification_id": expected_effective_id,
        "enforcement_evidence_id": expected_enforcement_id,
        "trusted_evidence_selection_id": selection_id,
        "ruleset_id": v1_receipt["ruleset_id"],
        "bypass_policy_id": v1_receipt["bypass_policy_id"],
        "review_assurance": v1_receipt["review_assurance"],
        "root_harness_identity": v1_receipt["root_harness_identity"],
        "v1_root_receipt_id": v1_receipt["root_receipt_id"],
        "admission_disposition": "AdmittedHistoricalRoot",
        "evidence_selection_basis": "trusted-phase-independent-expected-ids",
        "historical_scope": "exact-root-only",
        "current_admission": "not-evaluated",
        "receipt_attestation": "none",
        "self_hosted_runner_activation": "not-authorized",
        "scientific_authority": "none",
        "bootstrap_authority": "none",
    }
    receipt["root_receipt_id"] = _content_id(ROOT_RECEIPT_DOMAIN, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", type=Path)
    parser.add_argument("structural_verification", type=Path)
    parser.add_argument("effective_rules_verification", type=Path)
    parser.add_argument("root_subject", type=Path)
    parser.add_argument("enforcement_evidence", type=Path)
    parser.add_argument("--root-harness-identity", required=True)
    parser.add_argument("--expected-protection-policy-id", required=True)
    parser.add_argument("--expected-protection-verification-id", required=True)
    parser.add_argument("--expected-effective-rules-verification-id", required=True)
    parser.add_argument("--expected-root-subject-id", required=True)
    parser.add_argument("--expected-enforcement-evidence-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = compose_root_receipt_v2(
            policy=v1._load_json(args.policy),
            structural_verification=v1._load_json(args.structural_verification),
            effective_rules_verification=v1._load_json(args.effective_rules_verification),
            root_subject=v1._load_json(args.root_subject),
            enforcement_evidence=v1._load_json(args.enforcement_evidence),
            root_harness_identity=args.root_harness_identity,
            expected_protection_policy_id=args.expected_protection_policy_id,
            expected_protection_verification_id=args.expected_protection_verification_id,
            expected_effective_rules_verification_id=args.expected_effective_rules_verification_id,
            expected_root_subject_id=args.expected_root_subject_id,
            expected_enforcement_evidence_id=args.expected_enforcement_evidence_id,
        )
    except (OSError, v1.RootReceiptError, RootReceiptV2Error) as exc:
        print(f"trusted-main root receipt V2 invalid: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
