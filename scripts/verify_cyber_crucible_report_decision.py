#!/usr/bin/env python3
"""Verify a diagnostic-loop report and its separately signed decision receipt."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

try:
    from validate_cyber_crucible_diagnostic_loop import validate_report
except ModuleNotFoundError:
    from scripts.validate_cyber_crucible_diagnostic_loop import validate_report

try:
    from verify_cyber_crucible_decision_receipt import (
        DecisionVerificationError,
        ReplayLedger,
        canonical_json_digest,
        load_json,
        verify_decision_receipt,
    )
    from verify_cyber_crucible_verifier_provenance import (
        ProvenanceVerificationError,
        verify_verifier_provenance,
    )
    from package_cyber_crucible_verifier import BundleError, verify_bundle
except ModuleNotFoundError:
    from scripts.verify_cyber_crucible_decision_receipt import (
        DecisionVerificationError,
        ReplayLedger,
        canonical_json_digest,
        load_json,
        verify_decision_receipt,
    )
    from scripts.verify_cyber_crucible_verifier_provenance import (
        ProvenanceVerificationError,
        verify_verifier_provenance,
    )
    from scripts.package_cyber_crucible_verifier import BundleError, verify_bundle


class ReportDecisionBindingError(ValueError):
    """Report identity or bytes do not match the trusted decision context."""


def validate_input_receipt_binding(input_receipt: Any, expected_context: Any) -> str:
    """Bind the exact upstream signed receipt artifact to trusted expected context.

    The decision receipt authenticates what the independent verifier decided about
    this exact envelope. This function checks artifact linkage; it does not replace
    independent verification of the upstream signature.
    """
    if not isinstance(expected_context, dict):
        raise ReportDecisionBindingError("expected context must be an object")
    if not isinstance(input_receipt, dict) or set(input_receipt) != {"payload", "signature"}:
        raise ReportDecisionBindingError("input receipt envelope must contain exactly payload and signature")
    payload = input_receipt.get("payload")
    signature = input_receipt.get("signature")
    if not isinstance(payload, dict) or not isinstance(signature, str) or not signature:
        raise ReportDecisionBindingError("input receipt payload/signature has invalid shape")
    digest = canonical_json_digest(input_receipt)
    if digest != expected_context.get("input_receipt_digest"):
        raise ReportDecisionBindingError("input receipt digest does not match trusted expected context")
    for field in ("campaign_id", "subject_id", "attempt_id", "scenario_id", "scenario_revision", "scenario_digest"):
        if payload.get(field) != expected_context.get(field):
            raise ReportDecisionBindingError(f"input receipt does not match expected context field: {field}")
    return digest


def validate_report_binding(report: Any, expected_context: Any) -> str:
    """Verify the digest and stable run identity before authenticating its decision.

    The expected context must be obtained independently of the report/receipt pair.
    This function binds fields present in the diagnostic-loop report; campaign and
    policy provenance remain operator-supplied expected-context inputs.
    """
    if not isinstance(report, dict):
        raise ReportDecisionBindingError("diagnostic report must be an object")
    if not isinstance(expected_context, dict):
        raise ReportDecisionBindingError("expected context must be an object")
    if report.get("non_authorizing") is not True:
        raise ReportDecisionBindingError("diagnostic report must be explicitly non-authorizing")
    run = report.get("run")
    if not isinstance(run, dict):
        raise ReportDecisionBindingError("diagnostic report run must be an object")
    scenario = run.get("scenario")
    if not isinstance(scenario, dict):
        raise ReportDecisionBindingError("diagnostic report scenario must be an object")

    bindings = {
        "attempt_id": run.get("run_id"),
        "subject_id": run.get("subject_id"),
        "scenario_id": scenario.get("scenario_id"),
        "scenario_revision": scenario.get("revision"),
        "scenario_digest": scenario.get("scenario_digest"),
    }
    for field, actual in bindings.items():
        if actual != expected_context.get(field):
            raise ReportDecisionBindingError(
                f"diagnostic report does not match expected context field: {field}"
            )

    report_digest = canonical_json_digest(report)
    if expected_context.get("report_digest") != report_digest:
        raise ReportDecisionBindingError("diagnostic report digest does not match expected context")
    return report_digest


def verify_report_decision(
    report: Any,
    input_receipt: Any,
    decision_receipt: Any,
    decision_trust_policy: Any,
    expected_context: Any,
    *,
    verifier_artifact_path: str,
    provenance_policy: Any,
    replay_ledger: ReplayLedger,
    provenance_runner=None,
    now=None,
) -> dict[str, Any]:
    input_receipt_digest = validate_input_receipt_binding(input_receipt, expected_context)
    report_digest = validate_report_binding(report, expected_context)
    report_errors = validate_report(report)
    if report_errors:
        preview = "; ".join(report_errors[:8])
        suffix = f" (and {len(report_errors) - 8} more)" if len(report_errors) > 8 else ""
        raise ReportDecisionBindingError(f"diagnostic report contract rejected: {preview}{suffix}")

    # Provenance is checked at consumption time, not accepted as a mutable local JSON
    # result. The operator-controlled policy binds the exact verifier artifact digest.
    provenance = verify_verifier_provenance(
        verifier_artifact_path, provenance_policy, runner=provenance_runner
    )
    expected_build_digest = expected_context.get("verifier_build_digest")
    if provenance.get("provenance_verified") is not True:
        raise ReportDecisionBindingError("verifier artifact provenance was not verified")
    if provenance.get("artifact_sha256") != expected_build_digest:
        raise ReportDecisionBindingError(
            "verified verifier artifact digest does not match the signed decision context"
        )
    expected_provenance_policy_digest = expected_context.get("verifier_provenance_policy_digest")
    if provenance.get("policy_digest") != expected_provenance_policy_digest:
        raise ReportDecisionBindingError(
            "verified provenance policy digest does not match the signed decision context"
        )
    # An attestation proves a claim about an artifact; the bundle contract also checks
    # the exact embedded verifier/validator/schema/test inventory before consumption.
    bundle = verify_bundle(
        verifier_artifact_path,
        expected_source_commit=provenance["source_commit"],
    )
    if bundle.get("bundle_verified") is not True:
        raise ReportDecisionBindingError("verifier bundle inventory was not verified")
    if bundle.get("bundle_sha256") != provenance.get("artifact_sha256"):
        raise ReportDecisionBindingError("verified bundle digest differs from attested artifact digest")

    result = verify_decision_receipt(
        decision_receipt,
        decision_trust_policy,
        expected_context,
        now=now,
        replay_ledger=replay_ledger,
    )
    result["report_contract_valid"] = True
    result["verifier_provenance_verified"] = True
    result["verifier_artifact_sha256"] = provenance["artifact_sha256"]
    result["verifier_provenance_policy_digest_checked"] = True
    result["verifier_bundle_verified"] = True
    result["verifier_bundle_manifest_version"] = bundle["schema_version"]
    result["verifier_bundle_file_count"] = bundle["file_count"]
    result["input_receipt_digest"] = input_receipt_digest
    result["report_digest"] = report_digest
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, help="diagnostic-loop report JSON")
    parser.add_argument("--input-receipt", required=True, help="exact upstream signed receipt envelope JSON")
    parser.add_argument("--decision-receipt", required=True, help="signed verifier-decision receipt JSON")
    parser.add_argument("--trust-policy", required=True, help="operator-controlled decision signer policy JSON")
    parser.add_argument("--expected-context", required=True, help="independently supplied trusted expected context JSON")
    parser.add_argument("--verifier-artifact", required=True, help="exact verifier artifact whose provenance must be checked")
    parser.add_argument("--provenance-policy", required=True, help="operator-controlled verifier provenance policy JSON")
    parser.add_argument("--replay-db", required=True, help="protected persistent SQLite replay ledger")
    args = parser.parse_args(argv)
    try:
        result = verify_report_decision(
            load_json(args.report),
            load_json(args.input_receipt),
            load_json(args.decision_receipt),
            load_json(args.trust_policy),
            load_json(args.expected_context),
            verifier_artifact_path=args.verifier_artifact,
            provenance_policy=load_json(args.provenance_policy),
            replay_ledger=ReplayLedger(args.replay_db),
        )
        # Exit zero means the report contract and signature are verified. A valid
        # signed "reject" remains a rejection; consumers MUST inspect decision.
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except (BundleError, DecisionVerificationError, ProvenanceVerificationError, ReportDecisionBindingError, OSError) as exc:
        print(f"REJECTED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        # Malformed report JSON structures must fail closed without a traceback
        # becoming an accidental success signal for automation.
        print(f"REJECTED: unexpected validation failure ({type(exc).__name__})", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
