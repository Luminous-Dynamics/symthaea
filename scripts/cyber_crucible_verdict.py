"""Fail-closed combination rule for Cyber Crucible assessment results.

This module classifies already-collected evidence-check records. It does not
verify receipt signatures or grant authority to change a live system.
"""
from __future__ import annotations

import re
from typing import Any

VALID_STATUSES = {"pass", "fail", "inconclusive", "not_run"}
SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def is_correct_and_secure(
    functional_status: Any,
    security_status: Any,
    execution_mode: Any,
    required_evidence_refs: Any,
    required_evidence_checks: Any,
) -> bool:
    """Return True only when both verdicts and every evidence gate pass.

    Fail closed on unknown statuses, simulated/not-run execution, missing
    required evidence, failed/inconclusive check records, malformed check
    records, duplicate check IDs, or missing/malformed receipt digests.

    Receipt authenticity and the digest-to-content binding are responsibilities
    of the upstream evidence verifier; this function only checks the structure
    and status of records supplied by that verifier.
    """
    if functional_status != "pass" or security_status != "pass":
        return False
    if execution_mode != "real":
        return False
    if not isinstance(required_evidence_refs, list) or not required_evidence_refs:
        return False
    if any(not isinstance(ref, str) or not ref for ref in required_evidence_refs):
        return False
    if len(set(required_evidence_refs)) != len(required_evidence_refs):
        return False
    if not isinstance(required_evidence_checks, list) or not required_evidence_checks:
        return False

    seen_check_ids: set[str] = set()
    covered_refs: set[str] = set()
    for check in required_evidence_checks:
        if not isinstance(check, dict):
            return False
        check_id = check.get("check_id")
        status = check.get("status")
        refs = check.get("evidence_refs")
        receipt_digest = check.get("receipt_digest")
        if not isinstance(check_id, str) or not check_id or check_id in seen_check_ids:
            return False
        if not isinstance(status, str) or status not in VALID_STATUSES or status != "pass":
            return False
        if not isinstance(refs, list) or not refs:
            return False
        if any(not isinstance(ref, str) or not ref for ref in refs):
            return False
        if not isinstance(receipt_digest, str) or not SHA256_HEX.fullmatch(receipt_digest):
            return False
        seen_check_ids.add(check_id)
        covered_refs.update(refs)

    return set(required_evidence_refs).issubset(covered_refs)
