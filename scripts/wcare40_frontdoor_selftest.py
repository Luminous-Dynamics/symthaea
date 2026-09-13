#!/usr/bin/env python3
"""Adversarial checks for the official WCARE-40 front door."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile

import wcare40_selftest as base

HERE = Path(__file__).resolve().parent
FRONTDOOR = HERE / "wcare40-qualify.py"


def main() -> int:
    original_verifier = base.VERIFIER
    original_write_lineage = base.write_lineage
    base.VERIFIER = FRONTDOOR
    try:
        with tempfile.TemporaryDirectory(prefix="wcare40-frontdoor-") as tmp:
            root = Path(tmp)
            same_receipt = base.h("qualified-evidence")
            domains = {"a": base.builder_domains("a"), "b": base.builder_domains("b")}
            outcomes = {"a": "PASS", "b": "PASS"}
            receipts = {"a": same_receipt, "b": same_receipt}

            code, supported = base.execute_case(root, "frontdoor-supported", domains, outcomes, receipts)
            assert code == 0 and supported["disposition"] == "REPLICATION_SUPPORTED", supported
            assert base.is_hex64(supported["wcare40_frontdoor_sha256"]), supported
            assert base.is_hex64(supported["wcare40_core_verifier_sha256"]), supported

            def forge_outcome(
                case_root: Path,
                replica_id: str,
                outcome: str,
                receipt_digest: str,
                env_label: str,
            ):
                prepared_path, final_path, prepared_sha, final_sha = original_write_lineage(
                    case_root, replica_id, outcome, receipt_digest, env_label
                )
                if replica_id == "a":
                    final = json.loads(final_path.read_text())
                    # Keep the top-level claim favorable while making the exact command
                    # evidence say FAIL. WCARE-39 compare alone sees no immutable drift;
                    # the WCARE-40 front door must recompute and reject this mismatch.
                    final["subject_outcome"] = "PASS"
                    final["commands"][0]["subject_outcome"] = "FAIL"
                    final["commands"][0]["exit_code"] = 7
                    final_sha = base.write_json(final_path, final)
                return prepared_path, final_path, prepared_sha, final_sha

            base.write_lineage = forge_outcome
            code, forged = base.execute_case(root, "forged-final-outcome", domains, outcomes, receipts)
            assert code == 4 and forged["disposition"] == "REPLICATION_INVALID", forged
            assert "final_subject_outcome_recomputation_mismatch:a:FAIL" in forged["detail"], forged

    finally:
        base.VERIFIER = original_verifier
        base.write_lineage = original_write_lineage

    print(json.dumps({
        "authority": "MeasurementOnly",
        "classification": "PASS_WCARE40_FRONTDOOR_SELFTEST",
        "frontdoor_supported_path_verified": True,
        "forged_final_outcome_rejected": True,
        "runtime_authority_granted": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
