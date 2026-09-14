#!/usr/bin/env python3
"""WCARE-48V receipt-run-identity wrapper over the structural FINAL verifier.

This proves that a FINAL receipt names the preregistered GitHub run identity.
It does not independently prove that GitHub's hosted run record produced the
commit; that stronger host attestation is a separate theorem.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "scripts/wcare48v_verify_final.py"
EXACT_RUN_ID = "34836223949"
EXACT_RUN_NUMBER = "1"
EXACT_RUN_ATTEMPT = "1"

spec = importlib.util.spec_from_file_location("wcare48v_core", CORE_PATH)
assert spec is not None and spec.loader is not None
core = importlib.util.module_from_spec(spec)
spec.loader.exec_module(core)


def emit(result: dict, code: int) -> int:
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return code


def base_result(classification: str, detail: str, target) -> dict:
    return {
        "authority": core.AUTHORITY,
        "classification": classification,
        "detail": detail,
        "prepared_head": core.PREPARED,
        "target": target,
        "receipt_run_identity_bound": False,
        "github_host_run_attested": False,
        "final_child_structurally_valid": False,
        "lock_admitted": False,
        "wcare42_executable_qualification_established": False,
        "runtime_authority_granted": False,
    }


def main() -> int:
    root_proc = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if root_proc.returncode != 0:
        return emit(base_result("INVALID_VERIFIER", "not_in_git_worktree", None), 4)

    root = Path(root_proc.stdout.decode("utf-8").strip())
    target_revision = (
        sys.argv[1] if len(sys.argv) == 2 else "wcare-48-observed-lock-generation"
    )

    try:
        result = core.verify(root, target_revision)
        if result["classification"] == "TARGET_MISSING":
            result["receipt_run_identity_bound"] = False
            result["github_host_run_attested"] = False
            return emit(result, 3)

        target = result["target"]
        receipt = core.strict_json(core.bytes_at(root, target, core.RECEIPT))
        core.require(receipt["github_run_id"] == EXACT_RUN_ID, "receipt_run_id_mismatch")
        core.require(
            receipt["github_run_number"] == EXACT_RUN_NUMBER,
            "receipt_run_number_mismatch",
        )
        core.require(
            receipt["github_run_attempt"] == EXACT_RUN_ATTEMPT,
            "receipt_run_attempt_mismatch",
        )
        result["receipt_run_identity_bound"] = True
        result["github_host_run_attested"] = False
        result["github_run_id"] = EXACT_RUN_ID
        result["github_run_number"] = EXACT_RUN_NUMBER
        result["github_run_attempt"] = EXACT_RUN_ATTEMPT
        return emit(result, 0)
    except core.InvalidFinal as exc:
        return emit(
            base_result(
                "FINAL_CHILD_INVALID",
                str(exc),
                core.resolve_commit(root, target_revision),
            ),
            1,
        )
    except (core.InvalidVerifier, UnicodeError) as exc:
        return emit(base_result("INVALID_VERIFIER", str(exc), None), 4)


if __name__ == "__main__":
    raise SystemExit(main())
