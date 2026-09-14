#!/usr/bin/env python3
"""Fail-closed state campaign for WCARE-48H."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts/wcare48h_attest.py"


def run_case(payload: dict) -> tuple[int, dict]:
    with tempfile.TemporaryDirectory(prefix="wcare48h-selftest-") as tmp:
        path = Path(tmp) / "run.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        proc = subprocess.run(
            [sys.executable, str(VERIFIER), "--run-json", str(path)],
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return proc.returncode, json.loads(proc.stdout)


def exact(status: str, conclusion=None) -> dict:
    return {
        "id": 34836223949,
        "name": "WCARE-48 Observed Lock Generation",
        "path": ".github/workflows/wcare48-lock-generation.yml",
        "head_branch": "wcare-48-observed-lock-generation",
        "head_sha": "52d1d9fb741250ab8bcab205113689a8cc9431bb",
        "event": "push",
        "run_number": 1,
        "run_attempt": 1,
        "status": status,
        "conclusion": conclusion,
    }


def main() -> int:
    code, result = run_case(exact("queued"))
    assert code == 3, (code, result)
    assert result["classification"] == "HOST_RUN_INDETERMINATE"
    assert result["detail"] == "exact_run_not_completed"
    assert result["github_host_run_attested"] is False
    assert result["lock_admitted"] is False
    assert result["runtime_authority_granted"] is False
    assert isinstance(result["run_json_sha256"], str) and len(result["run_json_sha256"]) == 64

    code, result = run_case(exact("completed", "failure"))
    assert code == 1, (code, result)
    assert result["classification"] == "HOST_RUN_FAILED"
    assert "host_run_conclusion:failure" in result["detail"]
    assert result["github_host_run_attested"] is False

    wrong = exact("queued")
    wrong["run_attempt"] = 2
    code, result = run_case(wrong)
    assert code == 1, (code, result)
    assert result["classification"] == "HOST_RUN_FAILED"
    assert "run_identity_mismatch:run_attempt" in result["detail"]

    print("PASS_WCARE48H_FAIL_CLOSED_STATES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
