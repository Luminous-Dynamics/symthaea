#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adversarial conformance tests for EUREKA V2 qualification evidence tooling."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

INFRASTRUCTURE_EXIT = 125


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_env(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n"), path
    values: dict[str, str] = {}
    for line in text[:-1].split("\n"):
        key, value = line.split("=", 1)
        assert key not in values
        values[key] = value
    return values


def base_env(runner: Path, contract: Path, limit: int = 8192) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "EUREKA_COMMAND_CONTRACT_SHA256": sha256_file(contract),
            "EUREKA_STAGE_RUNNER_SHA256": sha256_file(runner),
            "EUREKA_SUBJECT_HEAD": "11" * 20,
            "EUREKA_SUBJECT_TREE": "22" * 20,
            "EUREKA_CARGO_LOCK_SHA256": "33" * 32,
            "EUREKA_WORKFLOW_SHA256": "44" * 32,
            "EUREKA_STAGE_LOG_LIMIT_BYTES": str(limit),
            "EUREKA_PYTHON_VERSION": (
                f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            ),
        }
    )
    return env


def run_stage(
    runner: Path,
    contract: Path,
    evidence_dir: Path,
    stage: str,
    scenario: str,
    *,
    limit: int = 8192,
    marker: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[bytes]:
    env = base_env(runner, contract, limit)
    env["EUREKA_SELFTEST_SCENARIO"] = scenario
    if marker is not None:
        env["EUREKA_SELFTEST_MARKER"] = str(marker)
    if extra_env is not None:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, str(runner), stage, str(contract), str(evidence_dir)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def write_contract(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
stage="${1:?stage required}"
case "${EUREKA_SELFTEST_SCENARIO:?scenario required}" in
  success)
    printf 'synthetic-%s-stdout\\n' "$stage"
    printf 'synthetic-%s-stderr\\n' "$stage" >&2
    ;;
  fail)
    printf 'synthetic-%s-failure\\n' "$stage" >&2
    exit 23
    ;;
  large-success)
    python3 - <<'PY'
import sys
sys.stdout.buffer.write(b'x' * 131072)
sys.stdout.buffer.flush()
PY
    printf 'child-completed\\n' > "${EUREKA_SELFTEST_MARKER:?marker required}"
    ;;
  signal)
    kill -TERM $$
    ;;
  *)
    echo 'unknown self-test scenario' >&2
    exit 97
    ;;
esac
""",
        encoding="utf-8",
    )
    path.chmod(0o700)


def assert_log_bound(receipt: dict[str, str], log: Path) -> None:
    assert receipt["log_retained_bytes"] == str(log.stat().st_size)
    assert receipt["log_sha256"] == sha256_file(log)


def test_stage_runner(runner: Path, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    contract = root / "synthetic-contract.sh"
    write_contract(contract)

    success_dir = root / "success"
    result = run_stage(runner, contract, success_dir, "check", "success")
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    success_receipt = parse_env(success_dir / "check.stage.env")
    success_log = success_dir / "check.combined.log"
    assert success_receipt["stage_disposition"] == "Passed"
    assert success_receipt["log_completeness"] == "Complete"
    assert success_receipt["command_exit"] == "0"
    assert success_receipt["python_version"] == base_env(runner, contract)["EUREKA_PYTHON_VERSION"]
    assert_log_bound(success_receipt, success_log)

    original_receipt_bytes = (success_dir / "check.stage.env").read_bytes()
    original_log_bytes = success_log.read_bytes()
    overwrite = run_stage(runner, contract, success_dir, "check", "success")
    assert overwrite.returncode != 0
    assert (success_dir / "check.stage.env").read_bytes() == original_receipt_bytes
    assert success_log.read_bytes() == original_log_bytes

    success_log.write_bytes(original_log_bytes + b"mutation")
    assert sha256_file(success_log) != success_receipt["log_sha256"]
    success_log.write_bytes(original_log_bytes)

    fail_dir = root / "failure"
    result = run_stage(runner, contract, fail_dir, "test", "fail")
    assert result.returncode == 23
    fail_receipt = parse_env(fail_dir / "test.stage.env")
    assert fail_receipt["stage_disposition"] == "Failed"
    assert fail_receipt["log_completeness"] == "Complete"
    assert fail_receipt["command_exit"] == "23"
    assert_log_bound(fail_receipt, fail_dir / "test.combined.log")

    signal_dir = root / "signal"
    result = run_stage(runner, contract, signal_dir, "clippy", "signal")
    assert result.returncode == 143
    signal_receipt = parse_env(signal_dir / "clippy.stage.env")
    assert signal_receipt["stage_disposition"] == "FailedBySignal"
    assert signal_receipt["command_exit"] == "signal-15"
    assert signal_receipt["log_completeness"] == "Complete"

    large_dir = root / "large"
    marker = root / "large-child-completed"
    result = run_stage(
        runner,
        contract,
        large_dir,
        "check",
        "large-success",
        limit=4096,
        marker=marker,
    )
    assert result.returncode == INFRASTRUCTURE_EXIT
    assert marker.read_text(encoding="utf-8") == "child-completed\n"
    large_receipt = parse_env(large_dir / "check.stage.env")
    large_log = large_dir / "check.combined.log"
    assert large_receipt["stage_disposition"] == "EvidenceTruncated"
    assert large_receipt["log_completeness"] == "Truncated"
    assert large_receipt["log_retained_bytes"] == "4096"
    assert int(large_receipt["log_observed_bytes"]) > 4096
    assert_log_bound(large_receipt, large_log)


def manifest_env(
    runner: Path,
    selftest: Path,
    manifest_tool: Path,
    workflow: Path,
    contract: Path,
) -> dict[str, str]:
    env = base_env(runner, contract)
    env.update(
        {
            "EUREKA_STAGE_SELFTEST_SHA256": sha256_file(selftest),
            "EUREKA_FORENSIC_MANIFEST_TOOL_SHA256": sha256_file(manifest_tool),
            "EUREKA_WORKFLOW_SHA256": sha256_file(workflow),
        }
    )
    return env


def test_manifest_tool(runner: Path, selftest: Path, manifest_tool: Path, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    contract = root / "manifest-contract.sh"
    write_contract(contract)
    workflow = root / "workflow.yml"
    workflow.write_text("name: synthetic-eureka-evidence-workflow\n", encoding="utf-8")
    workflow_sha = sha256_file(workflow)
    qualification_receipt = root / "qualification.env"
    qualification_receipt.write_text(
        "receipt_schema_revision=synthetic-v1\nexecution_authority_granted=false\n",
        encoding="utf-8",
    )
    stage_dir = root / "manifest-stages"
    for stage in ("check", "test", "clippy"):
        result = run_stage(
            runner,
            contract,
            stage_dir,
            stage,
            "success",
            extra_env={"EUREKA_WORKFLOW_SHA256": workflow_sha},
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")
    (stage_dir / "stage-evidence-summary.env").write_text(
        "check_disposition=Passed\n"
        "test_disposition=Passed\n"
        "clippy_disposition=Passed\n"
        "diagnostic_evidence_complete=true\n",
        encoding="utf-8",
    )

    env = manifest_env(runner, selftest, manifest_tool, workflow, contract)
    args = [
        sys.executable,
        str(manifest_tool),
        str(qualification_receipt),
        str(stage_dir),
        str(contract),
        str(runner),
        str(selftest),
        str(workflow),
    ]
    first = root / "manifest-1.env"
    second = root / "manifest-2.env"
    result = subprocess.run(args + [str(first)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    result = subprocess.run(args + [str(second)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert first.read_bytes() == second.read_bytes()
    manifest = parse_env(first)
    assert manifest["diagnostic_evidence_complete"] == "true"
    assert manifest["execution_authority_granted"] == "false"
    assert manifest["check_disposition"] == "Passed"
    assert manifest["check_log_sha256"] == sha256_file(stage_dir / "check.combined.log")
    assert len(manifest["manifest_commitment"]) == 64

    check_log = stage_dir / "check.combined.log"
    original = check_log.read_bytes()
    check_log.write_bytes(original + b"tampered")
    tampered = subprocess.run(
        args + [str(root / "manifest-tampered.env")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert tampered.returncode != 0
    check_log.write_bytes(original)

    unexpected = stage_dir / "unexpected.bin"
    unexpected.write_bytes(b"unexpected")
    extra = subprocess.run(
        args + [str(root / "manifest-extra.env")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert extra.returncode != 0
    unexpected.unlink()


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: eureka-v2-qualification-stage-runner-selftest.py <stage-runner> <manifest-tool>",
            file=sys.stderr,
        )
        return 2
    runner = Path(sys.argv[1]).resolve()
    manifest_tool = Path(sys.argv[2]).resolve()
    selftest = Path(__file__).resolve()
    if not runner.is_file() or not manifest_tool.is_file():
        print("self-test input tool missing", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="eureka-v2-evidence-selftest-") as temporary:
        root = Path(temporary)
        test_stage_runner(runner, root / "runner")
        test_manifest_tool(runner, selftest, manifest_tool, root / "manifest")
    print("EUREKA V2 qualification evidence conformance: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
