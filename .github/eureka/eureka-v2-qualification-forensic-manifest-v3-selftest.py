#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adversarial controls for the canonical EUREKA V2 forensic manifest v3."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile

DOMAIN = b"EUREKA.002.V2.BACKEND_QUALIFICATION_FORENSIC_MANIFEST_COMMITMENT.v3\x00"
STAGES = ("check", "test", "clippy")
HEAD = "11" * 20
TREE = "22" * 20
LOCK = "33" * 32


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse(path: Path) -> tuple[list[str], dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    assert text.endswith("\n")
    keys: list[str] = []
    values: dict[str, str] = {}
    for line in text[:-1].split("\n"):
        key, value = line.split("=", 1)
        assert key not in values, key
        keys.append(key)
        values[key] = value
    return keys, values


def write_contract(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
stage="${1:?stage required}"
case "${V3_SCENARIO:?scenario required}" in
  pass) printf 'v3-%s-pass\\n' "$stage" ;;
  fail) printf 'v3-%s-fail\\n' "$stage" >&2; exit 23 ;;
  *) exit 97 ;;
esac
""",
        encoding="utf-8",
    )
    path.chmod(0o700)


def env_for(producer: Path, runner: Path, stage_selftest: Path, workflow: Path, contract: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "EUREKA_SUBJECT_HEAD": HEAD,
            "EUREKA_SUBJECT_TREE": TREE,
            "EUREKA_CARGO_LOCK_SHA256": LOCK,
            "EUREKA_WORKFLOW_SHA256": sha(workflow),
            "EUREKA_COMMAND_CONTRACT_SHA256": sha(contract),
            "EUREKA_STAGE_RUNNER_SHA256": sha(runner),
            "EUREKA_STAGE_SELFTEST_SHA256": sha(stage_selftest),
            "EUREKA_FORENSIC_MANIFEST_TOOL_SHA256": sha(producer),
            "EUREKA_PYTHON_VERSION": f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "EUREKA_STAGE_LOG_LIMIT_BYTES": "8192",
        }
    )
    return env


def run_stage(
    runner: Path,
    contract: Path,
    stage_dir: Path,
    stage: str,
    scenario: str,
    env: dict[str, str],
) -> subprocess.CompletedProcess[bytes]:
    child_env = env.copy()
    child_env["V3_SCENARIO"] = scenario
    return subprocess.run(
        [sys.executable, str(runner), stage, str(contract), str(stage_dir)],
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def write_summary(stage_dir: Path, check: str, test: str, clippy: str, complete: bool) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "stage-evidence-summary.env").write_text(
        f"check_disposition={check}\n"
        f"test_disposition={test}\n"
        f"clippy_disposition={clippy}\n"
        f"diagnostic_evidence_complete={'true' if complete else 'false'}\n",
        encoding="utf-8",
    )


def run_producer(
    producer: Path,
    qualification: Path,
    stage_dir: Path,
    contract: Path,
    runner: Path,
    stage_selftest: Path,
    workflow: Path,
    output: Path,
    env: dict[str, str],
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            sys.executable,
            str(producer),
            str(qualification),
            str(stage_dir),
            str(contract),
            str(runner),
            str(stage_selftest),
            str(workflow),
            str(output),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def verify_commitment(path: Path) -> None:
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    assert lines[-1].startswith(b"manifest_commitment=")
    recorded = lines[-1].decode("ascii").strip().split("=", 1)[1]
    assert recorded == hashlib.sha256(DOMAIN + b"".join(lines[:-1])).hexdigest()


def clone_stage_dir(source: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file():
            (destination / item.name).write_bytes(item.read_bytes())


def main() -> int:
    if len(sys.argv) != 5:
        print(
            "usage: eureka-v2-qualification-forensic-manifest-v3-selftest.py "
            "<producer-v3> <stage-runner> <stage-selftest> <workflow>",
            file=sys.stderr,
        )
        return 2
    producer = Path(sys.argv[1]).resolve()
    runner = Path(sys.argv[2]).resolve()
    stage_selftest = Path(sys.argv[3]).resolve()
    workflow = Path(sys.argv[4]).resolve()
    for path in (producer, runner, stage_selftest, workflow):
        assert path.is_file(), path

    with tempfile.TemporaryDirectory(prefix="eureka-manifest-v3-") as temporary:
        root = Path(temporary)
        contract = root / "contract.sh"
        write_contract(contract)
        env = env_for(producer, runner, stage_selftest, workflow, contract)
        qualification = root / "qualification.env"
        qualification.write_text(
            "receipt_schema_revision=synthetic-v3\nexecution_authority_granted=false\nqualification_result=PASS\n",
            encoding="utf-8",
        )

        success = root / "success"
        for stage in STAGES:
            result = run_stage(runner, contract, success, stage, "pass", env)
            assert result.returncode == 0, result.stderr.decode(errors="replace")
        write_summary(success, "Passed", "Passed", "Passed", True)
        first = root / "manifest-first.env"
        second = root / "manifest-second.env"
        for output in (first, second):
            result = run_producer(
                producer, qualification, success, contract, runner, stage_selftest, workflow, output, env
            )
            assert result.returncode == 0, result.stderr.decode(errors="replace")
        assert first.read_bytes() == second.read_bytes()
        keys, values = parse(first)
        assert len(keys) == len(set(keys))
        assert values["forensic_manifest_schema_revision"] == "EUREKA.002.V2.BACKEND_QUALIFICATION_FORENSIC_MANIFEST.v3"
        assert values["workflow_sha256"] == sha(workflow)
        assert values["workflow_file_sha256"] == sha(workflow)
        assert values["command_contract_sha256"] == sha(contract)
        assert values["command_contract_file_sha256"] == sha(contract)
        assert values["stage_runner_sha256"] == sha(runner)
        assert values["stage_runner_file_sha256"] == sha(runner)
        assert values["stage_selftest_sha256"] == sha(stage_selftest)
        assert values["stage_selftest_file_sha256"] == sha(stage_selftest)
        assert values["forensic_manifest_tool_sha256"] == sha(producer)
        assert values["manifest_producer_file_sha256"] == sha(producer)
        assert values["execution_authority_granted"] == "false"
        verify_commitment(first)

        unexpected = success / "unexpected.bin"
        unexpected.write_bytes(b"unexpected")
        assert run_producer(
            producer,
            qualification,
            success,
            contract,
            runner,
            stage_selftest,
            workflow,
            root / "unexpected.env",
            env,
        ).returncode != 0
        unexpected.unlink()

        failure = root / "failure"
        result = run_stage(runner, contract, failure, "check", "fail", env)
        assert result.returncode == 23
        write_summary(
            failure,
            "Failed",
            "NotRunDueToPredecessorFailure:check",
            "NotRunDueToPredecessorFailure:check",
            True,
        )
        qualification.write_text(
            "receipt_schema_revision=synthetic-v3\nexecution_authority_granted=false\n",
            encoding="utf-8",
        )
        failure_manifest = root / "failure.env"
        result = run_producer(
            producer,
            qualification,
            failure,
            contract,
            runner,
            stage_selftest,
            workflow,
            failure_manifest,
            env,
        )
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        _, failure_values = parse(failure_manifest)
        assert failure_values["check_disposition"] == "Failed"
        assert failure_values["test_disposition"] == "NotRunDueToPredecessorFailure:check"
        assert failure_values["diagnostic_evidence_complete"] == "true"

        forged = root / "forged-summary"
        clone_stage_dir(failure, forged)
        write_summary(
            forged,
            "Failed",
            "NotRunDueToPredecessorFailure:test",
            "NotRunDueToPredecessorFailure:test",
            True,
        )
        assert run_producer(
            producer,
            qualification,
            forged,
            contract,
            runner,
            stage_selftest,
            workflow,
            root / "forged.env",
            env,
        ).returncode != 0

        late = root / "late"
        clone_stage_dir(failure, late)
        (late / "stage-evidence-summary.env").unlink()
        result = run_stage(runner, contract, late, "test", "pass", env)
        assert result.returncode == 0
        write_summary(late, "Failed", "Passed", "NotRunDueToPredecessorFailure:check", True)
        assert run_producer(
            producer,
            qualification,
            late,
            contract,
            runner,
            stage_selftest,
            workflow,
            root / "late.env",
            env,
        ).returncode != 0

        qualification.write_text(
            "receipt_schema_revision=synthetic-v3\nexecution_authority_granted=false\nqualification_result=PASS\n",
            encoding="utf-8",
        )
        assert run_producer(
            producer,
            qualification,
            failure,
            contract,
            runner,
            stage_selftest,
            workflow,
            root / "false-pass.env",
            env,
        ).returncode != 0

    print("EUREKA V2 forensic manifest v3 conformance: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
