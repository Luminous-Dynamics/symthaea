#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adversarial controls for the independent EUREKA forensic verifier v3."""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

DOMAIN = b"EUREKA.002.V2.BACKEND_QUALIFICATION_FORENSIC_MANIFEST_COMMITMENT.v3\x00"
HEAD = "11" * 20
TREE = "22" * 20
LOCK = "33" * 32
RUSTC = "rustc 1.96.0 (deadbeef 2026-08-20)"
CARGO = "cargo 1.96.0 (cafebabe 2026-08-20)"
STAGES = ("check", "test", "clippy")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_contract(path: Path) -> None:
    path.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
stage="${1:?stage required}"
case "${VERIFIER_V3_SCENARIO:?scenario required}" in
  pass) printf 'verifier-v3-%s-pass\\n' "$stage" ;;
  fail) printf 'verifier-v3-%s-fail\\n' "$stage" >&2; exit 23 ;;
  *) exit 97 ;;
esac
""",
        encoding="utf-8",
    )
    path.chmod(0o700)


def base_env(producer: Path, runner: Path, stage_selftest: Path, workflow: Path, contract: Path) -> dict[str, str]:
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


def qualification_receipt(workflow_sha: str, contract_sha: str, *, passed: bool) -> str:
    fields = [
        ("receipt_schema_revision", "EUREKA.002.V2.BACKEND_QUALIFICATION_RECEIPT.v2"),
        ("qualification_revision", "EUREKA.002.V2.BACKEND_QUALIFICATION.v2"),
        ("command_contract_revision", "EUREKA.002.V2.BACKEND_QUALIFICATION_COMMANDS.v2"),
        ("repository", "Luminous-Dynamics/symthaea"),
        ("event", "pull_request"),
        ("github_run_id", "1"),
        ("github_run_attempt", "1"),
        ("github_workflow_ref", "Luminous-Dynamics/symthaea/.github/workflows/eureka-v2-backend-qualification.yml@refs/pull/1/merge"),
        ("expected_subject_head", HEAD),
        ("subject_head", HEAD),
        ("subject_tree", TREE),
        ("cargo_lock_sha256", LOCK),
        ("workflow_sha256", workflow_sha),
        ("command_contract_sha256", contract_sha),
        ("rustc_version", RUSTC),
        ("cargo_version", CARGO),
        ("checkout_clean_before", "true"),
        ("claim_scope", "backend-build-test-lint-only"),
        ("execution_authority_granted", "false"),
        ("real_canary_executed", "false"),
        ("heldout_executed", "false"),
        ("confirmatory_evidence_minted", "false"),
    ]
    if passed:
        fields.extend(
            [
                ("postflight_head", HEAD),
                ("postflight_tree", TREE),
                ("postflight_cargo_lock_sha256", LOCK),
                ("postflight_workflow_sha256", workflow_sha),
                ("postflight_command_contract_sha256", contract_sha),
                ("checkout_clean_after", "true"),
                ("qualification_result", "PASS"),
            ]
        )
    return "".join(f"{key}={value}\n" for key, value in fields)


def run_stage(runner: Path, contract: Path, stage_dir: Path, stage: str, scenario: str, env: dict[str, str]) -> subprocess.CompletedProcess[bytes]:
    child = env.copy()
    child["VERIFIER_V3_SCENARIO"] = scenario
    return subprocess.run(
        [sys.executable, str(runner), stage, str(contract), str(stage_dir)],
        env=child,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def write_summary(stage_dir: Path, check: str, test: str, clippy: str, complete: bool) -> None:
    stage_dir.mkdir(parents=True, exist_ok=True)
    write(
        stage_dir / "stage-evidence-summary.env",
        f"check_disposition={check}\n"
        f"test_disposition={test}\n"
        f"clippy_disposition={clippy}\n"
        f"diagnostic_evidence_complete={'true' if complete else 'false'}\n",
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


def make_bundle(
    root: Path,
    producer: Path,
    runner: Path,
    stage_selftest: Path,
    workflow: Path,
    *,
    failing: bool,
) -> tuple[Path, Path, Path, Path, dict[str, str]]:
    root.mkdir(parents=True, exist_ok=True)
    contract = root / "contract.sh"
    write_contract(contract)
    env = base_env(producer, runner, stage_selftest, workflow, contract)
    stage_dir = root / "stage-evidence"
    qualification = root / "qualification.env"
    manifest = root / "manifest.env"
    if failing:
        result = run_stage(runner, contract, stage_dir, "check", "fail", env)
        assert result.returncode == 23
        write_summary(
            stage_dir,
            "Failed",
            "NotRunDueToPredecessorFailure:check",
            "NotRunDueToPredecessorFailure:check",
            True,
        )
        write(qualification, qualification_receipt(env["EUREKA_WORKFLOW_SHA256"], env["EUREKA_COMMAND_CONTRACT_SHA256"], passed=False))
    else:
        for stage in STAGES:
            result = run_stage(runner, contract, stage_dir, stage, "pass", env)
            assert result.returncode == 0, result.stderr.decode(errors="replace")
        write_summary(stage_dir, "Passed", "Passed", "Passed", True)
        write(qualification, qualification_receipt(env["EUREKA_WORKFLOW_SHA256"], env["EUREKA_COMMAND_CONTRACT_SHA256"], passed=True))
    result = run_producer(producer, qualification, stage_dir, contract, runner, stage_selftest, workflow, manifest, env)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return manifest, qualification, stage_dir, contract, env


def args(
    verifier: Path,
    manifest: Path,
    qualification: Path,
    stage_dir: Path,
    contract: Path,
    runner: Path,
    stage_selftest: Path,
    producer: Path,
    workflow: Path,
    *,
    expected_head: str = HEAD,
) -> list[str]:
    return [
        sys.executable,
        str(verifier),
        expected_head,
        str(manifest),
        str(qualification),
        str(stage_dir),
        str(contract),
        str(runner),
        str(stage_selftest),
        str(producer),
        str(workflow),
    ]


def run_verify(arguments: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def recommit(path: Path, changes: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[-1].startswith("manifest_commitment=")
    body_lines: list[str] = []
    seen: set[str] = set()
    for line in lines[:-1]:
        key, value = line.split("=", 1)
        if key in changes:
            value = changes[key]
            seen.add(key)
        body_lines.append(f"{key}={value}\n")
    assert seen == set(changes)
    body = "".join(body_lines).encode("utf-8")
    commitment = hashlib.sha256(DOMAIN + body).hexdigest()
    path.write_bytes(body + f"manifest_commitment={commitment}\n".encode("ascii"))


def rebind_qualification(manifest: Path, qualification: Path) -> None:
    recommit(
        manifest,
        {
            "qualification_receipt_bytes": str(qualification.stat().st_size),
            "qualification_receipt_sha256": sha(qualification),
        },
    )


def no_local_imports(verifier: Path) -> None:
    tree = ast.parse(verifier.read_text(encoding="utf-8"))
    allowed = {"hashlib", "pathlib", "re", "sys", "typing", "__future__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".", 1)[0] in allowed, alias.name
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert (node.module or "").split(".", 1)[0] in allowed, node.module


def main() -> int:
    if len(sys.argv) != 6:
        print(
            "usage: eureka-v2-qualification-forensic-verifier-v3-selftest.py "
            "<verifier-v3> <producer-v3> <stage-runner> <stage-selftest> <workflow>",
            file=sys.stderr,
        )
        return 2
    verifier = Path(sys.argv[1]).resolve()
    producer = Path(sys.argv[2]).resolve()
    runner = Path(sys.argv[3]).resolve()
    stage_selftest = Path(sys.argv[4]).resolve()
    workflow = Path(sys.argv[5]).resolve()
    for path in (verifier, producer, runner, stage_selftest, workflow):
        assert path.is_file(), path
    no_local_imports(verifier)

    with tempfile.TemporaryDirectory(prefix="eureka-offline-verifier-v3-") as temporary:
        root = Path(temporary)
        manifest, qualification, stage_dir, contract, _ = make_bundle(
            root / "pass", producer, runner, stage_selftest, workflow, failing=False
        )
        verify_args = args(verifier, manifest, qualification, stage_dir, contract, runner, stage_selftest, producer, workflow)
        result = run_verify(verify_args)
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        assert b"classification=QUALIFICATION_PASS_EVIDENCE" in result.stdout

        wrong = args(
            verifier,
            manifest,
            qualification,
            stage_dir,
            contract,
            runner,
            stage_selftest,
            producer,
            workflow,
            expected_head="aa" * 20,
        )
        assert run_verify(wrong).returncode != 0

        original_log = (stage_dir / "check.combined.log").read_bytes()
        (stage_dir / "check.combined.log").write_bytes(original_log + b"tamper")
        assert run_verify(verify_args).returncode != 0
        (stage_dir / "check.combined.log").write_bytes(original_log)

        forged = root / "forged-disposition.env"
        shutil.copyfile(manifest, forged)
        recommit(forged, {"check_disposition": "Failed"})
        assert run_verify(args(verifier, forged, qualification, stage_dir, contract, runner, stage_selftest, producer, workflow)).returncode != 0

        # Recommitting the manifest around a receipt with an unknown field must
        # still fail at the independent receipt grammar boundary.
        unknown_receipt = root / "unknown-receipt.env"
        unknown_receipt.write_text(
            qualification.read_text(encoding="utf-8").replace("checkout_clean_before=true\n", "shadow_field=true\ncheckout_clean_before=true\n"),
            encoding="utf-8",
        )
        unknown_manifest = root / "unknown-manifest.env"
        shutil.copyfile(manifest, unknown_manifest)
        rebind_qualification(unknown_manifest, unknown_receipt)
        assert run_verify(args(verifier, unknown_manifest, unknown_receipt, stage_dir, contract, runner, stage_selftest, producer, workflow)).returncode != 0

        original_qualification = qualification.read_text(encoding="utf-8")
        qualification.write_text(
            original_qualification.replace(f"subject_head={HEAD}\n", f"subject_head={'aa' * 20}\n", 1),
            encoding="utf-8",
        )
        rebind_qualification(manifest, qualification)
        assert run_verify(verify_args).returncode != 0

        fail_manifest, fail_qualification, fail_stage_dir, fail_contract, _ = make_bundle(
            root / "fail", producer, runner, stage_selftest, workflow, failing=True
        )
        fail_args = args(
            verifier,
            fail_manifest,
            fail_qualification,
            fail_stage_dir,
            fail_contract,
            runner,
            stage_selftest,
            producer,
            workflow,
        )
        result = run_verify(fail_args)
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        assert b"classification=FORENSICALLY_VALID_FAILURE" in result.stdout

        producer_copy = root / "producer-mutated.py"
        shutil.copyfile(producer, producer_copy)
        producer_copy.write_text(producer_copy.read_text(encoding="utf-8") + "\n# alternate producer\n", encoding="utf-8")
        split_manifest = root / "split-producer.env"
        shutil.copyfile(fail_manifest, split_manifest)
        recommit(
            split_manifest,
            {
                "manifest_producer_file_bytes": str(producer_copy.stat().st_size),
                "manifest_producer_file_sha256": sha(producer_copy),
            },
        )
        split_args = args(
            verifier,
            split_manifest,
            fail_qualification,
            fail_stage_dir,
            fail_contract,
            runner,
            stage_selftest,
            producer_copy,
            workflow,
        )
        assert run_verify(split_args).returncode != 0

        false_pass_manifest = root / "false-pass.env"
        shutil.copyfile(fail_manifest, false_pass_manifest)
        false_pass_qualification = root / "false-pass-qualification.env"
        false_pass_qualification.write_text(
            qualification_receipt(sha(workflow), sha(fail_contract), passed=True),
            encoding="utf-8",
        )
        rebind_qualification(false_pass_manifest, false_pass_qualification)
        false_pass_args = args(
            verifier,
            false_pass_manifest,
            false_pass_qualification,
            fail_stage_dir,
            fail_contract,
            runner,
            stage_selftest,
            producer,
            workflow,
        )
        assert run_verify(false_pass_args).returncode != 0

    print("EUREKA V2 offline forensic verifier v3 conformance: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
