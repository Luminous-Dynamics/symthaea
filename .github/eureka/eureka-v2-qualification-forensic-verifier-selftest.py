#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adversarial conformance tests for the independent EUREKA forensic verifier."""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

DOMAIN = b"EUREKA.002.V2.BACKEND_QUALIFICATION_FORENSIC_MANIFEST_COMMITMENT.v2\x00"
HEAD = "11" * 20
TREE = "22" * 20
LOCK = "33" * 32


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def canonical_stage_receipt(
    stage: str,
    disposition: str,
    command_exit: str,
    log: bytes,
    *,
    workflow_sha: str,
    contract_sha: str,
    runner_sha: str,
    python_version: str,
    completeness: str = "Complete",
    capture_error: str = "none",
    limit: int = 8192,
    observed: int | None = None,
) -> str:
    retained = len(log)
    if observed is None:
        observed = retained
    fields = [
        ("stage_receipt_schema_revision", "EUREKA.002.V2.BACKEND_QUALIFICATION_STAGE_RECEIPT.v1"),
        ("stage", stage),
        ("subject_head", HEAD),
        ("subject_tree", TREE),
        ("cargo_lock_sha256", LOCK),
        ("workflow_sha256", workflow_sha),
        ("command_contract_sha256", contract_sha),
        ("stage_runner_sha256", runner_sha),
        ("python_version", python_version),
        ("log_limit_bytes", str(limit)),
        ("log_observed_bytes", str(observed)),
        ("log_retained_bytes", str(retained)),
        ("log_sha256", hashlib.sha256(log).hexdigest()),
        ("log_completeness", completeness),
        ("capture_error", capture_error),
        ("console_emit_complete", "true"),
        ("command_exit", command_exit),
        ("stage_disposition", disposition),
        ("execution_authority_granted", "false"),
    ]
    return "".join(f"{key}={value}\n" for key, value in fields)


def qualification_receipt(workflow_sha: str, contract_sha: str, *, passed: bool) -> str:
    fields = [
        ("receipt_schema_revision", "EUREKA.002.V2.BACKEND_QUALIFICATION_RECEIPT.v2"),
        ("qualification_revision", "EUREKA.002.V2.BACKEND_QUALIFICATION.v2"),
        ("command_contract_revision", "EUREKA.002.V2.BACKEND_QUALIFICATION_COMMANDS.v2"),
        ("repository", "synthetic/eureka"),
        ("event", "selftest"),
        ("github_run_id", "1"),
        ("github_run_attempt", "1"),
        ("github_workflow_ref", "synthetic@selftest"),
        ("expected_subject_head", HEAD),
        ("subject_head", HEAD),
        ("subject_tree", TREE),
        ("cargo_lock_sha256", LOCK),
        ("workflow_sha256", workflow_sha),
        ("command_contract_sha256", contract_sha),
        ("rustc_version", "rustc 1.96.0 synthetic"),
        ("cargo_version", "cargo 1.96.0 synthetic"),
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


def base_env(workflow: Path, contract: Path, runner: Path, stage_selftest: Path, producer: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "EUREKA_SUBJECT_HEAD": HEAD,
            "EUREKA_SUBJECT_TREE": TREE,
            "EUREKA_CARGO_LOCK_SHA256": LOCK,
            "EUREKA_WORKFLOW_SHA256": digest(workflow),
            "EUREKA_COMMAND_CONTRACT_SHA256": digest(contract),
            "EUREKA_STAGE_RUNNER_SHA256": digest(runner),
            "EUREKA_STAGE_SELFTEST_SHA256": digest(stage_selftest),
            "EUREKA_FORENSIC_MANIFEST_TOOL_SHA256": digest(producer),
            "EUREKA_PYTHON_VERSION": f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }
    )
    return env


def make_bundle(
    root: Path,
    producer: Path,
    contract: Path,
    runner: Path,
    stage_selftest: Path,
    workflow: Path,
    *,
    failing: bool,
) -> tuple[Path, Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    stage_dir = root / "stage-evidence"
    stage_dir.mkdir()
    qualification = root / "qualification.env"
    manifest = root / "forensic-manifest.env"
    env = base_env(workflow, contract, runner, stage_selftest, producer)
    python_version = env["EUREKA_PYTHON_VERSION"]
    workflow_sha = env["EUREKA_WORKFLOW_SHA256"]
    contract_sha = env["EUREKA_COMMAND_CONTRACT_SHA256"]
    runner_sha = env["EUREKA_STAGE_RUNNER_SHA256"]

    if failing:
        log = b"synthetic check compiler failure\n"
        (stage_dir / "check.combined.log").write_bytes(log)
        write(
            stage_dir / "check.stage.env",
            canonical_stage_receipt(
                "check",
                "Failed",
                "23",
                log,
                workflow_sha=workflow_sha,
                contract_sha=contract_sha,
                runner_sha=runner_sha,
                python_version=python_version,
            ),
        )
        write(
            stage_dir / "stage-evidence-summary.env",
            "check_disposition=Failed\n"
            "test_disposition=NotRunDueToPredecessorFailure:check\n"
            "clippy_disposition=NotRunDueToPredecessorFailure:check\n"
            "diagnostic_evidence_complete=true\n",
        )
        write(qualification, qualification_receipt(workflow_sha, contract_sha, passed=False))
    else:
        for stage in ("check", "test", "clippy"):
            log = f"synthetic {stage} pass\n".encode("utf-8")
            (stage_dir / f"{stage}.combined.log").write_bytes(log)
            write(
                stage_dir / f"{stage}.stage.env",
                canonical_stage_receipt(
                    stage,
                    "Passed",
                    "0",
                    log,
                    workflow_sha=workflow_sha,
                    contract_sha=contract_sha,
                    runner_sha=runner_sha,
                    python_version=python_version,
                ),
            )
        write(
            stage_dir / "stage-evidence-summary.env",
            "check_disposition=Passed\n"
            "test_disposition=Passed\n"
            "clippy_disposition=Passed\n"
            "diagnostic_evidence_complete=true\n",
        )
        write(qualification, qualification_receipt(workflow_sha, contract_sha, passed=True))

    result = subprocess.run(
        [
            sys.executable,
            str(producer),
            str(qualification),
            str(stage_dir),
            str(contract),
            str(runner),
            str(stage_selftest),
            str(workflow),
            str(manifest),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return manifest, qualification, stage_dir


def verifier_args(
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


def run_verify(args: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def recommit_manifest(path: Path, changes: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[-1].startswith("manifest_commitment=")
    body_lines: list[str] = []
    changed: set[str] = set()
    for line in lines[:-1]:
        key, value = line.split("=", 1)
        if key in changes:
            value = changes[key]
            changed.add(key)
        body_lines.append(f"{key}={value}\n")
    assert changed == set(changes)
    body = "".join(body_lines).encode("utf-8")
    commitment = hashlib.sha256(DOMAIN + body).hexdigest()
    path.write_bytes(body + f"manifest_commitment={commitment}\n".encode("ascii"))


def assert_no_local_imports(verifier: Path) -> None:
    tree = ast.parse(verifier.read_text(encoding="utf-8"))
    allowed = {"hashlib", "pathlib", "re", "sys", "typing", "__future__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".", 1)[0] in allowed, alias.name
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert (node.module or "").split(".", 1)[0] in allowed, node.module


def update_manifest_qualification_identity(manifest: Path, qualification: Path) -> None:
    recommit_manifest(
        manifest,
        {
            "qualification_receipt_bytes": str(qualification.stat().st_size),
            "qualification_receipt_sha256": digest(qualification),
        },
    )


def main() -> int:
    if len(sys.argv) != 7:
        print(
            "usage: eureka-v2-qualification-forensic-verifier-selftest.py "
            "<verifier> <producer> <contract> <runner> <stage-selftest> <workflow>",
            file=sys.stderr,
        )
        return 2
    verifier = Path(sys.argv[1]).resolve()
    producer = Path(sys.argv[2]).resolve()
    contract = Path(sys.argv[3]).resolve()
    runner = Path(sys.argv[4]).resolve()
    stage_selftest = Path(sys.argv[5]).resolve()
    workflow = Path(sys.argv[6]).resolve()
    for path in (verifier, producer, contract, runner, stage_selftest, workflow):
        assert path.is_file(), path
    assert_no_local_imports(verifier)

    with tempfile.TemporaryDirectory(prefix="eureka-v2-offline-forensic-verifier-") as tmp:
        root = Path(tmp)
        manifest, qualification, stage_dir = make_bundle(
            root / "passing", producer, contract, runner, stage_selftest, workflow, failing=False
        )
        args = verifier_args(
            verifier, manifest, qualification, stage_dir, contract, runner, stage_selftest, producer, workflow
        )
        result = run_verify(args)
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        assert b"classification=QUALIFICATION_PASS_EVIDENCE" in result.stdout

        assert run_verify(
            verifier_args(
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
        ).returncode != 0

        original_log = (stage_dir / "check.combined.log").read_bytes()
        (stage_dir / "check.combined.log").write_bytes(original_log + b"tamper")
        assert run_verify(args).returncode != 0
        (stage_dir / "check.combined.log").write_bytes(original_log)

        extra = stage_dir / "unexpected.bin"
        extra.write_bytes(b"unexpected")
        assert run_verify(args).returncode != 0
        extra.unlink()

        producer_copy = root / "producer-copy.py"
        shutil.copyfile(producer, producer_copy)
        producer_copy.write_text(producer_copy.read_text(encoding="utf-8") + "\n# mutation\n", encoding="utf-8")
        assert run_verify(
            verifier_args(
                verifier,
                manifest,
                qualification,
                stage_dir,
                contract,
                runner,
                stage_selftest,
                producer_copy,
                workflow,
            )
        ).returncode != 0

        forged_manifest = root / "forged-manifest.env"
        shutil.copyfile(manifest, forged_manifest)
        recommit_manifest(forged_manifest, {"check_disposition": "Failed"})
        assert run_verify(
            verifier_args(
                verifier,
                forged_manifest,
                qualification,
                stage_dir,
                contract,
                runner,
                stage_selftest,
                producer,
                workflow,
            )
        ).returncode != 0

        # Rebind a structurally valid manifest to a qualification receipt whose
        # subject identity was forged. The independent verifier must inspect the
        # receipt semantics instead of accepting the updated file hash.
        qual_original = qualification.read_text(encoding="utf-8")
        qualification.write_text(
            qual_original.replace(f"subject_head={HEAD}\n", f"subject_head={'aa' * 20}\n", 1),
            encoding="utf-8",
        )
        update_manifest_qualification_identity(manifest, qualification)
        assert run_verify(args).returncode != 0
        qualification.write_text(qual_original, encoding="utf-8")
        # Recreate a clean manifest after the intentional forged-subject case.
        env = base_env(workflow, contract, runner, stage_selftest, producer)
        subprocess.run(
            [
                sys.executable,
                str(producer),
                str(qualification),
                str(stage_dir),
                str(contract),
                str(runner),
                str(stage_selftest),
                str(workflow),
                str(root / "passing" / "manifest-clean.env"),
            ],
            env=env,
            check=True,
        )

        fail_manifest, fail_qualification, fail_stage_dir = make_bundle(
            root / "failing", producer, contract, runner, stage_selftest, workflow, failing=True
        )
        fail_args = verifier_args(
            verifier,
            fail_manifest,
            fail_qualification,
            fail_stage_dir,
            contract,
            runner,
            stage_selftest,
            producer,
            workflow,
        )
        result = run_verify(fail_args)
        assert result.returncode == 0, result.stderr.decode(errors="replace")
        assert b"classification=FORENSICALLY_VALID_FAILURE" in result.stdout

        # A malicious/replaced producer could recommit a failure bundle around a
        # false PASS receipt. Rebind the manifest file identity too; rejection
        # must therefore come from independent receipt/stage semantics.
        with fail_qualification.open("a", encoding="utf-8") as handle:
            handle.write("qualification_result=PASS\n")
        update_manifest_qualification_identity(fail_manifest, fail_qualification)
        assert run_verify(fail_args).returncode != 0

    print("EUREKA V2 offline forensic verifier conformance: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
