#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Black-box differential campaign for EUREKA qualification receipt grammar.

The canonical oracle, forensic-manifest producer, and offline forensic verifier
are executed as independent programs. No implementation is imported into this
campaign. Valid receipts must be accepted by all three at their respective
claim ceilings; every one-field mutant must be rejected independently.
"""

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

PREFLIGHT = (
    "receipt_schema_revision",
    "qualification_revision",
    "command_contract_revision",
    "repository",
    "event",
    "github_run_id",
    "github_run_attempt",
    "github_workflow_ref",
    "expected_subject_head",
    "subject_head",
    "subject_tree",
    "cargo_lock_sha256",
    "workflow_sha256",
    "command_contract_sha256",
    "rustc_version",
    "cargo_version",
    "checkout_clean_before",
    "claim_scope",
    "execution_authority_granted",
    "real_canary_executed",
    "heldout_executed",
    "confirmatory_evidence_minted",
)
POSTFLIGHT = (
    "postflight_head",
    "postflight_tree",
    "postflight_cargo_lock_sha256",
    "postflight_workflow_sha256",
    "postflight_command_contract_sha256",
    "checkout_clean_after",
    "qualification_result",
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


def write_contract(path: Path) -> None:
    write(
        path,
        """#!/usr/bin/env bash
set -euo pipefail
stage="${1:?stage required}"
printf 'receipt-symmetry-%s-pass\\n' "$stage"
""",
    )
    path.chmod(0o700)


def base_fields(workflow_sha: str, contract_sha: str) -> list[tuple[str, str]]:
    return [
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


def receipt_text(workflow_sha: str, contract_sha: str, *, sealed: bool) -> str:
    fields = base_fields(workflow_sha, contract_sha)
    if sealed:
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


def mutate_field(text: str, key: str, replacement: str) -> str:
    lines = text.splitlines()
    seen = 0
    out: list[str] = []
    for line in lines:
        field, value = line.split("=", 1)
        if field == key:
            value = replacement
            seen += 1
        out.append(f"{field}={value}")
    assert seen == 1, (key, seen)
    return "\n".join(out) + "\n"


def run(command: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)


def runner_env(producer: Path, runner: Path, stage_selftest: Path, workflow: Path, contract: Path) -> dict[str, str]:
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


def build_pass_stages(runner: Path, contract: Path, stage_dir: Path, env: dict[str, str]) -> None:
    for stage in STAGES:
        result = run([sys.executable, str(runner), stage, str(contract), str(stage_dir)], env=env)
        assert result.returncode == 0, result.stderr.decode(errors="replace")
    write(
        stage_dir / "stage-evidence-summary.env",
        "check_disposition=Passed\n"
        "test_disposition=Passed\n"
        "clippy_disposition=Passed\n"
        "diagnostic_evidence_complete=true\n",
    )


def producer_command(
    producer: Path,
    receipt: Path,
    stage_dir: Path,
    contract: Path,
    runner: Path,
    stage_selftest: Path,
    workflow: Path,
    output: Path,
) -> list[str]:
    return [
        sys.executable,
        str(producer),
        str(receipt),
        str(stage_dir),
        str(contract),
        str(runner),
        str(stage_selftest),
        str(workflow),
        str(output),
    ]


def verifier_command(
    verifier: Path,
    manifest: Path,
    receipt: Path,
    stage_dir: Path,
    contract: Path,
    runner: Path,
    stage_selftest: Path,
    producer: Path,
    workflow: Path,
) -> list[str]:
    return [
        sys.executable,
        str(verifier),
        HEAD,
        str(manifest),
        str(receipt),
        str(stage_dir),
        str(contract),
        str(runner),
        str(stage_selftest),
        str(producer),
        str(workflow),
    ]


def oracle_command(oracle: Path, receipt: Path, workflow_sha: str, contract_sha: str) -> list[str]:
    return [
        sys.executable,
        str(oracle),
        str(receipt),
        HEAD,
        TREE,
        LOCK,
        workflow_sha,
        contract_sha,
    ]


def recommit_for_receipt(source_manifest: Path, destination: Path, receipt: Path) -> None:
    lines = source_manifest.read_text(encoding="utf-8").splitlines()
    assert lines[-1].startswith("manifest_commitment=")
    body: list[str] = []
    seen: set[str] = set()
    changes = {
        "qualification_receipt_bytes": str(receipt.stat().st_size),
        "qualification_receipt_sha256": sha(receipt),
    }
    for line in lines[:-1]:
        key, value = line.split("=", 1)
        if key in changes:
            value = changes[key]
            seen.add(key)
        body.append(f"{key}={value}\n")
    assert seen == set(changes)
    raw = "".join(body).encode("utf-8")
    commitment = hashlib.sha256(DOMAIN + raw).hexdigest()
    destination.write_bytes(raw + f"manifest_commitment={commitment}\n".encode("ascii"))


def assert_no_cross_imports(path: Path, forbidden_fragments: tuple[str, ...]) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
        for name in names:
            assert all(fragment not in name for fragment in forbidden_fragments), (path, name)


def main() -> int:
    if len(sys.argv) != 8:
        print(
            "usage: eureka-v2-qualification-receipt-symmetry-selftest.py "
            "<oracle> <producer> <verifier> <stage-runner> <stage-selftest> <workflow> <campaign-output>",
            file=sys.stderr,
        )
        return 2
    oracle = Path(sys.argv[1]).resolve()
    producer = Path(sys.argv[2]).resolve()
    verifier = Path(sys.argv[3]).resolve()
    runner = Path(sys.argv[4]).resolve()
    stage_selftest = Path(sys.argv[5]).resolve()
    workflow = Path(sys.argv[6]).resolve()
    campaign_output = Path(sys.argv[7]).resolve()
    for path in (oracle, producer, verifier, runner, stage_selftest, workflow):
        assert path.is_file(), path

    assert_no_cross_imports(producer, ("receipt-oracle", "forensic-verifier"))
    assert_no_cross_imports(verifier, ("receipt-oracle", "forensic-manifest"))
    assert_no_cross_imports(oracle, ("forensic-manifest", "forensic-verifier"))

    with tempfile.TemporaryDirectory(prefix="eureka-receipt-symmetry-") as temporary:
        root = Path(temporary)
        contract = root / "contract.sh"
        write_contract(contract)
        env = runner_env(producer, runner, stage_selftest, workflow, contract)
        stage_dir = root / "stage-evidence"
        build_pass_stages(runner, contract, stage_dir, env)
        workflow_sha = env["EUREKA_WORKFLOW_SHA256"]
        contract_sha = env["EUREKA_COMMAND_CONTRACT_SHA256"]

        # Both valid canonical receipt classes must be accepted independently.
        valid_unsealed = root / "valid-unsealed.env"
        write(valid_unsealed, receipt_text(workflow_sha, contract_sha, sealed=False))
        oracle_result = run(oracle_command(oracle, valid_unsealed, workflow_sha, contract_sha))
        assert oracle_result.returncode == 0 and b"VALID_UNSEALED_RECEIPT" in oracle_result.stdout
        unsealed_manifest = root / "valid-unsealed-manifest.env"
        producer_result = run(
            producer_command(producer, valid_unsealed, stage_dir, contract, runner, stage_selftest, workflow, unsealed_manifest),
            env=env,
        )
        assert producer_result.returncode == 0, producer_result.stderr.decode(errors="replace")
        verifier_result = run(
            verifier_command(verifier, unsealed_manifest, valid_unsealed, stage_dir, contract, runner, stage_selftest, producer, workflow)
        )
        assert verifier_result.returncode == 0 and b"FORENSICALLY_VALID_INCOMPLETE" in verifier_result.stdout

        valid_pass = root / "valid-pass.env"
        write(valid_pass, receipt_text(workflow_sha, contract_sha, sealed=True))
        oracle_result = run(oracle_command(oracle, valid_pass, workflow_sha, contract_sha))
        assert oracle_result.returncode == 0 and b"VALID_PASS_RECEIPT" in oracle_result.stdout
        valid_manifest = root / "valid-pass-manifest.env"
        producer_result = run(
            producer_command(producer, valid_pass, stage_dir, contract, runner, stage_selftest, workflow, valid_manifest),
            env=env,
        )
        assert producer_result.returncode == 0, producer_result.stderr.decode(errors="replace")
        verifier_result = run(
            verifier_command(verifier, valid_manifest, valid_pass, stage_dir, contract, runner, stage_selftest, producer, workflow)
        )
        assert verifier_result.returncode == 0 and b"QUALIFICATION_PASS_EVIDENCE" in verifier_result.stdout

        replacements = {
            "receipt_schema_revision": "bad-schema",
            "qualification_revision": "bad-qualification",
            "command_contract_revision": "bad-contract",
            "repository": "other/repository",
            "event": "push",
            "github_run_id": "0",
            "github_run_attempt": "00",
            "github_workflow_ref": " noncanonical ",
            "expected_subject_head": "aa" * 20,
            "subject_head": "aa" * 20,
            "subject_tree": "aa" * 20,
            "cargo_lock_sha256": "aa" * 32,
            "workflow_sha256": "aa" * 32,
            "command_contract_sha256": "aa" * 32,
            "rustc_version": "rustc 1.95.0 (deadbeef 2026-08-20)",
            "cargo_version": "cargo 1.95.0 (cafebabe 2026-08-20)",
            "checkout_clean_before": "false",
            "claim_scope": "other-scope",
            "execution_authority_granted": "true",
            "real_canary_executed": "true",
            "heldout_executed": "true",
            "confirmatory_evidence_minted": "true",
            "postflight_head": "aa" * 20,
            "postflight_tree": "aa" * 20,
            "postflight_cargo_lock_sha256": "aa" * 32,
            "postflight_workflow_sha256": "aa" * 32,
            "postflight_command_contract_sha256": "aa" * 32,
            "checkout_clean_after": "false",
            "qualification_result": "FAIL",
        }
        assert set(replacements) == set(PREFLIGHT + POSTFLIGHT)

        valid_pass_text = valid_pass.read_text(encoding="utf-8")
        mutant_count = 0
        for key in PREFLIGHT + POSTFLIGHT:
            mutant_count += 1
            receipt = root / f"mutant-{mutant_count:02d}-{key}.env"
            write(receipt, mutate_field(valid_pass_text, key, replacements[key]))

            # Independent oracle rejection.
            assert run(oracle_command(oracle, receipt, workflow_sha, contract_sha)).returncode != 0, key

            # Independent producer rejection.
            mutant_output = root / f"mutant-{mutant_count:02d}.manifest.env"
            assert run(
                producer_command(producer, receipt, stage_dir, contract, runner, stage_selftest, workflow, mutant_output),
                env=env,
            ).returncode != 0, key
            assert not mutant_output.exists(), key

            # Independent verifier rejection, even if an attacker recomputes the
            # manifest commitment around the changed receipt identity.
            rebound = root / f"mutant-{mutant_count:02d}-rebound.env"
            recommit_for_receipt(valid_manifest, rebound, receipt)
            assert run(
                verifier_command(verifier, rebound, receipt, stage_dir, contract, runner, stage_selftest, producer, workflow)
            ).returncode != 0, key

        structural_mutants: list[tuple[str, str]] = []
        structural_mutants.append(("unknown-field", valid_pass_text.replace("checkout_clean_before=true\n", "shadow_field=true\ncheckout_clean_before=true\n")))
        structural_mutants.append(("duplicate-field", valid_pass_text.replace("event=pull_request\n", "event=pull_request\nevent=pull_request\n")))
        lines = valid_pass_text.splitlines()
        lines[3], lines[4] = lines[4], lines[3]
        structural_mutants.append(("reordered-fields", "\n".join(lines) + "\n"))
        structural_mutants.append(("partial-postflight", receipt_text(workflow_sha, contract_sha, sealed=False) + f"postflight_head={HEAD}\n"))
        structural_mutants.append(("missing-pass-field", valid_pass_text.replace("checkout_clean_after=true\n", "")))

        for name, text in structural_mutants:
            receipt = root / f"structural-{name}.env"
            write(receipt, text)
            assert run(oracle_command(oracle, receipt, workflow_sha, contract_sha)).returncode != 0, name
            output = root / f"structural-{name}.manifest.env"
            assert run(
                producer_command(producer, receipt, stage_dir, contract, runner, stage_selftest, workflow, output),
                env=env,
            ).returncode != 0, name
            assert not output.exists(), name
            rebound = root / f"structural-{name}-rebound.env"
            recommit_for_receipt(valid_manifest, rebound, receipt)
            assert run(
                verifier_command(verifier, rebound, receipt, stage_dir, contract, runner, stage_selftest, producer, workflow)
            ).returncode != 0, name

        campaign_output.parent.mkdir(parents=True, exist_ok=True)
        campaign_output.write_text(
            "campaign_revision=EUREKA.002.V2.QUALIFICATION_RECEIPT_SYMMETRY.v1\n"
            f"single_field_mutants={mutant_count}\n"
            f"structural_mutants={len(structural_mutants)}\n"
            "valid_unsealed_agreement=PASS\n"
            "valid_pass_agreement=PASS\n"
            "all_mutants_rejected_by_oracle=true\n"
            "all_mutants_rejected_by_producer=true\n"
            "all_mutants_rejected_by_verifier=true\n"
            "shared_validation_imports=false\n"
            "execution_authority_granted=false\n",
            encoding="utf-8",
        )

    print("EUREKA V2 qualification receipt symmetry campaign: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
