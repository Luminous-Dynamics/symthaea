#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Adversarial conformance suite for the EUREKA qualification-receipt oracle."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys
import tempfile

HEAD = "11" * 20
TREE = "22" * 20
LOCK = "33" * 32
WORKFLOW = "44" * 32
CONTRACT = "55" * 32

PREFLIGHT = [
    ("receipt_schema_revision", "EUREKA.002.V2.BACKEND_QUALIFICATION_RECEIPT.v2"),
    ("qualification_revision", "EUREKA.002.V2.BACKEND_QUALIFICATION.v2"),
    ("command_contract_revision", "EUREKA.002.V2.BACKEND_QUALIFICATION_COMMANDS.v2"),
    ("repository", "Luminous-Dynamics/symthaea"),
    ("event", "pull_request"),
    ("github_run_id", "123456789"),
    ("github_run_attempt", "1"),
    (
        "github_workflow_ref",
        "Luminous-Dynamics/symthaea/.github/workflows/eureka-v2-backend-qualification.yml@refs/pull/1/merge",
    ),
    ("expected_subject_head", HEAD),
    ("subject_head", HEAD),
    ("subject_tree", TREE),
    ("cargo_lock_sha256", LOCK),
    ("workflow_sha256", WORKFLOW),
    ("command_contract_sha256", CONTRACT),
    ("rustc_version", "rustc 1.96.0 (abcdef123 2026-03-05)"),
    ("cargo_version", "cargo 1.96.0 (abcdef123 2026-03-05)"),
    ("checkout_clean_before", "true"),
    ("claim_scope", "backend-build-test-lint-only"),
    ("execution_authority_granted", "false"),
    ("real_canary_executed", "false"),
    ("heldout_executed", "false"),
    ("confirmatory_evidence_minted", "false"),
]
POSTFLIGHT = [
    ("postflight_head", HEAD),
    ("postflight_tree", TREE),
    ("postflight_cargo_lock_sha256", LOCK),
    ("postflight_workflow_sha256", WORKFLOW),
    ("postflight_command_contract_sha256", CONTRACT),
    ("checkout_clean_after", "true"),
    ("qualification_result", "PASS"),
]


def encode(fields: list[tuple[str, str]], *, terminal_newline: bool = True) -> bytes:
    text = "".join(f"{key}={value}\n" for key, value in fields)
    if not terminal_newline:
        text = text[:-1]
    return text.encode("utf-8")


def run(
    oracle: Path,
    receipt: Path,
    *,
    head: str = HEAD,
    tree: str = TREE,
    lock: str = LOCK,
    workflow: str = WORKFLOW,
    contract: str = CONTRACT,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [sys.executable, str(oracle), str(receipt), head, tree, lock, workflow, contract],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def write_and_run(
    root: Path,
    oracle: Path,
    name: str,
    fields: list[tuple[str, str]],
    *,
    terminal_newline: bool = True,
    head: str = HEAD,
    tree: str = TREE,
    lock: str = LOCK,
    workflow: str = WORKFLOW,
    contract: str = CONTRACT,
) -> subprocess.CompletedProcess[bytes]:
    path = root / f"{name}.env"
    path.write_bytes(encode(fields, terminal_newline=terminal_newline))
    return run(oracle, path, head=head, tree=tree, lock=lock, workflow=workflow, contract=contract)


def replace(fields: list[tuple[str, str]], key: str, value: str) -> list[tuple[str, str]]:
    out = [(k, value if k == key else v) for k, v in fields]
    assert sum(1 for k, _ in fields if k == key) == 1
    return out


def remove(fields: list[tuple[str, str]], key: str) -> list[tuple[str, str]]:
    out = [(k, v) for k, v in fields if k != key]
    assert len(out) + 1 == len(fields)
    return out


def assert_reject(result: subprocess.CompletedProcess[bytes], label: str) -> None:
    assert result.returncode != 0, (label, result.stdout, result.stderr)


def assert_stdlib_only(oracle: Path) -> None:
    tree = ast.parse(oracle.read_text(encoding="utf-8"))
    allowed = {"pathlib", "re", "sys", "typing", "__future__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".", 1)[0] in allowed, alias.name
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert (node.module or "").split(".", 1)[0] in allowed, node.module


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: receipt-oracle-selftest.py <oracle>", file=sys.stderr)
        return 2
    oracle = Path(sys.argv[1]).resolve()
    assert oracle.is_file()
    assert_stdlib_only(oracle)

    with tempfile.TemporaryDirectory(prefix="eureka-receipt-oracle-") as tmp:
        root = Path(tmp)
        unsealed = write_and_run(root, oracle, "valid-unsealed", PREFLIGHT)
        assert unsealed.returncode == 0, unsealed.stderr.decode(errors="replace")
        assert b"classification=VALID_UNSEALED_RECEIPT" in unsealed.stdout
        assert b"execution_authority_granted=false" in unsealed.stdout

        sealed_fields = PREFLIGHT + POSTFLIGHT
        sealed = write_and_run(root, oracle, "valid-pass", sealed_fields)
        assert sealed.returncode == 0, sealed.stderr.decode(errors="replace")
        assert b"classification=VALID_PASS_RECEIPT" in sealed.stdout

        mutations: list[tuple[str, list[tuple[str, str]], bool]] = [
            ("unknown", PREFLIGHT + [("shadow_authority", "false")], True),
            ("duplicate", PREFLIGHT + [("subject_head", HEAD)], True),
            ("missing", remove(PREFLIGHT, "subject_tree"), True),
            ("zero-run", replace(PREFLIGHT, "github_run_id", "0"), True),
            ("leading-zero-run", replace(PREFLIGHT, "github_run_attempt", "01"), True),
            ("bad-head", replace(PREFLIGHT, "subject_head", "AA" * 20), True),
            ("bad-tree", replace(PREFLIGHT, "subject_tree", "GG" * 20), True),
            ("bad-lock", replace(PREFLIGHT, "cargo_lock_sha256", "ZZ" * 32), True),
            ("wrong-head", replace(PREFLIGHT, "subject_head", "aa" * 20), True),
            ("wrong-tree", replace(PREFLIGHT, "subject_tree", "aa" * 20), True),
            ("wrong-lock", replace(PREFLIGHT, "cargo_lock_sha256", "aa" * 32), True),
            ("wrong-workflow", replace(PREFLIGHT, "workflow_sha256", "aa" * 32), True),
            ("wrong-contract", replace(PREFLIGHT, "command_contract_sha256", "aa" * 32), True),
            ("wrong-expected", replace(PREFLIGHT, "expected_subject_head", "aa" * 20), True),
            ("wrong-repository", replace(PREFLIGHT, "repository", "Other/repo"), True),
            ("wrong-event", replace(PREFLIGHT, "event", "push"), True),
            ("wrong-scope", replace(PREFLIGHT, "claim_scope", "everything"), True),
            ("wrong-rust", replace(PREFLIGHT, "rustc_version", "rustc 1.95.0 (abcdef123 2026-01-01)"), True),
            ("wrong-cargo", replace(PREFLIGHT, "cargo_version", "cargo 1.97.0 (abcdef123 2026-04-01)"), True),
            ("exec-authority", replace(PREFLIGHT, "execution_authority_granted", "true"), True),
            ("canary", replace(PREFLIGHT, "real_canary_executed", "true"), True),
            ("heldout", replace(PREFLIGHT, "heldout_executed", "true"), True),
            ("confirmatory", replace(PREFLIGHT, "confirmatory_evidence_minted", "true"), True),
            ("partial-postflight", PREFLIGHT + POSTFLIGHT[:2], True),
            ("pass-missing-postflight", remove(sealed_fields, "postflight_tree"), True),
            ("postflight-drift", replace(sealed_fields, "postflight_head", "aa" * 20), True),
            ("false-pass", replace(sealed_fields, "qualification_result", "FAIL"), True),
            ("dirty-after", replace(sealed_fields, "checkout_clean_after", "false"), True),
            ("no-newline", PREFLIGHT, False),
        ]
        reordered = PREFLIGHT.copy()
        reordered[8], reordered[9] = reordered[9], reordered[8]
        mutations.append(("reordered", reordered, True))
        for name, fields, terminal_newline in mutations:
            assert_reject(
                write_and_run(root, oracle, name, fields, terminal_newline=terminal_newline),
                name,
            )

        for name, kwargs in [
            ("external-head", {"head": "aa" * 20}),
            ("external-tree", {"tree": "aa" * 20}),
            ("external-lock", {"lock": "aa" * 32}),
            ("external-workflow", {"workflow": "aa" * 32}),
            ("external-contract", {"contract": "aa" * 32}),
        ]:
            assert_reject(write_and_run(root, oracle, name, PREFLIGHT, **kwargs), name)

        control = root / "control.env"
        control.write_bytes(encode(PREFLIGHT).replace(b"event=pull_request\n", b"event=pull_request\r\n"))
        assert_reject(run(oracle, control), "carriage-return")
        workflow_ref_space = replace(
            PREFLIGHT,
            "github_workflow_ref",
            " Luminous-Dynamics/symthaea/.github/workflows/eureka-v2-backend-qualification.yml@main",
        )
        assert_reject(write_and_run(root, oracle, "workflow-ref-space", workflow_ref_space), "workflow-ref-space")

    print("EUREKA V2 qualification receipt oracle conformance: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
