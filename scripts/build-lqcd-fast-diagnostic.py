#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build bounded, sanitized, non-authoritative diagnostics for LQCD qualification.

This script runs only after the canonical qualification attempt has been written.
Its output is a debugging aid. It is never an input to gate classification,
positive-receipt eligibility, or the canonical attempt digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

AUTHORITY = "NON_AUTHORITATIVE_DIAGNOSTIC"
SCHEMA_VERSION = "symthaea.focused-qualification-diagnostic.v1"
MAX_OUTPUT_BYTES = 48 * 1024
SUMMARY_OUTPUT_BYTES = 8 * 1024

ANSI_RE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))"
)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
        "[REDACTED_GITHUB_TOKEN]",
    ),
    (
        re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
        "Bearer [REDACTED]",
    ),
    (
        re.compile(
            r"(?i)\b(GITHUB_TOKEN|GH_TOKEN|ACTIONS_RUNTIME_TOKEN|TOKEN|PASSWORD|SECRET)=([^\s]+)"
        ),
        r"\1=[REDACTED]",
    ),
    (
        re.compile(r"://[^/\s:@]+:[^/\s@]+@"),
        "://[REDACTED]@",
    ),
)

COMMAND_PREFIXES = {
    "governance": "+ bash scripts/check-class-a-changes.sh --ci ",
    "cargo_metadata": "+ cargo metadata --locked --no-deps --format-version 1 ",
    "format": "+ cargo fmt -p symthaea-particle-physics -- --check ",
    "tests": "+ cargo test --locked -p symthaea-particle-physics --all-targets ",
    "clippy": (
        "+ cargo clippy --locked -p symthaea-particle-physics "
        "--all-targets -- -D warnings "
    ),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sanitize(text: str) -> tuple[str, int]:
    text = ANSI_RE.sub("", text)
    text = CONTROL_RE.sub("", text)
    redactions = 0
    for pattern, replacement in REDACTIONS:
        text, count = pattern.subn(replacement, text)
        redactions += count
    return text, redactions


def bounded_tail(text: str, limit: int) -> tuple[str, bool, int]:
    raw = text.encode("utf-8")
    original_bytes = len(raw)
    if original_bytes <= limit:
        return text, False, original_bytes
    tail = raw[-limit:]
    while tail and (tail[0] & 0xC0) == 0x80:
        tail = tail[1:]
    return tail.decode("utf-8", errors="replace"), True, original_bytes


def first_failed_gate(attempt: dict) -> str | None:
    required = attempt.get("qualification_profile", {}).get("required_gates", [])
    gates = attempt.get("gates", {})
    for name in required:
        if gates.get(name) not in {"PASS", "NOT_APPLICABLE"}:
            return name
    return None


def extract_gate_segment(log_text: str, gate: str | None) -> tuple[str | None, str]:
    if gate is None:
        return None, ""
    prefix = COMMAND_PREFIXES.get(gate)
    if prefix is None:
        return None, ""
    start = log_text.find(prefix)
    if start < 0:
        return None, ""
    line_end = log_text.find("\n", start)
    if line_end < 0:
        line_end = len(log_text)
    command = log_text[start:line_end].strip()

    next_command = log_text.find("\n+ ", line_end)
    end = len(log_text) if next_command < 0 else next_command + 1
    return command, log_text[start:end]


def build_diagnostic(attempt_bytes: bytes, log_bytes: bytes, producer_path: Path) -> dict | None:
    attempt = json.loads(attempt_bytes)
    terminal = attempt.get("terminal_disposition")
    verifier_error = attempt.get("verifier_error")
    gate = first_failed_gate(attempt)

    if terminal == "Passed" and verifier_error is None and gate is None:
        return None

    log_text = log_bytes.decode("utf-8", errors="replace")
    command, segment = extract_gate_segment(log_text, gate)
    if not segment:
        segment = log_text

    raw_segment_bytes = len(segment.encode("utf-8"))
    sanitized, redactions = sanitize(segment)
    output, truncated, sanitized_bytes = bounded_tail(sanitized, MAX_OUTPUT_BYTES)
    command_sanitized, command_redactions = sanitize(command or "")
    redactions += command_redactions

    gates = attempt.get("gates", {})
    subject = attempt.get("subject", {})
    verifier = attempt.get("verifier_authority", {})

    return {
        "schema_version": SCHEMA_VERSION,
        "authority": AUTHORITY,
        "authority_boundary": (
            "debugging aid only; never an input to qualification gates, "
            "terminal disposition, positive-receipt eligibility, or canonical receipt digest"
        ),
        "canonical_attempt_sha256": sha256_bytes(attempt_bytes),
        "terminal_disposition": terminal,
        "subject_sha": subject.get("checked_out_commit_sha"),
        "verifier_sha": verifier.get("checked_out_commit_sha"),
        "failure": {
            "gate": gate or "verifier_error",
            "gate_state": gates.get(gate) if gate else None,
            "command": command_sanitized or None,
            "gate_exit_code": None,
            "gate_exit_code_note": (
                "verifier-v2 records PASS/FAIL but does not emit the child process exit code; "
                "FAIL establishes a nonzero result"
                if gate
                else "not applicable"
            ),
            "qualifier_exit_code": 1,
            "output_tail": output,
            "raw_segment_bytes": raw_segment_bytes,
            "sanitized_output_bytes": sanitized_bytes,
            "captured_output_bytes": len(output.encode("utf-8")),
            "truncated": truncated,
            "redactions_applied": redactions,
            "sanitization_policy": "ansi-control-strip-plus-token-redaction-v1",
        },
        "producer": {
            "path": "scripts/build-lqcd-fast-diagnostic.py",
            "sha256": sha256_bytes(producer_path.read_bytes()),
        },
    }


def write_json(path: Path, value: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2) + "\n"
    path.write_text(payload)
    digest = sha256_bytes(payload.encode())
    Path(str(path) + ".sha256").write_text(f"{digest}  {path.name}\n")
    return digest


def append_summary(diagnostic: dict, digest: str) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    failure = diagnostic["failure"]
    display, _, _ = bounded_tail(failure["output_tail"], SUMMARY_OUTPUT_BYTES)
    display = display.replace("```", "``\u200b`")
    lines = [
        "### LQCD non-authoritative diagnostic",
        "",
        f"- Authority: `{AUTHORITY}`",
        f"- Terminal disposition: `{diagnostic.get('terminal_disposition')}`",
        f"- Failing gate: `{failure.get('gate')}`",
        f"- Subject: `{diagnostic.get('subject_sha')}`",
        f"- Verifier: `{diagnostic.get('verifier_sha')}`",
        f"- Canonical attempt SHA-256: `{diagnostic.get('canonical_attempt_sha256')}`",
        f"- Diagnostic SHA-256: `{digest}`",
        f"- Truncated: `{failure.get('truncated')}`",
        f"- Redactions applied: `{failure.get('redactions_applied')}`",
        "",
        "The diagnostic is a debugging aid only and cannot change qualification authority.",
        "",
        "```text",
        display,
        "```",
        "",
    ]
    with open(summary_path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def self_test() -> None:
    secret = "ghp_" + "A" * 24
    sample = f"\x1b[31merror\x1b[0m {secret} GITHUB_TOKEN=abcd1234\n"
    clean, redactions = sanitize(sample)
    assert "\x1b" not in clean
    assert secret not in clean
    assert "GITHUB_TOKEN=abcd1234" not in clean
    assert redactions == 2

    huge = "a" * (MAX_OUTPUT_BYTES + 1024)
    tail, truncated, original = bounded_tail(huge, MAX_OUTPUT_BYTES)
    assert truncated
    assert original == MAX_OUTPUT_BYTES + 1024
    assert len(tail.encode()) <= MAX_OUTPUT_BYTES

    attempt = {
        "terminal_disposition": "ClippyFailed",
        "verifier_error": None,
        "qualification_profile": {"required_gates": ["tests", "clippy"]},
        "gates": {"tests": "PASS", "clippy": "FAIL"},
        "subject": {"checked_out_commit_sha": "subject"},
        "verifier_authority": {"checked_out_commit_sha": "verifier"},
    }
    log = (
        "+ cargo test --locked -p symthaea-particle-physics --all-targets (cwd=/tmp/subject)\n"
        "tests passed\n"
        "+ cargo clippy --locked -p symthaea-particle-physics --all-targets -- -D warnings "
        "(cwd=/tmp/subject)\n"
        "warning: example lint\n"
        "+ git diff --quiet --ignore-submodules -- (cwd=/tmp/subject)\n"
    )
    diagnostic = build_diagnostic(
        json.dumps(attempt, sort_keys=True).encode(),
        log.encode(),
        Path(__file__),
    )
    assert diagnostic is not None
    assert diagnostic["failure"]["gate"] == "clippy"
    assert "example lint" in diagnostic["failure"]["output_tail"]
    assert "tests passed" not in diagnostic["failure"]["output_tail"]
    print("diagnostic_self_test=PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0

    if not args.attempt or not args.log or not args.output:
        parser.error("--attempt, --log, and --output are required unless --self-test is used")

    attempt_bytes = args.attempt.read_bytes()
    log_bytes = args.log.read_bytes()
    diagnostic = build_diagnostic(attempt_bytes, log_bytes, Path(__file__))
    if diagnostic is None:
        print("diagnostic=not-required")
        return 0

    digest = write_json(args.output, diagnostic)
    append_summary(diagnostic, digest)
    print(f"diagnostic={args.output}")
    print(f"diagnostic_sha256={digest}")
    print(f"authority={AUTHORITY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
