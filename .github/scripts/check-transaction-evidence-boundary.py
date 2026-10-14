#!/usr/bin/env python3
"""Fail closed if the durable privileged-execution evidence state machine regresses.

Required monotonic flow:
    Started -> Bound(execution commitment) -> Completed(same commitment)

The checker is intentionally source-structural. It complements runtime tests by
making the existence and ordering of the durable evidence transition itself a CI
invariant.
"""

from pathlib import Path
import sys

RELAY = Path("crates/domains/symthaea-spore/src/bin/ssh_relay.rs")
LEDGER = Path("crates/domains/symthaea-spore/src/bin/system_transaction.rs")


def fail(message: str) -> None:
    print(f"transaction-evidence-boundary: FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    for path in (RELAY, LEDGER):
        if not path.is_file():
            fail(f"required source file missing: {path}")

    relay = RELAY.read_text(encoding="utf-8")
    ledger = LEDGER.read_text(encoding="utf-8")

    required_ledger_tokens = (
        "enum JournalEventKind",
        "Started,",
        "Bound,",
        "Completed,",
        "execution_commitment: Option<ExecutionCommitment>",
        "pub(crate) fn bind_execution_commitment(",
        "event: JournalEventKind::Bound",
        "record.execution_commitment = Some(execution_commitment.clone())",
        "execution_commitment: transaction.execution_commitment.clone()",
    )
    for token in required_ledger_tokens:
        if token not in ledger:
            fail(f"ledger evidence boundary lost required token: {token!r}")

    bound_pos = ledger.find("JournalEventKind::Bound => {")
    complete_pos = ledger.find("JournalEventKind::Completed => {")
    if bound_pos < 0 or complete_pos < 0 or bound_pos >= complete_pos:
        fail("ledger replay state machine is not ordered Bound before Completed")

    completion_guard = ledger.find(
        "transaction ledger completion at line {} changes the execution commitment"
    )
    install_success_guard = ledger.find(
        "successful install transactions require a durable execution commitment"
    )
    if install_success_guard < 0:
        fail("successful install transactions lost mandatory execution evidence")

    if completion_guard < 0:
        fail("completion no longer rejects execution-commitment mutation")

    bind_method = ledger.find("pub(crate) fn bind_execution_commitment(")
    bind_append = ledger.find("event: JournalEventKind::Bound", bind_method)
    if bind_method < 0 or bind_append < 0:
        fail("durable execution binding append disappeared")

    install_start = relay.find('            "install" => {')
    if install_start < 0:
        fail("install mutation arm disappeared")
    install_end = relay.find('
            "', install_start + 20)
    install_body = relay[install_start:install_end if install_end >= 0 else len(relay)]

    required_relay_tokens = (
        'role: "install-script"',
        "transaction.bind_execution_commitment(",
        "transaction_ledger",
        "bind_execution_commitment(&transaction",
        "spawn_privileged_background_script_file(",
        "staged_script.file",
    )
    for token in required_relay_tokens:
        if token not in install_body:
            fail(f"install arm lost execution-evidence token: {token!r}")

    local_bind = install_body.find("transaction.bind_execution_commitment(")
    durable_bind = install_body.find("transaction_ledger.bind_execution_commitment(")
    spawn = install_body.find("spawn_privileged_background_script_file(")
    if not (0 <= local_bind < durable_bind < spawn):
        fail("install script is not locally bound, durably bound, then executed")

    if "staged_script.digest_hex.clone()" not in install_body:
        fail("install execution commitment is no longer derived from staged script digest")

    print(
        "transaction-evidence-boundary: PASS: "
        "durable Started -> Bound -> Completed execution evidence is present and "
        "install binds before privileged execution"
    )


if __name__ == "__main__":
    main()
