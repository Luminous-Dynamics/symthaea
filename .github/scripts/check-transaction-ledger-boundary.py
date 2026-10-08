#!/usr/bin/env python3
"""Fail closed if the transaction journal loses its durability invariants.

This is a structural ratchet, not a replacement for runtime tests. It protects
the journal properties that make the local append-only transaction protocol
auditable across future refactors.
"""

from __future__ import annotations

from pathlib import Path
import sys

SOURCE = Path("crates/domains/symthaea-spore/src/bin/system_transaction.rs")


def fail(message: str) -> None:
    print(f"transaction-ledger-boundary: FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def require(text: str, needle: str, description: str) -> None:
    if needle not in text:
        fail(f"{description}: missing {needle!r}")


def main() -> None:
    if not SOURCE.is_file():
        fail(f"source file missing: {SOURCE}")

    text = SOURCE.read_text(encoding="utf-8")

    # Source integrity.
    for marker in (
        "pub(crate) struct MutationLease",
        "pub(crate) struct TransactionLedger",
        "enum JournalEventKind",
        "fn open_ledger_file(",
        "fn append(&self, event: &JournalEvent)",
        "pub(crate) fn admit(",
        "pub(crate) fn lookup(",
        "fn random_operation_id()",
        "#[cfg(test)]\nmod tests {",
    ):
        require(text, marker, "ledger source-integrity sentinel")

    # Journal records must have explicit typed event kinds rather than an
    # unconstrained string field.
    require(text, "event: JournalEventKind,", "journal event must be typed")
    require(text, "JournalEventKind::Started", "started event constructor")
    require(text, "JournalEventKind::Completed", "completed event constructor")
    require(text, "match event.event {", "journal event dispatch")

    # Every writer serializes the complete size-check/write/fsync operation.
    append_start = text.find("fn append(&self, event: &JournalEvent)")
    append_end = text.find("\n    pub(crate) fn admit(", append_start)
    if append_start < 0 or append_end < 0:
        fail("append boundary disappeared")
    append = text[append_start:append_end]
    for needle in (
        "libc::flock(file.as_raw_fd(), libc::LOCK_EX)",
        "metadata.len()",
        "checked_add(serialized.len() as u64)",
        "let mut framed = serialized.into_bytes();",
        "framed.push(b'\\n');",
        "file.write_all(&framed)",
        "file.sync_all()",
        "self.directory.sync_all()",
    ):
        require(append, needle, "journal append durability invariant")

    # Readers synchronize with writers so a normal reader cannot parse a
    # record between its write and fsync.
    load_start = text.find("fn load(&self)")
    load_end = text.find("\n    fn append(&self, event: &JournalEvent)", load_start)
    if load_start < 0 or load_end < 0:
        fail("load boundary disappeared")
    load = text[load_start:load_end]
    for needle in (
        "libc::flock(file.as_raw_fd(), libc::LOCK_SH)",
        "read_until",
        "MAX_JOURNAL_EVENT_BYTES + 1",
        "line.last().copied() != Some(b'\\n')",
        "potentially partial event",
    ):
        require(load, needle, "journal reader framing invariant")

    # Final-record framing is part of durability semantics: a JSON object with
    # no delimiter is treated as potentially partial and is never accepted.
    require(text, "ledger_rejects_valid_json_without_record_delimiter",
            "journal partial-write regression test")

    # Admission is an idempotency boundary, not merely an append operation.
    # The exclusive lock must remain held across replay lookup, transaction-id
    # collision detection, and the start-event append so concurrent identical
    # request IDs cannot both become New.
    admit_start = text.find("pub(crate) fn admit(")
    admit_end = text.find("\n    pub(crate) fn has_successful_transaction(", admit_start)
    if admit_start < 0 or admit_end < 0:
        fail("transaction admission boundary disappeared")
    admit = text[admit_start:admit_end]
    for required in (
        'open_ledger_file(',
        "libc::LOCK_EX",
        "load_locked_file(&mut file)",
        "append_locked_file(&file",
    ):
        require(admit, required, "atomic transaction admission invariant")
    if "self.load()?" in admit or "self.append(" in admit:
        fail("transaction admission regained an unlocked load/append TOCTOU")
    require(
        text,
        "concurrent_duplicate_request_id_has_exactly_one_new_admission",
        "duplicate request-id concurrency regression test",
    )

    # Terminal recording has the same TOCTOU shape: the completion replay
    # check and terminal append must be one atomic decision, otherwise
    # conflicting concurrent outcomes could permanently poison the journal.
    completion_start = text.find("pub(crate) fn mark_completed_with_image_artifacts(")
    completion_end = text.find("\n}", completion_start)
    if completion_start < 0:
        fail("transaction completion boundary disappeared")
    # Locate the next major impl boundary rather than relying on a single brace,
    # which keeps this ratchet tolerant of formatting changes.
    completion_end = text.find("\n}\n\nfn read_fingerprint_key", completion_start)
    if completion_end < 0:
        fail("transaction completion implementation boundary disappeared")
    completion = text[completion_start:completion_end]
    for required in (
        'open_ledger_file(',
        "libc::LOCK_EX",
        "load_locked_file(&mut file)",
        "append_locked_file(&file",
    ):
        require(completion, required, "atomic transaction completion invariant")
    if 'libc::O_CREAT' in completion:
        fail("terminal completion must never recreate a missing transaction journal")
    if "self.load()?" in completion or "self.append(" in completion:
        fail("transaction completion regained an unlocked load/append TOCTOU")
    require(
        text,
        "concurrent_conflicting_completions_commit_exactly_one_terminal_outcome",
        "conflicting completion concurrency regression test",
    )

    # The mutation lease remains descriptor-bound and non-blocking.
    lease_start = text.find("fn acquire_at(path: &Path)")
    lease_end = text.find("\n    }\n}\n\nimpl Drop for MutationLease", lease_start)
    if lease_start < 0 or lease_end < 0:
        fail("mutation lease boundary disappeared")
    lease = text[lease_start:lease_end]
    for needle in (
        "O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC",
        "libc::openat(",
        "libc::LOCK_EX | libc::LOCK_NB",
    ):
        require(lease, needle, "mutation lease descriptor invariant")

    # Event replay must remain fail-closed on unknown schema values.
    require(text, "ledger_rejects_unknown_event_kind",
            "unknown-event regression test")
    require(text, "transaction ledger is malformed",
            "malformed journal remains a terminal load error")

    # Durable transaction IDs remain CSPRNG-derived rather than time/pid-derived.
    random_start = text.find("fn random_operation_id()")
    random_end = text.find("\n}", random_start)
    random_body = text[random_start:random_end]
    require(random_body, "getrandom02::getrandom", "transaction identity CSPRNG")
    if "SystemTime" in random_body or "Instant::now" in random_body:
        fail("transaction identity regained clock-derived entropy")

    print(
        "transaction-ledger-boundary: PASS: "
        "typed events, framed records, shared/exclusive journal locking, "
        "descriptor persistence, and CSPRNG transaction identity are present"
    )


if __name__ == "__main__":
    main()
