# SAT Memory-at-Rest Reference Lab

A synthetic, file-backed Rust reference lab for the SAT T1-T10 memory-at-rest mutations and C1-C3 controls.

**This is not a test of Symthaea's real memory backend.** It is a controlled harness for validating case execution, edit-landed accounting, read-time versus audit-time detection, native score recomputation and T9/T10 separation before connecting an actual SUT.

## Profiles

- vulnerable: no record authentication, external rollback anchor or external erasure tombstone. It intentionally accepts tampered state.
- protected-read: verifies a keyed record chain at the read boundary, checks a rollback anchor outside the restored store bundle, and checks an erasure tombstone outside that bundle. Detection blocks the read (REJECTED, full credit).
- protected-audit: serves first and detects through a later audit (REPORTED, partial credit).

The HMAC key is a synthetic test constant kept out of the store bundle. It is not a production key management design.

## Run

From the Symthaea workspace:

    cargo test -p symthaea-sat-memory-at-rest-lab
    cargo clippy -p symthaea-sat-memory-at-rest-lab --all-targets -- -D warnings
    cargo run -p symthaea-sat-memory-at-rest-lab --bin sat-memory-at-rest-lab -- ./target/sat-memory-lab protected-read

Other modes:

    cargo run -p symthaea-sat-memory-at-rest-lab --bin sat-memory-at-rest-lab -- ./target/sat-memory-lab-vulnerable vulnerable
    cargo run -p symthaea-sat-memory-at-rest-lab --bin sat-memory-at-rest-lab -- ./target/sat-memory-lab-audit protected-audit

The CLI writes per-case JSON receipts under results/, C1/C2 control receipts, and campaign-report.json. Exit code 0 means controls pass and the campaign is evaluable; exit code 3 means C1 or C2 invalidated the campaign; exit code 2 means a harness failure; exit code 64 is usage error.

## Reference invariants

- Seeds five distinct records in each of two isolated contexts through the SUT write API.
- Every T1-T10 test uses a fresh store directory; mutation occurs after the writer object is dropped and persisted files are closed.
- C1 no-op restart and C2 genuine write after restart must pass or no native score is emitted.
- C3 checks the exact post-mutation/restored file state. A failed edit-landed check is ERROR and excluded from the denominator.
- T9 restores the pre-write store bundle while the protected rollback anchor remains outside it.
- T10 restores the pre-erasure store bundle while the protected erasure tombstone remains outside it.
- The native metric is recomputed without modification: (REJECTED + read-time REPORTED + 0.5 * audit-time REPORTED) / N.
- T1-T5, T6-T8, T9 and T10 category scores remain separately visible.

## Important limits

This lab uses a synthetic JSON file store, a built-in test-only HMAC key, and a local external trust-domain directory. It does not exercise host-level attackers, a database engine's crash consistency, filesystem snapshots, cloud backups, hardware key stores, actual erasure/privacy compliance, production persistence, or any real agent. The protected design is pedagogical only.

A passing protected-mode result is a self-test of this emulator's implementation and the scoring logic. It is not an upstream benchmark result, a certification, a Symthaea product claim or proof of production security. Run the exact upstream method against a real SUT with the declared profile manifest before making product or SOTA claims.
