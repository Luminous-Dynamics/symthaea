# Acceptance-ledger head currentness

This crate authenticates and currentness-qualifies the exact head of an append-only `AttestationAcceptanceLedger`.

It deliberately separates three claims: signed head authenticity, contiguous anti-rollback observation, and challenge-bound exact-use currentness. A full supplied ledger is structurally replayed before tracking, and a later tracked head must still contain the previously observed authoritative head at the same ledger revision.

The output capability is non-serializable. It does not establish trusted wall-clock provenance, persistence of the local tracker, TPM correctness, readiness, or physical authority.
