# Attestation acceptance-ledger current head

This crate separates three claims that a caller-supplied acceptance ledger cannot establish by itself:

1. **head authenticity** — a reviewed authority signs one exact ledger ID, acceptance revision and acceptance-record digest;
2. **observed append-only continuity** — a tracker begins at revision 1 and rejects rollback, gaps, collisions, broken previous-head links and issuance-time regression; and
3. **exact-use currentness** — a separately scoped authority signature asserts that the exact tracked head is current for one exact challenge and use-time.

The acceptance revision is the head sequence; there is no parallel counter.

A successful `CurrentAcceptanceLedgerHead` remains a bounded authority assertion. It does not establish trusted time, platform correctness, or physical authority. A final provider bridge must still bind this exact revision/digest to the ledger-relative TPM+IMA anchor produced by the stateful acceptance layer.
