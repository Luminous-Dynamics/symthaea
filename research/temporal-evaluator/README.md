# Symthaea Temporal Evaluator

This is a separate, evaluator-side Rust package for paired retain/evict effects and exhaustive small-subset upper bounds. It is not a candidate runtime dependency.

The current source provides:

- seeded, reproducible retain-first / evict-first order assignment using evaluator-held randomness (SplitMix64 for reproducibility only; not cryptographic secrecy);

- paired intervention checks that require identical episode/task/future-stream/model/initial-state/candidate-feature digests and identical byte/compute/refresh/observation budgets;
- componentwise wrong-belief, abstention and missing-output exposure differences;
- group interventions to expose non-additive memory interactions;
- exact subset enumeration up to 16 candidate items;
- separate best-known subsets for each exposure component and a Pareto frontier, with no weighted scalar objective;
- hard checks for duplicate IDs, unordered selections, incorrect bytes, budget overruns, incomplete subset enumeration and exposure totals exceeding episode duration.

The exact subset result is an evaluator-only clairvoyant upper bound, not a deployable candidate policy. It must never be exposed to the candidate or used to train against confirmatory outcomes.

Package separation is an architectural boundary only. It does not provide OS-level confidentiality. Before confirmatory use, complete #9437: separate OS identities and protected oracle storage, candidate sandboxing, evaluator-owned signing keys, signed receipts, replay protection and adversarial filesystem/descriptor tests.

No Cargo build, format, Clippy or test execution is claimed until a Rust toolchain is available.

## Rollback-resistant witness checkpoints

The `witness` module adds pure canonical checkpoint and witness-attestation types plus a strict verifier. Evaluator checkpoints bind the accepted receipt-chain head and previous checkpoint digest. Witness attestations bind the checkpoint digest to a fresh evaluator challenge. The verifier requires a Byzantine fault-tolerant quorum satisfying n >= 3f+1 and 2f+1 <= q <= n-f. Responses are authenticated before conflict detection: malformed/invalid signatures and unknown identities are ignored; stale challenges are non-votes counted once per identity; exact retransmissions are deduplicated; only conflicting, valid signatures for the current evaluator and challenge disqualify a witness. A forged packet claiming an honest witness ID cannot make that identity appear to equivocate.

This package does not implement a remote witness service, durable anti-equivocation database, TPM counter, challenge CSPRNG, RPC protocol, key rotation or recovery ceremony. A witness service must persist its high-water head outside the evaluator rollback domain and refuse to sign two different digests for the same evaluator epoch. The pure verifier is not proof that a deployed witness is independent or durable. See `docs/research/ROLLBACK_RESISTANT_RECEIPT_WITNESS_V0_1.md` and #9631.
