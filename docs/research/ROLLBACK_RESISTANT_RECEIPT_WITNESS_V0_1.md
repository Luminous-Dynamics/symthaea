# Rollback-Resistant Receipt Checkpoints and Witness Quorum (v0.1)

**Tracking:** EVAL-ORACLE-003 / #9631  
**Evaluator implementation:** `research/temporal-evaluator/src/witness.rs`  
**Receipt implementation:** `research/evaluator-isolation/src/receipt.rs`  
**Status:** canonical checkpoint and quorum-verification primitives authored; no Rust build, witness service, TPM adapter or rollback experiment executed

## 1. The rollback problem

The local replay ledger in #9630 atomically consumes run IDs, nonces and per-key monotonic counters during ordinary operation. However, a filesystem snapshot restore can roll back the ledger and its high-water mark together. A local hash chain does not fix this if its head is stored in the same rollback domain.

The protection requires an independent state anchor. Candidate choices include a TPM 2.0 NV monotonic counter/sealed state, independent durable remote witnesses, or a combination. Each mechanism has a different availability, provisioning, recovery and trust model. The current source implements only canonical checkpoint and witness-response verification; it does not implement network or TPM I/O.

## 2. Canonical evaluator checkpoint

`LedgerCheckpointPayload` is serialized as a versioned, domain-separated little-endian binary record and signed by the evaluator's Ed25519 key. It binds:
- evaluator ID and evaluator signing-key ID;
- checkpoint epoch;
- highest accepted receipt sequence and monotonic counter;
- accepted receipt count;
- previous checkpoint digest;
- receipt-chain head digest;
- issue time.

The checkpoint signature is verified using the externally provisioned evaluator public key and strict Ed25519 verification. The receipt/oracle bytes are not exposed in the checkpoint; only commitments are used.

The evaluator should publish checkpoints only after the local receipt ledger and all included receipt records have been durably synced. A checkpoint's prior digest must match the previously accepted checkpoint. The witness adapter must itself verify the evaluator checkpoint signature and previous-head continuity before attesting.

## 3. Byzantine witness quorum

A plain majority is insufficient if a trusted witness can be Byzantine and equivocate. The verifier therefore requires the standard Byzantine quorum shape:

[
n ge 3f+1,qquad 2f+1 le q le n-f
]

where:
- (n) is the configured witness set size;
- (f) is the maximum Byzantine witness count;
- (q) is the required matching attestation threshold.

The included test policy uses (n=4), (f=1), (q=3). This ensures any two accepted quorums intersect in at least (f+1) identities, assuming the configured witness identities are independent and each honest witness durably refuses to sign two different heads for the same evaluator epoch.

This is not a generic “majority of whoever answered” rule. The trusted witness registry has a fixed size; missing responses reduce availability, not the threshold.

## 4. Fresh challenge and witness attestation

Every witness attestation binds:
- witness identity (derived from its allow-listed Ed25519 public key);
- evaluator ID;
- a fresh 256-bit evaluator challenge;
- checkpoint epoch;
- checkpoint digest;
- previous checkpoint digest;
- witness observation timestamp.

The evaluator must generate an unpredictable fresh challenge using an OS CSPRNG for each quorum query, persist it with the request, and reject zero/reused challenges. The pure verification API takes the challenge as input. A response with a stale challenge is a non-vote and cannot abort an otherwise valid quorum; it is counted as stale. Challenge generation and remote transport are not implemented in this prototype.

A witness service must:
1. authenticate the evaluator request;
2. verify the signed checkpoint and previous-head continuity;
3. check its own durable high-water epoch/head outside the evaluator's rollback domain;
4. refuse to sign a second digest for the same evaluator epoch;
5. sign the checkpoint digest and the evaluator's fresh challenge;
6. persist its accepted head and attestation state before acknowledging.

If the witness cannot durably commit its state, it must not attest.

## 5. Verifier behavior

`verify_checkpoint_quorum`:
- validates a strict BFT witness configuration;
- verifies the local evaluator checkpoint signature;
- verifies candidate responses against allow-listed keys using strict Ed25519;
- authenticates responses before deduplication/equivocation classification, so a forged packet claiming an honest witness ID cannot disqualify that honest witness;
- ignores unknown, malformed or invalidly signed responses rather than allowing unauthenticated input to poison a valid quorum;
- deduplicates byte-for-byte identical current-challenge retransmissions and excludes a witness only when it supplies conflicting, validly signed responses for the same evaluator and current challenge;
- treats stale-challenge responses as non-votes, counted once per witness identity, rather than aborting the quorum check or allowing replay floods to inflate the counter;
- groups attestations by evaluator epoch, checkpoint digest and previous digest;
- requires (q) matching witnesses for acceptance;
- detects local rollback only when a quorum confirms a higher epoch;
- rejects same-epoch conflicting heads only when a conflicting quorum attests them;
- tolerates a single faulty witness response without letting it override a valid current quorum;
- fails closed when a current quorum is unavailable or the witness set is stale.

A single higher-epoch or conflicting response cannot override a three-of-four quorum. This is intentional Byzantine fault tolerance, not a reason to accept an old local head when no current quorum exists.

## 6. Failure, availability and recovery

- **Witness partition/outage:** if the current quorum cannot be reached, checkpoint acceptance is blocked. Do not silently fall back to local-only replay protection.
- **Local behind a witnessed quorum:** reject and require recovery from a trusted backup plus reconciliation to the highest witnessed head.
- **Witnesses behind the local state:** block. The evaluator may have advanced without checkpointing or the witness set may have lost state; this is not automatically safe.
- **Same-epoch conflicting quorum:** reject, preserve both signed artifacts and enter incident review.
- **Key rotation:** requires overlapping old/new key authorization and explicit witness registry update; not implemented.
- **TPM reset/replacement:** requires an administrative re-enrollment ceremony with signed evidence and independent witness confirmation; not implemented.
- **Witness quorum migration:** needs a transition certificate acknowledged by the old quorum and the new quorum; not implemented.
- **Backup restoration:** never reset epoch/counter to match a local snapshot. Reconcile to an externally witnessed checkpoint first.

## 7. Tests authored (not executed)

Ten source-level test functions cover:
1. three-of-four confirmation of an exact current checkpoint;
2. stale witness reporting while a current quorum succeeds;
3. rollback detection when a witness quorum has a newer epoch;
4. same-epoch conflicting quorum rejection;
5. insufficient quorum and weak-policy rejection;
6. replay of old attestations yields no quorum under a fresh challenge;
7. one stale-challenge response cannot block a valid current quorum;
8. forged invalid-signature response under an honest witness ID, unknown identity, and valid equivocation handling cannot poison a valid quorum;
9. one Byzantine higher-epoch response cannot override a current quorum;
10. one Byzantine conflicting-head response cannot deny a current quorum.

No Cargo compilation, rustfmt, Clippy, Rust test execution, remote witness experiment or snapshot/restore campaign has run.

## 8. Claim ceiling

The source verifies signatures and quorum responses under an explicit trust/fault model. It does not prove that remote witness state is durable, independent or correctly operated. The real service must implement durable anti-equivocation and checkpoint continuity; a Byzantine quorum is only as strong as the independence of its witness identities, key custody and storage. No rollback-resistance claim is made until snapshot, crash, partition, equivocation, key-rotation and recovery tests run against deployed witnesses or TPM hardware.
