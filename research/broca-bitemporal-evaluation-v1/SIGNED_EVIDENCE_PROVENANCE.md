# Signed Evidence Provenance v1

**Status:** research-only Rust API. The existing Broca JSONL worker does not use this API yet. No compilation or unit-test run is claimed.

## Why a source signature is not enough

Three separate claims must be distinguished:

1. **Source attribution:** which registered source key signed the evidence payload?
2. **Trusted acceptance time:** when did the ingest service actually accept that signed payload?
3. **Policy authorization:** was that source allowed to make this claim, and how was it ranked under the policy applicable at the relevant time?

A source signature authenticates bytes and key possession. It does not prove the statement is true, authorize that source for an attribute, or provide a trustworthy transaction timestamp. In particular, an event signer must not be able to backdate its recorded_at field to smuggle an event into an earlier as-known query.

## Implemented research API

The branch adds crates/domains/symthaea-broca/src/memory_attestation.rs, exported from the Broca crate. The module uses the workspace's Ed25519 implementation through ed25519-dalek.

### Two signatures, two purposes

- **Source signature:** binds a domain-separated, normalized event payload to signer_key_id and verifies that the key is registered for the event's claimed source and event-signing role.
- **Trusted-ingress receipt:** a separately provisioned ingestion key signs the event ID, source signer ID, digest of the source signature, and accepted_at. The receipt signer must have the separate ingestion-receipt role.

The verified event's transaction time is overwritten with the signed accepted_at value before it enters PolicyPinnedLedger. The source-provided timestamp is therefore not used as the query cutoff time. The source key must also be valid at the trusted acceptance time, not merely at the timestamp the source claims.

SignedAuthorityPolicy separately verifies policy changes under an explicitly provisioned policy-signing key. Policy records are normalized and sorted for signing so a harmless reorder of policy sources does not change the signature payload.

EvidenceTrustStore::verify_batch verifies the policy envelopes and event envelopes. The resulting VerifiedLedgerBatch has private input fields, exposes policy/event digests, and can be converted to PolicyPinnedLedger only after verification.

### Authored Rust tests

The module includes 19 authored unit tests covering:
- valid source, policy and ingress signatures;
- source-payload and receipt-time tampering;
- source/key identity mismatch and unknown key rejection;
- role separation among policy, source-event and ingestion-receipt keys;
- source key validity checked at actual receipt time;
- source-claimed time cannot override trusted ingress time in a query;
- canonical policy source ordering;
- strict schema rejection;
- verified batch to ledger constructor flow; and
- malformed key, duplicate key ID and missing signing-role checks.

Run when the Rust workspace is available:

~~~sh
cargo test -p symthaea-broca memory_attestation --lib
cargo test -p symthaea-broca memory_governance --lib
~~~

These are recommended commands, not test results. The source file was added by the GitHub repository tool. This execution environment does not have Cargo/rustc or a full checkout, so compile behavior remains unverified.

## Threat model and remaining limitations

This design is safer than trusting self-asserted source names/ranks, but it is not a complete production trust system.

- **A signer can lie about the world.** A valid signature attributes a statement to a key; it does not make the statement true.
- **The trust store is a root of trust.** Public-key configuration must come from an authenticated deployment mechanism, not from the same event feed it validates.
- **The ingress key is privileged.** A compromised ingress key can sign false acceptance times. Production should anchor append-only receipts in a durable, monotonic log with rollback detection; key rotation/revocation needs an authenticated policy.
- **Policy time needs anchoring too.** The policy signer controls the signed policy's recorded timestamp. Production policy distribution should provide its own trusted receipt/effective-time rule and reject rollback.
- **Replay and exactly-once effects.** Event IDs are unique within a ledger, but distributed deduplication, monotonic sequence numbers, checkpointing and crash-safe log persistence are not implemented here.
- **Deletion and reinstatement.** Retractions preserve audit history. Real deletion/privacy obligations and retraction reversal require explicit, separately reviewed semantics.
- **No live worker integration.** The JSONL worker continues using its existing behavior until a separately reviewed integration change wires in a verified ledger and authenticated trust-store provisioning.

Do not treat hashes as signatures or signatures as truth. Do not promote this research module to production until the ingress log, signer registry, key rotation, policy anchoring and replay/rebuild semantics are implemented and reviewed.

## Research comparison

Recent work makes the target more specific than generic retrieval improvement:

- **TrustMem (EMNLP Findings 2026)** reports gains in memory consolidation reliability and reductions in omission, corruption and hallucination transitions relative to its evaluated baselines. Broca should compare its explicit admission/provenance invariants against trust-aware consolidation rather than claiming to replace it.
- **TAME / Trust-Memevo (EMNLP 2026)** studies trustworthiness degradation as agent memories evolve. The relevant test for Broca is whether trust and temporal invariants survive many benign updates as well as deliberate poisoning.
- **MemEvolve / EvolveLab (ICML 2026)** evolves memory architecture itself. Broca's differentiating hypothesis is that an optimizer may evolve storage/retrieval strategies inside hard evidence, authority, time and scope constraints.

These papers' reported values come from their own protocols and are not directly comparable with Broca's synthetic tests. Relevant links:
- https://aclanthology.org/2026.findings-emnlp.2419/
- https://aclanthology.org/2026.emnlp-main.2100/
- https://proceedings.mlr.press/v306/zhang26fa.html

### Future benchmark extension

Use two independent axes:
1. **Trust enforcement:** correct signer attribution, trusted time, authorization scope, retraction and anti-rollback behavior under adversarial mutations.
2. **Task utility:** official MEME and COLLABMEM tasks, then paired LongMemEval-V2/AgentRunbook-C and STATE-Bench runs.

Report the fraction of forged-source, untrusted-key, tampered-payload, backdated-event, unauthorized-retraction and replay attacks blocked, together with false rejection rates for valid events. A system that rejects everything is not useful; coverage and task success must accompany attack rejection. Keep task benchmark scores separate and compare only under their official protocols.
