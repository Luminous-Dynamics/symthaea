# BASALT source-epoch commitments and seal integrity — 2026-10-31

## Thesis

A source epoch needs an immutable commitment over the exact ledger bytes, source identity, ordered event records, classified loss/discard ranges, observer identity, terminal sequence bounds and supplied close evidence. A plain file hash does not explain which source epoch was sealed; a per-source commitment makes the semantics explicit and independently recomputable.

This implementation is intentionally **integrity-only**. A self-computed hash is not an authenticated seal. A successful report must never promote signature validity, signer authorization, anti-rollback, close-coordinator qualification or observer-delivery qualification.

## V1 representation

The verifier accepts the existing event-delivery ledger and a strict `basalt-source-epoch-seal-v1` record.

It binds:
- exact SHA-256 digest of the raw ledger bytes (not a reserialized JSON representation);
- source, epoch and partition identity;
- observer binary and configuration digests;
- first and final attempted sequence;
- published event count and ordered event-chain root;
- classified-range count and ordered loss/discard chain root;
- close-fence digest;
- producer-quiescence evidence digest and monotonic timestamp;
- consumer-drain evidence digest and monotonic timestamp; every published event's observation timestamp must be no later than this drain fence;
- generation and previous-seal digest; `previous_seal_digest` must be explicitly present (null for generation 1), and later generations require a digest.

### Canonical chain encoding

The hash chains are deterministic, domain-separated binary encodings:
- every string is UTF-8 preceded by a 32-bit big-endian byte length;
- every sequence, count and monotonic timestamp is an unsigned 64-bit big-endian integer;
- each event leaf includes source/epoch/partition, sequence, observation timestamp and the 32 raw digest bytes;
- each disposition leaf includes source/epoch/partition, first/last sequence and stable one-byte disposition/reason tags;
- each next chain state hashes a distinct domain label, previous state and leaf digest;
- final roots bind the leaf count;
- the outer seal commitment uses its own domain label and binds every seal field, including the raw-ledger digest and predecessor link.

The raw ledger-byte digest deliberately avoids signing a normalized or canonicalized reconstruction of the ledger. A later DSSE adapter should sign the exact seal payload bytes and payload type using DSSE's pre-authentication encoding. DSSE authenticates the byte string and type; it does not require inventing a canonical JSON signature format.

## Trust dimensions kept separate

| Dimension | V1 result | What is still required |
|---|---|---|
| Exact ledger-byte binding | Verified | Exact source bytes |
| Event and range chain match | Verified | Independent recomputation |
| Seal self-commitment | Verified | Domain-separated field encoding |
| Signature verification | Always false | DSSE envelope and trusted public key |
| Signer authorization | Always false | Pinned source/profile policy and key lifecycle |
| Anti-rollback | Always false | Durable monotonic checkpoint or independently witnessed append-only log |
| Close procedure qualification | Always false | Independent quiescence/drain coordinator and positive/negative controls |
| Observer delivery qualification | Always false | Held-out transport fault injection and sensor-profile qualification |

A hash chain proves order and mutation sensitivity, not origin. A predecessor digest proves only a claimed link unless the verifier also has a trusted latest checkpoint. An attacker able to rewrite the ledger and recompute all unkeyed digests can create a new self-consistent record; a signature and trusted monotonic witness are still required.

## Industry comparison and integration

- **DSSE** is the appropriate envelope layer for signing exact payload bytes together with an explicit payload type. The key ID in its envelope is only a hint, not an authorization decision.
- **in-toto Attestation Framework** separates predicate, statement, envelope and bundle. Use a source-epoch predicate bound to a statement subject digest; keep BASALT-specific fields in the predicate.
- **SLSA provenance** describes build provenance and builder-controlled completeness. A runtime observer's source-epoch seal is not a SLSA build-provenance claim and must not be represented as one.
- **Sigstore bundles** can carry verification material, signatures and transparency-log/timestamp evidence. They may be useful for portable verification, but do not prove that a close coordinator actually stopped all producers or drained every consumer queue.
- **Linux BPF ringbuf** reservation failure has more than one cause, including NMI-context lock acquisition failure; query values are momentary snapshots. Neither a zero counter nor an occupancy snapshot is a completeness proof.
- **Falco** metrics must be explicitly enabled. Its aggregate drop counter includes documented component counters; subset counters must not be added twice.
- **Tetragon** supplies useful process-execution/lifecycle telemetry and exec correlation metadata, but event semantics do not establish transport completeness by themselves.

BASALT should complement these tools: sensors generate telemetry; BASALT verifies whether a specific source epoch is structurally complete, correctly bound and independently qualified.

## Benchmark plan

The versioned benchmark includes clean controls, raw-byte mutation, event-root mutation, range-root mutation, source substitution, terminal-high-water mutation, observer/close-fence substitution, generation-link corruption, close-order reversal, commitment mutation omission of the required predecessor field and event-after-drain inconsistency (14 scenarios).

Safety targets:
- 100% rejection of every seeded ledger/chain/close/commitment mutation;
- zero false signature, authorization, anti-rollback or qualification promotion;
- all negative and invalid attempts retained in the denominator.

Efficiency targets (not measured):
- p95 verification at or below 250 ms for 100,000 events on a pinned reference machine;
- peak RSS at or below 128 MiB at that scale.

The independent unit is one source epoch from a pinned host boot, observer binary/configuration, kernel identity and task-lineage run. Individual events and mutations from one epoch are not independent statistical samples. Report exact denominators and use separate held-out confirmation runs.

## Current limitations

V1 validates the close timestamps against the ledger capture window and requires every event's observation timestamp to precede the consumer-drain acknowledgement, but the evidence digests are references only; it does not retrieve or authenticate the underlying evidence bytes. The booleans in the ledger remain supplied assertions. The generation link is syntax-checked only; the predecessor field is mandatory but there is no durable checkpoint to verify its value against. There is no signature implementation in V1. Therefore all authenticity, authorization, anti-rollback and qualification fields are hard false.

## Exit criteria for authenticated epoch seals

1. DSSE envelope verification over exact payload bytes and payload type.
2. Pinned trust root, source/profile signer authorization, key rotation and revocation.
3. Independently maintained anti-rollback checkpoint or witnessed append-only log.
4. Close coordinator with independently observed producer quiescence and consumer drain.
5. Positive/negative controls for early seal, in-flight producer, late consumer, missing event, replay and coordinator restart.
6. Independent observer qualification remains a separate final gate.

## References

- DSSE: https://github.com/secure-systems-lab/dsse
- in-toto Attestation Framework: https://github.com/in-toto/attestation
- in-toto Statement v1: https://github.com/in-toto/attestation/blob/main/spec/v1/statement.md
- SLSA build provenance: https://slsa.dev/spec/v1.0/provenance
- Sigstore Bundle: https://docs.sigstore.dev/about/bundle/
- Linux BPF ring buffer: https://docs.kernel.org/bpf/ringbuf.html
- Falco metrics: https://falco.org/docs/concepts/metrics/
- Falco dropped events: https://falco.org/docs/troubleshooting/dropping/
- Tetragon process execution: https://tetragon.io/docs/use-cases/process-lifecycle/process-execution/
