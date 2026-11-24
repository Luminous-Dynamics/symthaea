# SFEB MON-005 — witnessed evidence and anti-rollback checkpoints

Date: 2026-11-24
Status: executable Node.js reference model, locally tested; distributed deployment and external-log integration remain unqualified.
Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/13515

## Thesis

Do not replace DBOS, Temporal, PostgreSQL, Kafka or established transparency logs. Their durable execution and append-only logging capabilities are real strengths. SFEB should bind an external action to portable evidence: tenant, source receipt, current authority revision, objective fence, stable operation identity, sink receipt, independent postcondition readback, Merkle inclusion and a witnessed checkpoint.

Do not collapse these states: authorized, effect observed, evidence pending external anchor, externally anchored. A valid signature only proves that a trusted key signed a checkpoint; it does not prove that the source, authorization or sink receipt is truthful. Validate each boundary independently.

## Executed prototype

Source: benchmarks/frontend/monitoring/mon005-witnessed-evidence-v1.mjs.gz
Report: benchmarks/frontend/monitoring/mon005-witnessed-evidence-report-v1.json
README/run commands: benchmarks/frontend/monitoring/mon005-witnessed-evidence-README-v1.md

- Node.js v22.16.0; direct source syntax check, compressed stdin syntax check and execution passed.
- 39/39 deterministic cases passed.
- 496 Merkle inclusion proofs verified across every leaf position for tree sizes 1 through 31.
- A 3-of-4 Ed25519 test-witness quorum certified two tenant-scoped checkpoints with 12 and 18 records.
- One unavailable witness still permits a 3-signature quorum; two signatures are rejected.
- Negative controls reject payload tampering, record reorder/substitution, cross-tenant substitution, invalid inclusion paths, unknown/duplicate signers, quorum downgrade, checkpoint mutation, changed-content challenge reuse, stale replay, rollback/gap, same-epoch fork, broken checkpoint linkage and rewriting the already witnessed prefix even after recomputing the local record chain.
- An identical checkpoint/challenge retry returns the same signature; a changed checkpoint using the same challenge is rejected.

These are reference-model results, not distributed fault-tolerance results. All witnesses run inside one process; state is in memory; keys are deterministic test fixtures; no PostgreSQL, independent network services, HSM/KMS, production key lifecycle, real sink or independent readback exists. Canonicalization is a strict local JSON subset, not full RFC 8785 JCS. The prototype recomputes the full prior prefix and does not yet implement RFC 9162 consistency proofs.

## Why 3-of-4

Two 3-of-4 quorums intersect in at least two signers. Assuming no more than one Byzantine witness, at least one intersecting signer is honest. If every honest signer durably and atomically refuses to sign two different heads for the same tenant/epoch, two conflicting quorums cannot both form. This argument is conditional: four keys in one process, one cloud account or one administrator are not independent witnesses. Production needs independently operated services, durable monotonic state, atomic compare-and-set or uniqueness constraints, and verified log reads.

## Standards and ecosystem fit

- RFC 9162 defines Certificate Transparency Merkle Tree Hash, inclusion proofs and consistency proofs. This prototype follows the leaf/node domain-separated hash shape and implements inclusion proofs; consistency proofs are next.
- RFC 8785 defines JSON Canonicalization Scheme. Use a conformant implementation and official/interoperability vectors before Rust/Go/JavaScript signatures are deployed across languages.
- Sigstore Rekor already provides a transparency-log ecosystem and signed tree heads. Evaluate publishing SFEB checkpoint digests there rather than building a competing general-purpose log.
- The Trillian project documentation marks Trillian as maintenance mode and directs new log operators to Tessera; evaluate current Tessera interfaces and operational status before choosing an integration.

References:
- RFC 9162: https://www.rfc-editor.org/rfc/rfc9162.html
- RFC 8785: https://www.rfc-editor.org/rfc/rfc8785.html
- Sigstore security model: https://docs.sigstore.dev/about/security/
- Trillian project documentation: https://google.github.io/trillian/

## Fair industry comparison

DBOS Go RunAsTransaction is the same-datasource atomic transaction baseline. DBOS Rust steps must remain a separate at-least-once baseline. Temporal supplies durable event history and replay. PostgreSQL transactional outbox is a native local atomicity/dispatch baseline. Rekor/Tessera/CT-style logs supply transparency inclusion and append-only commitments. Compare native and SFEB-augmented arms using the same event trace, sink, idempotency key, payload sizes, failure schedule, hardware and durability settings.

SFEB's incremental question is: can an independent party prove which tenant-authorized operation occurred and that the sink's postcondition was read back, while detecting a stale authority, tenant substitution or rewritten history? No throughput comparison was run in MON-005.

## Proposed benchmark gates (not achieved)

- At least 1,000,000 adversarial attempts per release candidate across tamper, tenant substitution, stale authority, replay, rollback, fork, missing readback, bad signature, compromised witness and witness unavailability.
- Zero accepted unauthorized/cross-tenant effects, accepted rewritten witnessed prefixes, rollback checkpoints or conflicting 3-of-4 checkpoint heads under the stated one-Byzantine assumption.
- 100% of outcomes labelled externally anchored independently verify from source/authority/execution/outcome receipts, inclusion proof, checkpoint signatures and trusted checkpoint chain.
- Unknown sink outcomes remain unknown until authoritative readback; never mint a replacement action key to bypass reconciliation.
- Proposed <=10% p95/p99 overhead on matched native external-action workloads. Measure and report synchronous effect latency separately from asynchronous time-to-anchor.
- Sweep batches of 1/10/100/1,000/10,000 records, checkpoint intervals 250ms/1s/10s, 1/8/32/128/512 producers, and witness availability 1/4 through 4/4. Report throughput, p50/p95/p99 append/proof/quorum/anchor latency, bytes per proof, storage growth, signature CPU, recovery, witness egress and cost per independently verified/anchored action.

## Next qualification steps

1. Replace local canonicalization with a conformant RFC 8785 implementation and run cross-language test vectors.
2. Implement RFC 9162 consistency proofs and proof-size scaling tests.
3. Build four separate Rust witness services with durable per-tenant/epoch state, atomic uniqueness/CAS, independent failure domains and non-exportable production keys.
4. Test concurrent conflicting-head races, crash/restart, backup rollback, key rotation/revocation and one compromised witness.
5. Integrate one real idempotent/conditional-write external sink, enforce objective/policy fencing at the sink, and perform independent readback.
6. Publish SFEB checkpoint digests to an established transparency log and monitor split views using independent gossip/witness operators.
7. Run paired DBOS Go, DBOS Rust, Temporal and PostgreSQL outbox arms before making any SOTA or throughput superiority claim.