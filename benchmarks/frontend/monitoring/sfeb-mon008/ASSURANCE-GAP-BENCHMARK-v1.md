# MON-008 assurance-gap benchmark protocol

**Status:** preregistered research protocol proposal. No end-to-end industry comparison has yet been run.

## Research question

After a consequential external action completes, can an auditor independently establish what happened, under whose authority, against which resource and policy, with what source evidence, and whether the transparency view is consistent—without trusting the original orchestrator, database, or log operator?

The primary outcome is **portable independently audited external actions per second**, accompanied by assurance completeness and failure outcomes. Do not substitute workflow starts, completed workflow records, Merkle hashes, signatures, or log registrations for externally observed and independently audited effects.

## Baseline arms

Run the same application semantics, hardware, database durability, payloads, sink API, concurrency, retry policy and fault schedule across:

1. PostgreSQL transactional outbox + idempotent receiver.
2. DBOS durable transaction/step with the documented external-effect idempotency pattern.
3. Temporal Activity with stable operation identity and recorded event history.
4. SCITT / RFC 9942 receipt + RFC 9943 registration-policy path.
5. Sigstore Rekor transparency path.
6. SFEB assurance adapter over each of PostgreSQL, DBOS and Temporal.
7. Native sink control: direct conditional write and independently authorized readback.

Keep each component's natural durability and retry semantics. Report a no-op workflow tier separately; never compare it directly with a signed external mutation + readback + transparency proof workload.

## Workload tiers

| Tier | Operation | Main question |
|---|---|---|
| W0 | no-op durable workflow | raw orchestration ceiling only |
| W1 | one durable DB mutation | transaction/outbox cost |
| W2 | sink-enforced conditional mutation with stable operation ID | actual effect and replay safety |
| W3 | W2 + separately authorized readback + signed action statement | effect qualification |
| W4 | W3 + transparency registration + authenticated checkpoint + inclusion/consistency proof | anchoring cost |
| W5 | W4 + independent witnesses, key history and historical policy context | non-equivocation and trust continuity |
| W6 | W5 + loss/compromise of original operator DB and credentials + portable offline audit | operator-independent auditability |

Exercise 1 KiB, 16 KiB and 256 KiB evidence payload classes, reporting actual encoded sizes and proof sizes.

## Failure and adversarial corpus

Use deterministic fault seeds and preserve every attempt and observed sink state. Inject crashes and partitions before/after authorization, revocation, sink commit, acknowledgement, readback and transparency registration. Exercise duplicate and delayed messages; lost acknowledgements; worker restart; DB failover and snapshot restore; clock skew; stale fences; authority/key revocation; tenant/resource substitution; read-replica staleness; reader compromise; malformed and forged receipts; key rotation; root rollback; same-size forks; malformed Merkle paths; stale checkpoints; split views; witness outages and correlated witness compromise.

A timeout after a possible sink commit must stay indeterminate until the same operation identity is reconciled. A second mutation is prohibited while the first outcome is unknown. Application-side prechecks do not count as sink enforcement.

## Metrics

Performance: intent admissions/s, sink-accepted effects/s, independently effect-verified actions/s, externally anchored actions/s, offline-audited actions/s, p50/p95/p99 latency by tier, recovery/reconciliation time, time-to-anchor, proof bytes, evidence bytes, memory, CPU-seconds, IOPS, network transfer and total cost per qualified effect.

Assurance: duplicate effects, post-revocation writes, cross-tenant acceptances, stale-fence writes, false accept/reject counts, indeterminate rate and age, proportion verified with operator DB unavailable, proportion audited with no original credentials, time-to-detect checkpoint equivocation, policy/key history reconstructibility and witness failure-domain diversity.

Publish raw samples, exact machine and software versions, durability settings, fault seeds, sink implementation, source and verifier versions, concurrency, payload distribution, retries, confidence intervals and analysis scripts. Use at least five independent performance runs per point.

## Proposed gates (targets, not achieved)

- Zero accepted unauthorized, replayed, cross-tenant or stale-fence effects in at least 1,000,000 seeded negative attempts per release candidate. This is a campaign result, not a proof of zero risk.
- Zero duplicate actual sink mutations under same-operation replay.
- Two independently maintained implementations accept the same official and generated proof/receipt corpus; include malformed mutations and negative controls.
- 100% of bundles classified as qualified can be audited offline with original operator services and secrets unavailable.
- Every root/policy transition in a declared retention interval can be independently reconstructed from a retained authenticated history.
- All unresolved outcomes stay explicit; no unknown result is converted to success or retried with a fresh operation ID.
- Report overhead separately for local crypto, networked witnesses, transparency, sink mutation, independent readback and offline audit. Only compare each to a same-semantics baseline.

## Standards and integration discipline

Build on established building blocks rather than inventing a competing transparency service: RFC 9162 Merkle inclusion/consistency proofs; RFC 9942 COSE receipts; RFC 9943 Signed Statements and historical registration-policy/trust-anchor collateral; and existing systems such as Rekor. The MON-008 custom Ed25519 envelope is a research test profile only, not a COSE wire profile and not RFC 9942 interoperability.

## Paradigm-shift contribution: the Assurance Gap Challenge

Execute an external action, then deliberately remove trust in the operator, its database, and its credentials. Give a fresh auditor only a portable evidence bundle and independently configured trust roots. Ask the auditor to reconstruct authority, source identity, exact action, sink result, independent readback, checkpoint authenticity, append-only history and historical policy context.

Score each claim as verified, missing, contradicted, stale or indeterminate; measure false acceptance, detection time, audit time, bytes and cost. Publish the harness so DBOS, Temporal, outbox, SCITT/Rekor and new assurance systems can enter on equal terms.

The field-level shift is from **“how many workflow records were produced?”** to **“how many consequential external actions remain independently auditable after the operator is no longer trusted?”**
