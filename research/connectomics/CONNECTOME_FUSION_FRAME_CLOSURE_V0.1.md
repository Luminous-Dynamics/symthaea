# Connectome Fusion-Frame Closure V0.1

**Status:** research protocol; executable qualification required  
**Issue:** [#10433 — SCI-CONNECTOMICS-074](https://github.com/Luminous-Dynamics/symthaea/issues/10433)  
**Parents:** [#10414 provenance-completeness sampling](https://github.com/Luminous-Dynamics/symthaea/issues/10414), [#10416 sampling verifier](https://github.com/Luminous-Dynamics/symthaea/issues/10416), [#10406 evidence-dependence verifier](https://github.com/Luminous-Dynamics/symthaea/issues/10406)  
**Date:** 2026-11-04  
**Claim ceiling:** bounded event-population closure within named, qualified source scopes; no universal logging-completeness or biological-independence claim

## 1. The missing precondition

A statistically valid sample cannot detect a unit that had zero probability of entering its sampling frame.

The provenance-completeness audit in #10414 samples claimed-independent evidence-pair assertions. That is useful only after the assertion frame has been shown to cover the fusion operations that actually contributed evidence to the declared scientific claims.

A producer can omit an evidence record before frame construction. If the audit frame is derived only from that producer's manifest, the omitted dependency is absent from both the frame and the sample. Exact hypergeometric bounds then describe the wrong finite population.

V0.1 therefore introduces a separate precondition:

> Reconcile the frozen audit frame against independently governed records of actual evidence consumption, qualify the event-source coverage, and only then authorize provenance sampling.

This is not a second provenance-graph validator. It verifies the population boundary on which the sampling design relies.

## 2. Scope and unit of account

The target population is the finite set of evidence-dependence assertions actually used by fusion operations that contributed to a specified set of claims, certificates, results or downstream decisions during a declared interval and release scope.

The unit is a **fusion-use assertion**, not a model, artifact, file or arbitrary pair of objects.

A fusion-use assertion identifies:
- the operation and its stable execution identity;
- the consuming claim/query/certificate/result;
- the evidence records actually supplied to the operation;
- the fusion policy and verifier identities;
- the evidence-dependence bundle and source-release revisions;
- the exact projection from the operation's evidence set to the dependence assertions required by the fusion policy.

Do not form a Cartesian product of unrelated artifacts. The policy defines which pairs/relations must be represented for the actual evidence set.

The scope must be explicit: query families, claim types, release IDs, start/end times, event-time convention, ingestion-time convention, and the systems through which claims can be issued.

## 3. Three inputs, three different roles

### 3.1 Producer declaration

The producer's provenance bundle states which dependencies it knows and which independence assertions it relies upon. It is an input to reconcile, never the sole authority for frame membership.

### 3.2 Execution-of-record stream

A qualified execution source records actual invocations that consume evidence for claim/fusion operations. It must expose a bounded interval, unique event/use identities, sequence or equivalent completeness controls, retention coverage and source custody.

Examples include a fusion executor's append-only operation journal or a separately governed execution receipt service. A mutable application log with unknown retention is not sufficient.

### 3.3 Independent claim/certificate record

A second source records the issued claim, certificate or decision and its exact evidence-use references. It should be governed separately from the producer declaration and, where possible, independently from the execution stream.

Two logs are not independent merely because they have different names, APIs, storage paths or operators. Record shared writers, event buses, databases, ingestion jobs, retention, identity providers, clocks, administrative control and backup/restore paths. A shared upstream failure mode must remain visible.

If no source can enumerate actual use independently of the producer declaration, return NOT_ESTIMABLE.

## 4. Canonical evidence-use identity

The schema must distinguish an event's local ID from its cross-source identity.

A canonical use key is computed from a versioned, domain-separated encoding of:
- scope digest;
- query and claim IDs;
- fusion-operation type and operation ID;
- policy digest and verifier digest;
- source-release identities;
- sorted, typed evidence identities actually consumed.

Evidence identity must include immutable revision or content commitments. A display name, model family label or mutable database row ID is not sufficient.

The event identity algorithm, field normalization, Unicode handling, ordering and digest algorithm must be pinned. Hashes authenticate byte identity under that encoding; they do not prove that the event source recorded every operation.

Retries must be distinguishable from duplicate semantic use. The protocol must preserve both the physical execution-event identity and the idempotency/semantic-use key.

## 5. Bounded event-stream closure

Each source export must bind:

- source ID, source role, custodian and extraction identity;
- precise time/query/release scope;
- lower and upper sequence positions or equivalent completeness watermarks;
- event count and unique canonical-use count;
- ordered event digest (and optional Merkle inclusion material);
- duplicate, gap, retry, deletion, correction and late-arrival accounting;
- retention start/end and known access limitations;
- extraction software/configuration digest;
- shared-control and common-ancestry declarations.

Counts alone do not prove closure. A sequence gap can be hidden by a matching total count; a Merkle root can faithfully commit to an incomplete export. The auditor must validate the source's coverage contract and export provenance, not just recompute a digest.

For every required source, qualify whether it is complete for its declared source role and scope. If the sources have different coverage roles, pin the expected crosswalk relation and its coverage assumptions rather than forcing naive set equality.

## 6. Reconciliation algorithm

Given:
- producer assertion set P;
- qualifying execution-use population E;
- qualifying claim/certificate population C;
- policy projection function pi that maps actual consumed evidence to required dependence assertions;
- frozen candidate audit frame F;

perform the following steps.

1. Verify that all inputs use the same declared query/release/time scope or have an explicit, versioned scope mapping.
2. Validate source watermarks, retention, sequence continuity, unique event identities and source-export digests.
3. Validate each event-to-claim/certificate crosswalk using immutable IDs and policy digests.
4. Reconstruct the required assertion population R = pi(E) using the pinned fusion policy. Preserve operation multiplicity where distinct executions matter.
5. Reconcile execution events to claim/certificate records. Any missing or extra use must be explained by an explicit allowed state (for example, a failed invocation that never contributed to a claim).
6. Compare R with F and P. Preserve both anti-join directions and all multiplicities:
   - required assertions in R absent from the audit frame F;
   - assertions in F with no qualifying actual use in R;
   - actual uses present in E but absent from the claim/certificate record C;
   - claim/certificate uses in C absent from E.
7. Recompute exact counts and canonical digests. Never trust producer-supplied totals or a claimed 'closed' flag.
8. Emit a closure receipt bound to every source, scope, policy, projection, frame and reconciliation digest.

The output must not silently discard unresolved records or repair a frozen frame in place. A detected gap requires a corrected source export, a newly frozen frame and a new sampling commitment.

## 7. Fail-closed dispositions

The only dispositions are:

- `FRAME_CLOSED_WITHIN_DECLARED_SCOPE`: all required sources passed their named coverage contracts; the event/certificate reconciliation is exact; and the audit frame equals the policy-derived required assertion population for the bound scope.
- `FRAME_GAP_DETECTED`: at least one qualifying actual evidence use is absent from the audit frame, or required reconciliation shows an omitted use.
- `RECONCILIATION_CONFLICT`: sources disagree, mappings conflict, or records cannot be reconciled under the pinned contract.
- `SOURCE_COVERAGE_UNKNOWN`: a required source's retention, watermark, completeness or access scope is not established.
- `NOT_ESTIMABLE`: the available sources or policy do not support a bounded target population.

Only FRAME_CLOSED_WITHIN_DECLARED_SCOPE may authorize #10414 sampling. Its receipt is valid only for its exact query/release/time scope, event sources, projection policy and frame digest.

A closed frame does not mean every biological relationship was logged. It means the actual fusion uses represented by the qualified sources are covered by the frame within the receipt's boundary.

## 8. Change, correction and temporal semantics

A new, corrected, deleted or backfilled event after frame freeze invalidates closure for the affected scope. A new frame digest and sample commitment are required.

Define whether the audit uses event time or ingestion time. Late-arrival windows must be bounded and explicit. If a source cannot provide a finalization watermark for the requested interval, the scope is not closed.

A correction is a new event referencing the superseded event; do not rewrite history in a way that makes the prior audit frame appear complete retrospectively.

Closure receipts must be dependency-addressable. A change to a source export, event mapping, fusion policy, projection function, retention declaration or frame invalidates every affected receipt in its transitive dependency closure and no unrelated receipt without a path.

## 9. Adversarial challenge corpus

At minimum:

1. The claim engine consumes evidence B, but the producer manifest exports only A.
2. One execution stream omits an event that a separately governed claim/certificate source records.
3. Apparent sources share a hidden upstream writer or event bus.
4. A high-materiality operation is filtered before frame construction.
5. A sequence gap is masked by an unchanged aggregate count.
6. Duplicate event IDs mask an omitted unique use.
7. A backfilled use arrives after frame freeze.
8. Retention expiry removes old, high-risk events.
9. Evidence IDs collide or alias across releases/model revisions.
10. A frame assertion has no corresponding contributing fusion use.
11. An execution event has no corresponding claim/certificate record.
12. A query/certificate is issued through an unregistered alternate execution path.
13. A retry creates duplicate physical events but one semantic use.
14. Clock skew places an event on the wrong side of the interval boundary.
15. A Merkle root is valid for an incomplete export.
16. Source access is unavailable and must produce SOURCE_COVERAGE_UNKNOWN.
17. The fusion-policy projection omits a required pair relation.
18. A policy or verifier changes between execution and frame freeze.
19. The primary event source and purported independent export share a restore/retention failure.
20. A new evidence-use route is added without registering it in the closure source inventory.

Evaluator-only truth must remain outside frame construction and runtime reconciliation until the receipt is frozen.

## 10. Metrics and qualification goals

Report distinct measures; do not compress them into a single assurance score:

- false FRAME_CLOSED rate under enumerated omission attacks;
- recall for deliberately omitted use events when at least one qualified independent source records them;
- exact precision/recall of event-to-claim and event-to-evidence reconciliation;
- orphan-frame assertion rate;
- duplicate, gap, late-event and retention-loss detection;
- cross-source disagreement and shared-control coverage;
- source scope coverage and unobservable event fraction;
- frame construction and reconciliation cost;
- time-to-invalidation after source or policy change.

Proposed gates (goals, not results):
- zero false FRAME_CLOSED promotions across all enumerated adversarial controls;
- 100% detection of omitted use events when present in a source qualified for that event class;
- 100% invalidation of affected closure receipts after relevant late/backfilled/deleted events or scope changes;
- no #10414 sample or query-ready certificate if closure is unknown, conflicted or incomplete;
- unsupported event-source architectures return NOT_ESTIMABLE rather than assuming completeness.

A zero false-closure rate on a finite synthetic corpus is not a measured false-closure rate on deployed infrastructure.

## 11. Comparator matrix

Evaluate on the same frozen scopes and source availability:

1. Producer-manifest-only frame (unsafe baseline).
2. One execution source without cross-reconciliation.
3. Two logs naively assumed independent (negative control).
4. Policy-derived frame reconciled across qualified execution and claim/certificate sources.
5. Full census of the declared event ledger where feasible.
6. Adversarial hidden-use and incomplete-export suite.

Report both the result and the assumptions each comparator needs. A source pair with a shared failure mode must not receive credit for independent reconciliation.

## 12. Integration order

1. #10406 validates the semantics of supplied evidence-dependence graphs.
2. This work qualifies whether the frozen assertion frame covers actual fusion uses in its declared scope.
3. #10414 samples independence assertions in that closed frame.
4. #10416 recomputes sample membership, finite-population bounds and promotion gates.
5. #9255 binds all relevant receipts into the query-specific certificate and refuses promotion if any scope or dependency is stale.

These states remain separate: structural validity, frame closure, sampled provenance findings, dependence/fusion eligibility and biological claim qualification.

## 13. Implementation boundary

V0.1 should be a standalone Rust verifier for normalized, versioned event/frame/reconciliation bundles. It should:
- strictly decode the schema and reject unknown fields;
- canonicalize identities using one pinned algorithm;
- recompute source, event, projection, frame and anti-join digests;
- check sequence/watermark/retention constraints declared by the supported profile;
- validate event-to-claim/evidence mappings and exact multiplicity;
- recompute both anti-join sets and all counts;
- emit deterministic JSON and a non-promotion disposition for any unsupported source architecture;
- bind output to exact inputs and invalidate on relevant changes.

No network, shell execution, raw microscopy access, probabilistic inference, authoritative annotation writes, or generic event-ingestion service is in scope. The Rust verifier cannot independently authenticate a custodian's honesty or prove that an unlogged operation never occurred.

## 14. Claim ceiling

A passing receipt establishes closure only relative to the named source records, source roles, time interval, query population, release set, fusion policy and event-projection contract.

It does not prove that every real-world evidence use was logged, that apparently separate custodians have no hidden common control, that source records are truthful, or that biological evidence is independent. Where source coverage cannot be qualified, the correct outcome is NOT_ESTIMABLE.