# Engineering Decision Evidence Lifecycle and Requalification Benchmark v0.1

**Status:** synthetic benchmark protocol and preregistered controls  
**Date:** 2026-11-14  
**Issue:** [ENG-DECISION-REQUALIFICATION-001 #12165](https://github.com/Luminous-Dynamics/symthaea/issues/12165)  
**Claim ceiling:** synthetic bitemporal decision-lifecycle benchmark; no real engineering qualification.

## 1. Research question

When evidence, configuration, trust, authority or operating conditions change after an engineering decision, can an intelligent system determine what was justified at the original knowledge frontier K0, what remains justified at a later frontier K1, which decisions depend on the changed evidence, and when a previously supported decision must be requalified, withdrawn or marked indeterminate?

This is not another temporal ledger, event bus, trust registry, configuration manager, alerting service or evidence store. It evaluates the decision layer's consumption of owner-qualified change events and propagation of changed validity across typed decision dependencies.

## 2. Paradigm: a supported decision is a versioned claim, not a permanent fact

A decision is tied to an exact knowledge frontier, evidence manifest, target configuration, intended use and authority context. It is not permanently valid merely because a historical assessment once returned supported.

The benchmark preserves two non-interchangeable views:

- **As-of reconstruction at K0:** only information both effective and known by K0 may influence the historical disposition.
- **Current requalification at K1:** qualified events observed and effective by K1 may change current validity.

The event model distinguishes:

- effective_at: when the change applied in the world;
- observed_at: when the owner/decision system first knew it;
- source_sequence: owner event ordering where available;
- event_inventory_state: whether the relevant owner event inventory is complete;
- event_ordering_state: whether ordering is qualified or unresolved.

A late-arriving event can change current validity without rewriting the historical state justified using information available at K0. Conversely, a revocation known before K0 must block a claim still asserted as supported at K0.

## 3. Dependency-scoped requalification

Every decision names a frozen set of typed evidence/configuration/model dependencies. Each event names the dependencies it can affect.

Only decisions whose dependency sets intersect a qualified event's affected scope should be requalified. An unrelated configuration change, duplicate notification or informational event must not invalidate every decision in the workspace.

Unknown scope is not the same as empty scope. If the affected dependency set is unknown, or the owner event inventory is incomplete, the decision cannot remain supported merely because no matching event was found.

The event owner remains responsible for event authenticity, source completeness, effective/observed time and scope qualification. This benchmark's reference evaluator is not an event service or a trust root.

## 4. Trigger families

| Event family | Expected current handling |
|---|---|
| Evidence superseded or withdrawn | Requalification or withdrawal |
| Trust/key/attestation revocation | Withdraw affected claim and hold |
| Configuration or part-revision drift | Requalify affected configuration claims |
| Intended-use, rights or authority scope change | Requalify or withdraw |
| Expired evidence/owner receipt | Requalify or withdraw |
| Operating-envelope excursion | Withdraw/hold the affected operating claim |
| Solver/model calibration drift | Requalify dependent model-backed claims |
| Late-arriving event | Preserve K0; update K1 |
| Duplicate/irrelevant event | Do not invalidate unrelated claims |
| Unknown trigger scope / incomplete inventory / uncertain ordering | INDETERMINATE |

## 5. Contract and output states

For every decision, the evaluator expects the as-of state at K0, current state at K1, current action/disposition, exact reason codes, and an immutable original snapshot/dependency-manifest binding.

States: supported, requalification_required, withdrawn, indeterminate.

Candidate dispositions: no_requalification, hold_and_requalify, withdraw_claim_and_hold, hold_pending_evidence.

These are decision dispositions only. The assessment contract must carry effect_authority = none. No benchmark result authorizes release, manufacture, operation, actuator control or any other external effect.

## 6. Frozen synthetic corpus

The companion corpus contains ten known-answer cases covering no change, evidence supersession, late-arriving trust revocation, irrelevant duplicate notification, a two-decision scope-control baseline, configuration drift affecting only one of two decisions, operating-envelope breach, model calibration drift, unknown scope with incomplete event inventory and uncertain ordering, and revocation already known before K0.

Eight paired interventions specify INVARIANT, FLIP, late-arrival, scoped-invalidation or historical-rejection relations.

## 7. Metrics

- Critical trigger recall and precision, severity-weighted.
- Requalification latency from first observable qualified trigger to safe disposition.
- Stale-decision reuse after an observable critical trigger.
- Historical reconstruction fidelity at K0.
- Dependency-scoped invalidation precision.
- Late-arrival handling and bitemporal correctness.
- Duplicate/noise resilience.
- Unknown event ordering / incomplete inventory handling.
- Requalification burden and reviewer time, measured separately from correctness.

## 8. Proposed targets (not achieved results)

| Metric | Proposed target |
|---|---:|
| Frozen critical trigger recall | 100% |
| Critical stale-decision reuse | 0 |
| Retroactive rewriting of K0 snapshots | 0 |
| Dependency-scoped propagation precision | ≥95% on blinded held-out cases |
| Late-arrival disposition accuracy | ≥95% |
| Unknown scope/order/inventory promoted to supported | 0 |
| Irrelevant/duplicate event invalidation | 0 on frozen controls |
| Timely requalification latency | Median 0 steps; p95 ≤1 step on trace-complete controls |

A zero-failure synthetic corpus is a regression gate, not a population reliability estimate. Use independently authored task lineages, supplier/configuration families, and cluster-aware confidence intervals before generalizing.

## 9. Baseline ladder and ablations

Matched tasks, event inventories, owner receipts, budgets and action sets:

1. Human-only review.
2. Native PLM/ALM change and configuration process.
3. Vendor-native impact analysis and lifecycle traceability.
4. Static snapshot / no-requalification control.
5. Time-aware RAG without typed dependency propagation.
6. Typed dependency graph with event-scope matching.
7. Bitemporal event knowledge plus qualified owner receipts.
8. Symthaea decision evaluator with independent replay and defeater analysis.

Ablate effective_at vs observed_at, total vs partial event order, complete vs incomplete event inventory, typed dependency scope vs coarse tags, and historical snapshot preservation vs overwrite. Keep event bytes and candidate-visible sets matched.

## 10. Interoperability and industry posture

PLM, ALM, MES, ERP, supplier registries, trust/transparency services, telemetry systems and engineering-model registries remain systems of record. Consume their owner-qualified change events and receipts; do not replace their event stores.

The value proposition is independently reconstructable answers to which decisions depended on the changed item, what was knowable at K0, what changed at K1, why a decision remains valid/must be requalified/is withdrawn/is indeterminate, and whether stale use occurred after a critical trigger.

Product documentation from PTC, Dassault Systèmes, IBM and Ansys establishes substantial native PLM/ALM/change/safety-analysis capabilities; it is not independent comparative performance evidence. A superiority claim requires matched, blinded operational trials.

## 11. Explicit nonclaims

This benchmark does not establish continuous event-monitoring completeness, sensor coverage, universal event ordering, legal validity of revocation, physical safety, real-time actuation guarantees, external certification or vendor superiority. Those require owner-native validation and prospective engineering trials.
