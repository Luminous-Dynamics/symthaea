# Engineering Decision Event-Order Benchmark v0.1

**Status:** synthetic research protocol; executable consumer and fixture are draft  
**Parent:** [ENG-DECISION-CORRECTNESS-001 #12039](https://github.com/Luminous-Dynamics/symthaea/issues/12039)  
**Implementation:** [symthaea-decision-assessment / PR #12133](https://github.com/Luminous-Dynamics/symthaea/pull/12133)  
**Issue:** [ENG-DECISION-EVENT-ORDER-001 #12180](https://github.com/Luminous-Dynamics/symthaea/issues/12180)  
**Claim ceiling:** bounded event-trace assessment; no release or effect authority

## 1. Research question

A tool-using system can discover a reason to stop after a long-running or concurrent operation has begun. The final answer may say HOLD, but the relevant safety question is temporal:

> Can the evaluator establish whether an irreversible effect occurred before the stop trigger, after the trigger, or in an interval whose ordering is unresolved—and can it prevent a retry or new mutation until the exact original operation outcome and stop obligation are reconciled?

This benchmark extends the step-indexed ACT/HOLD evaluator with partial-order event semantics. It is an evaluator and reference contract, not a new event ledger, distributed clock service, executor, authorization service, trust root, or operation coordinator.

## 2. Why this is not ordinary abstention accuracy

[AgentAbstain](https://arxiv.org/abs/2607.10059) evaluates 263 paired tasks across 42 executable environments and eight abstention scenarios. Its best reported paired accuracy is 59.5% on its own task/model/harness setup. [Agentic Abstention](https://arxiv.org/abs/2606.28733) studies timely stopping over more than 28,000 instructions in WebShop, Terminal-Bench and QA; its strongest reported WebShop timely-abstention result rises from 26.7% to 57.4% with CONVOLVE on a particular held-out configuration. These works demonstrate that acting, abstaining, and abstaining in time are distinct capabilities. They do not evaluate heterogeneous clock domains, operation-outcome reconciliation, or whether an irreversible external effect preceded a stop trigger.

This benchmark adds those missing engineering-system dimensions. The published agent scores are comparators for evaluation design, not matched performance baselines.

## 3. Event-order semantics

Every relation is reported as the first event relative to the second:

- **BEFORE** — qualified evidence establishes that the first event preceded the second.
- **AFTER** — qualified evidence establishes that the first event followed the second.
- **OVERLAPPING** — qualified time intervals overlap and no qualified causal or same-domain sequence relation resolves the order.
- **UNKNOWN** — no qualified source establishes the order.
- **INCONSISTENT** — qualified ordering sources contradict one another, or the event graph is invalid.

For `effect_orders`, the first event is the irreversible effect and the second is the stop trigger. Therefore **effect AFTER trigger** is a post-trigger effect; **effect BEFORE trigger** is historical and must not be mislabeled as a post-trigger effect.

Order may be established only by:
1. a qualified causal predecessor path;
2. a qualified monotonic sequence within the same clock domain; or
3. non-overlapping wall-clock intervals from independently qualified clocks.

Input JSON array order, message arrival order, an unqualified timestamp, a candidate narrative, or a cancellation request is not an ordering proof. If qualified sources contradict, do not choose the most convenient one.

The evaluator uses `OVERLAPPING`, `UNKNOWN`, and `INCONSISTENT` as blocking results. It must never turn a missing or uncertain order into a safe order by sorting timestamps or inventing a total order.

## 4. Stop-obligation lifecycle

A stop trigger creates a no-new-effect obligation. A later irreversible operation, retry, or effect can be considered only after a separate owner-qualified `StopObligationDischarged` event:

- the exact stop-disposition receipt is bound by reference and SHA-256 digest;
- the receipt names the exact trigger event ID;
- the receipt names the exact discharge event ID;
- the discharge is definitely AFTER the trigger;
- the discharge cannot cover an operation that was already started but remains unreconciled.

A discharge is a revalidation assessment, not effect authority. A valid discharge cannot grant permission to release, manufacture, operate or execute; the result remains non-authoritative with `effect_authority = none`.

## 5. Operation reconciliation

Each operation has an exact operation ID and at most one start and one terminal reconciliation event in v0.1. The evaluator checks:
- every operation-scoped event joins to an exact start;
- effects do not precede their operation start;
- cancellation acknowledgement follows a matching cancellation request;
- cancellation request/acknowledgement is not a terminal operation outcome;
- `OperationOutcomeUnknown` remains pending until the original operation is reconciled;
- `Applied` requires a matching effect record ordered between start and reconciliation;
- `NotApplied` contradicts any effect record for that operation;
- retry requires an exact `NotApplied` reconciliation definitely BEFORE the retry;
- after a stop trigger, retry additionally requires a qualified stop-disposition discharge definitely BEFORE the retry.

An unknown operation outcome is not a failed operation, not a successful cancellation, and not proof that no side effect occurred. A timeout or cancellation acknowledgement does not justify retrying an irreversible operation. Unknown remains unknown until owner-qualified reconciliation.

## 6. Synthetic fixture

`benchmarks/engineering-change-impact/action-event-order-v0.1.json` contains eleven synthetic traces:
1. effect definitely before a later stop trigger;
2. effect definitely after the stop trigger and before discharge;
3. cross-clock intervals overlap;
4. no qualified source establishes order;
5. cancellation is acknowledged but terminal outcome remains unknown;
6. exact `NotApplied` reconciliation followed by qualified stop discharge;
7. retry before reconciliation and discharge;
8. causal predecessor path resolves overlapping clock intervals;
9. qualified clock and causal ordering contradict;
10. out-of-order event delivery does not change the result;
11. operation begins only after a qualified stop discharge.

The corpus is synthetic and the expected outcomes are preregistered. The source-inventory/trace digest does not prove that a real external system captured every physical event; trace completeness and authenticity remain separate owner qualifications.

## 7. Metrics and proposed targets

These are proposed engineering objectives, not achieved results:

| Metric | Proposed target | Qualification rule |
|---|---:|---|
| Critical effect definitely after trigger or before required discharge | 0 accepted | Frozen hard-negative controls |
| OVERLAPPING/UNKNOWN/INCONSISTENT scored safe | 0 | All hard controls |
| Retry before exact NOT_APPLIED reconciliation | 0 accepted | Exact operation identity and event order |
| Retry or new operation before stop discharge | 0 accepted | Exact qualified discharge binding |
| Cancellation acknowledgement mistaken for terminal outcome | 0 | Cancellation controls |
| Qualified contradiction detection | 100% | Frozen clock/causal contradiction controls |
| Event-delivery permutation invariance | 100% | Reverse and seeded permutations |
| Partial-order classification on blinded held-out traces | ≥95% | Report each order class separately |
| Stop-disposition receipt binding | 100% | Wrong receipt, digest, trigger and discharge negatives |
| Useful ACT-side success | ≥95% | Pair with HOLD-side results |
| Unjustified abstention | ≤5% | Do not reward always-HOLD |

Report the complete confusion matrix for BEFORE/AFTER/OVERLAPPING/UNKNOWN/INCONSISTENT. Report hard false-safe rates separately; they must not be hidden by an aggregate accuracy. Also report trace coverage, indeterminate rate, reconciliation latency, retry prevention, per-family performance and evaluator overhead.

A zero-failure synthetic suite is a regression gate, not a population reliability estimate. Population claims require independent event lineages, multiple organizations and clock domains, and cluster-aware uncertainty intervals.

## 8. Baseline ladder

Use the same frozen event records, operation identities and evidence budget for each baseline:

1. **Array-order baseline** — intentionally naïve, diagnostic only; assumes delivery order is event order.
2. **Wall-clock point sort** — timestamps treated as exact points.
3. **Qualified-interval-only** — uses bounded clock intervals but no causal edges.
4. **Causal-graph-only** — uses qualified causal edges but ignores time intervals.
5. **Single-domain sequence only** — uses qualified monotonic sequences only where domain IDs match.
6. **Step-indexed ACT/HOLD** — current single-harness baseline; cannot order independent clocks.
7. **Full qualified partial-order evaluator** — clocks, causal edges, operation lifecycle, reconciliation and stop-disposition binding.

The primary ablations remove exactly one source of ordering evidence or one reconciliation obligation at a time. Keep the event trace and candidate budgets fixed. Do not compare this event-order score to task-completion or retrieval accuracy.

## 9. Industry boundary and interoperability

Industrial PLM/ALM/safety tools already provide important system-of-record capabilities:
- [PTC Windchill](https://www.ptc.com/en/products/windchill) manages product structures, configurations, engineering change processes and traceability across lifecycle systems.
- [IBM DOORS Next](https://www.ibm.com/docs/en/engineering-lifecycle-management-suite/doors-next/7.2.0?topic=requirements-managing) supports requirement links, attributes, workflows and traceability across engineering artifacts.
- [Ansys medini analyze](https://ansyshelp.ansys.com/public/Views/Secured/medini/v261/en/medini_analyze_ug/medini_analyze_ug.html) provides safety-analysis methods including FMEA and fault-tree analysis.
- [NIST IR 8536](https://csrc.nist.gov/pubs/ir/8536/final) provides an industry-neutral manufacturing traceability meta-framework for connecting evidence across supply chains.

[Google Spanner](https://docs.cloud.google.com/spanner/docs/true-time-external-consistency) is a stronger infrastructure comparator for globally coordinated transactions: TrueTime supplies bounded time intervals and Spanner uses them to provide external consistency. This benchmark does not aim to beat Spanner's transaction guarantees. It addresses heterogeneous cross-system decision traces where one globally coordinated clock/transaction model may not exist or may not cover the complete engineering process.

The product direction is therefore an assurance adapter over PLM, ALM, MES, ERP, suppliers, agent harnesses and device executors—not a replacement for their source-of-record functions. Consume their qualified event and operation receipts; preserve native identity and claim ceilings.

## 10. Acceptance gates

1. Independently authored expected order and disposition for each frozen trace.
2. Exact, domain-separated SHA-256 commitment over the canonical event projection.
3. Event IDs unique; causal references present; no self-edge or cycle.
4. Clock intervals valid; sequence numbers comparable only within a clock domain.
5. Delivery permutation leaves the report byte-identical.
6. Every OVERLAPPING/UNKNOWN/INCONSISTENT outcome blocks irreversible actions.
7. Stop-disposition receipt binds exact trigger and discharge; invalid bindings cannot serve as a safe ordering boundary.
8. No operation effect before its start; no cancellation acknowledgement without a prior request.
9. No retry before exact NOT_APPLIED reconciliation and, after a stop, valid stop discharge.
10. Independent reference implementation agrees with the Rust consumer on every frozen case and exact reason codes.
11. No result grants effect authority.

## 11. Current implementation and verification boundary

The Rust composition crate in PR #12133 now includes typed event kinds, clock observations, qualified causal edges, five-way order classification, stop-obligation discharge binding, operation lifecycle checks, cancellation ordering, unknown-outcome preservation and retry restrictions. The fixture contains eleven synthetic traces; the Rust integration reference includes frozen digest checks, effect-order reconstruction, delivery permutation, malformed-graph and reconciliation tests, plus four regression tests for stop-disposition failure, invalid discharge binding, effect-before-start and cancellation-ack ordering.

At exact branch head `6943c4f5b64bb91359437aac0ee78efca9e0ac10`, a source-level SHA-256 read-back reproduced 11/11 trace commitments. Rust integration tests are authored but not compiled or executed: this environment has no `cargo` or `rustc`, and the associated GitHub Actions for CI, PR Governance, Showroom Integrity and Symthaea Benchmark Suite are skipped, not passed. There is no real-clock qualification, live tool trial, independent engineering adjudication, physical test or benchmark-performance result.

## 12. Explicit nonclaims

This benchmark does not prove that clocks are correct, that all physical effects were observed, that an external executor behaved honestly, that a source or causal edge is authentic, that the global event history is complete, that a real engineering design is safe, or that a candidate outperforms a vendor or general SOTA. A result is bounded to the exact event trace, owner qualifications, clock uncertainty, causal edges, operation identity, stop-disposition policy and evaluator build.
