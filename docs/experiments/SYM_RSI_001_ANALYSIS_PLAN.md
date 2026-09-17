# SYM-RSI-001 — Primary Analysis Plan

**Status:** pre-primary-outcome analysis plan. This document supplements the existing preregistration and freezes the C-vs-A decision rule before the fresh-execution entry point is introduced.

## Scope

This analysis plan applies only to the primary **C vs A** contrast:

- **A — Fixed exploration:** the frozen incumbent fixed-hash policy.
- **C — Exact replay policy improvement:** the single policy selected from the frozen eight-policy family using only the frozen training-replay corpus, after surviving the independent held-out replay gate.

It does not define the later D-vs-C counterfactual-dreaming analysis.

## Frozen fresh-execution sample

The fresh controlled evaluation uses exactly the preregistered `FreshExecution` seeds:

- 201
- 202
- 203
- 204

for each of the three frozen fixture domains:

1. `sym-rsi-branching-search-v1`
2. `sym-rsi-delayed-navigation-v1`
3. `sym-rsi-rugged-optimization-v1`

This yields **12 paired A/C comparisons**. No fresh seed may be added, removed, replaced, or rerun selectively under the same evidence lineage after primary outcomes are observed.

## Pairing

Every C observation is paired with A on the exact same:

- experiment ID,
- preregistration digest,
- subject digest,
- environment digest,
- domain adapter version,
- domain,
- split,
- seed.

A pair that does not satisfy the machine-checkable `PrimaryContrastReceipt` identity contract is invalid rather than approximately matched.

## Domain balancing

The primary quality effect is a **macro-average across the three domain-level mean deltas**, not a raw average over arbitrary observations. Each frozen domain therefore contributes equal weight to the primary quality conclusion.

For domain `d`:

`quality_delta_d = mean(C_quality - A_quality)` over seeds 201–204.

Primary macro quality delta:

`macro_quality_delta = mean_d(quality_delta_d)`.

Evaluator-call effects are reported both per-domain and as total paired call delta:

`total_call_delta = sum(C_calls - A_calls)`.

Negative call delta means C used fewer evaluator/environment calls.

## Frozen positive-result rule

A C-vs-A result is classified **PositiveUnderProtocol** only when all of the following hold:

1. a non-incumbent C policy was selected using training replay only;
2. that exact policy passed the independent held-out replay gate;
3. all 12 fresh A/C pairs are present and provenance-valid;
4. no authority-boundary or safety-constraint violation is recorded in either arm;
5. **every domain** satisfies `quality_delta_d >= -0.02`;
6. `macro_quality_delta >= -0.02`; and
7. at least one strict gain is present:
   - `macro_quality_delta > 0`, **or**
   - `total_call_delta < 0`.

The `0.02` bound is the same frozen held-out quality tolerance already encoded in the canonical experiment manifest. It is not changed after fresh outcomes are observed.

## Other dispositions

- **NoCandidatePromotion** — training replay retained the incumbent. Fresh seeds remain unconsumed by the C-vs-A runner.
- **BlockedByHoldout** — a non-incumbent training selection did not pass held-out replay. Fresh seeds remain unconsumed.
- **QualityNonInferiorityFailed** — one or more domain mean quality deltas are below `-0.02`, or macro quality delta is below `-0.02`.
- **NoStrictGain** — quality remained within tolerance but neither macro quality improved nor total evaluator calls decreased.
- **IntegrityFailure** — required pairs, lineage bindings, evidence digests, or zero-violation requirements are incomplete or invalid. This is not converted into a negative scientific result; the run is unqualified.

## Statistical treatment

The first mechanism test has only 12 deterministic/seed-controlled paired observations. It will therefore report:

- all 12 paired effects,
- domain-level means,
- macro-average quality delta,
- total evaluator-call delta,
- worst-domain quality delta,
- replay and held-out gate receipts.

No asymptotic p-value or post-hoc significance threshold is used for the primary conclusion. A later, larger-seed confirmatory lineage may preregister inferential statistics separately.

## Claim boundary

`PositiveUnderProtocol` supports only the preregistered bounded claim:

> Historical exact replay can improve exploration policy under the tested domains and protocol.

It does not establish recursive general intelligence, open-ended self-improvement, consciousness, universal monotonic improvement, or generalization beyond the tested distributions.
