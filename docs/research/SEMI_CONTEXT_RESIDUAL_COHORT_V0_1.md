# SEMI-CONTEXT-COHORT-001 — cross-context common-mode drift decomposition v0.1

**Issue:** #13906  
**Implementation:** symthaea-semiconductor::benchmark::context_residual_cohort  
**Status:** Pure Rust cohort estimator with 9 authored unit tests; exact-head validation pending.

## Motivation

The v0.1 streaming adapter has both a global context residual EMA and local tool/chamber residual EMAs. Updating the global EMA from one sample's context residual risks leaking a local chamber bias into every other context. A single observation cannot identify common-mode and local offsets without extra assumptions.

This module treats global/local drift as a cohort identifiability problem rather than an ordinary EMA tuning problem.

## Cohort model

Each observation binds:
- exact immutable streaming prediction receipt and its model, feature, policy and context identity;
- K1-qualified label ID/revision, full-payload digest and corrected numeric value;
- label/report availability and delivery times;
- common-cause group ID and its membership-evidence digest;
- calibration epoch and clock domain.

A cohort binds:
- one clock domain and calibration epoch;
- event-time window and close time;
- declared common-cause registry digest;
- owner-declared inventory completeness and cohort closure;
- minimum independent common-cause group count;
- all sealed observations.

## Estimator

For observation \(i\), calculate the frozen residual

\[
e_i = y_i - \widehat y_i,
\]

where \(\widehat y_i\) is the prediction value in its immutable receipt.

First calculate one robust median per declared common-cause group:

\[
m_g = \operatorname{median}\{e_i : g(i)=g\}.
\]

Then estimate common-mode residual from group medians, so a group with many chambers does not receive greater weight merely because it generated more records:

\[
\widehat g = \operatorname{median}\{m_g\}.
\]

For each context \(c\), estimate a local offset:

\[
\widehat \ell_c =
\operatorname{median}\{e_i:c(i)=c\} - \widehat g.
\]

The estimator is deterministic and input-order invariant. It uses a median at both aggregation levels; its 50% breakdown property applies over the declared independent-group units, not over arbitrary physical tools if the common-cause registry is wrong.

## Fail-closed states

The estimator returns UNKNOWN without a common-mode value when:
- the common-cause inventory is incomplete;
- the cohort has not been closed;
- fewer than the configured number of independent common-cause groups are represented.

It rejects malformed/tampered receipts, duplicate observation IDs, duplicate label/revision pairs, duplicate measurement events, K0 or ineligible reports, value/revision/payload mismatch, clock/epoch mismatch and observations outside the declared event/availability window.

Even a COMPLETE result sets physical_independence_proven=false and source_authenticity_proven=false. A content digest and a declared group ID do not prove physical independence or source authenticity.

## Authored adversarial tests

Nine Rust unit tests and two independent integration-oracle tests cover:
1. one common-cause group cannot identify global drift;
2. coherent shifts over independent groups produce a common-mode estimate and local offsets;
3. one extreme group outlier does not dominate the group-median estimator;
4. multiple chambers sharing one reference count as one group;
5. incomplete inventory or open cohort yields UNKNOWN;
6. input ordering does not change cohort/report digests;
7. late observations and calibration-epoch mismatch are rejected;
8. duplicate label/revision cannot be double-counted across contexts;
9. mismatched clock domain cannot enter the same cohort.

The independent oracle separately computes the group medians and median of group medians, then verifies that increasing observation count within one common-cause group does not increase that group's global estimator weight.

## Integration boundary

This v0.1 module is a pure decomposition result. It does not update the streaming adapter's global or local EMA state. A follow-on integration must define when a completed cohort result is eligible to promote a common-mode state, how local state updates are separated, and how later calibration revocation creates a new replay version without rewriting historical receipts.

## Benchmark

Compare:
- current single-stream global/context EMA as the baseline;
- local-only adaptation with the global state held;
- cross-context cohort median;
- trimmed mean and Huber/median-of-means variants as separately versioned estimators.

Adversaries:
- one local chamber outlier;
- coherent global shift across truly independent tool groups;
- shared reference/calibration common cause across multiple tools;
- unbalanced numbers of observations per group;
- missing/unknown group memberships;
- asynchronous clocks and calibration epochs;
- late label/revocation and out-of-order evidence;
- 10%, 20%, 30%, 40%, 49%, 50% contaminated groups.

Metrics:
- common-mode bias and absolute error;
- local-offset error and worst-context bias;
- false-shared-drift rate;
- global detection delay;
- group-level breakdown curve;
- UNKNOWN rate under incomplete evidence;
- deterministic replay and compute/memory.

## Claim ceiling

This is a robust statistical decomposition under declared cohort and common-cause grouping assumptions. It does not prove process causality, physically independent tools, actual root cause, yield improvement or safety. It does not mutate adapter state or authorize APC/R2R, recipe changes, wafer disposition or equipment commands.
