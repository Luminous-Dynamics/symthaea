# CMAQ-058 — Exact O(K) renewal aggregation for capital-restarted kernel betting

## Paradigm shift

CMAQ-055 is a correct full-reset all-start oracle, but creates one child monitor per start and updates every active child each observation. Its update cost grows quadratically with horizon.

CMAQ-057 emits one valid predictable log-factor per registered Fourier expert at each observation. For expert k, call it f_(k,t). The factor is generated from the expert's shared prequential policy state, which is updated after the current pair. We can restart capital at every possible start time while retaining that one shared predictable factor stream.

For the dyadic start prior nu_s, define:

    M_(s,k,t) = product_(i=s..t) f_(k,i)       for t >= s
    M_(s,k,t) = 1                              for t < s
    R_(k,t) = sum_(s<=t) nu_s * M_(s,k,t)
    U_t = sum_(s>t) nu_s

Then:

    R_(k,t) = f_(k,t) * (R_(k,t-1) + nu_t)

and the complete kernel × start mixture is:

    E_t = U_t + sum_k w_k * R_(k,t).

The recurrence is exact. It does not truncate or renormalize the start prior and does not need an FFT. It requires O(K) numeric restart bookkeeping per observation.

## Equivalent stable log recurrence

For each kernel k, maintain:

    C_(k,t) = C_(k,t-1) + log(f_(k,t))
    A_(k,t) = A_(k,t-1) + nu_t * exp(-C_(k,t-1))
    R_(k,t) = exp(C_(k,t)) * A_(k,t)

with log-domain log-add-exp for A and E. This is algebraically equivalent to the recurrence above and permits stable long-horizon operation within floating-point limits.

The exact rational dyadic start prior and future mass come from CMAQ-055. Kernel prior weights are the exact fixed weights in the CMAQ-056 canonical manifest. The reference-failure mode is not supported in this first implementation: the factor producer and typed risk plan must use direct conditional paired equality.

## Critical scope boundary: capital restart, not policy-state restart

CMAQ-058 uses one shared CMAQ-057 policy state per expert for the entire stream. Each start-time component begins its wealth at one at its own start, but its future factors use the shared prequential policy state, which may include observations before that start.

This is valid because each factor remains predictable with respect to the full filtration, and every fixed-start component is a product of future conditional e-factors starting from wealth one. But it is not the same strategy as CMAQ-055's full reset of both policy state and capital.

The direct full-reset CMAQ-055 oracle remains the reference comparator. The benchmark must report this distinction and measure whether retaining historical policy state helps or harms detection after late shifts.

## Implementation

- Constructs a CMAQ-057 factor producer in unthresholded mode. The parent restart mixture owns the single quarantine threshold.
- Receives one positive predictable log-factor per canonical Fourier expert at each step.
- Checks expert IDs, order, policy digest, manifest digest, trace digest, local sequence and pair identity.
- Uses exact rational start weights and unstarted residual mass from CMAQ-055.
- Maintains per-expert cumulative log factors and weighted inverse-prefix sums.
- Computes exact all-start wealth in log space, kernel posterior mass and future-start posterior mass.
- Uses the C057 prepare/commit API: prepares factors and staged policy state without cloning the growing replay-ID set, calculates all renewal states and posteriors, then commits once.
- Uses one child policy state and one pair-ID ledger, not one per start.
- First version is horizon-bounded and resource-bounded; no start truncation or hidden re-normalization.

## Complexity

Let K be the number of kernel experts and D the per-expert product-feature dimension.

- Factor production: O(KD) time and O(KD) policy state, from CMAQ-057.
- Restart mixture update: O(K) time and O(K) numeric restart state.
- Pair replay inventory: O(T) IDs in the single child policy monitor.
- Total: O(KD + K) per observation, not O(KT) or O(KT²) restart work.

## Exactness tests

The test suite compares the recurrence to a naive O(K*T²) enumerator on deterministic factor sequences with gains and losses. It compares:
- every per-kernel restart wealth;
- the total kernel × start mixture;
- future-start residual mass;
- posterior normalization.

The integration tests also verify that equal pairs preserve unit wealth and that the unthresholded CMAQ-057 child continues producing factors after its own 1/alpha threshold, leaving the parent to make the only quarantine decision.

## Benchmark programme

Compare CMAQ-055 full-reset oracle and CMAQ-058 capital-only restart on identical conditional paired streams. Include CMAQ-054 and CMAQ-057 as ablations; faithful SCPT and ICML 2026 kernel-family methods are the primary external comparators.

Keep PF-CCTM scalar-reference and Online RFF-MMD unconditional two-sample results in separate protocol tables. PF-CCTM's FFT portfolio is related motivation, not the same betting process or null.

Initial horizons: 128, 256, 512 and 1,024. Use at least 10,000 independent full-horizon null paths and 5,000 alternatives per primary cell; alpha 0.01, 0.05 and 0.10; misses count H+1. Report ever-alarm confidence intervals, unconditional power, restricted delay, worst-family performance, p99 factor-production and restart-update cost, and memory.

## Claim boundary

The renewal recurrence is exact only when every start-time component for expert k uses the same per-time predictable factor f_(k,t). It restarts capital, not policy state. Every factor stream must still be a valid conditional e-process increment under the same null. A digest/trace binds content and ordering; it does not prove stochastic assumptions or source authenticity.

No SOTA, Type-I calibration, physical-safety or deployment claim follows from the recurrence tests or smoke probe.
