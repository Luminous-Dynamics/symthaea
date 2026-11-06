# SPECTRA Independent Counterfactual Projection Index v0.1

**Status:** draft implementation; Rust compilation/tests and benchmark execution are pending.
**Base:** certificate-directed allocation and CRN projection reuse.
**Method:** unchanged "robust-certificate-contraction-allocation-v1".

## Motivation

CRN mode already reuses one exact scenario-wide projection because a scenario batch changes every action atomically. Independent-per-cell mode changes only one (action, scenario) utility interval, but previously rebuilt the full action × scenario map, re-aggregated every action, and re-found the global maxima for every candidate cell.

That repeated work is avoidable: an independent update changes exactly one action's robust interval. Every other action interval is invariant under that candidate projection.

## Implementation

The new IndependentCounterfactualProjectionIndex in ranking.rs:

1. Validates and freezes the action × scenario utility-interval grid once.
2. Computes every action's existing Maximin or lower-tail CVaR interval using the production aggregation function.
3. Caches the best and runner-up action lower endpoints and upper endpoints, with deterministic action-ID tie ordering.
4. For a candidate cell, replaces only that cell, recomputes only the affected action's robust lower/upper aggregates, and updates the global regret bound using the cached best/runner-up endpoints.
5. Reuses the caller's candidate metric reference and a per-action metric index, avoiding a repeated linear search through the entire metric grid in independent mode.

The full counterfactual_regret_upper_bound remains intact as the reference oracle. CRN still uses that full path because a CRN projection modifies every action in one scenario.

## Exactness contract

The index reuses aggregate_robust_utility unchanged. This preserves:

- Maximin's minimum endpoint;
- lower-tail CVaR's total_cmp ordering;
- scenario-ID tie-breaking;
- weighted fractional tail-mass inclusion;
- the same weighted accumulation order for each action;
- the same global regret expression max(0, max_a U(a) - max_a L(a)).

No inferential interval, familywise alpha, stopping rule, robust objective, scenario weight, action set, or score receipt is changed. This is a deterministic projection optimization, not an optimal allocation theorem or an expected-value-of-computation model.

## Complexity

Let A be the number of actions and S the number of scenarios.

- One-time metric indexing and robust-interval preparation is O(A·S log(A·S)) for ordered maps and O(A·S log S) for CVaR aggregation.
- Per independent candidate, Maximin projection is O(S) and CVaR projection is O(S log S); the global best/runner-up endpoint update is O(1).
- Across all A·S candidates, the worst-case projection phase is O(A·S²) for Maximin and O(A·S² log S) for CVaR, rather than rebuilding all A action aggregates for every candidate.
- CRN complexity is unchanged by this PR.

These are structural operation-count expectations, not measured latency results. The next optimization frontier is a per-action dynamic minimum index for Maximin and a deterministic weighted order-statistics structure for CVaR. Any such structure must preserve the registered ordering and a reviewed numerical-equivalence contract; a faster but arithmetically different reduction must not silently replace the oracle.

## Tests added

- Direct indexed projection equals the full recomputation oracle for Maximin and fractional-tail CVaR.
- A deterministic override corpus covers three robust-utility settings and 128 generated overrides per setting.
- Tied action/scenario values are preserved.
- Out-of-range and missing-scenario overrides fail closed.

These are Rust test declarations, not evidence that the tests have run. This environment currently has no cargo, rustc, or rustfmt; CI status must be checked at the exact PR head. No test, compile, formatting, clippy, or wall-time pass is claimed here.

## Locked benchmark plan

Use the same frozen receipts, root seeds, machine, compiler, and toolchain for the indexed path and the full oracle:

| Dimension | Required sweep |
|---|---|
| Actions | 2, 4, 8, 16, 32, 64, 128 |
| Scenarios | 4, 16, 64, 256, 1,024 |
| Utility | Maximin; CVaR tail mass 0.1, 0.25, 0.5, 0.9 |
| Geometry | unique minima; tied minima; fractional tail boundaries; broad and narrow intervals |
| Workload | independent mode; CRN control; sparse and dense competitive sets |
| Outputs | exact projected-bound parity; allocation-receipt replay; p50/p95/p99 allocator time; allocations and peak memory |

Acceptance gates:

1. Exact equality against the full oracle for all deterministic parity fixtures; any deliberate floating-point relaxation requires a separately reviewed error bound and cannot weaken certificate coverage.
2. Zero change to the allocation batch, method ID, certificate, or replay outcome on the locked parity corpus.
3. No wall-time superiority claim until the benchmark is run on the same host and toolchain.
4. Report all workload sizes and both medians and tail latency; do not extrapolate small synthetic fixtures to production scale.
