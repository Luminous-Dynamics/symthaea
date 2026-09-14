# State-tracking query-mix contract

This document freezes the semantics of `StateTrackingBenchmarkConfig::historical_query_rate` for the current state-tracking benchmark lineage.

## Exact meaning

`historical_query_rate` is **not** the target fraction of all generated queries that are historical. It is the Bernoulli probability of appending one additional historical query at each eligible scoring cadence.

After the observable initialization prefix, each eligible cadence always emits exactly three current-state queries:

1. one `EntityLocation` query,
2. one `ObjectOwner` query,
3. one `ObjectLocation` query.

If the historical augmentation draw succeeds, the generator then appends exactly one additional historical `ObjectLocation` query whose `as_of_event` precedes the event after which it is asked.

For `N` eligible cadences, the current-query count is therefore exactly `3N`. The historical count is a random variable `H` determined by the frozen benchmark RNG and satisfies `0 <= H <= N`; total queries are `3N + H`.

Boundary cases are exact:

- `historical_query_rate = 0.0` -> `H = 0`; all generated queries are current-state.
- `historical_query_rate = 1.0` -> `H = N`; every cadence has three current queries plus one historical query, so the historical fraction is exactly `1/4`, not `1.0`.

For intermediate probabilities, the configured rate must not be reported as an observed historical-query fraction. Experiment manifests and analyses should report the actual current and historical query counts produced by the deterministic seed.

## Compatibility rule

The current field name is retained for backward compatibility. Changing its meaning to a target mixture fraction would silently change existing benchmark semantics and therefore requires a new versioned benchmark/query-schedule contract.

A future protocol that needs an exact historical/current mixture should introduce an explicit schedule or mix specification rather than reinterpret this field.

## Scientific boundary

This contract clarifies an existing generator. It does not modify the query generator, frozen validity-capacity cases or seeds, evidence thresholds, or any `research_v0` outcome. It was frozen after exact-head qualification exposed a test that incorrectly interpreted rate `1.0` as an all-historical benchmark.
