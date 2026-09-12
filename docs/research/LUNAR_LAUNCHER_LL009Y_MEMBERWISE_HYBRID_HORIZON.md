# LL-009Y — Memberwise Site01 clone + geometry-aware far RMS horizon

LL-009Y numerically combines the two uncertainty mechanisms currently available for the Site01 study without pretending they are one probability model.

- **Near field / observer:** LL-009Q's finite NASA Site01 clone ensemble. Each exact member carries both its own near-field horizon and the site elevation from that same realization.
- **Far field:** LL-009V/W/X's Product 90 RMS semantics and distribution-free familywise geometry-aware envelope.
- **Spatial support:** exact LL-009R classifications/margins, preserved independently of vertical uncertainty.

## Ordering theorem

For Q member `j` and azimuth bin `b`, Y re-evaluates the far RMS envelope at the member's exact observer radius and forms

`H_full[j,b] = max(H_near[j,b] + margin_near, H_far[j,b] + margin_far)`.

Only after every member/bin full horizon exists does Y take the configured finite-ensemble nearest-rank statistic. This ordering is required because in general

`quantile(max(A,B)) != max(quantile(A), quantile(B))`.

The executable synthetic campaign includes a direct q=0.75 counterexample: two marginal arrays each have nearest-rank quantile 0°, while their memberwise maximum has quantile 10°.

## Exact reuse of the X support population

Observer height changes along the fixed Site01 lunar radial. In the observer's local tangent frame this changes no tangent component, so far support-point azimuth does not change. Y therefore reconstructs the exact X far population once and requires, per bin, both exact admitted count agreement and exact binary population SHA-256 agreement with X's `input_population_digest_sha256`.

Only the observer radius changes member by member. This is stronger than merely asserting that the same source files were used: it proves Y solves over the exact support population X qualified.

## Memberwise far solve

For each exact Q member observer radius, Y keeps X's whole-sky alpha and exact per-bin `alpha_b` budgets. It reconstructs a guaranteed-feasible W-like upper skyline using the exact W multiplier and then runs the same geometry-aware Markov/union-bound bisection against that observer radius.

The implementation is batched over Q members and chunked over support points so the real 100-member run does not require a giant `members × pixels` in-memory matrix. Every emitted member/bin far skyline is re-evaluated and must satisfy its inherited `alpha_b` budget.

## No unsupported Q × far probability coupling

Product 90's marginal RMS second-moment model does not establish the same moment bound after conditioning on a particular Q clone state. Y therefore treats each Q member as a deterministic observer/near-terrain **scenario** and geometrically re-evaluates the same marginal far RMS model at that observer radius.

Y does **not** claim that the far error distribution conditional on Q member `j` has the same RMS moments. It assumes neither independence nor a correlation model between Q and Product 90 error. A future joint-coupling/covariance theorem would be required before a joint Q×far probability statement could be promoted.

Across scenarios, Q remains a finite published empirical ensemble. Thus a Y q=0.99 result means the 0.99 nearest-rank empirical quantile across the exact published Q member scenarios of their scenario-parameterized far-RMS-enveloped full horizons. It is not a parent-population 99% confidence interval or a joint confidence statement, and Y does not multiply `0.99 × (1-alpha)` into a new confidence number.

## K/S handoff

Y emits `schema_version = ll009k.horizon-pack.v1` so LL-009S can consume the numeric bins directly. The pack carries `statistical_horizon_binding.status = bound` with mode `empirical_q_of_scenario_parameterized_distribution_free_far_envelopes`.

The binding hashes Q, V, W, X policy/receipt, R, L, the estimator/quantile, and a deterministic digest over every memberwise numerical solve. This prevents statistical side evidence from being attached to unrelated K numbers.

LL-009S may still downgrade the result until a later capability-reconciliation layer supersedes LL-009O's older far-field `unknown` classification and respects LL-009R's current spatial-support limitations. Numeric closure and claim promotion remain separate.

## Local logic campaign

The synthetic Rasterio 1.5.0 campaign passes deterministic replay, exact X per-bin population-digest reconstruction, observer-height-dependent far skyline changes with invariant bin population, per-member/per-bin far Markov risk within exact X budgets, exact Q near/observer member coupling, R spatial-margin application, memberwise max before quantile, the marginal-vs-memberwise quantile counterexample, and Q self-hash tamper rejection.

No real NASA Y receipt is claimed until the exact N/Q/V/W/X/R/L evidence chain has executed against acquired NASA bytes in the pinned GIS environment.
