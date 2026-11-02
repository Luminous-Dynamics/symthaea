# connectome-query-stop-calibration

Finite-sample, volume/cluster-disjoint calibration of future drift for a pre-registered scientific query.

## Purpose

The query-stability auditor (#9994) checks stability over already observed graph snapshots. This crate addresses a separate question: under a declared calibration stratum and fixed edit horizon, what finite-sample upper bound can we place on a target query's future maximum drift?

It uses split-conformal calibration of a simple residual predictor. It does not fit a complex model or claim conditional coverage.

## Calibration unit

One calibration unit is one independent source trajectory cluster. The input includes both:
- a source volume ID (the physical dataset volume);
- an independence-cluster ID (the highest shared animal/session/archive lineage that must not be counted twice).

Each cluster and each source volume must appear at most once per profile. The target source volume and target independence cluster must be absent from calibration. If several volumes share one animal or acquisition lineage, they cannot inflate n as independent calibration samples.

The group-disjointness report and explicit exchangeability assessment have separate digests and boolean gates. Group disjointness does not establish exchangeability. An adapter must verify both reports, the source group IDs, and the query stratum contents.

## Finite-sample method

All drift ratios are normalized by the query's frozen pre-registered materiality threshold.

For calibration trajectory i:
- O_i = maximum normalized drift relative to the rolling-origin anchor in the observed window;
- F_i = maximum normalized drift relative to the same anchor in the fixed future edit horizon;
- R_i = max(0, F_i - O_i).

For n independent calibration trajectories and miscoverage alpha:
- k = ceil((n + 1) * (1 - alpha));
- if k > n, no finite order-statistic bound exists and the status is INSUFFICIENT_CALIBRATION;
- otherwise q is the k-th sorted residual;
- target upper bound is O_target + q.

The candidate is ProjectedStableWithinHorizon only when:
- its current identified-set width / materiality threshold is <= 1; and
- its calibrated upper future-drift ratio is <= 1.

For alpha=0.05, n=18 gives k=19 and must abstain; n=19 gives k=19 and a finite bound. This minimum is an algebraic consequence of the finite-sample rank, not a recommended universal sample size. The research target of at least 30 independent calibration clusters is a stronger operational support target.

## Statuses and abstention

- InsufficientCalibration: k > n; no finite bound is reported.
- NotCalibratedGroupOverlap: group-disjointness audit does not support the target/calibration separation.
- NotCalibratedExchangeability: the declared target stratum is not supported as exchangeable with calibration.
- ProjectedStableWithinHorizon: both current uncertainty width and projected upper future drift meet tolerance.
- ProjectedUnstableWithinHorizon: either current uncertainty or the projected future-drift upper bound exceeds tolerance.

The report always sets universal completeness and all-future-edits certification to false. The status applies only to the named query, target volume, frozen stratum and future edit horizon.

## Assumptions and limits

Split-conformal finite-sample marginal coverage relies on exchangeability of calibration and target trajectory units in the declared stratum. It does not provide distribution-free conditional coverage and does not survive arbitrary temporal, species, acquisition or workflow shift. A boolean exchangeability field is not proof; it must be backed by an independently reviewed assessment artifact and checked by the adapter.

The residual is clipped at zero so the future upper bound is never reduced below the observed max-drift value. That makes the method conservative. If future trajectories are not independent at the declared cluster level, the finite-sample guarantee does not apply.

Rolling-origin calibration data must be constructed without lookahead: for every calibration trajectory, the observed window ends before the future horizon begins. The query definition, analysis code, metric, unit, materiality threshold, stratum, rolling-origin procedure and future edit horizon are fixed before residual calculation.

## Synthetic fixture

- Input schema: `research/connectomics/connectome-query-stop-calibration-input-v0.1.schema.json`.
- Report schema: `research/connectomics/connectome-query-stop-calibration-report-v0.1.schema.json`.
- Fixture schema: `research/connectomics/connectome-query-stop-calibration-fixture-v0.1.schema.json`.
- Known-answer cases: `research/connectomics/fixtures/connectome-query-stop-calibration-known-answer-v0.1.json`.

Eight cases cover 18/19 sample-size behavior, stable/unstable projected bounds, uncertainty-width blocking, group-overlap abstention, exchangeability abstention, duplicate independence clusters and target cluster overlap.

## Tests

Run in a Rust environment:

```sh
cargo test --locked --manifest-path tools/connectome-query-stop-calibration/Cargo.toml
```

Nineteen Rust unit tests are authored for the exact order-statistic rank, insufficient calibration, finite upper bounds, unstable/wide uncertainty, group overlap, source-volume and independence-cluster separation, duplicate IDs, invalid alpha/horizon/tolerance/ratios, empty calibration, deterministic order and evidence digests, calibration-trajectory manifest provenance and report-field propagation.

They have **not** been compiled or executed in the current authoring environment.

## Integration

- #10023 supplies group-disjoint split evidence.
- #9994 supplies observed query-stability trajectories.
- #10054 defines query-specific endpoint and materiality semantics.
- #10028 binds the stopping-calibration receipt into the cross-stage evidence envelope.

No real ConnectomeBench2 V2 history is included. The component has not estimated actual coverage, false-stop rate or biological completeness.

See [SCI-CONNECTOMICS-061](https://github.com/Luminous-Dynamics/symthaea/issues/10067).