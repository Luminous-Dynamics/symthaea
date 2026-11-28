# MON-048 — Evaluator mutation and oracle qualification

**Status:** benchmark protocol and seeded challenge manifest; runner pending  
**Date:** 2026-11-28  
**Tracks:** MON-043, MON-046, MON-047

## Thesis

A benchmark evaluator is part of the system under test. It must be challenged, not trusted because it emits a score. A recovery-assurance evaluator must detect seeded defects, preserve safe positive controls, and resist strategies that maximize apparent safety by returning `indeterminate` for everything.

## Challenge corpus

`CHALLENGES-v1.json` defines 24 deterministic cases: 3 positive controls and 21 negative/adversarial controls. Each case identifies the intended mutation, expected disposition and failure family. These are oracle fixtures, not production incident observations.

## Score dimensions

Report the following separately; do not collapse them into a single accuracy number.

- **False-qualified rate:** unsafe seeded cases returned as qualified / unsafe seeded cases.
- **False-blocked rate:** safe positive controls incorrectly blocked / safe positive controls.
- **Indeterminate rate:** indeterminate outcomes / all cases, stratified by expected disposition.
- **Decision coverage:** decisive outcomes (qualified or blocked) / cases whose oracle expects a decisive outcome.
- **Mutation detection:** detected evaluator mutations / applicable seeded mutations.
- **Invalidation precision:** unrelated claims preserved / unrelated claims in scoped-invalidation fixtures.
- **Invalidation recall:** affected dependent claims invalidated / affected claims in fixtures.
- **Admission safety:** unauthorized or stale mutation attempts rejected / seeded unauthorized attempts.
- **Latency and cost:** p50/p95/p99 evaluation latency, incremental recomputation time, and cost per adjudicated claim.

## Mandatory benchmark gates (proposed)

1. Zero false-qualified outcomes across all declared critical negative controls.
2. 100% detection of applicable deterministic critical mutations.
3. 100% correct disposition on the frozen known-answer corpus.
4. Zero unrelated-claim invalidations in scoped fixtures.
5. 100% dependent-claim invalidation in mandatory dependency fixtures.
6. Positive-control false-blocking reported with confidence intervals; initial engineering target <=1% only after sufficient safe-control sample size.
7. An all-indeterminate evaluator fails decision-coverage gate.
8. No aggregate score may compensate for a critical false qualification.

The 24-case starter corpus is too small to substantiate a <=1% statistical false-block rate. That threshold requires a larger, appropriately sampled safe-control corpus and confidence intervals.

## Oracle integrity and holdouts

- Keep public development fixtures separate from sealed evaluation holdouts.
- Commit a manifest digest before candidate evaluation; reveal holdout contents only after the candidate artifact is frozen.
- Have an independent oracle implementation produce expected dispositions.
- Mutation-test the oracle itself: alter expected labels, omit dependency edges, weaken signature checks, and introduce incorrect scope inventories. The benchmark must detect these oracle mutations or mark the affected run invalid.
- Preserve immutable raw inputs, candidate binary digest, evaluator configuration, environment, run output and adjudication record.
- A skipped test is not a pass. An indeterminate run is not a successful negative test.
- Never tune thresholds on the sealed holdout.

## Experimental design

Use paired runs over identical fixtures and runtime targets for native vendor validation and MON augmentation. Randomize run order where possible. Separate infrastructure clusters, seeds and failure domains in the statistical model. Report per-scenario results and confidence intervals, not only pooled averages.

## Anti-gaming properties

- Returning indeterminate for all inputs fails decision coverage.
- Blocking all inputs fails positive-control sensitivity.
- Qualifying all inputs fails critical negative controls.
- Invalidating all claims fails invalidation precision.
- Invalidating no claims fails invalidation recall.
- A benchmark result is invalid if the oracle, corpus or candidate digest is missing.

## Explicit non-claims

This artifact defines a proposed benchmark protocol and deterministic challenge cases. It does not implement a runner, establish a live vendor comparison, demonstrate a mutation score, establish production readiness, or claim SOTA. All thresholds are proposed engineering gates until measured and independently replicated.
