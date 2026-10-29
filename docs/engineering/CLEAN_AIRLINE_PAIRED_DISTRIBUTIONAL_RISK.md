# Clean Airline Paired Distributional Risk Bridge

**Workstream:** CLEAN-AVIATION-001 / CLEAN-AVIATION-002  
**Status:** Draft implementation slice; code and tests authored, not executed in this environment  
**Stack:** `research/clean-airline-paired-distributional-risk-001` on the paired-policy benchmark branch  
**Date:** 2026-10-29

## Why this slice exists

The paired-policy benchmark reports scenario design weights, paired costs and feasibility coverage. The distributional-robustness module evaluates cost deltas under a nominal probability distribution and a total-variation (TV) ambiguity radius. Those inputs are not interchangeable: a synthetic panel's design weights are not calibrated probabilities.

This slice creates a fail-closed bridge between the two modules. It makes the probability source, finite scenario support, cost quantization, risk radius, policy pair and source benchmark digest explicit.

## Contract

1. Compare only static, greedy and rolling-horizon policies. The perfect-information exact optimizer remains an optimistic oracle, not a deployable policy.
2. Require a nonempty source benchmark digest and the exact baseline scenario whose canonical digest is bound by that report.
3. Require a calibration-frequency table from a frozen calibration partition. Its scenario IDs must exactly match the paired outcome IDs. Duplicate or missing support is rejected.
4. Derive the TV radius by rerunning the repository's calibration function over the supplied frequencies and predeclared failure probability. The risk-gate radius must equal that calibrated radius; it cannot be selected after observing the holdout outcomes.
5. Ignore the benchmark's design weights for probability purposes. Use the calibration table's explicit nominal probabilities instead.
6. Require the exact realized scenario manifests and recompute each canonical scenario digest. Independently re-run the energy-hub plan verifier for both selected policies in every scenario, comparing the verifier's full metrics with the report. If a selected policy is infeasible, unresolved, absent, conflicting or fails independent re-verification in any scenario, do not calculate a distributionally robust cost score over a reduced subset. Report that policy's feasibility/infeasibility/unresolved weight through the paired benchmark's separate coverage fields.
7. Quantize floating-point currency costs using an explicit integer scale. The maximum candidate-minus-baseline rounding error is reported, and overflow / negative / non-finite costs fail closed.
8. Return the nominal and worst-case cost delta, scenario-level deltas, calibrated TV-radius report, configured risk assessment and a deterministic assessment digest.

## Important claim boundaries

- A content digest binds a report to bytes/fields; it does not authenticate who supplied the data, prove execution, or prove that calibration episodes are representative.
- The TV result is conditional on the declared finite support, nominal probabilities and calibrated radius. It is not a guarantee for omitted scenarios or arbitrary distribution shift.
- The calibration module's sampling allowance assumes independent observations from one fixed finite-support distribution. Temporal dependence, regime changes and missing scenario categories need separate treatment.
- An output marked eligible means only that the configured finite-support cost gates were met. It does not establish aircraft or equipment safety, physical-model validity, causal savings, climate improvement or authority to control airport assets.
- If a policy lacks a feasible result in any case, that is evidence to preserve in coverage reporting—not a zero cost and not an invitation to reweight the surviving cases.

## New test definitions

The module includes seven unit-test definitions intended to exercise:

- calibrated probabilities are used instead of the benchmark's design weights;
- a missing policy result rejects the full-support risk evaluation;
- calibration/evaluation support mismatch is rejected;
- an unrelated baseline scenario digest is rejected;
- realized-scenario support mismatch is rejected;
- a policy plan whose reported accounting metrics have been tampered with is rejected by independent re-evaluation; and
- a risk gate with a radius different from the calibrated radius is rejected.

These are **authored tests, not passing tests**. This runtime has no `cargo`, `rustc` or `rustfmt`, so no compilation or Rust test execution is claimed.

## Next evidence gates

1. Execute this module's tests and the complete stacked aviation suite against the exact published head SHA, not merely the latest branch name.
2. Add independent known-answer cases for quantization boundaries, probability-mass errors, duplicate IDs, all three policy comparisons and infeasible/unresolved status mappings.
3. Verify the assessment digest changes for changes to any calibration count/probability, cost scale, threshold, chosen policy, source benchmark digest or paired cost.
4. Use real historical calibration counts only after source rights, event semantics, chronological splits, event-cluster leakage and use scope have been checked by the existing data contract.
5. Compare the adapter with an independently implemented numerical oracle on hand-computable TV-ball cases. Preserve the full-support restriction; do not turn a selected feasible subset into a population-level risk claim.
6. Keep synthetic, replayed historical, simulated physical-system and field-measured evidence as separate classes.

No public data ingestion or scoring has begun for this work. The adapter improves the evaluation chain; it does not claim airport savings or SOTA performance.
