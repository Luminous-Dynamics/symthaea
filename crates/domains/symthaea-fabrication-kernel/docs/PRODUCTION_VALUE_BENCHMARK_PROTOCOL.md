# MFG-VALUE-001: Paired Production-Outcome Evaluation Protocol

## Purpose

Create a reusable evaluator for the question that matters: does an engineering/manufacturing intervention improve conforming output on the same workload and disturbance conditions without hiding safety, quality, or accounting failures behind a scalar score?

This is an implementation slice for [MFG-VALUE-001](https://github.com/Luminous-Dynamics/symthaea/issues/7608). It complements domain-specific work such as the FDM quality benchmark (#7960/#7961), the household-goods factory and supply-chain benchmark (#8053), and explicit manufacturing baseline calibration (#8015). It does not establish that any candidate improves production.

## Paradigm shift

A model or optimizer is not the product; the measured outcome is the product. Compare baseline and candidate on identical, frozen scenario IDs and workload identities. Preserve hard gates as non-compensatory constraints. Report the primitive outcomes independently; do not blend yield, energy, speed, safety, cost, and repairability into a weighted score that allows one dimension to conceal a regression in another.

The unit of evaluation is a paired scenario, not a cherry-picked best run. Scenario IDs should include seed and scenario-defining choices. The workload identity should cover work order, part mix, material/lot identity, starting state, process route, measurement boundary, and acceptance criteria. Each pair carries the same workload identity, input-unit count, and input material quantity.

## Implemented evaluator scope

The Rust module `symthaea_fabrication_kernel::production_value` provides a deterministic qualification boundary:

- Every run partitions all input units exactly into accepted, scrap, rework, and held units.
- Every run closes the declared material mass in integer milligrams across accepted output, scrap, rework, held material, and an explicit residual.
- Energy and elapsed time must be finite; duration must be positive; energy must be non-negative.
- Baseline and candidate scenario sets must match exactly. Duplicate scenarios, mixed policy labels, workload mismatches, malformed identities, arithmetic overflow, invalid counts, and non-conserving records reject the comparison.
- Summaries preserve units and boundaries. There is no blended score and no automatic claim of economic, environmental, or safety improvement.
- Candidate eligibility is a separate, non-compensatory result. Any critical safety violation, escaped nonconforming unit, or unresolved held unit makes the candidate ineligible.

This module is an arithmetic/structure oracle over supplied records. It is not a sensor adapter, authenticated metrology pipeline, causal estimator, optimizer, digital twin, or manufacturing controller. A valid report neither authenticates inputs nor proves that the outcomes are physical. Callers must retain source data, measurement methods, provenance, uncertainty, the exact software revision, and inclusion/exclusion rules.

## Minimum experimental protocol

1. Freeze the product/workload, acceptance specification, route, scenario list, seeds, measurement boundary, and stopping rule before running either policy.
2. Name/version the baseline and candidate. Give them identical input lots and scenario sequences. Use paired repetitions and disposition every planned case, including aborts and failures.
3. Prefer independent metrology. Distinguish sensor telemetry, model estimates, operator observations, and measured outcomes.
4. Reserve chronological and/or product-family holdouts. Do not tune on the final evaluation set. Prevent related runs from leaking across splits.
5. Report first-pass yield, conforming yield, scrap/rework/held fractions, energy per conforming unit, accepted units per machine-hour, uncertainty, and maintenance/downtime only when those data have actually been collected.
6. Report safety, escaped defects, and conservation failures as hard gates—not costs that can be offset by productivity.
7. Compare with simple transparent baselines before claiming added value from Symthaea, Symtropy, or Mycelix. Repeat at an independently reviewed site before generalizing.

## Proposed benchmarks (not achieved)

- **100% paired coverage:** every candidate run has one baseline run on the same scenario/workload; every planned case is dispositioned.
- **100% accounting closure:** unit partitions and the stated material boundary reconcile exactly at the declared integer units; otherwise no comparison is emitted.
- **Zero hard-gate violations:** no critical safety violations, escaped nonconforming units, or unresolved held units for candidate eligibility.
- **Prediction value:** on a preregistered real-data holdout, target at least 10% lower MAE than a named conventional baseline for each claimed forecast target/horizon, with uncertainty and all horizons reported.
- **Production value:** investigate a 5-percentage-point yield increase or 10% lower energy per conforming unit only where a powered paired physical pilot supports it without hard-gate regression.
- **Reproducibility:** frozen inputs and software version produce byte-identical structural reports; performance measurements separately record environment and p50/p95/p99 latency.

These are investigation thresholds, not universal specifications or achieved results. A synthetic fixture qualifies arithmetic, not production benefit. Do not decide that a target is meaningful until baseline variance, measurement uncertainty, sample size, and effect size are estimated.

## Negative controls

Reject or explicitly fail eligibility for duplicate scenario IDs; missing/extra scenarios; mixed policy labels; changed workload identity; changed fixed input quantities; unit partition off by one; material balance off by one milligram; zero/negative/non-finite duration; negative/non-finite energy; integer overflow; zero accepted units causing division by zero; critical safety violations; escaped nonconforming units; or held output. Future forecasting comparisons should additionally include label-shuffle controls and report matched conventional baselines. Those statistical tests are not implemented by this arithmetic module.

## Paired uncertainty without pseudo-replication

The companion `paired_inference` module provides a deterministic percentile cluster-bootstrap interval for the mean paired difference. It requires at least 20 independently sampled clusters and at least 1000 resamples. It records the seed, resample count, confidence level, mean and median difference, interval bounds, and the number of clusters where the candidate is better/worse/tied under a declared metric direction.

**Cluster before bootstrap.** The caller must aggregate observations inside a genuinely independent unit (for example, material lot, production day, or independent site) and provide one paired value per unit. Repeated pieces, time samples, or multiple scenarios from one lot must not be passed as separate independent clusters. Cluster IDs must be unique and paired; the function rejects duplicates and non-finite values. The interval is over an equal-cluster-weighted mean difference.

This is a percentile bootstrap description of uncertainty, not a causal identification procedure or a statistical proof of superiority. It emits no p-value and does not decide eligibility by whether a confidence interval excludes zero. The independence assumption, study design, missingness, stopping rule, measurement uncertainty, and direction of each metric must be justified outside the helper. The minimum cluster count is a conservative project gate, not a universal theorem; small or highly heterogeneous studies may require substantially more data or another preregistered method.

`PairedBootstrapConfig::default()` uses 10,000 bootstrap replicates and a fixed reproducible seed. Always report the exact configuration and cluster definition.

## Research anchors

- [NIST Smart Manufacturing Systems Test Bed](https://www.nist.gov/laboratories/tools-instruments/smart-manufacturing-systems-sms-test-bed): integrated CAD/CAM, machine tools, inspection, MTConnect data, and lifecycle information for manufacturing evaluation.
- [NIST Digital Twins for Advanced Manufacturing](https://www.nist.gov/programs-projects/digital-twins-advanced-manufacturing): interoperability, verification/validation, and quantified model uncertainty.
- [NASA-STD-7009B](https://standards.nasa.gov/standard/NASA/NASA-STD-7009): active modeling-and-simulation credibility standard covering verification, validation, and uncertainty.

These sources motivate the evaluation design; they do not validate this code or demonstrate a production improvement.

## Local verification

```bash
cargo fmt --check
cargo test -p symthaea-fabrication-kernel production_value
cargo clippy -p symthaea-fabrication-kernel --all-targets --all-features -- -D warnings
```

A passing unit-test suite qualifies the evaluator's logic only. It does not constitute a physical manufacturing trial or external validation.
