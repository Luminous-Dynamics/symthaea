# Vocal no-op tolerance calibration benchmark (VNOC-001)

**Date:** 10 November 2026  
**Status:** standalone distribution-free calibration utility and protocol; synthetic tests only

## Research question

How much can an acoustic metric change when the musical performance has not changed?

VCRB-001 needs a tolerance for every collateral metric. A threshold chosen after inspecting confirmatory results can hide failures. A threshold estimated from a handful of repeats can substantially understate tail variability. Testing many metric thresholds separately also creates a family-wise confidence problem.

VNOC-001 provides a predeclared, reproducible calibration rule.

## Statistical approach

For each metric, gather absolute differences from matched no-op repeats. The sampling unit must be an independent, representative calibration unit, not every overlapping analysis frame.

Use the sample maximum as a one-sided, distribution-free upper tolerance bound. For independent observations from a continuous population, if p is the minimum population coverage and n is the sample count, the confidence that the sample maximum covers at least p of the population is:

~~~text
1 - p^n
~~~

To obtain per-metric confidence 1 - alpha, choose:

~~~text
n >= ln(alpha) / ln(p)
~~~

For m simultaneously guarded metrics, VNOC allocates total family-wise alpha across metrics using Bonferroni:

~~~text
alpha_metric = alpha_family / m
~~~

This does not assume the different acoustic metrics are independent. The family-wise confidence lower bound is the Bonferroni union bound:

~~~text
1 - m * p^n
~~~

The calibrator rejects a sample set below the required n. For 95% population coverage and 95% family-wise confidence, the current protocol requires 59 independent calibration units for one metric and 104 for ten metrics.

References on nonparametric one-sided tolerance bounds and order statistics: JMP's tolerance-interval documentation and Minitab's exact nonparametric tolerance-interval method. These sources distinguish tolerance bounds (covering a population fraction with confidence) from confidence intervals around a mean.

## Engineering calibration policy

The statistical bound is the maximum absolute no-op delta. The policy can then apply a predeclared safety multiplier (not less than 1.0) and an engineering minimum floor. These are explicit policy parameters, not statistically estimated facts.

Calibrate separate strata whenever the distribution can change materially:

- renderer/backend and exact model/voicebank digest;
- analysis implementation/version and metric definition;
- singer/voice identity and register;
- language/dialect and phonetic context where relevant;
- sample rate, channel policy and preprocessing.

A calibration unit must be defined before collection. Recommended unit: an independently selected source take × performer/voice × relevant register × renderer artifact stratum, with matched no-op re-renders and a fixed analysis pipeline. Multiple windows from one take must not be misrepresented as independent units.

## Executable implementation

The dependency-free Rust crate at tools/vocal-noop-calibration:

- validates metric IDs, units and calibration-unit identity;
- requires every metric to cover the same set of unique calibration units;
- rejects missing, extra, duplicated, negative, non-finite and unit-mismatched observations;
- computes exact minimum sample size for requested coverage and confidence;
- applies Bonferroni error allocation across the metric family;
- computes the maximum observed absolute no-op delta, explicit safety margin and engineering floor;
- emits a complete calibration report with sample count, required sample count, per-metric confidence bound and family-wise lower bound.

It cannot prove independence or representativeness. Those are protocol responsibilities and must be documented in the calibration bundle.

## Relationship to VCRB-001 and VELB-001

- **VCRB-001** consumes the calibrated tolerance for every off-target acoustic metric.
- **VELB-001** measures waveform-level edit locality and requires renderer-specific protected-region tolerances.
- **VNOC-001** provides the statistically explicit calibration path for those tolerances; waveform and acoustic metrics should each be calibrated on their own correctly defined no-op samples.

Freeze and content-address the calibration report before confirmatory control interventions. If the renderer artifact, voicebank, analyzer, metric semantics or preprocessing changes, the corresponding calibration expires.

## Claim boundary

Synthetic tests establish arithmetic and fail-closed input validation only. They do not demonstrate that acoustic measurements are stable, that calibration strata are representative, or that Muse follows its controls. No SOTA or naturalness claim follows from this utility.

## References

- JMP, Statistical Details for Tolerance Intervals, nonparametric one-sided tolerance bounds: https://www.jmp.com/support/help/en/19.2/jmp/statistical-details-for-tolerance-intervals.shtml
- Minitab, Exact nonparametric tolerance intervals for continuous distributions and one-sided order-statistic bounds: https://support.minitab.com/en-us/minitab/help-and-how-to/quality-and-process-improvement/quality-tools/how-to/tolerance-intervals-nonnormal-distribution/methods-and-formulas/tolerance-intervals/
- NIST, Engineering Statistics Handbook, tolerance intervals and distinction between confidence and population coverage: https://www.itl.nist.gov/div898/handbook/
