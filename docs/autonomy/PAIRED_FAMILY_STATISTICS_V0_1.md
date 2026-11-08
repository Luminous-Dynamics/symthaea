# Family-clustered paired controller statistics v0.1

**Status:** deterministic statistics implementation; no driving performance results reported  
**Code:** `symthaea_vehicle::paired_statistics::summarize_paired_metric`

## Problem

A paired controller benchmark may contain many parameter variants of one scenario family. Treating every variant as an independent observation can make confidence intervals artificially narrow because the variants share geometry, actor-intent patterns, generation procedures and other common causes.

This module treats the declared failure family as the bootstrap resampling unit.

## Input contract

Each `PairedMetricObservation` contains:
- failure-family ID;
- globally unique scenario ID;
- advanced-controller metric value;
- baseline-controller metric value.

The caller must use the same candidate/route/exogenous conditions for the advanced and baseline values within each observation. The module validates identifiers and finite numeric values; it cannot authenticate that experimental pairing itself.

Direction is explicit:
- `LowerIsBetter`: improvement = baseline − advanced.
- `HigherIsBetter`: improvement = advanced − baseline.

Positive improvement always favors the advanced controller.

## Estimator

1. Group observations by failure-family ID.
2. Average advanced and baseline metric values inside each family.
3. Compute the paired improvement for each family.
4. Give each family equal weight in the macro-average, regardless of how many sibling variants it contains.
5. Bootstrap family means with replacement, drawing the same number of families per replicate.
6. Report the 2.5th and 97.5th percentiles of the deterministic bootstrap distribution.

The PRNG is a local SplitMix64 implementation with rejection sampling for unbiased bounded indices. Means divide each value before summation to avoid overflow for otherwise finite metrics; unrepresentable differences are rejected. The seed and bootstrap replicate count are included in the summary.

## Artifact integrity

The summary retains the exact paired input observations in canonical family/scenario order. Its validator:
- checks unique scenario identities and finite values;
- reconstructs each family’s scenario membership and advanced/baseline means;
- recomputes family improvements from the declared direction;
- recomputes macro mean, median, family win fraction and worst-family improvement;
- reruns the seeded family bootstrap and reconstructs its interval;
- checks the summary digest.

The digest is a BLAKE3 integrity fingerprint, not a signature, authentication or proof that the input observations came from a valid execution backend.

## Interpretation limits

The interval describes uncertainty across the declared set of failure families under the bootstrap assumptions. It is not a fleet-level safety interval, an estimate of event prevalence or a substitute for a representative sampling design. Families must be meaningfully distinct and defined before looking at final-holdout outcomes. If the family taxonomy is incomplete or highly dependent, the interval may still be overconfident.

This statistic does not combine multiple metrics into one score. Tracking error, progress, effort, safety constraints and runtime remain separate. The caller must pre-register non-inferiority margins and any selection rules.

For broad robustness statements, SAM's proposed evaluation target remains at least ten disjoint failure families and 10,000 executed holdout scenarios, with per-family results and denominators exposed.

## Controls

Nine unit tests are authored for:
- sibling aggregation before bootstrap;
- positive-improvement sign convention in both metric directions;
- deterministic replay for a frozen seed, independent of input row order;
- duplicate scenario rejection;
- insufficient-family and non-finite-value rejection;
- robustness to extreme but finite inputs and rejection of an unrepresentable paired difference;
- inconsistent aggregate rejection even after the summary digest is recomputed.

Tests have not been compiled or executed in this session.
