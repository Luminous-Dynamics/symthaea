# AOBENCH-013B: suite-level stochastic operating-characteristic metrics

**Date:** 2026-11-15  
**Parent:** [AOBENCH-013 / #12197](https://github.com/Luminous-Dynamics/symthaea/issues/12197)  
**Per-episode evaluator:** [AOBENCH-013A / PR #12216](https://github.com/Luminous-Dynamics/symthaea/pull/12216)  
**Implementation issue:** [#12217](https://github.com/Luminous-Dynamics/symthaea/issues/12217)

## Why a suite-level scorer is required

An individual sequential episode can show whether a particular trace respected its stopping boundary. It cannot estimate type-I error, power, confidence-sequence coverage or resource cost across a population of stochastic worlds. These are suite-level operating characteristics, not properties that can be inferred from one successful trace.

Recent work such as petri-bench (working draft v0.5) explicitly separates correctness, method rigor and efficiency, and audits repeated-testing / p-hacking and probe-only shortcuts. Its reported 540-episode study also shows why a single solve-rate score is not a sufficient scientific-method metric. AOBENCH therefore keeps its rates and integrity gates separate.

## Metrics and denominators

- **False-positive rate on null episodes:** false directional claims divided by evaluator-valid episodes whose true risk difference lies inside the frozen minimum-effect margin.
- **Qualified power on materially non-null episodes:** correctly claimed and process-qualified directional decisions divided by evaluator-valid episodes whose true effect is outside that margin. Abstentions, invalid traces and missed boundaries do not count as successful detections.
- **Final interval coverage:** final treatment-minus-control interval coverage divided by coverage-eligible episodes.
- **All-look confidence-sequence coverage:** whether every interim interval included the evaluator truth, divided by coverage-eligible episodes.
- **Correct claim rate:** truth-matched terminal claims divided by all evaluator-valid episodes; abstentions are not claims and do not count as correct.
- **Process qualification:** qualified episodes divided by all attempted episodes, including invalid-oracle episodes.
- **Resource means:** attempted samples, cost and risk per attempted episode, including failed and unqualified attempts.

Invalid evaluator-oracle episodes remain in process totals but are excluded from truth-conditioned rates. An episode with an unavailable or rejected observation is not coverage-eligible for the confidence-sequence rate. Episodes with zero qualified observations are also excluded from coverage denominators: a vacuously full-width interval with no interim look must not reward an always-abstain policy.

## Uncertainty

The summary reports 95% Wilson score intervals for every proportion. This avoids the poor small-sample behavior of a naive normal approximation. Wilson intervals are descriptive episode-level uncertainty intervals; they do not correct for shared generator families, repeated seeds from the same world, benchmark contamination or selection bias.

Confirmatory analysis still needs family-level splits and cluster-aware uncertainty. Do not treat episodes derived from a common SCM family or shared seed hierarchy as automatically independent.

## No composite leaderboard score

False-positive control, power, final coverage, all-look coverage, process qualification, correct-claim rate and total cost are not collapsed into a weighted score. A high power number cannot compensate for uncontrolled false positives. A good final-look coverage rate cannot hide poor all-look coverage. A truthful claim from an invalid process does not become a qualified discovery.

## Implementation and limits

The scorer consumes the per-episode reports produced by the Bernoulli evaluator. It does not generate random episodes, authenticate evidence, execute interventions, or prove the correctness of the per-episode confidence sequence. The deterministic unit tests verify denominators and aggregation mechanics, not empirical type-I error or power.

No benchmark score or SOTA superiority claim is made. Monte Carlo calibration, exact-head nested-workspace checks, family-disjoint confirmatory tests and comparison with fixed-sample / sequential / confidence-sequence baselines remain pending.
