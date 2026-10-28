# PITB weak-null calibration stress test (2026-10-28)

**Status:** exploratory independent simulation; not production inference, not Rust output, not repository CI, and not a proof-transfer result.

## Question

Does a lower-bound test used to qualify transfer utility retain reasonable behavior when the *target-weighted average effect is exactly zero*, but individual components have effects of different magnitudes and signs? This is a harder diagnostic than a strict null that sets every component effect to zero.

## Setup

- Binary paired potential outcomes; baseline success probability 0.35 and beta-binomial component difficulty with ICC 0.05.
- Component-specific effect heterogeneity is symmetric in sign and calibrated so the finite-target weighted expected average is zero.
- Pair-association fraction interpolates from independent potential outcomes to the Fréchet upper bound; it is not a Pearson correlation.
- 2,000 synthetic studies per configuration, 999 component-bootstrap replicates per study, and fixed published seeds in the accompanying local research artifact.
- Compared (1) target-weighted component-percentile bootstrap, (2) a simple cluster-paired t interval, and (3) an exploratory restricted wild-cluster bootstrap-t using Webb weights. The wild-cluster implementation has no leverage-adjusted residual correction and is a comparator only, not a recommended production procedure.
- The observed criterion is rejection when the nominal two-sided 95% interval's lower endpoint is above zero; the reference rate is therefore 2.5%. Degenerate/unavailable results are never counted as passes.

## Selected results

| Synthetic scenario | Method | Zero-null coverage | Lower-bound rejection | 95% Monte Carlo interval for rejection |
|---|---|---:|---:|---:|
| 20 balanced-ish components / 47 targets; effect amplitude 0.08; pair association 0.5 | Component percentile bootstrap | 94.25% | 3.40% | 2.69–4.29% |
| Same | Cluster-paired t | 95.05% | 2.80% | 2.16–3.62% |
| Same | Restricted wild-cluster t | Not computed by this harness | 2.60% | 1.99–3.39% |
| 20 unequal components / 80 targets; effect amplitude 0.08; pair association 0.5 | Component percentile bootstrap | 93.25% | 3.15% | 2.47–4.01% |
| Same | Cluster-paired t | 89.65% | 4.95% | 4.08–5.99% |
| Same | Restricted wild-cluster t | Not computed by this harness | 6.10% | 5.13–7.24% |
| 50 balanced-ish components / 118 targets; effect amplitude 0.08; pair association 0.5 | Component percentile bootstrap | 94.55% | 3.00% | 2.34–3.84% |
| Same | Cluster-paired t | 94.60% | 3.00% | 2.34–3.84% |
| Same | Restricted wild-cluster t | Not computed by this harness | 3.10% | 2.43–3.95% |

In the 20-component high-association sensitivity case (association fraction 0.9), the component-percentile interval was available in 96.9% of studies; coverage conditional on availability was 96.75%, and the rejection rate across all studies was 1.65%. This illustrates why interval availability must be reported beside conditional coverage: suppressing degenerate intervals can make the surviving intervals appear well calibrated while leaving some studies unresolved.

The Monte Carlo intervals quantify simulation noise only under this one data-generating family. They do not include uncertainty about whether the generator resembles PITB's eventual task population.

## Interpretation and decisions

1. **Do not declare a universal winner.** The t interval was better in one balanced case, while the percentile bootstrap was materially better than both comparators in the unequal-size case. None is qualified across the intended range.
2. **Component-size assumptions are part of the estimand.** Resampling components changes the observed size mix. If the planned task/component vector is fixed rather than a sample from a component superpopulation, the ordinary cluster bootstrap may target the wrong sampling process. Freeze the estimand and size-generating process before selecting an interval.
3. **Report three denominators separately:** all planned studies, interval-available studies, and intervals used for coverage/rejection. Never convert a suppressed interval to a pass or silently drop it.
4. **A small number of components is not repaired by a large target count.** Inference must be calibrated against the number of independent components and their size/leverage distribution.
5. **The wild-cluster result is a warning, not a recommendation.** This implementation deliberately lacks some small-sample corrections and should not be compared as though it were a qualified method.

## Next calibration gate

Before the confirmatory PITB run, use the actual eligible component-size vector once the relationship/quarantine audit is complete; document whether the target is a fixed finite collection or a sample from a broader task population. Then run a preregistered grid with at least 10,000 studies per core cell covering balanced and highly unequal sizes, different component counts, rare and common baseline success, several ICCs, negative/positive effect heterogeneity, and dependence. Publish Monte Carlo intervals for size and coverage, unconditional and conditional interval availability, power at the registered utility effect, and interval width.

Compare the current percentile method with at least one small-sample-corrected alternative and a randomization-based test only where its assignment/exchangeability assumptions are justified. An independent statistician should review the final design. These are proposed qualification steps, not an assertion that the gates have already passed.

## Reproduction boundary

The independent Python 3.13.5 / NumPy 2.3.5 / SciPy 1.17.0 implementation and complete JSON/CSV output are in the accompanying generated research pack for this work session. This note is a research record, not a second production scorer. The Rust power-planning and analysis binaries must be tested on the exact PITB head before their outputs are treated as project evidence.

## Connection to benchmark validity

This work qualifies only the statistical decision layer. It says nothing about model capability or SOTA. PITB's separate semantic gates remain necessary: exact target-statement correspondence, audited dependencies/axioms, negative mutations, family leakage quarantine, pinned checker/prover/resource budgets, and content-addressed receipts. A statistical interval cannot turn invalid or leaked proof outcomes into valid successes.


## Production-scoring integration update (2026-10-28; unverified source change)

The primary analyzer source now contains a separate weak-null power grid: requested target-weighted family-effect SDs of 5, 10 and 20 percentage points crossed with ICC values 0, 0.05, 0.10 and 0.20. Each simulated study centers its family effects to a zero target-weighted mean and uses a single common feasibility scale to keep treatment rates in [0,1]. The report records false-claim counts/rates, degenerate intervals and realized effect-SD diagnostics; analyze mode is authored to replay every grid cell from the validation-derived family-size vector and frozen seed. The report schema was advanced to `pitb.power-report.v1`, and the confirmatory power screen now requires both strict-null and heterogeneous weak-null upper bounds to be no greater than 5%.

This is the **source implementation contract**, not a replacement for the independent simulation tables above. No Rust compiler or exact-head Actions result was available when this change was authored; the new simulation and replay tests are present in source but unexecuted. In addition, the current Wilson bounds are per scenario rather than familywise simultaneous across the 12 heterogeneous cells, and the new model still assumes independent arm outcomes conditional on family difficulty. Pair-association, selective missingness, alternative component-size mechanisms and multiplicity-controlled simultaneous bounds remain required before treating this as qualified weak-null inference.