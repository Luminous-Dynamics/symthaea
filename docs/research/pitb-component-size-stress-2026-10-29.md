# PITB component-size and estimand stress screen

**Date:** 2026-10-29  
**Status:** executed exploratory synthetic calibration; not a qualified inference method, not a production Rust result, and not a theorem-prover result.

## Why this experiment exists

The PITB weak-null calibration notes already identify two unresolved assumptions: (1) what population the component bootstrap is intended to generalize to, and (2) how results change under alternative connected-component size distributions. I added a separate research-only NumPy simulation to probe those assumptions before any real transfer results are unsealed.

The paradigm shift is to treat the **estimand and dependence graph as part of the benchmark contract**, not merely reporting choices. A target-weighted effect across a frozen set of theorem targets and an equal-component-weighted effect answer different questions when component sizes are unequal and treatment effects vary with component size. Both can be informative, but neither may be silently substituted for the other.

## Reproduction

From the repository root:

```bash
python3 scripts/research/pitb_component_size_stress_v0.py \
  --studies 1000 \
  --bootstrap-reps 299 \
  --seed 20261029 \
  --out /tmp/pitb-component-size-stress-screen.json
```

The screen uses Python 3.13.5, NumPy 2.3.5, seed 20261029, 1,000 simulated studies per cell, and 299 component-bootstrap resamples per study. The model uses baseline success mean 0.35, component ICC 0.10, paired-outcome association fraction 0.50, and requested weighted component-effect standard deviation 0.08. Each synthetic study centers the true target-weighted effect to zero and samples paired binary potential outcomes. A two-sided percentile interval is computed by resampling whole components; the primary diagnostic gate requires an informative interval, an observed effect of at least +10 percentage points, and a positive lower endpoint.

The six size profiles vary from 20 to 60 components and include balanced, moderate-tail, heavy-tail, one-giant-component, and larger/smaller-cluster cases. Within each profile, component effects are either generated independently of size or deliberately aligned with log(component size), then centered under target weighting. These are adversarial synthetic conditions, not estimates of PITB's actual relationship graph.

## Results

The following are unconditional rates over the 1,000 simulated studies. Coverage is conditional on an informative interval; all intervals happened to be non-degenerate in this screen. Wilson intervals quantify Monte Carlo noise only.

| Profile | Effect vs. size | Components / targets | Largest component's target share | 95% interval coverage | Lower-bound rejection | False primary-gate pass |
|---|---|---:|---:|---:|---:|---:|
| Balanced 40 | Independent | 40 / 160 | 2.5% | 95.3% | 2.5% | 0.3% |
| Balanced 40 | Size-correlated | 40 / 160 | 2.5% | 94.1% | 3.5% | 0.2% |
| Moderate tail | Independent | 40 / 84 | 11.9% | 93.8% | 2.2% | 1.9% |
| Moderate tail | Size-correlated | 40 / 84 | 11.9% | 93.6% | 1.4% | 0.9% |
| Heavy tail | Independent | 40 / 120 | 12.5% | 94.8% | 2.4% | 0.8% |
| Heavy tail | Size-correlated | 40 / 120 | 12.5% | 93.5% | 2.3% | 0.9% |
| One giant component | Independent | 40 / 100 | 45.0% | 93.9% | 3.0% | 1.7% |
| One giant component | Size-correlated | 40 / 100 | 45.0% | 93.4% | 0.7% | 0.4% |
| Few clusters | Independent | 20 / 60 | 25.0% | 91.9% | 4.3% | 3.8% |
| Few clusters | Size-correlated | 20 / 60 | 25.0% | 90.7% | 1.3% | 1.0% |
| More clusters | Independent | 60 / 180 | 8.3% | 94.3% | 2.3% | 0.3% |
| More clusters | Size-correlated | 60 / 180 | 8.3% | 93.1% | 1.8% | 0.3% |

Selected Monte Carlo 95% Wilson intervals for the lower-bound rejection rate were 3.21–5.74% for the 20-component independent-effect case and 2.11–4.25% for the 40-component one-giant-component case. For the primary utility gate, the 20-component independent-effect rate was 3.8%, with Wilson interval 2.78–5.17%. The first interval's upper endpoint exceeds 5%, and the second is near the nominal threshold; therefore these results **do not establish** the prescribed simultaneous type-I error gate.

## Most important finding: two estimands can diverge

In the size-correlated scenarios, the true target-weighted treatment effect was centered to exactly zero in every generated study (up to floating-point representation), while the mean true equal-component effect was:

| Profile | True equal-component effect while target-weighted effect is zero |
|---|---:|
| Moderate tail | -4.94 percentage points |
| Heavy tail | -6.27 percentage points |
| One giant component | -6.83 percentage points |
| Few clusters | -6.83 percentage points |
| More clusters | -5.56 percentage points |

This is not a statistical bug. It is a change of estimand: equal-component weighting gives a singleton component the same weight as a large connected component, while target weighting gives components weight in proportion to the number of target theorems. PITB should freeze its primary target-weighted operational estimand and report equal-component weighting only as a clearly labelled diagnostic.

## Decisions and next gates

1. Keep the current 20-component minimum as a structural floor only. In these synthetic cases, 20 components were associated with the widest calibration shortfall and the highest observed false utility-gate rate. That does not prove the real PITB design fails, but it prevents treating the component count as sufficient evidence.
2. Before the test split is opened, compute the exact audited component-size vector, largest-component share, size coefficient of variation, effective sample-size diagnostics, and target count after exposure/eligibility quarantines. Use precisely that manifest for both power analysis and final inference.
3. Preregister a broader simulator grid over component count, size CV, maximum component share, baseline success, ICC, effect heterogeneity, effect-size correlation with component size, and paired-outcome association. Include selective missingness and alternate data-generating families.
4. Compare at least one small-sample-corrected method and a randomization-based method only where its assignment/exchangeability assumptions are actually met. The percentile component bootstrap is a candidate, not an endorsed universal solution.
5. Obtain independent statistical review, and test the production Rust implementation separately. This exploratory NumPy harness cannot qualify itself or substitute for exact-head CI evidence.

## External methodology context

Recent cluster-randomized trial literature also cautions that incomplete outcomes and unequal cluster structure require explicit assumptions and sensitivity analysis; double-robust estimators do not make non-ignorable missingness identifiable without further assumptions. See [Wang, Li and Wang, Biometrics (2026)](https://academic.oup.com/biometrics/article-abstract/82/1/ujag030/8494909) and the [systematic review of missing-data handling in cluster trials](https://journals.sagepub.com/doi/10.1177/17407745251378117). PITB is not a clinical trial, but the lessons about clustered sampling, missingness and estimand definition apply at the level of statistical design.

**Evidence boundary:** this is a reproducible, synthetic research screen. It does not authenticate PITB's dependency graph, establish validity of the current bootstrap procedure, report a proof-insight transfer, or claim SOTA performance.
