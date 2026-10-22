# Robust Information-Gain Noise Envelope (v0.1)

**Status:** exploratory implementation; no performance or SOTA claim.

## Hypothesis

A single plug-in observation-noise estimate can make experiment rankings brittle. A candidate may look best at a nominal low noise level but perform poorly if the real measurement process is noisier. A small, explicit uncertainty set over noise scales lets the selector prefer an experiment whose informativeness degrades less severely across plausible measurement conditions.

This is intentionally a narrow step beyond the nominal expected-information-gain selector. It does not reproduce R-IDeA, maximin robust Bayesian design, or any other published method.

## API

- `expected_information_gain_worst_case_nats` evaluates the same experiment under every supplied observation-noise standard deviation and returns the minimum information gain.
- `select_experiment_by_worst_case_information_gain_per_cost` ranks candidate experiments by that minimum divided by positive finite cost.

The noise envelope is a finite caller-supplied list, not an interval guarantee. Invalid/empty noise scenarios fail closed. Incomplete or non-finite predictions remain ineligible. The functions preserve deterministic input-order tie-breaking and do not alter the existing variance or nominal EIG selectors.

## Preconditions and non-claims

The underlying information-gain estimator still assumes a uniform prior over the supplied hypothesis instances, a common Gaussian observation-noise model within each scenario, and a hypothesis set that supplies finite predictions. Repeated/equivalent hypotheses can affect the implicit prior through multiplicity. A supplied envelope only tests the listed noise scales; it does not imply coverage between or beyond them.

Worst-case noise scoring does **not** solve hidden-variable or structural misspecification. Keep discrepancy audits and held-out validation separate: an empty candidate set, a confident singleton, and an unresolved set are different outcomes and all must be reported. Never turn a conservative acquisition score into a calibrated confidence claim.

## Local regression design

The focused Rust tests cover:

1. The robust score is no larger than any included scenario score.
2. A rank-reversal example where low-noise nominal EIG favors a four-way split, but the worst-case envelope favors a higher-signal-variance experiment when the noise scenario rises sharply.
3. Empty and invalid noise envelopes fail closed.

A separate numeric translation of the rank-reversal case predicts these scenario scores (nats): at standard deviation 0.1, candidate A ≈ 0.562 and B ≈ 1.386; at standard deviation 5.0, A ≈ 0.0326 and B ≈ 0.0244. This calculation verifies the intended test geometry only; it is not a Rust build or CI result.

## Benchmark protocol

Freeze the candidate pool and random seeds across all policies. Compare at minimum:

- seeded uniform selection;
- raw prediction variance;
- nominal EIG at a single estimated noise scale;
- worst-case EIG over a predeclared noise envelope;
- EIG plus scheduled audits;
- worst-case EIG plus the same audit schedule.

Use held-out discovery worlds clustered by structural family, not a random-seed-only split. Sweep both correctly specified and misspecified worlds, noise-scale calibration error, omitted hypotheses, heterogeneous noise, shifted experiment/query distributions, and experiment costs. Report full structural recovery, held-out transfer error, scientifically valid recovery, false confident-singleton rate, unresolved/empty-set rates, cost and queries to valid recovery, selector regret versus an evaluator-only oracle, runtime, and calibration/abstention.

Proposed confirmatory target (not a result): at least 20% lower median cost-to-valid-recovery than nominal EIG across held-out worlds, paired world-level 95% interval excluding zero, with no more than a 1 percentage-point scientific-validity regression. Under misspecification, separately target at least 20% relative reduction in false confident singletons while bounding clean-world recovery degradation to 5 percentage points. Publish negative results and all failure modes.

Initial matrix: at least 60 held-out worlds across 6 or more structural families, five noise/misspecification regimes, and at least 20 paired seeds per world. Treat seeds as repeated runs within worlds; use hierarchical/clustered intervals at world level and predeclare exclusions.

## External comparison and reading

Benchmark context must remain protocol-matched. NewtonBench is an interactive scientific-law discovery suite (324 tasks across 12 physics domains); SciLaws-Bench reports distinct numerical-fit, scientific-validity and structural-recovery dimensions. Their leaderboard scores are not directly comparable to this Rust simulator.

Research informing the direction:

- Tang, Sloman & Kaski, *Representative, Informative, and De-Amplifying: Requirements for Robust Bayesian Active Learning under Model Misspecification*, AISTATS 2026: https://proceedings.mlr.press/v300/tang26d.html
- Callahan & Catanach, *On the Misinformation in a Statistical Experiment*, AISTATS 2026: https://proceedings.mlr.press/v300/callahan26a.html
- Abdulsamad et al., *Maximin Robust Bayesian Experimental Design* (2026): https://arxiv.org/abs/2603.14094
- NewtonBench: https://arxiv.org/abs/2510.07172
- SciLaws-Bench: https://github.com/InternScience/SciLaws-Bench

The near-term scientific question is whether conservative acquisition improves outcome quality and false-confidence behavior at fixed cost—not whether its information score is numerically elegant.
