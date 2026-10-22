# Expected-Information-Gain Experiment Selection v0.1

**Status:** research implementation; focused CI is the validation gate. No superiority result is claimed by this proposal.

## Research question

The existing selector ranks candidate experiments by the variance of deterministic predictions across surviving hypotheses. That is a useful disagreement heuristic, but variance is not the same thing as expected reduction in uncertainty about which hypothesis generated an observation. A single extreme outlier can dominate variance even when a balanced experiment separates more of the plausible hypotheses. Variance also has no native way to price measurements.

This change adds an alternative selector, without replacing the current heuristic:

- `expected_information_gain_nats` estimates mutual information between a uniformly distributed live-hypothesis identity and a future observation.
- `select_experiment_by_information_gain_per_cost` ranks candidate experiments by expected information gain divided by user-supplied cost.
- Both fail closed for non-finite/incomplete predictions; the cost-aware selector skips candidates with invalid costs.
- Existing raw-variance APIs remain intact to preserve comparability and permit ablation.

## Model and numerical method

For candidate experiment `x`, hypothesis (h_i) predicts mean μ_i(x). Given common observation standard deviation σ > 0, assume:

```text
H ~ Uniform(live hypotheses)
Y | (H = h_i, x) ~ Normal(mu_i(x), sigma^2)
score(x) = I(H; Y | x) = E[log p(Y | H, x) / p(Y | x)]
```

The score is reported in nats. The current implementation approximates the Gaussian expectation using 256 deterministic midpoint quadrature points spanning ±8 standard deviations. It uses a log-sum-exp likelihood-mixture calculation, scales values before subtraction, clamps only the final numerical result to the information-theoretic interval [0, ln(number of hypotheses)], and has a zero-noise-underflow limit based on the entropy of exactly distinguishable prediction groups.

The cost-aware score is nats per cost unit. Callers must supply a common, scientifically justified noise scale and comparable costs. Candidates with missing/non-finite predictions are rejected rather than rewarded for silently shrinking the set of hypotheses being compared.

## What this does not establish

This is an approximate, equal-prior, homoscedastic-Gaussian design criterion, not a general Bayesian experimental-design engine. It does not yet support non-uniform posterior hypothesis probabilities, heteroscedastic noise, correlated measurements, nuisance-parameter integration, experiment safety constraints, or calibrated predictive distributions. Quadrature error is tested indirectly through invariants but does not yet have an independently validated error bound. Do not use this selector for real-world experimental decisions until its noise assumptions, bounds, and domain constraints are validated.

This code does not prove improved scientific discovery or SOTA performance. Unit tests demonstrate selected mathematical/numerical properties only.

## Falsifiable benchmark plan

### Baselines

Compare at equal observation and cost budgets:

1. Uniform random admissible experiment.
2. Existing raw prediction-variance selector.
3. Expected-information-gain selector, no cost normalization.
4. Expected-information-gain-per-cost selector.
5. Evaluator-only oracle over the same permitted candidate set, retained strictly as an upper bound and never exposed to the selector.

Freeze each selector's inputs and candidate ordering. Share world, seed, initial observations, admissible-action set, and query budget across methods. Record every attempted experiment, invalid prediction, cost, and rejected candidate.

### Test matrix

The existing visible three-family/nine-world reference evaluator is a unit-scale diagnostic, not enough for superiority claims. Expand it before drawing general conclusions:

- At least 60 held-out synthetic worlds spanning at least six structural families or transformation types; hold out complete families/transforms during tuning.
- Five observation-noise regimes including low, moderate, and severely signal-dominated noise.
- At least 20 paired seeds per world and selector for the confirmatory run.
- Include balanced partitions, rare outlier hypotheses, near-identical hypotheses, bounded domains, undefined regions, unequal measurement costs, and deliberate candidate-pool omissions.
- Report candidate budget, observation count, actual cost, wall time, and quadrature/evaluation overhead separately.

### Primary metrics

Report each independently; do not collapse them into a single composite headline score.

- Hidden structural-family recovery and full structural recovery.
- Held-out prediction error on a predeclared extrapolation/transfer split.
- Scientific-validity rubric pass rate and invalid/hacking rate.
- Median and 90th-percentile cost/query count to first valid recovery.
- Calibration/abstention and valid-prediction coverage.
- Selector regret versus the within-pool evaluator oracle.
- Sensitivity to noise, candidate-pool size, omitted hypotheses, numerical extremes, and cost misspecification.
- Runtime per scored candidate and total experiment-selection overhead.

For uncertainty, pair methods within world and seed, resample worlds as clusters first, then seeds within sampled worlds. Keep the candidate-pool oracle separate from model-visible scores. Freeze a benchmark manifest, corpus digest, code commit, toolchain, seed schedule, and scoring version in every receipt.

### Proposed advancement gate (target, not result)

Before describing the information-gain selector as better than the variance baseline, require a held-out confirmatory run to satisfy all of the following:

- At least +5 percentage points in full-structure recovery **or** at least 20% lower median cost to valid recovery.
- A paired 95% confidence interval excluding zero for the selected primary outcome across held-out worlds.
- No more than a 1 percentage-point regression in scientific validity, and no deterioration in invalid/hacking rate.
- No improvement claim from training/tuning worlds or from the evaluator-only oracle.

After this synthetic gate, attempt only permitted, version-pinned external tasks. SciLaws-Bench reports 118 literature-grounded problems across six disciplines and explicitly separates numeric fit, validity, and hidden-structure recovery; its task licenses vary and must be checked individually. NewtonBench reports 324 interactive discovery tasks across 12 physics domains. Results from these two suites must be reported separately and compared only with matched protocol, model/tool access, task subset, and budget. External scores are not inferred from the synthetic reference tests.

## Current evidence links

- [SciLaws-Bench paper and official results](https://arxiv.org/abs/2609.01552) — motivates separate fit, validity and structure metrics and explicit analysis of candidate selection.
- [SciLaws-Bench official project page](https://yiyihum.github.io/SciLaws-Bench/) — provides current results and task-specific license notes.
- [NewtonBench publication record](https://researchportal.hkust.edu.hk/en/publications/newtonbench-benchmarking-generalizable-scientific-law-discovery-i/) — motivates noise and complexity stress tests for interactive discovery.
- [SciMLBenchmarks](https://docs.sciml.ai/SciMLBenchmarksOutput/dev/) — a relevant future reference for numerical solver work-precision, but not a direct leaderboard for this hypothesis-selection method.
