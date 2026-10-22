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

The cost-aware score is nats per cost unit. Callers must supply a common, scientifically justified noise scale and comparable costs. Candidates with missing/non-finite predictions are rejected rather than rewarded for silently shrinking the set of hypotheses being compared. Candidates with zero information gain are also skipped; if no positive-information candidate remains, the selector returns `None` rather than spending a measurement with no expected discrimination.

## What this does not establish

This is an approximate, equal-prior, homoscedastic-Gaussian design criterion, not a general Bayesian experimental-design engine. It does not yet support non-uniform posterior hypothesis probabilities, heteroscedastic noise, correlated measurements, nuisance-parameter integration, experiment safety constraints, or calibrated predictive distributions. The uniform prior is over candidate instances, not canonical laws: duplicated/equivalent expressions can alter prior mass and bias experiment selection. A future version should accept calibrated posterior weights and/or normalize priors over validated structural-equivalence classes rather than pretending generic expressions can be safely deduplicated. A stress-grid regression test now compares the production 256-point midpoint estimate against a 2,048-point refinement across several prediction geometries and noise levels with a 0.005 absolute-difference tolerance. This is a convergence check against a higher-resolution estimate, not a rigorous error bound or an independent mathematical oracle. Do not use this selector for real-world experimental decisions until its noise assumptions, bounds, and domain constraints are validated.

This code does not prove improved scientific discovery or SOTA performance. Unit tests demonstrate selected mathematical/numerical properties only.

## Integrated law-discovery pilot

The focused workflow also runs `expected_information_gain_vs_variance_on_paired_seeds` inside the existing `symthaea-physics-bridge` multi-round discovery simulator. It compares the raw-variance selector with expected-information-gain-per-cost (equal unit cost in this first ablation), using the same 40 deterministic seeds, typed-generation setting, four-round budget and candidate-pool generation per seed. The report prints paired selector-only wins, both success rates, mean rounds, EIG selection failures, and 2,000-resample percentile intervals for EIG-minus-variance changes in success rate and rounds. Positive success deltas favor EIG; negative round deltas favor EIG. These seed-resampled pilot intervals do not replace world-clustered confirmatory inference. This is a small pilot diagnostic, not a confirmatory performance claim.

The simulator's actual noise scale depends on the hidden truth's output magnitude. The EIG selector must not inspect that truth, so this pilot estimates one shared standard deviation from the median absolute predictions across the live hypotheses and remaining candidate points. That is a visible homoscedastic approximation; an improvement or regression may reflect this mismatch as well as the selection rule. The benchmark must therefore report selection failures and noise regimes before any general conclusion.

## Scheduled audit policy for model mismatch

A literature-driven extension adds an explicitly separate policy arm, `InformationGainWithAudits`: it uses the information-gain selector on ordinary rounds, selects a seeded-uniform admissible experiment every second round, and—critically—treats a single surviving hypothesis as provisional. Once only one candidate remains, all remaining queries in the budget become validation audits rather than stopping immediately. A contradictory observation can therefore reject that candidate. This is not claimed to be optimal; it is a controlled test of whether periodic exploration and post-selection validation reduce false confidence when the available candidate pool does not contain the observation-generating process.

The integrated simulator now applies a controlled hidden additive discrepancy term proportional to base-law magnitude and a combination of sinusoidal input terms. This perturbation is designed to make observations misspecified relative to the finite candidate pool; it does not assert that the broad expression grammar cannot represent such a term. Selectors receive only their live candidate predictions and never the discrepancy amplitude or hidden transformed truth. The paired diagnostic compares plain EIG and EIG-with-audits on the same seeds and candidate-pool generation in two regimes:

- **Matched regime:** discrepancy amplitude is zero; track recovery to ensure audit sampling does not catastrophically harm ordinary discovery.
- **Misspecified regime:** discrepancy amplitude is 0.12; track how often all candidate hypotheses are eliminated (a detectable model-coverage failure) versus how often the procedure ends with exactly one candidate (false certainty under known misspecification).

The pilot reports paired 2,000-resample percentile intervals for the difference in false-singleton rate, discrepancy detection, and matched-world recovery. It also reports an explicit three-way outcome partition in the misspecified regime: candidate set collapsed to empty (detectable model-coverage failure), exactly one candidate remains (false certainty in this controlled regime), or multiple candidates remain (unresolved). The paired bootstrap pairs policies by seed but is not a world-clustered external benchmark. A lower false-singleton rate and higher discrepancy-detection rate are desired; the test does not assert that the audit policy wins.

This is inspired by recent research on misspecified experimental design: classical information-maximizing experiments can amplify inferential bias, while representativeness and error de-amplification are separate acquisition concerns. It does **not** implement the published R-IDeA criterion or the maximin/Sibson-MI robust-design method and must not be presented as a reimplementation of either.

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

### Misspecification-specific target (target, not result)

For the audit-augmented policy, preregister a stress sweep at systematic discrepancy amplitudes `0, 0.03, 0.06, 0.12, 0.24`, then repeat under heteroscedastic noise and shifted query distributions. Compare random, raw variance, plain EIG, and EIG-with-audits at equal observation budgets. The preliminary target is at least a 20% relative reduction in false-singleton decisions on held-out misspecified worlds, with a paired 95% interval excluding zero, while matched-world structural recovery drops by no more than 5 percentage points. Always publish the unresolved fraction and empty-set/model-coverage-failure fraction as well—lower false certainty alone can be achieved by refusing to decide, so all three outcomes matter. This is a target for future confirmatory runs, not a result from the current 40-seed pilot.

### Proposed advancement gate (target, not result)

Before describing the information-gain selector as better than the variance baseline, require a held-out confirmatory run to satisfy all of the following:

- At least +5 percentage points in full-structure recovery **or** at least 20% lower median cost to valid recovery.
- A paired 95% confidence interval excluding zero for the selected primary outcome across held-out worlds.
- No more than a 1 percentage-point regression in scientific validity, and no deterioration in invalid/hacking rate.
- No improvement claim from training/tuning worlds or from the evaluator-only oracle.

After this synthetic gate, attempt only permitted, version-pinned external tasks. SciLaws-Bench reports 118 literature-grounded problems across six disciplines and explicitly separates numeric fit, validity, and hidden-structure recovery; its task licenses vary and must be checked individually. NewtonBench reports 324 interactive discovery tasks across 12 physics domains. Results from these two suites must be reported separately and compared only with matched protocol, model/tool access, task subset, and budget. External scores are not inferred from the synthetic reference tests.

## External comparison snapshot (2026-10-22)

The live SciLaws-Bench project page lists fourteen models under one fixed ReAct-style harness (up to 30 turns, with a task-specific query budget in the Parallel setting). At this snapshot, GPT-6 reports aggregate Real numeric-fit `S_N=51.33%`, Real scientific-validity `S_V=82.91%`, and Parallel structural-recovery `S_S=72.46%`; GPT-5.5 reports 50.11%, 81.84%, and 58.26%, respectively. The paper's paired task-bootstrap supports the GPT-6-vs-GPT-5.5 increase in structural recovery, but not the gains in numeric fit or scientific validity. The leaderboard further reports only 53.6% fit-validity concordance over unequal-fit pairs. These are external agent leaderboard values, not comparable to the current synthetic Rust microbench and not outcomes from this PR.

NewtonBench provides a different external stress reference: 324 interactive discovery tasks across 12 physics domains, with systematic law alterations and difficulty variation across law and system complexity. Its authors report strong degradation as system complexity/noise increases. We should begin with a small permitted adapter or manually matched subset, preserving the original harness and task licenses, rather than reimplementing either benchmark and calling the scores equivalent.

## Current evidence links

- [SciLaws-Bench paper and official results](https://arxiv.org/abs/2609.01552) — motivates separate fit, validity and structure metrics and explicit analysis of candidate selection.
- [SciLaws-Bench official project page](https://yiyihum.github.io/SciLaws-Bench/) — provides current results and task-specific license notes.
- [NewtonBench official repository](https://github.com/HKUST-KnowComp/NewtonBench) — 324 interactive tasks across 12 physics domains; results show sensitivity to system complexity and noise.
- [AISTATS 2026: On the Misinformation in a Statistical Experiment](https://proceedings.mlr.press/v300/callahan26a.html) — information-maximizing designs may amplify inferential/model misspecification.
- [AISTATS 2026: R-IDeA active learning](https://proceedings.mlr.press/v300/tang26d.html) — motivates treating representativeness, informativeness, and error de-amplification as distinct concerns.
- [ICML 2026: Maximin Robust Bayesian Experimental Design](https://arxiv.org/abs/2603.14094) — adversarial distributional-robust formulation with Sibson mutual information; noted as a future research reference, not reimplemented here.
- [SciMLBenchmarks](https://docs.sciml.ai/SciMLBenchmarksOutput/dev/) — a relevant future reference for numerical solver work-precision, but not a direct leaderboard for this hypothesis-selection method.
