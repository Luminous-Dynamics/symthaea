# Robust Information Gain under Model Scenarios (v0.1)

**Status:** exploratory implementation; no performance or SOTA claim.

## Research shift

The initial noise-only envelope was too weak to count as a full model-robust design: under a fixed hypothesis set and additive Gaussian observation noise, information decreases with increasing noise by data-processing monotonicity. Taking the minimum over a list of noise levels therefore collapses to scoring at the largest level. That is a useful noise stress case, but it doesn't model structural uncertainty.

This revision adds explicit caller-defined *joint scenarios* that can vary both observation noise and the predictive model used to score hypotheses. The key experiment-selection question becomes: which query stays informative across several plausible predictive worlds, rather than which query best separates hypotheses under only one assumed world?

This is a finite scenario envelope, not a mathematical guarantee over all possible misspecifications. It does not reproduce R-IDeA, generalized Bayesian OED, or maximin robust Bayesian experimental design.

## API

- `InformationGainScenario { observation_std_dev, prediction_model }`: a finite scenario with a noise scale and an opaque caller-defined model-variant identifier.
- `expected_information_gain_over_scenarios_nats`: computes the minimum EIG across provided scenarios. A caller-supplied prediction closure maps each hypothesis/experiment/scenario to a finite outcome prediction.
- `select_experiment_by_scenario_robust_information_gain_per_cost`: selects by the minimum scenario EIG divided by positive finite experiment cost.
- The earlier `expected_information_gain_worst_case_nats` / `select_experiment_by_worst_case_information_gain_per_cost` remain noise-only conveniences; for a fixed predictive model they use the largest supplied Gaussian noise standard deviation.

Empty scenarios, invalid noise scales, incomplete/non-finite model predictions, and invalid costs fail closed. Existing variance and nominal-EIG baselines remain intact.

## Integrated law-discovery pilot

The `WorstCaseInformationGain` policy in `symthaea-physics-bridge` now scores each candidate against three predeclared *bundled stress scenarios*: (a) nominal predicted laws with a lower noise scale, (b) a first smooth prediction-discrepancy variant at nominal noise, and (c) a second smooth discrepancy variant at a higher noise scale. The predictive variants use smooth sinusoidal basis perturbations with frequencies different from the hidden benchmark discrepancy; the selector cannot read the simulator's ground-truth expression or the hidden discrepancy amplitude.

This is a deliberately small diagnostic, not a full factorial study. The paired pilot compares nominal EIG to scenario-robust EIG over shared deterministic seeds, in clean and systematically misspecified regimes. It reports recovery, false confident-singleton rates, detected candidate-set collapse, unresolved candidates, and paired percentile-bootstrap intervals. It does not assert that robustness must win.

## Regression coverage

Focused tests cover:

1. Noise-only worst-case EIG is bounded by each included scenario and rejects invalid envelopes.
2. A scenario rank-reversal example: nominal EIG prefers an experiment that perfectly splits all four hypotheses, while a plausible alternate prediction model collapses that experiment's predicted outcomes and makes another experiment the robust choice.
3. Invalid joint scenarios fail closed.
4. The in-bridge paired discovery pilot preserves shared seeds and disjoint misspecification outcomes (false singleton, detected model inadequacy, unresolved multi-hypothesis set).

## Confirmatory benchmark plan

Compare at minimum:

- seeded uniform selection;
- raw prediction variance;
- nominal EIG at a single estimated noise scale;
- noise-only worst-case EIG (the largest supplied noise scale);
- joint-scenario robust EIG;
- EIG with scheduled audits;
- joint-scenario robust EIG plus the same audit schedule;
- evaluator-only oracle, used solely as an upper bound.

Hold candidate pools and random seeds constant across arms. Split by *held-out structural world/family*, not by seed alone. Include correctly specified and systematically misspecified laws, omitted variables/terms, heteroscedastic and heavy-tailed noise, scale-estimation error, shifted query/test distributions, and varied experiment costs.

Report full structural recovery, held-out transfer error, scientific validity, false confident-singleton rate, unresolved/empty-set rates, cost and queries to valid recovery, model-misspecification detection delay, calibration/abstention, selector regret versus the evaluator-only oracle, runtime and acquisition overhead. Publish every metric and negative result.

Predeclared target (not a result): at least 20% lower median cost-to-valid-recovery than nominal EIG on held-out worlds, a world-clustered paired 95% interval excluding zero, and no more than a 1 percentage-point scientific-validity regression. Under misspecification, separately target at least 20% relative reduction in false confident singletons with clean-world recovery degradation no worse than 5 percentage points.

Initial confirmatory matrix: at least 60 held-out worlds spanning 6 or more structural families; five noise/misspecification regimes; at least 20 paired seeds per world. Treat repeated seeds as clustered within worlds and predeclare task exclusions. The small in-repo pilot is only a smoke/diagnostic test.

## External comparison

Protocol alignment matters more than a superficial leaderboard comparison. NewtonBench is an interactive scientific-law discovery benchmark with 324 tasks across 12 physics domains. SciLaws-Bench reports numerical fit, scientific validity, and structural recovery separately. Their scores are not directly comparable to Symthaea's Rust simulator unless the task subset, query budget, candidate representation, tool/model access, and evaluator are matched.

Current research informing the design:
- Tang, Sloman & Kaski, *Representative, Informative, and De-Amplifying: Requirements for Robust Bayesian Active Learning under Model Misspecification*, AISTATS 2026: https://proceedings.mlr.press/v300/tang26d.html
- Callahan & Catanach, *On the Misinformation in a Statistical Experiment*, AISTATS 2026: https://proceedings.mlr.press/v300/callahan26a.html
- Barlas, Sloman & Kaski, *Robust Experimental Design via Generalised Bayesian Inference* (revised 16 October 2026): https://arxiv.org/abs/2511.07671
- Abdulsamad et al., *Maximin Robust Bayesian Experimental Design*: https://arxiv.org/abs/2603.14094
- Catanach & Das, *Metrics for Bayesian Optimal Experiment Design under Model Misspecification*: https://arxiv.org/abs/2304.07949

The key claim to test is empirical: does joint-scenario scoring improve discovery reliability at a fixed cost, especially under model misspecification? If the paired, held-out experiment does not support that claim, do not advance the policy as an improvement.
