# CMAQ-007: Planner-Induced World-Model Exploitability

**Status:** protocol v0.3 + executable synthetic smoke harness; draft  
**Tracking issue:** [#9035](https://github.com/Luminous-Dynamics/symthaea/issues/9035)  
**Claim ceiling:** deterministic finite synthetic benchmark only; no SOTA, physical safety, trained-model, or production Symthaea claim

## Why this benchmark

A predictor may have low average error and still be unsafe or decision-poor after a planner searches many candidates. Optimization changes the distribution of selected predictions: it preferentially picks favorable errors. Average open-loop loss can therefore conceal selected-action optimism, regret and unsupported-action exploitation.

This slice isolates that effect using a deterministic 256-seed panel and nested candidate catalog sizes of 8, 32, 128 and 512. Each episode uses the same catalog prefix for all policies at a given size; methods are paired on identical hidden truths. Unsupported candidates are rare, have deliberately optimistic model predictions, and have overconfident uncertainty estimates. This makes the world model's errors policy-inducing rather than just uniformly noisy.

The synthetic support-confidence score is imperfect and overlaps between supported and unsupported actions. Support gates therefore expose a **risk–coverage frontier** rather than receiving an oracle support label.

Current literature supports this evaluation direction, not a claim that this harness is novel or state of the art. The 2026 [decision-making-centric world-model evaluation position](https://arxiv.org/abs/2606.15032) explicitly recommends testing policy ranking, optimization lift, exploitability and uncertainty calibration. The [2026 world-model benchmark survey](https://world-model-benchmarks.github.io/) distinguishes prediction-level evaluation from downstream decision utility; its survey snapshot reports 102 representative benchmarks, while its taxonomy includes 13 closed-loop interaction and 14 downstream-outcome-metric entries (categories overlap). A separate 2026 [embodied-world-model benchmark survey](https://arxiv.org/abs/2609.29669) argues that explicit world-model versus direct-policy comparisons and prediction-to-action tests remain scarce. These findings motivate the research question but do not establish a novelty claim for CMAQ-007.

## Run

From this directory:

```sh
cargo test --locked
cargo run --quiet --locked
```

The program prints one machine-readable JSON report with 44 planner-policy rows and four risk-control rows. The focused workflow validates schema, fixed seed/action-set protocol, oracle regret, fallback accounting, a prediction-error bound for this synthetic construction, the shuffled-score negative control, and the calibrated risk-control contract. A passing smoke test establishes only that the code and declared synthetic invariants work.

## Candidate and evaluator boundary

For this synthetic harness, the candidate selector receives predicted utility, a predicted uncertainty value, and a support-confidence score. The evaluator holds the true utility and support flag. In a later process-isolated qualification harness, candidate-facing records must omit evaluator truth, support flags and hidden structural-family identity. In this first slice, source code necessarily specifies the generator; this is not a sealed test set.

## Strategies

- **Direct fallback:** selects a fixed reference action with utility 0.35.
- **Predicted greedy:** selects the candidate with highest predicted utility.
- **Uncertainty penalty:** selects predicted utility minus twice its stated uncertainty. The unsupported family intentionally under-reports uncertainty, testing whether a generic penalty can correct structural blind spots.
- **Support gates:** filter candidate actions by score thresholds 0.40, 0.55, 0.75, 0.90 and 0.97, then select by predicted utility. No learned threshold is tuned on a qualification set.
- **Shuffled-support negative controls:** apply the same 0.55 and 0.97 gates to a separate seeded random score stream independent of the truth labels. This checks whether the apparent gate benefit requires informative support evidence instead of chance filtering. The shuffled score is generated with an independent RNG, preserving the main candidate/utility tape and the earlier reported outputs.
- **Oracle upper bound:** evaluator-only selector chooses the best true-utility action or direct fallback. This is not an eligible competitor.

All candidates and methods are evaluated at the same search sizes and episode seeds. Increasing search size uses a prefix of the same 512-candidate catalog for that episode, making the search-pressure contrast paired.

## Reported metrics

- **Prediction MAE:** mean absolute prediction error across the candidate catalog.
- **Selected-action optimism gap:** predicted utility minus true utility of the action that is selected.
- **Realized utility and regret:** selected true utility and regret against the evaluator-only best action available in that catalog (including the fallback).
- **Unsupported-selection rate:** fraction of episodes in which a selector chooses an unsupported candidate.
- **Coverage:** fraction of episodes in which a selector chooses a model candidate instead of the direct fallback.
- **Paired regret reduction versus predicted-greedy:** greedy regret minus the strategy's regret, with a normal-approximation 95% interval over the 256 paired episode differences.

The interval is an exploratory normal approximation, not a guarantee of calibration for arbitrary dependence. The independent sampling unit is the seeded episode. Report raw policy-level metrics and the complete risk–coverage frontier; do not choose one threshold after looking at qualification results.

## First exact-head smoke result (Rust)

The initial focused workflow passed on source commit `c1cf6921ee68f02492ebf8e68d98c246a3ddccbd`: **7 Rust tests passed**, and the JSON report/schema and fixed-panel invariants passed ([workflow run 40952051883](https://github.com/Luminous-Dynamics/symthaea/actions/runs/40952051883)). These numbers are the actual Rust workflow output for this commit, not the earlier kind of independent mirror result; this remains a synthetic smoke panel, not external qualification.

| Search candidates | Mean prediction MAE | Greedy regret | Greedy unsupported selections | Uncertainty-penalty regret | Uncertainty-penalty unsupported selections | Strict support gate (0.97) regret | Strict gate unsupported selections / coverage |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.0391 | 0.0422 | 14.5% | 0.0666 | 21.1% | 0.1616 | 1.2% / 43.8% |
| 32 | 0.0409 | 0.0911 | 29.7% | 0.1511 | 48.0% | 0.1439 | 2.0% / 91.0% |
| 128 | 0.0411 | 0.1569 | 55.9% | 0.2462 | 85.2% | 0.0731 | 8.2% / 100% |
| 512 | 0.0414 | 0.2414 | 89.5% | 0.2739 | 100% | 0.0646 | 15.2% / 100% |

The strict 0.97 gate table entry is reported as *unsupported selections / coverage*. For comparison, the direct fallback's regret rises only from 0.1475 at size 8 to 0.1991 at size 512, with zero unsupported selections and zero model-candidate coverage. The 0.55 support gate at size 512 still selects an unsupported candidate in 69.9% of episodes and has regret 0.1924.

**Interpretation:** in this chosen synthetic construction, mean prediction MAE changes only from roughly 0.039 to 0.041, while the greedy planner's unsupported-action selection rises from 14.5% to 89.5% as search size increases. A generic uncertainty penalty is worse here because the structural blind spots are overconfident; at 512 candidates it selects unsupported actions in every episode. A very strict support gate reduces that risk but cannot eliminate it under the deliberately overlapping support scores, and broad search eventually finds high-scored unsupported actions more often. The gate can also perform worse than fallback at low search size because it takes low-coverage risks.

This demonstrates the intended falsification signal for this synthetic setup; it does **not** show that all real planners behave this way, that a support gate universally fixes it, or that Symthaea outperforms an industry baseline. The same result must be challenged with corrupted support signals, correlated/horizon-amplified errors and independently held-out environments before any generalization claim.

## Shuffled-support negative control (Rust exact-head run)

The follow-up control passed at current source commit `22ce169ab360be31bc19f1cdaa69b15fa1ef8446`: **8 Rust tests passed**, and the 44-row report schema, oracle/fallback/risk–coverage invariants, and the preregistered informative-versus-shuffled strict-gate comparisons at candidate-set sizes 32, 128 and 512 passed ([exact-head workflow run 40952937400](https://github.com/Luminous-Dynamics/symthaea/actions/runs/40952937400)).

| Search candidates | Informative support gate .97: regret | Unsupported rate | Shuffled support gate .97: regret | Unsupported rate | Informative gate coverage | Shuffled gate coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 0.1616 | 1.2% | 0.1524 | 0.4% | 43.8% | 26.2% |
| 32 | 0.1439 | 2.0% | 0.1944 | 3.1% | 91.0% | 63.7% |
| 128 | 0.0731 | 8.2% | 0.1241 | 8.6% | 100% | 98.4% |
| 512 | 0.0646 | 15.2% | 0.0919 | 20.3% | 100% | 100% |

At size 8, the random score happens to have lower regret because it abstains more often; this is a useful reminder that one seed panel and one threshold do not prove superiority. From sizes 32 to 512, the informative score outperforms the shuffled score on both regret and unsupported-selection rate. But it still leaves substantial unsupported selection at 512 candidates, so support scoring is evidence—not a safety certificate. The workflow now encodes the large-search contrast as a regression assertion rather than a narrative-only observation.

## Joint risk- and utility-controlled planning (v0.3)

CMAQ-007 v0.3 replaces hand-picked thresholds with a calibration-selected abstention rule. The design treats world-model authority as conditional: a model action is used only if the calibration evidence meets three separately declared requirements.

For each search size (8, 32, 128 and 512), the procedure uses **2,048 calibration episodes** and **2,048 held-out qualification episodes** from distinct deterministic seed streams. Those streams are separate from the 256-episode descriptive panel. It examines a fixed six-threshold grid: 0.40, 0.55, 0.75, 0.90, 0.97 and 1.01. The last threshold forces direct fallback because support scores are below 1.01. Calibration episodes provide evaluator-observed supported/unsupported labels retrospectively; the candidate selector itself receives only the declared candidate-facing support score and predicted utility. This is an offline synthetic-calibration assumption, not a claim that live Symthaea knows support labels before execution.

For every threshold, the calibrator evaluates three quantities:

1. **Unsupported-selection risk:** the one-sided Clopper–Pearson upper confidence bound on the fraction of episodes in which the chosen model action is evaluator-labelled unsupported must be at most 5%.
2. **Fallback-downside risk:** the one-sided Clopper–Pearson upper confidence bound on the fraction of episodes in which the selected model action delivers utility no better than the direct fallback must be at most 5%. Episodes where the controller falls back are not counted as a downside event.
3. **Positive utility lift:** a one-sided Hoeffding lower bound on mean per-episode utility lift versus direct fallback must be strictly above zero. Candidate lift is bounded in this synthetic construction to a width of 0.50 utility units; fallback episodes contribute zero lift.

The six candidate thresholds × three criteria produce 18 bounds, so the procedure uses Bonferroni-adjusted α = 0.05/18 for each bound. Among thresholds satisfying **all three** constraints, it selects the one with the greatest calibration coverage; exact coverage ties prefer the stricter threshold. If no model threshold satisfies every criterion, it selects 1.01, explicitly authorizing fallback and zero model coverage. The model-gate flag distinguishes a genuinely certified threshold from this fallback sentinel.

The report records selected threshold, whether the gate was certified, calibration event counts/rates/upper bounds, the utility-lift estimate and lower bound, held-out event rates/coverage/utility lift/regret, and direct-fallback regret on the same qualification episodes. Qualification outcomes are not used to choose the threshold. The workflow checks the schema, distinct seed roots, sample counts, risk bounds, and positive utility bound for any certified model gate; it does not require a model to be used when none qualifies.

This rule is intentionally conservative, but it is still a synthetic experiment rather than a deployment guarantee. The stated confidence interpretation assumes calibration episodes are independent draws from the target episode distribution, support/downside labels are correct, and the candidate utility-lift range is valid. It does not cover arbitrary shift after calibration, adaptive adversaries, mislabelled or delayed support evidence, or dependence among episodes. If the controller abstains at larger search sizes, that shows the current estimator has not earned higher-coverage authority under these requirements. Do not weaken the bound after inspecting qualification outcomes; improve the support estimator, collect better calibration evidence, or improve the model's decision-value margin.

**Status:** v0.3 source and report-contract checks have been added; the exact-head run for this revision is pending. No performance result for the new three-constraint gate should be cited until its focused Rust workflow completes.

## Falsifiable research questions

1. Does search breadth increase the unsupported-action selection rate or selected-action optimism gap while mean prediction MAE remains comparatively stable?
2. Does the uncertainty penalty fail when the estimator is structurally overconfident outside support?
3. Can imperfect support gating reduce regret, and what coverage does it sacrifice?
4. Do the risk–coverage trade-offs survive support-score corruption, correlated errors, multiple rollout horizons and held-out reward families?
5. Does a calibrated, evidence-backed support gate beat direct fallback at matched compute and data cost on real task suites?

A negative answer is still useful. The benchmark must report where no method dominates, not tune away a difficult regime.

## Known limitations / next work

- No transition dynamics or long-horizon rollout in v0.1; utility is one-step and scalar.
- No learned world model; errors are generated by a fixed synthetic mechanism.
- Support score and model uncertainty are synthetic observables, not calibrated outputs from Symthaea.
- No process isolation, cryptographic provenance, real-world robot interface, or hardware latency/energy measurement.
- The search-pressure panel is a falsification smoke test, not an industry ranking and not a claim that the current construction beats any SOTA method.

Next steps: add correlated and horizon-amplified model error; adversarial policy-search intensity; distribution shift after policy selection; support-score corruption and delayed/noisy-label controls; expand held-out mechanism families; and then compare Symthaea's model-based policy with direct and robust baselines on a matched task and compute budget.
