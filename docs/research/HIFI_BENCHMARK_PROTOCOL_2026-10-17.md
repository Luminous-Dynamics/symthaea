# Open Hi-Fi Benchmark Protocol: Generalization Before Leaderboards

**Status:** proposed protocol and engineering targets; no participant data, physical measurements, or model performance results are claimed.  
**Snapshot:** 2026-10-17.  
**Parent research:** [HIFI-RESEARCH-001 #7888](https://github.com/Luminous-Dynamics/symthaea/issues/7888).  
**Human study scaffold:** [HP-PREF-001 #7889](https://github.com/Luminous-Dynamics/symthaea/issues/7889).

## Thesis: score decisions, not just curves

The paradigm shift is from optimizing a single response curve or one universal sound-quality number to a **falsifiable, uncertainty-aware design loop**:

1. Predict preference and acoustic behavior before the next measurement.
2. Select the experiment expected to reduce decision uncertainty most per total cost.
3. Measure with an explicit fixture, calibration record, repeatability estimate and raw-data hash.
4. Separate population preference from listener-specific adaptation.
5. Validate on identities unavailable during fitting—not on random rows drawn from the same products, listeners, excerpts or sessions.
6. Publish failures, uncertainty and evidence lineage alongside any winning score.

A smoother curve, a higher internal score or a synthetic benchmark win is not proof of a better listening experience.

## Four distinct benchmark regimes

Do not combine these regimes into one leaderboard; each measures a different generalization claim.

| Regime | What must be held out | What it answers |
|---|---|---|
| New listener | Listener identities from population-train and model-selection sets | Does the population model generalize to people it has not seen? |
| Personalization | Test ratings and excerpts from the calibration set; listener may appear in calibration and test but not in population training | Do a few permitted calibration judgments improve later predictions for this listener? |
| New product revision | Product revision IDs from every fitting/tuning split | Does the model generalize beyond the exact hardware samples used to build it? |
| New fixture / session | Fixture or measurement-session IDs according to the declared transfer claim | Does measurement interpretation survive the change in test setup? |

Each benchmark manifest must name its intended regime and declare group-level disjointness rules. Never assume that random row splitting is valid. In repeated-measures audio data, correlated rows may differ in filename while representing the same person, excerpt, product revision, reseat, fixture or session.

The new **hifi-benchmark-check** tool is a fail-closed **split-contract guard**, not a model evaluator. It checks identity fields, unique sample IDs, allowed roles and explicit test-versus-fit-role disjointness rules. It deliberately allows calibration/test listener overlap only when the declared rule does not prohibit it. It cannot prove that a dataset is honest, that the study was randomized, or that the model is good. The interchange contract is [hifi-split-manifest.schema.json](../../schemas/hifi-split-manifest.schema.json); use a full Draft 2020-12 validator for that schema and the custom Rust check for cross-row rules. The manifest must declare an evaluation regime, and the checker refuses regimes that omit their mandatory holdout dimensions.

## Baselines that should always be reported

1. Fixed transparent population target or a conventional published preference model where applicable.
2. Regularized population-only model using the same permitted training data.
3. Simple listener-calibration model with a frozen, documented form.
4. Hierarchical/personalized candidate using calibration responses only.
5. Majority/prior baseline and a non-personalized ablation.

All baselines receive identical permitted information. Tune only on development data. Keep a final untouched test set; record source revision, feature/profile hashes, preprocessing, exclusions and partition algorithm before test evaluation.

## Metrics and gates

### Primary preference metric

- **Primary:** held-out pairwise log loss, with participant-clustered uncertainty intervals.
- **Calibration:** Brier score and reliability/calibration plot; report calibration slope/intercept where sample size supports them.
- **Discrimination:** pairwise accuracy, ranking metrics and per-listener results as secondary metrics.
- **Heterogeneity:** report the fraction of listeners helped and harmed by personalization, not only the population average.
- **Robustness:** stratify by excerpt, product revision, fit/reseat, session and relevant listener subgroups; include uncertainty and missingness.

**Provisional engineering ambition:** the personalized candidate should reduce held-out pairwise log loss by at least **10% relative** to the strongest frozen transparent baseline, with listener-clustered uncertainty supporting an improvement and no material calibration degradation. This is a target for experiment planning, not an achieved result or a universally meaningful margin. Run a pilot and power/sensitivity simulation first, then freeze the smallest practically meaningful effect and the non-inferiority margin before opening the confirmatory test. If the sample cannot resolve the margin, report inconclusive rather than claiming a win.

### Physical and measurement gates

- Report repeatability across reseats, units, fixture/session changes and microphone orientation as applicable, including uncertainty—not only a mean curve.
- Report response deviation, distortion and output capability separately; do not turn distinct measures into one unvalidated fidelity score.
- Propagate measurement uncertainty through predicted design ranking. A design whose apparent lead is smaller than the uncertainty should be classified as unresolved.
- Before confirmatory data collection, preregister the measurement protocol, acceptable tolerance bands, decision rule and handling of invalid captures. Standards or regulatory compliance must never be inferred from app checks or self-described calibration.

### Active experiment selection

Compare the proposed selector against random, fixed-grid and simple uncertainty-sampling baselines under the same total budget. Count setup, calibration, failed captures, repeats and deferred confirmatory tests in cost. Primary metric: reduction in uncertainty or regret per unit total cost, assessed against a known synthetic ground truth first, then on a separately governed physical experiment. Synthetic success is a software sanity check only.

## Quantified launch goals

These are ambitious project targets, not industry standards or achieved results. Freeze dataset-specific thresholds and the analysis plan before confirmatory evaluation.

| Area | Target |
|---|---|
| Split-contract integrity | Reject 100% of seeded listener/excerpt/product/fixture/session leakage cases in the adversarial suite, with zero false passes; clean controls must pass. |
| Preference prediction | At least 10% relative reduction in held-out pairwise log loss over the strongest frozen transparent baseline; a listener-clustered 95% interval must support improvement. The pilot must justify the meaningful-effect threshold. |
| Calibration | No degradation beyond a preregistered Brier-score/calibration margin. Set that margin from pilot variability and decision consequences, not after seeing the confirmatory results. |
| Active measurement | At matched predeclared uncertainty or decision regret, target at least 20% less total measurement cost than the strongest fixed/random baseline across at least three synthetic task families; count repeats, failed captures and confirmation budget. |
| Robust design decisions | For candidates whose estimated advantage exceeds measurement uncertainty, target at least 95% stability of the down-selection under the preregistered reseat/unit/room/fixture perturbation set. Otherwise report the decision as unresolved. |
| Evidence reproducibility | 100% of artifacts used for a published result have verified hashes and pinned processing/source revisions; a clean rerun reproduces the report within declared numeric tolerances. |

The 10% and 20% targets are useful research ambitions, not guarantees that any model will achieve them. If a pilot shows the threshold is not distinguishable from measurement or sampling noise at a feasible sample size, revise the *future protocol before confirmation* or report that the proposed goal is currently unsupported. Never lower the threshold after observing the test result.

## Synthetic tests before human or hardware claims

Use deterministic synthetic data with known latent listener preferences and known confounders to test whether the pipeline can detect:

- listener leakage that falsely inflates random-row scores;
- excerpt, product-revision, fixture and session leakage;
- a personalization model that memorizes IDs rather than learning preferences;
- calibration failure under preference distribution shift;
- missing-not-at-random judgments and unequal listener activity;
- measurement noise large enough to reverse a purported winner;
- an experiment selector that overfits its own synthetic generator.

Required negative controls: shuffled labels, listener-ID-only predictor, deliberately leaking split, a strong transparent baseline and at least one distribution-shift test. The evaluator should make it impossible to represent these as real listener results.

## Research priorities, ordered by value

**P0 — Trustworthy evaluation.** Split-integrity guard, frozen baselines, report generator, deterministic negative controls and evidence provenance. Without this, subsequent model gains are easy to overstate.

**P1 — Useful personalization.** Determine when a few calibration judgments help, how much calibration is needed, which listeners are harmed, and when population preference should remain the safer default.

**P2 — Robust acoustic design.** Optimize lower-tail outcomes over plausible fit, reseat, unit and room variation. Show Pareto trade-offs among frequency response, distortion, maximum linear output, size, cost and repairability instead of hiding them in a single score.

**P3 — Information-efficient measurements.** Choose the next physical measurement by expected decision improvement per total cost, including uncertainty propagation and confirmation budget.

**P4 — Inter-lab reproducibility.** Recompute claims from versioned raw artifacts, fixture metadata, calibration files and documented processing, then test whether independent operators can reproduce the result.

## Initial command and status boundary

From the repository root:

~~~sh
cargo test -p symthaea-audio-evidence
cargo run -p symthaea-audio-evidence --bin hifi-benchmark-check -- schemas/examples/hp-pref-001-split-manifest.json
~~~

Expected successful fixture output includes **SPLIT CONTRACT OK**, seven observations and the explicit warning that model quality was not evaluated. A deliberately leaked fixture must exit non-zero. These commands are documented expectations only until run in a supported Rust environment; no test pass is claimed here.

## Decision rule

A benchmark win is accepted only when the split contract is valid, the strongest transparent baseline is beaten on the preregistered primary metric, uncertainty supports the claimed effect, calibration does not materially regress, negative controls behave as expected, and the report can be regenerated from pinned inputs. Otherwise: **no win / inconclusive**, with the failure retained.
