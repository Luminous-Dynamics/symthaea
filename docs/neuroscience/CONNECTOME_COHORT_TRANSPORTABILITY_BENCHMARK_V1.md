# Connectome cohort transportability and calibrated response benchmark V1

**Tracking:** #10752 — SCI-CONNECTOMICS-094
**Composes:** #9522, #9587, #10603, #10656, #10675, #10682
**Status:** synthetic contract qualification only; no biological score or cohort-level performance is claimed.

## 1. Research question

Does a structure-to-function model or intervention-selection policy remain calibrated on independently held-out animals and biological contexts, without turning missing observations or low-powered non-significant results into false functional negatives?

## 2. Why this is a separate capability

The C. elegans signal-propagation atlas measures perturbation responses for 23,433 neuron pairs across 113 WT-background animals, covering 186 of 188 head neurons. Some pairs were observed once, others up to 59 times. The authors report trial- and animal-level variation in response amplitude, shape and even sign, potentially reflecting state/history, neuromodulation, plasticity, expression and inter-animal wiring variation. The study also includes 18 unc-31 mutant animals and shows that dense-core-vesicle-mediated extrasynaptic signalling can contribute to measured response even when no direct wired connection is present.

The paper's classification is three-way: functionally connected, functionally non-connected under an explicit equivalence criterion, or insufficient evidence for either. Unmeasured pairs are a fourth, distinct state. A missing response measurement is not a functional zero; a non-significant difference is not an equivalence test.

## 3. Paradigm shift: transport-qualified prediction

Keep these layers explicit:
- structural anatomy and its source revision;
- measured animal-specific perturbation response;
- context/state/genotype and observation-kernel effects;
- predictive distribution for a new animal or target;
- model support domain and abstention reason;
- scientific query decision and downstream intervention value.

V1 qualifies **new-animal / known-pair** transport. Pair- and target-held-out generalization are separate future tracks; do not merge their denominators into the V1 score.

## 4. V1 split contract

- Split at the independent biological unit: animal. All trials, timepoints, neuron pairs and response kernels from one animal belong to exactly one of train, validation or lockbox.
- A known neuron pair may occur in train and lockbox animals in the `NEW_ANIMAL_SAME_PAIR` track. This is intentional and does not qualify as new-pair generalization.
- Models declare exact training animal IDs and an explicit supported-condition set. They may not claim support for a genotype/context absent from the training manifest unless an independently justified external-training receipt is supplied.
- A lockbox outcome must not influence model selection, threshold selection, calibration, nuisance-set construction, intervention choice or report editing.
- Canonical split SHA-256 is recomputed over the domain tag `connectome-cohort-transport-split-v1`, the ordered labels TRAIN/VALIDATION/LOCKBOX, and sorted animal IDs using length-prefixed UTF-8 fields. A stale digest or changed membership rejects the run.
- Each model training-manifest SHA-256 is recomputed over `connectome-cohort-training-manifest-v1`, model ID, split digest, sorted training animal IDs and sorted training-condition IDs. Validation and lockbox animals are never allowed in this manifest.
- Report per-animal metrics and the animal-macro aggregate. Do not count trials, frames, timepoints or pairs within one animal as independent biological replicates.

## 5. Observation dispositions

Every sampled animal/pair/condition record receives exactly one status:

| Status | Meaning | Metric treatment |
|---|---|---|
| `FUNCTIONAL_CONNECTION` | Evidence meets the frozen positive functional-response criterion | Binary positive label |
| `FUNCTIONAL_NON_CONNECTION` | A prespecified equivalence procedure supports the bounded null | Binary negative label; requires equivalence protocol and receipt |
| `INDETERMINATE` | Measured, but neither functional connection nor equivalence is established | Retain; exclude from binary scoring |
| `NOT_MEASURED` | No eligible observation was acquired | Retain in denominator/coverage accounting; never a negative |
| `FAILED` | Acquisition or processing failed | Retain; never a negative |
| `INACCESSIBLE` | Source evidence unavailable | Retain; never a negative |

`FUNCTIONAL_NON_CONNECTION` cannot be inferred from p > 0.05. It must bind an equivalence-protocol digest and a result receipt. A response may still be measured and indeterminate.

## 6. Models and predictions

Each model binds its implementation, training manifest, exact training animal IDs and supported biological conditions. For every measured lockbox observation, it emits either:
- `PREDICTED`: a probability of functional connection plus a predictive response mean and 90% interval; or
- `ABSTAINED_OUT_OF_DOMAIN`: no probability/response point prediction, with an out-of-domain receipt.

An out-of-domain condition not declared in the model's supported-condition set must be abstained, not silently scored as an ordinary prediction. A `NOT_MEASURED`, `FAILED` or `INACCESSIBLE` observation receives `NOT_APPLICABLE`, not an imputed prediction.

V1 checks contract integrity and computes finite lockbox sample metrics. It does not infer that a model is biologically calibrated from a tiny synthetic fixture, and it does not compute population confidence intervals.

## 7. Required comparator families

1. Anatomy-only direct-edge and shortest-path baseline.
2. Anatomy-only with fitted edge weights/signs, but no invented direct edges.
3. Empirical functional-atlas baseline.
4. Hierarchical per-animal response model with animal/target effects.
5. State/history-aware response model.
6. Joint wired plus extrasynaptic/latent pathway model.
7. #10675 robust query-decision intervention policy.
8. Random, expert-ranked, centrality and entropy/EIG intervention baselines.

All comparators require matched input access, training data, tuning budgets and lockbox rules. Report a Pareto frontier across predictive quality, decision regret, calibration, abstention and cost.

## 8. V1 metrics

**Binary functional-response quality:** Brier score on determinate observations only; binary scoring coverage; per-animal Brier; animal-macro Brier.

**Continuous response quality:** absolute error of observed response summaries; 90% interval coverage and sharpness; per-animal and animal-macro interval coverage. Full trajectories, CRPS and likelihood scores are future extensions requiring the underlying time-series source data and a validated predictive distribution.

**Decision and coverage quality:** measured/indeterminate/not-measured/failed/inaccessible counts; out-of-domain abstention counts; false functional-zero promotions; false query-resolution rate; expected/worst-case posterior query loss and decision regret when composed with #10675.

No row-level pooled metric may replace animal-macro reporting. No CI from the V1 fixture is interpreted as biological uncertainty.

## 9. Synthetic known-answer fixture

The starter fixture has two WT animals in train/validation and two lockbox animals: one held-out WT animal and one unc-31 mutant. Two neuron pairs are measured and one is intentionally not measured.

- The same pair appears in different animals, valid for new-animal/known-pair transport.
- One WT response is functional despite no direct EM edge, representing an extrasynaptic-style control.
- One mutant response is equivalently non-functional only because an explicit equivalence receipt is present.
- Two observations are indeterminate and two are not measured; none are converted to a negative.
- The anatomy-only model confidently misses the extrasynaptic-style response and has weak interval coverage.
- The context-aware model predicts on the supported WT domain and abstains on the unsupported mutant domain.
- Binary scores are reported only for determinate truth labels; interval metrics include measured responses; abstentions remain visible.

This fixture tests data contracts, split hygiene, denominator arithmetic and honest abstention. It is synthetic and does not reproduce the biological effect size or statistics of the published atlas.

## 10. Proposed promotion targets

These are research targets, not achieved results:
- zero animal leakage across train/validation/lockbox;
- zero functional-negative promotions from p > 0.05, missing, failed or inaccessible observations;
- 100% correct handling of indeterminate and not-measured known-answer controls;
- nominal 90% predictive interval coverage statistically compatible with 90% at the independent-animal level, with sharpness reported separately;
- at least 20% lower animal-macro CRPS or decision regret per total experiment cost than the strongest frozen baseline on preregistered held-out animals, with uncertainty supporting improvement;
- zero false query-resolution certificates while an opposing admitted world remains compatible;
- no hidden domain-support extrapolation: unsupported conditions must be identified and abstained or supported by a pinned external-training receipt.

Require adequate independent animal counts and predeclared subgroup support before promoting any threshold. The 18-animal unc-31 subset is not a license for unqualified population claims.

## 11. Staged execution

Stage 1: synthetic schema, oracle and negative controls.
Stage 2: obtain and pin trial-level source files, animal IDs, neuron registration, exclusions and exact q/q_eq procedures for the Randi et al. atlas.
Stage 3: reproduce published per-pair/condition summaries before any new model comparison.
Stage 4: grouped held-out-animal evaluation for WT, then a separately reported mutant-context transport track.
Stage 5: compose calibrated predictions with #10675 intervention policies on untouched animals.
Stage 6: prospective evaluation only with appropriate collaborators, approved protocols and exact data custody.

## 12. Claim ceiling

A passing V1 verifier establishes split integrity and finite-sample arithmetic for the supplied bundle. It does not authenticate source data, establish anatomical truth, prove mechanism, show population calibration, establish cross-species transfer, or make a small mutant cohort representative of all biological contexts.