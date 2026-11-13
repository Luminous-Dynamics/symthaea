# Human Experiential State Research
**Status:** research protocol proposal v0.1  
**Track:** exploratory human cognition research, separate from the software execution gateway  
**Named source family:** Monroe Institute Gateway Experience / Hemi-Sync exercises

## 1. Scope and separation

This programme asks whether structured auditory, relaxation, imagery, and attention practices change measurable cognitive performance or physiological state. It does not assume that the Gateway Experience's metaphysical explanations are correct.

This is **not** the Symthaea provider-execution gateway, the in-process SafetyGateway, a safety policy, or an authorization mechanism. Human-state observations must never authorize tool execution.

The Monroe Institute describes progressive exercises including Focus 10, Focus 12, Focus 15, and Focus 21. These labels should be treated as names for protocol conditions and participant-reported experiences, not validated biological states.

## 2. Evidence summary and prior uncertainty

The prior evidence does not justify claiming reliable neural entrainment or general cognitive enhancement from binaural beats:
- A 2023 systematic review of 14 studies on binaural-beat effects on brain oscillatory activity found five studies consistent with entrainment, eight contradictory, and one mixed; methods and EEG analyses were heterogeneous.
- A 2024 systematic review of 31 studies on auditory stimulation, autonomic arousal, cognition, and attention found the evidence mixed or insufficient for firm conclusions.
- A 2024 systematic review of randomized trials for non-clinical stress management found promising results in some studies but inconsistent effects, methodological limitations, and no established optimal protocol.
- A 2026 scoping review of 45 studies on binaural, monaural, and isochronic stimulation again found mixed outcomes and called for more standardized protocols.
- A 2026 systematic review/meta-analysis of meditation EEG oscillations found pooled changes in some frequency bands but substantial heterogeneity and sensitivity to analytic choices.

These findings justify controlled, transparent experimentation—not an assumption that a named frequency produces a particular state or that subjective experiences establish nonlocal perception.

## 3. Research questions

1. Do guided relaxation and attention protocols produce reproducible within-person changes in attention, working memory, recall, or divergent thinking?
2. Do binaural beats add benefit beyond matched audio, music, guided relaxation, or silence?
3. Which measured pre-task features predict task performance for a given person, and do those predictions generalize across sessions and participants?
4. Are changes explained by expectancy, relaxation, sleep, hearing, familiarity, or other confounds rather than beat frequency?
5. Do subjective state reports correspond to objective performance or physiological measurements, and where do they diverge?
6. Can Symthaea forecast the uncertainty of human task performance without claiming access to private experience?

## 4. Experimental design

### Phase 0 — protocol and instrument qualification
- Publish hypotheses, primary/secondary outcomes, analysis code, stopping rules, and exclusion rules before confirmatory data collection.
- Validate audio generation, headphone channel separation, playback level, timing, randomization, and event synchronization.
- Validate physiological sensors against appropriate references where possible.
- Conduct a safety review and obtain ethics approval/consent before recruiting or collecting human data.
- Use an initial feasibility/pilot sample only to estimate variance and refine the protocol; do not count it as confirmatory evidence.

### Phase 1 — randomized, counterbalanced crossover

Within-subject conditions, order counterbalanced with a Latin-square or equivalent design:
1. Gateway-style guided relaxation / imagery audio.
2. Binaural-beat audio with declared carrier and beat parameters.
3. Matched audio control with no binaural frequency difference.
4. Neutral spoken guidance or music control matched for duration and volume.
5. Quiet rest / silence.
6. An established attention or mindfulness control, only if its protocol is fully specified.

Use identical session duration, time-of-day windows where practical, playback equipment, volume limits, and task timing. Record previous sleep, caffeine, medication changes if voluntarily disclosed and ethically approved, hearing status, prior meditation/audio familiarity, and expectancy. Do not infer diagnoses from these variables.

Participants may recognize an audio condition, so do not claim full participant blinding. Keep outcome assessors and analysts blind to condition labels where feasible. Record condition guesses to measure expectancy.

### Phase 2 — outcome battery

Pre-register one primary endpoint per study. Candidate outcomes:
- sustained attention (e.g. omission/commission errors and reaction-time variability);
- working memory (accuracy and response time on a defined n-back or span task);
- delayed recall (predefined immediate and delayed scoring);
- divergent thinking (blinded, rubric-based fluency/flexibility/originality scoring);
- metacognitive calibration (confidence versus correctness).

Secondary measures:
- short validated subjective state inventories;
- EEG with full acquisition and preprocessing provenance;
- HRV, electrodermal activity, respiration, and heart rate where properly collected;
- comfort, adverse experiences, fatigue, and dropout.

Do not use EEG band power as a proxy for consciousness or treat a self-report as an objective performance gain.

### Phase 3 — transfer and replication
- Hold out participants, sessions, and (where available) an independent site.
- Evaluate whether any effect transfers across audio hardware and task variants.
- Replicate confirmatory results before publishing claims of robust benefit.
- Publish null, contradictory, adverse, and inconclusive outcomes.

## 5. Symthaea model programme

Represent each session as a time-stamped, provenance-carrying record with separate channels:

- intervention identity and exact audio digest;
- experimental condition and randomization assignment;
- participant/session pseudonymous identifier;
- subjective report and instrument version;
- physiological observations with sensor, calibration, quality flags, and units;
- task trials and scoring rubric;
- sleep/time-of-day/expectancy covariates where consented;
- missingness, exclusions, and protocol deviations;
- analysis profile, code digest, and evidence grade.

Candidate model baselines:
1. condition mean / participant-only baseline;
2. linear mixed-effects model with participant/session structure;
3. regularized linear model on pre-task features;
4. random forest and gradient-boosted trees;
5. temporal convolution/recurrent baseline where sequence length justifies it;
6. HDC/CfC temporal state model.

Compare at identical data splits and feature availability. Use leave-one-participant-out and future-session holdouts; do not let adjacent trials from one person leak across train/test. Use ablations for audio features, subjective reports, physiology, and baseline covariates.

Metrics:
- task score: accuracy, reaction-time distribution, and task-specific scoring;
- prediction: MAE/RMSE as appropriate, Brier score or log loss for probabilistic outcomes;
- calibration: reliability curves, calibration slope/intercept, expected calibration error;
- uncertainty: interval coverage and width;
- classification: AUROC plus AUPRC at the observed prevalence, sensitivity/specificity at declared thresholds;
- generalization: participant-held-out and site-held-out degradation;
- robustness: missing-sensor, motion-artifact, audio-device, and protocol-deviation tests;
- utility: improvement over simple baselines, not a raw score alone.

A proposed stretch target is a ≥10% relative reduction in held-out prediction error versus the strongest simple baseline, with a confidence interval excluding no improvement, no calibration degradation, and replication on a separate participant cohort. This is a research target, not a claim or an arbitrary acceptance gate for human benefit.

## 6. Gateway Experience claims requiring separate treatment

Claims such as remote viewing, out-of-body perception, nonlocal information access, or literal travel beyond ordinary time/space must not be inferred from relaxation, EEG changes, vivid imagery, confidence, or HDC/CfC model outputs.

If anomalous-perception hypotheses are ever studied, they require a separately preregistered, independently administered, double-blind protocol with:
- randomized target assignment by an independent custodian;
- sealed target pools and no leakage through metadata, timing, experimenter cues, or online systems;
- forced-choice or otherwise predeclared scoring;
- blinded judges and frozen scoring rubrics;
- correction for multiple comparisons and stopping rules;
- adversarial leakage audits and independent replication;
- public reporting of failures and null results.

This is not part of the first implementation phase and must not be used for consequential decisions.

## 7. Human safety, privacy, and governance

- Obtain ethics review and informed consent before human-subject research.
- Participation is voluntary; withdrawal must not require explanation.
- Use conservative listening levels, breaks, and an immediate stop option.
- Screen for hearing discomfort and provide non-audio alternatives.
- Do not represent these exercises as treatment or replace clinical care.
- Avoid collecting identifiable health data unless necessary, approved, and protected.
- Keep raw physiological/audio/session data under strict access controls; release only consented, de-identified derivatives with disclosure-risk review.
- Never use cognitive-state estimates for employment, insurance, credit, policing, or other high-stakes classification.

## 8. Acceptance criteria and claim ceiling

- Preregistered protocol and frozen analysis plan exist.
- Audio and sensor chains are qualified with calibration and digests.
- Randomization and blinding boundaries are auditable.
- A held-out participant/session result is reported against simple baselines.
- Confidence intervals, missingness, adverse outcomes, and null results are visible.
- Independent replication is required for strong causal or generalization claims.

A successful study can establish a measured effect of a specified intervention on a specified outcome in a specified population. It cannot establish a universal “Gateway state,” validate every Monroe claim, or establish nonlocal perception. Symthaea's model of a person's reports is not the person's subjective experience.

## 9. Selected references

- Ingendoh et al. (2023), [Binaural beats to entrain the brain? A systematic review](https://pubmed.ncbi.nlm.nih.gov/37205669/).
- Chee et al. (2024), [The effects of music and auditory stimulation on autonomic arousal, cognition and attention: A systematic review](https://pubmed.ncbi.nlm.nih.gov/38458383/).
- [Is non-clinical, personal use of binaural beats audio an effective stress-management strategy? (2024)](https://www.tandfonline.com/doi/full/10.1080/18387357.2024.2374759).
- Papadatou et al. (2026), [Amplitude Modulated Audio Signals in Brainwave Entrainment: A Scoping Review](https://www.tandfonline.com/doi/abs/10.1080/25742442.2026.2679402).
- [EEG oscillatory correlates of meditation practice: a systematic review and exploratory meta-analysis (2026)](https://pubmed.ncbi.nlm.nih.gov/42409211/).
- [Monroe Institute](https://www.monroeinstitute.org/) — programme descriptions; not evidence for the explanatory claims.
