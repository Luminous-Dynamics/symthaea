# Personalized Listener-Response Benchmark (MLRB) v0.1

## Purpose and boundary

This proposal adds a distinct empirical target to Symthaea–Muse: predict and evaluate what a listener reports experiencing during a particular listening episode, and whether a musical intervention serves the listener's explicitly chosen goal.

This is not the existing Sonata cognition study with different names. The current study compares fixed, random-valid, heuristic, and Symthaea composition policies using structural evidence, blinded preference, and artist-workflow outcomes. The adaptive outcome model predicts symbolic effects such as tension and density in the score; its contract explicitly does not learn listener preference. MLRB keeps these claims separate and introduces a listener-level response record for future opt-in collection.

No code change here establishes that Symthaea improves listener outcomes. The benchmark is a protocol and a data-contract foundation; empirical superiority requires prospective, adequately powered, independently inspectable evidence.

## Research motivation

- Kang and Herremans (2025) survey music-emotion datasets, models, evaluation standards, and outstanding challenges. https://doi.org/10.1109/TAFFC.2025.3583505
- Patil et al. (2025) review methods and datasets and discuss the subjective, dynamic, person-dependent, and context-dependent character of music responses. https://doi.org/10.1007/s42452-025-07178-9
- Assuncao, Piccolo, and Zaina (2022) review emotion/context-aware recommendation and identify feedback, activity, personality, learning, preference, and cold-start behavior as important dimensions. https://doi.org/10.1007/s11042-022-12110-z
- The 2024 scoping review of music emotion regulation finds a varied literature with different concepts, interventions, and measures rather than one universal response target. https://doi.org/10.3390/bs14090793
- A January 2026 paper on recommendations for constructed music-evoked episodic memories reports familiarity and genre among the stronger predictors of fit, supporting inclusion of listener meaning rather than reliance on acoustic affect alone. https://doi.org/10.1016/j.ijhcs.2025.103717

These sources motivate this work; they do not establish that any single architecture, including Symthaea, is best.

## Paradigm shifts

### 1. Predict response, not merely the emotion expressed by a recording

Keep separate labels for perceived musical affect, felt affect before listening, felt affect during or after listening, enjoyment, familiarity, and attainment of the listener's stated intent. Identifying a sad-sounding song does not demonstrate prediction of whether a particular person feels sad after hearing it.

### 2. Model change and trajectory, not only a static state

For channels with both baseline and follow-up observations, evaluate reported change and (when enough checkpoints exist) trajectory. The first target is an honest estimate of report change, not a claim that music caused it. Fatigue, concurrent activity, time, and regression to the mean can also explain changes.

### 3. Personalize with evidence and uncertainty

Use population-level evidence as a prior, then update from that listener's opt-in history. Evaluate separately: (a) cold start, with the participant absent from training; and (b) known listener, where only earlier sessions may predict strictly future sessions. With sparse history, shrink toward the population prior and widen uncertainty rather than invent a highly specific profile.

### 4. Separate prediction from causal intervention

Prediction asks what a listener may report. Causal evaluation asks whether offering or generating an alternative improves a prespecified outcome compared with a suitable control. Good prediction metrics alone do not prove benefit. Use randomized, counterbalanced within-person comparisons for intervention claims and keep assignment hidden during blinded analysis.

### 5. Use controllable generated counterfactuals

Generate matched alternatives from a common premise, then vary a declared factor such as rhythmic density, tempo, harmonic tension, dynamics, or timbral palette. Record the output hashes and actual feature differences. Do not claim to isolate one cause if several correlated features changed at once. Include same-composition/different-renderer controls where renderer effects are plausible confounds.

### 6. Optimize for the listener's stated goal, not engagement

Do not use session length, compulsion, replay count, or inferred vulnerability as a proxy for benefit. Goals may include matching a mood, shifting toward a target, sustaining focus, energizing, winding down, exploring, or enjoyment. The person remains the authority on whether the experience helped.

### 7. Treat cultural and personal meaning as first-class variables

Do not assume an identical acoustic feature means the same thing across traditions or people. Record only optional, coarse, participant-supplied context. Evaluate culture/genre slices when sample size permits and call underpowered slices non-estimable rather than reporting universal performance.

## Proposed data contract

The new Rust module, **src/listener_response.rs**, defines a versioned ListenerResponseRecord for one presentation:

- opaque, study-local participant and presentation tokens; no direct identifiers;
- SHA-256 identity for the exact presented audio, optional score hash, and optional model version;
- session and presentation order without wall-clock timestamps;
- listener-selected intent and optional activity/familiarity;
- separate self-reported pre-state, during checkpoints, post-state, and perceived musical affect;
- optional enjoyment and goal-attainment ratings;
- explicit consent confirmation and consent-protocol version;
- bounded, finite self-report scales and strictly increasing relative playback checkpoint times;
- incomplete reports remain representable so coverage and response propensity can be measured.

Scales in v1 are valence [-1, 1], arousal/tension/energy [0, 1], enjoyment 1–7, goal attainment 1–7, and familiarity 1–5. These are engineering encodings, not an exhaustive theory of human experience. UI wording and anchors require comprehension piloting across participants and languages. Affect values are self-reports; no camera, microphone, or physiological signal is part of this schema.

The schema rejects unknown schema versions and records without confirmed consent. A ShiftTowardTarget intent requires an explicitly listener-defined target. The record does not infer what that target should be. Missing dimensions remain missing; validation never silently imputes them.

## Evaluation tracks

### Track A — Perceived-affect prediction

Predict a listener's report of the affect expressed by a music stimulus. Use existing public emotion-labelled datasets where compatible, but call this perceived-affect prediction, not personal emotional impact. Document dataset, license, split, features, and label provenance.

### Track B — Individual felt-response prediction

Predict post-listening self-report from pre-listening response, stimulus/audio features, familiarity, declared intent, and coarse context. Primary target: future-session post-state or pre/post change. Never use post-listening fields, future sessions, or held-out-fold aggregates as features.

### Track C — Goal attainment

Predict the listener's own post-session goal-attainment report, then separately test whether using that prediction to select music improves the report prospectively. Predicted attainment is not clinical benefit.

### Track D — Adaptive composition

Generate candidates, freeze score/render identities, predict responses before exposure, then collect opt-in reports. Evaluate prediction error and human-rated musical quality separately. Do not alter a composition after seeing a held-out response and score it against that same response.

## Mandatory splits and leakage controls

Keep separate scorecards for:
1. unseen participants and unseen recordings;
2. known participants with strictly future sessions held out;
3. cross-style and cross-cultural transfer where evidence is sufficient.

Group near-duplicate versions and alternate renders of a composition in the same partition. Fit preprocessing and normalization from training data only. Keep variants of a listener/session block together. For causal studies, preregister assignment, outcomes, exclusions, analysis, and practical effect margin before unblinding. Use participant- and stimulus-aware uncertainty estimates when both create dependence.

## Baselines and ablations

At minimum compare:
1. no-change prediction (predict the listener's pre-state);
2. population-only model;
3. stimulus/audio-only model;
4. familiarity and preference/history-only model;
5. context and intent-only model;
6. combined non-personalized model;
7. personalized hierarchical/online model;
8. Symthaea temporal/HDC/CfC/FEP variant under equal data and compute budgets.

Model complexity earns no presumption of superiority. Select the strongest baseline on validation data. Ablate personal history, familiarity, context, intent, temporal checkpoints, and audio features independently.

Required negative controls: shuffled participant-token mapping, shuffled session order, a stimulus-identity-only memorization baseline, and a no-change/sham intervention when appropriate. A result that depends on identity or future-context leakage is a failed result.

## Scorecard and provisional targets

All thresholds are proposals for engineering prioritization. Finalize them from a separate pilot and preregister them before confirmatory outcomes are inspected.

| Dimension | Metrics | Initial target or release gate |
|---|---|---|
| Response prediction | MAE for continuous reports; Brier/log loss only for categorical targets | Target at least 10% relative MAE improvement over the strongest simple baseline on held-out future sessions; participant-clustered 95% CI must exclude no improvement |
| Calibration | Prediction-interval coverage and width; calibration curves | Nominal 90% intervals should achieve 85–95% empirical coverage on sufficiently large held-out data; report width as well as coverage |
| Individual utility | Per-listener error, goal attainment, user-rated usefulness | Report effect distribution across listeners, not just a grand mean; predefine usable-history rules |
| Intervention effect | Preregistered within-person difference on declared goal outcome, uncertainty interval, attrition, unwanted effects | No benefit claim unless the prespecified practical margin and inferential gate pass prospectively |
| Generalization | Held-out listener, recording, context, genre and cultural slices | No universal claim from a convenience sample; small/underpowered slices are non-estimable |
| Feedback burden | Prompts per session, response time, dropout and skipped-report rate | Keep reporting optional and low-friction; publish missingness and burden alongside accuracy |
| Integrity | Hash completeness, protocol adherence, leakage tests and independent reproduction | Fail closed on invalid identity, mismatched protocol, split leakage, or unrecorded deviations |

The 10% relative MAE target is a proposed engineering hurdle, not a published standard or clinical threshold. The calibration target is provisional and meaningful only with enough held-out observations. If pilot data show the margins are not reliable or practically meaningful, revise the plan before preregistration, never after confirmatory results are known.

## Execution plan

1. **Schema tranche (this PR):** land the Rust record, fail-closed range/digest validation, JSON round-trip tests, and tests proving missing responses remain missing.
2. **Offline baseline tranche:** add a deterministic benchmark runner with synthetic fixtures first; then add only licensed, compatible public datasets with documented label semantics.
3. **Collection tranche:** run an opt-in pilot with explicit consent, opaque study-local tokens, no passive sensing, and public synthetic fixtures. Size the pilot for feasibility and power the confirmatory study from pilot variance.
4. **Model tranche:** evaluate population prior, personalized update, and Symthaea under identical input, data, and compute budgets.
5. **Intervention tranche:** randomized/counterbalanced within-person comparison of recommended or generated variants, with a frozen primary outcome and no engagement-maximization objective.
6. **Release tranche:** reuse existing Muse V8–V13 manifest, blinding, confirmatory-analysis, publication, and replication machinery where compatible. Do not create a parallel governance stack.

## Schema-local verification

Run from repository root:

- cargo fmt --manifest-path crates/domains/symthaea-muse/Cargo.toml -- --check
- cargo test -p symthaea-muse --lib listener_response::tests
- cargo test -p symthaea-muse --lib

These commands have not been run in the authoring environment. Passing CI or a workflow that is pending/queued is not a pass. Schema tests cannot establish model accuracy or positive listener outcomes.

## Scope and privacy

No direct identifiers, precise wall-clock times, raw free-text diaries, facial/video/audio sensing, or physiological signals are necessary for v1. A future modality requires its own purpose, consent, retention, access, security, and evidence review. Stable participant tokens remain sensitive pseudonymous data: keep linkage keys separate, restrict access, define retention, and publish only consented records or aggregates. Never silently convert ordinary listening telemetry into psychological profiling.

## References

- Kang, J. and Herremans, D. (2025). *Are We There Yet? A Brief Survey of Music Emotion Prediction Datasets, Models and Outstanding Challenges.* IEEE Transactions on Affective Computing. https://doi.org/10.1109/TAFFC.2025.3583505
- Patil, S. et al. (2025). *Review on music emotion analysis using machine learning: technologies, methods, datasets, and challenges.* Discover Applied Sciences. https://doi.org/10.1007/s42452-025-07178-9
- Assuncao, W. G., Piccolo, L. S. G., and Zaina, L. A. M. (2022). *Considering emotions and contextual factors in music recommendation: a systematic literature review.* Multimedia Tools and Applications. https://doi.org/10.1007/s11042-022-12110-z
- *Scoping Review on the Use of Music for Emotion Regulation.* Behavioral Sciences (2024). https://doi.org/10.3390/bs14090793
- Raingeard de la Bletiere, P. et al. (2026). *A music recommendation system for constructed music-evoked episodic memories (CoMEEMs).* International Journal of Human–Computer Studies. https://doi.org/10.1016/j.ijhcs.2025.103717
