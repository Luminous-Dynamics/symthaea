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

### 8. Learn what to ask next, not just what to predict

Use active learning to select the next optional rating or musical contrast when it is expected to reduce uncertainty or distinguish competing predictions. Compare uncertainty/disagreement sampling with random prompts at equal participant burden. TROMPA-MER reports prior work using collective judgment and active learning for personalized music-emotion annotation, so MLRB should compete with rather than ignore this baseline. The objective is useful information per prompt, not maximizing time spent interacting.

Provisional efficiency target: reach the same prespecified held-out MAE with at least 30% fewer feedback prompts than random sampling, with the target frozen from pilot results before confirmation. Report prompt count, completion, skipped feedback, and subgroup coverage alongside the accuracy curve. This is a proposal, not an observed result.


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

## Deterministic temporal baseline evaluator

The Rust module src/listener_response_benchmark.rs (report schema v2) provides an evaluator for a global temporal cutoff. It reports MAE for four predictors of self-reported pre/post change: no change, a training-only population mean change, a listener-history mean change, and a population/personal shrinkage baseline. The personal estimate only uses prior sessions from that listener; another presentation in the same session is never treated as personal history. The report exposes scored counts, personal-history coverage, the training prior, and an embargoed interval between training and test windows.

Each record includes collection_sequence, a monotonic study-local ordinal assigned at collection. It is not a wall-clock time. The evaluator rejects duplicate sequences/episode IDs, unsupported schema versions, unconfirmed consent, malformed record fields, reversed participant session order, any listener session that crosses training/embargo/test partitions, invalid cutoffs, and empty train/test partitions. Records between cutoffs are embargoed. Missing outcomes remain excluded from channel MAE and are reported separately.

This is a deterministic baseline evaluator, not a learned model and not a causal estimator. It asks whether added personalization beats simple baselines on later observations; it cannot establish that music caused the reported change.

### Participant-aware uncertainty

Benchmark v2 adds two matched comparisons: personal-history prediction versus no-change, and personal-history prediction versus the population-delta prior. The machine-readable record includes the bootstrap replicate count, deterministic seed, and minimum participant threshold. For each comparison, calculate per-participant MAE improvement on the same eligible future test episodes, then report the mean and median participant effect, fraction of participants improved, number of participants, and matched episode count. A reproducible percentile bootstrap resamples participants (not episodes) with 2,000 replicates and a fixed, recorded seed.

To avoid presenting a highly unstable interval as if it were confirmatory, the runner withholds the bootstrap interval when fewer than ten eligible participants contribute. Ten is only an initial display gate for exploratory analysis, not a claim that ten participants provide adequate statistical power. A scientific superiority claim still requires prospective power analysis, preregistration, adequate independent participants, and a confirmatory design. The mean over episodes and equal-participant-weighted effect are deliberately both reported: one answers typical-episode performance, the other prevents frequent listeners from dominating the listener-level result. These v2 intervals resample participants while conditioning on the observed test-stimulus mix; they do not yet quantify generalization to a new population of musical works. A future confirmatory release should add a crossed participant-by-stimulus bootstrap or an appropriate multiway clustered analysis when stimulus reuse warrants it, and must freeze a primary comparison to control multiplicity.

## Additional methodological finding

A 2025 RecSys paper, “Time to Split: Exploring Data Splitting Strategies for Offline Evaluation of Sequential Recommenders,” reports that split design can change offline model rankings and that global temporal splitting protects against future-data leakage. We adopt an explicit global ordinal and separate future holdout to make the evaluation task auditable: https://doi.org/10.1145/3705328.3748164

## Runnable deterministic CLI

The binary \`muse_listener_response_benchmark\` consumes JSONL records (one ListenerResponseRecord per non-empty line) and emits a machine-readable report bound to the SHA-256 of the exact input bytes.

Synthetic smoke test (no participant data required):

\`\`\`sh
GIT_COMMIT_SHA="$(git rev-parse HEAD)" cargo run -p symthaea-muse --bin muse_listener_response_benchmark -- --demo
\`\`\`

Evaluate an explicitly prepared dataset:

\`\`\`sh
cargo run -p symthaea-muse --bin muse_listener_response_benchmark -- \
  --input responses.jsonl \
  --training-through 800 \
  --testing-from 1001 \
  --prior-strength 3.0 \
  --output report.json
\`\`\`

The interval between the training and testing cutoffs is embargoed. A whole participant-session must fit within one partition. The report includes training, embargo and test counts; a source digest; channel-wise MAE for baseline predictors; personal-history coverage; matched-subset baseline errors; and participant-clustered improvement estimates with fixed-seed bootstrap intervals when the minimum participant threshold is met. The demo uses synthetic placeholder audio identities and is explicitly tagged as synthetic; it is never human-listener evidence.

The output records the source JSONL SHA-256, runner version, and compile-time runner revision when the build environment provides GIT_COMMIT_SHA or GITHUB_SHA. If the revision is emitted as "unbound-at-build-time", the artifact is not fully source-revision-bound; use a build environment that sets GIT_COMMIT_SHA to the checked-out commit. The input digest identifies the exact JSONL bytes but does not prove that the underlying participant consent, media rights, or audio-byte claims are valid.

The CLI and synthetic tests are authored but not locally run in this environment. Exact-head Cargo formatting, tests, and a real-input smoke run remain required before promotion.

## Dataset strategy: don't confuse the tasks

A credible benchmark should use existing open work without pretending every dataset measures the same outcome:

| Track | Candidate source | What it can establish | Important limitation |
|---|---|---|---|
| Perceived-affect baseline | MERGE audio/lyrics dataset (Zenodo record 13939205; 3,554 audio clips, 2,568 lyric records, with balanced and bimodal subsets) | Audio-only, lyrics-only and fused prediction of music's perceived valence/arousal | Static labels are not measurements of a person's pre/post state change. Check rights for each audio/lyric item before redistribution. |
| Personalized judgments | TROMPA-MER (181 participants, 4,721 annotations, 1,161 excerpts) | Personalization of perceived and induced emotion annotations, familiarity/preferences, and active-learning strategies | Its annotations do not replace a prospective, paired pre/post listening-response study. Extracted features/models may need to be obtained from the authors; the project describes the dataset as publicly viewable. |
| Emotion-aware recommendation baseline | EmoMusicLJ / EmoMusicLJ-small and the published heterogeneity-aware deep Bayesian network (HDBN) | Compare top-K recommendation against a strong model that represents emotion and preference heterogeneity both across and within users | Interaction-based recommendation metrics do not prove that music caused an affective change. Reproduce its task and splits rather than comparing HR/NDCG directly with MLRB pre/post MAE. |
| Domain-specific retrieval | CalmSet, subject to its own data/license and governance review | Graded relevance for an affective music retrieval task in its target domain | Does not represent the general population and must not be used to claim universal effects. |
| Optional physiological research track | FEEL: AfNIRS–EEG benchmark (34 participants, 38 affective music clips) | External study of subject-dependent, cross-subject, and cross-music emotion recognition | Different task; multimodal fusion did not consistently outperform single modalities across scenarios. Not required for MLRB v1; any physiological data use needs separate privacy/ethics review. |
| Personalized response change | MLRB self-report records collected under a consented protocol | Whether prior-session data predicts a listener's reported pre/post change in a future session | Observational change is not a causal music effect. Causal claims require randomized/counterbalanced interventions. |

For the recommendation sub-track, include the published HDBN approach as a serious comparator rather than relying on popularity or a weak collaborative-filtering baseline alone. HDBN explicitly models emotion heterogeneity across users, emotional variation within a user, preference heterogeneity across users, and preference variation within a user. The model and its datasets are published, making a reproduction-oriented comparison appropriate. This is a ranking baseline only; its HR/Precision/NDCG/MRR results cannot stand in for post-listening response or causal benefit.

MERGE supplies dataset-provided splits, but MLRB must still create and record its own splits aligned to the intended question. Never assume an ordinary 70/15/15 random split is adequate for longitudinal personalization. Do not mirror third-party audio, lyrics, profile data, or derived personal records until their individual licenses and permitted uses are confirmed.

Sources:
- MERGE dataset record and available subsets: https://zenodo.org/records/13939205
- TROMPA-MER study, annotations and data availability: https://doi.org/10.1007/s10844-022-00746-0
- Heterogeneity-aware Deep Bayesian Network for emotion-aware recommendation (ACM TOIS, 2025): https://doi.org/10.1145/3733233
- CalmSet dataset / intended-use description: https://github.com/Sensify-Lab/CalmSet
- FEEL: AfNIRS–EEG Emotion Dataset and Benchmark Library (2026): https://doi.org/10.1109/TAFFC.2026.3733456

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
- cargo test -p symthaea-muse --lib listener_response_benchmark::tests
- cargo test -p symthaea-muse --bin muse_listener_response_benchmark
- cargo run -p symthaea-muse --bin muse_listener_response_benchmark -- --demo
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
