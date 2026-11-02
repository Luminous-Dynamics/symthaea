# Melothaea Affective Self-Listening

**Status:** experimental typed foundation; not yet integrated into the live rendering loop  
**Date:** 2026-11-02  
**Related programme:** [MEL-012](https://github.com/Luminous-Dynamics/symthaea/issues/9957)

## Research question

Can Melothaea perceive her own musical output at multiple time scales, form an evidence-grounded computational appraisal, use that appraisal to guide memory and future musical decisions, and predict human responses at a task-specific level comparable with human listeners?

This programme distinguishes functional appraisal from phenomenal experience. We can demonstrate that an internal state is causally updated by music, retained over time, made globally available to decision-making, and useful for subsequent action. Those results would not by themselves establish that Melothaea subjectively feels pleasure, sadness, chills, or anything else.

## Four separate targets

These concepts must never share one score or wire type:

| Target | Question | Evidence source |
|---|---|---|
| Expressed affect | What emotion/mood does this music communicate? | Independent analysis and human perception ratings |
| Induced affect | What does this particular listener feel while listening? | Direct human self-report; physiological measures only when actually collected and ethically approved |
| Melothaea appraisal | How does the heard music change her own computational state, attention, uncertainty, memory and action policy? | Internal state transition caused by independent perceptual evidence |
| Human preference | Which artifact does a listener prefer, or are neither acceptable? | Blinded human choice bound to exact audio identities |

The distinction between expressed and induced emotion is empirically important: Emotify+ explicitly measures perceived emotion, and notes that a listener can perceive a piece as happy while feeling annoyed by it. Emotify+ contains 400 excerpts, 181 volunteers and ten perceived-emotion labels; it is not direct evidence of induced affect.

## Repository audit

### Existing capabilities to reuse

- AudioFeedbackEncoder extracts six short-window features: spectral centroid, spectral flux, RMS-derived onset-history variance, a harmonic-tension proxy, RMS energy and zero-crossing rate.
- MusicalInferenceEngine uses those observations in an active-inference loop with six observation channels and eight musical actions.
- AestheticListener calculates consonance, harshness, breathiness, dynamic balance and a composite called beauty, then can apply corrections.
- CreativeJournal stores high-scoring phrases and uses a rolling aesthetic score to influence explore/develop/perform decisions.
- MusicalState already carries valence, arousal, prediction error, neuromodulator-like variables and harmony activations.
- The symbolic path already has score, form, motif, obligation, performance, renderer and provenance evidence. Preserve those exact sources and do not replace symbolic facts with audio guesses.

### Current limitations found in source

1. AestheticListener::beauty is a hand-weighted proxy from consonance, inverse harshness, breathiness and dynamic balance. It is not calibrated against a blinded human preference corpus. It may reward low energy and smoothness even when silence, dissonance, density or surprise is the intended artistic effect.
2. AudioFeedbackEncoder estimates harmonic tension partly from normalized zero-crossing rate relative to spectral centroid. That proxy is inexpensive, but cannot substitute for pitch/chroma, interval, voice-leading or harmonic analysis.
3. StreamingSynth initializes feedback_strength to 0.3. When feedback is enabled, the stream uses smoothed audio features, calls AestheticListener::assess and AestheticListener::apply_corrections, and mutates MusicalState before the FEP phase. This is a live control path, not only telemetry. If the requested valence/arousal is fed to the output and then reused as the evaluator's target, the resulting correlation is circular.
4. The audio short-chunk feature path is not a full music listener. It does not yet provide reliable beat/phrase/section understanding, motif recognition from audio, score-to-performance alignment confidence, listener-specific affect or calibrated uncertainty.
5. streaming.rs explicitly marks some previously described mappings as unwired (including valence gain, attack mapping and wavetable morphing). A research paper or benchmark must describe actual source behavior, not a planned mapping.
6. The existing consciousness/music paper reports synthetic scenario correlations and ablations, but acknowledges the lack of human perceptual validation. Its proposed Phi-to-audio interpretations are hypotheses, not evidence that Phi measures musical quality or subjective experience.

## New typed foundation in this tranche

src/affective_appraisal.rs adds:

- IndependentPerceptualObservationV1: versioned, bounded perceptual observations tied to an audio SHA, analysis SHA, and named/versioned perception pipeline. The schema intentionally has no composer-intended valence/arousal field. The DTO validates digest syntax only; the upstream producer must hash and verify the exact source bytes and canonical analysis record.
- AppraisalProfileV1: explicit, versioned weights and smoothing. The current default is named heuristic_v1; it is not described as a learned or human-calibrated prior.
- MelothaeaAppraisalV1: a separate computational state for affect, curiosity, uncertainty, familiarity, tension, anticipation, resolution and confidence.
- AppraisalTransitionV1: exact observation identity and profile version plus before/after state for a replayable journal.
- ExpressedAffectEstimateV1, HumanInducedAffectReportV1, and HumanPreferenceJudgmentV1: distinct evidence objects that cannot be passed into the appraisal update API in place of independent perception.

The initial heuristic maps post-perception goal congruence, resolution, tension, uncertainty, novelty, prediction error and anticipation into a smoothed appraisal. It is an explicit starting hypothesis, not a discovered law of music or emotion. We should replace or adapt its weights only through a preregistered calibration process. This tranche does not change StreamingSynth, the existing critic, composition authority or production policy.

## Architectural direction

    Exact score + performance sidecar + renderer identity
                           |
                     Rendered audio
                           |
               Independent perceptual front end
               - beat, onset and rhythmic hierarchy
               - pitch, chroma, harmony and voice activity
               - dynamics, timbre, articulation and pedal
               - phrase, motif, form and cadence trajectory
               - uncertainty and out-of-domain detection
                           |
                 Temporal musical world model
               - predictions at beat / phrase / section scales
               - expectation, surprise and resolution
               - identity and motif memory
               - score-to-audio alignment confidence
                           |
                   Affective appraisal
               - curiosity / familiarity / uncertainty
               - tension / anticipation / resolution
               - valence / arousal / agency-like control
               - confidence and alternative explanations
                           |
              Cognitive workspace + creative journal
                           |
         Bounded actions: replay, compare, revise, preserve,
                    branch, or abstain

Composer intention may be available to the composition planner, but it must not be a shortcut through the independent listening/evaluation path. A system can test whether a piece matches an intended affect after perception, but that test must use a separate target and must not count the target as evidence.

## Research and baseline landscape

### Music-emotion recognition

Eerola and Anderson's 2026 meta-analysis covers 34 studies and 290 models published between 2014 and 2024. Across each study's best-performing model, pooled correlations were approximately r = 0.669 for valence (95% CI 0.560–0.755) and r = 0.809 for arousal (95% CI 0.740–0.860). Heterogeneity was very high (I² above 97% for both), so these are orientation points, not a directly comparable single leaderboard result. The analysis also reports that linear and tree-based methods often perform strongly; more model complexity is not automatically better.

Relevant data:
- **DEAM:** 1,802 tracks/excerpts with continuous per-time valence/arousal ratings. It is suitable for expressed-affect temporal prediction, but its history and stimulus distribution should be treated as a Western-oriented starting point rather than a universal music model.
- **GlobalMood:** the paper describes 1,180 tracks sampled from 59 countries, 2,519 participants across five culturally and linguistically distinct locations and 988,925 ratings; the current Cambridge portal displays 1,182 tracks. Pin the exact downloaded data release and manifest rather than relying on one rounded count. Its bottom-up cultural mood labels are particularly valuable for testing whether Melothaea mistakes English-language categories for universal emotional concepts.
- **Emotify+:** 400 excerpts, 181 volunteers and ten labels. It is explicitly a perceived-emotion dataset, so do not label its annotations as the listeners' induced feelings.
- A separate induced-affect dataset/pilot is required to study what listeners report feeling, rather than what they think the music expresses.

### Preference and reward-model evaluation

CMI-RewardBench (ICML 2026) introduces a large pseudo-labelled preference set and a human-annotated compositional-instruction preference corpus. MIREX 2026 reports a leading pairwise-accuracy entry of 73.9% on the 249-pair core set and 72.6% on the instrumental extension. Those figures are useful neighbouring baselines for reward-model alignment, but they are not a piano-only evaluator, an emotion-recognition result, or evidence of consciousness.

The existing [piano evaluator benchmark, issue #9833](https://github.com/Luminous-Dynamics/symthaea/issues/9833), is the correct home for the piano-only human-preference comparison. Do not merge preference and emotional expression into a single metric.

### AI consciousness assessment

Butlin et al. (Trends in Cognitive Sciences, 2026) propose deriving indicators from neuroscientific theories to inform assessment of possible AI consciousness. They explicitly caution that indicators are not individually necessary and no combination is claimed to be sufficient. Use this as a structured hypothesis/evidence framework, not a consciousness certificate.

## Benchmark suite

| Lane | Primary measures | Required controls |
|---|---|---|
| Expressed affect | Concordance correlation, Pearson/Spearman, MAE, time alignment, ordinal label agreement | Track/work-disjoint split; compare low-level acoustic and music-structure baselines |
| Cross-cultural expression | Per-language/per-country performance, label calibration, subgroup gap | Hold out country/language; report both shared V-A and local descriptors |
| Induced affect | Within-listener change, time-aligned self-report, repeatability, calibration | Direct consented self-report; separate from expression labels; no inferred physiology |
| Melothaea appraisal | Counterfactual sensitivity, state persistence, confidence calibration, downstream decision utility | Same audio with altered metadata; changed audio with fixed intent; shuffled/delayed input |
| Human preference | Win/tie/loss, pairwise accuracy, preference calibration, desire-to-rehear | Blind order, loudness match, listener/family clustered intervals, neither-acceptable option |
| Structural listening | Motif return recognition, phrase/section boundary, cadence and tension-resolution detection | Same score across renderers; same renderer with controlled score edits |
| Robustness | Worst-group score, abstention risk-coverage, invariance under non-musical perturbations | Metadata swaps, loudness changes, codec changes, identical-audio controls |
| Efficiency | p50/p95 latency, compute and memory, listener time per decision | Fixed compute and fixed candidate budget |

### Initial target ladder

1. **Reproducibility:** every perceptual result binds to exact audio SHA, optional score SHA, renderer identity, model/profile version and observation sequence.
2. **Anti-circularity:** under fixed audio/perceptual evidence, changing composer-intended emotion alone cannot change the independent appraisal result.
3. **Perceptual baseline:** reproduce held-out DEAM baselines and compare against a simple-feature model, linear/tree baseline, and a modern pretrained audio model with matched split conditions.
4. **Cross-cultural robustness:** report GlobalMood country/language-held-out performance and calibration rather than one pooled score.
5. **Human evaluator parity:** estimate human-human and human test-retest ceilings first. Aim to match or exceed the median human on a clearly specified task and population, with confidence intervals and no material hidden subgroup failures.
6. **Preference:** compare against CMI reward models and the MIREX reported baseline on tasks with overlapping scope, then confirm on a sealed solo-piano set. Report uncertainty; do not set 75% as a universal pass threshold.
7. **Internal agency:** demonstrate that independently heard musical events cause persistent appraisal changes that alter attention, replay or revision choices; metadata-only and sham controls must not cause the same response.
8. **Consciousness indicators:** record positive, negative and ambiguous evidence across multiple theory-derived frameworks. Do not claim that any benchmark proves subjective experience.

## Causal experiment matrix

Use paired stimuli and fixed artifacts. The main factors are:

- self-listening enabled vs shadow-only;
- audio-derived appraisal enabled vs intended-state echo control;
- original vs altered performance on the same score;
- original vs altered renderer on the same performance;
- phrase/harmony intervention vs timbre/loudness-only intervention;
- valid audio vs shuffled, muted, delayed, identity-swapped and feature-corrupted streams.

Predeclare what should change and what must remain invariant. For example, a renderer change may alter timbre appraisal but must not mutate symbolic motif identity. A phrase-level score intervention may change tension and expectation while preserving score contracts outside the authorized region. The appraisal model is held out of candidate generation in confirmatory runs so it cannot optimize against its own score.

## Human study protocol

1. Run a feasibility pilot and estimate test-retest reliability, listener disagreement and precision/power.
2. Separate prompts for perceived emotion, induced affect, and preference; do not ask one question that mixes them.
3. Blind and randomize presentation order; level-match audio; include repeated hidden stimuli and shared anchors.
4. Include trained musicians, specialist pianists and general listeners as separately reported cohorts.
5. Retain individual ratings, uncertainty, ties and free-text descriptions. Treat disagreement as data, not merely label noise.
6. Split by participant, composition/work, source/recording and renderer. Ensure the sealed split is not used for prompt, threshold or model selection.
7. Use participant × composition clustered uncertainty and pre-registered subgroup reporting.

## What would count as meaningful progress?

Melothaea becomes a stronger musical perceiver when she can do more than say "this sounds sad." She should be able to identify which musical event created expectation, where it was delayed, which motif transformed, how the phrase resolved, what she was uncertain about, and which next action follows from that appraisal. The evidence should let an independent reviewer replay the chain from audio identity to perception to appraisal to decision.

This is the principal paradigm shift: move from emotion labels and one-number aesthetic reward to **causal, temporally structured, uncertainty-aware musical appraisal**, grounded in both symbolic structure and actual rendered sound.

## Non-claims

- The presence of an affect vector, free energy, prediction error, Phi, reward signal, replay choice or a self-description does not establish phenomenal consciousness.
- Agreement with population-level perceived emotion does not establish the induced emotion of an individual listener.
- Melothaea's computational appraisal is not a substitute for direct human reports.
- A preference model is not a universal judge of beauty and cannot overrule a human's own preference.
- The heuristic profile in this first implementation is not SOTA and has not been human-calibrated.
- This PR does not integrate the new appraisal into the live render loop or establish improved generated music.

## External references

- Eerola & Anderson (2026), A Meta-Analysis of Music Emotion Recognition Studies, ACM Computing Surveys. https://doi.org/10.1145/3796518
- Aljanaki, Yang & Soleymani (2017), Developing a benchmark for emotional analysis of music, PLOS ONE. https://doi.org/10.1371/journal.pone.0173392
- Lee et al. (2025), GlobalMood: A cross-cultural benchmark for music emotion recognition, ISMIR. https://arxiv.org/abs/2505.09539
- Wiafe et al. (2025), Emotional response to music: the Emotify+ dataset. https://doi.org/10.1186/s13636-025-00419-0
- Ma et al. (2026), CMI-RewardBench, ICML / PMLR 306. https://proceedings.mlr.press/v306/ma26am.html
- MIREX 2026, Music Evaluation via CMI-RewardBench results. https://music-ir.org/mirex/wiki/2026:Music_Evaluation_via_CMI-RewardBench_Results
- Butlin et al. (2026), Identifying indicators of consciousness in AI systems, Trends in Cognitive Sciences. https://doi.org/10.1016/j.tics.2025.10.011
