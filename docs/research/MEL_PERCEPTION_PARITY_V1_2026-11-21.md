# MEL-PERCEPTION-001 — paired audio/symbolic perception parity v1

**Date:** 21 November 2026  
**Programme:** [#10417 — reproducible external SOTA and industry comparison](https://github.com/Luminous-Dynamics/symthaea/issues/10417)  
**Implementation:** \`crates/domains/symthaea-muse/src/perception_parity.rs\`  
**Schema:** \`benchmarks/mel/schemas/perception-parity-v1.schema.json\`

## Why this is a separate benchmark

A system that answers a question from MIDI has demonstrated symbolic reasoning over a score representation. It has not necessarily demonstrated that it can hear the corresponding music from audio.

Carone, Roman and Ripollés (PMLR 303, 2026) explicitly separate these modalities for syncopation, transposition and chord-quality tasks. Their study reports near-ceiling MIDI performance for several models but much lower audio performance on syncopation and chord quality. Gemini 2.5 Pro approaches parity on transposition, showing the gap is task-dependent rather than a universal inability to listen.

Approximate visual estimates from their Figure 2:

| Model | Task | Audio | MIDI |
|---|---|---:|---:|
| Gemini 2.5 Flash | Syncopation | ≈29% | ≈98% |
| Gemini 2.5 Pro | Syncopation | ≈41% | ≈100% |
| Gemini 2.5 Flash | Transposition | ≈71% | ≈98% |
| Gemini 2.5 Pro | Transposition | ≈95% | ≈100% |
| Gemini 2.5 Flash | Chord quality | ≈20% | ≈87% |
| Gemini 2.5 Pro | Chord quality | ≈35% | ≈99% |
| Qwen2.5-Omni | Chord quality | ≈24% | ≈51% |

These are approximate chart estimates, not exact raw counts. The study does not measure full-song collaboration or production editing.

**Research implication:** separate the perception layer from the reasoning layer. An accurate downstream symbolic solver cannot repair a wrong transcription, onset grid or chroma estimate. Uncertainty from audio perception must propagate into all downstream musical claims.

## v1 contract

Every benchmark case binds:
- an immutable source-score SHA-256;
- an audio input SHA-256;
- a symbolic MIDI input SHA-256;
- the exact renderer-recipe SHA-256;
- one task and one ground-truth answer;
- a fixed development/validation/test split.

The audio and MIDI inputs are two views of the same underlying source score. Their input digests differ; the case identity and source-score digest bind them as a matched pair.

The v1 tasks are:
1. **Syncopation scoring:** ordinal level 0–4. Report exact accuracy and mean absolute ordinal error separately.
2. **Transposition detection:** binary same/different relational judgment.
3. **Chord-quality identification:** major, minor, dominant or diminished.

A model run binds a system/version, model-identity digest, preprocessing digest and prompt/policy digest. Every prediction identifies its case, modality and exact input digest.

## Scoring rules

For every task and modality, report:
- total cases;
- answered, abstained and missing counts;
- exact accuracy over all cases (abstentions and missing are incorrect);
- conditional accuracy over answered cases;
- coverage;
- syncopation mean absolute error where applicable;
- paired audio-vs-MIDI answer agreement;
- audio accuracy minus symbolic accuracy.

No prediction is silently dropped. Duplicate case/modality outputs, unknown case IDs, wrong input digests, task-incompatible answer types, invalid confidence and invalid syncopation levels fail the run.

The all-case denominator is intentional. An evaluator must not improve its reported accuracy by abstaining on hard audio inputs or by omitting failures.

## Engineering already implemented

The Rust module provides:
- typed task, modality and answer schemas;
- content-addressed paired case records;
- immutable model/preprocessing/prompt identity fields;
- input-digest verification by modality;
- duplicate/unknown/mismatched prediction rejection;
- explicit abstention and missing-output accounting;
- per-modality accuracy, coverage, confidence presence checks and syncopation MAE;
- modality gap and paired agreement metrics.

The current tests use deterministic in-memory fixtures to validate the contract and scoring mechanics. They do not yet run a real audio perception model, do not contain a public human-recorded dataset, and do not establish a Melothaea score for any MUSE task.

## Next research increments

1. Reproduce the MUSE Benchmark stimulus protocol using rights-cleared human performances and same-source MIDI encodings.
2. Add controlled nuisance transforms: instrument/timbre, octave, tuning offset, tempo, dynamics, room/reverb, codec, noise and expressive microtiming.
3. Build a transcription/onset/chroma uncertainty ledger. Keep pitch-salience, onset-grid and chord-label uncertainty distinct.
4. Add audio-native negative controls: spectrally similar but relationally different excerpts, transposed melody pairs, syncopation shifts with equal onset count, and chord inversions/voicings that preserve timbre statistics.
5. Add cross-genre and culturally diverse rhythm/harmony cases; do not assume Western tonal labels exhaust musical structure.
6. Compare the audio front-end and downstream solver separately. Report perceptual extraction accuracy and conditional solver accuracy given a correct representation.
7. Run a blinded musician cohort for ambiguous/expert labels; preserve annotator disagreement instead of forcing uncertain examples into deterministic labels.

## Prospective targets

These are research goals, not achieved results:

| Task | First gate | Stretch objective |
|---|---:|---:|
| Audio transposition detection | ≥90% | ≥95% with ≤5 pp audio/MIDI gap |
| Audio syncopation scoring | ≥60% exact accuracy | ≥70% exact accuracy and ≤0.5 ordinal MAE |
| Audio chord-quality identification | ≥60% | ≥70% |
| Cross-task paired agreement | Report paired agreement and coverage | ≥90% agreement where both modalities answer and the answer is unambiguous |
| Coverage | ≥95% on clean in-domain audio | ≥90% under declared nuisance transforms with explicit uncertainty |
| Model identity | 100% model/preprocessing/prompt identity bound | 100% independently replayable local runs |

These thresholds must be re-calibrated after the rights-cleared cohort, class balance and scorer are frozen. The external Gemini results are not a directly matched baseline until the same stimuli, prompts, answer encoding and evaluation split are reproduced.

## Sources

- Carone, Roman and Ripollés, [LLMs can read music, but struggle to hear it](https://proceedings.mlr.press/v303/carone26a.html), Proceedings of Machine Learning Research 303, 2026.
- Duan, Yu and Briot, [Recent advances in music generation: methods, evaluation, and challenges](https://link.springer.com/article/10.1007/s10462-026-11582-x), Artificial Intelligence Review, 2 June 2026.
- MIREX 2026, [Music Evaluation via CMI-RewardBench results](https://music-ir.org/mirex/wiki/2026%3AMusic_Evaluation_via_CMI-RewardBench_Results).

## Qualification boundary

A passing test suite establishes the Rust schema, digest-binding, validation and scoring implementation on deterministic fixtures. It does not establish perceptual accuracy, human musicality, broad cultural validity, real-time performance, or superiority to external models. Those require matched public stimuli, rights and independent listening evaluation.
