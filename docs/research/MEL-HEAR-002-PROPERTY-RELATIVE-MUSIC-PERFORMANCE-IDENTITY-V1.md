# MEL-HEAR-002 — Property-relative musical performance identity v1

**Status:** synthetic-first Rust evaluator; awaiting exact-head qualification  
**Date:** 27 November 2026  
**Companion:** #13908; MusicListenBench v1.0 adapter #13703 / PR #13857

## 1. Why this is separate from MusicListenBench v1.0

MusicListenBench v1.0 tests a clean comparison, an intentional musical FLIP, and an acoustic STAY such as room/EQ/hiss or relative-pitch-preserving transposition. The current MEL adapter reproduces that protocol without changing the official split or its metrics.

The new question is more demanding: **does a model recognise the same musical property across genuinely different performances?** A new take or performer changes many waveform details at once. It is not equivalent to adding hiss or EQ to the same recording.

The November 2026 MusicListenBench paper reports that a trained model called 27 of 42 same-chord two-strum real-recording pairs different. That finding motivates a matched performance-STAY extension; it is not a result from Melothaea. The official benchmark leaderboard says its self-reported results have not yet been reproduced by maintainers.

References:
- MusicListenBench paper (22 Nov 2026): https://arxiv.org/abs/2611.29375
- MusicListenBench official page and protocol: https://amaai-lab.github.io/MusicListenBench/
- MusicListenBench repository/data ledger: https://github.com/AMAAI-Lab/MusicListenBench

## 2. Paradigm shift: identity is property-relative

There is no universal binary “same music” relation. Each case identifies one musical property and asks whether that property is preserved across the comparison.

The v1 ontology separates:
- chord quality;
- root pitch class;
- voicing;
- relative interval sequence;
- melodic contour;
- rhythmic ratio pattern;
- pulse rate;
- instrument family.

Examples:
- Global transposition can preserve chord quality and relative interval sequence while changing root pitch class.
- Inversion can preserve chord quality while changing voicing.
- Global tempo change can preserve rhythmic ratios while changing pulse rate.
- A new strum can preserve chord quality but change articulation and possibly rhythm.
- A renderer change can preserve symbolic harmony while changing timbre; it is not a STAY for an instrument-family query.

The evaluator rejects incompatible property/factor combinations rather than forcing them into one generic label.

## 3. V1 comparison contract

Each triad contains:

1. **CLEAN:** a reference comparison. Expected relation is Same.

All three arms must use the exact same question bytes. The controlled variable is the audio/performance realization and its target-property relation—not prompt wording. The evaluator recomputes each question digest and rejects within-triad prompt drift, including drift paired with a freshly recomputed digest.
2. **Performance-STAY:** a different performance realization that preserves the named property. Expected relation is Same.
3. **Musical-FLIP:** a matched change to the named property. Expected relation is Different.

Performance-STAY factors:
- different take by the same performer;
- different performer with the same instrument family;
- strum or articulation change;
- expressive timing change;
- a different tempo preserving the rhythmic-ratio pattern;
- a different renderer for the same performance.

Musical-FLIP factors map one-to-one to the target-property ontology. The Rust evaluator validates the declared factor, exact audio/question digest syntax, recomputed question hash, semantic A/B option mapping, distinct audio assets, model/runtime/decoding identity consistency and performance identity metadata.

The performance-STAY validator requires:
- different take and performance IDs for new takes/performers;
- a distinct performer identity for the different-performer factor;
- a preserved instrument-family identity for the same-instrument comparison;
- same take/performance identity but changed renderer for renderer-only STAY;
- tempo scaling only for the rhythmic-ratio property, never pulse rate;
- strum/articulation changes not to stand in for voicing or rhythmic-ratio invariance.

Musical-FLIP uses the property-specific flip factor and a distinct audio digest. For InstrumentFamily, the instrument-family reference digest must change; for non-timbre properties, instrument family and renderer must remain fixed. The pairing manifest is still a supplied digest, not authenticated evidence that exactly one semantic variable changed.

## 4. Metrics — separate, no composite

Report independently:
- CLEAN accuracy/coverage;
- performance-STAY false-difference rate;
- musical-FLIP miss rate;
- each error rate conditional on a correct CLEAN answer;
- joint exact CLEAN + Performance-STAY + Musical-FLIP rate;
- unparseable and abstention counts per condition;
- slices by target property, performance-STAY factor and musical-FLIP factor;
- performance-STAY false-difference slices by performer, take and renderer;
- source-work and performer cluster counts.

Do not average chord quality with root identity or relative interval sequence. Do not pool acoustic nuisance STAY with performance STAY. Do not call a small synthetic result a human-performance improvement.

## 5. Synthetic controls and current scope

The committed eight-triad panel spans chord quality, root, interval sequence, rhythmic ratio, instrument family and voicing, with multiple performance-STAY factors. It deliberately contains:
- one wrong CLEAN response;
- one false-difference answer on Performance-STAY;
- one missed Musical-FLIP.

This is a deterministic plumbing fixture. Audio bytes are not materialized/rehashed, target-property references are not authenticated, no model is run and there is no musician study.

The report sets:
- `target_property_reference_authenticated = false`;
- `pairing_manifests_authenticated = false`;
- `audio_bytes_recomputed = false`;
- `population_confidence_interval_computed = false`.

## 6. Dataset and rights ladder

1. Start with symbolic-score counterfactuals and independently rendered controls. These establish exact target-property semantics, not real performer generalization.
2. Audit a rights-cleared GuitarSet pilot. GuitarSet's 360 excerpts across six players provide time-aligned chord/beat/pitch-related information, but performer and progression coverage is limited. Its exact version, annotation scope and permissible transformations must be reviewed before use: https://guitarset.weebly.com/ and https://doi.org/10.5281/zenodo.3371780.
3. Consider IDMT-SMT-Chords and IDMT-SMT-Chord-Sequences for cross-instrument controlled comparisons. They are explicitly evaluation-only under CC BY-NC-ND 4.0; do not train on them or redistribute transformed audio without permission:
   - https://www.idmt.fraunhofer.de/en/publications/datasets/chords.html
   - https://www.idmt.fraunhofer.de/en/publications/datasets/chord-sequences.html
4. Build a separately consented, compensated musician cohort with repeated takes and controlled changes. Obtain distinct purpose grants for annotation, internal evaluation, public release, training and commercial use.

No raw audio or dataset archive belongs in Git.

## 7. Prospective SOTA targets (not results)

- First reproduce MusicListenBench v1.0 on the exact official split; leave its official scores unchanged.
- Reach 100% on deterministic target-property synthetic controls.
- On a preregistered real-performance cohort, reduce false-difference on repeated same-chord takes below the reported 27/42 result, with performer- and source-work-clustered uncertainty.
- Improve performance-STAY while preserving target-specific musical-FLIP sensitivity.
- Demonstrate transposition invariance for relative intervals while still detecting a root-pitch-class FLIP.
- Set sample size and non-inferiority margins after a musician-reviewed pilot and power analysis. The proposed 30-source-work floor is a reporting warning, not a power calculation.

A credible superiority claim requires exact baseline reproduction, matched model/prompt/runtime/inference budgets, source-work/performer-disjoint evaluation, preregistered primary outcomes, ablations, independent human labels, source-clustered intervals and no material regression in any hard-gated property stratum.

## 8. Claim ceiling

Passing the Rust tests proves the evaluator's structural contract only. It does not prove that the audio assets differ in the declared manner, that the labels are authentic, that Melothaea recognises musical identity across performances, or that musicians benefit. No SOTA or human-level claim is made.
