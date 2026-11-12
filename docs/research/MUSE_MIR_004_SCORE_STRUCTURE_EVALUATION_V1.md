# MUSE-MIR-004 — Score-structure and measure-alignment evaluation (V1 design)

**Parent evaluator:** [MUSE-MIR-001 / #11597](https://github.com/Luminous-Dynamics/symthaea/pull/11597)  
**Part topology:** [MUSE-MIR-003 / #11660](https://github.com/Luminous-Dynamics/symthaea/pull/11660)  
**Experiment receipts:** [MUSE-MIR-002 / #11637](https://github.com/Luminous-Dynamics/symthaea/pull/11637)  
**Date:** 12 November 2026  
**Status:** design/specification; no score-structure metric is implemented by this document

## Why a fourth evaluation layer

The current MIDI evaluator measures note events, and MUSE-MIR-003 measures whether matched events preserve logical PartIds. Neither can answer whether a human can read the result as a score.

A transcription may recover most pitches and onsets yet place notes in the wrong bars, infer the wrong meter, lose tuplets, flatten rubato, omit rests, or produce voice-crossing/engraving errors. Conversely, a notation system can make a visually plausible score that is acoustically wrong. These are different failure modes and must have separate metrics.

MUSE-MIR-004 proposes a score-structure benchmark that evaluates:

1. beat/downbeat and tempo-map recovery;
2. meter and measure/barline placement;
3. note placement within a measure, not merely absolute MIDI seconds;
4. rhythmic-value and tuplets/rest structure;
5. part/staff/voice consistency;
6. semantic score correctness and notation edit burden;
7. acoustic/rendered playback fidelity and musician usability as separate axes.

No scalar aggregate should obscure a weak dimension. Report a vector of metrics, with the dataset, tolerance and aggregation method attached to every value.

## Research evidence and comparison boundaries

### Commercial-product reports

The October 2026 Klangio vendor benchmark runs 130 one-minute clips across 13 sets. Its reported averages include raw-note F1 at a 100 ms onset tolerance, score F1, bar F1, time-signature accuracy and score-error rate. It reports Klangio 0.82 raw-note F1 / 0.67 score F1 / 0.72 bar F1 / 92% meter accuracy / 0.41 score-error rate; Songscription 0.77 / 0.47 / 0.47 / 78% / 0.75; MuScriptor 0.73 / 0.29 / 0.17 / 43% / 1.00. The same report says MuScriptor fell back to default tempo on 117 of the 130 clips. These are the vendor's own results, include internal sets, and are not an independently reproduced leaderboard.

The score F1 compares the right pitch, right bar and right position on a 32nd-note grid. The benchmark allows a global offset of up to three bars and tempo doublings. Bar F1 counts barlines within one-sixth of a beat of annotated downbeats. Score-error rate is described as the share of notes a user would still need to edit. Exact formulas must be reconstructed from the benchmark publication before claiming numerical parity.

Source: https://klang.io/blog/music-transcription-benchmark-2026/

### Songscription / Play Anything report

The October 2026 report states:
- 250 full-length songs: headline MIDI composite 59.5 for Songscription, 50.6 Klangio, 49.7 Mirelo, 46.6 MuScriptor, 46.3 YourMT3+, 35.8 Basic Pitch.
- 200 multi-instrument recordings: MIDI composite 52.2, 45.1, 41.0, 40.4, 39.8, 32.7 in the same respective ranking.
- 28 held-out professional multi-instrument score references: score composite 61.1, 46.5, 43.8, 40.0, 36.8, 31.8.

The MIDI composite is an average of onset F1, onset-offset F1 and frame F1. The sheet-music composite averages MUSTER and OMR-NED. These are not directly interchangeable with raw F1, Klangio's 100 ms raw-note F1, or MuScriptor's D_Test results. The report is vendor-published, and each product is shown in its best-scoring mode.

Source: https://playanything.app/blog/state-of-the-art-music-transcription

### Open multi-instrument challenge

The 2025 AMT Challenge paper, published in 2026, reports eight valid systems. MIROS scores 0.5998 F1, 0.6558 precision, 0.5724 recall and 0.7391 overlap; YourMT3-YPTF-MoE-M scores 0.5938 F1; MT3 baseline scores 0.3932 F1. This is a separate constrained challenge protocol, not comparable to the vendor tests or MulTTiPop.

Source: https://arxiv.org/abs/2603.27528

### MulTTiPop

MulTTiPop contains 572 time-aligned commercial-pop segments (3.5 hours), from 374 songs and 263 artists. Its artist-separated test split has 403 segments / 152 minutes. The paper's Table 4 reports YourMT3+ exact-instrument onset F1 11.51 and harmonic/percussive onset F1 28.74 at 50 ms; MT3 reports 10.87 and 27.82. The arXiv abstract separately states that the best model reaches 38% Onset F1, which is not reconciled with the visible Table 4 values. Resolve the version/protocol discrepancy before using 38% as a target. The authors explicitly recommend evaluation-only use, not training. Obtain only the permitted linked audio segments and respect the underlying rights.

Source: https://arxiv.org/abs/2607.08756

## Paradigm shifts

### 1. Infer the metrical structure before committing notation

Replace “transcribe notes, then snap to a grid” with a joint structure-and-event inference graph:

audio evidence → beat/downbeat hypotheses → tempo map and meter hypotheses → measure/barline lattice → part/voice hypotheses → note-event hypotheses → notated score

The structure is a latent hypothesis with confidence and alternatives, not a single global BPM guess. Notes constrain beat hypotheses; beat/measure consistency constrains note placement; the system retains competing interpretations when the audio is ambiguous.

### 2. Separate the performance clock from the notated score clock

The performance clock contains expressive timing, rubato, swing, humanization, tempo drift and audio/render delay. The score clock contains rational beat positions, measures, tuplets, pickups, ties and notated durations. Store an explicit alignment map between them. Never overwrite observed performance timing with quantized notation, or mistake an aligned audio offset for a musical barline.

### 3. Use a typed musical score graph, not a flat MIDI note list

The canonical score representation should include work/section, measure, beat position, part, staff, voice, note/rest, duration, pitch spelling, ties, tuplets, grace notes/ornaments, articulations and dynamics. PartId remains the logical identity; MIDI track/channel and GM patch are realization metadata. Every score edit preserves event lineage or emits an explicit remap.

### 4. Make rhythm and notation uncertainty explicit

Represent meter, downbeat, beat phase, tempo-map segments, tuplets, pickup length and voice assignment as versioned hypotheses with confidence and evidence references. Abstain from a false exact score rather than silently converting ambiguous rhythm into simple quarter/eighth notes.

### 5. Treat human corrections as labelled evidence and product feedback

A useful co-creation system should know which structure the musician corrected, which unaffected parts were preserved, how much time the correction saved, and whether the musician accepted the result. The benchmark should measure musician effort and correction locality, not only model similarity.

## Proposed scorecard

The first implementation should be a deterministic evaluator over a normalized MusicXML/score representation plus reference and prediction MIDI/audio. Keep these families separate.

| Dimension | Metric proposal | Unit / interpretation |
|---|---|---|
| Beat tracking | beat precision/recall/F1 and continuity | beat events with a documented tolerance |
| Downbeat tracking | downbeat F1 | downbeat events, same declared tolerance |
| Tempo | segment-wise tempo MAE, median absolute log-ratio, tempo-change recall | BPM and dimensionless ratio |
| Meter | exact meter accuracy, per-measure accuracy, abstention coverage | exact numerator/denominator; report coverage |
| Barlines | barline precision/recall/F1 | candidate downbeat within 1/6 beat for Klangio parity |
| Score-note placement | score F1 on rational bar/beat positions | correct pitch, part/voice and position |
| Rhythm values | duration/value accuracy; tuplet/rest/tie F1 | notated duration classes and structure |
| Voice/part topology | MUSE-MIR-003 track/lane purity and part coverage | structural, not acoustic instrument identity |
| Score edit burden | MUSTER metrics and OMR-NED | lower edit/error rates are better |
| Acoustic fidelity | separate rendered-audio similarity and calibrated human preference | renderer- and dataset-specific |
| Human utility | median correction time, edits/minute, accept-without-edit rate, undo rate | musician workflow outcomes |

Metric versions must specify canonicalization, note matching, part/voice matching, grid resolution, pickup-bar policy, tempo-doubling policy, instrument hints and empty-reference handling.

### MUSTER and OMR-NED

MUSTER compares two MusicXML scores using edit-distance-style score transcription error metrics, with separate measures for musical score components and voices. It should be included as a pinned external implementation rather than reimplementing only a subset.

OMR-NED is a normalized edit-distance metric for optical music recognition and score-symbol errors. The Sheet Music Benchmark introduces it for Humdrum **kern and analyzes symbols such as noteheads, beams, pitches and accidentals. For audio-to-score evaluation, define the precise MusicXML-to-notation normalization path before comparing a value to OMR-NED publications; do not apply a raster OMR metric blindly to arbitrary MusicXML encodings.

Sources:
- MUSTER: https://amtevaluation.github.io/
- Sheet Music Benchmark / OMR-NED: https://arxiv.org/abs/2506.10488

## V1 data and protocol design

### Dataset policy

- **Klangio 130-clip benchmark:** attempt reproduction only where the clips, references and formulas are legitimately available. Vendor internal sets are not automatically reproducible.
- **Songscription:** use its published 200 multi-instrument and 28 score values as external context only unless the exact sample manifest, references and system settings can be lawfully obtained.
- **AMT Challenge:** reproduce the public challenge evaluation protocol with all documented constraints, not only its headline score.
- **MulTTiPop:** evaluation-only; use its artist-separated dev/test splits exactly; do not train or tune on its test set.
- **Public symbolic score sets:** pair aligned audio/MIDI/score sources such as MAESTRO/ASAP for piano, URMP for ensembles and Slakh2100 for synthetic multitrack controls. Record training overlap and source licenses.
- **Commissioned corpus:** build a rights-cleared ensemble dataset with full score, stems/audio, expressive performance, performer annotations and permission for commercial evaluation. Include non-Western and mixed-meter traditions through co-designed annotations and paid expert review.

### Statistical protocol

- Pre-register hypotheses, primary metrics, splits, tolerances and exclusion rules.
- Freeze audio, reference score, renderer, soundfont, resampler, adapter, model checkpoint, settings and exact hashes.
- Report performance per dataset, instrument family, genre, polyphony/density, tempo, meter, rubato, recording quality and overlap stratum.
- Use paired bootstrap confidence intervals clustered by song/artist, not by individual note.
- Report both macro-per-song and micro-per-note aggregates; do not let long tracks dominate all results.
- Blind musician A/B studies, loudness-match audio, randomize order and report inter-rater reliability.
- Separate audio-only, score-conditioned, instrument-hinted and human-corrected settings; never put them on one leaderboard row.
- Keep evaluation-only content out of model training and tuning.

## Proposed targets (research goals, not achieved results)

1. Reproduce the Klangio benchmark's own score definitions before using 0.72 bar F1, 92% meter accuracy and 0.67 score F1 as like-for-like baselines.
2. On an independently held-out, rights-cleared ensemble corpus, target at least 0.80 bar F1 and 0.95 exact meter accuracy at declared coverage, then improve score F1 beyond 0.67 with lower edit burden than 0.41 under a reconstructed matching protocol. These are stretch goals.
3. For the public AMT Challenge protocol, first reproduce MIROS 0.5998 F1 and YourMT3+ 0.5938; then target a statistically significant gain at matched data, compute, instrument hints and post-processing.
4. Improve score-structure calibration: expected calibration error below 0.05 for meter/downbeat hypotheses, and coverage-risk curves that show when the model abstains rather than fabricates a measure structure. These are proposed engineering thresholds, not published baselines.
5. On musician workflow, target at least 30% lower median correction time than the best reproduced baseline without degrading score correctness, with 95% confidence intervals and stratification by musician expertise.
6. On every benchmark, preserve raw note-event metrics and publish the full scorecard. Never claim a win from a single composite while a material submetric regresses.

## Implementation sequence

1. Freeze an independent normalized MusicXML score IR and canonicalization version.
2. Add MIDI time-signature, tempo-map and measure-grid extraction without changing existing note-event scoring.
3. Add reference/prediction beat and downbeat events plus beat/downbeat matching.
4. Add barline and meter scoring, including pickup and tempo-doubling controls.
5. Add score-note position/duration/voice metrics over rational measure positions.
6. Add pinned MUSTER/OMR-NED parity fixtures and document non-equivalent fields.
7. Add a benchmark runner that emits dataset/model/adapter/runtime/rights receipts from MUSE-MIR-002.
8. Add blind musician correction-time studies and public, aggregate-only reporting.

## Non-goals for V1

- Claiming acoustic instrument identity from MIDI track, channel or GM program.
- Treating a global tempo estimate as a complete expressive timing model.
- Conflating raw event F1, score F1, OMR-NED and listening preference.
- Releasing copyrighted commercial audio or reference scores without permission.
- Claiming industry parity based on vendor numbers whose exact samples, formulae or system settings are unavailable.
