# MUSE-MIR-019 — Source-native MusicXML notation-event evaluator (V1)

**Issue:** #12043  
**Parent:** MUSE-MIR-004 score-structure evaluation  
**Input contract:** MUSE-MIR-018 per-part MusicXML notation graphs  
**Date:** 14 November 2026  
**Status:** implementation and adversarial tests authored; awaiting exact-head Rust qualification

## Why this layer is necessary

MUSE now has three distinct symbolic evidence layers:

1. MIDI event evaluation scores note events in performance time.
2. Measure/barline evaluation scores explicit notated measure boundaries.
3. MusicXML notation-event preservation retains the written score: pitched/rest/unpitched events, written pitch spelling, durations, grace/cue/chord relationships, ties, tuplets, staff/voice, instrument references, score-order attributes and auxiliary events.

Until MIR-019, these source-native MusicXML events could be retained and committed but not compared independently. Converting them back to MIDI notes before evaluation would erase the very notation semantics MIR-018 preserves.

## Paradigm shift

Do not treat transcription as one scalar accuracy value. Evaluate a score as a structured object with independent claims:

- event presence and score position;
- written pitch spelling;
- performed symbolic duration;
- written rhythmic value;
- rest and unpitched semantics;
- ties and tuplets;
- voice/staff and part identity;
- instrument-reference validity;
- score-order attributes and auxiliary events.

A system can be excellent at recovering sounding pitch while poor at written spelling, rests, ties or staff assignment. Those are different defects and should produce different diagnostics.

## Implemented V1

Adds `crates/domains/symthaea-muse/src/musicxml_notation_eval.rs`, exported behind the existing `theory` feature.

### Inputs and identity

The API accepts:
- reference and prediction `MusicXmlPartNotationGraphV1` arrays;
- explicit one-to-one reference-part to prediction-part bindings;
- a rational quarter-beat onset tolerance.

No part matching is inferred from pitch range, MIDI channel, staff number, or voice role. Every part in both graph sets must be bound exactly once.

Reference and prediction must share:
- one score-clock identity;
- the same measure-map canonicalization version;
- the same notation-event canonicalization version.

Missing or unsupported measure structure and Unknown notation events yield an explicit Unavailable outcome. They do not become zero, perfect scores or partial event arrays.

### Event alignment

V1 performs deterministic, one-to-one maximum-cardinality matching.

Candidate edges are permitted only when:
- the event family agrees: pitched, rest, or unpitched;
- the exact rational score onset lies within the declared inclusive tolerance.

Duration, pitch spelling, staff, written value, ties and tuplets are not matching prerequisites. They are evaluated after alignment. This is deliberate: a wrong note length must not prevent the note from aligning and thereby hide the duration error.

Candidate generation sorts by exact rational onset and source ordinal, uses a sliding window, and has a total comparison budget. Matching uses iterative Hopcroft–Karp, avoiding recursion-depth risk. Source ordinals are not compared across documents. Chord-member relationships are evaluated after event matching by resolving each reference base-note anchor through the matched base-event mapping.

### Independent scorecard dimensions

The report emits separate precision/recall/F1 counts for:
- all event alignment;
- pitched, rest and unpitched event alignment;
- exact written pitch spelling (step, alteration string and octave);
- rest and measure-rest semantics;
- unpitched display pitch;
- performed duration;
- written note type, dot count and time-modification ratio;
- grace/cue/chord flags and chord-anchor relation;
- voice/staff identity;
- note instrument-reference equality and resolution against declared score-instrument IDs;
- sound ties and visual tied notation separately;
- tuplet notation source structure;
- score-order attribute events and their exact source values;
- auxiliary events and their exact source values;
- backup/forward cursor movement and its exact semantics.

The evaluator reports mean and maximum absolute onset and duration errors in quarter-beat units among eligible aligned events. It does not average the dimensions into a weighted composite.

Undefined precision/recall remain None when a denominator is absent. When both denominators exist but no true positives are found, F1 is 0, not None.

### Evidence receipt

Every evaluated report binds:
- canonical SHA-256 of the reference graphs;
- canonical SHA-256 of the prediction graphs;
- part binding and tolerance configuration digest;
- evaluator and schema versions;
- all independent metrics and error summaries;
- a report SHA-256.

Replay recomputes the entire result and rejects any changed input, mapping, config, metric or digest. The digest is an integrity commitment, not a producer signature or proof of rights.

### Resource and integrity controls

- 20,000 total note/attribute/auxiliary/cursor events per part.
- 512 part bindings maximum.
- 200,000 total events across both graph sets.
- 2,000,000 candidate comparisons across all matched parts.
- Checked rational arithmetic and bounded matching.
- Reject duplicate part mappings, duplicate part IDs, invalid event source ordinals, invalid measure locators, nonpositive durations, chord anchors that do not resolve to a preceding base event, and score-clock/canonicalization mismatch.

## Tests authored

The focused evaluator module contains 16 tests, including:
- identical graph match and report replay;
- pitch and duration errors visible after onset alignment;
- one prediction cannot match two reference events;
- wrong event family cannot match;
- Unknown structure returns Unavailable;
- score-clock mismatch refusal;
- explicit renamed-part binding and duplicate binding rejection;
- independent attribute and auxiliary event scoring;
- tampered report rejection;
- exact rational tolerance edge;
- iterative augmenting-path adversary;
- exhaustive maximum-matching comparison against an oracle over all 3×3 bipartite graphs;
- zero-true-positive F1 semantics;
- independent sound tie / visual tie / tuplet / instrument reference / performance directive checks;
- unresolved score-instrument ID detection.

A focused Rust 1.96 workflow is added at `.github/workflows/muse-musicxml-event-eval.yml`. It checks formatting and runs the evaluator, notation-event parser and MusicXML graph tests. The workflow is configured but no pass is claimed until GitHub reports the exact-head result.

## Industry and benchmark comparison

### MIREX 2026 Audio-to-Score

The public result table illustrates why composer-held-out performance must be reported separately from familiar-set performance:

| Track and split | System | WER ↓ | CER ↓ | LER ↓ | MV2H ↑ |
|---|---|---:|---:|---:|---:|
| Blind / Quartets familiar | Hidalgo-Centeno | 0.0286 | 0.0276 | 0.0792 | 0.9492 |
| Blind / QuartetsMIREX held-out | Hidalgo-Centeno | 0.3385 | 0.3258 | 0.7460 | 0.6053 |
| Staves-Informed / Quartets familiar | Lu et al. | 0.2160 | 0.2385 | 0.5983 | 0.8796 |
| Staves-Informed / QuartetsMIREX held-out | Lu et al. | 0.3177 | 0.3481 | 0.7456 | 0.8179 |

Blind and Staves-Informed are separate tracks because the latter supplies staff metadata. The held-out result is materially lower on the familiar-to-unseen-composer generalization axis. MV2H aggregates multipitch, voice separation, metrical alignment, note value and harmonic analysis; MIR-019 does not reproduce that metric.

Sources:
- https://futuremirex.com/mirex/wiki/2026:Audio-to-Score_Transcription_Results
- https://music-ir.org/mirex/wiki/2026:Audio-to-Score_Transcription

### Commercial product comparison

Klangio's October 2026 vendor-authored benchmark reports, on its own 13-set/130-clip evaluation, approximately:
- 0.82 raw-note F1 at 100 ms on nine aligned sets;
- 0.67 mean score F1;
- 0.72 bar F1;
- 92% time-signature accuracy.

These are useful industry reference points, not directly comparable MIR-019 scores: the datasets, tokenization, tolerances and aggregation differ, and some sets are internal to the vendor.

Source: https://klang.io/blog/music-transcription-benchmark-2026/

### Notation versus playback

The August 2026 dual-evaluation study covers 24 pipeline configurations, 230 piano recordings, 23 works, 30 performers and six composers, plus a listening study with over 100 participants. It reports that notation similarity and rendered-playback similarity can prefer different pipelines. MUSE therefore keeps score correctness, source MIDI event preservation, rendered-audio similarity and musician correction burden separate.

Source: https://arxiv.org/abs/2608.04511

## Qualification targets (not achieved results)

1. 100% exact agreement on deterministic graph fixtures.
2. Zero false-positive event matches on wrong-family, duplicate, wrong-part and shifted-onset negative controls.
3. 100% deterministic maximum-cardinality agreement with the exhaustive oracle on bounded cases.
4. 100% replay rejection after mutating any report or graph input.
5. 100% Unavailable outcomes for unknown or unsupported notation structures.
6. Composer/song-held-out evaluation with song/composer-clustered 95% confidence intervals.
7. At least 30% lower median expert correction time than a matched baseline without increased critical notation errors. This is a human-study target, not a measured result.

## Scope boundary

- MIR-019 compares symbolic MusicXML graph outputs. It does not transcribe audio.
- It does not claim MIREX MV2H, WER, CER, LER, MUSTER, OMR-NED or CLEWS metric equivalence.
- It does not claim superiority to any vendor or SOTA system.
- Playback similarity must be evaluated only when an original audio performance exists, with renderer and soundfont identity pinned.
- The original uploaded MusicXML bytes remain authoritative; the semantic event graph is a bounded, namespace-aware representation, not a lossless XML serializer.
