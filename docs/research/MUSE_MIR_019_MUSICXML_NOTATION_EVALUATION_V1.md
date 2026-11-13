# MUSE-MIR-019 — Multidimensional MusicXML score-event evaluation (V1)

**Issue:** #11901  
**Parent implementation:** MUSE-MIR-018 source-native MusicXML notation events  
**Date:** 13 November 2026  
**Status:** evaluator and focused fixtures authored; not compiled or executed

## Why this benchmark exists

MUSE-MIR-018 gives Muse a source-native per-part event surface for written score semantics. The next question is whether a predicted score preserves those semantics. A MIDI note F1 or a generic score similarity value cannot identify whether errors come from written pitch spelling, rests, tuplets, ties, voice/staff assignment, directions, attribute changes or measure traversal.

V1 compares two source-native MusicXML per-part notation graphs. It is model-agnostic: a candidate system can produce MusicXML, be parsed by the same source adapter, and be scored without running the vendor's model inside Muse.

## Design principles

1. **No weighted headline scalar.** Every metric remains an independently inspectable dimension.
2. **Presence is not correctness.** Event matching first establishes one-to-one event presence; pitch, duration, written value, tie and staff correctness are then evaluated separately.
3. **Unknown is not a zero.** Unsupported or unresolved measure/notation layers refuse scoring.
4. **Score clocks are explicit.** Inputs must share a part ID, score-clock identity and supported canonicalization version. No hidden transposition, tempo scaling, pickup shifting, dynamic time warping or barline alignment.
5. **Bounded and replayable.** Input graph hashes, metric configuration and all outputs are bound into a deterministic report commitment.
6. **Missing annotations stay visible.** Assignment accuracy uses only reference events that actually carry the relevant label; missing reference labels and unexpected prediction labels are separate counts.

## Input and refusal contract

Inputs:
- reference MusicXmlPartNotationGraphV1;
- prediction MusicXmlPartNotationGraphV1;
- explicit rational position and duration tolerances plus work budgets.

The evaluator also rejects a written pitch alteration that is not a valid finite decimal lexeme. It normalizes valid decimal strings as exact reduced rationals, so `0`, `0.0` and `0.000` are equivalent, while malformed values do not receive a lexical-equality pass.

The evaluator refuses when either side has:
- a non-Supported explicit measure map;
- a non-Supported notation-event layer;
- an unknown schema/canonicalization version;
- a mismatched part identity or score-clock identity;
- empty, overlapping, discontinuous or noncanonical measure spans;
- invalid event part/measure bindings, duplicate/non-contiguous source ordinals, out-of-measure positions or invalid cursor moves;
- an exceeded event, candidate-comparison or matching-work budget.

Measure spans must begin at score coordinate zero and be contiguous exact rational quarter-beat spans.

V1 defaults:
- note/auxiliary/attribute/cursor position tolerance: **1/16 quarter-note beat**;
- performed-duration tolerance: **1/16 quarter-note beat**;
- maximum structural units per part: 10,000, including measures;
- maximum candidate-pair comparisons per event family: 2,000,000;
- maximum augmenting-path edge visits per event family: 10,000,000.

These are protocol defaults, not universal musical tolerances. Reports retain the full configuration. Sensitivity panels should rerun at 1/32, 1/16, 1/8 and 1/4 quarter-note beats rather than tune a tolerance on the evaluation holdout.

## Matching

### Note/rest/unpitched event presence

A note event can match only when:
- its event family is the same (pitched, rest or unpitched);
- its exact rational score onset is within the configured position tolerance.

Pitch, duration, written value, voice/staff, tie and tuplet semantics are deliberately not candidate filters. They are measured after event-presence matching. Thus a wrong pitch at the correct onset remains a matched-but-incorrect pitch instead of disappearing as a note deletion plus insertion.

Each note event can match at most one event on the other side. V1 uses a deterministic, iterative augmenting-path algorithm that finds maximum-cardinality bipartite matching. Candidate adjacency follows source order; when several maximum-cardinality matchings exist, source order supplies the deterministic tie-break. V1 does not claim minimum-cost or minimum-onset-error assignment among all maximum matchings; the actual error distribution for the selected deterministic pairing is reported.

Auxiliary events match on element family and score position. Attribute events match on attribute element and position. Cursor operations match on direction and exact before/after positions within tolerance. Event payload correctness is reported separately from presence.

## Independent metrics

| Dimension | V1 result |
|---|---|
| Note/rest/unpitched presence | Precision, recall and F1 with TP/FP/FN |
| Matched note onset | Absolute error distribution: mean, median, p95, p99 and within-tolerance count |
| Written pitch spelling | Exact step + octave + numeric alteration equality; omitted alteration and explicit zero are equivalent, decimal lexical forms such as “0” and “0.000” are normalized exactly, enharmonic spellings remain distinct |
| Rest semantics | Measure-rest flag and display-pitch correctness |
| Unpitched display | Display-step/display-octave correctness |
| Duration presence | Whether both events carry a performed duration |
| Performed duration | Absolute rational quarter-beat error distribution and within-tolerance rate |
| Written value | Exact note type and dot-count accuracy |
| Voice/staff/instrument | Accuracy only where reference identity exists; missing-reference and unexpected-prediction counts separately |
| Sound ties | Exact sound-tie semantics |
| Visual ties | Exact visual tied notation semantics |
| Time modification | Exact actual/normal and normal-type/dot semantics |
| Tuplet notation | Exact bracket/number/placement/source-subtree semantics, independently from performed ratio |
| Full note source subtree | Exact structured source-tree match rate |
| Auxiliary events | Presence precision/recall/F1 and exact source-subtree accuracy |
| Attribute timeline | Presence precision/recall/F1 and exact source-subtree accuracy |
| Cursor operations | Presence precision/recall/F1 and exact movement semantics |

Accuracy dimensions are conditional on the corresponding matched event family. Note presence F1 exposes deleted/inserted events; a pitch accuracy of 1.0 over a tiny matched subset must not be read without the presence counts.

Percentiles use the nearest-rank rule ceil(p * n) - 1. If both sides have no observations for a metric, the value is unavailable; when exactly one side is nonempty and there are no true positives, F1 is zero.

## Report integrity

The report binds:
- reference and prediction graph SHA-256;
- part ID, score-clock ID and canonicalization version;
- evaluator algorithm version and complete tolerance/budget config;
- all event counts, independent metrics, onset/duration error distributions and assignment coverage counts.

Replay recomputes the graph digests, matching, every reported dimension and the domain-separated report commitment. A changed input, config, count or metric invalidates the replay result.

The hashes establish content consistency, not the identity of the parser, truth annotator or model producer.

## Tests authored

19 focused tests currently cover:
- identical graph yields perfect independent dimensions, including onset, duration, note, auxiliary, attribute and cursor outputs;
- wrong pitch spelling does not suppress note-event presence;
- equivalent decimal alteration lexemes compare numerically exactly, including implicit natural versus explicit zero;
- extra chord member is a false positive without consuming a second reference note;
- deleted event is a false negative;
- tie, tied, time-modification and tuplet semantics remain separate dimensions;
- absent reference voice/staff/instrument identity is reported separately from unexpected prediction labels;
- exact rational onset/duration tolerance and onset-error distributions;
- part/clock/canonicalization mismatch and Unsupported states refuse scoring;
- noncanonical/discontinuous measure map and malformed event order refuse scoring;
- report replay rejects changed graphs and tampered metrics;
- event, candidate-comparison and augmenting-path work budgets fail closed;
- augmenting-path matching reaches maximum cardinality where a greedy first pairing would fail.

Tests are authored, not compiled or run.

## External comparison and honest SOTA strategy

There is no direct numerical leaderboard comparison between this evaluator and AMT transcription systems yet. It is a score-semantics measurement layer, not a transcription model.

Relevant 2026 reference points:

- **MIREX 2026 Audio-to-Score:** Blind Hidalgo-Centeno reports MV2H 0.9492 on familiar Quartets but 0.6053 on composer-held-out QuartetsMIREX; Blind Cummins reports 0.9478 and 0.5981. The Staves-Informed Lu et al. track reports 0.8796 and 0.8179, but receives staff metadata and must remain a separate assisted track. These results show why composer-held-out generalization and assistance conditions must be published separately.
- **Klangio October 2026 benchmark:** on 130 one-minute clips over 13 sets, Klangio reports mean score F1 0.67, bar F1 0.72, time-signature accuracy 92%, and raw note F1 0.82 at 100 ms over nine aligned sets. Its bar-line tolerance is 0.175 of the median beat interval, while this evaluator uses score-clock quarter-beat tolerances; the numbers are not directly comparable. The evaluation includes internal/vendor-authored sets.
- **Dual Evaluation for Music Transcription (August 2026):** on 24 pipelines and 230 piano recordings, notation similarity and playback similarity favor different pipelines. The paper reports CLEWS as the cheapest playback metric with the best correlation to listener judgments among those evaluated.

Sources:
- https://futuremirex.com/mirex/wiki/2026:Audio-to-Score_Transcription_Results
- https://klang.io/blog/music-transcription-benchmark-2026/
- https://arxiv.org/abs/2608.04511

### Qualification targets (not achieved results)

1. Reproduce public MIREX Blind and Staves-Informed metrics separately on familiar and held-out composer splits. Do not compare these score-event metrics directly to MV2H.
2. On an independently annotated, rights-cleared, composer-separated MusicXML holdout, target at least 0.95 exact preservation accuracy for supported notation constructs, with every dimension published separately.
3. Achieve zero false-success for unsupported schema, missing score clocks, invalid event positions, malformed source ordinals and budget exhaustion.
4. Report note-presence F1, exact spelling, performed-duration errors, written-value correctness, ties, tuplets, staff/voice and auxiliary events independently; show coverage next to every conditional accuracy.
5. Add song/composer-clustered 95% confidence intervals and sensitivity panels over tolerance choices.
6. Evaluate expert correction time separately. A proposed product target is at least 30% lower median correction time than a matched transcription baseline, with human hints/staff metadata as separate protocol tracks.

The opportunity is not to maximize one score. It is to make score-transcription failures diagnosable and make the final notated artifact measurably more editable and reliable for musicians.

## Verification boundary

- Tests are authored but not compiled or executed.
- No local cargo, rustc or rustfmt toolchain is available in the authoring environment.
- No exact-head CI pass, vendor benchmark reproduction, annotated-holdout result, human correction-time study, MIREX MV2H parity or SOTA win is claimed.
- The focused workflow `.github/workflows/muse-musicxml-notation.yml` is already present on this PR's target base branch. It includes the evaluator module path and a dedicated test command; the first exact-head execution is still pending and must be observed rather than inferred.
