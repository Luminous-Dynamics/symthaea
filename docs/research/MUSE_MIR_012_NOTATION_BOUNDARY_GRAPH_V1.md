# MUSE-MIR-012 — Notation boundary graph and MusicXML semantics (V1)

**Parent programme:** #11677 MUSE-MIR-004 score-structure evaluation  
**Implementation issue:** #11727  
**Branch:** `research/melothaea-musicxml-boundary-graph-v1`  
**Date:** 12 November 2026  
**Status:** implementation and tests authored; not compiled or executed in the authoring environment

## Why this increment exists

A measure sequence is not the complete semantics of a score.

MusicXML 4.0 defines measure-level `non-controlling="yes"` for multimetric music: the measure's left barline does not coincide with left barlines in other parts. It also permits <barline> elements inside a measure, including middle-positioned subdivision bars in complex meters. Barline children may encode repeats, multiple endings, segno/coda signs, fermatas and wavy lines.

A single global barline list therefore loses polymetric structure. A simple measure-span list also loses internal notation/playback events. The system must preserve part-specific notation and the exact location of every supported event, while refusing to label a lossy projection as complete.

Official specifications:
- MusicXML 4.0 [partwise measure](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/measure-partwise/)
- MusicXML 4.0 [barline](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/barline/)
- MusicXML 4.0 [repeat](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/repeat/)
- MusicXML 4.0 [ending](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/ending/)

## Representation: three separate layers

### 1. Part-specific notated measure maps

`parse_musicxml_part_measure_maps` emits one exact measure sequence per MusicXML part.

Each map:
- retains the source part ID;
- shares the same caller-supplied score-clock identity as sibling parts;
- represents measure spans in reduced rational quarter-note beats;
- preserves pickup / incomplete / irregular / unclassified measure kinds;
- retains the zero-based indices of measures marked non-controlling;
- carries a canonicalization version and source-artifact identity.

The per-part map is the default representation for polymetric scores. The single-global-map API remains conservative: it returns Unknown for non-controlling measures, incompatible part extents or barline semantics it cannot represent. It never chooses one part's meter and projects it onto the others.

### 2. Typed notation-boundary graph

`parse_musicxml_part_notation_graph` emits, per part:
- the exact measure map;
- non-controlling measure indices;
- every explicit MusicXML barline event in source order;
- left / middle / right location;
- exact local measure offset and absolute score-clock offset;
- source event ID and measure locator;
- bar style and divisions override;
- repeat direction, repeat count and winged form;
- ending number, type and text;
- segno and coda marker presence plus optional identifiers;
- fermata type/text/content;
- wavy-line attributes;
- footnote/level annotations;
- unrecognized attributes and child elements, retained as explicit opaque annotations rather than silently discarded.

A middle barline is located at the exact current measure cursor. Left and right barlines use the explicit measure origin and end. Multiple events at the same rational coordinate are retained as distinct events; no deduplication is performed.

The map-only API withholds Supported status whenever the caller would otherwise lose internal bars, repeat/ending semantics, unknown extensions or other unmodelled barline data. The graph API preserves those events instead.

### 3. Playback traversal — deliberately separate

The notation graph does **not** expand repeats, multiple endings, D.S./D.C., coda jumps or volta navigation into duplicated measures.

The graph records the notated event and its source location. A future playback-traversal engine must derive a separate performed sequence, identify visited measure/ending instances, and retain a trace back to the notated source. This prevents a repeat from inflating notated measure count or barline F1.

## Exactness, commitments and replay

- Canonical positions are reduced rational quarter-note beats, not floating-point seconds or MIDI ticks.
- Part maps share one score-clock identity; part ownership remains a separate field.
- Cross-part boundary comparison is exact. It reports coincident boundaries, unmatched required boundaries and unmatched non-controlling boundaries separately.
- A graph commitment is SHA-256 over canonical serialized graph content, including measure spans, non-controlling flags and all barline events.
- Replay reparses the same supplied source and compares the complete graph; changed offsets, IDs, extension data or repeat counts fail validation.

A digest is a content commitment, not a digital signature, proof of source authenticity, or independent endorsement of the annotation. Source rights, provenance and authority must be checked separately.

## Input and unsupported cases

V1 supports MusicXML `score-partwise`. `score-timewise` is Unsupported. A part with unresolved divisions, cursor underflow, malformed duration, mid-measure time-signature change that cannot be normalized, or unresolved measure extent yields Unknown or an explicit error.

MIDI time-signature meta-events are not treated as barline sequences. MIDI needs an explicit provenance-bound barline sidecar or remains Unknown/Unsupported for this score-structure metric.

The existing note-centric `Score` type still does not store explicit measure boundaries. The adapter exposes a separate structure artifact so this information is not silently lost at import. Integrating that artifact into an end-to-end score IR is subsequent work.

## Tests authored

The adapter test module now includes fixtures for:
- constant 4/4 and 3/4;
- one-beat pickup;
- mixed 7/8 → 5/8 → 4/4;
- divisions changes;
- backup/forward multi-voice cursor handling;
- rest-only measures;
- unresolved and empty measures;
- cross-part measure disagreement;
- 3/4 versus 2/4 polymeter with non-controlling measures;
- exact cross-part alignment and score-clock mismatch;
- duplicate part IDs;
- malformed and mid-measure time-signature changes;
- internal middle barlines;
- forward/backward repeats and multiple endings;
- segno/coda, fermata and extension capture;
- multiple same-offset events without deduplication;
- map-only API abstention when richer graph data is required;
- canonical graph commitment and replay tampering.

The tests are authored, not executed. No local `cargo`, `rustc` or `rustfmt` toolchain is available in the authoring environment.

## Benchmark and industry comparison

These are distinct tasks and should remain distinct scorecard rows.

| Benchmark / reference | Published result | What it measures | Comparability boundary |
|---|---:|---|---|
| MIREX 2026 BeatThis, JSIR | Beat F1 94.70; CMLc 58.62; CMLt 83.50 | Beat/downbeat tracking | Does not measure notated barline reconstruction |
| MIREX 2026 BeatThis, GTZAN | Beat F1 89.02; CMLc 76.00 | Beat tracking on GTZAN | Dataset-specific; distinct from score structure |
| Klangio, Oct 2026 vendor benchmark | Score F1 0.67; bar F1 0.72; meter accuracy 0.92 | Commercial sheet-music transcription | Vendor-authored, 13 sets, its own bar tolerance; not a public independent leaderboard |
| A Dual Evaluation for Music Transcription (2026) | 24 pipelines; notation and playback objectives select different systems | Score notation similarity vs rendered performance similarity | Piano-focused study; renderer/converter is part of the evaluated pipeline |
| GlobalTap (Nov 2026) | 200 excerpts, 59 countries, 746 participants across 10 recruitment countries | Listener tapping and culturally diverse metrical interpretation | Multi-reference beat judgments, not score barline ground truth |

Sources:
- [MIREX 2026 Audio Beat Tracking results](https://music-ir.org/mirex/wiki/2026:Audio_Beat_Tracking_Results)
- [Klangio Music Transcription Benchmark 2026](https://klang.io/blog/music-transcription-benchmark-2026/)
- [A Dual Evaluation for Music Transcription](https://arxiv.org/abs/2608.04511)
- [AMT-Dual-Eval code and human-evaluation data](https://github.com/pingw220/AMT-Dual-Eval)
- [GlobalTap: Globally Diverse Beat Tracking Benchmark](https://arxiv.org/abs/2611.11137)

The Dual Evaluation study is especially important architecturally: symbolic notation similarity and playback similarity are complementary objectives, and the MIDI-to-score converter changes which pipeline is favored. Melothaea should report at least **notation faithfulness**, **rendered-performance faithfulness**, and **intent/protected-content preservation** separately. Neither a score similarity metric nor a listener preference score proves that an edit fulfilled the musician's intended request.

GlobalTap also argues against forcing one universal metrical interpretation where listeners disagree systematically. MUSE should preserve separate metrical hypotheses / annotations and report per-reference as well as aggregated scores.

## Qualification targets — not achieved results

1. **Synthetic contract fixtures:** 100% exact correctness across rational offsets, pickup, mixed meter, repeat/ending annotations and multiple part maps.
2. **False-success controls:** zero Supported results when a measure-only consumer would drop internal bars, repeat/ending semantics, unknown barline extensions or unresolved score-clock data.
3. **Replay integrity:** zero acceptance of altered graph content; 100% replay success for unchanged, pinned inputs.
4. **Polymeter:** 100% preservation of per-part maps and exact coincident/unmatched boundary classification on deterministic 2/4-versus-3/4 fixtures.
5. **Empirical barline reconstruction:** stretch target ≥0.80 F1 on a held-out, rights-cleared, musician-annotated score corpus at a preregistered tolerance of 1/8 quarter-note beat; also publish precision, recall, coverage, boundary displacement and pickup/partial-bar results.
6. **Industry comparison:** reproduce the exact public/vendor protocol before claiming numerical superiority. Klangio's 0.72 bar F1 is a reference point, not a directly comparable baseline until its exact dataset, alignment, tolerance and aggregation are reproduced.
7. **Human utility:** separately measure musician correction minutes, readability, edit success, protected-event violations and blinded preference. Cluster uncertainty by work/composition/artist, not by excerpt.

Do not collapse these dimensions into one weighted score. A system that abstains on ambiguous or unsupported score structure is preferable to a system that fabricates a confident barline map.

## Next implementation stages

- Cross-check the MusicXML adapter against a second independent parser and schema validation.
- Add timewise MusicXML support.
- Add a true notated-boundary graph IR shared by MusicXML, MIDI sidecars and score-native imports.
- Implement a separate repeat/ending playback traversal with exact visited-measure provenance.
- Add score-to-render conformance using pinned renderer/soundfont versions, then evaluate the rendered output independently from notation similarity.
- Run the barline evaluator against a held-out score corpus and publish per-part / per-meter results with clustered confidence intervals.
