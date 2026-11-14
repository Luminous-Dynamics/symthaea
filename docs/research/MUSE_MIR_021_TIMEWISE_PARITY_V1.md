# MUSE-MIR-021 — MusicXML score-timewise graph parity (V1)

**Issue:** #12078  
**Parent:** MUSE-MIR-018 source-native notation-event preservation  
**Implementation:** `musicxml_barline_adapter.rs`, `musicxml_notation_events.rs`  
**Date:** 14 November 2026  
**Status:** implementation and tests authored; not compiled or executed in this environment

## Why this is a real format gap

MusicXML 4.0 defines two document structures that contain the same musical score data but order the hierarchy differently:

| Layout | Structure |
|---|---|
| `score-partwise` | part → measure → music data |
| `score-timewise` | measure → part → music data |

The MusicXML tutorial explicitly explains that either layout can be primary and provides XSLT stylesheets to convert between them. A reader is expected to recognize both roots or convert one to the other.

The previous MUSE adapter accepted only `score-partwise`. A legal `score-timewise` source therefore failed before the notation-event layer, even though the inner part content uses the same MusicXML notes, attributes, cursor moves and barlines.

Official references:
- [MusicXML 4.0 score-timewise](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/score-timewise/)
- [MusicXML 4.0 timewise measure](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/measure-timewise/)
- [MusicXML 4.0 file structure tutorial](https://www.w3.org/2021/06/musicxml40/tutorial/structure-of-musicxml-files/)
- [MusicXML 4.0 score-timewise example](https://www.w3.org/2021/06/musicxml40/musicxml-reference/examples/score-timewise-element/)

## Implementation

### 1. Layout-neutral source measure inputs

Adds a small internal source-node pair:

- `container`: the original outer `<measure>` node;
- `content`: the original music-data container for that part.

For partwise input both fields point to the same `<measure>` node. For timewise input the container is the outer measure and content is the nested `<part>` node.

This is intentionally not an XML serialize/transform/reparse path. Both adapters pass the original `roxmltree` nodes to the same exact-duration, cursor, barline, playback, part-metadata and notation-event logic.

### 2. Canonical part ordering

- Partwise input follows the source `<part>` order, preserving the existing parser behavior.
- Timewise input follows the required `<part-list>/<score-part>` declaration order, regardless of the order in which nested `<part>` blocks appear inside a measure.
- Duplicate/empty score-part IDs are rejected.
- Every declared part must appear exactly once in each timewise measure in V1.
- Undeclared, missing or duplicate timewise part blocks are rejected.
- Multiple part lists and non-part children directly inside a timewise measure are rejected rather than ignored.

### 3. Measure-level versus part-level semantics

The input pair keeps these roles distinct:
- outer timewise measure attributes provide source measure number, ID, display text, pickup/implicit and non-controlling flags;
- nested part content provides that part's own divisions timeline, cursor movements, notes, rests, directions, barlines and playback controls.

Each part receives its own exact measure map. A timewise root does not imply one global barline lattice. Non-controlling measures and polymetric parts remain separate.

### 4. Shared consumers

The same paired source nodes feed:
- exact measure-span calculation;
- source measure locators;
- typed barline event parsing;
- playback-control capture;
- notation-event parsing for notes, rests, unpitched events, ties, tuplets, voice/staff, written duration and attribute timelines;
- part-list metadata and instrument declarations.

Playback-control source paths are normalized relative to the part-content node, so the same event has the same semantic path in both layouts.

The legacy `parse_musicxml_measure_map`, the per-part `parse_musicxml_part_measure_maps`, and the full `parse_musicxml_part_notation_graph` now use the same source layout normalizer.

### 5. Canonicalization and historical receipts

The shared measure-map canonicalization string is now:

`musicxml-partwise-timewise-explicit-measures-quarter-beats-v2`

The notation-event canonicalization is:

`musicxml-partwise-timewise-source-ordered-exact-quarter-beats-v2`

New partwise and timewise graphs therefore advertise the same layout-neutral contract. Existing stored partwise receipts with the previous v1 strings remain replayable: the replay validator recognizes the historical strings and applies them only when validating the old stored graph. It does not silently downgrade a new v2 graph.

Measure XML IDs are layout-specific: partwise IDs live on per-part measure nodes; timewise IDs live on the shared outer measure. The graph preserves those original source locators. The parity test normalizes only this field when comparing the otherwise identical source score semantics.

## Tests authored

The adapter test module now contains 45 tests (previously 37 before this increment), including:
1. Full partwise/timewise graph parity across rational spans, barline events, playback controls, notation events and part metadata.
2. Reversed timewise nested part order with canonical output order from the part list.
3. Timewise pickup and non-controlling measure locator preservation.
4. Per-part independent measure spans in a polymetric timewise score.
5. Global common-lattice measure map support when part spans agree.
6. Required part-list and duplicate-part-list rejection.
7. Missing, duplicated and undeclared timewise part block rejection.
8. Unexpected direct children under a timewise measure rejection.
9. Tampered measure locator replay rejection.
10. Historical v1 partwise map/event canonicalization replay compatibility.
11. Default-namespace MusicXML timewise parse, playback-path normalization and replay.

The existing notation-event parser suite remains at 19 authored tests and is run by the focused MusicXML workflow.

## Benchmark and competitive implications

This does not improve transcription accuracy by itself. It removes a format-coverage blind spot that would otherwise exclude a valid score layout from training, editing or evaluation.

The scorecard remains multidimensional:
- notation-event presence and written pitch;
- sounding duration versus written type/dots and tuplet ratios;
- voice/staff/part identity;
- tie and tuplet notation;
- exact measure/barline structure;
- rendered playback similarity;
- uncertainty and parser coverage.

Do not claim a score increase because the adapter now parses an additional format. The appropriate future benchmark is a paired-layout parity cohort: convert rights-cleared reference scores between partwise/timewise using the official transform and compare normalized per-part graphs, then evaluate transcriptions against the same score semantics regardless of source layout.

## Verification boundary

- Tests are authored but have **not** been compiled or executed.
- No local `cargo`, `rustc` or `rustfmt` executable is available in the authoring environment.
- The existing focused workflow `.github/workflows/muse-musicxml-notation.yml` covers the touched adapter, event parser, Studio summary tests and Add Music wasm check. It is triggered by relevant PR paths, but no exact-head result for this branch is claimed.
- No parser parity against an independent MusicXML library, external corpus accuracy, SOTA or industry-win claim is made.
- V1 requires every declared part exactly once in every timewise measure. Omitted/empty part blocks remain a deliberate fail-closed limitation until a schema- and corpus-qualified absent-part semantic is designed.
