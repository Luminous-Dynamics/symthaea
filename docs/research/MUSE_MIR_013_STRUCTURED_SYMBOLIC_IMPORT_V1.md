# MUSE-MIR-013 — Structured symbolic import with notation evidence (V1)

**Parent:** #11677 MUSE-MIR-004 score-structure programme  
**Issue:** #11779  
**Branch:** `research/melothaea-structured-symbolic-import-v1`  
**Date:** 12 November 2026  
**Status:** implementation and tests authored; not compiled or executed in this authoring environment

## Problem

Muse Studio's existing symbolic import path returns a note-centric `symthaea_music_theory::Score`. That representation carries notes, parts, tempo and a legacy meter scalar, but it does not preserve the original MusicXML measure sequence, per-part polymeter, or explicit barline/repeat/ending semantics.

The new MUSE-MIR-009/010/011/012 adapters can now represent those semantics. MUSE-MIR-013 wires them into the actual symbolic import route without changing the legacy parser's return type.

A separate integration defect was also present: `symbolic_import.rs` and `teaching_corpus.rs` existed and Muse Studio called them, but neither module was exported from `symthaea-muse/src/lib.rs`. The repository's 2026-07-30 CI failure catalogue already identified these missing exports as a likely Muse tests/studio build blocker.

## Backward-compatible import API

The existing functions remain unchanged:

- `parse_symbolic(bytes, format) -> Result<Score, String>`
- `parse_musicxml(bytes) -> Result<Score, String>`
- `parse_midi(bytes) -> Result<Score, String>`

V1 adds:

```rust
parse_symbolic_with_structure(
    bytes,
    format,
    score_clock_id,
) -> Result<StructuredSymbolicImportV1, String>
```

The result contains:
- the original `Score`, unchanged;
- the source format;
- a SHA-256 digest computed internally from the exact input byte slice;
- a structure result that is one of:
  - `MusicXmlPartGraphs(Vec<MusicXmlPartNotationGraphV1>)`;
  - `Unknown { reason }`;
  - `Unsupported { reason }`.

The source digest is not caller-supplied. For MusicXML, the adapter's source ID is `sha256:<digest>`, and the score-clock identity is derived by the Studio route from that exact digest. Each part retains its own identity while sharing one score coordinate frame.

## Format-specific behavior

### MusicXML

The old note-centric parser still creates the `Score`. In parallel, the new path calls `parse_musicxml_part_notation_graph` to preserve:
- explicit per-part measure spans and pickup/partial-measure classes;
- polymetric and non-controlling measure indices;
- exact rational score-beat coordinates;
- left/middle/right barline events and source order;
- repeats, endings, segno/coda, fermatas and wavy-line markers;
- opaque unknown barline extensions;
- source IDs, part IDs, score-clock identity and canonicalization version.

Per-part Unknown map statuses are retained in the graph. A graph adapter error after a successful legacy Score parse becomes an explicit Unknown structure result; it does not discard the Score. An empty score-clock identifier is still rejected as an invalid call contract.

### MIDI

The current MIDI importer returns a Score, but a time-signature meta-event list does not establish a complete notated measure-boundary sequence or pickup semantics. V1 therefore reports Unknown until an explicit, provenance-bound barline sidecar is supplied.

### MuseScore JSON

The current JSON route deserializes the note-centric Score schema. It does not yet preserve a separate measure-boundary graph, so the structured import result reports Unknown rather than deriving bars from `Score::meter`.

## Muse Studio persistence and API

The `POST /api/music/import` route now calls `parse_symbolic_with_structure`, rather than only calling the legacy parser.

Every new imported-work summary contains an optional, backwards-compatible `notation_structure` summary:
- schema/version;
- status: `supported`, `partially_unknown`, `unknown`, or `unsupported`;
- exact source SHA-256;
- canonical graph SHA-256 where a graph exists;
- part/support/unknown counts;
- measure, barline-event and non-controlling-measure counts;
- unavailable part IDs and an explanatory note.

For MusicXML, the complete graph is persisted beside the source, score, summary and audition as `notation-graph.json` within the import's atomic staging transaction.

New endpoint:

`GET /api/music/import/{work_id}/notation-graph`

It:
1. reads the import summary (which also validates the content-derived work ID);
2. obtains the expected graph digest from `notation_structure`;
3. reads and deserializes the stored graph;
4. recomputes the canonical graph digest;
5. returns the graph only when the digest matches.

Older imported-work summaries deserialize with `notation_structure: None`; their legacy source, score and audition behavior is unchanged. The Muse UI now displays structural status/counts and links to the graph endpoint when a graph was retained.

The digest establishes content consistency against the stored receipt, not source authenticity or rights authorization. The import authorization, content license, private-project scope and existing no-global-learning boundaries remain separate and unchanged.

## Tests authored

### Symbolic importer

Seven focused tests now cover:
- the legacy MusicXML note import;
- returning Score plus exact per-part notation graph;
- source SHA-256 equality to the supplied byte slice;
- repeat/ending semantics surviving structured import;
- MuseScore JSON preserving Score while structure remains Unknown;
- MIDI preserving Score while structure remains Unknown without a barline sidecar;
- duplicate part IDs producing Unknown structure rather than discarding a valid legacy Score;
- an empty MusicXML score-clock ID rejected.

### Muse Studio summary helper

Two focused tests cover:
- supported MusicXML graph summary counts and persisted sidecar digest round-trip;
- Unknown structure summary with explanation and no graph sidecar.

The UI/API additions are authored but have not been compiled in this environment.

## Qualification boundaries

- Tests are authored, not executed.
- No local `cargo`, `rustc` or `rustfmt` exists in the authoring environment.
- No exact-head CI result or Muse Studio build pass is claimed.
- No empirical barline score, vendor parity or SOTA claim is made by this integration.
- MIDI sidecar parsing, MuseScore structured-measure schema support, MusicXML timewise support and repeat playback traversal remain separate follow-up work.

## Next stages

1. Verify `cargo fmt --check -p symthaea-muse`, focused importer tests and Studio compile/test on the exact head.
2. Add a provenance-bound MIDI barline sidecar adapter and a MuseScore schema extension.
3. Add a separate repeat/ending playback traversal that produces a performed sequence with traceable notated-measure provenance.
4. Add MusicXML schema validation and an independent parser parity harness.
5. Evaluate score notation fidelity, rendered-performance fidelity, and musician intent/preservation as separate benchmark axes.
