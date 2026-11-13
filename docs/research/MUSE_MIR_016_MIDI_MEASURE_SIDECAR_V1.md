# MUSE-MIR-016 — Explicit MIDI measure sidecar and PPQ-invariant maps (V1)

**Issue:** #11798  
**Parent:** MUSE-MIR-013 structured symbolic import  
**Implementation:** `crates/domains/symthaea-muse/src/midi_measure_sidecar.rs`  
**Date:** 12 November 2026  
**Status:** adapter, structured import and Studio wiring authored; not compiled or executed in this authoring environment

## Why a sidecar is necessary

Standard MIDI Files include a Time Signature meta-event (FF 58), Marker (FF 06), Cue Point (FF 07), track labels and tempo metadata. The time-signature event encodes a numerator/denominator and metronome-click parameters; it does not provide a complete typed sequence of per-part measure spans, pickup boundaries, irregular measures or polymetric lattices.

Therefore MUSE must not derive barlines from:
- the last-seen FF 58 time signature;
- a fixed `Score::meter` scalar;
- note density or the first note onset;
- marker/cue text interpreted heuristically as a measure boundary.

References:
- MIDI Association, [Standard MIDI Files](https://midi.org/standard-midi-files)
- MIDI Field Guide, [Standard MIDI File chunks, timing and meta-events](https://midi.community/docs/smf/)
- Somascape, [SMF format and Format 1 tempo/time-signature conventions](https://somascape.org/midi/tech/mfile.html)

The standard meta-event catalog identifies FF 58 as Time Signature and FF 06/07 as Marker/Cue Point. These are not a typed measure-span data structure. This sidecar supplies that missing annotation explicitly.

## Sidecar schema

Schema identifier: `muse-midi-measure-sidecar-v1`

The JSON envelope contains:
- exact source `midi_sha256`;
- `ticks_per_quarter` required to match the MIDI header's Metrical timing division;
- caller-bound `score_clock_id`;
- one or more part records;
- each part's unique `part_id`, source MIDI `source_track_index`, optional source `channel` (0–15), and ordered measure spans;
- `source_channel: null` means the entire track is intentionally one composite part; a channel number maps only that MIDI channel. Multiple sidecar parts may share a track when they use distinct explicit channels.
- every measure's `start_tick`, exclusive `end_tick`, `MeasureKind`, optional source number/ID/display text, `implicit` flag and `non_controlling` flag.

V1 maps one sidecar part to one source MIDI track. Every note-bearing source track must have an explicit map; empty conductor/metadata tracks may be omitted. MIDI format-0 files may map the combined track as one composite part, or split it into multiple channel-specific part maps. This preserves distinct channels when the source file combines instruments in one track. Format-1 tracks share the file's score-clock identity and may have independent measure lattices. Format-2 files contain independent sequential patterns and are rejected by V1 rather than incorrectly assigning them one shared score clock.

## Exact normalization

For source PPQ (P) ticks per quarter note and tick coordinate (t), the canonical position is:

[
  q = rac{t}{P}
]

The implementation reduces that fraction with the existing `RationalBeat` type. No floating-point conversion or audio-time estimate is used.

For example, a 4/4 measure at 96 PPQ ends at tick 384; the same measure at 480, 960 or 1920 PPQ ends at 1920, 3840 or 7680 ticks respectively. Every one normalizes to exactly 4 quarter-note beats.

Sidecar validation checks:
- MIDI parses and has Metrical PPQ timing; SMPTE-time division is rejected;
- exact source SHA-256 equality;
- exact PPQ equality with the header;
- a non-empty, matching score-clock identity;
- supported schema version;
- one or more parts, unique part IDs and unique source track indices;
- source track indices in range;
- every note-bearing channel represented exactly once by a channel-specific map or a whole-track composite map (empty conductor/metadata tracks may be omitted);
- non-empty, ordered measure lists;
- first measure starts at tick zero;
- all spans have positive length and are exactly contiguous;
- ticks fit the checked rational arithmetic domain;
- no gaps, overlaps, reversed measures or zero-length measures.

The resulting per-part `MeasureMap` has canonicalization version `midi-explicit-ppq-ticks-to-rational-quarter-beats-v1`. Source identity includes track and channel scope. Original tick coordinates and source measure locators remain available beside the normalized spans.

## Provenance and replay

The adapter records:
- SHA-256 of the exact MIDI input;
- SHA-256 of the exact sidecar JSON bytes;
- shared score-clock identity;
- PPQ;
- per-track maps, measure locators and non-controlling indices.

A canonical receipt commitment binds the normalized sidecar maps. Replay reparses the exact MIDI and sidecar byte slices and compares the complete receipt. Changed MIDI bytes, PPQ, measure spans, source locators or classification flags fail replay.

These hashes establish content identity and replay consistency. They do not authenticate the annotator, prove ownership/rights or establish the truth of manually supplied boundaries. Dataset authority and rights receipts remain separate.

## Structured symbolic import

Existing `parse_symbolic` and `parse_symbolic_with_structure` signatures remain unchanged.

The new `parse_symbolic_with_midi_measure_sidecar` entry point:
1. parses MIDI through the existing note-centric Score importer;
2. computes source SHA-256 from the exact bytes;
3. validates the explicit sidecar against the exact MIDI bytes and caller score-clock ID;
4. returns the existing Score plus `MidiExplicitMeasureSidecar(receipt)`.

MIDI imports without a sidecar continue to return Unknown structure. No FF 58 or legacy meter inference is introduced.

## Muse Studio integration

The private `POST /api/music/import` endpoint accepts an optional multipart field named `measure_sidecar`. It is valid only when the uploaded source is MIDI.

For a valid sidecar import, Studio:
- includes the exact sidecar SHA-256 in the content-derived import identity, so changing annotations cannot reuse a stale import record for the same MIDI;
- persists the raw `midi-measure-sidecar.json`;
- persists normalized `midi-measure-maps.json`;
- exposes a backward-compatible `measure_map_sha256` field in `notation_structure`;
- adds `GET /api/music/import/{work_id}/midi-measure-maps`;
- recomputes the canonical receipt commitment and replays the exact stored MIDI plus raw sidecar bytes before returning the maps. The endpoint rejects a modified source file, raw sidecar or normalized receipt.

The Add Music page now has an optional MIDI measure-sidecar picker and a link to inspect the verified per-track measure maps. MusicXML continues to use the separate notation-graph endpoint.

## Tests authored

### Adapter (15 tests)

- PPQ 96/480/960/1920 produce identical rational quarter-beat measure spans;
- source digest mismatch;
- PPQ mismatch;
- SMPTE time-code refusal;
- nonexistent source track, duplicate part/track and omitted note-bearing track rejection;
- empty score-clock identity;
- gap, overlap and zero-length span controls;
- two polymetric track lattices on a shared score clock;
- pickup, irregular, implicit, non-controlling and source-locator preservation;
- replay and commitment tamper rejection.

### Structured importer (2 new tests)

- valid MIDI + explicit sidecar returns Score and exact maps;
- wrong MIDI digest and wrong score-clock sidecars are rejected.

### Muse Studio helper (1 new test)

- normalized MIDI receipt summary, JSON persistence and canonical digest round-trip.

The Studio getter additionally replays the persisted MIDI/sidecar inputs against the stored normalized receipt before serving maps. That filesystem endpoint path is authored but has not been exercised by a live server test in this environment.

Existing MusicXML note-only import, structured graph import, MuseScore/MIDI Unknown behavior, and private import tests remain in the branch. Tests are authored, not executed.

## Benchmark implications and targets

This adapter enables MUSE to evaluate MIDI-derived explicit measure annotations using the same rational measure-map and barline-evaluation infrastructure as MusicXML. It does not improve MIDI transcription accuracy by itself.

Qualification targets:
1. 100% PPQ-invariance across 96/480/960/1920 synthetic equivalents.
2. Zero accepted source-digest or PPQ mismatch controls.
3. 100% exact span and part-lattice preservation on sidecar fixtures.
4. Zero false-success when no sidecar is supplied.
5. Compare MusicXML and MIDI sidecar adapters against shared score-clock fixtures with identical measure maps and evaluator protocol.
6. Do not claim SOTA or real-world barline accuracy until a rights-cleared annotated MIDI corpus is collected and an independently verified sidecar authoring protocol exists.

## Qualification boundary

- Tests are authored but have **not** been compiled or executed.
- No local `cargo`, `rustc` or `rustfmt` toolchain is available.
- No exact-head CI pass, live Studio upload, external parser parity or benchmark score is claimed.
- The sidecar is explicit human/producer-supplied annotation, not an automatically inferred boundary detector.
- V1 does not encode MIDI repeats, D.C./D.S. playback navigation or multi-track note-channel routing beyond one part per track.
