# MUSE-MIR-018 — MusicXML notation-event preservation (V1)

**Issue:** #11837  
**Parent:** MUSE-MIR-004 score-structure evaluation  
**Integrates with:** MUSE-MIR-012 MusicXML measure/barline graph  
**Stack:** MUSE-MIR-017 source MIDI audition provenance  
**Date:** 13 November 2026  
**Status:** implementation and tests authored; not compiled or executed

## Motivation: preserve the score, not just the notes

MUSE has three related but distinct representations:

1. The legacy `Score`: convenient pitched notes for playback, composition and existing tooling.
2. The MUSE-MIR-012 graph: explicit per-part measure spans, source measure locators, barline/repeat/endings and playback controls.
3. The MUSE-MIR-018 notation-event layer: written and performed-symbolic note/rest semantics, plus score-order attributes, kept beside the graph.

The first two are insufficient for readable-score reconstruction and editing. Flattening MusicXML into a list of pitched notes loses rests, grace/cue notes, unpitched percussion, written values, ties, tuplets, staff/voice assignments, chord groupings and score-order attributes.

## MusicXML semantics that must not be conflated

MusicXML explicitly distinguishes:
- `<tie>` sound semantics from `<tied>` visual notation;
- `<time-modification>` performed-duration ratios from `<tuplet>` displayed bracket/number structure;
- written note type/dots from divisions-based performed duration;
- grace notes from ordinary-duration notes;
- chord members from sequential notes that advance the measure cursor.

References:
- [MusicXML 4.0 note](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/note/)
- [MusicXML 4.0 time-modification](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/time-modification/)
- [MusicXML 4.0 duration](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/duration/)
- [MusicXML notation basics](https://www.w3.org/2021/06/musicxml40/tutorial/notation-basics/)
- [MusicXML voice example](https://www.w3.org/2021/06/musicxml40/musicxml-reference/examples/voice-element/)

## Implementation

Adds `crates/domains/symthaea-muse/src/musicxml_notation_events.rs`, exported behind the existing `theory` feature and integrated into `MusicXmlPartNotationGraphV1`.

### Note/rest/unpitched events

Each source event records:
- monotonically increasing source-event ordinal and measure index/source number;
- exact rational quarter-beat score position;
- exact rational performed duration (or none for a grace note);
- exactly one event kind: written pitch spelling (step, exact source alter string, octave), rest (including measure-rest flag/display pitch), or unpitched event (display pitch);
- grace attributes (slash, steal time, make time), cue and chord-member flags;
- chord-member anchor ordinal, voice, staff and instrument IDs;
- written type/dot count and exact time-modification actual/normal ratio;
- sound tie records and visual tie records as separate fields;
- tuplet notation source subtrees, beam/stem/notehead/accidental records;
- full note subtree as a typed source tree, including articulation/ornament and extension annotations.

Chord members share their immediately preceding base note's onset and do not advance the cursor. Grace chords are supported without fabricating a non-zero duration. A chord member longer than its base note, or without a resolvable base, makes the event layer Unknown.

### Score-order attributes and cursor movement

Each child of MusicXML `<attributes>` becomes an ordered attribute event at its exact cursor position. This preserves divisions, key, meter, clefs, staves, transpose, staff details and future child elements in source order.

`<backup>` and `<forward>` become explicit cursor-movement events. A forward event retains voice/staff and its source subtree, not just the final cursor position.

Opaque source subtrees preserve ordered element children, attributes and ordered mixed text/element content. This is a structural source representation, not a byte-for-byte lexical XML archive: namespace prefixes, comments and insignificant lexical formatting are not promised as preserved. The original uploaded XML remains the authoritative byte artifact.

### Fail-closed and compatibility behavior

- The event layer emits all events or `Unknown`; it never leaks a partial event list.
- Invalid divisions/durations, cursor underflow/overflow, unresolved chord anchors and event/source-tree budget exhaustion produce Unknown.
- A measure map can remain Supported while its notation-event layer is Unknown; consumers must read both status values independently.
- Existing Score importer signatures and note-only MIDI metrics are unchanged.
- New graph commitments include notation events.
- Old graph JSON without the field deserializes to a legacy Unknown sentinel; serialization skips the sentinel, keeping old graph commitment shape stable.
- Replay validation projects newly parsed graphs back to the legacy shape when validating an old receipt. New receipts compare the full event layer, so tampering is rejected.

Resource budgets:
- 200,000 notation events per part;
- 32 nested source-tree levels;
- 500,000 retained source-tree nodes per part.

## Tests authored

### Notation-event parser (12 tests)

- rest and measure-rest, grace, cue, unpitched, microtonal pitch spelling, voice/staff and instrument ID;
- chord member onset anchoring and no cursor advance;
- triplet sounding duration alongside written eighth-note type, dot and tuplet display structure;
- sound tie versus visual tied notation, with slur/articulation retained in the source tree;
- score-order key/clef attribute changes at exact beat offsets;
- backup/forward movement across simultaneous voices;
- source order and unknown extension retention;
- event extent beyond the explicit measure span withholds all partial events;
- ambiguous chord / invalid duration fail-closed behavior;
- grace-chord grouping;
- ordered mixed text/element content;
- forward movement voice/staff identity.

### Integrated graph (32 adapter tests total)

New integration tests verify:
- notation events are part of the per-part graph and graph commitment;
- deterministic replay accepts the same source and rejects a tampered event position;
- legacy graph serialization omits the sentinel and legacy replay remains valid;
- unsupported mid-measure meter change does not leak partial notation events.

These are authored tests. They have not been compiled or executed in this environment.

## How this advances the benchmark programme

This creates the missing reference surface for score-level evaluation, complementing:
- MUSE-MIR-001: MIDI note/event metrics (pitch/onset/duration/program);
- MUSE-MIR-009: explicit measure/barline structure;
- MUSE-MIR-012: MusicXML measure/barline/repeat graph;
- MUSE-MIR-015: dual notation/playback benchmark protocol.

It enables future, separate score metrics for:
- pitch spelling exactness, note/rest/unpitched event coverage;
- written duration vs. performed duration;
- tie-chain sound continuity vs. visual tie notation;
- time-modification ratio, tuplet nesting and bracket accuracy;
- voice/staff/instrument identity;
- clef/key/meter/transposition timeline edit distance;
- chord grouping and grace-note preservation;
- source coverage and Unknown/abstention rates.

No overall weighted score is introduced. No MIREX MV2H, MUSTER, OMR-NED or CLEWS equivalence is claimed.

## Qualification targets (not achieved results)

1. 100% exact rational position agreement on deterministic MusicXML fixtures.
2. Zero accepted malformed-duration, cursor-underflow, invalid-division and ambiguous-chord controls.
3. 100% preservation of known note/rest/unpitched, written type/dots, time-modification, tie/tied and tuplet semantics in the fixture corpus.
4. Zero partial event lists when the event layer cannot establish source positions.
5. 100% replay rejection after mutation of event position, note kind, voice/staff, tie or tuplet semantics.
6. Legacy graph receipt replay remains valid without re-writing old stored artifacts.
7. On a rights-cleared, composer-separated score corpus: publish each dimension with song/composer-clustered 95% CIs, exact tolerances and parser coverage. Benchmark target setting follows baseline reproduction, not guessed headline scores.

## Verification boundary

- Tests are authored, not compiled or executed.
- No local `cargo`, `rustc` or `rustfmt` toolchain is available in the authoring environment.
- No exact-head CI pass, external corpus score, parser parity result or SOTA claim is asserted.
- The retained source tree is a structured semantic representation, not a replacement for storing the original MusicXML bytes.
