# MUSE-MIR-014 — Bounded repeat-aware playback traversal (V1)

**Parent:** #11727 MUSE-MIR-012 notation-boundary graph  
**Integration:** #11779 MUSE-MIR-013 structured symbolic import  
**Issue:** #11782  
**Date:** 12 November 2026  
**Status:** Rust implementation and tests authored; not compiled or executed in this environment

## Paradigm boundary: notation is not playback

A notated score is a stable document. A performance is an ordered traversal through that document. Repeats and endings change the order in which a performer visits measures; they do not create new notated measures.

Duplicating measures in the source score to represent a repeat would corrupt:
- barline and measure-count evaluation;
- stable source measure identity;
- score-edit diffs and protected-content invariants;
- provenance from performance instances back to the authored notation.

MUSE-MIR-012 preserves repeat/ending markers in an immutable per-part notation graph. MUSE-MIR-014 derives a separate playback-visit sequence from that graph.

## V1 supported semantics

- Forward repeat at a left or right measure boundary.
- Backward repeat at a left or right measure boundary.
- Backward repeat with an explicit positive integer `times`.
- Missing backward `times` uses the MusicXML default of two total passes.
- Backward repeat without a preceding forward marker uses the start of the notated part as the implicit repeat start.
- Multiple sequential, non-overlapping repeat regions.
- The MusicXML `after-jump` attribute is preserved in the source graph. V1 accepts absent/`no`, but returns Unsupported for `yes` because D.C./D.S. jump semantics are not implemented.
- Every source measure's number (including non-numeric labels), document ID, display text, pickup/implicit flag and non-controlling flag is retained in a dedicated measure locator. Playback visits bind directly to this locator rather than trying to recover the source measure number from whichever barline event happens to exist.

V1 expands each part independently. MusicXML's non-controlling per-part measure lattices remain valid and do not imply synchronized barlines between parts.

## V1 explicitly Unsupported

The current traversal returns Unsupported rather than guessing when it encounters:
- nested or overlapping repeat regions;
- numbered endings / voltas;
- segno or coda jumps;
- D.C. / D.S. or other navigation semantics;
- `after-jump="yes"` repeat semantics;
- MusicXML `<sound>` navigation attributes: `dacapo`, `dalsegno`, `segno`, `coda`, `tocoda`, `forward-repeat`, `fine`;
- selective `time-only` controls on sounds, notes, lyrics, ties and other playback-related elements;
- unrecognized repeat attributes;
- repeat markers on middle barlines;
- unknown repeat directions;
- unrecognized barline attributes or child elements that may affect playback;
- an unmatched forward-repeat marker.

Repeat count zero, malformed count strings, and count values above the configured bound are errors. Event and expansion budgets are checked before visit allocation.

## Output contract

The traversal report includes:
- schema and algorithm version;
- canonical digest of the complete source notation graph;
- part identity;
- exact traversal configuration and bounds;
- ordered repeat regions with start/end measure indices and total passes;
- expanded playback visits with:
  - zero-based playback ordinal;
  - part ID;
  - source notated measure index;
  - source MusicXML measure number when available;
  - repeat region ID and one-based repeat pass when applicable;
- canonical report digest.

A visit is a reference to a notated measure, not a copied measure. The graph and its measure count remain unchanged.

The report digest is a content commitment, not a signature or proof of source authenticity. Replay recomputes the entire traversal from the graph/config and rejects altered input or output.

## Example

For a four-measure part with a forward repeat at the left boundary of measure 1 and a backward repeat at the right boundary of measure 2 (default two passes), the notated sequence is:

`[0, 1, 2, 3]`

The playback visits are:

`[0, 1, 0, 1, 2, 3]`

The notation graph still contains exactly four measures. No barline or note-evaluation denominator is inflated by the playback expansion.

## Authored fixtures

The new Rust module `crates/domains/symthaea-muse/src/repeat_traversal.rs` includes 13 focused tests:
1. no repeats: each notated measure visited once;
2. default forward/backward repeat: two total passes;
3. explicit repeat count three;
4. backward repeat with implicit score-start;
5. two sequential repeat regions;
6. nested repeats -> Unsupported;
7. ending, segno and coda navigation -> Unsupported;
8. unknown barline extension -> Unsupported;
9. zero/malformed repeat counts -> error;
10. repeat-pass and total-visit budget exhaustion;
11. source graph/report digest replay and tamper rejection;
12. after-jump and MusicXML sound-navigation controls -> Unsupported;
13. repeated visits preserve non-numeric source measure labels.

The parent MusicXML adapter now has 29 focused tests, including:
- after-jump and unknown repeat attribute preservation;
- preservation of `<sound>` attributes and selective `time-only` controls;
- source measure numbers, IDs, display text, pickup flags and non-controlling flags.

The test fixtures exercise MusicXML parsing into the actual notation-graph API, not a mock graph.

## Qualification boundaries

- Tests are authored, not compiled/executed.
- No local Rust toolchain is available in the authoring environment.
- No exact-head CI pass, DAW playback parity, or performance-timing claim is made.
- MusicXML semantics are based on the official 4.0 `<repeat>` contract: `times` is the number of times the repeated section is played; `after-jump` changes repeat behavior after a D.C./D.S. jump. The adapter now preserves both fields and V1 refuses the unsupported post-jump case. See the [MusicXML 4.0 repeat reference](https://www.w3.org/2021/06/musicxml40/musicxml-reference/elements/repeat/) and [schema annotation](https://www.w3.org/2021/06/musicxml40/listings/musicxml.xsd/).
- V1 does not implement endings or navigation jumps.
- V1 preserves playback-affecting `<sound>` and `time-only` controls, but refuses traversal when those controls would alter the visit sequence or note/lyric eligibility.
- V1 does not schedule concurrent parts against a conductor timeline.
- A separate renderer/performance layer must still establish tempo-map, fermata, rubato, swing and audio-clock behavior.

## Next stages

1. Implement numbered endings/voltas and nested repeats with a bounded state machine and independent notation fixtures.
2. Add D.C./D.S./segno/coda traversal with loop protection and explicit visit traces.
3. Derive a multi-part conductor timeline without forcing polymetric parts onto one barline sequence.
4. Compare expansion against independent notation/DAW playback oracles.
5. Add score-to-render conformance as a separate evaluation axis from notation similarity and musician intent.
