# MUSE-MIR-020 — Localized MusicXML score correction ledger (V1)

**Issue:** #12030  
**Parent:** MUSE-MIR-018 source-native notation graph and MUSE-MIR-019 score-event evaluator  
**Implementation:** `musicxml_correction_ledger.rs`  
**Date:** 14 November 2026  
**Status:** implementation and focused tests authored; not compiled or executed

## Motivation

MUSE-MIR-019 publishes independent score-event metrics. Those aggregates tell us that a prediction differs, but not the exact edits a musician or reviewer should make.

The 2026 ISMIR work *Measure for Measure* introduces Musical Correction Rate (MCR), which parses **kern, ABC and MusicXML, aligns at measure level, and localizes errors to individual notes. Its human study reports closer association with estimated correction effort than existing metrics. The paper's open review also calls out cross-format conversion bias, unsupported notation, pitch/key assumptions, and the danger of making broad SOTA claims using only a newly proposed metric.

MUSE does not claim MCR parity. V1 adds a transparent correction ledger from the exact MIR-019 matching result. It is an explanation layer—not a new score and not an estimate of minutes saved.

References:
- https://ismir2026program.ismir.net/poster_520.html
- https://amtevaluation.github.io/
- https://arxiv.org/abs/2608.04511

## Design

The public MIR-019 metrics remain unchanged. A new internal evaluation API returns both:
- the existing multidimensional report;
- the exact note, auxiliary, attribute and cursor event match pairs used to produce that report.

`build_musicxml_correction_ledger` consumes those pairs. It does **not** invoke a second aligner.

The output ledger is deterministic, source-localized and bounded:
- insert in prediction;
- delete from prediction;
- replace one field.

Each action includes:
- deterministic SHA-256 action ID;
- part and event family;
- operation and field path;
- reference/prediction measure index and source measure number;
- reference/prediction source-order ordinal;
- exact rational quarter-beat location on each side, where applicable;
- SHA-256 of the before and after values;
- a stable explanation code.

The ledger intentionally does not copy untrusted XML text or full source subtrees into its action list. It points to the source event ordinal and binds the before/after field digests.

## Correction dimensions

### Note / rest / unpitched events

- Missing reference event → `InsertInPrediction`.
- Unexpected prediction event → `DeleteFromPrediction`.
- Matched events generate field-level replacements for differences in onset, written pitch spelling, rest/unpitched display, performed duration, written type/dots, grace/cue/chord semantics, mapped chord anchor, voice/staff, note instrument IDs, sound ties, visual ties, time modification, tuplets, stem, notehead, accidental, beams and source note subtree.

Presence matching remains MIR-019's one-to-one family/onset matching. A wrong pitch therefore remains a matched event with a pitch correction; it does not vanish from the note-presence denominator.

Chord anchors are translated through the existing matched source-order map. A chord member whose reference base note cannot be mapped receives an unresolved-anchor action rather than silently comparing two absent targets.

### Auxiliary, attribute and cursor events

Unmatched events create insertion/deletion actions. Matched events create typed field replacements for exact position, retained source subtree, cursor direction, voice/staff, before/after coordinates and duration.

### Measures and part metadata

Measure spans and locators are compared independently from the note list. Actions identify changed span start/end/kind and measure number/source ID/display text/implicit/non-controlling flags.

Part metadata actions are emitted only when both graphs commit the metadata layer. The ledger handles part declaration presence, part name/abbreviation, score-instrument IDs, nested MIDI-instrument setup and complete score-part source subtree. If either graph predates the metadata layer, those actions abstain; missing historical metadata is not treated as proof of an absent source part-list.

## Aggregates and calibration

The report summarizes:
- actions by family, operation and field path;
- number of affected measures;
- unique affected reference and prediction source events;
- metadata action coverage;
- total unweighted action count.

There are no default effort, severity or time weights. Counting an action is not the same as measuring how long a musician takes to make it. A future weighted correction-effort model requires a preregistered, real musician study with observed edits and time-to-accept, and must publish its calibration cohort, uncertainty and generalization limits.

## Integrity and resource limits

- The MIR-019 evaluator performs all input validation and matching once. Unknown/Unsupported graphs or malformed source metadata refuse the ledger.
- Action ordering is deterministic by measure, exact score position, family, source ordinals, operation, field path and action ID.
- Action IDs and the report commitment are domain-separated canonical SHA-256 digests.
- Replay recomputes the MIR-019 result, shared match set, action list, aggregates and final commitment.
- The default action cap is 50,000 (hard V1 maximum 50,000).
- The default serialized report cap is 16 MiB (hard V1 maximum 16 MiB).
- A budget overflow returns an error; no partial correction ledger is presented as complete.

The source graph and the musician's source file remain authoritative. Correction actions do not mutate source notation or override a human correction.

## Tests authored

Eight focused tests cover:
1. identical graphs produce zero actions and deterministic replay;
2. a missing note and an extra note produce separate delete/insert actions with measure locators;
3. a wrong written pitch creates a replace-field action while MIR-019 note-presence F1 remains 1.0;
4. multiple field errors on one event yield multiple actions but one unique affected source event;
5. part name and nested MIDI setup changes are separate actions, while pre-metadata legacy graphs abstain;
6. auxiliary source changes are localized to their measure;
7. action-budget overflow fails closed;
8. replay rejects a tampered action and changed graph.

## Qualification boundary

- Tests are authored, not compiled or executed.
- No human correction-time study, MCR parity, correction-effort correlation or SOTA claim is made.
- The action list is a deterministic explanation of MIR-019's matching and field differences; it is not a calibrated estimate of human work.
- Vendor OMR metrics are not directly comparable without matching format, corpus, annotation, tolerance, track and aggregation.
