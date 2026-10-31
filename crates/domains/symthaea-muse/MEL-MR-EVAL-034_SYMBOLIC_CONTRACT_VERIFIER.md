# MEL-MR-EVAL-034 — independent symbolic edit-contract verification

**Status:** theory-feature Rust verifier implemented; eleven adversarial source tests authored; no compile/test run or audio/renderer claim made.

## Research hypothesis

A frozen edit contract is a signed statement of intent and constraints. Its hash proves only that the statement has not changed. It does not prove the editor created a target Score that satisfies the requested transformation.

The verifier must sit after the symbolic transform and before audio rendering:

1. Planner emits a frozen edit contract.
2. Transform produces a concrete target Score.
3. Independent verifier hashes source/target Scores and checks the operation and protected symbolic properties.
4. Audio renderer executes the verified target.
5. Independent audio and renderer evidence discharges remaining obligations.

## Supported operations

The implementation intentionally uses the existing narrow EVAL-010/EVAL-011 policy:

### pitch-at

- Resolves a canonical event identifier, `score_event_index:N`, against the onset-sorted Score::events() stream that Muse renders, not the backing storage order.
- Requires the selected target pitch to equal the declared MIDI note and to differ from the source pitch.
- Requires all other source/target notes to be exactly equal.
- Requires selected-note non-pitch attributes, key, tempo, meter and total score length to remain unchanged.
- Rejects alternate event-ID encodings and out-of-range indices.

### transpose

- Requires a nonzero integer semitone delta. Fractional semitone deltas are not silently rounded to MIDI.
- Requires every target note pitch to equal source pitch plus the declared delta.
- Rejects MIDI range overflow/clamping.
- Requires the key tonic to transpose by the same pitch-class delta while preserving tonality.
- Preserves every non-pitch note field bit-exactly (including float-valued dynamics), plus tempo, meter and total length.

The verifier does not extend the supported operation list. Instrumentation, dynamics, articulation, cent-level tuning, audio-only edits and timbral edits remain outside this gate.

## Implementation

Module: `src/symbolic_edit_verifier.rs`, gated behind the existing `theory` feature.

Public interface:
- `symbolic_score_sha256`: domain-separated digest over the canonical serialized Score.
- `verify_symbolic_edit_contract`: compares actual source/target Score inputs to the frozen contract.
- `SymbolicEditVerification`: includes source/target/contract hashes, a symbolic verdict, a separate overall verdict, verified IDs, failed IDs, pending external IDs and failures. A correct symbolic edit can have symbolic `pass` and overall `incomplete` until audio/renderer evidence arrives.
- `verify_symbolic_edit_verification`: re-runs the check from source, target and contract and rejects tampered reports.

A symbolic pass does not imply an end-to-end pass. The current pitch contracts also require:
- edited-region RMS preservation;
- protected-remainder RMS preservation;
- protected-remainder spectral preservation;
- renderer configuration digest binding.

Those four obligations are explicitly returned as pending external evidence when the symbolic checks pass. A score-only report is therefore `incomplete`, not success.

Any symbolic failure takes precedence over pending evidence. A clean symbolic result cannot compensate for a missing acoustic proof, and an audio metric cannot compensate for a wrong pitch or collateral note edit.

## Adversarial source tests

Eleven tests are authored:

1. Valid pitch-at change passes symbolic checks but remains audio-incomplete.
2. No-op pitch request is rejected even when the target Score hash matches.
3. Wrong target pitch and edit on the wrong event are rejected.
4. Collateral edit to a protected note is rejected.
5. Selected-note timing change and score metadata drift are rejected.
6. Whole-score transpose checks every pitch and the key tonic.
7. Partial transpose and MIDI range clamping are rejected.
8. Source/target digest mismatches are rejected.
9. Tampered verification receipts fail replay.
10. Noncanonical score event IDs fail closed.
11. Event IDs resolve against the onset-sorted renderer event stream even when backing note storage is deliberately unsorted.

## Qualification targets

- 100% rejection of deliberately invalid symbolic fixtures.
- 100% acceptance of exact valid pitch-at/transpose fixtures.
- Zero false acceptance for no-op, wrong-event, collateral-note and out-of-range controls.
- 100% of score-only results with audio/renderer obligations classified as incomplete.
- Deterministic Score and report commitments with successful replay on identical inputs.

These are gates, not measured results.

## Integration path

- #8985 / EVAL-010: frozen edit-contract schema and structural policy.
- #9006 / EVAL-011: operation-specific obligation completeness and canonical units.
- #9307 / EVAL-033: valid alternative realizations. The verifier can accept any score that satisfies the typed symbolic contract; it does not compare the output to only one target MIDI.
- #9284 / EVAL-030: paired-target benchmark adapter. Record symbolic contract success separately from target-waveform distance.
- #9275 / EVAL-029: renderer effects configuration and route experiment.
- #9218 / EVAL-025: replay-verified FluidSynth support evidence and acoustic binding.

## Claim boundary

This module verifies only symbolic pitch-at/transpose transformations. It does not establish that a renderer used the verified Score, that output PCM corresponds to it, that protected audio is perceptually unchanged, or that listeners accept the result. An end-to-end certificate requires the independent renderer receipt, acoustic preservation report and perceptual validation to be composed with this symbolic report.

The eleven tests are authored but not compiled or run in the current environment. No CI pass or measured editing improvement is claimed.