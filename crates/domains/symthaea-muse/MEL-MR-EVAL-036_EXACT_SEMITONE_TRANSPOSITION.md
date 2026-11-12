# MEL-MR-EVAL-036 — Operation-Specific Symbolic Counterfactual Validation

## First qualified operation: integer-semitone transposition

**Status:** source and synthetic tests authored on 12 November 2026. Eleven EVAL-017 source-level tests are authored in total; exact-head Rust execution, formatting and Clippy remain pending. This draft implements only the transposition slice. It does not enable onset, duration, velocity, instrument, or fine-tuning operations.

**Parent:** [#9355 — Add operation-specific symbolic counterfactual validation](https://github.com/Luminous-Dynamics/symthaea/issues/9355)

**Base verifier:** [#9128 — MEL-MR-EVAL-017 replayable counterfactual event influence](https://github.com/Luminous-Dynamics/symthaea/pull/9128)

## Why this operation-specific boundary matters

An acoustic difference is not proof that the requested symbolic operation occurred. A render can differ because a protected part was changed, the target was altered incorrectly, or the renderer configuration drifted.

The counterfactual verifier now treats a transposition as an exact symbolic contract before acoustic influence measurement can proceed.

## Implemented

- Enables `EventIntervention::Transpose { semitones }` only for non-zero signed integer intervals in [-127, 127].
- Keeps the original source Score bytes and Score digest as the lineage root.
- Derives the counterfactual primary-pitch inventory internally from the supplied source pitch vector and the declared event index + semitone delta.
- Rejects MIDI pitch underflow/overflow outside 0..127.
- Requires full source-event coverage in both baseline and counterfactual performed-event maps; the target remains present in both.
- Requires the same target descendant count and stable descendant ordinals, MIDI track, channel, MIDI note-on ordinal, onset/end microseconds and frame projections.
- Requires every target descendant pitch to equal its baseline pitch plus exactly the declared semitone delta.
- Requires every non-target descendant to remain exact, including its MIDI note-on ordinal.
- Keeps two pitch commitments in the replayable influence report: the baseline score pitch inventory and the transformed counterfactual pitch inventory.
- Adds `verify_renderer_audio_binding_for_score_transposition`, which derives the transformed inventory from the typed Score and exact intervention rather than accepting a caller-authored expected target pitch list.
- Preserves Remove semantics unchanged. ShiftOnset, ChangeDuration and ChangeVelocity remain rejected until the event-map representation carries exact PPQ/ticks/tempo and velocity fields with their own validators.

## Negative controls and test coverage

The new fixtures exercise:

1. Valid +2 semitone transposition with all protected events and timing preserved.
2. Wrong declared interval versus the receipt-bound transformed pitch inventory.
3. Zero interval rejection.
4. MIDI upper- and lower-bound overflow rejection.
5. Target timing drift rejection.
6. Non-target pitch mutation rejection.
7. Existing Remove behavior and protected-event mutation rejection remain in the same focused test module.

The tests use exact synthetic MIDI and synthetic PCM. They do not claim real FluidSynth listening quality, performer acceptability, cent-level tuning correction or benchmark superiority.

## Deliberate limitations

- This is integer MIDI semitone transposition, not cents or microtonal pitch.
- Performed-event lineage currently carries rounded microseconds and PCM frame projections, not raw MIDI ticks, PPQ and tempo-map identity. Therefore onset and duration deltas remain disabled.
- The current lineage does not retain note-on velocity. Therefore velocity operations remain disabled.
- Instrument/program/bank/percussion changes remain disabled until those identities are represented and preserved in the map.
- A valid symbolic transposition receipt is not proof that the resulting music is aesthetically preferable or that the human requested that exact operation; those remain separate intent and listening evaluations.

## Qualification

The focused exact-head workflow is `.github/workflows/muse-counterfactual-transposition.yml`. It verifies the event SHA, resolves workspace metadata, checks rustfmt, runs the EVAL-017 test module with the `theory` feature, and runs Clippy with warnings denied.

**No Rust test, fmt, Clippy or live FluidSynth pass is claimed until the workflow executes successfully on the exact commit.**
