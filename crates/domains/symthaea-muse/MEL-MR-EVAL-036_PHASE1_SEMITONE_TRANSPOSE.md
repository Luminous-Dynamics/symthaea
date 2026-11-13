# MEL-MR-EVAL-036 — Phase 1: Exact Semitone Transposition

**Date:** 13 November 2026  
**Status:** source implementation and deterministic tests authored; focused Rust qualification pending  
**Parent:** [#9355 — Operation-specific symbolic counterfactual validation](https://github.com/Luminous-Dynamics/symthaea/issues/9355)  
**Implementation branch:** research/mel-mr-eval-036-note-transpose-20261113

## Why this tranche

The canonical MEL-MR-EVAL-017 influence verifier previously admitted only event removal. Its intervention enum reserved transpose, onset shift, duration change and velocity change, but the verifier's source-map and event-count rules correctly assumed removal. Simply allowing those enum variants would turn an unverified declaration into a misleading acoustic receipt.

Phase 1 adds exactly one operation: **integer MIDI semitone transposition of one source Score event**. The target remains present in both performed-event maps. No other event is allowed to change.

This is the smallest non-removal operation that can be established using the current exact MIDI parser and event lineage without inventing tick-domain, controller, fine-tuning or instrumentation evidence.

## Exact contract

Given the baseline Score primary MIDI pitch vector p, the target event index i, and a declared signed semitone delta d:

\[
p'_j =
\begin{cases}
p_i + d & j=i \\
p_j & j\ne i
\end{cases}
\]

Admission requires:

- d != 0, and -127 <= d <= 127;
- both baseline and transformed target pitches remain in MIDI range 0..127;
- the exact typed source Score serialization still matches both renderer receipts;
- the baseline MIDI is reparsed and proves the untransformed Score pitch inventory;
- the counterfactual MIDI is independently reparsed and proves the transformed inventory derived by this verifier;
- the source event remains present in both maps;
- target descendant count and descendant ordinals are unchanged;
- target track, channel, note-on ordinal, onset/end microseconds and PCM frame projections are unchanged;
- each target descendant's MIDI pitch is exactly its baseline pitch plus d;
- every non-target descendant retains event identity, track, channel, MIDI pitch, timing and frame projections; note-on ordinals must also remain unchanged for transposition;
- renderer identity/configuration is matched by the existing EVAL-017 paired-render contract.

The operation's target-pitch transformation is derived inside the Score-facing verifier. A caller cannot provide an arbitrary transformed pitch inventory and ask the receipt to trust it.

## Implementation

The canonical source is crates/domains/symthaea-muse/src/counterfactual_influence.rs.

- validate_intervention retains Remove and admits only valid nonzero integer Transpose deltas.
- counterfactual_primary_pitches derives the expected transformed primary-pitch vector from the baseline vector and stable Score event index, rejecting zero delta and MIDI underflow/overflow.
- verify_renderer_audio_binding_for_score_intervention serializes the typed Score, checks the exact receipt Score bytes, derives baseline pitches from Score::events(), applies the typed operation, and verifies the actual parsed MIDI map.
- validate_input now has two explicit branches:
  - Remove: target absent in counterfactual; all other source events retained.
  - Transpose: every source event remains; target pitch is changed by the exact delta; all target timing/identity fields and all non-target fields remain unchanged.
- CounterfactualInfluenceReport retains the baseline pitch-inventory digest and adds a distinct counterfactual_expected_primary_pitches_sha256. The report version is now v0.2 and its digest domain is v2 because the serialized report schema changed. The repeatability-calibration schema/version remains unchanged.

## Deterministic tests authored

The module now contains thirteen test functions, including an exhaustive finite-domain property test over all 128 MIDI pitches, all signed deltas -127..=127, and both target indices. The additional transposition cases cover:

1. exact +2-semitone target transformation with preserved lineage/timing and replay-verifiable influence report;
2. a valid renderer receipt for +3 semitones falsely declared as +2, which must reject;
3. a non-target pitch change accompanying a claimed target transposition, which must reject;
4. upper MIDI boundary overflow, lower boundary underflow and zero-delta rejection.

The existing Remove tests remain in place.

## Deliberate non-claims

- Integer MIDI transposition is not cent-level tuning correction. SMEBench's A4=440 operation includes ±50-cent deviations; that requires an independently verified fine-tuning/DSP control path.
- This does not enable ShiftOnset, ChangeDuration or ChangeVelocity. The event-map v0.1 currently lacks the raw PPQ/absolute-tick and velocity fields needed for those operations.
- This does not enable instrument replacement, extraction, doubling, drum operations or track remapping.
- A valid symbolic transposition and a measured paired-render difference do not establish perceptual quality, musical plausibility or human preference.
- A receipt hash is local content-integrity evidence, not trusted-host or cryptographic renderer attestation.
- No Rust compilation, test execution, formatting or Clippy result is claimed until the focused exact-head workflow completes.

## Next phases

- **Phase 2 — onset/duration:** version the performed-event map to retain PPQ, absolute onset/end ticks and tempo-map commitment; then verify exact tick deltas without floating-point time round-trips.
- **Phase 3 — velocity/instrumentation:** retain note-on velocity, program/bank/percussion identity and channel/track scope before enabling those interventions.
- **Fine tuning:** separate cent-level tuning from integer MIDI pitch and verify the actual renderer/DSP control path.
- **External evaluation:** keep SMEBench operation-contract success, target-reference distances, protected-content preservation and blinded perceptual acceptance as separate metrics. Never claim a single target waveform defines all valid realizations.

## Benchmark targets (prospective)

- 100% pass for exact valid transposition fixtures across positive and negative semitone deltas and boundary-adjacent notes.
- 100% rejection for wrong-delta, wrong-target, target-timing-change, non-target-pitch-change, missing/duplicate descendant and MIDI-range overflow controls.
- Zero false acceptance on no-op and renderer-context mismatch controls.
- Report transposition separately from all unsupported SMEBench operation families.
- Require held-out human listening before asserting perceptual-quality or musical-equivalence improvements.

These are qualification targets, not achieved results.
