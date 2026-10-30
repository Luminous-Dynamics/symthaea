# MEL-MR-EVAL-033 — contract-valid symbolic realization equivalence

**Status:** intent-level pitch-choice evaluator implemented; eleven adversarial source tests authored; no compile or test execution claimed.

## Why this exists

A paired benchmark target is a controlled example of one acceptable result. It is not necessarily the unique valid realization of the user's request. Exact target waveform/score equality can therefore reject a musically valid alternative, while source similarity can reward a no-op.

EVAL-033 separates:
1. the requested musical change;
2. preservation of every protected score property;
3. similarity to the benchmark reference as a diagnostic only.

## Implemented first vertical slice

Module: `src/realization_equivalence.rs` (feature `theory`).

The first operation is an event-scoped pitch-choice contract. It binds to:
- the exact source Score content digest;
- a stable event ID in the onset-ordered Score event stream;
- a sorted set of at least two permitted MIDI notes;
- an optional paired-target Score digest.

The permitted set must exclude the source pitch. The operation therefore cannot be completed by returning the source unchanged.

A candidate is accepted only when:
- its target event resolves unambiguously to the bound source event;
- its selected pitch is in the permitted set and differs from the source pitch;
- event count is unchanged;
- every non-target note retains its pitch;
- every event retains its exact non-pitch identity (part, onset, duration, velocity, voice role, emphasis, section intensity);
- score-level key, tempo, meter and total-beat context remain identical.

The event join uses pitch-independent exact identity. Duplicate source or candidate identity keys are ambiguous and fail closed. This avoids nearest-onset matching, but the current Score schema lacks persistent event UUIDs, so duplicate event identities remain unsupported.

## Reference target semantics

The report exposes three separate reference measurements:
- `exact_reference_match`: complete serialized Score SHA-256 equality;
- `semantic_reference_match`: equality of score context and event multiset, independent of note-storage order;
- `reference_satisfies_contract`: whether the supplied paired reference itself satisfies the intent/preservation predicates.

None of these values gates candidate acceptance. A valid candidate with a different allowed pitch from the benchmark reference is accepted and marked `accepted_alternative`. A different serialization with identical musical content is not misreported as a new musical alternative.

The audit is replayable: it binds source, candidate, optional reference, contract digest, independent predicates, violations and report digest. `verify_symbolic_realization_report` recomputes the report from the original inputs.

## Adversarial tests authored

1. A valid allowed pitch differing from the paired target is accepted.
2. The paired reference itself is accepted when it satisfies the contract.
3. An unchanged source pitch is rejected even though it maximizes source similarity.
4. Editing the wrong event is rejected.
5. An unrequested pitch change to a protected event is rejected.
6. Collateral timing/dynamics changes break event identity and are rejected.
7. Event insertion/deletion is rejected.
8. Note-storage reordering is not counted as a new musical realization.
9. Ambiguous source event identity is rejected.
10. The source pitch cannot appear in the allowed alternative set.
11. Stale source binding and report tampering are rejected.

## Limits — intentionally narrow

This first slice does **not** yet verify:
- instrument replacement or doubling;
- track insertion/deletion;
- polyphonic event-identity reconstruction when multiple notes have identical non-pitch identity;
- articulation/duration alternatives;
- fine tuning in cents;
- audio-only/real-audio edits;
- human-perceptual validity.

It is a symbolic contract primitive, not proof of audible quality. It should not be used to score SMEBench's instrumentation/drum edits or MMAE general audio instructions unless a compatible operation contract is added. Unsupported tasks must be reported as unsupported/abstained.

## Qualification targets

- 100% acceptance of every hand-constructed candidate in the permitted pitch set that preserves all protected properties.
- 100% rejection of no-op, out-of-set pitch, wrong-event and collateral-change controls.
- 100% rejection of duplicate/ambiguous event identities.
- 100% report replay success on untampered inputs and rejection on tampered inputs.
- On future external data, report acceptance separately from exact/semantic reference match; stratify by operation family and source lineage.

These are test/qualification targets, not measured performance.

## Next generalization steps

1. Add typed operation families only when the source schema can identify their target scope without guessing: duration/articulation, dynamics envelopes, track/instrument and arrangement operations.
2. Add stable persistent event UUIDs to the Score schema so polyphonic/duplicate-identity edits do not need conservative abstention.
3. Define allowed transformation spaces (sets, intervals, predicates) rather than copying one target score as the only accepted output.
4. Connect each operation contract to the existing symbolic and audio preservation gates; audio acceptance remains independent.
5. Calibrate perceptual acceptability with EVAL-021 blinded listener evidence and keep the automated judge audit from EVAL-032 separate.

## Validation boundary

Eleven Rust tests are authored. The current environment does not provide cargo, rustc or rustfmt; compilation, execution and external dataset qualification remain unverified. No benchmark score, perceptual improvement or SOTA claim is asserted.