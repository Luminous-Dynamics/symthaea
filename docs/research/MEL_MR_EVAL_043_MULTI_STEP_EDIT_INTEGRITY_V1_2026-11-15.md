# MEL-MR-EVAL-043 — Multi-step edit integrity contract v1

**Date:** 2026-11-15  
**Status:** Source and tests authored; exact-head qualification pending  
**Issue:** [#12243](https://github.com/Luminous-Dynamics/symthaea/issues/12243)  
**Stack base:** EVAL-042 [PR #12201](https://github.com/Luminous-Dynamics/symthaea/pull/12201)

## Why a separate sequence benchmark

Single-step editing scores do not establish that a session remains correct after repeated operations. The relevant unit is a state-transition chain: each step must consume the exact predecessor, change only its declared support, re-establish hard obligations, and preserve the history of every prior decision.

EVAL-042 measures selection pressure over a candidate pool. EVAL-043 measures cumulative correctness over edit operations and session history. EVAL-040 remains the only terminal transaction authority. The sequence contract consumes EVAL-042 selection receipts, EVAL-041 realization receipts and independent per-step evidence identities; it does not reimplement their authority.

## Stage 1 implementation

The standalone crate is \`tools/mel-music-sequence-integrity-harness\`.

### Frozen state model

- A request binds session identity, source-lineage digest, initial artifact digest, canonical initial symbolic-state digest, operation-plan digest and an expected length of 1, 2, 4 or 8.
- A symbolic event binds stable event and part IDs, onset, duration, pitch, velocity and program.
- Event arrays must be sorted by stable event ID and have unique IDs. The state digest is recomputed from the canonical serialized event sequence with a domain-separated SHA-256 commitment.
- Every step binds instruction identity; source lineage; input and output artifact identities; input and output symbolic-state identities; distinct EVAL-042 selection, EVAL-041 realization and verification-report digests; declared allowed event IDs; full output state; per-step gates and cost counters.
- Each subsequent step must consume both the exact previous artifact digest and exact previous symbolic-state digest. The initial symbolic state is itself committed by the frozen request.
- An event added, deleted or changed outside the declared event support is rejected. A changed symbolic state cannot retain the same artifact digest.
- Every step independently records target edit, protected preservation, rights, integrity, source lineage and realization evidence. Unknown and rejected are not success.
- A failed/unknown hard-gate step remains in the result as a rejected session. A later step cannot proceed from it and cannot retroactively turn the session green.
- Candidate attempts, failed attempts, render calls, retries and manual interventions accumulate over the whole session.

### Executable controls

Fifteen unit tests are authored. They cover:
- exact two-step ancestry and cumulative cost;
- unauthorized collateral event changes;
- stale artifact and symbolic-state parents;
- binding the initial event state to the frozen source;
- changed event state with an unchanged artifact digest;
- later hard-gate checks after earlier steps passed;
- rejected/unknown hard gates and no retroactive laundering;
- incomplete sessions;
- tampered state digests;
- canonical ordering and duplicate event IDs;
- distinct selection/realization/verification receipt identities;
- candidate-attempt accounting;
- preregistered session-length ladder bounds.

The eight-step example produces a deterministic symbolic session in which each operation changes one event and leaves the other seven unchanged. It is a known-answer contract fixture, not an audio-quality result.

## Qualification

The exact-head workflow uses Rust 1.96.0, verifies the checked-out PR head, stages the isolated crate outside the large workspace, normalizes and checks staged formatting, runs tests and the eight-step example, runs Clippy with warnings denied, and uploads source/schema/lockfile identities.

**No compilation, test, formatting or Clippy result is claimed until that workflow passes for the current exact head.** As with the EVAL-042 harness, rustfmt currently normalizes a staged copy; it does not yet prove that committed source text was originally formatted.

## Deliberate limits

This stage is symbolic MIDI-like event state only. It does not:
- render or listen to audio;
- measure perceptual continuity, groove, timbre, dynamics or human preference;
- execute real edits or invoke a model, plugin, DAW or external renderer;
- cryptographically validate the referenced EVAL-042/EVAL-041 receipts (it validates their syntax and distinct identities; upstream verifiers own trust);
- implement commuting/non-commuting operation permutations, undo/redo, branch/merge or multi-parent lineage DAGs;
- authorize commit, export, publication or sharing.

The initial artifact digest is treated as an upstream verified identity; binding it to exact bytes is owned by EVAL-035 / EVAL-041. The symbolic state hash proves only the canonical event snapshot supplied to this harness.

## Prospective targets

- Zero unauthorized symbolic event changes outside the declared event support on deterministic fixtures.
- Zero hard-gate false accepts at any step.
- 100% exact symbolic-state restoration for future undo/redo fixtures.
- 100% canonical convergence for declared commuting operation pairs.
- 100% explicit conflict outcomes for declared non-commuting pairs.
- At least 98% per-step hard-obligation success and 85% complete 8-step session acceptance on a preregistered real-audio confirmatory cohort, after power analysis.
- Lower p95 cost per fully accepted sequence than repeated regenerate-from-source at matched operation mix and verified quality.

These are prospective targets, not measured results.