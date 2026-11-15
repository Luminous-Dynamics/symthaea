# MEL-MR-EVAL-043 — Multi-step edit integrity contract v1

**Date:** 2026-11-15  
**Status:** Stage 1 and Stage 2 focused qualification passed at exact head `f4ff5f3971f3acc3a8ea4b5e7b61f56c00cbe18d`  
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

Twenty-one unit tests are authored. They cover:
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
- preregistered session-length ladder bounds;
- read/write-set commutativity and execution-order convergence;
- same-event operation conflicts and stale patch preconditions;
- exact undo/redo with history-receipt tamper detection;
- disjoint branch merge with both parents retained;
- conflicting branch merge with exactly one explicit resolution per conflict.

The eight-step example produces a deterministic symbolic session in which each operation changes one event and leaves the other seven unchanged. It is a known-answer contract fixture, not an audio-quality result.

## Stage 2 — operation algebra and collaborative history

The harness now adds deterministic symbolic operation patches with exact before-state preconditions and explicit read/write sets.

- **Commutativity:** an operation pair is declared independent only when write/write and read/write intersections are empty and both execution orders yield the same canonical state digest.
- **Non-commutativity:** same-event writes are not silently serialized; a stale precondition fails and the pair is marked non-commuting.
- **Undo/redo:** the operation, before/after state digests and inverse patch are bound into a domain-separated history receipt. Undo is accepted only from the exact after-state; redo only from the exact before-state. Tampering or stale state fails closed.
- **Branch/merge:** every branch commits to the same base state. The merge plan computes event-level patches, retains both branch IDs, and surfaces conflicting event IDs. The resolver requires exactly one explicit choice for each conflict; it never auto-picks a winner.

This is a symbolic operation algebra, not yet a complete multi-track DAW or collaborative CRDT. It does, however, create the core contract needed to test reorderability, reversibility and conflict semantics before introducing rendered audio.

## Exact-head qualification

The focused Melothaea multi-step sequence integrity workflow passed on exact head `f4ff5f3971f3acc3a8ea4b5e7b61f56c00cbe18d` (run 7):

- Rust 1.96 exact checkout and SHA assertion: passed.
- Sequence JSON Schema validation: passed.
- Staged isolated-crate formatting: passed.
- Standalone compile/tests: **21 passed, 0 failed**.
- Deterministic eight-step preservation example: passed.
- Clippy with warnings denied: passed.
- Exact source/schema/dependency identity artifact uploaded: `mel-sequence-f4ff5f3971f3acc3a8ea4b5e7b61f56c00cbe18d`.

Formatting is normalized on the staged copy; this is not a claim that committed source text was already rustfmt-clean. Repository-wide CI, Workflow Syntax and PR Governance were skipped by path filters, not passed.

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