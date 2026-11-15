# MEL-MR-EVAL-043 — Multi-step edit integrity contract v1

**Date:** 2026-11-15  
**Status:** Stage 1 and Stage 2 patch algebra qualified at exact head `fefc148b1dfd1db898f1641dbee2756dbbfe5afc`; two-parent branch-merge extension authored and awaiting exact-head qualification  
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
- Explicit Undo must restore the exact artifact and symbolic-state digest consumed by the operation being undone. Redo must restore the exact output of that operation and is only valid immediately after its matching undo. A new Apply after Undo invalidates the redo edge.
- Every step independently records target edit, protected preservation, rights, integrity, source lineage and realization evidence. Unknown and rejected are not success.
- A failed/unknown hard-gate step remains in the result as a rejected session. A later step cannot proceed from it and cannot retroactively turn the session green.
- Candidate attempts, failed attempts, render calls, retries and manual interventions accumulate over the whole session.

### Executable controls

Forty-three unit tests are authored. Thirty-six passed on exact head `fefc148b1dfd1db898f1641dbee2756dbbfe5afc`; seven new two-parent merge tests are awaiting the current exact-head workflow. The tests cover:
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
- exact undo restoration, rejection of near-miss restoration, exact redo replay, redo-without-undo rejection and invalidation of redo after a new edit;
- preregistered session-length ladder bounds;
- read/write-set commutativity and execution-order convergence;
- same-event operation conflicts and stale patch preconditions;
- exact undo/redo with history-receipt tamper detection;
- disjoint branch merge with both parents retained;
- conflicting branch merge with exactly one explicit resolution per conflict.

The eight-step example produces a deterministic symbolic session in which each operation changes one event and leaves the other seven unchanged. It is a known-answer contract fixture, not an audio-quality result.

## Stage 2 — operation algebra and collaborative history

The harness now adds deterministic symbolic operation patches with exact before-state preconditions and event-level write sets derived from canonical patch deltas and exact before-value preconditions.

- **Commutativity:** disjoint event-write patches are replayed in both orders; a commuting declaration passes only when both orders converge to the same canonical state digest.
- **Non-commutativity:** divergent same-event writes are not silently serialized; a stale before-value fails and an explicit conflict disposition is required.
- **Undo/redo:** the operation, before/after state digests and inverse patch are bound into a domain-separated history receipt. Undo is accepted only from the exact after-state; redo only from the exact before-state. Tampering or stale state fails closed.
- **Branch/merge:** each parent snapshot commits to the exact common base and its own head. Disjoint writes merge automatically; divergent writes to the same event require exactly one sorted per-event resolution. The receipt records both branch IDs, both parent-state digests, the conflict set, resolutions and merged-state digest. Replay recomputes the merge; digest-only re-sealing of a fabricated result is rejected.

This is a symbolic operation algebra, not yet a complete multi-track DAW or collaborative CRDT. It does, however, create the core contract needed to test reorderability, reversibility and conflict semantics before introducing rendered audio.

## Exact-head qualification

The focused Melothaea multi-step sequence integrity workflow passed on exact head `fefc148b1dfd1db898f1641dbee2756dbbfe5afc` (run 15):

- Rust 1.96 exact checkout and SHA assertion: passed.
- Sequence JSON Schema validation: passed.
- Staged isolated-crate formatting: passed.
- Standalone compile/tests: **36 passed, 0 failed** (sequence ledger, patch algebra, undo/redo and operation-order controls).
- Deterministic eight-step preservation example: passed.
- Clippy with warnings denied: passed.
- Exact source/schema/dependency identity artifact uploaded: `mel-sequence-fefc148b1dfd1db898f1641dbee2756dbbfe5afc` (SHA-256 `17e4311e58b633d0e27fcf1a8f746b2dd3b9af9d56c44f23e09c56de01b8fe78`). The new branch-merge implementation adds seven further tests; those remain pending until the current exact-head workflow passes.

Formatting is normalized on the staged copy; this is not a claim that committed source text was already rustfmt-clean. Repository-wide CI, Workflow Syntax and PR Governance were skipped by path filters, not passed.

## Deliberate limits

This stage is symbolic MIDI-like event state only. It does not:
- render or listen to audio;
- measure perceptual continuity, groove, timbre, dynamics or human preference;
- execute real edits or invoke a model, plugin, DAW or external renderer;
- cryptographically validate the referenced EVAL-042/EVAL-041 receipts (it validates their syntax and distinct identities; upstream verifiers own trust);
- implement a distributed CRDT, arbitrary DAG history synchronization or a production collaborative DAW merge engine;
- authorize commit, export, publication or sharing.

The initial artifact digest is treated as an upstream verified identity; binding it to exact bytes is owned by EVAL-035 / EVAL-041. The symbolic state hash proves only the canonical event snapshot supplied to this harness.

## Prospective targets

- Zero unauthorized symbolic event changes outside the declared event support on deterministic fixtures.
- Zero hard-gate false accepts at any step.
- 100% exact symbolic-state restoration on deterministic undo/redo fixtures.
- 100% canonical convergence for declared commuting operation pairs.
- 100% explicit conflict outcomes for declared non-commuting pairs.
- At least 98% per-step hard-obligation success and 85% complete 8-step session acceptance on a preregistered real-audio confirmatory cohort, after power analysis.
- Lower p95 cost per fully accepted sequence than repeated regenerate-from-source at matched operation mix and verified quality.

These are prospective targets, not measured results.

## External frontier comparison — 2026-11-15

The strongest current directly comparable single-step agent result surfaced in this review is Harmony (arXiv:2611.14724, posted 2026-11-10). On 420 MMAE examples across four operations—addition (115), removal (149), replacement (79) and volume adjustment (77)—it reports pooled IFR 62.18%, CR 87.71% and EMR 22.62% (95/420 exact matches), compared with 13.57% EMR for original WavCraft and 8.33% for the Audio Editing Challenge Agent baseline on that same 420-example subset. This is not the full 2,000-example public MMAE scope and should not be compared directly to its 7.45% Agent EMR. Harmony uses a GPT-5 mini planner, Qwen3-Omni audio observation, SAM Audio separation, Stable Audio 3 generation and Basic Pitch / pYIN for supported music rendering; it retains an N=8 internal SAM reranker. Its paper explicitly says the current system is single-step, incurs multiple LLM-call cost, and can produce abrupt transitions. Its small listener study preferred Harmony on overall editing (69.53% of 128 responses) and preservation (56.67% of 120 responses), but this is not a large confirmatory trial.

The ICASSP 2027 Agent Track now defines a harder public target: 500 previously unreleased evaluation examples, autonomous planning/tool use/intermediate inspection/refinement, no human intervention, open-source models/tools/data, model weights publicly released before 2026-11-01, and all non-LLM models/tools running locally. The leaderboard's published deadline is 2026-11-25; participant IDs and submission keys are privately distributed. Our recently authored November code does not itself prove eligibility against the pre-November-1 release rule. Any entry must first confirm that the chosen pre-cutoff components, deployment arrangement, registration and organizer authorization satisfy the official rules. The exact public 2,000-example baseline remains a separate comparison scope.

Commercial studio comparisons are capability comparisons, not measured benchmark scores. Suno Studio 2.0 now exposes MIDI editing/piano roll, automation, built-in effects, custom plugin generation and chat-based clip generation/arrangement. Moises Studio advertises live collaborative sessions, stems, MIDI conversion, and an Agentic Mode whose actions are visible and reversible, with final decisions kept by musicians. Therefore, "reversible edits" alone is not a sufficient differentiation. The intended Melothaea contribution is a stronger machine-checkable guarantee: exact source/state ancestry, allowed event-support locality, per-step independent hard-gate replay, append-only failed-attempt accounting, and eventually multi-parent conflict-aware merge proofs. Google Lyria RealTime is a low-latency streaming, steerable instrumental-generation comparator, not evidence of exact edit-session correctness.

Sources: https://arxiv.org/abs/2611.14724 ; https://audio-editing-challenge.github.io/leaderboard/ ; https://audio-editing-challenge.github.io/track2/ ; https://suno.com/release-notes/studio-2 ; https://moises.ai/newsroom/product-announcements/moises-launches-studio-collaborative-workspace/ ; https://ai.google.dev/gemini-api/docs/realtime-music-generation
