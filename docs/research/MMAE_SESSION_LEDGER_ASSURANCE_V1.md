# MMAE-TRACE-001 — Multi-turn audio edit session assurance

**Date:** 10 November 2026  
**Status:** standalone append-only ledger evaluator and synthetic controls; no agent performance or official MMAE score claimed  
**Owner:** https://github.com/Luminous-Dynamics/symthaea/issues/11508

## Why this is distinct from per-edit evaluation

MMAE Track 2 includes multi-instruction, multi-audio, multi-round and multi-hop editing. A multi-turn agent can produce a plausible final waveform while silently violating a prior instruction or applying a later edit to the wrong parent output.

VMEA-001 evaluates a case's instruction and preservation rubrics. VCCR-001 evaluates a bounded sequence of corrective renders for a fixed control objective. MMAE-TRACE-001 adds the conversation/session layer: it verifies that active constraints survive across user turns and that the final output is the terminal result of the recorded edit chain.

## Paradigm: event-sourced intent ledger

Each session binds its initial audio manifest, frozen context and initial output. Events are appended in order and linked by event digests. A later turn does not implicitly erase any earlier constraint. An instruction or preservation requirement is removed from the active set only by an explicit, same-role supersession event that names the replacement.

Every successful edit binds the exact parent output digest, action receipt, system artifact and new output digest. Failed/unsupported/abstained tool attempts are retained separately and cannot be erased by a later successful retry.

Each observation binds one active constraint, one output digest, one rubric profile, one evaluator artifact and one evidence receipt. An observation against an old output cannot qualify the final output after another edit occurs.

## V1 invariants

1. SessionStarted occurs once at event index zero and matches the frozen session specification.
2. Event indexes are contiguous; event digests are unique; every event after the first has the exact previous event digest.
3. User turns are contiguous and start at one.
4. Constraint IDs are unique.
5. Supersession requires an existing active old constraint and existing active replacement; self-supersession, role changes and repeat supersession are rejected.
6. Edit step IDs are contiguous; every edit consumes the exact current output digest.
7. An observation applies only to a currently active constraint and exact current output.
8. Rubric profile digest on observation equals the profile declared by the constraint.
9. Final output digest equals the last successful output.
10. Finalization requires an observation for every active constraint against the exact final output.
11. Every final active constraint must pass for QUALIFIED; any definite failure is NONCOMPLIANT; otherwise any uncertainty is INDETERMINATE.
12. No event or mutation is accepted after finalization.

The event hash values are supplied by a canonical serializer outside this crate. The evaluator validates hash format, uniqueness and link continuity; it does not claim cryptographic recomputation.

## Experiment design

Build matched session fixtures representing:
- two independent turns that preserve both instructions;
- an edit that intentionally changes an earlier constraint through explicit supersession;
- a turn that accidentally undoes an earlier constraint;
- two source audio inputs and a multi-hop chain;
- a preserved melody/timbre/accompaniment constraint that must survive later turns;
- failed tool attempts and retries;
- unsupported operations and explicit abstention;
- stale-output and event-order attacks.

Compare:
- a stateless per-turn tool agent;
- a rolling-context agent without explicit constraint ledger;
- an event-sourced ledger agent with active-constraint verification;
- no-op and deliberate regression controls.

Report per-turn instruction satisfaction, final active-constraint satisfaction, stale-parent detection, supersession errors, failed-attempt retention, invalid finalization count, event-chain integrity, tool calls, render cost and session latency.

## MMAE Track 2 context

The November 10, 2026 released Track 2 test input card contains 500 cases spanning single, multi-part, multi-instruction, multi-audio, multi-round and multi-hop complexity. Its rubric labels remain private. The official baseline reports IFR 44.13%, CR 76.09% and EMR 8.00% on that track.

This ledger can add engineering session-integrity evidence, but it cannot reproduce or replace the official Qwen3-Omni judging. Report official IFR/CR/EMR separately if an authorized official evaluation is available.

## Proposed qualification targets

Targets, not observed outcomes:
- 100% detection of stale parent, skipped/duplicated event, invalid supersession and post-finalization mutation fixtures;
- 100% retention of failed/unsupported/abstained attempts;
- zero QUALIFIED sessions with any active constraint FAIL or INDETERMINATE;
- 100% final-output digest agreement with the final event;
- demonstrate improvement over stateless and rolling-context baselines on final active-constraint satisfaction without increasing invalid supersession or false acceptance.

## Related work

- #11503 / PR #11506 VMEA-001 task-aware rubric scoring
- #11482 / PR #11488 VCCR-001 bounded correction trajectories
- #11356 renderer-neutral VocalEditPatch
- #11354 VELB-001 waveform locality
- #11402 independent versioned acoustic observations
- ICASSP 2027 MMAE Track 2 official test set