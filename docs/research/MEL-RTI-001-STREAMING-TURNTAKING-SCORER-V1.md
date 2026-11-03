# MEL-RTI-001 — Streaming musical turn-taking scorer V1

Status: implementation proposal / draft PR. The Rust regression tests are authored but require exact-head Rust execution before any pass claim.

## Purpose

This document describes the first reusable evaluator for [#10307 — streaming musical turn-taking and timely ensemble response](https://github.com/Luminous-Dynamics/symthaea/issues/10307).

The scorer evaluates supplied session annotations and per-chunk predictions. It does **not** detect cadences, phrase boundaries, rests, invitations or handoffs from waveform bytes. It does not render output audio and does not prove that a runner actually supplied only the prefix declared in its trace.

## External benchmark relationship

The HEARTBEAT Challenge 2027 (ICASSP 2027 GC-15) asks whether an audio agent should WAIT, RESPOND, CONTINUE or YIELD as streaming input arrives. It includes offline and online turn-taking, plus online proactive understanding. Its protocol requires each input chunk to be logged, including chunks without a response, and prevents response timestamps from being backdated.

Official reference: https://heartbeat-challenge.github.io/

Melothaea V1 uses the four-action vocabulary but is not a drop-in HEARTBEAT adapter. The musical annotations need event semantics such as cadence, phrase boundary, pickup, solo handoff and deliberate continuation. Official leaderboard participation and eligibility are not assumed.

## Contract design

### Artifact identity and clock

Each truth and prediction record binds to:
- exact audio SHA-256 (syntax checked, bytes not independently loaded here);
- sample rate, channel count and total frame count;
- session ID and schema version.

All timing is in half-open sample-frame intervals. A later adapter may add beat-domain timing once the canonical beat-to-performance map in #9477 is available and bound to the rendition.

### Chunk coverage

A session is partitioned into ordered chunks with stable zero-based indices. The intervals must:
- begin at frame zero;
- be non-empty and in range;
- be contiguous with no gap or overlap;
- finish exactly at the artifact's total frame count.

There must be one prediction record per chunk. An omitted chunk is not interpreted as WAIT.

### Action semantics

| Action | Operational meaning | Effect timestamp |
|---|---|---|
| WAIT | Preserve listening space; take no new floor | None |
| RESPOND | Start a new musical contribution after an invitation | Actual/scheduled response onset |
| CONTINUE | Sustain the current contribution rather than prematurely yield | None |
| YIELD | Relinquish the musical floor | Actual/scheduled output-stop boundary |

Ground truth supplies an effect target and permitted window only for RESPOND and YIELD. The effect window must begin at or after the end of the input chunk that led to the decision.

### Causal-clock checks

Keep two clocks separate:

1. **Session sample-frame clock:** input chunk boundaries and musical response/yield event positions.
2. **Monotonic runner clock:** chunk receipt, action decision, and actual effect emission timestamps in nanoseconds.

For every prediction chunk:
- observed-through frame must equal the current chunk's end frame;
- chunk receipt and decision times must be monotonic across the session;
- decision time cannot precede chunk receipt;
- a RESPOND or YIELD effect frame, when supplied, cannot precede the observed-through frame;
- the monotonic effect-emission time, when supplied, cannot precede the decision;
- a WAIT or CONTINUE decision cannot carry an action-effect timestamp.

Decision latency is computed from monotonic `decision_at_ns - chunk_received_at_ns`; action-effect latency is computed from monotonic `effect_at_ns - decision_at_ns`. Musical effect-window error is computed separately in sample frames and converted using the source audio sample rate. This prevents musical timeline offsets from being misreported as compute or actuation latency.

These checks reject declared lookahead, missing chunks and backdating. They do not prove the external runner's actual model-input isolation; that requires a trusted runner or independently captured input receipts.

## Metrics reported separately

- Four-class confusion matrix, top-1 action accuracy, per-action precision/recall/F1, macro-F1 and balanced accuracy.
- RESPOND/YIELD effect-window success, signed effect-time error and p95 absolute timing error.
- WAIT false-activation rate.
- CONTINUE premature-yield count.
- Post-yield non-WAIT decisions before the next expected response invitation.
- Decision latency p50/p95/p99 from the monotonic runner clock.
- Action-effect latency p50/p95/p99 from the monotonic runner clock.
- Parse failures separately from valid WAIT decisions.

Action classification and timed action execution are deliberately separate. A model can select RESPOND correctly but fail to produce the response effect within the permitted window.

The post-yield metric is trace-level action leakage, not an acoustic measurement of lingering output energy. A later renderer-bound verifier must inspect the exact PCM bytes and output event schedule before making an audible-silence claim.

## Deterministic regression fixtures

The module includes synthetic checks for:
1. all four actions with valid timing windows;
2. action confusion and premature yield;
3. parse failure distinct from WAIT;
4. lookahead declaration;
5. a decision timestamp before the current chunk end;
6. an effect timestamp before the decision;
7. missing chunks;
8. chunk gaps and mismatched artifact identity;
9. effect timestamps on untimed actions;
10. exact input commitment sensitivity;
11. post-yield non-WAIT behavior during a reference listening interval.

## Qualification targets (proposed, not results)

- 100% chunk coverage and monotonicity for any scored trace; otherwise invalid.
- 0 accepted lookahead or backdated action-effect timestamps.
- 100% synthetic action accuracy on deterministic, non-ambiguous controls.
- 0 false activations on WAIT controls and 0 premature yield on CONTINUE controls.
- Report per-action performance, latency distributions and confidence intervals clustered by session/audio identity.
- Set live millisecond thresholds from tempo-, instrument- and device-specific musician pilots, not from an arbitrary global limit.

## Next stages

1. Add dataset aggregation and session-cluster bootstrap intervals.
2. Add explicit audio-output event receipts to measure acoustic onset/offset, repeated/duplicate responses and actual post-yield leakage.
3. Add cue-conditioned adapters for HEARTBEAT's offline action labels where semantics match; preserve official splits and scoring.
4. Add symbolic score/beat-aligned annotations and multiple simultaneous collaborator roles.
5. Connect to a live MIDI/audio runner only after input-prefix and output-clock receipts can be independently observed.
6. Run randomized musician pilots before claiming improved ensemble coordination, agency, or musical quality.
