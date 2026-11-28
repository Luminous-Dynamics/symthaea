# Symthaea Causal Turn Decision Gate v0.1

**Date:** 2026-11-27  
**Status:** Rust decision gate and tests authored; detector and benchmark adapter remain separate work

## Purpose

The gate sits between a turn detector and the semantic event stream. It does not listen to audio or detect a floor transfer. It accepts a candidate WAIT, RESPOND, CONTINUE or YIELD action and decides whether that action is safe and well-formed enough to become a turn-state event.

This makes the boundary explicit: the detector proposes; the gate validates; the presentation policy separately decides whether an authorized output may be rendered.

## Decision contract

Each candidate contains:
- unique event ID and monotonically increasing event sequence;
- source monotonic clock-domain ID;
- observed and expiry timestamps in that same domain;
- proposed WAIT/RESPOND/CONTINUE/YIELD action;
- confidence in [0, 1];
- calibration identifier for the score source.

The gate rejects wrong-clock, future, expired, duplicate, non-monotonic or sequence-gapped candidates. It enforces a legal reciprocal state graph and returns exactly one of:
- **Emitted** — a validated action and decision timestamp/latency;
- **Abstained** — confidence is below the action-specific threshold.

Abstention is not silently rewritten to WAIT or RESPOND. The event ID and sequence are consumed on both abstention and structurally valid invalid-transition outcomes; a later fallback cannot reuse the same pause under a new timestamp.

## Action-specific confidence

Default thresholds:
- WAIT: 0.55
- CONTINUE: 0.76
- RESPOND: 0.82
- YIELD: 0.94

These are conservative initial policy values, not calibrated empirical thresholds. The constructor requires YIELD >= RESPOND >= CONTINUE. Thresholds must be tuned on a calibration split and frozen before a held-out benchmark evaluation.

Why the asymmetry? A false YIELD can truncate an otherwise coherent response, and a false RESPOND can interrupt the partner. These outcomes should have a higher decision threshold than continuing an already-held floor.

## Legal state transitions

- Initial → WAIT or RESPOND.
- WAIT → WAIT, RESPOND or YIELD.
- RESPOND → RESPOND, CONTINUE, YIELD or WAIT.
- CONTINUE → CONTINUE, YIELD or WAIT.
- YIELD → YIELD, WAIT or RESPOND.

A forbidden transition is rejected and consumes that candidate event. The current accepted action remains unchanged.

## Implementation

`crates/domains/symthaea-soma/src/embodiment_turn_control.rs` exports:
- `TurnCandidate`
- `TurnControlThresholds`
- `TurnDecision`
- `TurnAbstention`
- `TurnControlOutcome`
- `TurnDecisionGate`

The gate is bounded to 65,536 candidates per session and requires a host-supplied monotonic time in the exact same clock domain. It reports event-to-decision latency by subtracting timestamps only within that domain. It does not access a wall clock or game clock.

## Tests

Fifteen Rust tests are authored for:
- causal decision and client-clock latency;
- strict YIELD threshold and low-confidence abstention;
- consumption of abstained and invalid-transition event IDs;
- duplicate rejection;
- action threshold ordering;
- finite/bounded confidence;
- clock-domain mismatch;
- future and expired candidates;
- clock regression;
- sequence gap/regression;
- forbidden state transitions;
- valid WAIT → RESPOND → CONTINUE → YIELD sequence;
- capacity exhaustion;
- canonical decimal timestamps.

Tests are authored, not compiled or executed. No detector, microphone, TurnBench adapter, or official evaluation is integrated.

## Benchmark connection

The gate supports the internal TurnBench protocol:
- at most one decision per candidate event;
- no late fallback for the same pause;
- event timestamps remain causal;
- action-specific thresholds and abstention can be evaluated;
- calibration IDs and decision latency are explicit.

The official TurnBench v1 scorer has documented weaknesses and is only a diagnostic lane. Use the one-decision hidden-point protocol internally and migrate to TurnBench v2 if/when its fixed hidden decision-point evaluation becomes available.

## Limitations and next work

- A confidence number is not calibrated merely because it is in [0, 1]. Measure Brier score, log loss and reliability on held-out data.
- The gate does not infer emotion, intention or consent.
- It does not determine microphone permission or authorize speech/audio output.
- It is not a substitute for a causal streaming turn detector.
- Detector output, gate output, spoken output and observed user-facing output need distinct receipts if the controller is connected to live speech.

Next: build an official TurnBench trace adapter and fixed-decision scorer, then integrate a causal detector on frozen, permitted audio data.
