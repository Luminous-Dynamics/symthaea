# MEL-RMI-EXP-001 — expressive reciprocal interaction v2 reference panel

**Date:** 22 November 2026  
**Parent:** #10210 — expressive-performance extension to RMI-Bench  
**Related:** #10088 RMI-Bench, #10155 policy-state isolation, #10190 predictive transparency, #13070 offline Track A runner

## Why this panel exists

RMI Track A v1 deliberately models motif, rhythm, tempo and ensemble role. That is necessary but insufficient for live expressive collaboration. #10210 calls for a separately versioned v2 panel covering velocity/dynamics, articulation, rubato/tempo contour, register/voicing and phrase gesture.

This first reference increment makes the core one-feature causal contract executable for both directions without changing or backfilling the frozen v1 benchmark. It is a synthetic qualification oracle, not a claim that Melothaea itself has passed.

## Representation

Each event is keyed by a persistent event ID and retains five distinct expressive fields:

- velocity (1–127);
- articulation (detached, staccato, tenuto, legato, accent);
- rubato offset in microseconds from the declared score-time event;
- MIDI register/pitch (0–127);
- phrase gesture (neutral, pickup, sustain, release, pause, cadential).

A trial binds source Score digest, policy ID/version, deterministic seed, target dimension, target event IDs and four frozen paired frames per causal direction: control/treatment input and control/treatment response.

## Independent causal directions

The evaluator reports partner-to-agent and agent-to-partner separately. A change in the human-side input must not be conflated with a change in the agent's response, and agent-to-partner sensitivity must not be inferred from the existence of an agent action alone.

Each paired arm must satisfy:

1. Control and treatment contain exactly the same event identities.
2. The input changes only the declared target dimension and only on scoped events.
3. The output response changes only the declared target dimension and only on scoped events.
4. The target feature actually changes in the response (fixed response is not sensitive).
5. Follow/contrast expectations are checked for ordered numeric features; categorical features require explicit any-change semantics.
6. All trial content, policy identity, version, seed and source Score digest are bound by the trial SHA-256.

This panel explicitly rejects imitation of unrelated dimensions, collateral expressive changes, out-of-scope event mutation, missing event identities and stale/tampered trial content.

## Executable work

The isolated Rust 2024 crate at tools/mel-rmi-expressive-v2 contains:

- strict v2 types for the five expressive dimensions;
- independent partner-to-agent and agent-to-partner pair evaluation;
- a content-sealed JSON trial and report CLI;
- one unsealed known-answer velocity/rubato positive fixture;
- twelve deterministic tests covering velocity, rubato, articulation/phrase gesture, fixed response, wrong-feature change, out-of-scope mutation, response relation, duplicate identity and digest tampering;
- a dedicated Rust 1.96 exact-head workflow.

## Benchmark targets (not achieved results)

- 100% acceptance of registered target-specific positive controls.
- 100% rejection of fixed-response and wrong-feature controls.
- 0 collateral expressive changes in the strict single-feature synthetic panel.
- 100% preservation of note-event identities across the paired arms.
- 100% deterministic replay of the same frozen trial and report.
- Partner-to-agent and agent-to-partner pass rates are reported separately, never averaged into a single causal-reciprocity score.

The synthetic panel must not set a universal human timing threshold. Live latency and beat-phase performance will be measured through the runner/CPAL/playback clock chain (#10391) and qualified per device, tempo, velocity and instrument.

## External baseline and claim boundary

The 2026 jam_bot paper models velocity, trains free-form call-and-response at varying tempi and compensates MIDI output latency. It reports beat alignment F1 0.0678 without compensation, median 0.3386 with a fixed upper-bound offset, 0.7638 with average latency and 0.7834 with per-note/per-velocity compensation on a specific player-piano experiment with five non-overlapping windows. This is a direct live-performance baseline, not a universal SOTA number.

A passing synthetic trial establishes that this evaluator can distinguish target-sensitive positive controls from fixed, wrong-feature or collateral-changing outputs under the declared finite contract. It does not show real performer adaptation, improved musicality, perceived expressivity, stylistic generalization or superiority to jam_bot. Those require matched live hardware tests and a preregistered musician study.
