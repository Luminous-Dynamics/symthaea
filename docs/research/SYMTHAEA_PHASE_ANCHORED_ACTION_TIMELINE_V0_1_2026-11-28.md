# Symthaea phase-anchored action timeline v0.1 — 2026-11-28

## Research thesis

A conversational embodiment should not let each renderer independently decide when speech, lip motion, gaze, gesture, expression, posture and locomotion occur. The system should compile a canonical communicative event into a small, clock-bound action schedule. Renderer-specific animation becomes execution of a schedule, not an independent source of intent.

This extends phase-aware action conditioning research such as ActAvatar (CVPR 2026), which separates global prompt context from temporally anchored phase-specific control to reduce interference between audio-driven lip synchronization and text-driven action control. Symthaea's contribution is the additional separation of **intent, authorization, scheduling, actuation, and observed output**.

## Contract implemented

Source: `crates/domains/symthaea-soma/src/embodiment_action_timeline.rs`.

- Typed channels: speech, viseme, gesture, gaze, expression, posture, locomotion and spatial cue.
- Typed phases: prepare, onset, sustain, release and recovery.
- Each segment binds to a single monotonic source clock, source event, canonical cue digest and payload digest.
- Schedule order is deterministic; segment IDs are unique; time ranges are bounded by the parent expiry.
- A channel cannot overlap itself in this version. Cross-channel synchronization requires an explicit group that includes multiple channels.
- Cancellation is bound to the exact plan digest and source clock. Supersession requires an explicit replacement plan ID.
- Maximum v1 schedule duration is 60 seconds; maximum segment count is 256.

The digest is a content-integrity identifier, not a signature, trusted timestamp, authorization, media provenance or proof of actuator execution.

## Paradigm shift

**From animation clips to causal action contracts.** The action timeline is a typed plan with explicit phases, synchronization groups, source clock and expiry. A "speaking" visual should be derived from the same scheduled utterance as the audio, rather than inferred from a renderer's local animation timer.

**From output completion claims to independent observation.** Timeline cancellation means the plan was cancelled. It does not mean audio stopped, the compositor removed a frame, or a motor actuator halted. Those require independent output observers and receipts.

**From one monolithic avatar model to interoperable action channels.** A speech backend, viseme renderer, gesture controller and spatial display can consume the same event/phase schedule while preserving different visual grammars.

## Next integration steps

1. Compile canonical turn/action intent into this schedule only after the existing one-candidate/one-decision turn gate.
2. Bind speech generation chunk IDs and viseme frame timestamps to the same monotonic clock; do not align through wall-clock timestamps.
3. Add a runtime scheduler with cancellation checkpoints at chunk boundaries and a bounded maximum stop latency.
4. Connect cancellation to the existing receipt lifecycle; only an independent output observer may produce Observed or Stopped.
5. Add matched counterfactual tests: same canonical cue across all seven forms; interrupt during Prepare, Onset, Sustain and Release; policy revocation during cross-channel output; stale clock; out-of-order chunks.
6. Benchmark both perceptual continuity and interruption safety. A beautiful gesture that continues after a user barge-in is a failure.

## Benchmarks and targets (proposed, not results)

| Metric | Initial target |
|---|---:|
| Same cue digest preserved across all forms | 100% |
| Phase/segment schedule deterministic on replay | 100% |
| Unauthorized or expired segment submitted | 0 |
| Post-cancellation new segment dispatch | 0 |
| p95 interrupt-to-audio-stop | <=150 ms |
| p95 interrupt-to-visual-stop | <=100 ms |
| Speech/viseme A/V offset | <=80 ms p95 |
| Cross-channel phase-order violations | 0 |
| Human-rated action/cue temporal fit | >=4.3/5 |
| Task success regression vs audio-only | <=1 pp |

Latency targets are engineering hypotheses. They must be measured end-to-end on named hardware and transports and reported with p50/p95/p99, not inferred from scheduler timestamps.

## Qualification status

The deterministic contract and authored tests are committed at source level. The module has not been compiled or executed. No renderer consumes the schedule yet, no real audio/viseme clock is bound, no actuator is controlled by this plan, and no physical/output stop latency has been measured.
