# KAMA-DIALOGUE-001O / VOICE-FDX benchmark v1

## Purpose

Provide a deterministic benchmark harness for the VOICE-FDX-001A turn-taking controller. The benchmark measures interaction-control behavior from explicit scripted event traces and does not treat TTS quality or subjective naturalness as substitutes for stop/yield correctness.

## Evidence separation

Each case reports three independent result families:

1. **trace status** — whether the scripted controller interaction executed or was rejected/incomplete;
2. **semantic status** — whether the scenario-specific turn-taking invariant was established;
3. **latency-budget status** — whether measured latencies stayed within an explicitly declared case budget.

These are deliberately not collapsed into one score.

```text
low latency != correct turn-taking
correct turn-taking != natural conversation
natural conversation != stop/yield correctness
```

## Scenario vocabulary

The initial benchmark supports:

- interruption/yield;
- explicit stop;
- backchannel continuity;
- de-escalation acknowledgement;
- response latency;
- mixed interaction traces.

Cases are exact ordered scripts with nanosecond-domain logical timestamps. Case identity is domain-separated and binds the case ID, session epoch, scenario, ordered event script, and declared latency budgets.

## Measurements

Where the underlying controller emits evidence, the harness records separately:

- user interruption -> output stopped latency;
- explicit stop -> output stopped latency;
- SlowDown -> compliant-plan acknowledgement latency;
- user-floor release -> assistant speech-start latency;
- backchannel count;
- final user-floor / assistant-output / stop-latch / de-escalation state.

A missing required receipt produces `NotEstablished`; it is not silently interpreted as zero latency.

## Runtime rejection

If a scripted event is rejected by VOICE-FDX-001A, the case is reported as `RuntimeRejected` with the exact zero-based rejected step and controller error representation. Runtime rejection is not converted into a poor latency value.

For example, attempting to start assistant speech while the user floor remains occupied is a trace/controller failure, not a slow-response measurement.

## Scenario semantics

### InterruptionYield

Requires an interruption receipt and assistant output to be silent after the scripted yield acknowledgement.

### ExplicitStop

Requires a stop receipt, a latched stop state, and silent assistant output after acknowledgement.

### BackchannelContinuity

Requires a recorded backchannel while assistant output remains active; a backchannel may not silently become an interruption/yield.

### DeescalationAcknowledgement

Requires a de-escalation receipt and a cleared de-escalation latch after the compliant-plan acknowledgement.

### ResponseLatency

Requires a measured response latency from user-floor release to assistant speech permit and a released user floor.

## Latency budgets

Budgets are case metadata, not universal human-comfort thresholds. They allow reproducible comparisons under a declared benchmark policy but do not establish a clinical, accessibility, product-quality, or human-naturalness threshold.

A measured latency may exceed its budget while semantic control remains correct. Conversely, a fast interaction with missing semantic evidence is not a semantic pass.

## External benchmark alignment

This harness is the internal deterministic substrate for issue #4773. Later adapters can map external full-duplex evaluation traces into this evidence grammar while preserving the original benchmark's own population, noise conditions, task definitions, and scoring semantics.

## Nonclaims

This tranche does not implement or qualify:

- ASR or end-of-turn classification;
- TTS or expressive prosody quality;
- acoustic echo cancellation;
- real-person voice likeness authorization;
- semantic dialogue generation;
- subjective human naturalness;
- physical authority;
- universal latency targets.

No PASS is established until an exact-head qualifier executes successfully against the frozen source subject.
