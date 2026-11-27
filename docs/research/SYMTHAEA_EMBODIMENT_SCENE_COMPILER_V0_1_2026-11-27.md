# Symthaea Embodiment Scene Compiler v0.1

**Date:** 2026-11-27  
**Status:** deterministic CPU-side scene compiler and regression tests authored; graphics renderer not yet integrated

## Purpose

The first visible embodiment must be testable without requiring a GPU, camera, microphone, network, or windowing runtime. The scene compiler turns a validated canonical cue and an opaque visual presentation permit into a declarative list of visual primitives.

It is a contract between the semantic/policy core and a later renderer (Bevy first, puppet renderer afterward). The scene compiler does not animate, display, speak, capture sensors, or control actuators.

## Integrity contract

Compilation is allowed only if the permit and canonical cue agree on:
- session, clock domain and event identity;
- sequence and event kind;
- semantic key;
- source identity/build;
- privacy class and event expiry;
- the complete canonical-cue digest.

The cue digest is deterministic, domain-separated BLAKE3 over the serialized complete canonical cue. It binds the payload and provenance as well as the event ID and semantic category. It is a content-integrity link, not a digital signature, source authentication, or evidence that a source was honest.

A modified AttentionFocus target with the same event ID and semantic key is rejected. The presentation receipt also carries the same canonical-cue digest.

## Scene grammar

Each form receives the same canonical event and meaning, but uses a different visual vocabulary:

| Form | Core | Connection | Marker |
|---|---|---|---|
| Living Constellation | star node | filament | orbit |
| Puppet Theatre | marionette joint | puppet thread | label anchor |
| Mycelial Being | hypha node | hypha | branch |
| Swarm | particle | particle trail | marker |
| Living Instrument | resonator | string line | marker |
| Hearth Companion | ember | hearth arc | marker |
| Cognitive Mandala | mandala center | radial line | marker |

Semantic mappings:
- Listening: quiet core and optional connection.
- Speaking: stronger core and flowing connection, only when the source says speaking is active.
- WAIT: stable low-amplitude turn marker.
- RESPOND: localized turn pulse.
- CONTINUE: sustained connection motif.
- YIELD: outward/radiating handoff motif.
- Backchannel: marker weighted only by the source-reported intensity.
- Attention: focus marker and optional source-reported strength.
- Expression intent: a presentation marker, not an emotion classifier.
- Uncertainty: branching alternatives without mapping uncalibrated uncertainty into a probability gauge.
- Evidence: provenance-linked marker with accessible status text; no colour/glow-as-proof encoding.
- User control: static control marker.

The compiler clamps all output weights to the permit's effective intensity and forces every element to Static when reduced motion is enabled. Elements with no source-supplied calibrated weight retain None; renderer defaults must not be reported as source confidence.

## Privacy and accessibility

The scene frame carries event/source/build/clock/grant IDs, sequence, semantic key, canonical-cue digest, epistemic status, privacy class, event time and expiry, plus the accessibility description produced by CanonicalCue. Raw evidence-reference identifiers and semantic target payloads are not copied into the renderer-facing frame; their integrity remains bound by the canonical-cue digest. Sensitive subjects are redacted from generic accessible text. Restricted data cannot be compiled because no policy permit may be issued for it.

Essential state must be communicated through shape and text, not colour alone, animation alone, sound alone, or gaze. Reduced-motion mode makes all primitives static.

## Code and tests

- Implementation: crates/domains/symthaea-soma/src/embodiment_scene.rs
- Exported as symthaea_soma::embodiment_scene.
- Eight authored Rust regression tests cover deterministic compilation, all seven forms, reduced motion, privacy-aware accessibility and no raw-reference disclosure, uncertainty branching, evidence-status honesty, channel mismatch, and modified-payload rejection.
- The tests are authored, not compiled or executed. No Bevy renderer, frame timing, accessibility user study, or hardware profiling is claimed.

## Next renderer integration

The Bevy adapter should:
1. accept only a scene frame produced from an Authorized/Submitted lifecycle and the exact cue digest;
2. render these primitives without changing the semantic key, epistemic status, provenance or accessibility text;
3. record the adapter acknowledgement as Submitted, then compositor/frame observation as Observed;
4. record an observed stop after pause/cancel, rather than treating a stop command as proof;
5. preserve camera/microphone off and no actuator/network dependencies for the initial desktop demo.
