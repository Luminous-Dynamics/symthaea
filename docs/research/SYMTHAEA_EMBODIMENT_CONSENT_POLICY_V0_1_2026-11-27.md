# Symthaea Embodiment Consent and Presentation Policy v0.1

**Date:** 2026-11-27  
**Status:** implementation proposal and Rust module added; tests authored but not executed

## Central design shift

An embodiment cue is a **proposal for expression**, not permission to express it.

A listening pulse, spoken response, haptic acknowledgement, spatial effect, or public display can affect another person. The fact that the cognitive system emitted a cue must never implicitly authorize every renderer, output channel, destination, or intensity.

The architecture therefore separates five things:

1. **Observation:** what a source reported, with modality and provenance.
2. **Semantic event:** what Symthaea proposes to communicate.
3. **User permission:** which form, output channel, destination, and intensity are allowed for a bounded session and time window.
4. **Policy decision:** whether this specific cue may be presented now.
5. **Presentation outcome:** what the renderer actually did, independently observed and recorded.

The event stream is not an authority system. Canonical projection preserves meaning but does not grant permission. A presentation permit is not proof of execution, and an execution receipt is not proof that the user understood or consented to the experience.

## Current implementation

The Rust module at crates/domains/symthaea-soma/src/embodiment_policy.rs adds a renderer-independent policy gate.

- A new policy instance starts paused, reduced-motion enabled, with zero grants.
- Grants are session- and monotonic-clock-domain-scoped, expire, and name output channel, destination, allowed forms, and maximum intensity.
- The same output scope is replaced on grant refresh; an older broader grant cannot remain active behind a narrower replacement.
- Pause, event cancellation, and per-channel revocation deny subsequent evaluations.
- Local-sensitive cues are restricted to local-private destinations.
- Restricted cues are denied by this policy version.
- A visual request is static while reduced motion is enabled.
- Requested output intensity is clamped to the grant maximum without rewriting the semantic cue or its epistemic metadata.
- The function returns a presentation permit and performs no rendering, sensor capture, network transmission, or actuation.

The policy is intentionally small and synchronous. The production caller must serialize policy updates and presentation checks in a synchronization domain that makes pause/revocation visible before a subsequent output tick.

## Trust and safety boundaries

This module does **not**:
- authenticate the person or process that created a consent grant;
- prove that a user was informed, free from coercion, or understood the consequences;
- authorize camera or microphone capture;
- authorize remote transmission merely because a visual or audio output was granted;
- provide physical actuator safety, emergency-stop, collision avoidance, or force limits;
- establish that a renderer complied with a permit;
- provide cryptographic signatures or tamper-evident audit storage.

Those require separate capabilities and independent evidence. In particular:

- Sensor permissions must be independent from output permissions.
- Remote transmission must be a separate, explicit destination grant.
- Physical actuation needs a dedicated safety controller, independent emergency stop, rate/force/workspace limits, and fail-safe state. A visual embodiment permit must never be reused as an actuator permit.
- Consent provenance should eventually bind the grant to a user-controlled interaction, policy version, session, exact capabilities, expiry, and revocation identity. Authentication and storage belong in the trusted consent boundary, not in this policy module.
- A future presentation receipt should distinguish *permitted*, *submitted to renderer*, *observed on output*, and *confirmed stopped*. These are different states.

## State machine and flow

```text
untrusted source event
        |
        v
event schema + session/clock/freshness/replay validation
        |
        v
canonical semantic cue (meaning and provenance preserved)
        |
        v
default-deny presentation policy
  | deny                         | permit
  v                               v
no output                    renderer adapter
                                  |
                                  v
                         independently observed outcome
                                  |
                                  v
                         presentation receipt
```

An event is accepted only in its own validated sequence and clock domain. The policy then checks the cue against a current, exact grant. A renderer must re-check cancellation and policy state immediately before the visible/audio/haptic output tick. A pause or revocation must take precedence over queued presentation work.

## Benchmark contract

The policy belongs in E0 correctness, not only in a qualitative ethics checklist.

Required deterministic negative controls:
- no grant;
- wrong session or clock;
- expired grant or event;
- ungranted form;
- ungranted channel or destination;
- LocalSensitive cue to shared/remote destination;
- Restricted cue to any destination;
- event cancelled after it was queued;
- pause or channel revocation between decision and output;
- non-finite or out-of-range intensity;
- reduced-motion enabled;
- grant narrowed after a broader grant.

Targets (not results):
- zero unauthorized permits across the complete deterministic negative-control corpus;
- 100% agreement with expected allow/deny decisions;
- local pause/revocation effective before the next renderer update;
- p99 policy evaluation below 1 ms on a named device at 10,000 decisions/s, measured rather than inferred;
- no semantic or provenance change caused by intensity clamping or motion reduction;
- no camera/microphone activation from any presentation grant.

Use fault injection to test a stale permit, queue delay, renderer restart, process pause, and concurrent revocation. The exact policy version, consent-grant fixture, trace, renderer version, hardware, and output observation must be recorded.

## Next steps

1. Validate the module with the workspace's pinned Rust toolchain and exact-head tests.
2. Authenticate consent grants at a separate user-controlled boundary; add deterministic provenance fixtures.
3. Define a renderer submission/outcome receipt without implying that a permit proves output.
4. Integrate the policy into the first Living Constellation renderer, with pause/revocation checked at the last responsible moment.
5. Keep sensors and physical actuators behind independent permission and safety contracts.
