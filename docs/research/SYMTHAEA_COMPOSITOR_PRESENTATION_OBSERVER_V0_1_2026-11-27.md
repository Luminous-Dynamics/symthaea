# Symthaea Compositor Presentation Observer v0.1

**Date:** 2026-11-27  
**Status:** platform research and qualification protocol; no observer implementation or hardware result claimed  
**Parent:** presentation outcome receipts and interaction-trace verification

## 1. Objective

Close the evidence gap between "the renderer accepted ECS entities" and "a particular frame was actually presented by the user's display system".

The renderer must not certify its own output. A qualified observer should obtain presentation feedback from a compositor, display subsystem, or another independently qualified output source and bind that feedback to the exact semantic event and authorization that produced the frame.

The observer can establish that a frame update was reported as presented to an output. It cannot establish that a human looked at the screen, understood the frame, or experienced it. Symthaea must not claim those things from presentation telemetry.

## 2. Evidence vocabulary

Keep these states distinct:

1. **Compiled:** the immutable semantic scene and geometry were produced.
2. **AdapterAccepted:** the Bevy adapter accepted the ECS entities.
3. **BackendSubmitted:** a frame/update was submitted to the window/compositor presentation path.
4. **PresentedToOutput:** a qualified compositor/display source reported that the associated update was presented.
5. **Discarded:** the compositor/display source reported that the update was never presented.
6. **StopRequested:** the host requested pause, cancel, revocation, or replacement.
7. **SafeFramePresented:** an authorized blank/neutral frame or equivalent safe output was independently reported as presented.
8. **SurfaceDetached:** a separately qualified output source reported that the relevant surface was unmapped or detached.
9. **OutcomeUnknown:** the observer did not provide enough evidence to resolve the terminal display state.
10. **Failed:** a known presentation or observer failure was reported.

Existing receipt phases may continue to expose Submitted, Observed, and Stopped, but their meanings must be strict:
- Submitted requires the adapter/backend submission acknowledgement.
- Observed requires qualified PresentedToOutput evidence.
- Stopped requires qualified evidence that the prior content is no longer presented, such as SafeFramePresented or a platform-specific, independently qualified detachment result.
- StopRequested is never itself Stopped.
- Discarded is never Observed.
- Observer loss, absent feedback, or an incomparable clock remains OutcomeUnknown.

"Presented to output" is a display-pipeline statement, not proof of human attention or comprehension.

## 3. Linux Wayland: first implementation target

The stable Wayland presentation-time protocol (wp_presentation) is a concrete starting point.

For a wl_surface commit, a client can request presentation feedback. The compositor later reports one of two outcomes:
- presented: the update was displayed, with a presentation timestamp, refresh interval, sequence counter, and flags;
- discarded: the update was not displayed, for example because it was superseded or its surface was destroyed.

The protocol also supplies a presentation clock ID. The reported flags distinguish properties including vsync synchronization, hardware-provided clock information, hardware completion signalling, zero-copy, fixed refresh, and variable refresh. Not every compositor/output is required to provide hardware-backed timing.

Reference: [Wayland presentation-time protocol](https://wayland.app/protocols/presentation-time).

### Required Wayland behaviour

- Detect whether the compositor exposes the required protocol/version; never assume availability.
- Request one feedback object for each frame update that needs qualification.
- Correlate the feedback with a unique local presentation-request ID and the exact backend submission.
- Record presented and discarded as separate terminal outcomes.
- Record the selected output, refresh interval, sequence counter, clock ID, and all available flags.
- Record whether timestamp and completion are hardware-backed or compositor-estimated.
- If the compositor restarts, the surface is destroyed, the observer disconnects, or feedback is missing, preserve OutcomeUnknown unless an explicit terminal outcome is received.
- Do not store raw screen pixels in the receipt ledger.

A presentation timestamp is meaningful only in the clock domain identified by the protocol. Do not subtract it from the source event's monotonic timestamp unless a measured clock relationship with a bounded error is available. Store both clocks and their calibration metadata; otherwise report presentation latency as unavailable.

## 4. Other output backends

The observer abstraction should make platform guarantees explicit rather than pretend every backend provides the same evidence.

| Backend | Potential evidence source | Qualification notes |
|---|---|---|
| Linux / Wayland | wp_presentation presented/discarded feedback | First target; protocol presence, compositor support, clock and hardware flags must be recorded |
| Vulkan | VK_KHR_present_id + VK_KHR_present_wait; VK_GOOGLE_display_timing where supported | Present-wait completion is a presentation-completion signal; timing precision and support are device/driver dependent |
| Windows | DXGI frame statistics | PresentCount / PresentRefreshCount can describe frames presented to a monitor; do not equate API Present calls with monitor presentation |
| Other platforms | platform compositor/display timing APIs | Add only after a platform-specific capability and failure matrix exists |

References:
- [Vulkan present-wait extension](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_present_wait.html)
- [Vulkan window-system integration](https://docs.vulkan.org/spec/latest/chapters/VK_KHR_surface/wsi.html)
- [Microsoft DXGI_FRAME_STATISTICS](https://learn.microsoft.com/en-us/windows/win32/api/dxgi/ns-dxgi-dxgi_frame_statistics)

These APIs do not all expose the same guarantees. A common interface must not erase the source's confidence, clock domain, hardware flags, or limitations.

## 5. Observer trust boundary

The observer's evidence source must be outside the renderer's self-report path.

Minimum requirements:
- compositor/display feedback is the authoritative source for PresentedToOutput;
- the adapter cannot directly construct an Observed or Stopped receipt;
- the observer records its own build/version and platform backend;
- observer output is append-only or tamper-evident at the evidence boundary;
- missing, malformed, duplicated, stale, or contradictory feedback fails closed;
- the source output ID and presentation request ID are unique within the session;
- the observer does not receive camera/microphone permissions and does not capture raw media;
- policy rechecks and stop requests remain in the host's live authorization path.

A same-process callback may be a useful integration step, but it is not automatically an independent trust boundary. Prefer a separate observer process or a separately permissioned component that consumes compositor feedback and signs/attests its own receipt when a platform trust root exists. If that trust root is absent, report the observer as software-independent but not hardware-attested.

## 6. Binding presentation to the semantic event

Each observation record should bind at least:

- session ID and monotonic event sequence;
- semantic event ID and canonical-cue digest;
- grant ID, policy version, form, channel, destination, and privacy class;
- output/surface ID and backend identity;
- backend submission ID and observer presentation-request ID;
- presentation result: presented, discarded, failed, or unknown;
- observer clock domain and presentation timestamp;
- clock calibration source, offset/drift interval, and uncertainty when cross-clock latency is computed;
- refresh interval, sequence counter, vsync/hardware-completion/zero-copy flags where available;
- observer build, OS/compositor/backend versions, and capability profile.

Do not copy raw audio, video, sensitive semantic targets, or private evidence references into these records. The existing cue digest binds semantic content without exposing it to the observer.

## 7. Revocation and stop protocol

Revocation can occur while a frame is in flight. A local policy denial does not retroactively erase a frame already queued in the compositor.

The host should:
1. invalidate the active permit immediately;
2. prevent new frame submissions under that permit;
3. submit a safe neutral output or detach the surface according to the platform contract;
4. retain the exact last submitted frame and policy version;
5. await the observer's terminal feedback;
6. record Stopped only when the prior output's cessation is independently established.

Important fault cases:
- old frame submitted before revocation but presented after revocation;
- safe frame submitted but discarded;
- safe frame presented on a different output than the original;
- display topology changes during an active session;
- compositor restart or observer loss during stop;
- surface destroyed before presentation feedback;
- stale feedback arrives after a newer frame;
- two frames are reordered or a frame is duplicated;
- presentation timestamp and source timestamp cannot be compared safely.

If a post-revocation frame is reported presented, record a policy/output violation. Do not hide it by using the later stop receipt as the only outcome.

A discarded safe frame does not prove the old content disappeared. A stop request does not prove the display cleared. An output detachment event counts only if that platform-specific event has been qualified to establish cessation.

## 8. Qualification and fault injection

### Deterministic tests

- one submitted frame receives one presented outcome;
- one submitted frame receives one discarded outcome;
- duplicate feedback is rejected;
- stale feedback cannot overwrite a newer terminal state;
- missing feedback remains OutcomeUnknown;
- mismatched surface/output/request IDs are rejected;
- clock-domain mismatch prevents latency computation;
- clock-calibration uncertainty is propagated;
- hardware flags are preserved rather than inferred;
- observer restart cannot replay a terminal receipt;
- stop requested with safe frame discarded does not become Stopped;
- stop requested with safe frame presented can become Stopped when the original surface/output binding is preserved;
- presentation after revocation is explicitly detected.

### Runtime tests

On named compositor, GPU, driver, display, and refresh profiles, measure:
- submit-to-present p50/p95/p99;
- source-event-to-presented-frame latency, when clocks are comparable;
- discarded-frame rate;
- observer feedback loss;
- revocation-to-safe-frame-presented latency;
- frame reordering/duplication;
- VRR and multi-monitor behaviour;
- CPU overhead and power cost.

### Hard gates

- zero renderer-self-certified Observed/Stopped receipts;
- zero duplicate or contradictory terminal observations accepted;
- zero false Stopped outcomes when a safe frame was discarded or the observer was unavailable;
- zero post-revocation presentation events hidden from the trace;
- 100% clock-domain provenance for every measured latency;
- all unsupported observer capabilities explicitly reported.

Targets are not measured results. No specific stop-latency threshold should be declared until the chosen compositor and device profile have been measured.

## 9. Implementation sequence

1. Define a renderer-neutral observer event type and lifecycle state machine in Rust.
2. Add JSON schema and positive/negative fixtures to the existing interaction trace bundle.
3. Build a Linux Wayland capability probe and wp_presentation feedback adapter.
4. Route observations through a separate receipt writer; remove any ability for the Bevy adapter to mint Observed/Stopped.
5. Add the fault-injection cases above, including delayed presentation after revocation.
6. Qualify on a named Wayland compositor, GPU/driver, display, and refresh profile.
7. Add Vulkan and Windows backends only after their evidence differences are represented explicitly.

## 10. Qualification status

This document is a protocol proposal. No Wayland, Vulkan, DXGI, compositor, display, driver, or hardware integration has been implemented or tested. The existing Bevy acknowledgement remains ECS acceptance only. The observer must not be presented as implemented until the backend feedback is collected, correlated, fault-injected, and tested on a named runtime.


## 12. Source implementation increment (2026-11-27)

A renderer-neutral Wayland feedback correlation ledger now exists at:

`crates/domains/symthaea-soma/src/embodiment_presentation_feedback.rs`

It:
- binds one feedback request to session, event, cue digest, grant, policy version, surface, frame, backend commit and compositor clock;
- accepts exactly one Presented or Discarded terminal result for a registered request;
- rejects mismatched surface/frame/clock identities and presentation timestamps earlier than the submission sample;
- preserves output ID, refresh interval, sequence counter and raw protocol flags, including unknown future bits;
- creates `OutputObservation(Compositor, Started)` only for a Presented result;
- creates no observation for Discarded and never emits Stopped;
- bounds outstanding requests and completed records;
- permits a host to record OutcomeUnknown while leaving a request pending for delayed feedback resolution.

The receipt ledger now rejects `AdapterCallback` as evidence for both Observed and Stopped. Compositor and external-observer sources remain distinct from adapter submission acknowledgements.

Ten observer regression tests and one additional receipt-source regression test are authored, not compiled or executed. The Wayland protocol adapter, protocol capability discovery, real compositor callback, separate observer process, receipt persistence, and named-device runtime qualification are not implemented yet.

The observer proves a compositor-reported presentation event, not that a human looked at or understood the frame. This increment intentionally does not claim that presentation of a later frame proves the earlier frame has stopped; Stopped remains a separately qualified evidence requirement.
