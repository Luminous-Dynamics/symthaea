# Symthaea Real-Time Interaction Contract v1

**Status:** Architecture/measurement freeze; no runtime behavior change  
**Program:** INT-000R  
**Parent architecture:** `SYMTHAEA_INTERACTION_RUNTIME_ARCHITECTURE_V1.md`  
**Base:** `architecture/interaction-runtime-v1-r1@a187d1d7dade27fc6854dfa6a840396f4cf2b48c`

## Purpose

Freeze the responsiveness, scheduling, cancellation, backpressure, and latency-measurement semantics that every interactive Symthaea surface must obey.

The governing principle is:

> Slow thought must never imply a slow nervous system.

A deep cognitive operation may legitimately take milliseconds or seconds. That must not make keypresses, cancellation, sensory ingestion, voice interruption, state provenance, status inspection, or authority/evidence handling unresponsive.

This document narrows INT-000. It does not change cognition, add autonomous ticking, alter execution authority, or claim measured latency results.

## Scope

This contract applies to:

- the canonical `SymthaeaRuntime` planned by INT-003;
- Unix IPC and service request handling;
- the Ratatui/Crossterm TUI;
- Leptos/WebSocket clients;
- Bevy/gRPC research visualization;
- microphone/STT ingress;
- TTS/vocal-tract output;
- future phone, Zellij, desktop, robotics, and embodied clients.

It does not prescribe a single transport or renderer.

## Existing foundations to preserve

The current repository already contains several correct real-time patterns that should be generalized rather than replaced.

### Live microphone ingress

`src/perception/audio_stream.rs` already separates the audio callback from cognition:

```text
cpal callback
    -> lock-free ring buffer
    -> background STT/HDC worker
    -> bounded channel
    -> nonblocking latest-value drain by perception
```

The callback does not block on cognition. The worker may drop stale derived sensory state when the consumer falls behind. That is the expected sensory-plane shape.

### Streaming language generation

The language stack already exposes streaming generation through `LLMBackend::generate_streaming` and `LLMOrgan::query_streaming_async`, with streaming implementations in multiple backends.

New interaction work should route these incremental results into the runtime semantic-event plane rather than inventing a second token-stream abstraction.

### Web and gRPC telemetry

WebSocket and gRPC streaming paths already exist. Their transport machinery should be adapted to shared runtime semantics rather than replaced solely for architectural uniformity.

## Current hazards motivating this contract

The following current patterns are transition hazards. They are not claims that the components are unusable.

1. `src/bin/symthaea.rs` holds the service's `Symthaea` mutex across the full asynchronous query processing call.
2. Service `Speak` currently invokes synchronous speech handling inside the async request path.
3. `src/shell/ipc_client.rs::subscribe_metrics_watch` currently returns a local watch receiver but does not by itself establish a continuously reading subscription pump.
4. The shell uses a synchronous Crossterm poll loop with a 100 ms tick and some `Runtime::block_on` paths.
5. `src/voice/live_voice.rs::speak_async` pre-synthesizes a complete utterance before the background audio-push thread begins.
6. The live audio output ring can buffer approximately seconds of audio, so stopping synthesis is not equivalent to immediate silence.
7. `VoiceSynthesisChannel` uses unbounded request/response channels despite describing latest-wins behavior and declaring a completed-audio cap.
8. The Bevy dashboard receives telemetry through an unbounded standard MPSC channel and drains arbitrary backlog during an update.
9. The browser UI decodes complete base64 mental-movie frame sets on receipt of ordinary telemetry.
10. Current telemetry paths sometimes treat a lagged subscriber as a transport failure rather than an explicit recoverable sequence gap.

These are the concrete seams this contract constrains.

## Responsiveness theorem

For one runtime identity:

```text
interactive control path
+ bounded state/event/sensory queues
+ separately durable evidence path
+ cooperative cancellation
+ single cognitive authority
        ↓
responsive interfaces even during slow cognition
```

The following is prohibited:

```text
slow cognition
    -> owns every lock / queue / renderer
    -> blocks status
    -> blocks cancellation
    -> blocks input
    -> blocks evidence
```

## Scheduling classes

The runtime SHOULD classify work by semantic urgency. The initial classes are architectural categories, not yet a required Rust enum.

### Critical

Examples:

- cancellation and interruption;
- audio device callback safety;
- authority revocation;
- shutdown/stop signals;
- evidence-integrity failure notification;
- bounded-queue overload signals that affect safety.

Properties:

- must not wait behind ordinary cognition;
- must not perform unbounded allocation or blocking I/O in real-time callbacks;
- may preempt or invalidate lower-priority work;
- failures must be explicit.

### Interactive

Examples:

- keypress and pointer input;
- session control;
- response-start/delta delivery;
- current state/status inspection;
- user-visible connection/provenance change;
- action confirmation UI.

Properties:

- low bounded latency is more important than processing every intermediate state;
- must remain responsive during long cognitive operations;
- may use latest-wins semantics for render state.

### Realtime

Examples:

- microphone-derived auditory state;
- VAD/voice interruption signals;
- audio playback chunks;
- high-rate visual/sensor state required for interaction.

Properties:

- bounded queues only;
- explicit drop/coalesce policy;
- cannot create unbounded backlog;
- no blocking work on device callbacks.

### Background

Examples:

- long language generation after first response has begun;
- expensive research telemetry;
- mental-movie production;
- indexing and embedding work;
- optional introspection projections;
- maintenance not required for immediate interaction.

Properties:

- cancellable where practical;
- may yield to interactive/realtime work;
- must not own global interface locks while blocked.

### Evidence

Examples:

- authorization decisions;
- execution receipts;
- audit records;
- scientific measurement records;
- qualification artifacts.

Properties:

- not latest-wins;
- loss must not be hidden as normal telemetry drop;
- failure behavior is defined by the owning evidence/authority protocol;
- must remain logically separate from UI render state.

## Runtime-plane QoS contract

INT-000 defines state, semantic-event, sensory, and evidence planes. This document freezes their real-time behavior.

### State plane

Recommended primitive family: latest-value watch/snapshot.

Required properties:

- bounded memory independent of producer rate;
- slow subscribers may skip intermediate values;
- every externally rendered state carries explicit provenance;
- a subscriber can identify the runtime and latest semantic cursor represented;
- reconnect/resnapshot is normal, not exceptional failure.

A state consumer must prefer the newest valid state over replaying stale UI backlog.

### Semantic event plane

Recommended primitive family: bounded multi-subscriber event stream.

Required properties:

- each ordered event carries runtime-local monotonic `EventSeq`;
- loss is detectable as a sequence gap;
- slow consumers cannot stall the runtime;
- a lagged client transitions to explicit gap/recovery handling rather than silently assuming continuity;
- large binary payloads remain referenced by identity, not embedded in the normal event stream.

### Sensory plane

Required properties:

- bounded buffers at every handoff;
- modality-specific drop policy;
- source timestamps are metadata, not ordering authority;
- high-rate sensors cannot starve interactive, authority, or evidence work;
- raw/high-volume payloads are not duplicated into every UI subscriber by default.

Suggested default semantics:

| Sensory class | Default overload policy |
| --- | --- |
| acoustic feature/HDC snapshots | latest-wins/coalesce |
| camera/screen frames | drop stale frames, preserve newest key/reference frame |
| pointer/device motion | coalesce where semantics permit |
| finalized human utterance | do not silently drop; promote to semantic event |
| safety/interrupt sensor signal | critical path; explicit failure if undeliverable |

### Evidence plane

Required properties:

- evidence retention is never inferred from successful UI delivery;
- state/event overload cannot discard authority/evidence records implicitly;
- if durable evidence is mandatory for an effect, the effect protocol defines fail-closed behavior;
- evidence ordering/identity is preserved by the evidence subsystem, not a display timestamp.

## Backpressure invariants

### RT-I1 — no unbounded interactive queues

A queue reachable from normal UI, voice, sensory, or streaming response traffic must have a documented bound or a single-latest-value semantic.

### RT-I2 — stale state may be dropped, semantic loss may not be hidden

State can latest-win.

Events may be dropped only with detectable sequence discontinuity.

Evidence may be lost only through an explicit failure path defined by its owning protocol.

### RT-I3 — producer rate cannot define memory growth

A producer that runs faster than a consumer must not cause memory consumption to grow without a configured bound.

### RT-I4 — overload policy is part of the API

Each bounded queue must document one of:

- latest-wins;
- drop-newest;
- drop-oldest;
- coalesce;
- block outside a realtime/interactive path;
- fail closed;
- spill to explicitly durable storage.

"Bounded" without a defined overload policy is incomplete.

## Cancellation contract

Cancellation is a first-class runtime signal, not a UI convention.

A turn/session implementation SHOULD support hierarchical cancellation so one user interruption can wake all cooperating work belonging to that turn.

Conceptual propagation:

```text
user interruption
    -> TurnId cancellation
        -> cognition/generation task
        -> response delta producer
        -> phrase planner
        -> TTS/vocal-tract renderer
        -> audio generation queue
        -> presentation state
```

Cancellation must not grant authority or mutate evidence history retroactively.

### Cancellation generations

Long-lived streaming outputs SHOULD carry a generation/utterance identity.

When utterance generation `N` is invalidated, downstream consumers MUST reject stale `N` output even if it arrives after generation `N+1` has started.

This is required because cancellation and buffered I/O race in real systems.

## Voice interruption contract

Barge-in is successful only if all of the following occur:

1. speech/activity is detected for the new user utterance;
2. the active assistant turn is marked interrupted/cancelled;
3. stale response generation stops or its future output is ignored;
4. stale voice synthesis stops or its future chunks are ignored;
5. already queued stale playback is flushed/invalidated within a bounded interval;
6. the new utterance enters cognition with a new `UtteranceId`;
7. acoustic and semantic representations of that utterance retain correlated identity.

`stop_synthesizing()` without invalidating already buffered audio is not sufficient.

### Voice buffering

The voice renderer SHOULD use a small explicit ahead-of-playback budget rather than treating seconds of queued audio as harmless.

Initial engineering range for experimentation: approximately 100-250 ms of playable audio ahead of the device.

This is not a final product target. It is intended to make interruption measurable and to avoid multi-second stale playback.

## TUI event-loop contract

The TUI SHOULD evolve from synchronous poll/tick ownership toward independent asynchronous sources:

```text
terminal events
runtime state changes
runtime semantic events
render cadence
maintenance cadence
cancellation/shutdown
        -> select/reducer
        -> render
```

Required properties:

- no foreground `block_on` on ordinary UI interaction paths;
- rendering cadence is independent of cognition cadence;
- a keypress does not wait for telemetry refresh;
- expensive completion/context work may be debounced/cancelled;
- terminal redraws may skip unchanged state;
- disconnect must not fabricate live cognition.

## Service ownership contract

The planned `SymthaeaRuntime` remains the sole mutable cognitive authority, but interface liveness must not depend on locking that mutable owner for every observation.

The runtime SHOULD expose an immutable latest snapshot independently of long-running mutable operations.

Conceptually:

```text
                 commands/observations
clients  ------------------------------> runtime owner
   ^                                        |
   |                                        |
   +----- latest immutable state -----------+
   +----- ordered semantic events ----------+
```

A long `process()` operation may serialize cognitive mutation while status/cancel/state delivery remain live through separately owned synchronization primitives.

## IPC duplex contract

A framed duplex connection SHOULD have one logical reader responsible for parsing inbound frames.

Responses and unsolicited events/state updates are routed after parsing; foreground request methods must not race to consume the same reader.

Required properties:

- subscribe actually sends a subscription request;
- subscription updates arrive without requiring another foreground request;
- request/response correlation is explicit;
- unsolicited state/events cannot be mistaken for the response to a different request;
- reconnect creates a new runtime/session continuity decision explicitly;
- slow subscribers cannot backpressure the runtime indefinitely.

## Visual-client contract

### Leptos/browser

The everyday web presence SHOULD consume typed state/event semantics as they become available.

Large mental-movie/image data should move away from ordinary JSON state packets toward referenced/binary/artifact delivery where practical.

Browser main-thread work SHOULD be bounded per received state/event so one telemetry message cannot decode an arbitrarily large visualization backlog before input/rendering resumes.

### Bevy research dashboard

Bevy may render at the display frame rate while consuming much slower authoritative state.

It SHOULD interpolate/animate presentation between authoritative updates rather than demanding a cognitive update per rendered frame.

State transfer should be latest-wins. Ordered research events and imagination frames may use separately bounded queues.

A Bevy update must not drain an unbounded historical telemetry backlog in one frame.

## Latency measurement model

Latency metrics are observational engineering measurements. They do not establish semantic event order and are not part of deterministic replay equality.

Use monotonic process-local clocks (`Instant`-family semantics) for local duration measurement.

Wall-clock timestamps may be emitted as diagnostic metadata but are not the source of truth for ordering.

### Required measurement points

The runtime instrumentation SHOULD make the following boundaries observable:

```text
T0 observation_received
T1 observation_accepted
T2 response_started
T3 first_response_delta
T4 response_finished
T5 first_speakable_phrase_ready
T6 first_audio_queued
T7 first_audio_playable/device-visible
Ti interruption_detected
Tc cancellation_propagated
Ts stale_playback_silent
```

Useful derived metrics include:

- ingress acceptance = `T1 - T0`;
- time to response start = `T2 - T0`;
- time to first delta = `T3 - T0`;
- phrase-to-audio queue = `T6 - T5`;
- interruption-to-cancellation = `Tc - Ti`;
- interruption-to-silence = `Ts - Ti`;
- total turn generation = `T4 - T0`.

### Queue instrumentation

Every bounded realtime/interactive queue SHOULD expose at least:

- configured capacity;
- current/high-water depth where practical;
- dropped/coalesced count;
- lag/gap count;
- disconnected consumer count where relevant.

Metrics must not themselves create unbounded logging or high-rate allocations.

## Initial engineering targets

These are starting performance gates for measurement and design. They are **not current benchmark claims** and may be revised after host-specific evidence exists.

| Metric | Initial target |
| --- | ---: |
| local keypress -> next eligible render, p95 | <= 30 ms |
| local runtime state publish -> client receipt, p95 | <= 100 ms |
| local semantic event publish -> client receipt, p95 | <= 50 ms |
| interruption detected -> stale playback silent, p95 | <= 200 ms |
| partial speech feedback after usable audio context | approximately 100-300 ms |
| speakable phrase ready -> first playable audio, p95 | <= 500 ms |
| state queue memory growth under producer flood | O(configured bound), not O(duration) |
| stalled visual subscriber effect on cognition | no unbounded delay/backlog |

The latency program SHOULD report distributions (at minimum p50/p95/p99 and sample count) rather than a single best-case number.

## Determinism boundary

The following MUST NOT enter deterministic replay equality merely because they are instrumented:

- wall-clock timestamps;
- scheduler wake timing;
- transport latency;
- render frame timing;
- audio device callback timing;
- queue residence time;
- network RTT.

Replay identity should remain based on runtime configuration, seeds, ordered semantic observations/events, and explicitly deterministic state.

## Required negative controls

### Long-think liveness

Inject a cognitive operation that intentionally takes approximately 10 seconds.

During it:

- status/state inspection remains responsive;
- cancel remains responsive;
- UI animation/render remains responsive;
- incoming critical interrupt signals remain processable;
- evidence/control paths do not wait behind ordinary render work.

### State flood

Publish at least 10,000 state updates faster than the client can render.

Pass condition:

- memory remains bounded;
- client converges to latest state;
- skipped intermediate state is expected and visible through cursor/freshness semantics.

### Event flood

Overflow a deliberately slow semantic-event subscriber.

Pass condition:

- producer continues;
- consumer detects a sequence gap;
- resnapshot/recovery path is explicit;
- no silent continuity claim is made.

### Sensory flood

Overproduce microphone/visual sensory data.

Pass condition:

- configured drop/coalesce counters increase;
- memory remains bounded;
- action/control/evidence paths remain responsive;
- finalized user semantic observations are not silently discarded as if they were raw frames.

### Voice barge-in

Begin a long assistant utterance, then inject user speech.

Pass condition:

- interruption is detected;
- active turn becomes cancelled/interrupted;
- stale generation no longer reaches presentation;
- stale queued playback becomes silent within the configured bound;
- a new utterance identity becomes foreground;
- no stale prior-generation chunk plays after the new generation has taken ownership.

### Frozen visual client

Freeze a web or Bevy consumer while runtime updates continue.

Pass condition:

- cognition continues;
- runtime memory remains bounded;
- resumed client either receives latest state or detects a semantic gap and re-synchronizes.

### Evidence-stall separation

Artificially slow an evidence sink.

Pass condition depends on the owning authority protocol, but UI/sensory work must not pretend the evidence succeeded. Where evidence is required before an effect, the effect must remain fail-closed.

## PR implementation train constrained by this contract

### INT-002 — provenance hardening

First executable interface hardening after INT-001 qualifies.

- disconnected state becomes `Unknown` or explicitly `Stale`;
- deliberate fallback demos become `Simulation`;
- no synthetic drift may render as `Live`;
- no authority behavior changes are bundled into this PR.

### INT-003A — canonical runtime ownership

Introduce `SymthaeaRuntime` behind the compatibility facade.

- one mutable cognitive owner;
- turn-synchronous semantics preserved;
- immutable latest state independently readable;
- control/cancel path not serialized behind long read-only UI inspection;
- no autonomous heartbeat yet.

### INT-003B — latency instrumentation

Add the measurement boundaries defined above.

- process-local monotonic duration measurement;
- queue/drop/gap counters;
- percentile-reporting test/benchmark harness;
- no wall-clock values in deterministic replay equality.

### INT-004 — runtime planes and cancellation

Implement explicit state/event/sensory/evidence transport semantics.

- bounded queues;
- documented overload policy;
- `EventSeq` gap detection;
- turn/session cancellation propagation;
- slow-client isolation tests.

### INT-005A — real Unix IPC duplex pump

- one reader task;
- correlation routing;
- real subscription request;
- autonomous metrics/state receive path;
- reconnect semantics;
- bounded unsolicited update handling.

### INT-007A — asynchronous TUI foundation

- event-stream based terminal input;
- async select/reducer loop;
- no normal-path `block_on`;
- separate input/render/maintenance/state cadences;
- cancellable/debounced expensive completion work.

### INT-005C — semantic response streaming

- `ResponseStarted`;
- bounded `ResponseDelta` events;
- `ResponseFinished`;
- one event vocabulary for TUI/web/voice;
- cooperative turn cancellation.

### INT-009 — visual convergence

- typed state/events;
- latest-wins visual state;
- bounded imagination/artifact streams;
- explicit gap recovery;
- per-frame processing budget for research visualization.

### INT-008A — preemptible voice output

- incremental synthesis off the interactive path;
- utterance/generation identity on audio chunks;
- small explicit ahead-of-playback budget;
- queue flush/invalidation;
- interruption-to-silence measurement.

### INT-008B — bounded cognitive voice pipeline

- replace unbounded voice queues;
- incremental audio chunks rather than whole utterance responses;
- preserve self-hearing through the same acoustic representation path;
- retain source attribution.

### INT-008C — duplex `VoiceSession`

- VAD/endpointing;
- partial/final transcript events;
- correlated acoustic + semantic `SignalObservation`s;
- barge-in and cancellation;
- backend-replaceable VAD/ASR/renderers.

## Review rules for future interface PRs

A future interactive/runtime PR should be treated as incomplete if it introduces any of the following without an explicit justification and test:

- an unbounded queue on a normal interactive/sensory path;
- a blocking operation inside an audio callback;
- a UI method that owns the sole cognitive mutex while awaiting network/model work;
- a subscription API without an independently progressing receive path;
- stale/simulated state rendered as live;
- event loss without a sequence gap;
- cancellation that cannot invalidate already-buffered stale output;
- evidence records transported with lossy latest-state semantics;
- a renderer cadence coupled directly to cognitive-cycle frequency;
- latency assertions based only on best-case measurements.

## Deliberate non-claims

INT-000R establishes no claim that:

- the current runtime meets any latency target;
- the current voice stack achieves sub-200 ms interruption;
- the current TUI is async;
- current IPC subscriptions are fully duplex;
- the current Bevy/web clients are backlog-safe;
- cognition should tick continuously;
- higher cognitive frequency is inherently better;
- any consciousness, intelligence, or scientific metric improves because of this architecture.

It is a contract for how those claims may later be measured.

## Immediate next gate

Do not modify the frozen INT-001 exact head while its qualification workflows are pending.

The next executable child after INT-001 qualification should be the narrow provenance-hardening tranche. In parallel, runtime ownership/latency instrumentation can be prepared against the frozen interface contract, but must not claim executable qualification until its exact source and dependency lineage are frozen and tested.
