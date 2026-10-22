# Symthaea Interaction Runtime Architecture v1

**Status:** Architecture freeze / no runtime behavior change  
**Program:** INT-000  
**Base:** `main@a4168072a9cd29fafbb99928c33005ee4ba8d924`

## Purpose

Freeze the architectural boundary between Symthaea's cognitive runtime and its human/system interfaces before migrating the current REPL, TUI, web UI, voice, telemetry, and future ambient integrations.

This document does **not** create a new cognitive architecture. It constrains how the existing architecture is converged so that terminal, voice, visual, API, robotics, and future embodied surfaces observe and interact with the same running Symthaea instead of reconstructing overlapping local copies.

## Governing theorem

```text
one runtime identity
+ one authoritative mutable cognitive state
+ modality-neutral observations
+ typed state/event/sensory/evidence planes
+ capability-bound action authority
        ↓
many independently replaceable interfaces
```

not:

```text
one interface
= one locally reconstructed Symthaea
```

A terminal, browser, voice session, visualization, plugin, phone client, or robot bridge is an interface to a runtime. It is not a second cognitive authority.

## Existing foundations to preserve

This architecture deliberately builds on existing repository substrates rather than replacing them.

### `symthaea-communication`

`SignalObservation` is the canonical modality-neutral observation substrate. It already preserves modality, raw samples, derived features, text, uncertain spans, timing, calibration, source identity, environment metadata, and content-addressed observation identity.

New interface work SHOULD adapt external input into this substrate rather than creating parallel top-level audio/text/image observation families.

### Cognitive subsystem managers

The staged cognitive architecture already defines:

```text
CycleSnapshot
    -> CognitiveSubsystem::process(...)
    -> SubsystemOutput
    -> OutputCollector
    -> integrated state update
```

The manager path is currently transitional and may run beside legacy inline implementations. INT work MUST converge onto this existing proposal/integration model rather than adding another competing subsystem framework.

### `Symthaea`

The current high-level `Symthaea` facade remains a compatibility and ergonomic API during migration. A future `SymthaeaRuntime` MAY become the explicit owner of a running mind, but the migration MUST preserve existing call sites until replacement paths are qualified.

### SCIP

The Symthaea Cognitive Interchange Protocol remains a semantic/cognitive peer-interchange protocol. It is not the human-interface session protocol.

Interface events MAY reference SCIP or grounded concept graph artifacts, but UI/runtime transport semantics MUST NOT silently become SCIP semantic authority.

## Core invariants

### INT-I1 — single cognitive authority

Within one runtime identity, there is exactly one authoritative mutable cognitive state.

Clients MAY cache, render, replay, or simulate state. They MUST NOT represent client-local state as authoritative runtime state.

### INT-I2 — interfaces are replaceable projections

TUI, web, voice, Bevy, API, Zellij, desktop, mobile, and future embodied clients MUST be replaceable without changing cognitive semantics.

No interface technology may become a prerequisite for cognition to continue.

### INT-I3 — observation ingress is modality-neutral

Human/environment inputs SHOULD enter through `symthaea-communication::SignalObservation` or an explicitly versioned successor preserving equivalent provenance and modality semantics.

A transcript is one representation of an utterance, not the utterance's complete identity.

### INT-I4 — cognitive telemetry never grants authority

Phi, Psi, coherence, arousal, confidence, free energy, thermodynamic load, attention, consciousness labels, or any other cognitive metric MAY influence deliberation or request stronger review.

They MUST NOT manufacture capabilities, execution permission, filesystem authority, network authority, system-management authority, or physical-action authority.

```text
changed cognitive telemetry
!= expanded authority
```

### INT-I5 — transport is not semantic authority

JSON, MessagePack, protobuf/gRPC, Unix sockets, HTTP, WebSocket, in-process channels, and future transports are encodings/carriers.

Transport-specific DTOs MUST NOT independently define conflicting semantic meanings for the same runtime fact.

### INT-I6 — sequence establishes runtime event order

Every ordered semantic runtime event MUST carry a monotonic runtime-local sequence number.

Wall-clock timestamps are useful metadata but MUST NOT be the sole semantic ordering primitive.

### INT-I7 — missing telemetry is explicit

A client that lacks a current authoritative value MUST represent that absence explicitly.

Required provenance vocabulary for externally rendered runtime state:

```text
Live
Stale
Replay
Simulation
Unknown
```

`Disconnected` is a connection condition, not permission to invent plausible state.

Synthetic/default values MAY be used for tests, design previews, demos, or explicit simulation, but MUST be labeled as such and MUST NOT be surfaced as live cognition.

### INT-I8 — backpressure semantics are plane-specific

State, semantic events, high-volume sensory data, and durable evidence have different retention and backpressure requirements. They MUST NOT be collapsed into one undifferentiated broadcast stream.

### INT-I9 — UI observation is non-authoritative

A UI rendering, gauge, avatar pose, color, animation, waveform, mental-movie frame, or summarized label is presentation.

```text
presentation
!= cognitive state authority
!= scientific evidence
!= execution authority
```

### INT-I10 — introspection is structured telemetry

Research interfaces MAY expose structured cognitive telemetry, subsystem attribution, attention, uncertainty, prediction errors, manager outputs, memory events, or explicitly modeled internal representations.

The interface architecture does not require or define free-form hidden reasoning transcripts as a runtime surface.

## Runtime identity model

Future typed interface work SHOULD provide explicit identities for at least:

- runtime;
- client/session;
- turn;
- utterance;
- observation;
- event sequence;
- action proposal;
- authorization/decision;
- action result;
- artifact.

IDs MUST have clearly documented scope. A UI correlation ID MUST NOT silently become an evidence identity or authorization identity.

## Four logical runtime planes

### 1. State plane

Purpose: latest-known renderable runtime state.

Examples include consciousness/phenomenology measurements, affect, attention, current cognitive depth/mode, thermodynamic/resource state, sensor status, voice state, active task, and connectivity.

Semantics: latest-wins is acceptable. A slow client may skip intermediate states if it can identify the newest state and its provenance.

Recommended implementation family: watch/snapshot semantics.

### 2. Semantic event plane

Purpose: ordered interaction/runtime events.

Examples include observation acceptance, utterance partial/final, response started/delta/finished, attention shift, memory retrieval, tool/action proposal, authorization outcome, action result, artifact production, voice interruption, and error/recovery events.

Semantics: sequence-bearing and gap-detectable. Clients MUST be able to detect missed events.

### 3. Sensory plane

Purpose: bounded high-volume streams such as microphone frames, video/screen frames, high-rate HDC sensory states, and device telemetry.

Semantics: bounded queues with modality-specific retention/drop policies. Sensory overload MUST NOT starve action/audit/control paths.

The existing audio-stream pattern (real-time callback -> bounded buffering/worker -> nonblocking consumption) is a model for this plane.

### 4. Evidence/audit plane

Purpose: durable or explicitly retained authority/evidence records such as action authorization decisions, execution receipts, security denials, epistemic provenance, research measurements, and qualification/audit events.

Semantics: evidence MAY NOT inherit latest-wins/drop-on-overload behavior from UI telemetry. Loss or failure MUST be explicit and fail according to the owning authority protocol.

## Turn-synchronous migration first

INT MUST NOT use runtime convergence as an excuse to silently introduce unconstrained continuous cognition.

The first `SymthaeaRuntime` migration SHOULD preserve current turn-synchronous behavior:

```text
observation
    -> bounded cognitive work
    -> response/state update
```

Only after deterministic replay, state lineage, scheduling, and resource gates qualify should the runtime add explicit cycle causes such as:

```text
ExternalObservation
InternalHeartbeat
IdleCycle
DreamCycle
SleepCycle
SensoryInterrupt
```

A continuously ticking mind is a separately measurable behavioral change.

## Voice architecture

Voice SHOULD be modeled as a session rather than blocking `listen() -> String`.

One captured utterance SHOULD produce correlated observations under one utterance identity:

```text
captured utterance
    ├─ auditory/acoustic/HDC perception
    └─ semantic ASR / preserved human text
```

The semantic transcript MUST NOT erase the auditory observation lineage.

Future full-duplex voice SHOULD support explicit lifecycle states such as:

```text
Listening
SpeechDetected
Transcribing
UserUtteranceFinal
Thinking
Speaking
Interrupted
Recovering
```

Barge-in requires cancellation to propagate through generation, phrase scheduling, rendering, and playback.

### Voice rendering boundary

Cognition SHOULD produce backend-independent vocal intent. Renderers MAY include Kokoro, formant synthesis, and the Symthaea vocal tract.

```text
semantic response
+ cognitive/prosodic state
    -> VocalIntent
    -> VoiceRenderer
    -> audio
    -> optional voice-quality feedback observation
```

A renderer backend MUST NOT become cognitive authority.

## Terminal architecture

The canonical TUI remains terminal-emulator agnostic.

Ratatui/Crossterm are the current Rust TUI foundation and SHOULD be consolidated rather than replaced merely to target a specific terminal emulator.

Terminal support SHOULD be capability-based:

- baseline text/Unicode/style/input;
- enhanced keyboard capability where available;
- synchronized rendering where available;
- inline graphics where available;
- deterministic text fallback where unavailable.

Alacritty, WezTerm, Ghostty, Kitty, and other terminals are capability environments, not separate Symthaea products.

### Zellij

A future Zellij plugin is an optional peripheral client. It MUST NOT host cognitive state.

Permissions SHOULD be minimal by default. Reading pane contents, intercepting input, running commands, or writing stdin are separate capabilities and MUST require explicit enablement rather than being implied by installation.

## Visual architecture

### Presence UI

The Leptos/browser UI SHOULD remain the primary low-friction human-facing visual presence.

Its normal mode SHOULD emphasize legible state such as listening, focus, uncertainty, memory, imagination, speaking, resource load, and connection/provenance rather than exposing every research metric simultaneously.

### Cognitive twin / research UI

The Bevy dashboard SHOULD remain a separate research/introspection instrument for detailed topology, telemetry, imagination, neuromodulators, subsystem outputs, and experimental visualization.

A research dashboard and an everyday conversational presence are different products over the same runtime.

### Administrative GUI

NixOS/Spore/Nixward system-management interfaces SHOULD remain explicitly administrative. They MUST NOT be confused with Symthaea's general visual presence merely because they share cognitive telemetry.

## Action lifecycle

The intended authority-safe lifecycle is:

```text
cognition proposes
    -> deterministic policy / capability evaluation
    -> required operator authorization
    -> exact authorized execution
    -> action result
    -> result re-enters cognition as observation
```

Cognitive state may request caution or refuse to propose an action. It cannot create authority.

Action identifiers and authorization identifiers MUST remain distinct from UI request IDs and cognitive event IDs.

## Current migration hazards

The following current patterns motivate the migration and MUST NOT be copied into new surfaces:

1. Multiple top-level interactive paths independently own cognition/LLM/action/voice state.
2. Service JSON, shell MessagePack, gRPC telemetry, and UI-local DTOs duplicate semantic structures.
3. Shell IPC currently contains overlapping request/response families.
4. `MetricsSnapshot::default()` currently contains plausible cognitive values rather than an explicit absence/provenance state.
5. Some clients can fall back to simulated/local metrics without making that distinction a first-class semantic property.
6. Manager-based cognition is still dual-written beside legacy inline paths.
7. Voice transcription is available through the communication-provider contract while blocking microphone `listen()` remains a placeholder; live acoustic/HDC perception exists elsewhere.

These are migration observations, not claims that the corresponding components are unusable.

## Compatibility strategy

Migration MUST be additive and reversible.

- Existing public `Symthaea` entry points remain usable while runtime ownership moves behind them.
- Existing service transports MAY be adapted to new semantic types before old wire shapes are removed.
- Existing TUI/shell capabilities are preserved during extraction into reusable TUI components.
- Existing voice renderers remain available behind a common renderer contract.
- Existing web/Bevy interfaces remain separately buildable.
- Existing manager/inline cognition paths require parity evidence before any legacy path is removed.

## Planned implementation train

### INT-001 — shared interface types

Add a small shared crate for runtime/session/turn/utterance/event identities, provenance/freshness, capability negotiation, and compact semantic events.

It MUST stay transport-neutral and WASM-friendly.

### INT-002 — provenance hardening

Replace unlabeled plausible fallback metrics with explicit `Live | Stale | Replay | Simulation | Unknown` state throughout shell/UI-facing paths.

A disconnected client MUST NOT display synthetic cognition as live cognition.

### INT-003 — canonical runtime owner

Introduce a unique runtime owner behind the existing `Symthaea` facade while preserving turn-synchronous behavior.

This PR moves ownership, not scientific semantics.

### INT-004 — typed runtime planes

Implement state, event, sensory, and evidence/audit planes with explicit sequencing and backpressure semantics.

### INT-005 — service transport convergence

Adapt Unix IPC, HTTP/JSON, WebSocket, and telemetry transports onto the shared semantic model while retaining compatibility adapters as needed.

### INT-006 — REPL convergence

Remove duplicate top-level cognitive ownership from REPL paths and implement the currently incomplete remote orchestration path against the canonical runtime.

### INT-007 — reusable TUI foundation

Extract the large shell application into reusable Ratatui components, standardize compatible Ratatui/Crossterm versions, and move to an async/cancellation-aware event loop.

Suggested views: Converse, Observe, Act, Shell, Senses, Research.

### INT-008 — duplex voice session

Unify microphone capture, auditory/HDC perception, ASR, utterance identity, partial/final transcript events, renderer selection, streaming playback, cancellation, and barge-in.

### INT-009 — typed visual clients

Move Leptos and Bevy toward shared typed runtime DTOs/events and eliminate duplicated semantic telemetry structures where practical.

### INT-010 — finish manager convergence

Retire dual-write inline cognitive paths manager by manager after parity and regression evidence.

### INT-011 — controlled continuous cognition

Add separately qualified heartbeat/idle/dream/sleep cycle causes only after deterministic replay and resource/behavior gates exist.

### INT-012 — ambient clients

Add optional Zellij/desktop/mobile/Spore surfaces as thin clients. No cognitive authority moves into them.

## Acceptance properties for the program

The implementation program SHOULD eventually provide executable tests for these properties.

### Transport equivalence

Equivalent semantic requests over supported transports produce equivalent semantic runtime events aside from transport-specific metadata.

### Sequence/gap detection

A client that misses semantic events can detect the gap and request/resume from a known state boundary rather than silently assuming continuity.

### Replay

Given the same deterministic runtime configuration, genesis/seed state, and ordered observations, the deterministic semantic portion of the runtime can be replayed without wall-clock ordering becoming authority.

### Slow-client isolation

A stalled UI subscriber cannot stall cognition. State can latest-win; event loss must be detectable; evidence loss follows its own fail-closed contract.

### Fake-state negative control

With the runtime disconnected, no UI path may classify placeholder/simulated/default values as `Live`.

### Authority invariance

Changing only cognitive telemetry cannot expand the set of actions permitted under an unchanged authority context.

### Sensory backpressure

Flooding audio/video cannot create unbounded memory growth or starve control/evidence paths.

### Voice interruption

A new user utterance can cancel/suspend active response generation and playback without speaking stale queued output afterward.

### Manager migration parity

Before removing a dual-write inline path, frozen fixtures/replays must show that the manager-based path preserves the behavior/measurements claimed by that migration or explicitly documents the intended semantic change.

## Deliberate non-claims

INT-000 establishes no:

- runtime implementation;
- cognition improvement;
- consciousness result;
- continuous-mind behavior;
- TUI version upgrade;
- voice/STT/TTS quality gain;
- action authorization improvement;
- transport interoperability result;
- UI fidelity result;
- manager parity result;
- scientific qualification.

It is an architecture boundary only.

## Immediate next gate

INT-001 should be the first executable/source tranche. It should introduce only the minimal shared interface identities, provenance/freshness model, capability handshake, and compact semantic event vocabulary needed for later adapters.

INT-001 should not yet rewrite the service, shell, web UI, voice stack, cognitive loop, or action execution path.
