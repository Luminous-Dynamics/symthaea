# Symthaea Polyembodiment Architecture v0.1

**Date:** 2026-11-27  
**Status:** research RFC; proposed architecture, not an implemented or qualified capability  
**Scope:** digital avatars, expressive puppets, spatial/abstract forms, mobile and desktop renderers, and future physical companions

## Executive decision

Treat Symthaea as one cognitive and communicative system with multiple selectable embodiments, not as a single avatar and not as a cast of unrelated agents.

The proposed architecture is **polyembodiment**:

1. Symthaea emits typed, time-stamped, provenance-bearing interaction events.
2. A policy layer decides which events may be expressed, to whom, and through which channels.
3. Multiple embodiment adapters translate the same semantic events into distinct visual, aural, haptic, or physical expression.
4. Independent evaluation checks that the expression is timely, intelligible, accessible, privacy-preserving, and faithful to the source event.

The core remains independent of any renderer, rig, asset format, or vendor. A body is an interface and an affordance set; it is not the identity or proof of an inner state.

This is a proposal. It does not establish that Symthaea is conscious, sentient, emotionally aware, or equivalent to a human. Visualisations must not imply such claims.

## 1. Why this is the right problem

The current commercial frontier is still heavily human-avatar oriented:

- Epic's MetaHuman Animator supports real-time or offline facial/body performance generation from video, mobile capture, or audio. It is a strong reference for human-character animation and capture workflows.
- VideoFDB v1.4, updated 2026-11-25, evaluates full-duplex audiovisual interaction. Its leaderboard includes full-duplex and cascaded systems, and separates perception from generation. The leading entries are Anam Cara-5 Preview (3.87 perception overall) and 4.44 generation overall; these are benchmark-specific scores, not universal measures of intelligence.
- VRM 1.0 provides a portable humanoid-avatar extension for glTF, including humanoid bones, expressions, gaze, and spring-bone behavior.
- Khronos glTF is a runtime 3D asset-delivery format with keyframe node animation and morph-target animation.

These systems establish useful baselines. They do not, by themselves, solve a renderer-independent, multi-form embodiment whose expressive state is tied to inspectable cognitive events.

Symthaea should not compete by trying to out-photoreal MetaHuman or by pretending a generated face is a mind. The differentiation should be:

- one identity expressed through multiple non-human and human-compatible forms;
- continuous, full-duplex interaction rather than strictly turn-batched animation;
- explicit mapping from source events to rendered motion;
- low-power, local-first operation and privacy controls;
- accessible, user-governed embodiment selection;
- portable assets and reproducible evaluation.

## 2. Paradigm shifts

### 2.1 From one avatar to polyembodiment

Do not hard-code identity into a humanoid mesh. Define a stable identity profile and a family of interchangeable renderers. A transition between forms should preserve conversation context, current speech timing, user preferences, and interaction permissions.

### 2.2 From animation commands to semantic events

The cognitive system should not directly command a renderer with opaque values such as "look happy" or "move hand 3." It should publish typed events such as "listening," "yielding the conversational floor," "presenting two competing hypotheses," or "evidence is incomplete." Each renderer can express the same event using its own visual grammar.

### 2.3 From turn-based speech to full-duplex presence

Listening, turn-yielding, backchannels, gaze/attention cues, and speech generation should be modeled as concurrent streams. The renderer must be able to show appropriate listening behavior while the user is still speaking, and must be able to cancel or revise an animation when the source event changes.

### 2.4 From simulated personality to inspectable expression

Puppets and expressive characters should be presentation forms, not secretly independent personalities. A puppet may stage a question, a counterargument, a remembered scene, or an explanation. The shared source event and role must remain inspectable.

### 2.5 From vendor lock-in to portable embodiment assets

Use open asset interchange where it fits. Do not force every form into a human skeleton. Keep generic particle, graph, swarm, and kinetic-object representations in a renderer-neutral scene description.

### 2.6 From aesthetic quality to measurable communication

A beautiful animation that is late, misleading, distracting, inaccessible, or semantically wrong is a failed interface. Evaluate interaction quality and trustworthiness, not merely visual fidelity.

## 3. Embodiment family

The initial canonical family should contain:

| Form | Role | Primary expressive vocabulary | Best use |
|---|---|---|---|
| Living Constellation | Canonical presence | nodes, filaments, fields, topology, pulses | default identity and conversation |
| Puppet Theatre | Expressive cast | gesture, posture, staging, prop, silhouette | storytelling, teaching, dialogue and role-play |
| Mycelial Being | Relational view | branching paths, connection, growth, separation | memory relations, provenance and distributed work |
| Swarm | Collective view | density, synchrony, dispersion, local motion | multi-process activity and coordination |
| Living Instrument | Rhythmic/sensory view | rhythm, timbre, resonance, kinetic response | music and creative collaboration |
| Hearth Companion | Low-power physical presence | light, simple motion, haptics, sound | ambient home/workspace interaction |
| Cognitive Mandala | Explanatory view | layers, regions, links, uncertainty boundaries | system introspection and research |

These are distinct visual grammars for one source of events. Each needs a reduced-motion version, a static fallback, and a text-accessible description.

### Puppet design rules

- Puppets are not separate agents unless a real, separately identified agent is involved.
- Each puppet carries a role label such as narrator, questioner, explainer, or counterpoint.
- A puppet's gestures must not silently claim emotion, certainty, consent, or action.
- A cast scene must disclose when it is staging multiple viewpoints from one system.
- Users can dismiss, freeze, replace, or disable any puppet.
- Avoid deceptive human likeness by default. Stylised, abstract, animal-free, and non-anthropomorphic designs should be first-class.

## 4. Proposed architecture

    Symthaea cognitive and communication processes
                     |
                     v
       Typed Embodiment Event Stream
                     |
                     v
       Consent, Policy and Arbitration
                     |
                     v
       Stable Identity + Form Registry
          /       |        |       \
         v        v        v        v
     Constell.  Puppets  Mycelium  Swarm
         |        |        |        |
         +--------+--------+--------+
                     |
                     v
         Shared timebase / event receipts
                     |
          +----------+-----------+
          |          |           |
        Audio      Haptics    Physical output
                     |
                     v
          Independent evaluation

The core must not depend on Bevy, Android Views, Unreal, VRM, or a cloud avatar provider. A Bevy renderer is a promising Rust-first 3D target; the existing Soma Android UI can remain a lightweight mobile shell. The current Android demo layout already references particle, mandala, and garden layers. The implementation sources and build wiring for those referenced views must be verified before treating them as reusable runtime components.

### 4.1 Typed event contract

The first Rust implementation should define a versioned event envelope with the following minimum fields:

| Field | Meaning | Rule |
|---|---|---|
| schema | contract identifier and version | required |
| session_id | interaction session | required, opaque |
| event_id | globally unique within session | required |
| sequence | strictly increasing stream sequence | required |
| observed_at_monotonic_ns | source observation/decision clock | required |
| expires_at_monotonic_ns | event freshness deadline | required for transient events |
| event_kind | enumerated semantic event | required |
| payload | typed event-specific values | required |
| source | component and source event identity | required |
| epistemic_status | measured, derived, inferred, proposed, or presentation-only | required |
| confidence | calibrated confidence where inference is involved | optional; never fabricated |
| evidence_refs | content identifiers for supporting receipts | optional |
| privacy_class | public, local-sensitive, or restricted | required |

The event stream must not contain raw audio/video by default. It should carry derived, minimal signals and references to separately governed artifacts. Monotonic timestamps govern ordering and latency; wall-clock timestamps are metadata only.

Candidate event kinds:

- conversation: listening, speaking, interruption, backchannel, yield, wait, continue;
- attention: focus target, focus shift, attention confidence;
- expression: intensity, rhythm, gesture intent, prosodic contour;
- cognition display: competing hypotheses, uncertainty band, retrieval relation;
- evidence: unverified, supported, stale, contradicted, unavailable;
- user control: requested form, reduced motion, pause, cancel, consent state.

Do not encode human emotional labels as ground truth. When affect inference is enabled, mark it as inferred, retain confidence/calibration metadata, make it optional, and allow the user to disable it.

### 4.2 Freshness, ordering and cancellation

- Reject duplicate, regressing, expired, or wrong-session events.
- A later cancellation/supersession event must stop the obsolete animation within one render update.
- Never backdate a visual or audio effect to before its triggering input was observed.
- Keep source-decision time, renderer receipt time, first-visible-frame time, and physical-effect time separate.
- Missing events are missing data, not implicit WAIT or neutral emotion.
- On disconnect, stale state decays to a clearly defined neutral/static state; it must not continue implying live awareness.

### 4.3 Form registry

Each renderer declares:

- form_id and version;
- supported event kinds;
- visual/audio/haptic channels;
- required hardware and sensor permissions;
- asset manifest and digest;
- reduced-motion and low-power profiles;
- fallback renderer;
- known limitations;
- accessibility descriptions.

A form may ignore unsupported events, but it must not reinterpret them as stronger claims. Unsupported event kinds are logged as unsupported, not silently guessed.

### 4.4 User agency and consent

The user has priority over automatic form selection. Automatic adaptation may suggest a form, but the user's explicit preference wins. No form may activate camera, microphone, location, actuator, or network access merely because the form supports it. The default camera policy remains off unless explicitly enabled.

Transitions should be predictable: offer a preview, preserve a stable location for conversation controls, and avoid mid-utterance identity changes. Include a quick "return to canonical form" action and a global pause.

## 5. Interoperability

- **glTF 2.0 / 2.0.1:** baseline for portable 3D scene assets, node transforms, skeletal animation, and morph targets.
- **VRM 1.0:** optional adapter for humanoid puppets where humanoid bones, expressions, and gaze are useful. It is not the core embodiment contract.
- **Custom semantic scene graph:** required for constellation, mycelial, swarm, instrument and kinetic-object forms. It should describe semantic primitives and constraints, not vendor-specific render instructions.
- **Bevy:** preferred initial Rust renderer candidate, subject to a minimal prototype and device profiling.
- **Android Soma:** preferred near-term device surface, subject to source/build inventory and performance verification.
- **MetaHuman / Unreal:** benchmark and optional interoperability target, not a required dependency or canonical runtime.

Keep asset licenses, source hashes, rig versions, animation provenance, and redistribution rights in the asset manifest.

## 6. Correctness and epistemic integrity

The renderer is a presentation layer, not an authority.

- A "verified" visual requires a corresponding verifier receipt; colour, glow, or animation alone cannot signal verification.
- An uncertainty animation must be derived from a real uncertainty measure or explicitly labelled as illustrative.
- A memory-recall animation must bind to a real retrieval event, not merely to a text mention.
- A "thinking" animation must not imply that a hidden computation is occurring when the system is idle.
- No Φ value or embodiment multiplier may be presented as evidence of subjective experience.
- Separate system telemetry, inferred user state, artistic animation, and speculative interpretation in both API and UI.

The current embodied-consciousness source includes hard-coded embodiment multipliers and categorical language such as embodiment being required for consciousness. Those values are not qualified evidence and must not be reused as an avatar score or consciousness claim. This RFC makes no such claim.

## 7. Runtime and accessibility budgets

Initial qualification targets, not measured results:

| Measure | Target |
|---|---:|
| Local event-to-first-visible-frame | p95 <= 100 ms |
| Stale-event presentation | 0 accepted |
| Sequence / replay / session invariant violations | 0 accepted |
| Renderer frame pacing | 60 Hz desktop; 30 Hz minimum mobile target |
| Dropped frames during a 15-minute reference run | < 1% |
| Semantic event-to-render mapping on deterministic fixtures | 100% expected mappings |
| User cancellation to stopped animation | <= 1 render update |
| Default raw audio/video retention | 0 |
| Disabled sensor activation | 0 |
| Reduced-motion / static fallback | available for every form |

Measure all targets on named hardware, with build identity, power mode, thermal state, resolution, rendering backend, and workload recorded. A result on one laptop or phone does not establish broad device compatibility.

Accessibility requirements: keyboard and touch control, screen-reader descriptions, text alternatives for nonverbal cues, colour-independent status encoding, reduced motion, adjustable intensity, no compulsory gaze interaction, and no sound-only alerts.

## 8. Implementation sequence

### P0 — Inventory and freeze semantics

- Verify the actual source and build state of Soma's Android demo views.
- Inventory the current embodiment, HDC/CfC telemetry, audio, and turn-taking event producers.
- Reuse the semantics of the existing MEL-RTI-001 streaming scorer where applicable, while keeping speech and music-specific labels distinct.
- Freeze the event envelope and deterministic fixtures before renderer work.

### P1 — Minimal renderer-neutral Rust contract

- Add an isolated Rust module or crate with typed events, validation, freshness, sequence, session and cancellation rules.
- Keep it independent of GPU, Android, network and AI dependencies.
- Add tests for ordering, expiry, duplicate, cancellation, malformed payload, privacy class, and unsupported event kinds.
- Emit synthetic fixtures and a replayable event log.

### P1 initial implementation record — 2026-11-27

A renderer-independent event module has now been added in
`crates/domains/symthaea-soma/src/embodiment_event.rs` and exported from
`symthaea-soma`:

- Typed event and payload enums for conversation, attention, expression,
  uncertainty, evidence status, and user controls.
- Single-event validation for schema version, identifiers, monotonic timestamp
  encoding, expiry ordering, payload/event-kind agreement, inference confidence,
  evidence receipt requirements, and valid control payloads.
- Per-session stream validation for sequence continuity, duplicate identities,
  session binding, future observations, expiry, and cancellation references.
- No renderer, network, camera, microphone, GPU, or actuator dependencies.
- Regression tests are authored, including stream-ordering negatives, expiry,
  inference confidence, evidence receipts, cancellation, and JSON wire round-trip.

**Qualification boundary:** this is implementation code, not a verified build.
The exact branch has not yet been compiled or run through Cargo tests, CI, a JSON
Schema validator, or device profiling. It has not been connected to a renderer.
The event envelope does not itself authenticate source identity or authorize
hardware actions; those remain separate policy and trust-boundary work.

### P0 inventory finding — Soma Android demo source

The current `symthaea-soma/android/demo/src/main` tree contains the manifest and
Android resources/layouts, but no Java/Kotlin source files. The layout references
custom classes including `ParticleFieldView`, `ConsciousnessMandalaView`, and
`ConsciousnessGardenView`; their class definitions were not found in that demo
source tree. The layout also contains an empty `avatarRow` container, not a
verified working avatar selector. Treat this as a UI/resource shell until the
actual app source set and build target are located. Do not build the renderer
assuming these views already exist.

### P1.1 — Canonical semantic projection and deterministic replay

A second renderer-neutral module has been added:
`crates/domains/symthaea-soma/src/embodiment_projection.rs`.

- Defines the seven planned embodiment form identifiers.
- Converts typed events into a single canonical semantic-cue vocabulary.
- Keeps epistemic status, confidence, evidence references, privacy class,
  source identity, event identity, and time provenance with the cue.
- Provides an accessible text alternative that describes reported events
  without inventing emotion or claiming that a requested action completed.
- Replays ordered event traces without a GPU, renderer, microphone, camera,
  or network and can project the same canonical cue to each form identifier.
- Tests deterministic replay, sequence preservation, invalid trace rejection,
  and equality of cue semantics across all seven forms.

This is a semantic projection and replay harness, not a visual renderer. The
form IDs are present; no Bevy scene, puppet rig, or animation implementation is
claimed.

### P1 protocol hardening — 2026-11-27

The event contract now additionally:
- binds monotonic timestamps to an opaque clock-domain epoch and rejects mixed
  clock domains within one session;
- caps a session at 65,536 accepted events, requiring a new session epoch at the
  bound so replay-protection memory cannot grow without limit;
- records accepted cancellations, exposes `is_cancelled`, and rejects repeated
  cancellation of the same event;
- validates action-specific control payloads so a pause cannot smuggle a form,
  cancel target, or intensity field.

The caller-supplied `now_monotonic_ns` must come from the event's clock domain.
Cross-device replay requires an explicit clock synchronization/mapping protocol;
raw monotonic timestamps from different processes or devices must not be compared.

### P2 — Canonical Living Constellation

- Implement a low-cost procedural 2D/3D renderer driven only by the event contract.
- Map listening, turn-taking, speech, uncertainty and evidence states to distinct, documented visual grammar.
- Add static and reduced-motion fallbacks.
- Verify source-event-to-render receipts.

### P3 — Puppets and alternate forms

- Implement one stylised puppet rig and one non-humanoid form.
- Feed both the same fixtures; compare semantic consistency and latency.
- Add transitions between forms without losing session identity or conversational timing.
- Add the mycelial and swarm renderers only after the event contract proves useful.

### P4 — Full-duplex interaction

- Connect streaming audio events and optional visual cues.
- Evaluate on VideoFDB-compatible tasks and the project-specific benchmark plan.
- Ensure the renderer can backchannel during user speech and cancel a cue when context changes.

### P5 — Mobile and physical embodiments

- Integrate with Soma after the Android source/build inventory.
- Profile on Pixel 8 Pro and low-resource configurations.
- Prototype the Hearth Companion as a low-risk display/light/haptic endpoint before any mobility or manipulation.
- Keep physical actuators separately permissioned and fail-safe.

Each phase has an explicit evidence gate. A design document, synthetic test, compiled module, phone demo, and independently measured user study are different maturity levels and must be reported separately.

## 9. References and existing project seams

- NVIDIA VideoFDB project and benchmark: https://research.nvidia.com/labs/amri/projects/video-fdb/
- VideoFDB leaderboard v1.4 (2026-11-25): https://research.nvidia.com/labs/amri/projects/video-fdb/leaderboard.html
- Epic MetaHuman Animator: https://dev.epicgames.com/documentation/en-us/metahuman/metahuman-animator
- Epic real-time animation: https://dev.epicgames.com/documentation/en-us/metahuman/realtime-animation
- VRM 1.0 specification: https://github.com/vrm-c/vrm-specification/tree/master/specification/VRMC_vrm-1.0
- Khronos glTF registry: https://registry.khronos.org/glTF/
- Existing Symthaea streaming turn-taking evaluator: MEL-RTI-001 in this repository's docs/research directory.
- Existing Soma surface: crates/domains/symthaea-soma/android/demo/src/main/res/layout/activity_main.xml.
- Existing embodiment research source: crates/core/symthaea-core/src/hdc/embodied_consciousness.rs.
- Existing physical embodiment wire roadmap: docs/HOLON_SOMA_ROADMAP.md.

## 10. Decision requests for future review

This RFC chooses a semantic event contract and multiple rendering adapters as the default direction. It does not lock the project to a final visual design, animation engine, humanoid asset standard, or physical hardware. Those choices should follow prototype measurements and user evaluation, not preference alone.
