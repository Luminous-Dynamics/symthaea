# Mycelial Presence: Evidence-Driven Morphogenesis and Cognitive Instrument

**Date:** 2026-11-21  
**Status:** Research and implementation specification; no morphogenesis engine or live cognitive graph producer is claimed implemented.  
**Parent work:** [PR #13012](https://github.com/Luminous-Dynamics/symthaea/pull/13012)  
**Companion:** [Renderer evolution](MYCELIAL_RENDERER_EVOLUTION_2026-11-21.md)

## Executive decision

The next architectural step is not "more procedural branches". It is to introduce a **semantic topology layer** and a separate **morphology realization layer**.

The semantic topology records what the runtime actually observed or produced: memory entities, retrieval episodes, associations, prediction candidates, plans, and operation lifecycles. The morphology layer maps those stable identities into spatial curves, junctions, membranes, cells, and light. A renderer consumes both, but owns neither.

This creates a living organism whose form can evolve while preserving an auditable answer to:

- What runtime event caused this visible structure to appear?
- Which subsystem emitted it?
- Is it observed, derived, or hypothetical?
- What is its lifecycle state?
- Can the same event trace reproduce the same semantic state and canonical geometry?
- Can the user inspect the evidence without the visual metaphor becoming an authority?

**Do not build a fake neural network out of beautiful branches.** A branch is only a cognitive connection when an owning subsystem emits a typed, ordered, provenance-bearing event that creates or updates that connection.

## 1. Paradigm shift: two coupled graphs, not one decorative graph

### 1.1 Semantic topology (authoritative data)

A versioned, renderer-neutral graph should represent:

- **Entities:** stable references to memory items, concepts, tools, tasks, observations, or other runtime-owned objects.
- **Relations:** typed edges such as retrieved-from, associated-with, supports, contradicts, predicts, depends-on, or executed-by.
- **Episodes:** bounded events such as retrieval, association formation, prediction generation/resolution, planning, and tool execution.
- **Evidence:** source subsystem, event identity, sequence, monotonic time, provenance class, and source trace/digest.
- **Epistemic status:** observed, derived, candidate/hypothesis, resolved, rejected, stale, or unavailable.
- **Lifecycle:** created, updated, activated, completed, cancelled, failed, superseded, or expired.

The graph's identities must be independent of screen position, branch generation, rendering mode, colour, or animation phase.

### 1.2 Morphology realization (derived presentation)

A separate geometry state maps semantic identities to:

- filament curves and thickness profiles;
- branch junctions and terminal growth fronts;
- local clusters and orbital structures;
- membrane surfaces and nucleus layers;
- activity pulses and candidate-path overlays;
- confidence/uncertainty visual channels when supplied by a calibrated source.

Every derived geometry element carries the semantic IDs and evidence references from which it was built. The geometry layer may approximate, simplify, or hide a structure for performance, but it may not silently alter the semantic graph.

### 1.3 Why this separation matters

A semantic edge can persist while its on-screen branch is culled, simplified, occluded, or represented in 2D. A renderer can change from Bevy to WebGPU without changing the cognitive graph. An uncertainty estimate can be absent without the organism inventing a visual equivalent. A prediction can remain a candidate without being drawn as a completed memory.

This is the boundary that makes Mycelial Presence an instrument instead of an animation.

## 2. Morphogenesis: how the organism should grow

A strong design can combine several biological and computational ideas, but each should be used for the part it is good at.

| Approach | Useful role | Do not use it for |
|---|---|---|
| Deterministic branching grammar / L-system | A stable baseline morphology and style families | Claiming that grammar-generated branches represent real cognition |
| Space-colonization growth | Steering tips toward a known set of spatial attractors or semantic anchors | Inventing attractors from text or renderer timing |
| Physarum-inspired transport adaptation | Local strengthening, rerouting, and resilience experiments on an existing relation graph | Treating an analogy to slime-mold behaviour as proof that cognition is biologically equivalent |
| Reaction-diffusion / Turing patterns | Membrane mottling, surface detail, pigmentation, and subtle material variation | Creating semantic nodes, retrieval edges, or prediction paths |
| Diffusion-limited aggregation | Optional high-level silhouette or decorative growth studies | The authoritative graph: it is sensitive to initial conditions and can form tangled, hard-to-inspect structures |

The Physarum literature is particularly interesting because transport networks can emerge through local adaptation and trade-offs between efficiency, cost, and fault tolerance. A recent distributed Physarum optimal-transport method explicitly targets local execution rather than requiring a central graph solve. These are candidate algorithms for an experimental morphology/route layer, not direct models of Symthaea cognition.

References:
- Tero et al., *Rules for biologically inspired adaptive network design*, Science (2010): https://doi.org/10.1126/science.1177894
- *Physarum-based approach to distributed optimal transport on graphs*, Scientific Reports (2026): https://doi.org/10.1038/s41598-026-61965-x

### 2.1 Proposed deterministic growth loop

1. Receive a validated semantic event from its owning runtime adapter.
2. Validate schema version, session identity, sequence, provenance, finite values, bounds, and lifecycle transition.
3. Apply the event idempotently to the semantic topology.
4. Derive a stable growth command keyed by event identity and graph revision.
5. Advance morphology in bounded, deterministic steps.
6. Emit a geometry delta containing stable semantic IDs, added/updated/hidden elements, and the source event IDs.
7. Record the input event digest and resulting canonical geometry digest in the trace.
8. Render the delta. Renderer-local lighting and breathing remain explicitly non-semantic.

For reproducibility, the growth seed must derive from stable event/graph identity rather than wall-clock time or a global random stream. Use fixed-step updates and canonical coordinate quantization where cross-platform byte-stability is required. Where GPU math makes exact byte identity unrealistic, compare semantic topology hashes exactly and geometry using a specified tolerance.

### 2.2 Growth and maturation rules

- A new verified relation may create a new branch with a stable identity.
- Repeated retrieval may increase a display activation or thickness channel only when the runtime supplies a counted retrieval record. It must not be conflated with truth, relevance, or confidence.
- Contradictory evidence may create a visibly distinct cross-link or conflict junction; it must not silently erase either relation.
- A prediction creates a candidate branch with a clearly different visual grammar. Resolution, rejection, or cancellation must be an explicit lifecycle event.
- Completed tool execution settles its temporary execution pathway; it does not imply that the tool result is correct.
- Pruning is a presentation decision. Semantic evidence remains available in the graph and trace even when a branch is visually dormant or not rendered.
- Missing, stale, or contradictory inputs preserve the last valid state and expose diagnostics. Unknown must not be rendered as certainty.

### 2.3 The key visual distinction

Separate three kinds of motion and make that distinction discoverable:

1. **Semantic motion:** caused by a qualified runtime event; carries an event ID and provenance.
2. **Derived motion:** a deterministic interpolation or layout response to a semantic change; linked to the triggering event and clearly identified as derived.
3. **Ambient motion:** purely decorative breathing, twinkle, or drift; never changes semantic state and can be fully disabled.

This is essential to prevent users from mistaking an idle animation for evidence that an internal process is active.

## 3. Typed event contract to build

The current embodiment contract already handles activity, gaze, speech lifecycle, visemes, and explicit uncertainty. The following are proposed additions, not implemented runtime producers:

- `CognitiveNodeCreated` / `CognitiveNodeUpdated`
- `CognitiveEdgeCreated` / `CognitiveEdgeUpdated` / `CognitiveEdgeRetracted`
- `RetrievalStarted` / `RetrievalCompleted`
- `AssociationProposed` / `AssociationAccepted` / `AssociationRejected`
- `PredictionCandidateCreated` / `PredictionResolved` / `PredictionRejected`
- `PlanCreated` / `PlanStepStarted` / `PlanStepCompleted` / `PlanCancelled`
- `OperationStarted` / `OperationProgressed` / `OperationCompleted` / `OperationFailed`

Every event should carry:

- schema version, session ID, stable event ID, sequence number, monotonic timestamp;
- producer subsystem and typed provenance;
- stable entity/relation/episode IDs;
- explicit lifecycle transition and relevant parent/correlation IDs;
- bounded payload and finite numeric values;
- trace linkage or source digest;
- optional calibrated confidence/uncertainty, clearly distinguished from activation or frequency.

Do not add generic `is_thinking`, `is_feeling`, or `is_remembering` booleans. Model concrete operations and evidence-bearing transitions instead. The UI can summarize a validated event stream, but it cannot create the events.

## 4. Visual system: more organism, less neon diagram

### 4.1 Filaments and junctions

The current cubic Bézier/tube representation is a good portable starting point. The next geometric quality step is to make a branch look like one grown structure rather than several tubes joined by a glowing bead.

Investigate:

- variable radius profiles along arc length, not only linear start/end taper;
- curvature-aware control points and bounded torsion;
- smooth parent-child radius blending;
- watertight junction patches or implicit-surface blending for high-quality mode;
- stable low-cost junction bulbs for mobile mode;
- mesh self-intersection and degenerate-triangle checks;
- stable per-edge material and semantic selection ranges.

A high-quality junction should preserve a readable fork, avoid pinched necks, and blend smoothly into both child branches. A cheaper junction may use an instanced node, but the benchmark should quantify the visual trade-off.

### 4.2 Nucleus and membranes

Give the nucleus a true layered spatial structure:

- inner core with restrained emissive detail;
- one or more thin translucent shells;
- subtle rim response / Fresnel approximation;
- spatially separated internal structures;
- controlled transmission and roughness rather than blanket transparency;
- no full-scene bloom washout.

Use standard PBR as the baseline; introduce custom WESL materials only when a reference frame demonstrates a measurable visual improvement. glTF material extensions can help with static interchange, but the live event/geometry model remains authoritative.

### 4.3 Lighting and atmosphere

- Use one clear key/fill/rim hierarchy.
- Make depth readable through occlusion, controlled contrast, and spatial separation before adding fog.
- Use particles sparingly and deterministically for baseline ambient detail.
- Add event-driven light propagation only along the associated semantic path.
- Let the user turn down bloom, particle density, motion, and contrast independently.
- Keep a clean presentation preset that remains beautiful in a small window and at low brightness.

### 4.4 Colour is not the only channel

Each activity should have a stable combination of topology, pulse shape, texture, and label. Colour may reinforce a distinction but must not be the only carrier. Uncertainty, conflict, unavailable data, pending operations, and failure must all have distinct non-colour cues.

## 5. Cognitive Instrument interactions

The Cognitive mode should be an inspectable instrument, not a dense "brain dashboard".

### Interaction model

- Hover/focus: reveal a short label and current lifecycle.
- Select: pin a semantic entity, relation, or episode and expose provenance.
- Follow: trace a retrieval or prediction path through its actual source edges.
- Compare: show competing candidate paths without visually choosing a winner.
- Replay: scrub a recorded event trace deterministically.
- Inspect: open source subsystem, event sequence, timestamp, trace digest, and evidence status.
- Reset view: return to the ambient organism without changing underlying state.

### Three stable modes

**Ambient Presence:** beautiful, calm, low motion, minimal labels. Good for a desktop companion or home display.

**Cognitive Instrument:** graph structure, selection, provenance, uncertainty, and replay. Good for research and debugging.

**Conversational Presence:** responsive to confirmed speech lifecycle and supplied gaze/viseme signals. A humanoid or face-like skin may be an optional adapter, never the required identity of the organism.

All three modes consume the same graph and event contract. They may change camera, density, material, and layout, but not event meaning.

## 6. Rendering ecosystem and industry comparison

Do not force one renderer to win every environment.

| System / approach | Strong comparative capability | Mycelial Presence response |
|---|---|---|
| Bevy 0.20 / native wgpu | Rust-native 3D scene, PBR, mesh and shader flexibility | Primary native research renderer; validate the isolated adapter before adding custom shaders |
| WebGPU | Modern browser GPU rendering and compute | Preferred browser GPU path when an adapter and device are actually available |
| WebGL 2 | Broadly useful browser GPU fallback | Keep a conventional vertex/fragment pipeline for devices without WebGPU |
| SVG | Portable, inspectable, static and low-power output | Retain as fallback, snapshot, debugging and accessibility surface |
| Rive / Live2D | Strong authored state-driven 2D animation and expressive parameter control | Useful comparison or optional expressive skin; not the canonical cognitive graph |
| Unreal MetaHuman | High-fidelity human character animation, including real-time audio/video workflows | Benchmark as a facial embodiment baseline, not as a like-for-like cognitive instrumentation competitor |
| NVIDIA ACE / Audio2Face | Audio-to-facial-animation and blendshape-driven digital human tooling | Integrate or compare for optional voice embodiment; do not rebuild specialist facial animation without evidence of advantage |

Industry references:
- Bevy 0.20 release and renderer changes (8 October 2026): https://bevy.org/news/bevy-0-20/
- WebGPU publication history (Candidate Recommendation Draft, 16 November 2026): https://www.w3.org/standards/history/webgpu/
- MDN `GPU.requestAdapter()`: availability is limited, the API is secure-context-only, and the browser may return no adapter: https://developer.mozilla.org/en-US/docs/Web/API/GPU/requestAdapter
- Epic MetaHuman real-time animation: https://dev.epicgames.com/documentation/en-us/metahuman/realtime-animation
- NVIDIA Audio2Face-3D NIM: https://docs.nvidia.com/nim/digital-human/a2f-3d/latest/
- Live2D parameter control: https://docs.live2d.com/en/cubism-sdk-manual/parameters/
- Rive state machines: https://github.com/rive-app/help-center/blob/master/runtimes/state-machines.md
- Khronos glTF transmission material extension: https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_transmission

Bevy 0.20's release notes describe WESL adoption and a new mesh-shader path. The native adapter should keep standard mesh rendering as its compatibility baseline; browser parity must not depend on a native-only mesh-shader feature. WebGPU remains a capability-detected path, not a mandatory requirement.

**Competitive thesis:** MetaHuman and Audio2Face solve high-fidelity human embodiment; Live2D and Rive solve authored expressive 2D motion. Mycelial Presence should differentiate through a non-humanoid organism, semantic event fidelity, provenance inspection, deterministic replay, renderer portability, accessibility, and low-power operation. Beat those tools only on matched dimensions, not with a synthetic all-in-one "avatar score".

## 7. Benchmark programme: Mycelial Presence Evaluation Suite (MPES)

A benchmark should be a reproducible corpus, not a screenshot contest. The same versioned model, event trace, viewport, visual quality preset, and hardware must be supplied to every renderer.

### Track A — semantic fidelity

| Metric | Initial gate |
|---|---:|
| Accepted unsupported semantic states | 0 |
| Rejected stale/invalid lifecycle transitions | 100% of defined negative controls |
| Provenance retained through model and renderer | 100% |
| Deterministic event replay | 100% canonical semantic-state parity |
| Cross-renderer semantic divergence | 0 |
| Geometry element without valid source or renderer-local classification | 0 |

### Track B — geometric quality

| Metric | Initial target |
|---|---:|
| Parent-child endpoint gap, normalized by local branch length | ≤0.1% |
| Tangent discontinuity at smooth child starts | ≤5° |
| Degenerate triangles in accepted mesh | 0 |
| Non-finite or out-of-bound geometry accepted | 0 |
| Unreported self-intersections in high-quality junction set | 0 |
| Stable geometry digest for fixed quantized inputs | 100% |

These are proposed acceptance gates. The high-quality junction representation must be compared with the lower-cost bulb-based representation using the same input graph.

### Track C — responsiveness and performance

| Metric | Desktop target | Mobile target |
|---|---:|---:|
| Sustained frame-time P95 | ≤16.7 ms | ≤33.3 ms |
| Event-to-first-visible-frame P95 | ≤50 ms | ≤100 ms |
| Frame-time P99 | ≤33.3 ms | ≤50 ms |
| Unbounded event queue growth | 0 | 0 |
| Memory growth after 30-minute steady-state run | 0 unbounded growth | 0 unbounded growth |
| Reduced-motion residual semantic/ambient animation | 0 | 0 |

Measure cold start, warm start, model construction, geometry update, upload, render submission, present, CPU/GPU time, memory/VRAM, draw calls, allocations, event backlog, and energy per minute separately. Report device, driver, OS, resolution, quality preset, thermal state, and renderer fallback frequency.

### Track D — human legibility and calibrated trust

Use a blinded, counterbalanced study. Compare text-only, static graph, generic glowing orb, expressive 2D puppet, high-fidelity human avatar, and evidence-connected Mycelial Presence.

| Metric | Initial target |
|---|---:|
| Cognitive event classification macro-F1 | ≥90% |
| Playback lifecycle recognition accuracy | ≥95% |
| Provenance/source comprehension | ≥90% |
| Distinguish candidate prediction from completed fact | ≥95% |
| Distinguish no uncertainty signal from supplied uncertainty signal | ≥90% |
| Required accessibility checks passed | 100% |
| Unsupported anthropomorphic inference | No worse than neutral-control interface |

Collect confidence intervals and task completion time, and preregister hypotheses. Do not interpret participant preference as evidence of semantic correctness. Do not infer trust from engagement or time spent.

### Track E — accessibility and power

- reduced-motion mode removes all ambient motion, not just selected loops;
- keyboard focus and pointer selection both work;
- captions and labels explain non-colour channels;
- high-contrast and low-brightness presets remain legible;
- static image/export is available;
- density, bloom, contrast, motion, and animation intensity can be independently reduced;
- mobile thermal and energy behaviour is measured over long sessions.

### Baseline discipline

Run SVG, a conventional WebGL 2 renderer, WebGPU, and native Bevy against the same trace and model. Add Rive/Live2D for expressive 2D comparisons and MetaHuman/Audio2Face only for voice/facial-animation tracks. Never compare unmatched workloads, different hardware, different asset complexity, or cold-start measurements against warmed steady-state measurements.

All targets are research gates, not achieved results or current SOTA claims.

## 8. Execution roadmap

### P0 — make the present branch verifiable
- [x] Add renderer-neutral spatial model and bounded CPU mesh.
- [x] Add isolated Bevy 0.20.1 adapter source.
- [x] Define evidence-aware event/speech lifecycle and trace replay.
- [x] Apply rustfmt corrections discovered by the focused renderer workflow.
- [x] Correct the focused workflow's mismatched temporary-workspace path.
- [ ] Obtain a passing exact-head format/test/Clippy run.
- [ ] Compile and run the Bevy preview; capture a reference frame on a supported device.
- [ ] Record actual startup, frame-time and memory measurements.

### P1 — morphology without semantic fabrication
- [ ] Add a separate versioned semantic graph model and bounded graph event corpus.
- [ ] Add deterministic event-to-growth commands and idempotent replay.
- [ ] Add geometry-to-semantic-ID lookup for every branch/node.
- [ ] Add stable geometry hashes with documented cross-platform tolerance.
- [ ] Add variable-radius profiles and smooth junction experiments.
- [ ] Add topology integrity, mesh self-intersection and degenerate-triangle tests.

### P2 — first Cognitive Instrument
- [ ] Connect one genuine producer end-to-end (retrieval is the recommended first vertical).
- [ ] Show one retrieved entity and its actual source edge, with selectable provenance.
- [ ] Add an association producer only after its owning subsystem can emit typed evidence.
- [ ] Add prediction candidate/resolve/reject lifecycle with an explicit hypothesis visual.
- [ ] Add graph selection, trace replay, and no-data/unknown states.

### P3 — renderer and interaction maturity
- [ ] Native Bevy scene compile/run and rendered golden-image corpus.
- [ ] WebGPU adapter with successful-adapter detection and device-loss fallback.
- [ ] WebGL 2 fallback and deterministic SVG projection.
- [ ] Accessibility audit and mobile thermal/power evaluation.
- [ ] Optional glTF/GLB snapshot export; do not use it as the event or graph contract.

### P4 — research frontier
- [ ] Compare space-colonization and Physarum-inspired growth against a deterministic grammar baseline.
- [ ] Evaluate reaction-diffusion only for surface/material detail.
- [ ] Explore local graph adaptation under deletion, contradiction, and changing retrieval routes.
- [ ] Test whether a non-humanoid topology improves process comprehension and reduces anthropomorphic over-attribution.
- [ ] Publish MPES traces, scripts, scoring criteria, hardware metadata and negative results.

## 9. First implementation slice — source added

The first bounded slice is now present on the public research branch. It is **source-level implementation, pending exact-head formatting, compilation and test validation**.

Implemented in source:
1. `cognitive_topology.rs`: schema-v1 nodes, edges and episodes; bounded collections; explicit RuntimeEvent, RuntimeDerived and SyntheticFixture provenance; fail-closed validation for IDs, duplicate references, dangling links, event metadata consistency, sequence/time ordering, activation ranges, trace digests and resource limits.
2. `tests/fixtures/cognitive_topology_v1.json`: a clearly synthetic retrieval path, two competing associations and one unresolved prediction.
3. `CognitiveTopology::canonical_digest`: BLAKE3 digest over canonical ID-sorted semantic state, independent of array ordering and renderer configuration.
4. `cognitive_layout.rs`: a hash-anchored baseline plus a deterministic force-directed v1 refinement. The refinement uses bounded pairwise repulsion, relation-edge attraction, and a same-session previous-position regularizer. It does not change the semantic graph or convert epistemic status into force strength.
5. `CognitiveSpatialProjection` binds session identity and exposes a geometry digest over sorted stable IDs, link endpoints, and coordinates quantized to 0.001 world units.
6. `examples/cognitive_topology_replay.rs`: validates the fixture, emits semantic and geometry digests, and runs a CPU-only layout microbenchmark (20 warmups, configurable sample count, default 500; P50/P95 reported).
7. Unit-test source covers checked-in fixture parity, order-independent semantic/layout output, semantic mutation, dangling endpoints, renderer-local provenance rejection, conflicting metadata, non-monotonic times, invalid activation, oversized graph bounds, previous-session rejection, and bounded temporal anchor drift.
8. `cognitive_svg.rs`: a static, non-animated, accessible SVG graph view that distinguishes node kinds by shape, epistemic states by outline pattern, candidate/prediction links by line style, and retains source event/producer/sequence/provenance in inspectable attributes and titles.
9. The replay CLI accepts an optional third output path to write this static SVG. Its synthetic-fixture banner explicitly warns that it is not live cognitive telemetry.

This is not yet a smooth biological-style morphogenesis engine, live cognitive producer or human-validated instrument. The force layout is an initial relation-aware baseline, not a measured layout winner. The SVG is a static evidence inspection view, not a 3D GPU render or verified aesthetic result. CPU layout timings are not GPU frame or visual-quality results.

Remaining:
- exact-head formatting, compilation, all tests, Bevy adapter tests and Clippy;
- a real retrieval producer from its owning subsystem;
- static graph render and selectable provenance interaction;
- comparison against space-colonization and Physarum-inspired derived layouts;
- mesh junction quality and rendered-device benchmarks;
- a matched human-factors study.

Implementation and follow-on research are tracked in [#13066](https://github.com/Luminous-Dynamics/symthaea/issues/13066), [#13077](https://github.com/Luminous-Dynamics/symthaea/issues/13077), and [#13079](https://github.com/Luminous-Dynamics/symthaea/issues/13079).

## 10. Non-negotiable invariants

- Mycelial Presence is never an authority for actions, permissions, tools, payments, or system state.
- Runtime producers own semantic claims; renderers cannot infer them from text, timing, audio buffers, lighting, or motion.
- Unknown, stale, contradictory, or untrusted data never creates a new authoritative state.
- Pruning or LOD never destroys semantic evidence.
- Predictions remain visually and structurally distinct from observed or completed facts.
- Activation/frequency is not confidence, relevance, or truth.
- Renderer-local animation is always distinguishable from event-driven change and fully disableable.
- Benchmarks bind results to exact commit, schema version, input digest, model version, hardware and settings.
- CI pending, skipped, or cancelled is not a pass; compilation, rendered output, physical performance, and human evaluation are separate evidence classes.

## Research references checked 2026-11-21

- Bevy 0.20 release notes: https://bevy.org/news/bevy-0-20/
- W3C WebGPU publication history: https://www.w3.org/standards/history/webgpu/
- MDN GPU requestAdapter: https://developer.mozilla.org/en-US/docs/Web/API/GPU/requestAdapter
- Tero et al. (2010), Physarum network adaptation: https://doi.org/10.1126/science.1177894
- Distributed Physarum optimal transport (2026): https://doi.org/10.1038/s41598-026-61965-x
- Epic MetaHuman real-time animation: https://dev.epicgames.com/documentation/en-us/metahuman/realtime-animation
- NVIDIA Audio2Face-3D NIM: https://docs.nvidia.com/nim/digital-human/a2f-3d/latest/
- Live2D parameter control: https://docs.live2d.com/en/cubism-sdk-manual/parameters/
- Rive state machines: https://github.com/rive-app/help-center/blob/master/runtimes/state-machines.md
