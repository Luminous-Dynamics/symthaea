# Mycelial Presence: Renderer Evolution

**Date:** 2026-11-21  
**Status:** Spatial model v1 implemented; GPU renderer is a proposed next adapter  
**Implementation branch:** `research/symthaea-embodiment-contract-v0`

## Decision

**Keep SVG, but stop treating SVG as the canonical organism.** The canonical representation is now a deterministic, renderer-neutral 3D model. SVG is its portable 2D projection, inspection format, snapshot/debug surface, and compatibility fallback. A future Bevy/WGPU renderer can consume the same spatial filaments, orbital cells, and evidence-bearing runtime snapshot.

This is a format evolution, not a format replacement.

The distinction matters:

- A *model* defines identity, topology, geometry, and the observed signals that are permitted to influence presentation.
- A *renderer* chooses tessellation, materials, lighting, shaders, camera, and frame scheduling.
- A *runtime adapter* remains the only authority for cognitive activity, speech playback, gaze, and uncertainty.
- A renderer-local animation must never be represented as evidence of a runtime event.

## Why not replace SVG with one new format?

| Technology | Best role | Limitation for Mycelial Presence |
|---|---|---|
| SVG | Portable vector output, inspectable geometry, deterministic snapshots, 2D fallback | DOM-shaped scene graph and filters are not the right foundation for rich volumetric filaments, high particle counts, or 3D camera motion |
| Canvas 2D | Pixel effects and a simple browser raster surface | Does not itself provide a 3D scene, retained semantic nodes, or accessibility; much of the renderer must be written manually |
| WebGL 2 | Broadly deployed browser GPU rendering | Mature and useful fallback, but requires a separate GPU-oriented adapter and explicit resource/shader management |
| WebGPU / `wgpu` | Modern programmable GPU rendering, compute, instancing, and a path shared by native Rust and supported browsers | Browser support and platform/device availability are not uniform; initialization can fail even on a browser exposing the API |
| Bevy | Native real-time 3D scene, cameras, meshes, materials, lighting, input, and lifecycle | Larger engine/runtime footprint; the existing repository dashboard is still on Bevy 0.15 and is not yet wired to the embodiment contract |

SVG and Canvas are both useful, but replacing SVG with Canvas alone would not produce the qualitative leap requested. The significant step is spatial geometry plus a GPU pipeline.

## Canonical spatial model v1

Implemented in `crates/domains/symthaea-canvas/src/mycelial_model.rs`:

- Versioned `MycelialModel` output with stable deterministic generation.
- Cubic 3D Bézier filaments with stable IDs, parent IDs, generation, terminal markers, and tapered start/end radii.
- A low-discrepancy golden-angle primary arrangement, deterministic branching, and bounded 3D control points.
- Orbital cells with spatial depth.
- The observed activity, speech lifecycle, viseme, gaze, and uncertainty values remain attached to the model rather than being converted into inferred renderer state.
- A viseme is forwarded only when the snapshot's speech lifecycle confirms `Playing`.
- User-selected `Ambient`, `Cognitive`, and `Conversational` modes change presentation complexity, not the meaning or provenance of runtime events.

The SVG adapter now projects these same spatial curves into a lightly depth-compressed 2.5D view. It no longer invents an independent 2D branching topology.

The v1 model does **not** yet emit a GPU mesh, volume texture, or shader material. It gives future renderers an authoritative spatial input from which to build those assets. The 3D renderer is therefore planned, not delivered.

## Rendering architecture

```text
Runtime event adapters
        |
        v
Evidence-aware EmbodimentSnapshot + trace
        |
        v
MycelialModel v1 (renderer-neutral 3D curves + observed signals)
        |
        +--> SVG projection (current, portable fallback/debug/export)
        |
        +--> Bevy / wgpu renderer (next native 3D adapter)
        |
        +--> WebGPU renderer (later browser GPU adapter)
        |
        +--> static / reduced-motion renderer
```

### Native renderer

The preferred native experiment is a separate Bevy 0.20 adapter, not a rushed migration of the existing dashboard:

1. Convert cubic Bézier centre-lines into tapered tube or ribbon meshes. Parallel-transport frames are preferable to naïve Frenet frames for stable tube orientation near low curvature.
2. Keep topology and mesh identity stable; update per-frame uniforms and a small set of dynamic buffers rather than rebuilding the entire mesh every frame.
3. Use a restrained emissive core, thin transmissive membrane, grazing-angle rim light, and depth-aware branch materials. Avoid indiscriminate bloom, which destroys branch separation.
4. Instance terminal spores and orbital cells; apply level-of-detail and a strict particle budget on mobile.
5. Animate from the shared monotonic frame context. Reduced-motion mode must disable camera drift, branch breathing, orbit animation, and particle motion.
6. Keep camera motion user-controlled or very subtle. The organism should remain legible without motion.

The current `symthaea-bevy-dash` package is on Bevy 0.15 and currently visualizes a separate cognitive dashboard. Bevy 0.20 adds sprite material extensions and WESL shader support. Its release notes also state that mesh shaders are not supported on web platforms. Treat the Bevy adapter as native-first; do not make advanced mesh shaders a requirement for browser parity.

The workspace currently declares `wgpu = 27`, while Bevy 0.20's current render crate manifest uses `wgpu = 30`. Do not casually share GPU resources or force a workspace-wide version bump. Keep renderer integration behind a separate crate boundary until its dependency graph, feature set, platform matrix, and build costs are measured.

### Browser renderer

Use feature detection and a deliberate fallback ladder:

1. WebGPU when `navigator.gpu` and an adapter/device are actually available.
2. WebGL 2 for capable browsers without WebGPU.
3. SVG for static/low-power/unsupported environments.

Do not infer support from user-agent strings. The WebGPU API is still marked as not Baseline by MDN and is only available in secure contexts in supporting browsers. Adapter acquisition can fail, so the fallback must be tested, not merely documented.

Do not use Bevy's non-web mesh-shader path as a browser requirement. Use conventional vertex/fragment pipelines for cross-platform geometry until the supported browser/device matrix proves otherwise.

## Cognitive visualization without invented telemetry

The current embodiment event contract supports activity, speech output, viseme, gaze, and explicitly supplied uncertainty. These signals now have visual channels in the model and SVG presentation.

A future Cognitive Instrument should render actual retrieval/association graphs, prediction branches, planning trajectories, and tool execution only after their owning runtime subsystems publish typed, ordered, provenance-bearing events. The renderer must not infer those graphs from the conversational text, elapsed time, color, or decorative filament motion.

The required expansion is a separate set of typed graph/trajectory events with:
- session sequence and monotonic timestamp;
- source subsystem and signal provenance;
- stable node/edge/trajectory identifiers;
- explicit lifecycle transitions (created, updated, completed, cancelled, failed);
- bounded sizes and finite coordinates/weights;
- trace/replay fixtures and negative controls.

Until those producers exist, the current mode may display only the signals that are actually present. Decorative branches are not retrieval edges.

## Visual quality improvements to pursue in the GPU adapter

- **Organic morphology:** tapered branching tubes with curvature-continuous child junctions, non-uniform branch lengths, deterministic local asymmetry, and a controlled range of thickness.
- **Depth and material:** a true 3D nucleus with a translucent outer membrane, internal layered core, subtle Fresnel rim, and restrained subsurface-like scattering approximation.
- **Spatial light:** soft key/fill/rim arrangement, depth-aware fog only if it helps separation, no global overexposure.
- **Causal motion:** only observed runtime events alter semantic channels; renderer-local motion is slow, deterministic, and independently labelled.
- **Topology as an instrument:** later, qualified retrieval edges and prediction trajectories may pulse along the matching filament paths, with provenance discoverable on selection.
- **Accessibility:** captions/text labels, static representation, reduced motion, contrast checks, and user control over visual intensity.

## Benchmark plan

No performance result is claimed by this design note. The following are targets and measurement requirements.

### Matched renderer benchmark

Render the same versioned `MycelialModel` and same sequence of `EmbodimentSnapshot` events through SVG, WebGL 2, WebGPU, and native Bevy. Fix viewport sizes, output scale, quality settings, device, driver, power mode, and scene/event trace.

Report:

- frame-time P50/P95/P99 and low-percentile FPS, not average FPS alone;
- CPU time, GPU time, memory/VRAM, allocation count, and shader/mesh compilation time;
- event-to-first-visible-frame latency P50/P95/P99;
- model-build, scene-update, render-submit, and present costs separately;
- dropped or stale semantic events and visual-state divergence;
- energy per minute and thermal throttling on a mobile device;
- first-run and warmed-up results, plus renderer fallback frequency.

### Initial targets (not achieved results)

| Measure | Desktop target | Mobile target |
|---|---:|---:|
| Sustained frame-time P95 | <= 16.7 ms | <= 33.3 ms |
| Event-to-first-visible-frame P95 | <= 50 ms | <= 100 ms |
| Stale/reordered accepted events | 0 | 0 |
| Invalid viseme outside confirmed playback | 0 | 0 |
| Reduced-motion residual animation | 0 | 0 |
| Nondeterministic geometry for fixed model inputs | 0 | 0 |

Targets must be revised only after measuring real supported hardware. Quality/performance trade-offs must be reported per mode, not hidden in a single score.

### Correctness benchmark

- Deterministic model bytes/geometry for a fixed schema version, mode, and snapshot.
- Parent-before-child topology, finite coordinates, bounded radii, unique IDs, and valid terminal count.
- Event replay produces the same semantic model state.
- Queued/synthesized audio never creates speech-pulse or viseme evidence.
- Renderer changes do not change event acceptance, sequencing, provenance, or trace hashes.
- SVG and GPU output may differ artistically, but both must preserve the same state and event semantics.

## Phasing

1. **Completed in this increment:** canonical renderer-neutral 3D spatial model; deterministic branching; shared SVG projection; three presentation modes; gaze and confirmed-playback layers.
2. **Next:** render-golden and model-hash fixtures; validate SVG output in Chromium and reduced-motion; compare mode node/geometry budgets.
3. **Native GPU prototype:** a small, isolated Bevy 0.20 adapter that consumes `MycelialModel`, draws 3D tubular filaments and the nucleus, and reports frame metrics. Keep it out of the core semantic crate.
4. **Browser GPU adapter:** WebGPU with WebGL 2 fallback and SVG fallback; capability detection and device-loss tests.
5. **Cognitive Instrument extension:** only after real producers for retrieval, association, prediction, and planning traces are available.
6. **Conversational Puppet:** a separate, optional embodiment adapter (e.g. VRM/Live2D) sharing the event contract; never force the mycelial identity into a humanoid rig.

## References checked 2026-11-21

- Bevy 0.20 release notes: https://bevy.org/news/bevy-0-20/
- Bevy 0.20 render dependency manifest: https://github.com/bevyengine/bevy/blob/main/crates/bevy_render/Cargo.toml
- MDN WebGPU API support and secure-context notes: https://developer.mozilla.org/en-US/docs/Web/API/WebGPU_API
- W3C WebGPU Candidate Recommendation Draft, 2026-11-16: https://www.w3.org/TR/2026/CRD-webgpu-20261116/
- Cross-device/browser animation evaluation: https://doi.org/10.3390/jimaging12010045
