# Mycelial Presence: Renderer Benchmark Protocol and Optimization Priorities

**Date:** 2026-11-21  
**Status:** Research protocol and engineering targets; no new renderer measurements are claimed.  
**Parent:** [PR #13012](https://github.com/Luminous-Dynamics/symthaea/pull/13012)  
**Related implementation:** [#13066 — CognitiveTopology v0](https://github.com/Luminous-Dynamics/symthaea/issues/13066)

## Executive decision

The next rendering improvement should not begin with a custom shader or GPU-computed morphogenesis. It should begin with **workload-aware representation, incremental updates, and fair benchmark instrumentation**.

Recent 2026 browser-engine comparisons show why:
- an October 2026 cross-device draw-call benchmark includes an RTX 2070 Windows machine and a Pixel 8 Pro, and reports materially different per-draw CPU costs between engines and APIs;
- a separate WebGPU/WebGL2 benchmark finds that instancing can radically reduce CPU cost, while WebGPU is not consistently faster on every workload;
- results change by browser, GPU, driver, engine version, and device thermal conditions.

These reports are not Mycelial Presence benchmarks and their workloads are not directly interchangeable with our scene. They are useful baseline evidence for selecting experiments, not evidence that one engine will win our workload.

**Primary optimization hypotheses:**
1. Keep semantic topology and event replay independent of renderer frame rate.
2. Batch static filament geometry and instance repeated terminal/junction/cell primitives where measurement confirms the benefit.
3. Update only dirty semantic subgraphs and their geometry ranges after a qualified event.
4. Introduce screen-space LOD and density budgets rather than merely changing fixed branch counts between modes.
5. Optimize for an explicit quality/performance/energy Pareto frontier, not a single FPS headline.

## 1. Current source-derived workload

The current renderer source creates a deterministic model with these branch counts:

| Mode | Primary roots | Filaments per root | Total filaments | Terminal filaments | Non-terminal filaments |
|---|---:|---:|---:|---:|---:|
| Ambient | 8 | 15 | 120 | 64 | 56 |
| Cognitive | 10 | 15 | 150 | 80 | 70 |
| Conversational | 8 | 7 | 56 | 32 | 24 |

The native Bevy preview tessellates each filament with 12 axial intervals and 6 radial segments. That gives 78 vertices and 432 indices per filament:

| Mode | Approx. tube vertices | Approx. tube indices |
|---|---:|---:|
| Ambient | 9,360 | 51,840 |
| Cognitive | 11,700 | 64,800 |
| Conversational | 4,368 | 24,192 |

These are arithmetic estimates from the checked-in algorithm/configuration, not measured GPU allocations or draw counts.

The scene also spawns separate junction, terminal spore, and orbital-cell entities that share mesh/material assets. Whether Bevy batches these effectively must be measured. Do not assume entity count equals draw-call count, and do not assume shared mesh/material guarantees the same batching behaviour on every render path.

## 2. External comparison evidence (checked 2026-11-21)

### 2.1 Cross-device WebGL2 / WebGPU draw-call benchmark

The 1 October 2026 web-engines-compare report measures PlayCanvas, Three.js, and Babylon.js on three systems, including:
- Windows: Intel i7-10700F and NVIDIA RTX 2070;
- Android: Pixel 8 Pro, Tensor G3 and Mali-G715;
- macOS: M4 Max.

The report's default draw-call workload uses 100 unique materials and 100 unique meshes, shadows, five warmup frames and twenty measured frames per count, with backend and engine versions pinned.

Its reported mean CPU microseconds per draw call include:

| Engine/version | Windows WebGL2 | Windows WebGPU | Pixel 8 Pro WebGL2 | Pixel 8 Pro WebGPU |
|---|---:|---:|---:|---:|
| PlayCanvas 2.23.0 | 1.66 µs | 1.22 µs | 2.16 µs | 1.63 µs |
| Three.js r186 | 2.10 µs | 4.68 µs | 2.43 µs | 5.66 µs |
| Babylon.js 9.29 | 4.83 µs | 7.34 µs | 5.93 µs | 9.25 µs |

Interpretation:
- Backend results depend on engine, device and workload.
- The data do not support a blanket claim that WebGPU is faster than WebGL2.
- This is a draw-call microbenchmark with particular scenes/settings; do not transfer its ratios to Bevy or Mycelial Presence.
- The RTX 2070 system is not necessarily comparable to our laptop's mobile-TGP RTX 2070, and the Pixel thermal window is deliberately limited.

Reference: https://github.com/mvaligursky/web-engines-compare/blob/main/results/2026-10-01/README.md

### 2.2 WebGPU/WebGL2 engine benchmark

The November 2026 WebGPU Engine Bench evaluates nine 3D web workloads with the same generated geometry, transforms, materials, lights, camera, 1920×1080 output, 4× MSAA and ACES tone mapping across browser engines.

The benchmark's own results emphasize that:
- optimized instanced paths can support tens of thousands of repeated meshes at much lower CPU cost than per-object paths;
- engine-specific fast paths matter substantially;
- the best backend depends on the workload;
- GPU timing support and browser frame caps complicate cross-browser comparisons;
- the published run is one Apple M1 Pro machine, so it cannot establish a universal engine ranking.

Reference: https://endel.github.io/webgpu-webgl-benchmarks/

### 2.3 Dynamic graph visual stability

A 2026 survey of mental maps and visualization describes the tension between preserving positions for existing graph elements and maintaining the readability of a new graph state. Excessive position constraints can preserve familiarity while obscuring new structures.

A 2020 event-driven graph drawing paper and recent temporal graph work also use node movement, crowding and other event-based criteria to evaluate evolving layouts.

Implication for Mycelial Presence:
- stable semantic IDs must retain spatial identity where possible;
- new edges must still be allowed to form legible pathways;
- layout stability and local clarity should be optimized jointly, not one at the expense of the other;
- benchmark both visual displacement and actual task performance.

References:
- 2026 survey: https://doi.org/10.1186/s42492-026-00225-1
- Event-based dynamic graph drawing: https://doi.org/10.1111/cgf.14615
- Stable connected-component visualization: https://doi.org/10.1177/1473871620972339

### 2.4 Biological growth inspiration

Runions, Lane and Prusinkiewicz's 2007 space-colonization algorithm uses spatial attraction points and competition for space to produce controllable branching structures. This is well suited to exploring how semantic anchors may guide morphology, provided those anchors are already present in the evidence graph.

A 2026 Scientific Reports article applies Physarum-inspired local sensing and trail reinforcement to adaptive multi-robot communication networks. It is a useful prior for an experimental topology-adaptation algorithm, not proof that such a method is suitable for cognitive visualization.

References:
- https://doi.org/10.2312/NPH/NPH07/063-070
- https://www.nature.com/articles/s41598-025-33456-y

## 3. Architecture: asynchronous semantics, incremental morphology, renderer-owned frame loop

Use three clocks with explicit boundaries:

### Clock A — semantic event clock

The owning runtime emits ordered, provenance-bearing events. These are replayable and authoritative for what happened. Rendering frame rate never changes event acceptance or graph semantics.

### Clock B — morphology/layout clock

A deterministic, bounded worker converts semantic topology deltas into spatial updates. It can coalesce geometry work, but it cannot coalesce away semantically distinct events from the authoritative trace.

A layout revision should:
- preserve all unchanged semantic IDs;
- move existing anchors only when the configured layout policy permits it;
- identify newly introduced/changed elements;
- emit a geometry delta with source event IDs and before/after digests;
- remain reproducible from the topology plus layout algorithm/version/parameters.

### Clock C — render/presentation clock

The renderer interpolates toward the latest accepted geometry state and draws at the display's cadence. Renderer-local interpolation is not a new semantic event.

This separation makes a stable 60 FPS ambient display possible even while no cognitive events occur, and prevents renderer stalls from corrupting the semantic event stream.

## 4. Optimization hypotheses and implementation experiments

### H1 — batch/instance repeated primitives

Current repeated terminal spores, junction bulbs and orbital cells share geometry/materials but are spawned as separate ECS entities.

Experiment:
- baseline existing entity path;
- shared mesh/material with renderer's normal batching;
- explicit instanced representation if the Bevy 0.20 API and rendering path allow it;
- merged static geometry as a control.

Measure draw calls, render CPU, GPU time, memory, first-frame cost, upload cost, and visual differences. Keep the simplest path that meets the target. Do not add custom instancing code if the standard renderer already batches the workload equivalently.

### H2 — per-filament dirty ranges

The current tube mesh already records stable per-filament vertex/index ranges. Use them as a foundation for partial updates:
- unchanged branches retain their buffers;
- new/changed branches upload only affected ranges;
- completed event paths may alter an overlay/material parameter without rebuilding geometry;
- LOD changes rebuild only the affected representation.

Compare full rebuild, whole-mesh upload, and range-based update on identical event traces. Keep the canonical semantic model CPU-owned and renderer-neutral.

### H3 — screen-space adaptive LOD

Mode selection is currently a coarse user-facing complexity preset. Add a separate screen-space quality policy:
- high curvature and selected/provenance-pinned branches receive higher tessellation;
- distant/occluded low-salience branches receive fewer radial/axial segments;
- terminal particles are density-limited per projected pixel;
- selected evidence paths are never removed from the semantic model, even if their rendering is simplified;
- the same model digest must survive every LOD choice.

Report geometry count, edge/filament visibility, shape error, semantic selection accuracy and frame time per LOD.

### H4 — temporal-coherent morphogenesis

For incremental graph evolution, compare:
1. deterministic golden-angle baseline;
2. graph-aware stable anchor layout;
3. space-colonization layout driven by explicit node anchors;
4. Physarum-inspired local adaptation on an already validated graph.

Measure node movement, branch displacement, edge crossings, angular resolution, path visibility, local readability, update cost and user ability to track a selected node/path. A visually organic result that makes a known memory jump to a new location on every update is a regression.

### H5 — material detail is resolution-adaptive

Use reaction-diffusion/noise only for material surface patterns, not semantic topology. Compare standard PBR materials first, then custom WESL rim/transmission and restrained subsurface-like approximation. Evaluate:
- image-space perceptual difference at target resolutions;
- shader compile/startup time;
- GPU frame time;
- device compatibility;
- low-power/mobile quality.

Do not require a custom shader if a conventional PBR material provides equivalent visual quality.

## 5. Mycelial Presence Renderer Benchmark (MPRB)

All renderers receive the same:
- schema-versioned canonical topology and MycelialModel;
- quantized geometry input;
- event trace and time-step schedule;
- viewport/resolution/device-pixel ratio;
- camera path;
- material/lighting budget;
- animation/reduced-motion mode;
- warmup and measurement schedule.

Include SVG, Bevy 0.20 native renderer, browser WebGL2 and browser WebGPU. Optional engine comparisons (PlayCanvas, Three.js, Babylon.js) are useful for renderer microbenchmarks but do not become the semantic system of record.

### Test workloads

| Workload | Purpose |
|---|---|
| Idle ambient baseline | Startup, stable frame cost, power and beauty |
| Dense canopy | Tube geometry and vertex/index throughput |
| Spore/junction ladder | Entity count, instancing and draw-call sensitivity |
| Sparse event delta | Event-to-visible update latency |
| Burst events | Queue behaviour, update coalescing and backlog |
| Graph growth | Temporal layout stability and visual continuity |
| Selection/follow path | Provenance lookup and highlighting latency |
| Reduced motion | Verify zero residual movement and lower idle cost |
| Device loss/fallback | Recovery correctness and semantic-state preservation |
| 30-minute idle + 30-minute active | Memory, thermal behaviour and energy |

### Device matrix

Minimum:
- Linux/Windows desktop or laptop with NVIDIA RTX 2070-class hardware, recording exact laptop GPU TGP, driver and CPU;
- Pixel 8 Pro, recording browser, Android version, display resolution, thermal state and power mode;
- one integrated-GPU/low-power desktop or laptop;
- one Apple GPU device if available for an independent driver/API family.

Do not infer Pixel 8 Pro capabilities from desktop tests or assume WebGPU wins on mobile.

### Metrics

Report distributions, not a single average:
- frame-time P50/P95/P99 and low-percentile FPS;
- CPU submission time and GPU frame time separately;
- draw calls, pipeline switches, vertices, triangles, instances and buffer uploads;
- GPU/CPU memory, allocations and resource lifetime;
- cold startup, time to first visible frame and shader/pipeline compilation;
- event-to-first-visible-frame P50/P95/P99;
- geometry-delta compute time and bytes uploaded;
- event queue depth, backlog age, stale/rejected events and replay divergence;
- energy per minute and temperature/thermal throttling;
- renderer startup/fallback/device-loss rates.

### Proposed engineering targets

These are targets, not achieved measurements:

| Metric | Desktop | Mobile |
|---|---:|---:|
| Sustained P95 frame time | ≤16.7 ms | ≤33.3 ms |
| P99 frame time | ≤33.3 ms | ≤50 ms |
| P95 event-to-visible frame | ≤50 ms | ≤100 ms |
| P95 geometry-delta computation | ≤5 ms | ≤10 ms |
| Unbounded queue growth | 0 | 0 |
| Accepted stale/out-of-order semantic events | 0 | 0 |
| Reduced-motion residual animation | 0 | 0 |
| Memory growth after warm steady state | no unbounded growth | no unbounded growth |

These targets are not a universal SOTA assertion. They are release gates for our chosen device classes and workloads, subject to calibration after the first measurements.

## 6. Semantic and visual correctness are co-equal gates

Performance optimization is invalid if it changes what the organism claims.

- Semantic topology digest is independent of renderer, camera, LOD, colour, bloom, animation phase and frame rate.
- Geometry deltas preserve the source semantic IDs and event references.
- Predictions remain candidates; a rendering shortcut cannot upgrade epistemic status.
- A selected evidence path stays traceable when geometry is hidden or simplified.
- Coalescing render updates cannot delete or reorder authoritative events.
- Replaying the same event trace and layout version reproduces the same semantic graph and deterministic layout output (or a documented numeric tolerance where floating-point implementations differ).
- Reduced motion disables ambient, orbital, breathing, twinkle and interpolated semantic motion; static state remains legible.

## 7. Human-factors quality

Use a blinded, counterbalanced study. Benchmark:
- time to locate a selected memory or relation;
- accuracy following a retrieval/prediction path after incremental updates;
- accuracy distinguishing observed, derived and candidate states;
- provenance/source comprehension;
- false belief that ambient animation indicates internal activity;
- subjective organic plausibility, visual comfort and aesthetic preference;
- ability to interpret the instrument in reduced-motion and colour-vision-accessible modes.

Include task types where stable positions help and tasks where new relationships require clear movement. Do not assume a stable mental map always improves every task: the design should find the appropriate balance.

## 8. Next engineering sequence

1. **Compile baseline:** fix the exact source errors exposed by focused validation; do not weaken tests or switch off the adapter.
2. **Reference frame:** capture Ambient/Cognitive/Conversational, both motion settings, on a named GPU device.
3. **Counters first:** add reproducible counters for entities, draw calls, buffer bytes, triangles, frame time and CPU update cost.
4. **Primitive batching experiment:** measure the current spore/junction/cell path, then compare instancing or merged geometry.
5. **Incremental event update prototype:** use a synthetic fixture only, with explicit synthetic provenance; compare full rebuild to dirty-range update.
6. **CognitiveTopology integration:** use issue #13066; preserve event provenance and stable IDs.
7. **Layout experiment:** baseline graph layout vs space-colonization vs Physarum-inspired derived layout, using identical semantic topology.
8. **Browser renderer:** start with a minimal equivalent workload in WebGL2 and WebGPU; test adapter/device failure and fallback.
9. **MPRB release report:** publish raw machine-readable results, scripts, commit/model/trace digests and all failures.

## 9. Current limitations and nonclaims

- No successful compile/test/Clippy result is asserted by this document.
- No native rendered frame is yet available as a verified artifact.
- No local RTX 2070 laptop or Pixel 8 Pro performance numbers are claimed.
- No WebGPU or WebGL2 browser adapter is implemented in this increment.
- No instancing or partial-range renderer optimization is claimed implemented.
- No human-factors or accessibility study has been conducted.
- External engine benchmark numbers are not directly transferable to Bevy or our custom scene.
- Physarum- or plant-inspired geometry is a procedural analogy, not evidence that Symthaea has biological growth or cognition.

## References

- Web-engine draw-call comparison, 1 October 2026: https://github.com/mvaligursky/web-engines-compare/blob/main/results/2026-10-01/README.md
- WebGPU/WebGL2 Engine Bench: https://endel.github.io/webgpu-webgl-benchmarks/
- 2026 survey of mental maps and visualization: https://doi.org/10.1186/s42492-026-00225-1
- Event-Based Dynamic Graph Drawing without the Agonizing Pain: https://doi.org/10.1111/cgf.14615
- Stable visualization of connected components in dynamic graphs: https://doi.org/10.1177/1473871620972339
- Runions et al., Modeling Trees with a Space Colonization Algorithm (2007): https://doi.org/10.2312/NPH/NPH07/063-070
- Physarum-inspired decentralized mesh networks for multi-robot systems (2026): https://www.nature.com/articles/s41598-025-33456-y


## 2.5 Bevy 0.20 direct browser-engine comparison (11 October 2026)

A particularly relevant independent comparison now exists: **"Six 3D engines measured in the browser"** compares pinned Bevy 0.20.0, Three.js r186, Babylon.js 9.30.0, PlayCanvas 2.23.2, Godot 4.7.2 and DriftEngine 4.13.0. The DriftEngine vendor authored the comparison, so treat it as a useful disclosed benchmark, not a neutral certification. The page publishes the methodology, scene captures, engine versions and where the author’s engine loses.

Reference: https://driftengine.dev/blog/webgpu-engines-measured (11 October 2026).

### Desktop workload results relevant to Mycelial Presence

Desktop test host: AMD Radeon RX 9070 XT, Chrome/Vulkan, 1280×720, pixel ratio 1, vsync/frame cap off; 120 warmup frames then up to 600 timed frames, medians and P95 values.

| Workload | Bevy 0.20.0 WebGPU | Bevy 0.20.0 WebGL2 | Interpretation |
|---|---:|---:|---|
| 1,000 separate objects median / P95 | 3.12 / 4.02 ms | 2.41 / 3.29 ms | Automatic batching makes entity count different from draw-call count |
| 10,000 separate objects median / P95 | 10.58 / 11.75 ms | 10.77 / 12.04 ms | Bevy performed strongly on this particular scene |
| One mesh repeated as instances, 100,000 median / P95 | 24.12 / 25.65 ms | 94.71 / 97.27 ms | The instancing API/path was substantially different from separate-object batching |
| One mesh repeated as instances, 1,000,000 median / P95 | 211.10 / 218.15 ms | 895.04 / 908.40 ms | Neither path reaches 60 FPS at this scale |
| Sponza scene frame time | 4.17 ms | 9.45 ms | Different, richer scene; not directly comparable to our canopy |

The test’s explanation matters: Bevy automatically batches separate objects, but the one-object-many-instances case includes entity processing and Bevy’s automatic instancing; render and ECS costs were not separately isolated. The benchmark therefore supports two separate hypotheses, not one blanket conclusion:
1. Standard Bevy ECS entities can batch effectively when they share the right assets/material path.
2. Bevy’s explicit instancing/instance-processing path may be a distinct bottleneck and must be measured separately.

### Phone workload results relevant to the mobile renderer

Phone host: Samsung Galaxy S23 Ultra, Adreno 740, Chrome Android, USB-tethered, 1280×720, pixel ratio 1, vsync on.

| Workload | Bevy WebGPU | Bevy WebGL2 |
|---|---:|---:|
| Separate objects held at 60 FPS | 12,288 | 12,288 |
| One mesh repeated as instances held at 60 FPS | 53,248 | 13,824 |
| 256-light workload held at 60 FPS (reported limit) | 624 | 4,096+ |
| Sponza frame time | 25.0 ms | 20.8 ms |
| Sponza first frame | 6,850 ms | 7,659 ms |

The phone result is an especially important warning: Bevy’s best desktop/backend ordering does not transfer directly to a mobile GPU. The report also says phone temperature can move the ceilings. Its Galaxy S23 Ultra is not our Pixel 8 Pro, and its desktop RX 9070 XT is not our RTX 2070 laptop; the figures must remain device-specific external reference points.

### Benchmark consequence

Add two distinct primitive workloads to MPRB:
- **Many ECS primitives with shared mesh/material:** measures Bevy's standard batching path for terminal spores, junction bulbs and orbital cells.
- **One mesh, many explicit instances:** measures instance submission/processing separately from ECS entity count.

Also report time-to-first-visible-frame and browser payload size for any future Bevy WebAssembly delivery. The external benchmark measures several seconds to first Sponza frame for Bevy web builds; native desktop startup is a different path and must not be conflated with browser cold start.

Do not conclude that Bevy is universally faster/slower from these rows. Use this source to make the Mycelial workload design more representative and to insist on separate CPU/ECS, draw-submission, GPU, startup and mobile thermal measurements.
