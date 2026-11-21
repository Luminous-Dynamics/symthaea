# Symthaea Embodiment Programme — Research and Engineering Plan

**Date:** 2026-11-20  
**Status:** Research plan + first semantic-contract increment  
**Implementation branch:** research/symthaea-embodiment-contract-v0  
**Scope:** Canonical identity, renderer-neutral puppet protocol, cognitive-state fidelity, evaluation, and competitive positioning.

## Executive thesis

Symthaea should not compete first on photorealistic digital humans. NVIDIA, Epic, and specialist avatar vendors already invest heavily in facial reconstruction, audio-driven animation, and production rendering.

The stronger research opportunity is **evidence-connected embodiment**: a renderer-independent interface that lets a system communicate its actual interaction and cognitive-runtime events while preserving provenance, uncertainty, temporal order, and limitations.

The avatar is a presentation layer, not the source of truth. The runtime remains authoritative.

The programme has three synchronized embodiment families:

1. **Mycelial Presence** — an abstract, non-human, bioluminescent identity for ambient and low-distraction interaction.
2. **Conversational Puppet** — an optional expressive face or character for direct dialogue, with gaze, phoneme/viseme motion, and accessible controls.
3. **Cognitive Instrument** — a transparent view of active runtime signals, memory/retrieval associations, predictions, tool lifecycle, and uncertainty.

All embodiments share one semantic event contract and one identity system. They do not need to share geometry, skeletons, shaders, or rendering engines.

## What exists already in Symthaea

Repository inspection on 2026-11-20 found that this work should extend rather than replace existing components:

- crates/domains/symthaea-canvas already defines CognitiveSnapshot, an aesthetic engine, a declarative scene graph, SVG rendering, and temporal animation helpers.
- CognitiveSnapshot already carries selected telemetry such as prediction error, topology values, affect dimensions, harmony activations, and cycle count.
- animation.rs already supplies a monotonic frame context and frame-rate-independent smoothing helpers.
- symthaea-vocal-tract has speech synthesis pathways, but that is not itself a viseme/face-animation bridge.
- Android's symthaea-soma demo already has visual embodiment components and an avatar-selection area.

The existing canvas crate documents a <500µs pipeline target. That is a repository target, not an independently verified performance result. The new work should reuse its temporal contract and avoid claiming that a rendered visualization proves an underlying cognitive or experiential state.

Repository references:
- https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-canvas/src/lib.rs
- https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-canvas/src/animation.rs
- https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-canvas/Cargo.toml
- https://github.com/Luminous-Dynamics/symthaea/blob/main/crates/domains/symthaea-vocal-tract/src/speech.rs
- https://github.com/Luminous-Dynamics/symthaea/tree/main/crates/domains/symthaea-soma/android/demo

## Research landscape (checked 2026-11-20)

### NVIDIA

- NVIDIA Audio2Face-3D is a serious production baseline for audio-driven facial animation, ARKit-compatible blendshape streaming, and real-time digital-human applications.
- Its current performance documentation uses the **1st percentile FPS** to define max stream capacity: the published stream count is chosen so that 99% of measured FPS remain above 30. The tables are tied to specific GPUs and complete server configurations. This is a better capacity reference than average FPS, but it is not directly comparable to a local SVG renderer or a Pixel device.
- The same performance page exposes first-packet latency distributions and successful-request counts. SEFB should borrow this reporting discipline: event-to-first-visible-frame P50/P95, frame-time P1/P5, number of concurrent streams meeting a quality threshold, dropped-event counts, and a fixed hardware/software manifest.
- NVIDIA ACE/Audio2Face is therefore a strong optional backend and comparative baseline, not a replacement for Symthaea's semantic event contract.

Sources:
- https://docs.nvidia.com/ace/audio2face-3d-microservice/latest/text/interacting/performance.html
- https://docs.nvidia.com/nim/digital-human/a2f-3d/latest/
- https://developer.nvidia.com/ace-for-games

### Epic MetaHuman

Epic documents real-time MetaHuman animation from mono cameras, mobile capture, and audio sources. The documentation makes an important distinction: the Realtime Audio Solver inside an audio-driven Performance asset is still an offline processing workflow, and differs from a live MetaHuman Audio Live Link Source. The real-time audio solver also does not produce the same head-motion controls as the broader offline solve. MetaHuman is a high-quality Unreal-centered reference, not a lightweight cross-renderer core.

Sources:
- https://dev.epicgames.com/documentation/en-us/metahuman/realtime-animation
- https://dev.epicgames.com/documentation/en-us/metahuman/audio-driven-animation
- https://dev.epicgames.com/documentation/metahuman/metahuman-animator-in-unreal-engine

### VRM, VRMA, and Live2D

- VRM 1.0 defines humanoid structure, facial expressions, and gaze controls. Its specification also defines an execution order for humanoid bones, gaze, expressions, constraints, and spring bones.
- VRM Animation 1.0 (.vrma) is a glTF extension for reusable humanoid bone, expression and gaze animation. Its stated goal is to let the same animation file apply across compatible VRM models and applications.
- VRM 1.0 specifies a recommended update order for humanoid bones, gaze, expressions, constraints and spring bones. A future adapter should respect that order rather than inventing a competing skeletal evaluation pipeline.
- Live2D Cubism provides parameter-driven animation and lip-sync input paths that are useful references for a lightweight 2D/2.5D puppet.

Use these as optional adapters, not as the universal canonical model: the Mycelial Presence is not inherently humanoid.

Sources:
- https://github.com/vrm-c/vrm-specification/tree/master/specification/VRMC_vrm-1.0
- https://vrm.dev/en/vrma/
- https://github.com/vrm-c/vrm-specification/blob/master/specification/VRMC_vrm_animation-1.0/README.md
- https://docs.live2d.com/en/cubism-sdk-manual/lipsync/

### Additional 2026 speech-driven avatar baselines

The 2026 literature adds three distinct backends that should be represented in the comparison matrix:

- **Let Toon Talk** (Applied Sciences, 13 May 2026) combines parametric adaptation for heterogeneous cartoon geometry with identity-adaptive flow matching for speech-driven motion. On its MEAD evaluation it reports MVE 1.17 mm, LVE 5.19 mm, FDD 0.11, and per-frame motion inference time 1.30 ms. On VOCASET it reports MVE 1.56 mm, LVE 4.50 mm, FDD 0.10, and 1.22 ms/frame. These are the paper's own experimental conditions and should be treated as matched-baseline candidates, not universal deployment figures.
- **CartoonTalk** (IEEE TVCG 2026) combines 2D cartoon keypoints with 3D motion coefficients and reports state-of-the-art results on its cartoon datasets. The accessible abstract does not provide the metric table, so do not invent a numerical score. Its key comparison is cartoon-specific topology handling rather than ordinary human face animation.
- **AnyTalk / AnyTalk_RT** (IEEE TVCG 2026, early access) adapts a pretrained video-diffusion model to a target 3D character using rendered no-motion images, then estimates blendshape coefficients; the authors also describe a distilled real-time variant. This is a useful baseline for arbitrary character support and adaptation cost. It is not zero-shot in the strict no-optimization sense because the target character has a character-specific adaptation step.

These papers imply that the Conversational Puppet should not be tied to a single face geometry or motion model. The canonical Symthaea event protocol should drive interchangeable backends: a light parameter rig, VRM/VRMA, cartoon-specialized models, or arbitrary-character motion adapters.

Sources:
- https://www.mdpi.com/2076-3417/16/10/4840
- https://doi.org/10.1109/TVCG.2026.3687365
- https://pubmed.ncbi.nlm.nih.gov/42030183/
- https://dalab.sungshin.ac.kr/

### Evaluation literature and interaction quality

- A 2025 benchmark study of speech-driven 3D facial animation explicitly reports that the field lacks consensus on standardized datasets and evaluation methods. This is a direct reason to publish matched traces, hardware metadata, model versions, and measurement scripts rather than comparing unrelated paper numbers.
- A 2026 IEEE Computer Graphics and Applications overview formalizes avatar interoperability as asset containers, semantic control interfaces, and streaming/transport rules. It emphasizes end-to-end latency, robustness, temporal stability, signal-domain distortion, and multi-criteria subjective assessment. This maps closely to the proposed SEFB tracks.
- Established facial-animation metrics include LSE-D/LSE-C for lip synchronization, LVE/LMD for geometric articulation, and perceptual/video metrics for appearance. They do not measure runtime-state truthfulness.
- Human-likeness is not automatically better. A 2025 meta-analysis finds the effects of human-like social cues depend on moderators and context. A 2026 CHI study of AI-mediated video found reduced perceived trustworthiness and confidence in judgments under some conditions. These findings are not direct evaluations of Symthaea-like companions, but they justify testing transparency, embodiment choice, and anthropomorphic over-attribution rather than assuming a more human face improves trust.
- A commercial avatar vendor's 2026 benchmark uses live conversations, repeated within-participant comparisons, and post-session ratings. Its public methodology is a useful design reference, but its vendor sponsorship and full-pipeline scoring must be disclosed if used as a comparator.

Sources:
- https://onlinelibrary.wiley.com/doi/10.1111/cgf.70073
- https://pubmed.ncbi.nlm.nih.gov/42447007/
- https://www.mdpi.com/2078-2489/15/11/675
- https://www.nature.com/articles/s41599-025-05618-w
- https://doi.org/10.1145/3772318.3790845
- https://anam.ai/tools/avatar-quality-benchmark

### Facial fidelity benchmark references

For the optional humanoid backend, use established facial-animation datasets and metrics rather than inventing a Symthaea-only score:

- The 2025 ICCV MemoryTalker paper evaluates on VOCASET and BIWI using Face Vertex Error (FVE), Lip Vertex Error (LVE), Fréchet Inception Distance (FID), Lip Dynamic Time Warping (LDTW), and Lip-max. Its table reports LVE 0.293 on VOCASET and 0.187 on BIWI, with corresponding baseline comparisons against FaceFormer, CodeTalker, SelfTalk and UniTalker.
- The 2025 Computer Graphics Forum benchmark evaluates deterministic and non-deterministic models on BIWI, Multiface and 3DMEAD. It reports LVE, Mean Vertex Error (MVE), upper-face dynamics deviation (FDD), and diversity/coverage-oriented metrics for non-deterministic models, plus a 7-point human assessment of lip-sync and realism. The authors report that rankings change across datasets and objective scores do not always align with subjective ratings.
- Therefore, facial-backend SOTA claims must be dataset- and metric-specific. Compare exact dataset versions, subject splits, preprocessing, mesh topology, audio sampling, inference hardware and stochastic sampling protocol. Never use one paper's best table value as a universal target.

Sources:
- https://openaccess.thecvf.com/content/ICCV2025/papers/Kim_MemoryTalker_Personalized_Speech-Driven_3D_Facial_Animation_via_Audio-Guided_Stylization_ICCV_2025_paper.pdf
- https://onlinelibrary.wiley.com/doi/10.1111/cgf.70073

### Baseline policy for humanoid and cartoon puppets

Before attempting a facial-animation SOTA claim:

1. Reproduce one deterministic baseline and one expressive/probabilistic baseline on a fixed benchmark split.
2. Include 2026 comparisons where code/assets permit: Let Toon Talk, CartoonTalk and AnyTalk_RT, alongside FaceFormer, CodeTalker, UniTalker and MemoryTalker.
3. Report FVE/LVE/MVE/FDD, LDTW and LSE-D/C where their definitions apply; report diversity/MEE/CE for stochastic generators. Include LPIPS/SSIM/PSNR only for rendered-image tasks, not as substitutes for geometric articulation.
4. Report both motion-decoder latency and end-to-end first-visible-frame latency. Include per-character adaptation time, memory, GPU/CPU usage and the exact rendering workload.
5. Aim to reproduce baselines within their reported confidence intervals first. Only then set a research target of at least 10% lower LVE/LDTW or at least 10% better subjective lip-sync preference on the same split, while not degrading MVE/FDD, identity consistency, diversity or latency. Treat this as a target, not a claim.
6. Evaluate unseen identities separately from seen identities. Target a zero-shot renderer-neutral motion interface across at least 10 held-out avatar identities; report how any per-character fine-tuning changes that claim.

For Symthaea, the primary new contribution is still semantic fidelity. A facial engine can win LVE while the overall system loses if it lip-syncs audio while its speech output is only queued, shows a tool operation as completed before the tool returns, or depicts uncertainty that the runtime never emitted.

## Proposed architecture

### 1. Semantic control plane

Runtime adapters emit typed events with:
- strict per-session sequence number;
- monotonic timestamp;
- provenance (RuntimeEvent, RuntimeDerived, UserControlled, or RendererLocal);
- a typed event payload.

The event is the authority. Animation curves, facial expressions, glow, particles, and camera movement are renderer decisions.

### 2. Evidence-aware reduction

A session-scoped event sequencer assigns one monotonic timestamp and strictly increasing sequence across producers. A deterministic reducer turns the ordered events into a renderer-facing snapshot. It rejects timestamp regression, invalid normalized values, impossible speech-output transitions, invalid provenance, and viseme events outside confirmed playback.

The reducer must not infer listening, retrieving, associating, predicting, playback, or tool execution from visual timing or generic conversation text. Only the owning runtime adapter may publish those states.

### 3. Presentation adapters

| Adapter | First implementation | Later options |
|---|---|---|
| Mycelial Presence | Existing SVG scene graph + event-driven overlays | Bevy/WGPU volumetric filaments |
| Conversational Puppet | Lightweight 2D/2.5D rig and event-driven mouth/gaze | VRM 1.0 + VRMA, Live2D, custom glTF |
| Cognitive Instrument | Existing canvas visualization with explicit source labels | Interactive topology, retrieval and tool trace |
| Accessibility | Text/captions, reduced motion, static representation | Spatial audio and user-controlled haptics |

### 4. Identity and embodiment continuity

Keep stable identity traits separate from renderer-specific assets:
- canonical silhouette and core geometry;
- palette and material language;
- stable motion vocabulary;
- naming and mode semantics;
- user-selected embodiment preferences;
- accessible fallback.

The identity is not the mesh. A user should be able to move from the abstract presence to a face and back without losing the conversational session or semantic state.

### 5. Causal animation rules and orthogonal lifecycle channels

Cognitive activity and speech playback are deliberately separate. An internal retrieval or association can continue while audio plays; the avatar must not collapse both into one mutually exclusive "Speaking" state.

The semantic contract distinguishes:
- Cognitive activity: Listening, Perceiving, Retrieving, Associating, Predicting, Planning, ExecutingTool, AwaitingTool, Interrupted and Fault.
- Speech output lifecycle: Synthesizing → Ready → Queued → Playing → Completed, with explicit Failed, Interrupted and Unavailable exits.
- Articulation: viseme samples with their own timestamps and provenance; accepted only while the speech lifecycle is explicitly Playing.

The reducer validates speech-state transitions and rejects impossible sequences such as Queued → Completed without an observed Playing state. The only signal that authorizes lip motion is an explicit RuntimeEvent that playback has actually started. Synthesis completion and audio queue insertion are not playback proof.

Examples of acceptable mappings:
- Listening only after the audio/input adapter emits a listening event.
- Associating only while the relevant retrieval/association subsystem reports an active operation.
- Predicting only when a prediction operation is underway.
- Playing only after the playback backend confirms audio has started; the signal must not be inferred from generated PCM, an enqueued buffer, or a text response.
- VisemeChanged only while Playing and aligned to the audio timeline.
- ExecutingTool only while a tool invocation is active.
- Uncertainty shown only when an explicit uncertainty signal exists; do not manufacture it from a mood estimate.
- RendererLocal effects must remain distinguishable from runtime-derived evidence.

Never map internal affect values directly to human emotions such as happiness, sadness, or affection without a separately validated and clearly labeled interpretation layer.

## First engineering increment in this branch

Added to symthaea-canvas:

- embodiment.rs: typed cognitive activity, independent speech-output lifecycle, visemes, gaze, provenance, and uncertainty signals;
- a session-scoped monotonic event sequencer and `EmbodimentSession` facade: one publish operation assigns an event ID/time, validates and reduces it, and appends the accepted event to the session trace; rejected signals do not mutate the authoritative snapshot or trace (sequence gaps are allowed);
- strict sequence ordering, monotonic timestamps, provenance checks, finite [0, 1] validation and fail-closed speech-state transition validation;
- visemes accepted only while a RuntimeEvent confirms SpeechOutputState::Playing;
- stale visemes cleared when playback stops; reset clears transient state and speech lifecycle;
- negative controls for invalid provenance, invalid lifecycle transitions, stale events, and visemes outside confirmed playback;
- mycelial_presence.rs: deterministic non-humanoid presentation, activity glyphs, explicit uncertainty halo, gaze/playback overlays, and a 2.5D SVG projection;
- mycelial_model.rs: versioned renderer-neutral 3D cubic-Bézier filaments, parent topology, tapered radii, orbital cells, and provenance-bearing observed signals;
- mycelial_mesh.rs: bounded CPU tessellation into indexed tube geometry using parallel-transport frames, with per-filament mesh ranges and invalid/oversized-model rejection;
- three user-selected presentation modes (Ambient, Cognitive, Conversational) that alter visual complexity without changing runtime signal semantics;
- examples/mycelial_spatial_mesh.rs: inspect filament counts and generated mesh vertices/triangles for each mode;
- embodiment_trace.rs: bounded, versioned event corpus, deterministic replay, activity/playback interval summaries and exact event-index failure reporting;
- tests/fixtures/embodiment_trace_v1.json: a reference trace that covers listening, synthesis, ready, queueing, playback, visemes, completion, uncertainty and reset;
- examples/embodiment_trace_replay.rs: a standalone CLI that validates a trace, performs warmup and repeated replay, reports P50/P95 reducer latency, and emits a BLAKE3 digest plus platform metadata;
- low-amplitude hyphae expansion and asynchronously phased spore twinkle, with both disabled by reduced-motion handling;
- geometric motifs tested separately from palette changes to avoid relying on hue alone.

Run the semantic reducer benchmark with:

`cargo run -p symthaea-canvas --example embodiment_trace_replay -- --trace crates/domains/symthaea-canvas/tests/fixtures/embodiment_trace_v1.json --iterations 500`

The replay CLI measures event reduction only. It does **not** measure browser paint, SVG rasterization, audio playback, GPU frame time or human understanding.

Export the rendered presence from the same trace:

`cargo run -p symthaea-canvas --example mycelial_presence_demo -- --trace crates/domains/symthaea-canvas/tests/fixtures/embodiment_trace_v1.json --through-sequence 8 --output /tmp/symthaea-presence.svg --reduced-motion`

Repository source inspection found that `src/cognitive_loop/voice_channel.rs` returns synthesized PCM in `VoiceResponse`, while `src/voice/repl_voice.rs` contains separate playback paths (Rodio queueing and an optional live-voice backend). A returned PCM buffer or a successful queue append is not proof that audio is actually playing. This branch therefore defines the lifecycle contract but does not yet falsely wire synthesis completion to `Playing`.

**Not implemented yet:** live runtime producers, actual playback-start callbacks, audio-aligned viseme inference, a running frontend/session integration, VRM loading, live gaze tracking, platform packaging, and measured benchmark results. The SVG adapter now projects the canonical 3D model and remains the portable/debug renderer, not a completed conversational puppet. A native Bevy/WGPU scene, GPU material/shader, and browser WebGPU/WebGL adapter are still not implemented.

## Renderer evolution — 2026-11-21

The Mycelial Presence no longer treats SVG as its source of truth. A deterministic renderer-neutral spatial model and bounded CPU tube-mesh tessellator are now in the branch. SVG is retained as the inspectable, portable 2D projection and fallback. An isolated Bevy 0.20.1 native PBR preview adapter now consumes the same mesh, with topology-driven teal-to-violet vertex colour, terminal spores, an emissive nucleus, translucent membrane, orbital cells, and natural HDR bloom with filmic tonemapping. Its source has not yet been compiled or visually verified. Custom WESL materials, GPU-generated geometry and browser WebGPU/WebGL adapters remain future work.

This avoids a renderer-specific rewrite and keeps cognitive event semantics independent from geometry, materials, and camera motion. The new Bevy adapter is excluded from the root workspace to preserve the existing Bevy 0.15 dashboard and workspace `wgpu 27` boundary. A focused CI workflow now builds an isolated two-crate validation workspace, avoiding unrelated source-only orphan directories in the full repository workspace; the first failure identified and corrected a root-workspace discovery issue, and a new run is pending.

Detailed decision, browser/native compatibility considerations, resource limits, visual direction, and matched benchmark targets:
[MYCELIAL_RENDERER_EVOLUTION_2026-11-21.md](MYCELIAL_RENDERER_EVOLUTION_2026-11-21.md)

## Benchmark programme: Symthaea Embodiment Fidelity Benchmark (SEFB)

I did not identify an established leaderboard that jointly measures avatar realism, runtime-state fidelity, provenance honesty, accessibility, local performance, and user trust. A 2025 benchmark study specifically reports a lack of consensus in speech-driven 3D facial-animation evaluation. Create a matched, reproducible evaluation suite rather than claiming a new SOTA score from unrelated papers.

### Track A — semantic fidelity

| Metric | Measurement | Initial acceptance target |
|---|---|---:|
| Event-to-state correctness | Replay labelled event traces and compare semantic snapshots | 100% contract conformance |
| Provenance preservation | Confirm all displayed signals preserve their source class | 100% |
| Unsupported-state rate | Count visual claims without supporting events | 0 |
| Replay determinism | Same event trace produces byte-identical canonical JSON snapshots | 100% |
| Stale-event rejection | Inject duplicated/reordered/out-of-window events | 100% of defined negative controls rejected |

### Track B — temporal and rendering performance

| Metric | Measurement | Initial target |
|---|---|---:|
| Event-to-visible latency | Event receipt to first submitted frame, P50/P95 | P50 ≤33 ms; P95 ≤80 ms |
| Desktop rendering | P95 frame time on fixed reference hardware | ≤16.7 ms at 60 FPS |
| Mobile rendering | Sustained frame time on a named Android device | ≤33.3 ms at 30 FPS |
| Audio/viseme alignment | Playback-clock timestamp versus visible articulatory onset | P50 ≤40 ms; P95 ≤80 ms |
| Playback-state integrity | Attempted illegal transitions and mouth motion outside confirmed playback | 100% rejected / 0 unsupported mouth motion |
| Queue health | Event age, backlog depth, dropped/coalesced events | No unbounded backlog |
| Long-run stability | 30-minute thermal/memory/latency trace | No crash; bounded memory; report throttling |

Targets are proposed engineering gates, not current results or vendor-independent SOTA. Report hardware, renderer, model, asset complexity, resolution, thermal state, and whether preprocessing is included.

For facial animation, use more than one metric: LSE-D/LSE-C for audio-visual sync where compatible, lip landmark/vertex error where ground truth exists, and blinded human ratings for expressiveness. Do not conflate these measurements.

### Track C — human legibility and trust calibration

| Metric | Study design | Initial target |
|---|---|---:|
| Cognitive-state recognition | Blind participants classify idle/listening/associating/predicting/executing | ≥90% macro-F1 |
| Playback-state recognition | Participants distinguish synthesis-ready, queued, playing and completed | ≥95% accuracy |
| Provenance comprehension | Identify runtime event vs derived signal vs renderer-local effect | ≥90% accuracy |
| Uncertainty comprehension | Participants correctly recognize absent vs present uncertainty signal | ≥90% |
| Accessibility | Reduced-motion, static mode, captions, non-color cues | 100% required checks |
| Anthropomorphic overclaim | Participants falsely infer unsubstantiated feelings or capabilities | No increase versus neutral-control interface |

Conduct controlled comparisons against:
1. no avatar / text-only;
2. generic speaking orb;
3. conventional expressive 2D puppet;
4. audio-driven humanoid baseline;
5. Symthaea evidence-connected embodiment.

Pre-register hypotheses, separate familiarity effects from the interface effect, report confidence intervals, and publish anonymized materials and scoring scripts.

### Track D — optional calibrated uncertainty

Only run this track when the underlying runtime exports a calibrated probability or uncertainty estimate. Measure Brier score and expected calibration error (ECE); initial research goals are Brier ≤0.10 and ECE ≤0.05 on held-out tasks. If the runtime does not supply a calibrated value, show "not available" rather than inventing a meter.

## How we compare and where to try to outperform

| Dimension | NVIDIA ACE / Instant4D | Epic MetaHuman | Symthaea research objective |
|---|---|---|---|
| Facial fidelity | Strong specialist baseline | Strong production baseline | Use 2026 Let Toon Talk / CartoonTalk / AnyTalk_RT and established mesh baselines as adapters; do not claim to beat them without matched tests |
| Animation throughput | Audio2Face publishes max-stream capacity using P1 FPS so 99% of samples stay above 30 FPS on named GPU/server configurations | Real-time capture/animation workflow; live and offline solvers differ | Match on named hardware, exact audio workload and fixed assets; report P1/P5 frame time and stream capacity |
| Cognitive event semantics | Not the central published goal | Not the central published goal | First-class semantic contract |
| Provenance and uncertainty | Emotion/animation controls exist | Solve controls exist | Preserve the source and confidence of every signal |
| Embodiment diversity | Mostly human/digital-human pipeline | Human/digital-human pipeline | Abstract, humanoid, technical, and accessible modes |
| Offline replay and audit | Must be evaluated per integration | Must be evaluated per integration | Deterministic event-trace replay |

**Competitive strategy:** integrate with the best facial renderer when appropriate, while making Symthaea's semantic embodiment protocol the thing that is open, portable, auditable, and benchmarked. A high-quality face is a replaceable backend; a truthful cognitive interface is the product-level differentiator.

## Execution roadmap

### P0 — contract and identity
- [x] Inspect current canvas and speech-related crates.
- [x] Add the first renderer-neutral event/snapshot contract.
- [x] Add sequence, timestamp, range, and speech-state negative controls.
- [ ] Run focused Cargo tests and workspace checks on the branch.
- [ ] Review event provenance policy and version the schema.
- [ ] Adopt a canonical identity sheet and design tokens.

### P1 — real runtime signals
- [ ] Add adapters for speech output lifecycle, tool lifecycle, perception, retrieval/association, and prediction.
- [ ] Ensure event producers emit only from the authoritative subsystem and share one session sequencer.
- [ ] Add a real playback-start observer to the Rodio/live-voice and browser playback paths; synthesis and queueing remain separate.
- [ ] Add audio-clock timestamps and a calibrated viseme adapter; retain a static/no-mouth fallback if no viseme source is present.
- [x] Add versioned trace recording format, deterministic replay and a reference fixture.
- [ ] Add cross-session trace capture, redaction policy and stable corpus publication.

### P2 — first visible puppet
- [x] Add a Mycelial Presence SVG scene adapter on the feature branch.
- [ ] Embed the adapter in a live frontend/session and measure actual event-to-frame latency.
- [ ] Build a 2D/2.5D face with separate gaze, viseme, and activity channels.
- [ ] Add captions, static mode, reduced-motion mode, and user choice of embodiment.
- [ ] Measure latency, frame time, and power on desktop and Pixel 8 Pro.

### P3 — interoperability and quality
- [ ] Add optional VRM 1.0/VRMA import/export for humanoid embodiments.
- [ ] Evaluate Live2D and NVIDIA Audio2Face as comparative/adaptor backends.
- [ ] Add cross-renderer semantic-parity tests.
- [x] Add a trace replay CLI reporting reducer P50/P95 latency, exact-input BLAKE3 digest and OS/architecture metadata.
- [ ] Run the first SEFB benchmark with fixed traces, hardware metadata, and human-study protocol.

### P4 — research frontier
- [ ] Explore a non-humanoid procedural mycelial puppet with event-reactive topology.
- [ ] Investigate low-rank motion codes / HDC-coded gesture primitives with explicit, interpretable decoding.
- [ ] Explore causal predictive animation: pre-render likely next motion while preserving cancellation and correction when the event changes.
- [ ] Explore on-device avatar rendering and adaptive quality under thermal/power constraints.
- [ ] Publish benchmark results, negative results, and matched-baseline comparisons.

## Research hypotheses worth testing

1. **Semantic motion codes:** Can an HDC representation encode reusable gesture primitives while a small decoder preserves deterministic, bounded animation? This is exploratory; do not assume HDC is better than conventional motion embeddings.
2. **Predictive pre-animation:** Can a renderer reduce perceived latency by preparing candidate motion from an explicitly pending operation, without displaying it as completed? The UI must visibly distinguish anticipation from a confirmed event.
3. **Non-humanoid expressivity:** Can users identify runtime states from an organic mycelial representation as accurately as from a face, with lower anthropomorphic over-attribution?
4. **Renderer portability:** Can identical event traces produce semantically equivalent states across SVG, Live2D, and VRM backends?
5. **Evidence-first personalization:** Can users customize color, motion, face, and density without changing the underlying meaning of the event cues?

## Governance and release gates

- No avatar is allowed to authorize a tool, action, permission, payment, or system state.
- No renderer-local animation may be promoted to a runtime fact.
- Unknown, stale, or contradictory events must not create a new authoritative state; retain the last valid snapshot and expose the error to diagnostics.
- User audio/video must not be retained by default for benchmark telemetry; any study collection requires explicit consent and a documented retention policy.
- Every benchmark result must include commit SHA, event-corpus digest, asset/model versions, hardware, settings, measurement scripts, and raw aggregate data.
- CI queued, skipped, or pending is not a pass. Separate local/static checks, CI results, and physical-device measurements.

## Decision

Build the **semantic embodiment layer first**, then connect renderers. Make Mycelial Presence the canonical default, a humanlike face an optional conversational skin, and Cognitive Instrument the transparent technical mode. This is the fastest route to a distinctive first release while preserving the option to adopt industry-leading facial animation backends later.
