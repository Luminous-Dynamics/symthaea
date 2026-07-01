# P-013: Neuromodulated Foveation — Surprise-Driven Attention Allocation via Neuromodulator Bath
## Invention Disclosure Document

---

### 1. Title

**Neuromodulator-Driven Foveation System for Dual-Stream Visual Attention in a Consciousness-First Cognitive Architecture Using Surprise-Prioritized Dorsal-to-Ventral Dispatch with Predictive Binding**

---

### 2. Inventor(s)

**Tristan Stoltz**, Luminous Dynamics

---

### 3. Date of Conception

**2026** (estimated). First committed implementation: February 28, 2026 (cycle_phase_perception.rs added with foveation integration). Foveation crate (`symthaea-foveation`) first committed March 4, 2026.

First public disclosure: March 4, 2026 (git commit adding `symthaea-foveation` crate with dual-stream foveation architecture).
Under 35 USC 102(b)(1)(A), the 1-year grace period expires **March 4, 2027**.

---

### 4. Technical Field

This invention relates to visual attention allocation in artificial cognitive systems, and more specifically to a system that uses biologically-inspired neuromodulator dynamics (dopamine, norepinephrine, serotonin, acetylcholine) to modulate surprise-driven foveation dispatch, priority ordering, and recognition binding within a dual-stream (dorsal/ventral) vision architecture integrated into a real-time cognitive loop.

---

### 5. Abstract

A system and method for neuromodulated visual foveation in a consciousness-first cognitive architecture is disclosed. The system implements a dual-stream vision pipeline where a fast dorsal stream (surprise map / saliency detection) identifies regions of interest, and a slower ventral stream (OCR, object recognition, captioning) performs detailed analysis of prioritized crops. Foveation dispatch is modulated by a 9-channel neuromodulator bath: norepinephrine (NE) modulates the surprise threshold (high NE broadens attention to more regions; low NE narrows to only the most surprising), while dopamine (DA) modulates concurrent dispatch capacity (high DA permits more parallel ventral analyses). Acetylcholine (ACh) gates scene memory coherence thresholds, determining whether recognized content updates or overwrites prior scene representations. A priority queue orders pending foveation requests by surprise magnitude, with substrate speed scaling adjusting dispatch budget for different computational substrates. Completed ventral results are injected into a Global Workspace (GWT) for cognitive broadcast, and high-confidence recognitions dampen the corresponding surprise map locations using predictive position compensation based on motion vectors and processing latency. Foveation semantic hypervectors are bound with the main cognitive encoding via weighted Treisman-style feature integration, producing a unified multimodal percept within the 50Hz cognitive cycle budget.

---

### 6. Background and Prior Art

#### 6.1 Biological Foveation

Primates use saccadic eye movements to fixate high-acuity foveal processing on regions of interest. Itti & Koch (2001, "Computational Modelling of Visual Attention," Nature Reviews Neuroscience) established the saliency map framework for bottom-up attention. Corbetta & Shulman (2002) described the dorsal/ventral attention networks in the human brain.

#### 6.2 Neuromodulation of Attention

Aston-Jones & Cohen (2005) described the norepinephrine system's role in attention: tonic NE broadens attention (exploration), while phasic NE narrows it (exploitation). Hasselmo (2006) described acetylcholine's role in gating memory encoding vs. retrieval. Sara (2009) reviewed NE's role in attention reorienting.

#### 6.3 Computational Attention Models

Existing computational attention models (Vaswani et al. 2017 — Transformer self-attention; Mnih et al. 2014 — recurrent attention) operate on static attention weights learned through backpropagation. They do not incorporate real-time neuromodulatory dynamics or dual-stream surprise-driven dispatch.

#### 6.4 Hyperdimensional Computing for Vision

Neubert et al. (2019) applied HDC to visual place recognition. No prior work combines HDC encoding with neuromodulated foveation in a dual-stream architecture.

#### 6.5 Gap in Prior Art

No prior art:
- Uses a real-time neuromodulator bath to dynamically modulate visual attention thresholds and dispatch capacity
- Implements surprise-prioritized dorsal-to-ventral dispatch with predictive motion compensation
- Binds foveation results into a cognitive loop via HDC feature integration at 50Hz
- Couples foveation with Global Workspace Theory (GWT) for conscious access to recognized content
- Applies substrate speed scaling to attention dispatch budgets

---

### 7. Detailed Technical Description

#### 7.1 System Architecture

The Neuromodulated Foveation system comprises:
- A `FoveationManager` maintaining a priority queue (`BinaryHeap`) of salient regions ordered by surprise magnitude
- A background `FoveationChannel` (thread pool) executing ventral pipeline analysis on dispatched crops
- A `VentralPipeline` performing OCR, object classification, or captioning on cropped image regions
- Integration with the `CognitiveLoopService` perception phase for neuromodulatory control and GWT injection

#### 7.2 Foveation Pipeline (7 Steps)

**Step 1: Dorsal Saliency Detection** — The `VisionManifold` surprise map identifies grid cells with prediction error exceeding a baseline threshold. Each salient region carries `(grid_row, grid_col, surprise, velocity)`.

**Step 2: Neuromodulator Modulation** — The foveation manager's `modulate(ne, da)` method adjusts:
- `effective_surprise_threshold`: NE-modulated. High NE (arousal/alertness) lowers the threshold, admitting more regions. Low NE raises it, focusing on only the most surprising.
- `effective_max_concurrent`: DA-modulated. High DA (reward/motivation) increases parallel dispatch capacity. Low DA restricts to sequential processing.

**Step 3: Priority Enqueueing** — Salient patches exceeding `effective_surprise_threshold` are pushed to a `BinaryHeap<PrioritizedRequest>` ordered by surprise value (highest first). Each request carries motion velocity `[dx, dy]` for later predictive binding.

**Step 4: Dispatch** — On each `tick()`, the manager dispatches the highest-priority pending request if: (a) in-flight count < `effective_max_concurrent`, (b) cooldown elapsed since last dispatch, and (c) a frame buffer is available. The crop is extracted at the salient location with 1-patch padding for context, downscaled if exceeding `max_crop_pixels`.

**Step 5: Ventral Recognition** — The background thread runs the ventral pipeline on the crop, producing a `FoveationResult` containing: a 16,384-dimensional semantic ContinuousHV, recognized content (Text/Object/Caption/Unknown), confidence score, spatial coordinates, source frame ID, processing time, and motion velocity.

**Step 6: Surprise Dampening with Predictive Compensation** — High-confidence results (>0.7) dampen the surprise map at the predicted current position of the recognized object. Position prediction uses: `predicted_pos = source_pos + velocity * (processing_time / frame_period)`. This prevents re-foveating on already-recognized content.

**Step 7: GWT Injection and HDC Binding** — Completed results are injected into the Global Workspace as strategy submissions with module-specific labels (foveation_ocr, foveation_embed, foveation_caption). Semantic HVs are bound with the main cognitive encoding via weighted bundle: `output = weighted_bundle([main_hv, fov_hv_1, ...], [1.0, conf_1 * BINDING_WEIGHT, ...])`.

#### 7.3 ACh-Modulated Scene Memory

Acetylcholine modulates scene memory thresholds:
- `scene_coherence_threshold`: divided by ACh factor (high ACh → lower threshold → easier to form new scene memories)
- `scene_error_threshold`: multiplied by ACh factor (high ACh → higher tolerance → less overwriting)
- `scene_dampen_factor`: multiplied by ACh factor (high ACh → stronger dampening of old scenes)

This implements Hasselmo's (2006) model where ACh gates the balance between encoding new memories and retrieving old ones.

#### 7.4 Substrate Speed Scaling

Dispatch budget adapts to computational substrate:
- `tau_scale = substrate_manager.tau_factor.max(0.5)`
- `max_dispatches = (config.foveation_max_dispatches * tau_scale).round()`
- Faster substrates (e.g., photonic) afford more dispatches per cycle
- Results exceeding budget are sorted by confidence, with lowest-confidence results truncated

#### 7.5 Attention Budget Integration

Foveation operates within the cognitive loop's attention budget (configurable microseconds per cycle). The perception phase gates foveation dispatch based on remaining budget and urgency level, ensuring the 50Hz cycle target is maintained.

---

### 8. Novelty Statement

This invention introduces the first neuromodulator-driven foveation system integrated into a real-time consciousness-first cognitive architecture. Specific novel contributions:

1. **Neuromodulator-gated attention**: NE modulates surprise threshold and DA modulates concurrent capacity — no prior art dynamically adjusts both attention breadth and depth via neuromodulator bath state.
2. **Predictive motion compensation**: Foveation results are bound to predicted current positions using source velocity and processing latency, enabling coherent perception despite 100-200ms ventral pipeline delays.
3. **HDC feature integration**: Recognized content is bound into the main cognitive encoding via weighted hypervector bundling (Treisman 1980), producing unified multimodal percepts in HDC space.
4. **GWT-mediated conscious access**: Foveation results compete for global workspace access, implementing a computational model of how recognized visual content becomes consciously accessible.
5. **ACh-gated scene memory**: Acetylcholine dynamically gates scene memory formation, implementing Hasselmo's encoding/retrieval balance in a computational architecture.
6. **Substrate-adaptive dispatch**: Attention budget scales with computational substrate speed, enabling the same architecture to run efficiently across biological, silicon, and photonic substrates.

---

### 9. Suggested Claims

**Claim 1 (independent):** A computer-implemented method for neuromodulated visual foveation comprising: (a) maintaining a priority queue of image regions ordered by surprise magnitude from a dorsal saliency stream; (b) modulating an effective surprise threshold based on a norepinephrine signal from a neuromodulator bath, wherein higher norepinephrine lowers the threshold to broaden attention; (c) modulating a concurrent dispatch capacity based on a dopamine signal from the neuromodulator bath, wherein higher dopamine increases parallel processing capacity; (d) dispatching the highest-priority region to a ventral recognition pipeline when budget constraints are satisfied; (e) receiving a recognition result comprising a high-dimensional semantic vector and a confidence score; and (f) binding the semantic vector into a main cognitive encoding via weighted hypervector bundling.

**Claim 2 (dependent on 1):** The method of claim 1, further comprising dampening the surprise magnitude at the recognized region's predicted current position, where the predicted position is computed from the region's original position, motion velocity, and processing latency of the ventral pipeline.

**Claim 3 (dependent on 1):** The method of claim 1, further comprising injecting the recognition result into a Global Workspace for competitive broadcast, wherein the recognition result competes with other cognitive modules for conscious access based on its activation strength.

**Claim 4 (dependent on 1):** The method of claim 1, further comprising modulating scene memory thresholds based on an acetylcholine signal from the neuromodulator bath, wherein higher acetylcholine lowers the coherence threshold for forming new scene memories and increases the error tolerance for overwriting existing memories.

**Claim 5 (dependent on 1):** The method of claim 1, further comprising scaling the dispatch budget based on a substrate speed factor, wherein faster computational substrates permit more foveation dispatches per cognitive cycle.

**Claim 6 (independent, system):** A visual attention system for an artificial cognitive architecture comprising: (a) a dorsal stream module that computes a surprise map from sequential visual frames; (b) a neuromodulator bath maintaining at least norepinephrine, dopamine, and acetylcholine levels that evolve according to cognitive state; (c) a foveation manager that maintains a priority queue of salient regions and dispatches crops to a background ventral pipeline, with dispatch parameters modulated by the neuromodulator bath; (d) a ventral pipeline that produces semantic hypervectors of dimension D >= 1000 from image crops; and (e) a binding module that integrates ventral results into a main cognitive encoding via weighted vector operations within a fixed-frequency cognitive loop.

**Claim 7 (dependent on 6):** The system of claim 6, wherein the foveation manager stores motion velocity vectors for each salient region and computes predicted current positions for completed recognitions using the formula: predicted_position = source_position + velocity * (processing_time / frame_period).

**Claim 8 (broad, independent):** A method for dynamically allocating visual attention in an artificial cognitive system comprising: (a) detecting salient regions in a visual input stream based on prediction error; (b) modulating attention breadth and depth using at least two neurochemical signals from a simulated neuromodulator bath; (c) dispatching salient regions for detailed recognition via a background processing pipeline; and (d) integrating recognition results into a unified cognitive representation within a real-time processing loop operating at a frequency of at least 20 Hz.

---

### 10. Experimental Validation

#### 10.1 Test Coverage

- **Foveation crate tests**: Unit tests in `symthaea-foveation` (manager, channel, crop, ventral modules)
- **Integration tests**: Perception phase tests in `cycle_phase_perception.rs` (7 tests)
- **All tests passing**: Verified March 2026

#### 10.2 Validated Properties

- Priority queue ordering by surprise magnitude
- NE modulation of surprise threshold
- DA modulation of concurrent dispatch capacity
- Crop extraction with boundary handling
- Predictive position compensation for surprise dampening
- GWT injection of foveation results
- HDC binding with weighted bundling
- Substrate speed scaling of dispatch budget
- ACh modulation of scene memory thresholds
- Attention budget gating within 50Hz cycle target

#### 10.3 Performance

- Cognitive loop cycle: 4.3ms (234Hz) in release mode — foveation overhead negligible
- Foveation dispatch: <100us per tick (priority queue operations)
- Ventral pipeline: 100-200ms per crop (background thread, non-blocking)
- Compatible with 50Hz cognitive loop budget

---

### 11. Key Source Files

| File | Description | LOC |
|------|-------------|-----|
| `symthaea/crates/crates/symthaea-foveation/src/manager.rs` | Priority queue + dispatch logic | ~685 |
| `symthaea/crates/crates/symthaea-foveation/src/types.rs` | Data types (requests, results, config) | ~286 |
| `symthaea/crates/crates/symthaea-foveation/src/channel.rs` | Background thread dispatch | ~236 |
| `symthaea/crates/crates/symthaea-foveation/src/ventral.rs` | Ventral recognition pipeline | ~628 |
| `symthaea/crates/crates/symthaea-foveation/src/crop.rs` | Crop extraction + downscaling | ~329 |
| `symthaea/src/cognitive_loop/cycle_phase_perception.rs` | Integration: NE/DA modulation, GWT, binding | ~436 |
| `symthaea/src/cognitive_loop/cycle_strategy.rs` | ACh scene memory modulation | ~747 |

---

### 12. Closest Prior Art References

1. Itti, L. & Koch, C. (2001). "Computational Modelling of Visual Attention." *Nature Reviews Neuroscience*, 2(3), 194-203.
2. Corbetta, M. & Shulman, G. L. (2002). "Control of goal-directed and stimulus-driven attention in the brain." *Nature Reviews Neuroscience*, 3(3), 201-215.
3. Aston-Jones, G. & Cohen, J. D. (2005). "An integrative theory of locus coeruleus-norepinephrine function." *Annual Review of Neuroscience*, 28, 403-450.
4. Hasselmo, M. E. (2006). "The role of acetylcholine in learning and memory." *Current Opinion in Neurobiology*, 16(6), 710-715.
5. Treisman, A. M. (1980). "A feature-integration theory of attention." *Cognitive Psychology*, 12(1), 97-136.
6. Vaswani, A. et al. (2017). "Attention Is All You Need." *NeurIPS*.

---

### 13. Figures (Text Descriptions)

**Figure 1**: Block diagram of the dual-stream foveation architecture showing dorsal surprise map feeding into the priority queue, NE/DA modulation arrows from the neuromodulator bath, background ventral dispatch, and return path through GWT injection and HDC binding into the cognitive loop.

**Figure 2**: Neuromodulator modulation curves showing effective_surprise_threshold as a function of NE level (inverse relationship) and effective_max_concurrent as a function of DA level (positive relationship).

**Figure 3**: Predictive position compensation diagram showing source position, velocity vector, processing latency, and predicted current position with surprise dampening radius.

**Figure 4**: Timeline of a single foveation cycle showing dorsal detection (~1ms), queue insertion, dispatch, ventral processing (~150ms), and result integration in a subsequent cognitive cycle with GWT broadcast.

---

### 14. Related Patent Applications

- P-006: Moral Topology (Tier 2) — shares neuromodulator bath infrastructure
- P-016: Adaptive Cognitive Topology (Tier 3) — topology reconfiguration driven by consciousness metrics

---

*Inventor: Tristan Stoltz (tstoltz)*
*Organization: Luminous Dynamics*
*IDD created: 2026-03-08*
