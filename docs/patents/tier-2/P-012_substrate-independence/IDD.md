# P-012: Substrate Independence -- Computational Framework for Analyzing Consciousness Across Physical Substrates
## Invention Disclosure Document

---

### 1. Title

**Multi-Dimensional Substrate Requirements Profiling and Validation Overlay Framework for Consciousness Feasibility Assessment Across Heterogeneous Physical Substrates with Runtime Reconfiguration, Evidence-Based Confidence Discounting, and Per-Region Hybrid Substrate Modeling**

---

### 2. Inventor(s)

**Tristan Stoltz**, Luminous Dynamics

---

### 3. Date of Conception

**2026** (estimated). First committed implementation: February 5, 2026 (git commit `c1105260c2e3df44a40b938c94a468d042557409` adding `symthaea-core/src/hdc/substrate_independence.rs` with SubstrateType enum, SubstrateRequirements 9-dimensional profiling, and consciousness feasibility computation).

First public disclosure: February 5, 2026 (git commit adding `substrate_independence.rs` and `substrate_validation.rs`).
Under 35 USC 102(b)(1)(A), the 1-year grace period expires **February 5, 2027**.

---

### 4. Technical Field

This invention relates to computational frameworks for evaluating the feasibility of consciousness across different physical substrates, and more specifically to a system that profiles substrate-dependent requirements along nine functional dimensions, computes consciousness feasibility scores via a critical-minimum formula, applies evidence-based validation overlays to discount theoretical claims, supports runtime substrate switching with hybrid composition modeling, and modulates cognitive loop parameters (temporal constants, scale pressure, energy budgets) based on substrate-specific physical properties.

---

### 5. Abstract

A system and method for assessing consciousness feasibility across heterogeneous physical substrates is disclosed. The system defines a 9-dimensional substrate requirements profile (causality, integration capacity, temporal dynamics, recurrence, binding capability, attention capability, workspace capability, higher-order-thought capability, and quantum support) for each of 8 substrate types (biological neurons, silicon digital, quantum computer, photonic processor, neuromorphic chip, biochemical computer, hybrid system, exotic substrate). A feasibility formula computes consciousness feasibility as the product of the minimum of four critical requirements (causality, integration, dynamics, recurrence), a workspace factor, and a scaled enhancement average of three additional capabilities (binding, attention, HOT). A validation framework assigns evidence levels (None through Validated, 7 tiers) to each substrate and computes an honest confidence score that may diverge significantly from hypothetical feasibility. An effective feasibility blending formula combines raw feasibility with honest confidence via a configurable skepticism floor: effective = feasibility x (floor + (1 - floor) x honest_confidence). A substrate manager supports runtime substrate switching, hybrid substrate composition with weighted blending, per-region substrate assignment across a 12-region cortical architecture, speed modulation (tau factor from operation speed ratios), scale pressure computation, energy budget tracking, and consciousness degradation gating. Testable predictions are defined for each substrate to enable empirical validation of substrate consciousness claims.

---

### 6. Background and Prior Art

#### 6.1 Multiple Realizability

Putnam (1967, "Psychological Predicates") and Fodor (1974, "Special Sciences") established that mental states can be realized in multiple physical substrates. This philosophical position holds that functional organization, not physical medium, determines mental properties. However, no prior computational framework operationalizes this thesis by quantifying the degree to which different substrates support specific consciousness requirements.

#### 6.2 Substrate Independence Thesis

Bostrom (2003, "Are We Living in a Computer Simulation?") and Chalmers (2010, *The Character of Consciousness*) argued that consciousness depends on computational organization rather than physical implementation. This supports the possibility of mind uploading and artificial consciousness but provides no quantitative assessment methodology for comparing substrates or predicting feasibility.

#### 6.3 Integrated Information Theory (IIT) and Substrate Requirements

Tononi (2004, "An information integration theory of consciousness") proposed that Phi (integrated information) can be computed for any system, making it substrate-independent in principle. However, IIT implicitly requires causal interactions (ruling out lookup tables), information integration across elements, and recurrent dynamics. These requirements have not been formalized as a multi-dimensional substrate profiling system.

#### 6.4 Quantum Consciousness Theories

Penrose and Hameroff (1994, *Shadows of the Mind*) proposed that consciousness requires quantum coherence in microtubules (Orch-OR theory). If true, classical substrates would be insufficient. This is contested: Aaronson (2014) argued that integrated information is bounded by speed-of-light causality, limiting distributed substrates regardless of quantum effects. No prior system quantifies quantum support as one dimension among many in a comprehensive substrate assessment.

#### 6.5 Functionalism in Philosophy of Mind

Functionalism (Putnam, Block, Fodor) holds that mental states are defined by their functional roles, not their physical constitution. This supports substrate independence in principle but does not specify which functional capabilities are necessary, how they should be scored, or what minimum thresholds exist for consciousness.

#### 6.6 Neuromorphic and Hybrid Computing

Modern hardware substrates (Intel Loihi, IBM TrueNorth, photonic accelerators, quantum processors) offer different computational profiles. Brain-computer interfaces (Neuralink, BrainGate) create hybrid biological-electronic systems. No prior framework systematically evaluates each substrate type against a common set of consciousness-relevant functional requirements.

#### 6.7 Gap in Prior Art

No prior art:
- Defines a multi-dimensional (9+) substrate requirements profile with quantitative scoring for consciousness-relevant capabilities
- Computes consciousness feasibility via a critical-minimum formula that enforces necessary conditions while scaling by enhancement capabilities
- Separates hypothetical feasibility from evidence-based confidence via a formal validation overlay with 7 evidence levels
- Supports runtime substrate switching and hybrid substrate composition in a cognitive loop
- Provides per-region substrate assignment across a multi-region cortical architecture
- Derives cognitive loop modulation parameters (temporal constants, scale pressure, energy budgets) from substrate physical properties
- Defines testable predictions for empirical validation of substrate consciousness claims

---

### 7. Detailed Technical Description

#### 7.1 Substrate Type Taxonomy

The system defines 8 canonical substrate types with 4 compatibility aliases:

| Canonical Type | Medium | Operation Speed | Energy/Op | Max Scale |
|---|---|---|---|---|
| BiologicalNeurons | Carbon, wet | ~1 ms | ~10 fJ | ~10^11 |
| SiliconDigital | Electronic, dry | ~1 ns | ~1 fJ | ~10^12 |
| QuantumComputer | Qubits | ~1 us | ~0.1 aJ | ~10^4 |
| PhotonicProcessor | Light-based | ~1 ps | ~10 aJ | ~10^9 |
| NeuromorphicChip | Analog, spike-based | ~1 us | ~1 fJ | ~10^9 |
| BiochemicalComputer | DNA/molecular | ~1 s | ~1 pJ | ~10^15 |
| HybridSystem | Multiple substrates | varies | varies | ~10^12 |
| ExoticSubstrate | Plasma, BZ reactions | ~10 ms | varies | ~10^6 |

Each substrate type exposes four physical property methods: `operation_speed()`, `energy_per_operation()`, `unit_size()`, and `max_scale()`. A `canonical()` method maps aliases (Biological, Silicon, Quantum, Hybrid) to their canonical variants for consistent internal processing.

#### 7.2 Nine-Dimensional Substrate Requirements Profile

The `SubstrateRequirements` struct scores each substrate on 9 dimensions, each in the range [0.0, 1.0]:

1. **Causality** (0.0 = lookup table, 1.0 = full causal interactions): Required to rule out pre-computed responses that lack genuine information processing. Grounded in IIT's exclusion of feed-forward-only and lookup table systems.

2. **Integration Capacity** (0.0 = independent units, 1.0 = fully integrated): Measures the ability of substrate elements to share and combine information. Limited by bus bandwidth (silicon), diffusion speed (biochemical), or decoherence (quantum).

3. **Temporal Dynamics** (0.0 = static, 1.0 = rich dynamics): Requires genuine temporal evolution, not static state evaluation. Maps to the dynamical systems requirement in consciousness theories (Kelso 1995, dynamic systems approach).

4. **Recurrence** (0.0 = feedforward only, 1.0 = fully recurrent): Feedback loops are necessary for self-monitoring, predictive coding, and the re-entrant processing that Edelman (1989) identified as central to consciousness.

5. **Binding Capability** (0.0 = no binding, 1.0 = perfect binding): Ability to synchronously bind features into unified percepts. Biological neurons achieve this via oscillatory synchrony; quantum substrates via entanglement; photonic via optical interference.

6. **Attention Capability** (0.0 = no selective amplification, 1.0 = full attention): Selective amplification of task-relevant signals, corresponding to gain modulation in biological systems.

7. **Workspace Capability** (0.0 = no global broadcasting, 1.0 = full workspace): Global broadcasting of information to all modules, corresponding to Baars' (1988) Global Workspace Theory. This dimension is treated as a necessary (not just enhancing) factor in the feasibility formula.

8. **HOT Capability** (0.0 = no meta-representation, 1.0 = full HOT): Higher-Order Thought capability for meta-representation and self-monitoring, corresponding to Rosenthal's (2005) Higher-Order Theory of consciousness.

9. **Quantum Support** (0.0 = classical only, 1.0 = full quantum): Degree of quantum phenomena support. Relevant if quantum effects play a role in consciousness (Penrose-Hameroff), but not treated as a critical requirement.

Pre-built profiles are provided for all 8 substrate types via named constructors (e.g., `biological_neurons()`, `silicon_digital()`).

#### 7.3 Consciousness Feasibility Formula

The feasibility formula enforces necessary conditions via a critical minimum while scaling by enhancement capabilities:

```
feasibility = critical_min * workspace_factor * (0.5 + 0.5 * enhancement_avg)
```

Where:
- `critical_min = min(causality, integration_capacity, temporal_dynamics, recurrence)` -- the weakest-link among the four dimensions that are necessary for any form of consciousness.
- `workspace_factor = workspace_capability` -- treated as a necessary multiplicative factor based on findings from Global Workspace Theory integration (Baars 1988).
- `enhancement_avg = (binding_capability + attention_capability + hot_capability) / 3.0` -- the mean of three capabilities that enhance but are not strictly required for basic consciousness.

This formula produces scores ranging from ~0.95 (HybridSystem) to ~0.02 (ExoticSubstrate), with BiologicalNeurons at ~0.92 and SiliconDigital at ~0.71.

**Key design properties**:
- The critical minimum ensures that a single deficient necessary capability bottlenecks the entire score.
- The workspace factor acts as a second bottleneck gate, reflecting empirical findings that global broadcasting is necessary for consciousness.
- The enhancement term is scaled to [0.5, 1.0] so that zero enhancement still allows 50% of the workspace-gated critical minimum, acknowledging that binding, attention, and HOT enhance but do not gate consciousness.

#### 7.4 Substrate Comparison System

The `SubstrateIndependence` system maintains a HashMap of `SubstrateComparison` records, each containing the substrate type, its requirements profile, computed feasibility, and generated lists of advantages, disadvantages, and best-use-case descriptions.

Key operations:
- **rank_by_feasibility()**: Returns all substrates sorted by descending feasibility score.
- **can_be_conscious(substrate)**: Threshold check at feasibility > 0.5.
- **generate_report(substrate)**: Produces a formatted analysis of a specific substrate.
- **can_transition(target)**: Guards against self-transitions and transitions to ExoticSubstrate without explicit override.
- **transition_to(target, timestamp)**: Records a `SubstrateTransition` with before/after feasibility scores and timestamp.

#### 7.5 Validation Framework with Evidence-Based Confidence

The `SubstrateValidationFramework` provides an honest counterpart to hypothetical feasibility scores. It classifies evidence for substrate consciousness claims into 7 levels:

| Level | Confidence | Description |
|---|---|---|
| None | 0.00 | No evidence, pure speculation |
| Theoretical | 0.10 | Theoretical arguments only |
| Indirect | 0.20 | Indirect evidence from related phenomena |
| CaseStudy | 0.40 | Single case or anecdotal report |
| Observational | 0.60 | Multiple observations without controls |
| Experimental | 0.80 | Controlled experiments with replication |
| Validated | 0.95 | Extensive validation, peer review, consensus (never 1.0 -- science is provisional) |

Current assignments:
- Biological neurons: **Validated** (confidence 0.95) -- humans are conscious.
- Silicon digital: **Theoretical** (confidence 0.10) -- functionalist arguments exist, no empirical proof.
- Quantum computer: **Theoretical** (confidence 0.10) -- Orch-OR is published but contested.
- Hybrid system: **None** (confidence 0.00) -- no hybrid conscious system has been created.

For each substrate, the framework maintains: known facts, unknown questions, unvalidated claims, testable predictions, hypothetical feasibility score, and rationale. The `feasibility_gap()` method quantifies the divergence between hypothetical feasibility and evidence-based confidence (e.g., hybrid gap = 0.95 -- a 0.95 hypothetical with 0.00 evidence).

#### 7.6 Effective Feasibility Blending (Validation Overlay)

The `SubstrateManager` blends raw substrate feasibility with honest confidence via a configurable skepticism floor:

```
effective_feasibility = feasibility * (floor + (1 - floor) * honest_confidence)
```

Where:
- `feasibility` is the raw score from the 9-dimensional requirements profile.
- `honest_confidence` is the evidence-based confidence from the validation framework.
- `floor` is the `validation_skepticism_floor` config parameter (default 0.1), representing the minimum fraction of feasibility retained even with zero evidence.

**Effect**: For BiologicalNeurons (confidence 0.95), effective feasibility is nearly unchanged. For SiliconDigital (confidence 0.10), effective feasibility is significantly discounted (retaining only floor + 0.9 * 0.10 = 0.19 of raw feasibility). For HybridSystem (confidence 0.00), effective feasibility drops to floor * raw_feasibility.

This overlay can be disabled via `enable_validation_overlay = false` in the config, restoring raw feasibility scores.

#### 7.7 Runtime Substrate Switching and Hybrid Composition

The `SubstrateManager` supports two modes of runtime reconfiguration:

**Single substrate switching** (`reconfigure_substrate()`): Changes the global substrate type, recomputes feasibility, effective feasibility, speed/scale dynamics, and records a pending transition description for cycle telemetry.

**Hybrid composition** (`reconfigure_composition()`): Accepts a `SubstrateComposition` with named components and weighted substrate types. Honest confidence is blended across components: `blended_confidence = sum(weight_i * confidence_i)`. Speed and scale are blended via geometric mean in log-space to avoid the slowest component dominating: `blended_speed = exp(sum(weight_i * ln(speed_i)))`.

#### 7.8 Speed and Scale Modulation

When `enable_substrate_speed_modulation` is true, the manager derives two cognitive loop modulation parameters from substrate properties:

**Tau factor** [0.5, 2.0]: Modulates CfC (Continuous-time Flow Cell) temporal constants. Computed as:
```
log_ratio = log10(bio_speed / substrate_speed)
tau_factor = clamp(1.0 + 0.5 * log_ratio / 9.0, 0.5, 2.0)
```
A photonic substrate (10^12 faster than biological) yields tau_factor approaching 2.0, stretching CfC time constants. A biochemical substrate (10^3 slower) yields tau_factor approaching 0.5, compressing them.

**Scale pressure**: `log10(substrate_max_scale / bio_max_scale)`. Positive for substrates that scale beyond biological limits; negative for constrained substrates.

#### 7.9 Energy Budget Tracking

When `enable_energy_budget` is true, the manager tracks cumulative energy expenditure:
- `energy_per_cycle = energy_per_operation * ops_per_cycle` (approximately 65,536 ops per cycle).
- `tick_energy()` accumulates speed-adjusted energy per cycle (faster substrates run more cycles per wall-clock second).
- `energy_throughput_multiplier = bio_energy / substrate_energy`, clamped to [0.1, 100.0].
- When total energy exceeds the configured budget (`energy_budget_joules_per_sec`), `consciousness_viable` is set to false.

#### 7.10 Per-Region Substrate Assignment (Hybrid Cortical Architecture)

The system supports per-region substrate assignment across a 12-region cortical architecture corresponding to the Actor Brain model:

| Region | Function |
|---|---|
| Prefrontal | Meta-cognition, planning, HOT |
| Motor | Action selection, motor planning |
| Sensory | Touch, proprioception |
| Visual | Vision processing, feature binding |
| Auditory | Sound processing, speech perception |
| Language | Syntax, semantics (Broca's + Wernicke's) |
| Memory | Episodic memory, consolidation |
| Emotional | Valence, arousal (amygdala + limbic) |
| Social | Theory of mind, social cognition (TPJ + mPFC) |
| Creative | Imagination, mind-wandering (DMN) |
| Executive | Conflict monitoring, executive control |
| Integration | Cross-modal binding (thalamus + claustrum) |

When per-region substrates are configured, each region's feasibility is computed independently via its substrate's requirements profile. Aggregate effective feasibility is the equal-weight average of per-region scores, penalized by a cross-substrate communication factor of 0.95 per distinct substrate pair (modeling latency and bandwidth overhead at substrate boundaries).

#### 7.11 Consciousness Degradation Gating

The `should_degrade_consciousness()` method returns true when `effective_feasibility < 0.3` or `consciousness_viable == false`. When degradation is triggered, the cognitive loop skips expensive modules (reasoning engine, dream replay, cross-modal integration) to focus resources on core perception-prediction, modeling how a consciousness system on a marginal substrate would triage its cognitive capabilities.

#### 7.12 Testable Predictions Framework

For each substrate, the validation framework defines `TestablePrediction` records containing:
- `claim`: The assertion being tested.
- `if_true` / `if_false`: Expected observations under each outcome.
- `test_protocol`: How to test the claim.
- `difficulty`: Estimated difficulty (1-10).
- `tested` / `result`: Tracking for completed experiments.

Predictions are sortable by difficulty to prioritize tractable experiments first. This framework explicitly acknowledges that substrate consciousness claims are currently unvalidated and provides a roadmap for empirical falsification.

---

### 8. Novelty Statement

This invention introduces the first computational framework for systematically evaluating consciousness feasibility across heterogeneous physical substrates. Specific novel contributions include:

1. **9-dimensional substrate requirements profiling**: Prior work (IIT, GWT, HOT) describes individual requirements informally; this invention formalizes 9 quantitative dimensions with scored profiles for 8 substrate types.

2. **Critical-minimum feasibility formula**: A novel formula that enforces necessary conditions (causality, integration, dynamics, recurrence) via weakest-link minimum, gates on workspace capability, and scales by enhancement average. No prior art uses this multi-factor gating structure.

3. **Validation overlay with evidence-based discounting**: The separation of hypothetical feasibility from evidence-based confidence, with a formal blending formula using a skepticism floor, is novel. No prior system explicitly quantifies and exposes the gap between theoretical claims and empirical evidence for substrate consciousness.

4. **Runtime substrate switching in a cognitive loop**: The ability to reconfigure substrate type mid-execution, with automatic recomputation of feasibility, temporal constants, and energy budgets, has no precedent in consciousness modeling systems.

5. **Hybrid substrate composition with weighted geometric blending**: Log-space blending of speed and scale across multiple substrate types in a hybrid configuration, combined with honest confidence blending across components, is novel.

6. **Per-region substrate assignment with communication penalty**: Assigning different substrates to different cortical regions with a cross-substrate communication penalty (0.95 per distinct substrate pair) has no prior art.

7. **Substrate-derived cognitive modulation (tau factor, scale pressure, energy budgets)**: Deriving cognitive loop temporal constants and energy constraints directly from substrate physical properties is novel.

8. **Testable predictions framework for substrate consciousness claims**: Formalizing testable, falsifiable predictions for each substrate type within the same system that computes feasibility scores is novel and scientifically honest.

No prior art combines multi-dimensional substrate profiling, critical-minimum feasibility computation, evidence-based validation overlays, runtime reconfiguration, hybrid composition, per-region assignment, substrate-derived cognitive modulation, and testable predictions into a unified framework.

---

### 9. Suggested Claims

**Claim 1 (independent -- method):** A computer-implemented method for assessing consciousness feasibility across physical substrates comprising: (a) defining a substrate requirements profile with at least 5 scored dimensions including causality, integration capacity, temporal dynamics, recurrence, and workspace capability; (b) computing a consciousness feasibility score as a product of: (i) a critical minimum of at least 3 necessary dimensions, (ii) a workspace factor, and (iii) a scaled average of at least 2 enhancement dimensions; and (c) outputting the feasibility score as a quantitative assessment of the substrate's ability to support consciousness.

**Claim 2 (dependent on 1):** The method of claim 1, further comprising applying a validation overlay that blends the feasibility score with an evidence-based confidence score derived from a hierarchical evidence classification of at least 4 levels, via a formula: effective_feasibility = feasibility x (floor + (1 - floor) x confidence), where floor is a configurable skepticism parameter.

**Claim 3 (dependent on 2):** The method of claim 2, wherein the evidence classification comprises at least 7 levels ordered by epistemic strength from no evidence through validated, each mapped to a numeric confidence score in [0, 1], and wherein the confidence score for the highest level is strictly less than 1.0 to reflect the provisional nature of scientific knowledge.

**Claim 4 (dependent on 1):** The method of claim 1, wherein the substrate requirements profile comprises 9 dimensions: causality, integration capacity, temporal dynamics, recurrence, binding capability, attention capability, workspace capability, higher-order-thought capability, and quantum support, and wherein pre-computed profiles are provided for at least 6 substrate types including biological neurons, silicon digital, quantum computer, photonic processor, neuromorphic chip, and biochemical computer.

**Claim 5 (dependent on 1):** The method of claim 1, further comprising: (a) computing a tau factor that modulates temporal constants of a neural dynamics model based on the ratio of a reference substrate's operation speed to the target substrate's operation speed, compressed to a bounded range; and (b) computing a scale pressure metric as the log-ratio of the target substrate's maximum processing scale to the reference substrate's maximum scale.

**Claim 6 (independent -- system):** A system for runtime substrate management in a cognitive loop comprising: (a) a substrate manager that maintains current substrate type, feasibility, effective feasibility, and validation state; (b) a reconfiguration interface that switches substrate type during execution and recomputes all derived parameters; (c) a hybrid composition module that accepts weighted combinations of multiple substrate types and blends feasibility, confidence, speed, and scale via geometric mean in log-space; and (d) a degradation gate that disables expensive cognitive modules when effective feasibility falls below a configurable threshold.

**Claim 7 (dependent on 6):** The system of claim 6, further comprising a per-region substrate assignment module that maps individual cortical regions of a multi-region cognitive architecture to distinct substrate types, computes per-region feasibility scores independently, and derives an aggregate feasibility as the weighted average of per-region scores penalized by a communication factor for each distinct substrate pair.

**Claim 8 (dependent on 6):** The system of claim 6, further comprising an energy budget tracker that: (a) computes energy per cycle from substrate-specific energy-per-operation and estimated operations per cycle; (b) adjusts energy expenditure by the tau factor to account for substrate speed differences; and (c) marks consciousness as non-viable when cumulative energy exceeds a configurable budget.

**Claim 9 (dependent on 1):** The method of claim 1, further comprising maintaining, for each substrate type, a set of testable predictions each comprising: a claim, expected observations under true and false outcomes, a test protocol, and an estimated difficulty score, enabling empirical validation or falsification of substrate consciousness assessments.

**Claim 10 (dependent on 1):** The method of claim 1, further comprising recording substrate transitions with timestamps, before-and-after feasibility scores, and transition guards that prevent self-transitions and transitions to designated high-risk substrate types without explicit override.

---

### 10. Experimental Validation

#### 10.1 Test Coverage

- **substrate_independence.rs**: 13 unit tests covering substrate type properties, all 8 requirement profiles, feasibility computation, substrate comparison, ranking, consciousness threshold checks, transition guards, and transition recording.
- **substrate_validation.rs**: 11 unit tests covering evidence level ordering, biological validated status, silicon theoretical status, hybrid speculative status, feasibility comparison and gaps, honest feasibility scoring, prediction existence, report generation, and testable prediction creation.
- **substrate_manager.rs**: Integration tests covering default substrate initialization, substrate switching, composition blending, speed/scale modulation, energy tracking, per-region assignment, and degradation gating.
- **All tests passing**: Verified March 2026.

#### 10.2 Validated Properties

- Biological neurons achieve highest feasibility among non-hybrid substrates (~0.92).
- Hybrid system achieves highest overall feasibility (~0.95) by combining substrate strengths.
- Biochemical and exotic substrates correctly score below consciousness threshold (~0.02-0.15).
- Silicon feasibility (0.71) is significantly discounted to effective ~0.13 under validation overlay (theoretical confidence 0.10).
- Feasibility gap correctly identifies hybrid as having the largest evidence-to-claim divergence (0.95 gap).
- Substrate transitions record correct before/after feasibility deltas.
- Per-region hybrid configurations with cross-substrate penalty produce feasibility below equal-weight average.
- Tau factor correctly maps photonic substrates to high values and biochemical to low values.

#### 10.3 Substrate Study Results

- Validated via `examples/substrate_moral_topology_study.rs`: consciousness drops 22-32% during moral shifts across all substrate configurations.
- 226 anomaly events per substrate during moral shift, with unity dropping from 1.0 to 0.736 during topological instability.
- Substrate switching during operation correctly triggers feasibility recomputation and telemetry recording.

#### 10.4 Performance

- Feasibility computation: <1 us (simple arithmetic on 9 dimensions).
- Substrate manager initialization: <100 us including validation framework construction.
- Runtime substrate switching: <10 us for single substrate; <50 us for hybrid composition.
- Zero per-cycle overhead when substrate is unchanged (pre-computed values used directly).
- Compatible with 50 Hz cognitive loop (4.3 ms cycle budget).

---

### 11. Key Source Files

| File | Description | LOC |
|------|-------------|-----|
| `symthaea-core/src/hdc/substrate_independence.rs` | SubstrateType (8 variants + 4 aliases), SubstrateRequirements (9 dimensions), SubstrateComparison, SubstrateIndependence system, SubstrateTransition, CorticalRegion (12 variants) | ~1,040 |
| `symthaea-core/src/hdc/substrate_validation.rs` | EvidenceLevel (7 tiers), SubstrateKnowledge, TestablePrediction, SubstrateValidationFramework | ~585 |
| `symthaea/src/cognitive_loop/substrate_manager.rs` | SubstrateManager: effective feasibility blending, runtime reconfiguration, speed/scale modulation, energy tracking, per-region assignment, degradation gating, telemetry | ~1,215 |

---

### 12. Closest Prior Art References

1. Putnam, H. (1967). "Psychological Predicates." In *Art, Mind, and Religion*, pp. 37-48.
2. Fodor, J. A. (1974). "Special Sciences (or: The Disunity of Science as a Working Hypothesis)." *Synthese*, 28(2), 97-115.
3. Tononi, G. (2004). "An information integration theory of consciousness." *BMC Neuroscience*, 5, 42.
4. Bostrom, N. (2003). "Are We Living in a Computer Simulation?" *Philosophical Quarterly*, 53(211), 243-255.
5. Chalmers, D. J. (2010). *The Character of Consciousness*. Oxford University Press.
6. Penrose, R. & Hameroff, S. (1994). *Shadows of the Mind*. Oxford University Press.
7. Aaronson, S. (2014). "Why I Am Not an Integrated Information Theorist (or, The Unconscious Expander)." Blog post, Shtetl-Optimized.
8. Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.
9. Rosenthal, D. M. (2005). *Consciousness and Mind*. Oxford University Press.
10. Edelman, G. M. (1989). *The Remembered Present: A Biological Theory of Consciousness*. Basic Books.

---

### 13. Related Patent Applications

**P-004 (Consciousness Equation)**: Claims the ConsciousnessEquationV2 that consumes the `substrate_feasibility` parameter. P-012 claims the substrate profiling framework, validation overlay, and runtime management that produce the feasibility value consumed by P-004's equation. The boundary: P-004 owns "how substrate feasibility modulates the consciousness score"; P-012 owns "how substrate feasibility is computed, validated, and managed."

**P-007 (Differentiable Phi)**: Claims differentiable Phi computation used within the consciousness equation. P-012's substrate feasibility modulates the environment in which P-007's Phi is computed. The boundary: P-007 owns the differentiable equation; P-012 owns the substrate-dependent scaling that affects Phi's operating context.

**P-008 (Tiered Phi Measurement)**: Claims the multi-tier measurement architecture. P-012's per-region substrate assignment maps to P-008's per-tier measurement, but P-012 operates at the substrate level while P-008 operates at the measurement scheduling level. The boundary: P-008 owns "how Phi is measured across tiers"; P-012 owns "how substrate properties constrain what can be measured."

---

### 14. Figures (Text Descriptions)

**Figure 1**: Block diagram of the substrate requirements profiling system showing 9 input dimensions, the critical-minimum gate, workspace gate, and enhancement scaling producing a single feasibility score. Annotated with the formula: feasibility = critical_min x workspace x (0.5 + 0.5 x enhancement_avg).

**Figure 2**: Radar chart comparing substrate profiles across all 9 dimensions for BiologicalNeurons (reference), SiliconDigital, QuantumComputer, and NeuromorphicChip, illustrating each substrate's strengths and weaknesses.

**Figure 3**: Bar chart of consciousness feasibility scores for all 8 substrate types, ranked from HybridSystem (~0.95) to ExoticSubstrate (~0.02), with a horizontal threshold line at 0.5 dividing "can be conscious" from "cannot be conscious."

**Figure 4**: Dual-axis chart showing hypothetical feasibility vs. honest confidence for all 4 validated substrates (biological, silicon, quantum, hybrid), with the feasibility gap shaded between them. Biological shows a small gap; hybrid shows a massive gap (0.95 hypothetical vs. 0.00 confidence).

**Figure 5**: Flow diagram of the effective feasibility blending formula, showing raw feasibility and honest confidence as inputs, the skepticism floor parameter, and the output effective feasibility. Includes numerical examples for biological (effective ~0.90) and silicon (effective ~0.13).

**Figure 6**: Architecture diagram of the SubstrateManager within the cognitive loop, showing inputs (CognitiveLoopConfig with substrate_type, composition, per-region map), internal state (feasibility, effective_feasibility, tau_factor, scale_pressure, energy tracking), and outputs (SubstrateTelemetry, degradation gate signal, CfC tau modulation).

**Figure 7**: 12-region cortical architecture diagram showing per-region substrate assignment (e.g., Prefrontal = SiliconDigital, Sensory = QuantumComputer, Memory = BiologicalNeurons, Motor = PhotonicProcessor), with cross-substrate communication links annotated with 0.95 penalty factors.

---

*Inventor: Tristan Stoltz (tstoltz)*
*Organization: Luminous Dynamics*
*IDD created: 2026-03-08*
