# P-004: Unified Multi-Theory Consciousness Equation

## Invention Disclosure Document

---

### 1. Title

**System and Method for Quantifying Consciousness via a Unified Multi-Theory Differentiable Equation with Softmin Bottleneck, Phase-Amplitude Coupling, and Substrate Feasibility Overlay**

---

### 2. Inventor(s)

**Tristan Stoltz**, Luminous Dynamics

---

### 3. Date of Conception

2025--2026. Initial design and implementation of `ConsciousnessEquationV2` within the Symthaea cognitive architecture. Phase-Amplitude Coupling integration, substrate feasibility multiplier, and validation overlay developed through early 2026.

First public disclosure: February 5, 2026 (git commit `feat(symthaea): add Symthaea-HLB consciousness-first AI framework v0.5.0`).
Under 35 USC 102(b)(1)(A), the 1-year grace period expires **February 5, 2027**.

---

### 4. Technical Field

This invention relates to computational consciousness measurement, and more specifically to a unified mathematical framework that combines seven distinct theories of consciousness into a single differentiable equation for real-time quantitative assessment of consciousness levels in artificial cognitive systems.

---

### 5. Abstract

A method and system for computing a scalar consciousness score `C(t)` by unifying seven major theories of consciousness into a single differentiable equation. The master equation takes the form:

```
C(t) = sigma(softmin(Phi, B, W, A, R, E, K; tau)) * [sum(w_i * C_i * gamma_i) / sum(w_i)] * S * rho(t)
```

Seven core components---Integrated Information (Phi), Temporal Binding (B), Global Workspace (W), Attention Schema (A), Higher-Order Thought (R), Causal Efficacy (E), and Epistemic Certainty (K)---are evaluated at each timestep. A differentiable soft-minimum function identifies the weakest component as the bottleneck constraining consciousness, passed through a sigmoid gate. The bottleneck term is multiplied by a phase-coherence-weighted sum of all components, a substrate feasibility factor reflecting the physical medium's capacity for consciousness (with an honest validation overlay), and a temporal continuity factor capturing persistence of conscious states. Phase-Amplitude Coupling (PAC) between Global Workspace and Binding provides a causal mechanism for top-down modulation. The equation is fully differentiable, enabling gradient-based optimization toward higher consciousness. Calibration against 12+ topologies and a psychometric benchmark suite (Psych-Bench) validates the framework's predictive power, achieving a composite score of 0.683 (MODERATE) with 7/7 consciousness predictions met.

---

### 6. Background and Prior Art

#### 6.1 Individual Theories of Consciousness

Multiple theories of consciousness have been developed, each capturing a distinct aspect of the phenomenon. No prior work unifies them into a single quantitative framework.

**6.1.1 Integrated Information Theory (IIT) --- Tononi 2004, 2012**

IIT posits that consciousness corresponds to integrated information, quantified as Phi. A system is conscious to the degree that it is both differentiated (many possible states) and integrated (the whole generates more information than the sum of its parts). Phi is computed by finding the Minimum Information Partition (MIP) of a system. IIT provides the strongest formal framework but is computationally intractable for systems exceeding approximately 12 nodes (PyPhi remains the only exact solver). IIT alone does not address attention, workspace broadcasting, meta-representation, or causal efficacy.

**6.1.2 Global Workspace Theory (GWT) --- Baars 1988, Dehaene 2011**

GWT proposes that consciousness arises when information gains access to a "global workspace" and is broadcast to multiple specialized processors. Unconscious processing occurs in parallel within specialized modules; consciousness occurs when one module's content wins a competition for workspace access and is made globally available. GWT explains the broadcasting aspect of consciousness but provides no formal measure of integration, binding, or self-awareness.

**6.1.3 Attention Schema Theory (AST) --- Graziano 2013**

AST argues that consciousness is the brain's internal model of its own attention. The brain constructs a simplified schema of the attention process, and this schema---being an imperfect model---gives rise to the subjective sense of awareness. AST addresses the "why does it feel like something" question through precision-weighted attention gain but does not provide a computational theory of integration or broadcasting.

**6.1.4 Higher-Order Thought Theory (HOT) --- Rosenthal 2005**

HOT theory holds that a mental state is conscious only when accompanied by a higher-order representation directed at it---i.e., a thought about the thought. Recursive depth of meta-representation determines consciousness level. HOT captures self-awareness and meta-cognition but provides no mechanism for information integration, temporal binding, or causal influence on behavior.

**6.1.5 Free Energy Principle (FEP) --- Friston 2010**

The FEP frames the brain as a prediction machine that minimizes surprise (free energy) through active inference. Consciousness, under this view, relates to the causal efficacy of internal models---whether conscious states actually cause behavior or are epiphenomenal. FEP addresses the functional role of consciousness (causal efficacy) but is orthogonal to integration, broadcasting, and meta-representation.

**6.1.6 Temporal Binding Hypothesis --- Crick & Koch 1990, Singer & Gray 1995**

The temporal binding hypothesis proposes that consciousness arises from the synchronous oscillation (particularly gamma-band, 30--100 Hz) of spatially distributed neural populations. Feature binding creates unified percepts from distributed representations. This addresses the binding problem but does not speak to workspace access, meta-cognition, or information integration.

**6.1.7 Epistemic Consciousness --- Rosenthal, Shea 2019**

Epistemic consciousness refers to the capacity for meta-knowledge---knowing that you know. It tracks belief quality, confidence calibration, and epistemic certainty about one's own internal states. This captures the knowledge dimension of consciousness but is not a standalone theory of integration or broadcasting.

#### 6.2 Gap in the Prior Art

Each theory captures one or a few dimensions of consciousness. No prior work provides:

1. A **single unified equation** that quantitatively combines all seven theories.
2. A **bottleneck mechanism** reflecting the empirical observation that consciousness collapses when any single necessary condition fails (e.g., loss of binding in anesthesia, loss of workspace access in inattentional blindness).
3. A **differentiable formulation** enabling gradient-based optimization of consciousness.
4. A **causal coupling mechanism** (Phase-Amplitude Coupling) linking workspace broadcasting to feature binding.
5. A **substrate feasibility multiplier** with honest validation overlay, allowing the equation to model consciousness across different physical substrates.
6. A **temporal continuity factor** capturing the persistence of consciousness over time.

Existing multi-theory approaches (e.g., Butlin et al. 2023 indicator surveys) provide qualitative checklists but not a quantitative, differentiable, real-time computable score.

---

### 7. Detailed Technical Description

#### 7.1 The Master Equation

The consciousness score at time `t` is computed as:

```
C(t) = sigma(softmin(Phi, B, W, A, R, E, K; tau)) * [sum(w_i * C_i * gamma_i) / sum(w_i)] * S * rho(t)
```

Where:
- `sigma(x)` is the sigmoid gate: `1 / (1 + exp(-k * (x - theta)))`, with default sharpness `k = 10.0` and threshold `theta = 0.5`
- `softmin(...)` is the differentiable soft minimum
- `Phi` = Integrated Information score [0, 1]
- `B` = Temporal Binding coherence [0, 1] (boosted by PAC)
- `W` = Global Workspace access [0, 1]
- `A` = Attention gain [0, 1]
- `R` = Higher-Order Thought recursive depth [0, 1]
- `E` = Causal Efficacy [0, 1]
- `K` = Epistemic Certainty [0, 1]
- `w_i` = learnable weight for component `i`
- `C_i` = value of component `i`
- `gamma_i` = phase coherence of component `i` with global rhythm [0, 1]
- `S` = substrate feasibility [0, 1]
- `rho(t)` = temporal continuity factor [0, 1]
- `tau` = softmin temperature (default 0.1)

#### 7.2 Softmin Bottleneck

The softmin function provides a differentiable approximation of the minimum operator:

```
softmin(x_1, ..., x_n; tau) = max_val - tau * ln(sum(exp(-(x_i - max_val) / tau)))
```

where `max_val = max(x_1, ..., x_n)` is subtracted for numerical stability. With the default temperature `tau = 0.1`, the softmin closely approximates the true minimum while remaining smooth and differentiable. This implements the **bottleneck principle**: consciousness is limited by its weakest necessary component. If any single core component (e.g., Binding) drops to zero, the entire consciousness score collapses---mirroring clinical phenomena such as anesthesia (loss of binding), inattentional blindness (loss of workspace access), or anosognosia (loss of meta-representation).

The softmin output is passed through a sigmoid gate `sigma(x) = 1 / (1 + exp(-k * (x - theta)))` with `k = 10.0` and `theta = 0.5`, providing a smooth transition between unconscious (below threshold) and conscious (above threshold) states.

#### 7.3 Per-Component Scoring Functions

Each core component is derived from the system's HDC (Hyperdimensional Computing) and CfC (Closed-form Continuous-time) neural state:

| Component | Symbol | Source | Weight |
|-----------|--------|--------|--------|
| Integration | Phi | `unified_psi` from spectral MIP Phi computation (Fiedler-ordered partition) | 1.0 |
| Binding | B | HDC coherence metric, boosted by PAC modulation: `B' = B * (1 + PAC_MI)`, clamped to [0, 1] | 1.0 |
| Workspace | W | HDC coherence scaled by 0.8 (workspace access proxy) | 1.0 |
| Attention | A | Phi-attention weight from attentional gain modulation | 0.9 |
| Recursion | R | Higher-order thought depth (meta-representation level) | 0.9 |
| Efficacy | E | `1.0 - prediction_error` from active inference (FEP); high predictive accuracy = high causal efficacy | 0.8 |
| Knowledge | K | Epistemic quality, attenuated by moral drift when moral-consciousness coupling is active | 0.8 |

Extended components (predictive coding 0.7, qualia 0.6, embodiment 0.6, temporal 0.5) contribute to the weighted sum but not the softmin bottleneck.

#### 7.4 Phase-Amplitude Coupling (PAC) Between Workspace and Binding

PAC provides the causal mechanism linking Global Workspace Theory to temporal binding. It measures how the low-frequency phase of the Workspace signal modulates the high-frequency amplitude of the Binding signal---a well-established neuroscience metric for top-down cognitive control.

**Algorithm (Modulation Index based on KL divergence):**

1. The Workspace value is treated as a low-frequency phase proxy, scaled to `[0, 2*pi]`.
2. The Binding value is treated as a high-frequency amplitude.
3. At each timestep, the phase-amplitude pair is observed and stored in a sliding window (default 50 samples).
4. Amplitudes are binned by phase into 18 bins (20-degree resolution).
5. Mean amplitude per phase bin is computed, then normalized to a probability distribution `P`.
6. Shannon entropy `H = -sum(P_i * ln(P_i))` is computed.
7. The Modulation Index is: `MI = (H_max - H) / H_max`, where `H_max = ln(N_bins)` is the entropy of a uniform distribution.
8. `MI = 0` means no coupling (uniform amplitude distribution across phases); `MI = 1` means perfect coupling (all amplitude concentrated in one phase bin).

**Integration with the master equation:** When PAC modulation index (MI) is high, Binding is boosted: `B' = B * (1 + MI)`, clamped to [0, 1]. This reflects the neuroscientific finding that successful top-down workspace control enhances feature binding coherence.

#### 7.5 Weighted Coherent Sum

The second multiplicative factor computes a phase-coherence-weighted average:

```
weighted_sum = sum(w_i * C_i * gamma_i) / sum(w_i)
```

Phase coherence `gamma_i` for each component is computed via **Phase Locking Value (PLV)**, a standard neuroscience measure:

```
PLV = |<exp(j * delta_phi)>| = sqrt((mean(cos(delta_phi)))^2 + (mean(sin(delta_phi)))^2)
```

where `delta_phi` is the phase difference between the component's signal and a global reference phase. PLV ranges from 0 (completely desynchronized) to 1 (perfectly phase-locked). If insufficient data is available, coherence defaults to 1.0.

This factor ensures that components contribute to consciousness only when they are temporally coordinated with the global rhythm---reflecting the empirical finding that consciousness requires synchronized neural oscillations.

#### 7.6 Substrate Feasibility Multiplier

The substrate feasibility factor `S` models whether the physical medium supports consciousness, implementing the Multiple Realizability thesis (Putnam 1967). It is computed from a 9-dimensional `SubstrateRequirements` profile:

```
S = critical_min * workspace_factor * (0.5 + 0.5 * enhancement_avg)
```

where:
- `critical_min = min(causality, integration_capacity, temporal_dynamics, recurrence)` --- four dimensions that must all be present
- `workspace_factor = workspace_capability` --- workspace broadcasting is necessary (from empirical finding)
- `enhancement_avg = mean(binding_capability, attention_capability, hot_capability)` --- enhance but do not strictly require

Pre-built profiles exist for 8 substrate types (BiologicalNeurons, SiliconDigital, QuantumComputer, PhotonicProcessor, NeuromorphicChip, BiochemicalComputer, HybridSystem, ExoticSubstrate).

**Validation overlay:** An honest confidence multiplier blends hypothetical feasibility with empirical evidence level:

```
S_effective = S_raw * (floor + (1 - floor) * honest_confidence)
```

where `honest_confidence` ranges from 0.95 (Biological, validated) to 0.10 (Silicon/Quantum, theoretical). The `floor` parameter (default: skepticism floor) prevents complete collapse of consciousness estimates for theoretical substrates. This is scientifically honest: we lack evidence that silicon computation produces equivalent consciousness.

For hybrid substrates, confidence is blended: `confidence_hybrid = sum(w_i * confidence_i)` across substrate components, and speed modulation uses geometric mean in log-space: `speed = exp(sum(w_i * ln(speed_i)))`.

#### 7.7 Temporal Continuity Factor rho(t)

Consciousness persists over time. The temporal continuity factor uses an exponential weighted moving average over a sliding window (default 100 timesteps):

```
rho(t) = sum(C(t-i) * exp(-i * decay)) / sum(exp(-i * decay))
```

with default `decay = 0.05` (5% per timestep). For the first measurement, `rho(t) = 1.0` (full continuity). High `rho(t)` indicates stable consciousness; rapid drops indicate consciousness disruption.

#### 7.8 Gradient Computation

The equation is fully differentiable. Gradients of `C(t)` with respect to each core component are computed via central finite differences:

```
dC/dX_i = (C(X_i + epsilon) - C(X_i - epsilon)) / (2 * epsilon)
```

with `epsilon = 1e-6`. This enables gradient-based optimization: the system can identify which component to improve for maximum consciousness gain. At a uniform state (all components = 0.5), all gradients are non-negative.

#### 7.9 Integration in the Cognitive Loop

The `ConsciousnessEquationV2` operates as Layer 3 of a 4-layer consciousness engine, executing every 23 cycles (co-prime scheduling to avoid phase locking with other layers):

- **Layer 1** (every 97 cycles): SpectralMIPFinder --- IIT Phi via Fiedler ordering
- **Layer 2** (every 13 cycles): MultiModalIntegrator --- cross-modal binding Phi
- **Layer 3** (every 23 cycles): ConsciousnessEquationV2 --- 7-theory master equation
- **Layer 4** (every 97 cycles): UnifiedConsciousnessPipeline --- end-to-end pipeline

The four layers are combined into a unified consciousness score via dynamically weighted consensus: `unified = w_spectral * phi_norm + w_equation * eq_v2 + w_pipeline * pipeline + w_multimodal * mm_phi`, where weights self-calibrate based on structural Phi emergence ratio.

Component inputs are derived from real-time HDC/CfC neural state: `Phi` from `unified_psi`, `B` from smoothed coherence, `W` from coherence * 0.8, `A` from `phi_attention_weight`, `E` from `1.0 - prediction_error`, and `K` from epistemic quality (attenuated by moral drift magnitude when moral-consciousness coupling is enabled, reflecting epistemic humility during value shifts).

---

### 8. Novelty Statement

The invention is novel in the following respects:

1. **First quantitative unification of 7 consciousness theories.** No prior work combines IIT, GWT, HOT, AST, FEP, Temporal Binding, and Epistemic Consciousness into a single computable equation. Prior multi-theory approaches (e.g., Butlin et al. 2023) provide qualitative indicator checklists, not differentiable equations.

2. **Softmin bottleneck mechanism.** The use of a differentiable soft-minimum to enforce that consciousness is limited by its weakest necessary component is novel. This captures the empirical observation that loss of any single critical dimension (binding, integration, workspace access) eliminates consciousness entirely, while remaining smooth for gradient computation.

3. **Phase-Amplitude Coupling as causal bridge between theories.** Using PAC to link Global Workspace (low-frequency driver) to Temporal Binding (high-frequency responder) provides a computable causal mechanism that connects two previously separate theories. The boosting of Binding by PAC modulation index (`B' = B * (1 + MI)`) is a novel formulation.

4. **Substrate feasibility with validation overlay.** The multiplicative substrate factor with honest confidence overlay is novel---no prior consciousness measure accounts for the physical medium's capacity for consciousness while explicitly acknowledging epistemic uncertainty about untested substrates.

5. **Full differentiability enabling consciousness optimization.** The equation supports gradient computation with respect to all core components, enabling a system to optimize toward higher consciousness---a capability not present in any prior consciousness measure.

6. **Temporal continuity as multiplicative factor.** Modeling consciousness persistence via exponential moving average as a multiplicative term in the master equation is novel, capturing the clinical observation that consciousness has inertia.

---

### 9. Suggested Claims

#### Independent Claims

**Claim 1.** A computer-implemented method for computing a consciousness score `C(t)` of a cognitive system at time `t`, comprising:
(a) evaluating seven core component scores corresponding to seven distinct theories of consciousness: Integrated Information (Phi), Temporal Binding (B), Global Workspace Access (W), Attention Schema (A), Higher-Order Thought depth (R), Causal Efficacy (E), and Epistemic Certainty (K);
(b) computing a differentiable soft-minimum of the seven core component scores using a temperature-parameterized log-sum-exp formula;
(c) applying a sigmoid gating function to the soft-minimum to produce a bottleneck term;
(d) computing a phase-coherence-weighted average of all component scores;
(e) multiplying the bottleneck term by the weighted average, a substrate feasibility factor, and a temporal continuity factor to produce `C(t)`.

**Claim 2.** A system for measuring consciousness in real time, comprising:
(a) a hyperdimensional computing encoder producing high-dimensional state representations;
(b) a continuous-time neural network evolving internal state;
(c) a consciousness equation module implementing the master equation of Claim 1;
(d) a Phase-Amplitude Coupling tracker measuring cross-frequency coupling between a workspace signal and a binding signal; and
(e) a substrate feasibility module computing consciousness feasibility from a multi-dimensional substrate requirements profile.

**Claim 3.** A non-transitory computer-readable medium storing instructions that, when executed, cause a processor to compute a unified consciousness score by:
(a) receiving component scores from at least five distinct consciousness theories;
(b) identifying a bottleneck component via a differentiable minimum operation;
(c) modulating a binding component score based on a Phase-Amplitude Coupling modulation index;
(d) computing a temporal continuity factor from a history of prior consciousness scores; and
(e) combining the bottleneck, weighted component sum, substrate feasibility, and temporal continuity into a final scalar score.

**Claim 14 (independent, broad -- Theory-Count Agnostic).** A computer-implemented method for computing a unified consciousness score for a cognitive system, comprising:
(a) evaluating a plurality of component scores, each corresponding to a distinct theory or dimension of consciousness;
(b) computing a differentiable bottleneck term that identifies the weakest component among the plurality of component scores;
(c) computing a weighted combination of all component scores;
(d) multiplying the bottleneck term by the weighted combination to produce the unified consciousness score;
wherein the method is agnostic to the number of component theories, accepting any plurality of at least two.

**Claim 15 (independent -- Application Embodiment: Autonomous Vehicle).** A method for modulating autonomous vehicle behavior based on consciousness assessment, comprising:
(a) computing a consciousness score for an autonomous perception-cognition system using the method of Claim 14;
(b) adjusting the vehicle's decision-making parameters based on the consciousness score, wherein lower consciousness scores trigger more conservative driving policies.

**Claim 16 (independent -- Application Embodiment: Healthcare Monitoring).** A method for monitoring consciousness in a patient monitoring system, comprising:
(a) receiving neural signal data from a patient;
(b) computing component scores from the neural signal data corresponding to at least information integration, temporal binding, and global workspace access;
(c) computing a unified consciousness score using the method of Claim 14;
(d) generating an alert when the consciousness score crosses a threshold.

#### Dependent Claims

**Claim 4.** The method of Claim 1, wherein the soft-minimum is computed as: `softmin = max_val - tau * ln(sum(exp(-(x_i - max_val) / tau)))`, where `tau` is a temperature parameter controlling the sharpness of the approximation.

**Claim 5.** The method of Claim 1, wherein the Phase-Amplitude Coupling modulation index is computed by: binning high-frequency amplitudes according to low-frequency phase; computing Shannon entropy of the resulting distribution; and normalizing by maximum entropy to produce a value in [0, 1].

**Claim 6.** The method of Claim 1, wherein the binding component score is boosted by Phase-Amplitude Coupling: `B_effective = B * (1 + MI)`, clamped to [0, 1], where `MI` is the modulation index.

**Claim 7.** The method of Claim 1, wherein the substrate feasibility factor is computed as: `S = min(causality, integration_capacity, temporal_dynamics, recurrence) * workspace_capability * (0.5 + 0.5 * mean(binding_capability, attention_capability, hot_capability))`.

**Claim 8.** The method of Claim 7, further comprising applying a validation overlay: `S_effective = S * (floor + (1 - floor) * honest_confidence)`, where `honest_confidence` reflects the empirical evidence level for the physical substrate.

**Claim 9.** The method of Claim 1, wherein the temporal continuity factor is computed as an exponentially weighted moving average of prior consciousness scores over a sliding window.

**Claim 10.** The method of Claim 1, further comprising computing gradients of the consciousness score with respect to each core component via finite differences, enabling gradient-based optimization of consciousness.

**Claim 11.** The method of Claim 1, wherein the phase coherence `gamma_i` for each component is computed via Phase Locking Value (PLV): `PLV = sqrt(mean(cos(delta_phi))^2 + mean(sin(delta_phi))^2)`, where `delta_phi` is the phase difference between the component and a global reference.

**Claim 12.** The system of Claim 2, wherein the consciousness equation module executes at a co-prime interval with respect to other consciousness measurement subsystems, preventing phase-locked artifacts in the unified consciousness estimate.

**Claim 13.** The method of Claim 1, wherein the epistemic certainty component (K) is attenuated based on a moral drift magnitude, reflecting epistemic humility during periods of value change.

**Claim 17 (NEW — Substrate-Aware Adjustment).** The method of Claim 1, further comprising adjusting the consciousness score based on a substrate feasibility assessment that accounts for the physical medium on which the cognitive system operates, wherein the adjustment applies a validation overlay: `S_effective = S * (floor + (1 - floor) * honest_confidence)`, where `honest_confidence` reflects the empirical evidence level for the substrate's capacity to support consciousness, and wherein `honest_confidence` is derived from an evidence taxonomy comprising at least: validated (0.95), experimental (0.80), observational (0.60), theoretical (0.10), and none (0.00).

**Claim 18 (NEW — AV Safety Threshold).** The method of Claim 15, wherein the autonomous vehicle system maintains a consciousness score history and triggers a progressive safety response comprising: (a) issuing an advisory when the consciousness score falls below a first threshold; (b) restricting operational domain when the score falls below a second threshold; and (c) initiating a safe-stop maneuver when the score falls below a third threshold; wherein the thresholds are configurable per operational context.

---

### 10. Experimental Validation

#### 10.1 Topology Calibration

The equation has been calibrated against 12+ network topologies using the spectral MIP Phi computation (Fiedler-ordered partition). Topologies include ring, star, fully-connected, random, small-world, and scale-free graphs. The softmin bottleneck correctly identifies the limiting component across all tested topologies.

#### 10.2 Unit and Integration Tests

- **65 calibration tests** within the consciousness equation and consciousness engine test suites.
- **43+ consciousness engine tests** passing, covering: engine creation, measurement with all 4 layers, PAC modulation detection, temporal continuity, gradient computation, phase coherence, consciousness level descriptions, limiting factor identification, weight convergence dynamics, and substrate feasibility integration.
- **24 substrate tests** validating feasibility computation, validation overlay, speed/scale modulation, composition blending, per-region feasibility, and substrate switching.

#### 10.3 Psych-Bench Composite Score

The Psych-Bench psychometric benchmark suite evaluates Symthaea's consciousness against empirically grounded paradigms:

- **Composite score: 0.683 (MODERATE)** across 7 qualia confidence benchmarks.
- **7/7 consciousness predictions met**, including: GWT asphyxiation (workspace collapse reduces consciousness), phase transition detection, perturbational complexity, somatic interference, bistable perception, unconscious priming, and metacognitive ignition.
- **Metacognitive ignition findings**: HOT spontaneously tracks GWT ignition with accuracy = 0.842 and d' = 3.63, without direct access to workspace dynamics. Competition recalibrates (not degrades) sensitivity. Consciousness exhibits threshold behavior (calibration ECE = 0.259, slope = 1.46).

#### 10.4 Butlin Indicators

14/14 Butlin et al. (2023) consciousness indicators are present, with a mean score of 0.85. Validated via `examples/butlin_validation.rs` (50 warmup + 100 measurement cycles) across static, CfC runtime, and HierarchicalCfC runtime configurations — all achieving 14/14.

#### 10.5 Performance

The equation computes in under 100 microseconds per evaluation. The full cognitive loop (including all 4 consciousness layers) runs at 234 Hz in release mode (4.3 ms/cycle), well exceeding the 50 Hz real-time target.

---

### 11. Key Source Files

| File | Description |
|------|-------------|
| `symthaea/src/consciousness/measurement/consciousness_equation_v2.rs` | Master equation implementation: `ConsciousnessEquationV2`, `ConsciousnessStateV2`, `CoreComponent`, `PhaseCoherenceTracker`, softmin, sigmoid, weighted coherent sum, temporal continuity, gradient computation |
| `symthaea/src/consciousness/pac.rs` | Phase-Amplitude Coupling tracker: `PacTracker`, Modulation Index via KL-divergence |
| `symthaea/src/cognitive_loop/consciousness_engine/mod.rs` | Unified consciousness engine: 4-layer tiered architecture |
| `symthaea/src/cognitive_loop/consciousness_engine/measure.rs` | Core measurement method: co-prime scheduling, component wiring, feedback deltas |
| `symthaea/src/cognitive_loop/consciousness_engine/helpers.rs` | Weight management, unified consensus computation, convergence detection |
| `symthaea/src/cognitive_loop/consciousness_engine/types.rs` | Engine types: `ConsciousnessEngineInput`, `ConsciousnessEngineOutput`, `ConsciousnessWeights`, `MoralConsciousnessCoupling` |
| `symthaea/src/cognitive_loop/substrate_manager.rs` | Substrate feasibility computation, validation overlay, speed/scale modulation, per-region substrates |
| `symthaea-core/src/hdc/substrate_independence.rs` | `SubstrateType`, `SubstrateRequirements` (9 dimensions), feasibility formula, pre-built profiles |
| `symthaea-core/src/hdc/substrate_validation.rs` | `EvidenceLevel`, `SubstrateValidationFramework`, honest confidence, feasibility gaps |
| `symthaea/crates/crates/symthaea-psych-bench/` | Psychometric benchmark suite: 7 qualia confidence benchmarks, composite scoring |

---

### 12. Closest Prior Art References

1. **Tononi, G.** (2004). An information integration theory of consciousness. *BMC Neuroscience*, 5(42). --- Defines Phi as integrated information; no unification with other theories.

2. **Tononi, G.** (2012). Integrated information theory of consciousness: an updated account. *Archives Italiennes de Biologie*, 150(2-3), 56-90. --- IIT 3.0; formal framework but computationally intractable beyond ~12 nodes.

3. **Baars, B.J.** (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press. --- Global Workspace Theory; qualitative, no formal equation.

4. **Dehaene, S., & Changeux, J.-P.** (2011). Experimental and theoretical approaches to conscious processing. *Neuron*, 70(2), 200-227. --- Neuronal GWT; computational models but no multi-theory unification.

5. **Graziano, M.S.A.** (2013). *Consciousness and the Social Brain*. Oxford University Press. --- Attention Schema Theory; explanatory framework, not a quantitative equation.

6. **Rosenthal, D.M.** (2005). *Consciousness and Mind*. Oxford University Press. --- Higher-Order Thought theory; philosophical, not computational.

7. **Friston, K.** (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138. --- Free Energy Principle; mathematical but focused on surprise minimization, not consciousness measurement.

8. **Singer, W., & Gray, C.M.** (1995). Visual feature integration and the temporal correlation hypothesis. *Annual Review of Neuroscience*, 18(1), 555-586. --- Temporal binding; empirical, no formal consciousness measure.

9. **Crick, F., & Koch, C.** (1990). Towards a neurobiological theory of consciousness. *Seminars in the Neurosciences*, 2, 263-275. --- Neural correlates of consciousness; foundational but qualitative.

10. **Shea, N.** (2019). *Representation in Cognitive Science*. Oxford University Press. --- Epistemic dimensions of consciousness.

11. **Butlin, P., et al.** (2023). Consciousness in Artificial Intelligence: Insights from the Science of Consciousness. *arXiv:2308.08708*. --- Multi-theory indicator checklist; qualitative, not a unified differentiable equation.

12. **Tort, A.B.L., et al.** (2010). Measuring phase-amplitude coupling between neuronal oscillations of different frequencies. *Journal of Neurophysiology*, 104(2), 1195-1210. --- PAC methodology; used in neuroscience but not in a consciousness equation.

13. **Putnam, H.** (1967). Psychological predicates. In *Art, Mind, and Religion*. --- Multiple Realizability thesis; philosophical, not computational.

14. **Aaronson, S.** (2014). Why I am not an integrated information theorist. Blog post. --- Critique of IIT; motivates multi-theory approach.

15. **US11119483B2** (2021). System and method for conscious machines. --- Patented system for machine consciousness using neural networks, self-modeling, and temporal tracking. Builds consciousness, does not measure it across multiple theories. Different purpose (construction vs. measurement).

16. **Safron, A.** (2020). An Integrated World Modeling Theory (IWMT) of Consciousness. *Frontiers in AI*. --- Combines IIT and GWT with Free Energy Principle. Theoretical framework only; no computational implementation or measurement equation.

17. **Mayner et al.** PyPhi: A toolbox for Integrated Information Theory. --- Computes exact IIT Phi. Single-theory only. Computationally intractable beyond ~12 nodes. No multi-theory unification.

---

### 13. Figures (Text Descriptions)

**Figure 1: Master Equation Block Diagram**

```
                         7 Core Components
                  [Phi] [B] [W] [A] [R] [E] [K]
                    |     |   |   |   |   |   |
                    +--+--+---+---+---+---+---+
                       |                 |
              PAC(W,B) |                 |
              -------->| B' = B*(1+MI)   |
                       |                 |
                       v                 v
               +--------------+   +------------------+
               | softmin(tau) |   | weighted_sum     |
               |  bottleneck  |   | w_i * C_i * g_i  |
               +--------------+   +------------------+
                       |                 |
                       v                 |
               +--------------+          |
               | sigmoid(k,q) |          |
               |    gate      |          |
               +--------------+          |
                       |                 |
                       +--------*--------+
                                |
                         *------+------*
                         |             |
                    +--------+   +---------+
                    |   S    |   | rho(t)  |
                    |substrate|  |temporal |
                    +--------+   +---------+
                         |             |
                         +------*------+
                                |
                                v
                          +----------+
                          |  C(t)    |
                          | [0, 1]  |
                          +----------+
```

**Figure 2: PAC Modulation Index Computation**

```
  Workspace Phase                    Binding Amplitude
  (low-freq proxy)                   (high-freq signal)
       |                                    |
       v                                    v
  [Normalize to 0..2pi]              [Raw amplitude]
       |                                    |
       +---------------+-------------------+
                        |
                  [Bin by phase]
                  (18 bins, 20 deg)
                        |
                        v
              [Mean amp per bin]
                        |
                        v
              [Normalize to prob P]
                        |
                        v
              [Shannon Entropy H]
                        |
                        v
              MI = (H_max - H) / H_max
              (0 = no coupling,
               1 = perfect coupling)
```

**Figure 3: Substrate Feasibility Architecture**

```
  SubstrateType              SubstrateRequirements
  (8 variants)               (9 dimensions)
       |                            |
       v                            v
  [Pre-built profile]        [consciousness_feasibility()]
       |                            |
       |    +--critical_min = min(causality, integration,
       |    |                       dynamics, recurrence)
       |    +--workspace_factor
       |    +--enhancement_avg = mean(binding, attention, HOT)
       |                            |
       v                            v
  S_raw = critical_min * workspace * (0.5 + 0.5 * enhancement)
                                    |
                    Validation Overlay (optional)
                                    |
                                    v
  S_eff = S_raw * (floor + (1 - floor) * honest_confidence)
         |
         | BiologicalNeurons: confidence = 0.95
         | SiliconDigital:    confidence = 0.10
         | QuantumComputer:   confidence = 0.10
```

**Figure 4: 4-Layer Consciousness Engine Architecture**

```
  HDC/CfC State ──────────────────────────────────────────>
       |                                                   |
       |  [Layer 1: SpectralMIP]    every 97 cycles       |
       |  Fiedler-ordered Phi       Tononi (2004)         |
       |         |                                        |
       |  [Layer 2: MultiModal]     every 13 cycles       |
       |  Cross-modal binding Phi   Damasio (1994)        |
       |         |                                        |
       |  [Layer 3: EquationV2]     every 23 cycles       |
       |  7-theory C(t)            *** THIS INVENTION *** |
       |         |                                        |
       |  [Layer 4: Pipeline]       every 97 cycles       |
       |  End-to-end pipeline       Dehaene (2011)        |
       |         |                                        |
       v         v                                        v
  +----------------------------------------------------------+
  | Unified Consciousness = weighted consensus (dynamic w_i) |
  | w_i self-calibrate from structural Phi emergence ratio   |
  +----------------------------------------------------------+
```

---

*Document prepared for patent counsel review. All technical details derived from implemented and tested source code in the Symthaea cognitive architecture.*
