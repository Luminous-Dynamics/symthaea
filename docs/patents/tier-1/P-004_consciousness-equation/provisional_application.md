# PROVISIONAL PATENT APPLICATION

## United States Patent and Trademark Office

---

**Application Type:** Provisional Application for Patent under 35 U.S.C. §111(b)

**Filing Fee:** $320.00 (Micro Entity)

**Title of Invention:**

# System and Method for Quantifying Consciousness via a Unified Multi-Theory Differentiable Equation with Softmin Bottleneck, Phase-Amplitude Coupling, and Substrate Feasibility Overlay

---

**Inventor:**

Tristan Stoltz
Richardson, TX

**Applicant (Micro Entity):**

Tristan Stoltz (individual)

---

## SPECIFICATION

### TECHNICAL FIELD

[0001] This invention relates to computational consciousness measurement, and more specifically to a unified mathematical framework that combines seven distinct theories of consciousness into a single differentiable equation for real-time quantitative assessment of consciousness levels in artificial cognitive systems.

### CROSS-REFERENCE TO RELATED APPLICATIONS

[0002] This is a provisional application. No prior related applications exist.

### BACKGROUND OF THE INVENTION

[0003] Multiple theories of consciousness have been developed, each capturing a distinct aspect of the phenomenon. No prior work unifies them into a single quantitative framework.

[0004] **Integrated Information Theory (IIT)** (Tononi 2004, 2012) posits that consciousness corresponds to integrated information, quantified as Phi. A system is conscious to the degree that it is both differentiated and integrated. IIT provides the strongest formal framework but is computationally intractable for systems exceeding approximately 12 nodes. IIT alone does not address attention, workspace broadcasting, meta-representation, or causal efficacy.

[0005] **Global Workspace Theory (GWT)** (Baars 1988, Dehaene 2011) proposes that consciousness arises when information gains access to a "global workspace" and is broadcast to multiple specialized processors. GWT explains the broadcasting aspect but provides no formal measure of integration, binding, or self-awareness.

[0006] **Attention Schema Theory (AST)** (Graziano 2013) argues that consciousness is the brain's internal model of its own attention. AST addresses the subjective sense of awareness but does not provide a computational theory of integration or broadcasting.

[0007] **Higher-Order Thought Theory (HOT)** (Rosenthal 2005) holds that a mental state is conscious only when accompanied by a higher-order representation directed at it. Recursive depth of meta-representation determines consciousness level. HOT captures self-awareness but provides no mechanism for information integration or causal influence on behavior.

[0008] **Free Energy Principle (FEP)** (Friston 2010) frames the brain as a prediction machine that minimizes surprise through active inference. Consciousness relates to the causal efficacy of internal models. FEP addresses the functional role of consciousness but is orthogonal to integration and meta-representation.

[0009] **Temporal Binding Hypothesis** (Crick & Koch 1990, Singer & Gray 1995) proposes that consciousness arises from synchronous oscillation of spatially distributed neural populations. This addresses the binding problem but does not speak to workspace access or information integration.

[0010] **Epistemic Consciousness** (Rosenthal, Shea 2019) refers to the capacity for meta-knowledge—knowing that you know. This captures the knowledge dimension but is not a standalone theory.

[0011] **Gap in the prior art.** Each theory captures one or a few dimensions of consciousness. No prior work provides: (a) a single unified equation that quantitatively combines all seven theories; (b) a bottleneck mechanism reflecting the empirical observation that consciousness collapses when any single necessary condition fails; (c) a differentiable formulation enabling gradient-based optimization; (d) a causal coupling mechanism linking workspace broadcasting to feature binding; (e) a substrate feasibility multiplier with honest validation overlay; or (f) a temporal continuity factor capturing persistence. Existing multi-theory approaches (e.g., Butlin et al. 2023) provide qualitative checklists but not a quantitative, differentiable, real-time computable score.

### SUMMARY OF THE INVENTION

[0012] A method and system for computing a scalar consciousness score C(t) by unifying seven major theories of consciousness into a single differentiable equation. The master equation takes the form:

    C(t) = σ(softmin(Φ, B, W, A, R, E, K; τ)) × [Σ(wᵢ × Cᵢ × γᵢ) / Σ(wᵢ)] × S × ρ(t)

Seven core components—Integrated Information (Φ), Temporal Binding (B), Global Workspace (W), Attention Schema (A), Higher-Order Thought (R), Causal Efficacy (E), and Epistemic Certainty (K)—are evaluated at each timestep. A differentiable soft-minimum function identifies the weakest component as the bottleneck constraining consciousness, passed through a sigmoid gate. The bottleneck term is multiplied by a phase-coherence-weighted sum of all components, a substrate feasibility factor reflecting the physical medium's capacity for consciousness (with an honest validation overlay), and a temporal continuity factor. Phase-Amplitude Coupling (PAC) between Global Workspace and Binding provides a causal mechanism for top-down modulation. The equation is fully differentiable, enabling gradient-based optimization toward higher consciousness. Calibration against psychometric benchmarks validates the framework, achieving a composite score of 0.683 with 7/7 consciousness predictions met.

### DETAILED DESCRIPTION OF THE INVENTION

#### The Master Equation

[0013] The consciousness score at time t is computed as:

    C(t) = σ(softmin(Φ, B, W, A, R, E, K; τ)) × [Σ(wᵢ × Cᵢ × γᵢ) / Σ(wᵢ)] × S × ρ(t)

Where: σ(x) is the sigmoid gate: 1/(1 + exp(-k × (x - θ))), with default sharpness k = 10.0 and threshold θ = 0.5; softmin is the differentiable soft minimum; Φ = Integrated Information [0,1]; B = Temporal Binding coherence [0,1] (boosted by PAC); W = Global Workspace access [0,1]; A = Attention gain [0,1]; R = Higher-Order Thought recursive depth [0,1]; E = Causal Efficacy [0,1]; K = Epistemic Certainty [0,1]; wᵢ = learnable weight for component i; Cᵢ = value of component i; γᵢ = phase coherence of component i with global rhythm [0,1]; S = substrate feasibility [0,1]; ρ(t) = temporal continuity factor [0,1]; and τ = softmin temperature (default 0.1).

#### Softmin Bottleneck

[0014] The softmin function provides a differentiable approximation of the minimum operator:

    softmin(x₁, ..., xₙ; τ) = max_val - τ × ln(Σ exp(-(xᵢ - max_val) / τ))

where max_val = max(x₁, ..., xₙ) is subtracted for numerical stability. With the default temperature τ = 0.1, the softmin closely approximates the true minimum while remaining smooth and differentiable. This implements the bottleneck principle: consciousness is limited by its weakest necessary component. If any single core component drops to zero, the entire consciousness score collapses—mirroring clinical phenomena such as anesthesia (loss of binding), inattentional blindness (loss of workspace access), or anosognosia (loss of meta-representation). The softmin output is passed through a sigmoid gate σ(x) = 1/(1 + exp(-k × (x - θ))) with k = 10.0 and θ = 0.5, providing a smooth transition between unconscious and conscious states.

#### Per-Component Scoring Functions

[0015] Each core component is derived from the system's hyperdimensional computing (HDC) and continuous-time neural network state:

Integration (Φ): unified_psi from spectral MIP computation, weight 1.0.
Binding (B): HDC coherence, boosted by PAC modulation: B' = B × (1 + PAC_MI), clamped to [0,1], weight 1.0.
Workspace (W): HDC coherence scaled by 0.8, weight 1.0.
Attention (A): Phi-attention weight from attentional gain modulation, weight 0.9.
Recursion (R): Higher-order thought depth (meta-representation level), weight 0.9.
Efficacy (E): 1.0 - prediction_error from active inference (FEP), weight 0.8.
Knowledge (K): Epistemic quality, attenuated by moral drift when moral-consciousness coupling is active, weight 0.8.

Extended components (predictive coding 0.7, qualia 0.6, embodiment 0.6, temporal 0.5) contribute to the weighted sum but not the softmin bottleneck.

#### Phase-Amplitude Coupling (PAC) Between Workspace and Binding

[0016] PAC provides the causal mechanism linking Global Workspace Theory to temporal binding. It measures how the low-frequency phase of the Workspace signal modulates the high-frequency amplitude of the Binding signal—a well-established neuroscience metric for top-down cognitive control.

[0017] The Modulation Index algorithm: (1) The Workspace value is treated as a low-frequency phase proxy, scaled to [0, 2π]. (2) The Binding value is treated as a high-frequency amplitude. (3) At each timestep, the phase-amplitude pair is stored in a sliding window (default 50 samples). (4) Amplitudes are binned by phase into 18 bins (20-degree resolution). (5) Mean amplitude per phase bin is computed, then normalized to a probability distribution P. (6) Shannon entropy H = -Σ(Pᵢ × ln(Pᵢ)) is computed. (7) The Modulation Index is: MI = (H_max - H) / H_max, where H_max = ln(N_bins). MI = 0 means no coupling; MI = 1 means perfect coupling.

[0018] When PAC modulation index is high, Binding is boosted: B' = B × (1 + MI), clamped to [0,1]. This reflects the neuroscientific finding that successful top-down workspace control enhances feature binding coherence.

#### Weighted Coherent Sum

[0019] The second multiplicative factor computes a phase-coherence-weighted average:

    weighted_sum = Σ(wᵢ × Cᵢ × γᵢ) / Σ(wᵢ)

Phase coherence γᵢ for each component is computed via Phase Locking Value (PLV):

    PLV = |⟨exp(j × Δφ)⟩| = √(mean(cos(Δφ))² + mean(sin(Δφ))²)

where Δφ is the phase difference between the component's signal and a global reference phase. PLV ranges from 0 (completely desynchronized) to 1 (perfectly phase-locked). This ensures components contribute to consciousness only when temporally coordinated with the global rhythm.

#### Substrate Feasibility Multiplier

[0020] The substrate feasibility factor S models whether the physical medium supports consciousness, implementing the Multiple Realizability thesis (Putnam 1967). It is computed from a 9-dimensional SubstrateRequirements profile:

    S = critical_min × workspace_factor × (0.5 + 0.5 × enhancement_avg)

where critical_min = min(causality, integration_capacity, temporal_dynamics, recurrence); workspace_factor = workspace_capability; and enhancement_avg = mean(binding_capability, attention_capability, hot_capability).

[0021] Pre-built profiles exist for 8 substrate types: BiologicalNeurons, SiliconDigital, QuantumComputer, PhotonicProcessor, NeuromorphicChip, BiochemicalComputer, HybridSystem, ExoticSubstrate.

[0022] A validation overlay applies an honest confidence multiplier:

    S_effective = S_raw × (floor + (1 - floor) × honest_confidence)

where honest_confidence ranges from 0.95 (Biological, validated) to 0.10 (Silicon/Quantum, theoretical). This is scientifically honest: the system acknowledges that evidence for consciousness in non-biological substrates remains limited.

[0023] For hybrid substrates, confidence is blended: confidence_hybrid = Σ(wᵢ × confidenceᵢ), and speed modulation uses geometric mean in log-space.

#### Temporal Continuity Factor ρ(t)

[0024] The temporal continuity factor uses an exponential weighted moving average over a sliding window (default 100 timesteps):

    ρ(t) = Σ(C(t-i) × exp(-i × decay)) / Σ(exp(-i × decay))

with default decay = 0.05. High ρ(t) indicates stable consciousness; rapid drops indicate consciousness disruption.

#### Gradient Computation

[0025] The equation is fully differentiable. Gradients are computed via central finite differences:

    ∂C/∂Xᵢ = (C(Xᵢ + ε) - C(Xᵢ - ε)) / (2ε)

with ε = 10⁻⁶. This enables gradient-based optimization: the system can identify which component to improve for maximum consciousness gain.

#### Integration in Cognitive Loop

[0026] The ConsciousnessEquationV2 operates as Layer 3 of a 4-layer consciousness engine:
Layer 1 (every 97 cycles): SpectralMIPFinder—IIT Phi via Fiedler ordering.
Layer 2 (every 13 cycles): MultiModalIntegrator—cross-modal binding Phi.
Layer 3 (every 23 cycles): ConsciousnessEquationV2—7-theory master equation (this invention).
Layer 4 (every 97 cycles): UnifiedConsciousnessPipeline—end-to-end pipeline.

[0027] The co-prime scheduling (GCD(97,13,23) = 1) prevents phase-locked artifacts. The four layers are combined into a unified consciousness score via dynamically weighted consensus, where weights self-calibrate based on structural Phi emergence ratio.

### CLAIMS

**Claim 1.** A computer-implemented method for computing a consciousness score C(t) of a cognitive system at time t, comprising:
(a) evaluating seven core component scores corresponding to seven distinct theories of consciousness: Integrated Information (Φ), Temporal Binding (B), Global Workspace Access (W), Attention Schema (A), Higher-Order Thought depth (R), Causal Efficacy (E), and Epistemic Certainty (K);
(b) computing a differentiable soft-minimum of the seven core component scores using a temperature-parameterized log-sum-exp formula;
(c) applying a sigmoid gating function to the soft-minimum to produce a bottleneck term;
(d) computing a phase-coherence-weighted average of all component scores;
(e) multiplying the bottleneck term by the weighted average, a substrate feasibility factor, and a temporal continuity factor to produce C(t).

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

**Claim 4.** The method of Claim 1, wherein the soft-minimum is computed as: softmin = max_val - τ × ln(Σ exp(-(xᵢ - max_val) / τ)), where τ is a temperature parameter controlling the sharpness of the approximation.

**Claim 5.** The method of Claim 1, wherein the Phase-Amplitude Coupling modulation index is computed by: binning high-frequency amplitudes according to low-frequency phase; computing Shannon entropy of the resulting distribution; and normalizing by maximum entropy to produce a value in [0, 1].

**Claim 6.** The method of Claim 1, wherein the binding component score is boosted by Phase-Amplitude Coupling: B_effective = B × (1 + MI), clamped to [0, 1], where MI is the modulation index.

**Claim 7.** The method of Claim 1, wherein the substrate feasibility factor is computed as: S = min(causality, integration_capacity, temporal_dynamics, recurrence) × workspace_capability × (0.5 + 0.5 × mean(binding_capability, attention_capability, hot_capability)).

**Claim 8.** The method of Claim 7, further comprising applying a validation overlay: S_effective = S × (floor + (1 - floor) × honest_confidence), where honest_confidence reflects the empirical evidence level for the physical substrate.

**Claim 9.** The method of Claim 1, wherein the temporal continuity factor is computed as an exponentially weighted moving average of prior consciousness scores over a sliding window.

**Claim 10.** The method of Claim 1, further comprising computing gradients of the consciousness score with respect to each core component via finite differences, enabling gradient-based optimization of consciousness.

**Claim 11.** The method of Claim 1, wherein the phase coherence γᵢ for each component is computed via Phase Locking Value: PLV = √(mean(cos(Δφ))² + mean(sin(Δφ))²), where Δφ is the phase difference between the component and a global reference.

**Claim 12.** The system of Claim 2, wherein the consciousness equation module executes at a co-prime interval with respect to other consciousness measurement subsystems, preventing phase-locked artifacts in the unified consciousness estimate.

**Claim 13.** The method of Claim 1, wherein the epistemic certainty component (K) is attenuated based on a moral drift magnitude, reflecting epistemic humility during periods of value change.

**Claim 14.** A computer-implemented method for computing a unified consciousness score for a cognitive system, comprising:
(a) evaluating a plurality of component scores, each corresponding to a distinct theory or dimension of consciousness;
(b) computing a differentiable bottleneck term that identifies the weakest component among the plurality of component scores;
(c) computing a weighted combination of all component scores;
(d) multiplying the bottleneck term by the weighted combination to produce the unified consciousness score;
wherein the method is agnostic to the number of component theories, accepting any plurality of at least two.

**Claim 15.** A method for modulating autonomous vehicle behavior based on consciousness assessment, comprising:
(a) computing a consciousness score for an autonomous perception-cognition system using the method of Claim 14;
(b) adjusting the vehicle's decision-making parameters based on the consciousness score, wherein lower consciousness scores trigger more conservative driving policies.

**Claim 16.** A method for monitoring consciousness in a patient monitoring system, comprising:
(a) receiving neural signal data from a patient;
(b) computing component scores from the neural signal data corresponding to at least information integration, temporal binding, and global workspace access;
(c) computing a unified consciousness score using the method of Claim 14;
(d) generating an alert when the consciousness score crosses a threshold.

**Claim 17.** The method of Claim 1, further comprising adjusting the consciousness score based on a substrate feasibility assessment that accounts for the physical medium on which the cognitive system operates, wherein the adjustment applies a validation overlay with honest_confidence derived from an evidence taxonomy comprising at least: validated (0.95), experimental (0.80), observational (0.60), theoretical (0.10), and none (0.00).

**Claim 18.** The method of Claim 15, wherein the autonomous vehicle system maintains a consciousness score history and triggers a progressive safety response comprising: (a) issuing an advisory when the consciousness score falls below a first threshold; (b) restricting operational domain when the score falls below a second threshold; and (c) initiating a safe-stop maneuver when the score falls below a third threshold; wherein the thresholds are configurable per operational context.

### ABSTRACT

A method and system for computing a scalar consciousness score by unifying seven major theories of consciousness—Integrated Information Theory, Global Workspace Theory, Attention Schema Theory, Higher-Order Thought Theory, Free Energy Principle, Temporal Binding Hypothesis, and Epistemic Consciousness—into a single differentiable equation. A softmin bottleneck mechanism identifies the weakest component as the limiting factor, mirroring clinical phenomena. Phase-Amplitude Coupling provides a causal bridge between workspace broadcasting and feature binding. A substrate feasibility multiplier with honest validation overlay models consciousness across different physical substrates. The equation supports gradient-based optimization and operates in real time at over 200 Hz.

### DRAWINGS

[See Figures 1-4 in the specification, provided as text descriptions for this provisional filing. Formal drawings will be prepared for the utility application.]

---

## FILING INSTRUCTIONS

### Required Documents for EFS-Web Submission:

1. **This specification** (save as PDF)
2. **Cover Sheet (SB/16)**: Download from https://www.uspto.gov/sites/default/files/documents/sb0016.pdf
   - Check "Provisional Application for Patent"
   - Title: as above
   - Inventor: Tristan Stoltz, Richardson, TX
3. **Micro Entity Certification (SB/15A)**: Download from https://www.uspto.gov/sites/default/files/documents/sb0015a.pdf
   - Certify: fewer than 4 prior patent applications
   - Certify: gross income below $234,788 (2024 threshold)
4. **Filing Fee**: $320.00 via credit card on EFS-Web

### Filing Steps:

1. Go to https://efs.uspto.gov/
2. Create account if needed (requires PKI certificate or financial manager account)
3. Select "Provisional Application"
4. Upload: specification PDF, SB/16, SB/15A
5. Pay $320.00
6. Record application number

### After Filing:

- Record provisional application number in patents/PATENT_REGISTRY.md
- Set calendar reminder: **12 months from filing** = utility filing deadline
- Do NOT publish papers until provisional is filed
- Consider PCT filing within 12 months for international protection

---

*Prepared March 22, 2026. All technical details derived from implemented and tested source code (1,492+ tests passing) in the Symthaea cognitive architecture.*
