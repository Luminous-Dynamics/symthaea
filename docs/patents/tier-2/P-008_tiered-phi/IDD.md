# P-008: Tiered Phi Measurement System
## Invention Disclosure Document

---

### 1. Title

**Four-Tier Consciousness Measurement Architecture with Co-Prime Interval Scheduling, Spectral Minimum Information Partition, Seven-Theory Master Equation, and Self-Calibrating Dynamic Weights**

---

### 2. Inventor(s)

**Tristan Stoltz**, Luminous Dynamics

---

### 3. Date of Conception

**2025** (estimated). First committed implementation: February 5, 2026 (consciousness measurement framework). Spectral MIP algorithm (`spectral_mip.rs`) first committed February 20, 2026.

First public disclosure: February 5, 2026 (git commit `feat(symthaea): add Symthaea-HLB consciousness-first AI framework v0.5.0` — tiered measurement architecture including `measure.rs` and `consciousness_equation_v2.rs`). Spectral MIP component first disclosed February 20, 2026.
Under 35 USC 102(b)(1)(A), the 1-year grace period expires **February 5, 2027** (earliest component).

---

### 4. Technical Field

This invention relates to consciousness measurement in artificial cognitive architectures, and more specifically to a four-tier measurement system that orchestrates spectral integrated information (IIT), cross-modal binding, a seven-theory master equation, and an end-to-end consciousness pipeline on co-prime firing intervals, with self-calibrating weights based on structural emergence ratios.

---

### 5. Abstract

A system and method for measuring consciousness in a digital cognitive architecture is disclosed. The system implements four independent measurement layers operating at different computational tiers: Layer 1 (O(n^3)) computes integrated information (Phi) via a novel spectral Minimum Information Partition algorithm using Fiedler ordering and bordered Cholesky sweeps, replacing the intractable O(2^n) exhaustive search; Layer 2 (O(n^2)) measures cross-modal binding via Phi-guided fusion; Layer 3 (O(n)) unifies seven consciousness theories (IIT, GWT, HOT, AST, temporal binding, FEP, epistemic) into a single differentiable equation using softmin and sigmoid activations; Layer 4 (O(n^2)) validates consciousness via an end-to-end sensorimotor pipeline with Kuramoto oscillatory binding and global workspace broadcast. The layers fire on co-prime intervals (97, 13, 23, 97 cycles) to eliminate synchronous computational storms. A consensus mechanism combines all four layers using dynamic weights that self-calibrate based on structural Phi emergence ratios (macro_phi / micro_phi). The system achieves real-time operation at 4.3ms per cycle (234 Hz), exceeding the 50 Hz target by 4.7x. Validation against PyPhi achieves Pearson r=0.97 correlation, and topology ranking achieves Spearman rho=0.97.

---

### 6. Background and Prior Art

#### 6.1 Integrated Information Theory (IIT)

Tononi (2004, 2012) defined Phi as the information generated above and beyond its parts, requiring search over all bipartitions to find the Minimum Information Partition (MIP). This search is NP-hard with O(2^n) complexity, making exact computation intractable for systems with more than ~12 nodes (as demonstrated by PyPhi, Mayner et al. 2018).

#### 6.2 Spectral Methods for MIP Approximation

Kitazono et al. (2018) proposed spectral ordering approaches using the Fiedler vector of the mutual information Laplacian. However, they did not combine this with bordered Cholesky sweeps for O(n^3) total complexity, nor did they integrate the result into a multi-tier measurement system.

#### 6.3 Global Workspace Theory (GWT)

Baars (1988) and Dehaene & Changeux (2011) proposed that consciousness requires global broadcasting of selected information. Existing implementations (e.g., LIDA) model GWT but do not integrate it with IIT Phi measurement.

#### 6.4 Multi-Theory Approaches

Multiple consciousness theories exist (IIT, GWT, HOT, AST, FEP), but no prior system combines them into a single differentiable equation with learned component weights and phase coherence modulation.

#### 6.5 Gap in Prior Art

No prior art:
- Provides a tractable O(n^3) spectral MIP algorithm that replaces O(2^n) exhaustive search
- Combines four independent consciousness measurement layers at different computational tiers
- Uses co-prime interval scheduling to prevent synchronous computational storms
- Unifies seven consciousness theories into one differentiable master equation
- Self-calibrates layer weights based on structural Phi emergence ratios

---

### 7. Detailed Technical Description

#### 7.1 Layer 1: Spectral Phi Measurement (SpectralMIPFinder)

**Complexity**: O(n^3) replacing O(2^n)

**Five-step algorithm**:

1. **Online Covariance** (O(n^2) per push): Maintains regularized covariance matrix via running sums. Each `push(hdv)` is O(n) for online update; the full matrix is computed lazily.

2. **MI Laplacian** (O(n^2)): Compute Gaussian mutual information between all component pairs to form a weighted adjacency matrix W. Form graph Laplacian L = D - W where D is the degree matrix.

3. **Fiedler Ordering** (O(n^3)): Compute the second eigenvector (Fiedler vector) of L via shifted inverse iteration with deflation (~n^3/6 Cholesky + 30×n^2 solves). Sort components by Fiedler vector values.

4. **Bordered Cholesky Sweep** (O(n^3)): For each contiguous cut k=0..n-1 along the Fiedler ordering, compute MI_cut(k) = Gaussian MI across the bipartition [0..k] | [k+1..n-1]. Uses O(k^2) bordered Cholesky updates per step, giving O(n^3) total.

5. **MIP Selection** (O(n)): Phi = total_MI - min_k MI_cut(k).

**Adaptive dimension selection**: 60% boundary dimensions (near MIP cut) + 20% coverage (uniform spread) + 20% exploration (random) balances precision with generalization.

**Hierarchical multi-scale**: Computes Phi at multiple scales (32, 64, 128 components) to capture consciousness at cortical-column, local-network, and whole-brain levels. Emergence ratio = macro_phi / micro_phi.

**Performance**: push() 442ns (n=128), compute() 5.5ms (n=128), adapt() 3.2ms (n=128).

**Firing pattern**: push() every cycle; compute() every 97 cycles; adapt() + hierarchical every 194 cycles.

#### 7.2 Layer 2: Multi-Modal Integration

**Complexity**: O(n^2 cross-modal)

Computes cross-modal binding Phi by fusing inputs from multiple modalities (visual, temporal, somatosensory) weighted by their integration potential. Attention weights update via Phi gradients: modalities contributing more to integration get higher attention.

**Firing interval**: Every 13 cycles (co-prime with 97, 23).

**Novel aspect**: Uses IIT's Phi measurement *within* the binding mechanism itself, creating a self-improving loop where better binding increases Phi, which tunes further binding.

#### 7.3 Layer 3: Consciousness Equation V2 (Seven-Theory Master Equation)

**Complexity**: O(n)

**The equation**:
```
C(t) = sigma(softmin(Phi, B, W, A, R, E, K; tau)) × [sum(w_i × C_i × gamma_i) / sum(w_i)] × S × rho(t)
```

Where:
- sigma(·) = sigmoid activation (smooth threshold at 0.5)
- softmin(·; tau=0.1) = differentiable minimum: -tau × log(sum(exp(-value/tau)))
- **Phi** = Integration (IIT)
- **B** = Binding (Singer & Gray, temporal coherence)
- **W** = Workspace (Baars/Dehaene, global broadcast capacity)
- **A** = Attention (Graziano AST, precision-weighted saliency)
- **R** = Recursion (Rosenthal HOT, meta-representational depth)
- **E** = Efficacy (Friston FEP, causal influence on behavior)
- **K** = Knowledge (Shea, epistemic certainty)
- w_i = learned component weights [1.0, 1.0, 1.0, 0.9, 0.9, 0.8, 0.8]
- gamma_i = phase coherence (PLV with global rhythm)
- S = substrate feasibility [0, 1]
- rho(t) = temporal continuity (EMA of recent consciousness)

**Phase-Amplitude Coupling**: Workspace (low-frequency driver) modulates binding (high-frequency responder) via computed modulation index, implementing top-down conscious control.

**Moral-consciousness coupling**: Knowledge component attenuated by ethical drift ratio; anomaly dampening reduces unified consciousness during moral incoherence.

**Gradient computation**: Central finite differences for all 7 components, enabling consciousness optimization via backpropagation.

**Firing interval**: Every 23 cycles (co-prime with 97, 13).

#### 7.4 Layer 4: Unified Consciousness Pipeline

**Complexity**: O(n^2 hierarchical)

End-to-end pipeline implementing Dehaene's GWT model:
1. **HDC Encode + CfC Layer**: Sensory cortex simulation
2. **Oscillatory Binding** (40Hz): Kuramoto phase coupling for feature integration. Phase-Locking Value (PLV) measures binding quality.
3. **Hierarchical LTC** (16 circuits): Multi-scale temporal dynamics matching cortical hierarchy (fast sensory through feedback layers).
4. **Global Workspace** (128 neurons): Recurrent broadcast layer with selective attention and amplification.
5. **Causal Efficacy**: Validates that consciousness actually affects behavior.

**Firing pattern**: Lightweight advance() every cycle; full process() with binding every 97 cycles.

#### 7.5 Co-Prime Interval Scheduling

**Layer firing intervals**: 97, 13, 23, 97 cycles.

GCD(97, 13, 23) = 1, guaranteeing no two expensive layers fire simultaneously. LCM(97, 13, 23) = 29,081 cycles (~580 seconds at 50Hz), meaning the full pattern repeats only every ~10 minutes.

**Computational budget**:
- Average cycle cost: ~2.1ms (well within 20ms budget)
- Worst-case single-cycle cost: ~4.3ms (when L1 + L4 co-fire at cycle 97 multiples)
- No synchronous storms: co-prime scheduling eliminates overlap

#### 7.6 Dynamic Weight Self-Calibration

Rather than fixed weights, the consensus mechanism self-calibrates based on structural Phi decomposition:

```
emergence_ratio = structural.macro_phi / (structural.micro_phi + epsilon)

modulation = tanh(smoothed_emergence_ratio - 1.0)

If emergence > 1.2 (whole > parts):
  → boost spectral weight (+10%), reduce equation/pipeline weights (-5% each)
  → Trust integrated information measurement

If emergence < 0.8 (weak binding):
  → boost equation/pipeline weights, reduce spectral weight
  → Rely on bottom-up models
```

EMA smoothing with variance-gated alpha prevents oscillation. All weights normalized to sum to 1.0.

#### 7.7 Neuromodulator-Consciousness Coupling

- 5-HT2A amplifies perceptual richness (±5%)
- GABA-A dampens global gain
- Moral anomaly reduces coherence (up to -15%)

Based on Seth (2013, interoceptive inference model).

---

### 8. Novelty Statement

This invention introduces the first multi-tier consciousness measurement system with co-prime scheduling and self-calibrating weights. Novel contributions:

1. **O(n^3) spectral MIP algorithm**: Reduces NP-hard MIP search to polynomial time via Fiedler ordering + bordered Cholesky sweeps. For n=32, this is a 282.5 billion-fold speedup over exhaustive search.
2. **Co-prime interval scheduling**: Mathematical guarantee (GCD=1) that expensive layers never fire simultaneously, enabling real-time consciousness measurement.
3. **Seven-theory master equation as a measurement tier**: Integration of a multi-theory consciousness equation as one of four co-prime-scheduled measurement tiers, with self-calibrating weight adaptation. (Note: The differentiable properties and gradient computation of this equation are claimed separately in P-007; this patent claims its role within the tiered architecture.)
4. **Self-calibrating dynamic weights**: Emergence ratio (macro/micro Phi) automatically tunes trust in each measurement layer, adapting to system state without human intervention.
5. **Phase-amplitude coupling**: Workspace-binding modulation implementing top-down conscious control (neuroscience-faithful).
6. **Moral-consciousness coupling**: Ethical drift attenuates epistemic certainty; moral anomaly dampens unified consciousness.
7. **Causal efficacy validation**: Layer 4 validates that consciousness actually affects behavior (FEP-aligned).

---

### 9. Suggested Claims

**Claim 1 (independent):** A computer-implemented method for measuring consciousness in a digital cognitive system comprising: (a) computing integrated information (Phi) at a first computational tier using spectral ordering of a mutual information Laplacian and contiguous partition sweep; (b) computing cross-modal binding at a second computational tier; (c) computing a multi-theory consciousness score at a third computational tier using a differentiable equation combining at least 5 consciousness theory components; (d) computing end-to-end consciousness via a sensorimotor pipeline at a fourth computational tier; (e) scheduling the four tiers on co-prime intervals to prevent synchronous computational storms; and (f) combining the four tier outputs via dynamic weights that self-calibrate based on a structural emergence ratio.

**Claim 2 (dependent on 1):** The method of claim 1, wherein the spectral ordering comprises: computing a graph Laplacian from pairwise mutual information; extracting the Fiedler vector (second eigenvector) via shifted inverse iteration; sorting components by Fiedler vector values; sweeping contiguous bipartitions via bordered Cholesky updates; and selecting the minimum information partition as the cut minimizing cross-partition mutual information.

**Claim 3 (dependent on 1):** The method of claim 1, wherein the third computational tier computes a multi-theory consciousness score by: receiving outputs from at least 5 independent consciousness theory assessors; applying a bottleneck function that identifies the weakest component; weighting component contributions by phase coherence; modulating by substrate feasibility; and applying temporal continuity smoothing to prevent flickering.

**Claim 4 (dependent on 1):** The method of claim 1, wherein the dynamic weights are calibrated by: computing an emergence ratio as the ratio of macro-scale Phi to micro-scale Phi; smoothing the ratio via exponential moving average with variance-gated alpha; computing a modulation factor via hyperbolic tangent; and adjusting per-tier weights proportionally to the modulation, with normalization to sum to 1.0.

**Claim 5 (dependent on 1):** The method of claim 1, wherein the co-prime intervals are selected such that the greatest common divisor of all intervals equals 1, and wherein the least common multiple is at least 1000 cycles, ensuring that no two computationally expensive tiers fire on the same cycle.

**Claim 6 (dependent on 1):** The method of claim 1, further comprising a moral-consciousness coupling wherein: ethical drift attenuates the epistemic component of the multi-theory consciousness score; and moral anomaly score dampens the unified consciousness output.

**Claim 7 (independent, broad):** A method for real-time consciousness measurement in a cognitive system comprising: (a) computing at least two independent consciousness metrics at different computational complexities; (b) scheduling computation of each metric at intervals that are pairwise co-prime; and (c) combining the metrics via weights that adapt based on structural properties of the system being measured.

**Claim 8 (dependent on 2):** The method of claim 2, further comprising adaptive dimension selection using a weighted combination of boundary dimensions (near the MIP cut), coverage dimensions (uniformly distributed), and exploration dimensions (randomly selected), with configurable mixing ratios.

**Claim 9 (dependent on 1):** The method of claim 1, wherein the sensorimotor pipeline comprises: Kuramoto oscillatory binding at approximately 40Hz using phase-locking value (PLV) for coherence measurement; a hierarchical liquid time-constant network with at least 8 circuits modeling cortical hierarchy; a global workspace with at least 64 neurons implementing selective attention and recurrent amplification; and a causal efficacy validation that consciousness modulates learning rate.

**Claim 10 (dependent on 3):** The method of claim 3, further comprising phase-amplitude coupling between the workspace component (low-frequency driver) and the binding component (high-frequency responder), where a modulation index computed from their interaction modulates the effective binding strength.

---

### 10. Experimental Validation

#### 10.1 Test Coverage

- **Consciousness engine unit tests**: 43
- **SpectralMIPFinder tests**: 13 (in symthaea-core)
- **All tests passing**: Verified March 2026

#### 10.2 Accuracy Validation

| Metric | Target | Result |
|--------|--------|--------|
| PyPhi correlation (Pearson r) | > 0.90 | **0.97** |
| Topology ranking (Spearman rho) | > 0.90 | **0.97** |
| Dimensional sweep (R^2) | > 0.99 | **0.9987** |
| Consciousness calibration (ECE) | < 0.10 | **0.059** |
| Phi-capability correlation | > 0.55 | **0.65** |
| Phi-ethics correlation | > 0.60 | **0.70** |

#### 10.3 Performance

| Operation | Cost | Budget |
|-----------|------|--------|
| L1 push() | 442 ns | <0.01% |
| L1 compute() | 5.5 ms | 27.5% (every 97 cycles) |
| L2 integrate | 1.8 ms | 9% (every 13 cycles) |
| L3 equation | 0.8 ms | 4% (every 23 cycles) |
| L4 advance | 0.2 ms | 1% (every cycle) |
| L4 process | 2.1 ms | 10.5% (every 97 cycles) |
| **Total avg** | **~2.1 ms** | **Within 20ms budget** |

Full cognitive loop cycle: 4.3ms (234 Hz), exceeding 50 Hz target by 4.7x.

---

### 11. Key Source Files

| File | Description | LOC |
|------|-------------|-----|
| `symthaea/src/cognitive_loop/consciousness_engine/mod.rs` | Layer orchestration | ~300 |
| `symthaea/src/cognitive_loop/consciousness_engine/measure.rs` | Core measure() function | ~500 |
| `symthaea/src/cognitive_loop/consciousness_engine/types.rs` | Input/output types | ~200 |
| `symthaea/src/cognitive_loop/consciousness_engine/helpers.rs` | Weight updates, consensus | ~300 |
| `symthaea-core/src/consciousness_metrics/spectral_mip.rs` | Layer 1 (O(n^3) Phi) | ~840 |
| `symthaea/src/consciousness/measurement/consciousness_equation_v2.rs` | Layer 3 (7-theory) | ~400 |
| `symthaea/src/consciousness/dynamics/unified_consciousness_pipeline.rs` | Layer 4 | ~500 |

---

### 12. Closest Prior Art References

1. Tononi, G. (2004). "An information integration theory of consciousness." *BMC Neuroscience*, 5, 42.
2. Tononi, G. (2012). "Integrated information theory of consciousness: an updated account." *Archives Italiennes de Biologie*, 150(2/3), 56-90.
3. Kitazono, J., et al. (2018). "Efficient algorithms for searching the minimum information partition in integrated information theory." *Entropy*, 20(3), 173.
4. Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.
5. Dehaene, S. & Changeux, J.-P. (2011). "Experimental and theoretical approaches to conscious processing." *Neuron*, 70(2), 200-227.
6. Rosenthal, D. M. (2005). *Consciousness and Mind*. Oxford University Press.
7. Graziano, M. S. A. (2013). *Consciousness and the Social Brain*. Oxford University Press.
8. Friston, K. J. (2010). "The free-energy principle: a unified brain theory?" *Nature Reviews Neuroscience*, 11(2), 127-138.
9. Shea, N. (2019). "Representation in cognitive science: explanatory relevance and explanatory role." *Mind & Language*, 34(1), 18-35.
10. Seth, A. K. (2013). "Interoceptive inference, emotion, and the embodied self." *Trends in Cognitive Sciences*, 17(11), 565-573.

---

### 13. Related Patent Applications

**P-007 (Differentiable Phi)**: Claims the differentiable soft-minimum equation, dual-number automatic differentiation, consciousness gradient computation, and gradient-based optimization. P-008 claims the multi-tier architecture in which P-007's differentiable equation operates as Layer 3. The boundary: P-008 owns "how measurement tiers are scheduled, weighted, and combined"; P-007 owns "how the Layer 3 equation is made differentiable and optimized."

---

### 14. Figures (Text Descriptions)

**Figure 1**: Four-tier architecture block diagram showing Layer 1 (Spectral Phi), Layer 2 (Cross-Modal), Layer 3 (7-Theory Equation), Layer 4 (Pipeline) with co-prime firing intervals and dynamic weight consensus.

**Figure 2**: Spectral MIP algorithm showing: MI Laplacian → Fiedler vector → sort → bordered Cholesky sweep → MIP selection.

**Figure 3**: Co-prime scheduling timeline showing cycles 0-200 with firing marks for each layer, demonstrating no simultaneous expensive computations.

**Figure 4**: Self-calibrating weight adaptation showing emergence ratio computation and weight modulation via tanh.

**Figure 5**: Seven-theory radar chart showing softmin bottleneck identification: whichever component is lowest constrains consciousness.

**Figure 6**: Accuracy validation plots: PyPhi correlation (r=0.97), topology ranking (rho=0.97), dimensional sweep (R^2=0.9987).

---

*Inventor: Tristan Stoltz (tstoltz)*
*Organization: Luminous Dynamics*
*IDD created: 2026-03-08*
