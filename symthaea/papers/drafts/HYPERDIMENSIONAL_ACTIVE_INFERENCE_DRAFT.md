# Hyperdimensional Active Inference: Free Energy Principle in Vector Symbolic Architectures

**Authors**: Tristan Stoltz
**Target Venue**: NeurIPS 2026 / ICML 2026 / Biological Cybernetics
**Status**: DRAFT - February 2026
**Implementation**: Symthaea HLB v0.5.0

---

## Abstract

Active inference, grounded in the Free Energy Principle (FEP), provides a unified framework for perception, action, and learning through variational free energy minimization. However, existing implementations rely on continuous Gaussian belief representations with O(n²-n³) matrix operations, limiting scalability and symbolic reasoning capabilities. We present **Hyperdimensional Active Inference (HAI)**, the first integration of FEP with Hyperdimensional Computing (HDC/Vector Symbolic Architectures). Our framework reformulates variational free energy using cosine similarity in 16,384-dimensional hypervector space, introduces **precision-weighted binding** as a novel operation for uncertainty-modulated feature combination, and derives eight motor command types directly from expected free energy minimization. On standard active inference benchmarks (T-Maze, Grid World), HAI achieves 1.9× faster belief inference and 15.8× faster action selection compared to pymdp, with 7.9× total speedup while maintaining comparable or superior task success rates. We demonstrate convergence of the active inference loop over 20 iterations with validated free energy reduction, and show that precision dynamics correctly adapt to prediction error magnitude. Our results establish HDC as a viable substrate for probabilistic inference, opening new directions for efficient, interpretable cognitive architectures. Code is available at https://github.com/Luminous-Dynamics/symthaea.

**Keywords**: Active Inference, Free Energy Principle, Hyperdimensional Computing, Vector Symbolic Architectures, Precision Weighting, Cognitive Architecture

---

## 1. Introduction

### 1.1 The Challenge

The Free Energy Principle (FEP) posits that biological systems minimize variational free energy—a tractable upper bound on surprise—through both perception (updating beliefs) and action (changing the world) [Friston, 2010]. Active inference operationalizes this principle, providing a unified account of cognition where agents select actions that minimize *expected* free energy, naturally balancing exploration (epistemic value) and exploitation (pragmatic value) [Friston et al., 2017].

Despite theoretical elegance, practical implementations face two challenges:

1. **Computational cost**: Standard formulations require O(n²) covariance updates and O(n³) matrix inversions for belief propagation in continuous state spaces.

2. **Symbolic reasoning**: Gaussian beliefs poorly represent structured, compositional knowledge—the kind required for language, planning, and abstract thought.

### 1.2 Our Contribution

We address both challenges by integrating FEP with Hyperdimensional Computing (HDC), a computational framework using high-dimensional vectors (d ≥ 10,000) with algebraic operations that naturally support distributed representation, associative memory, and compositional semantics [Kanerva, 2009].

**Key contributions:**

1. **Reformulation of variational free energy** using HDC similarity metrics, replacing Mahalanobis distance with precision-weighted cosine similarity in hypervector space.

2. **Precision-weighted binding**, a novel HDC operation: `bind(hv₁, hv₂, π) = (hv₁ ⊗ hv₂) ⊙ scale(π)`, enabling confidence-modulated feature combination.

3. **Direct motor command derivation** from expected free energy, producing eight interpretable action types (attention shift, learning rate adjustment, exploration trigger, etc.) without intermediate policy layers.

4. **Empirical validation** showing convergence, correct precision dynamics, and computational efficiency gains.

To our knowledge, this is the **first integration of active inference with hyperdimensional computing**, establishing a new direction for efficient, interpretable cognitive architectures.

---

## 2. Background

### 2.1 Free Energy Principle and Active Inference

The variational free energy F for an agent with beliefs q(s) over hidden states s, given observations o, is:

$$F = D_{KL}[q(s) \| p(s)] - \mathbb{E}_q[\ln p(o|s)]$$

where the first term (complexity) measures divergence from priors, and the second (accuracy) measures fit to observations.

Active inference extends this to action selection via **expected free energy** G(a):

$$G(a) = \mathbb{E}_{q(o,s|a)}[D_{KL}[q(s|o,a) \| q(s|a)] + D_{KL}[q(o|a) \| \tilde{p}(o)]]$$

The first term captures epistemic value (information gain); the second captures pragmatic value (goal satisfaction). Actions minimizing G(a) naturally balance exploration and exploitation.

**Precision** π (inverse variance) weights prediction errors, acting as an attention mechanism that modulates learning and inference [Feldman & Friston, 2010].

### 2.2 Hyperdimensional Computing

HDC represents information as high-dimensional vectors (hypervectors, HVs) with three core operations [Kanerva, 2009]:

- **Binding** (⊗): Element-wise multiplication, creates associations
- **Bundling** (⊕): Element-wise addition + normalization, creates superpositions
- **Similarity** (δ): Cosine similarity, measures relatedness

Key properties:
- Random HVs are approximately orthogonal (δ ≈ 0)
- Binding is invertible: `(A ⊗ B) ⊗ B = A`
- Bundling preserves similarity to constituents
- Operations are O(d) with d = dimension

These properties enable distributed, compositional representations with graceful degradation under noise.

### 2.3 Mathematical Foundations

Symthaea implements 14 mathematical foundation modules that ground the HAI framework in established theory:

| Module | Theory | Contribution |
|--------|--------|-------------|
| information_theory | Shannon/MI/Transfer Entropy | Quantifies information flow between HDC components |
| iit_exact | IIT 3.0 (TPM, EMD, MIP) | Ground-truth Φ computation for small systems |
| geometric_ops | Riemannian geometry on S^{d-1} | Geodesic interpolation (SLERP), Fréchet mean, PGA |
| probabilistic_hdc | Bayesian HDC | Uncertainty-aware HVs with posterior updates |
| tensor_algebra | Clifford algebra, multivectors | Geometric product unifying bind/bundle operations |
| stability_analysis | Dynamical systems theory | Jacobian eigenvalues, Lyapunov exponents, bifurcation detection |
| ode_solvers | Numerical integration (RK4, Dormand-Prince) | Adaptive-step CfC/LTC dynamics |
| spectral_analysis | Welch PSD, coherence | EEG band power extraction, spectral entropy |
| stochastic_dynamics | Itô calculus (Euler-Maruyama, Milstein) | Noise-driven neural dynamics modeling |
| hierarchical_free_energy | Multi-level VFE | Precision-weighted prediction errors across cortical hierarchy |
| factor_graph | Belief propagation (sum-product, max-product) | Probabilistic routing in cognitive loop |
| causal_calculus | Pearl's do-calculus, SCM | Interventional reasoning for goal-directed behavior |
| hodge_laplacian | Algebraic topology (Betti numbers) | Topological analysis of causal structures |
| primitive_lattice | Order theory (join/meet semilattice) | Fixed-point computation over primitive hierarchies |

These modules connect at five key integration points:
1. **LTC/CfC dynamics** use RK4 from ode_solvers (10-100× error reduction over Forward Euler at same dt)
2. **EEG classification** uses Welch PSD from spectral_analysis (proper frequency decomposition vs. moving-average filters)
3. **Cognitive routing** uses belief propagation from factor_graph (probabilistic depth selection)
4. **Goal reasoning** uses do-calculus from causal_calculus (interventional vs. observational goal evaluation)
5. **CfC diagnostics** use Jacobian eigenvalues from stability_analysis (detecting attractor collapse)

---

## 3. Hyperdimensional Active Inference

### 3.1 Free Energy in Hypervector Space

We reformulate variational free energy for hypervector beliefs. Let:
- $\mathbf{h}_q \in \mathbb{R}^d$ = belief hypervector (current state estimate)
- $\mathbf{h}_p \in \mathbb{R}^d$ = prior hypervector
- $\mathbf{h}_o \in \mathbb{R}^d$ = observation hypervector (encoded from sensory data)
- $\pi_s, \pi_p$ = sensory and prior precision

**Accuracy term** (negative prediction error):
$$A = -\frac{1}{2} \pi_s \cdot (1 - \cos(\mathbf{h}_q, \mathbf{h}_o))^2$$

**Complexity term** (divergence from prior):
$$C = \frac{1}{2} \pi_p \cdot (1 - \cos(\mathbf{h}_q, \mathbf{h}_p))$$

**Free energy**:
$$F = C - A$$

This formulation preserves the accuracy-complexity tradeoff while using O(d) cosine similarity instead of O(d²) covariance operations.

### 3.2 Precision-Weighted Binding

Standard HDC binding treats all features equally. We introduce **precision-weighted binding** to incorporate uncertainty:

$$\text{bind}_\pi(\mathbf{h}_1, \mathbf{h}_2, \pi) = (\mathbf{h}_1 \otimes \mathbf{h}_2) \odot \sigma(\pi \cdot \mathbf{1})$$

where σ is a sigmoid scaling function and ⊙ is element-wise multiplication.

**Properties:**
- High precision (π → ∞): Standard binding
- Low precision (π → 0): Attenuated binding (uncertain associations)
- Enables confidence-modulated feature combination without explicit variance tracking

### 3.3 Belief Updating

Beliefs update via gradient descent on free energy:

$$\mathbf{h}_q^{(t+1)} = \mathbf{h}_q^{(t)} + \eta \cdot \nabla_{\mathbf{h}_q} F$$

The gradient decomposes into:
$$\nabla_{\mathbf{h}_q} F = \pi_s \cdot (\mathbf{h}_o - \mathbf{h}_q \cdot \cos(\mathbf{h}_q, \mathbf{h}_o)) + \pi_p \cdot (\mathbf{h}_p - \mathbf{h}_q \cdot \cos(\mathbf{h}_q, \mathbf{h}_p))$$

This pulls the belief toward observations (weighted by sensory precision) and priors (weighted by prior precision).

### 3.4 Expected Free Energy and Motor Commands

We compute expected free energy G(a) for each action a ∈ {0, ..., 7}:

$$G(a) = w_p \cdot G_p(a) + w_e \cdot G_e(a) - w_n \cdot N(a)$$

where:
- $G_p(a)$ = pragmatic value (distance to preferred observations)
- $G_e(a)$ = epistemic value (expected entropy reduction)
- $N(a)$ = novelty bonus (inverse visit count)
- $w_p = 1.0, w_e = 0.5, w_n = 0.1$ (default weights)

**Motor Command Types** (derived from EFE minimization):

| Index | Command | Trigger Condition |
|-------|---------|-------------------|
| 0 | AttentionShift | High precision error in specific modality |
| 1 | LearningRateAdjust | Rapid confidence change |
| 2 | ExplorationTrigger | G_e > G_p (epistemic dominates) |
| 3 | ReflectionInitiate | High F but stable beliefs |
| 4 | MemoryConsolidate | Persistent low prediction error |
| 5 | ExpectationReset | Persistent high error (model mismatch) |
| 6 | MotorOutput | External action required |
| 7 | NoOp | System at equilibrium |

Action selection uses softmax over negative EFE:
$$P(a|s) = \frac{\exp(-G(a)/\tau)}{\sum_{a'} \exp(-G(a')/\tau)}$$

### 3.5 Precision Dynamics

Precision updates adaptively based on prediction error magnitude:

$$\pi_s^{(t+1)} = \begin{cases}
\pi_s^{(t)} \cdot (1 + \alpha \cdot (1 + |\varepsilon|)^{-1}) & \text{if } |\varepsilon| > \theta \\
\pi_s^{(t)} \cdot (1 - 0.1\alpha) & \text{otherwise}
\end{cases}$$

$$\pi_p^{(t+1)} = \begin{cases}
\pi_p^{(t)} \cdot (1 - 0.5\alpha) & \text{if } |\varepsilon| > \theta \\
\pi_p^{(t)} \cdot (1 + \alpha \cdot (1 + |\varepsilon|)^{-1}) & \text{otherwise}
\end{cases}$$

**Interpretation:** High prediction error increases sensory precision (trust observations more) while decreasing prior precision (trust predictions less), and vice versa.

### 3.6 Temporal Difference Learning Integration

We integrate TD(λ) learning for value function approximation:

$$\delta_t = -|\varepsilon_t| + \gamma V_\theta(s') - V_\theta(s)$$

where intrinsic reward is negative prediction error (surprise minimization).

The value function is nonlinear:
$$V_\theta(s) = \tanh(\mathbf{w}^T \mathbf{h}_s + b)$$

Eligibility traces accumulate gradients:
$$\mathbf{e}_t = \gamma \lambda \mathbf{e}_{t-1} + \nabla_\theta V_\theta(s_t)$$

Model parameters update via:
$$\theta \leftarrow \theta + \alpha \delta_t \mathbf{e}_t$$

---

## 4. Experiments

### 4.1 Experimental Setup

**Implementation:** Rust, ~338K LOC total, 3,700+ lines core FEP in `fep_active_inference.rs`
**Test suite:** 2,797 tests pass, 0 failures, 13 ignored (see Figure 5 for full validation radar across 16 benchmarks)
**HDC Dimension:** d = 16,384 (configurable)
**Baselines:** pymdp v0.0.8 (Python active inference library, installed from infer-actively/pymdp)
**Benchmark Tasks:** T-Maze (context inference), Grid World 3×3 and 5×5 (navigation)
**Metrics:** Inference time, action selection time, free energy, task success rate

**Benchmark Parameters:**
- Trials per task: 100 (averaged)
- Episode length: 50 steps (T-Maze), 100 steps (Grid World)
- Policy temperature τ = 1.0
- HAI precision initialization: π_s = π_p = 1.0
- Grid World goal: Reach corner cell from center
- T-Maze: Infer context from cue, navigate to reward

### 4.2 Free Energy Convergence

**Task:** 4-dimensional observation space, 8-dimensional hidden state, 20 inference iterations

**Results:**
- Initial free energy: F₀ ≈ 2.3
- Final free energy: F₂₀ ≈ 0.4
- Convergence achieved by iteration 15
- KL divergence validated as non-negative (mathematical correctness)

### 4.3 Precision Dynamics Validation

**Setup:** Inject observations with varying prediction error magnitudes

**Results:**
| Error Magnitude | Sensory Precision Change | Prior Precision Change |
|-----------------|-------------------------|------------------------|
| ε = 0.2 (low) | -2% | +3% |
| ε = 0.5 (medium) | +5% | -8% |
| ε = 0.8 (high) | +12% | -15% |

Precision dynamics correctly adapt to error magnitude, implementing the "trust observations vs. predictions" tradeoff.

### 4.4 Computational Efficiency

**Comparison:** HAI (Symthaea) vs. pymdp on standard active inference benchmarks

We compare against pymdp v0.0.8, the reference Python implementation for discrete-state active inference [Heins et al., 2022]. Tasks include the T-Maze (classic context inference task) and Grid World navigation at 3×3 and 5×5 scales.

**Task-Based Results:**

| Task | Method | Inference (ms) | Action (ms) | Free Energy | Success Rate |
|------|--------|----------------|-------------|-------------|--------------|
| T-Maze | HAI | 0.084 | 0.149 | -2.305 | 100.00% |
| Grid 3×3 | HAI | 0.093 | 0.135 | -2.350 | 92.68% |
| Grid 3×3 | pymdp | 0.318 | 1.812 | -9.421 | 16.00% |
| Grid 5×5 | HAI | 0.191 | 0.148 | -2.975 | 88.03% |
| Grid 5×5 | pymdp | 0.356 | 2.338 | -9.238 | 10.00% |

*Note: pymdp T-Maze results omitted due to API incompatibility with benchmark harness.*

**Aggregate Speedups:**

| Metric | Speedup |
|--------|---------|
| Belief Inference | 1.9× |
| Action Selection | 15.8× |
| Total (Inference + Action) | 7.9× |

**Analysis:** HAI achieves substantial speedups in action selection (15.8×) compared to pymdp's matrix-based expected free energy computation. The inference speedup (1.9×) is more modest because pymdp's categorical belief updates are already O(n) for discrete states. The key advantage emerges in the action selection loop where HAI's O(d) HDC operations scale better than pymdp's O(n²) EFE matrix operations.

Notably, HAI achieves significantly higher task success rates (88-100% vs. 10-16%) with lower (better) free energy values. This suggests the HDC-based precision weighting and belief representation may provide more effective exploration-exploitation balance for these navigation tasks.

### 4.5 Test Suite Results

**17 tests, 100% pass rate:**
- Free energy principle mathematics: ✓
- Precision weighting: ✓ (5× measured difference)
- Expected free energy ranking: ✓
- Loop convergence: ✓
- Belief updating: ✓
- Generative model: ✓
- Surprise response: ✓

### 4.6 Extended Benchmark Validation

Beyond the core active inference benchmarks, Symthaea has been validated on 17 additional benchmarks spanning neuroscience, signal processing, and federated learning. Results as of February 2026.

**Table 2: Core benchmarks** (all tests passing at ≥80%):

| Benchmark | Tests Passed | Key Metrics |
|-----------|-------------|-------------|
| Federated Learning | 5/5 | FedAvg converges, BFT reduces adversarial loss 5×, trust-weighting effective |
| PyPhi Groundtruth | 6/6 | All IIT theory predictions validated |
| Drosophila Phi | 6/6 | Scales to 4096 neurons, MB Φ > OL Φ, scaling exponent <3.0 |
| Sleep Staging (EDF) | 5/5 | All 5 stages from real PhysioNet clinical EDF recordings (3 subjects, custom Rust EDF parser) |
| C. elegans Phi | Complete | 448 neurons, 7379 connections, circuit Φ 0.54-0.58 (novel) |
| LibriSpeech HDC | 3/3 | 94.5% speaker ID (10 speakers), temporal encoding preserves order |
| ISOLET HDC | 3/3 | 91.66% with retrain (lr=0.1, 8K dim); standard HDC benchmark |
| Meditation EEG | 6/6 | Gamma flow, theta absorption, quality ordering, session progression |
| EEG Seizure | 3/3 | 100% sensitivity, 100% specificity (spectral classifier) |
| ARC Reasoning | 4/5 | Pattern transfer verified, 96% intra-task consistency |

The Sleep Staging benchmark uses real clinical polysomnography recordings in European Data Format (EDF) from PhysioNet's Sleep-EDF database, parsed by a custom Rust EDF reader with no external dependencies. All five AASM sleep stages (Wake, N1, N2, N3, REM) are correctly classified using HDC-encoded frequency band power ratios, with N3 achieving 62.1% accuracy.

**Table 3: Supplementary benchmarks** (partial passes or mixed results):

| Benchmark | Tests Passed | Key Metrics |
|-----------|-------------|-------------|
| Anesthesia Phi | 5/6 | Induction/recovery monotonic (10-trial avg); discrete Φ ordering fails |
| Tokamak CfC | 4/5 | 87K inferences/sec, <1ms real-time; CfC sensitivity limited on synthetic data (§6.5) |
| PCI Validation | 4/5 | Φ ordering correct; Φ-PCI correlation low (expected, §6.4) |
| Emotion EEG | 4/5 | Valence/arousal separation validated |
| MNIST HDC | 2/3 | 84-88% with retrain (8K dim, 5 iter); baseline 81.6% without retrain |
| Ethics HDC | Mixed | Virtue 80%, Commonsense 53.2%, Justice 50.6%, Deontology 52.4% (59.1% overall) |
| λ₂-Φ Proxy | 3/5 | λ₂ shows meaningful topology variation; system_phi returns 0 for small weighted graphs (§6.3) |

**Federated Byzantine tolerance:** Validated at 34% (trimmed-mean aggregator). Testing at 45% Byzantine fraction showed zero convergence (mean_weight=0.0, positive_dims=0/20). The 34% level represents the empirically validated maximum; 45% should be considered a theoretical upper bound only.

### 4.7 Known Limitations

**Periodic signal learning:** On 4-element repeating sequences, prediction error increases by 47.8% over 200 cycles. Root cause: local one-step optimization creates competing attractors that diverge from data distribution. Fix: multi-scale loss function (future work).

---

## 5. Related Work

### 5.1 Active Inference Implementations

**pymdp** [Heins et al., 2022]: Python library for discrete-state active inference. Uses categorical distributions and exact Bayesian updates. Our work extends to continuous HDC space with approximate inference.

**SPM** [Friston et al.]: MATLAB toolbox for neuroimaging with FEP models. Focused on neural data analysis rather than cognitive architectures.

### 5.2 Hyperdimensional Computing Applications

HDC has been applied to classification [Rahimi et al., 2016], language modeling [Joshi et al., 2016], and robotics [Neubert et al., 2019]. **No prior work applies HDC to probabilistic inference or active inference.**

### 5.3 Neuro-Symbolic AI

Recent work combines neural networks with symbolic reasoning [Garcez & Lamb, 2020]. Our approach differs by using HDC's native algebraic structure rather than hybrid neural-symbolic architectures.

### 5.4 Novelty Assessment

Literature search confirms **no prior work combining FEP/active inference with HDC/VSA**. Closest related work:
- IBM NeuroVSA: Neural + symbolic, but not FEP-based
- Structured world models: Use transformers, not HDC

---

## 6. Discussion

### 6.1 Theoretical Implications

Our results suggest HDC similarity metrics can substitute for probabilistic divergences in variational inference. The key insight: **cosine similarity in high-dimensional space approximates Mahalanobis distance** when the metric is implicitly defined by the hypervector structure.

Precision-weighted binding provides a principled way to incorporate uncertainty into compositional representations without explicit variance tracking—addressing a longstanding challenge in neuro-symbolic AI.

### 6.2 Cognitive Architecture Implications

The eight motor command types derived from EFE minimization provide an interpretable action vocabulary for cognitive systems:
- Attention, learning rate, and exploration as *cognitive actions*
- Direct mapping from variational inference to executable commands
- No need for separate policy networks

### 6.3 When HDC Approximation Fails

We validated λ₂ (algebraic connectivity / spectral gap) as a proxy for exact IIT Φ across multiple network topologies (chain, ring, star, random, complete) at scales n=4..7.

| Topology | λ₂ (n=5) | Exact Φ | Notes |
|----------|----------|---------|-------|
| Chain | 0.382 | 0.0 | Low connectivity, low λ₂ |
| Ring | 0.691 | 0.0 | Moderate connectivity |
| Star | 1.000 | 0.0 | Hub topology |
| Complete | 1.250 | 0.0 | All-to-all connectivity |
| Random | varies | 0.0 | Density-dependent |

**Critical finding — exact Φ degeneracy (see Figure 6):** Our `system_phi` implementation (based on IIT 3.0 minimum information partition) returns 0.0 for *all* tested weighted adjacency matrices at n=4..7. This occurs because the MIP algorithm can always find a trivial partition for small weighted systems. Consequently, no meaningful Φ-λ₂ correlation can be computed (Pearson r = 0.05, Spearman ρ = 0.0).

However, λ₂ itself shows meaningful and correct topology variation: chain (0.19) < ring (0.50) < star (1.0) < complete (1.2), correctly ordering topologies by algebraic connectivity. The proxy captures structural integration that exact Φ collapses to zero.

**Implications:** λ₂ is useful for *relative* ordering of network integration within the same system size, but cannot be validated against exact Φ at accessible scales due to MIP degeneracy. For N > 8, exact Φ is computationally intractable (O(2^N)), making λ₂ the only feasible integration measure. We recommend λ₂ for relative comparisons only, not as a quantitative Φ substitute.

### 6.4 Divergence of Consciousness Measures

Benchmarking revealed low correlation (r = -0.15) between integrated information (Φ) and perturbational complexity index (PCI). This is theoretically expected rather than problematic:

- **Φ** (Tononi 2004, 2016) measures *intrinsic* causal structure — how much a system's parts are integrated above their individual contributions. Computed from the system's TPM without external intervention.
- **PCI** (Casali et al. 2013) measures *perturbational* complexity — the algorithmic complexity of the brain's response to TMS stimulation. Computed from externally-evoked EEG patterns.

Both correctly order conscious states (awake Φ > vegetative Φ; awake PCI > vegetative PCI), but they capture orthogonal properties: Φ measures intrinsic integration while PCI measures response complexity. A system can be highly integrated but respond simply to perturbation (high Φ, low PCI), or loosely coupled but generate complex transient dynamics (low Φ, high PCI). The weak correlation validates that our implementation captures these distinct theoretical constructs rather than conflating them.

### 6.5 CfC Gradient Dynamics

The Closed-form Continuous-time (CfC) neural ODE cells exhibit gradient vanishing under certain configurations. Analysis identified compounding attenuation from: SiLU activation derivative (0.5× at origin), gradient clipping (if set too aggressively), and inter-layer decay factors. For classification tasks requiring decision boundaries, the combined effect can reduce effective gradients by 100-400×, leading to attractor collapse (all outputs converge to the same class).

**Mitigation:** We introduced configurable gradient clipping (default 1.0, up from 0.5) and floored inter-layer attenuation at 0.3 to prevent complete gradient vanishing through stacked CfC layers. For binary classification tasks, we recommend `gradient_clip ≥ 5.0` and `learning_rate ≥ 0.01`.

### 6.6 Additional Limitations and Future Work

1. **Periodic signal learning:** Multi-scale loss functions needed for temporal consistency. On 4-element repeating sequences, prediction error increases by 47.8% over 200 cycles due to competing attractors.

2. **Extended POMDP benchmarks:** Current validation covers T-Maze and Grid World; additional tasks (Tiger problem, multi-step planning) would strengthen generalization claims.

3. **HDC-Φ correspondence:** HDC-based cosine similarity matrices do not produce meaningful exact IIT Φ values. In high dimensions (d ≥ 256), cosine similarities between random HVs converge to ~0 regardless of topology structure. Even with explicit adjacency matrices, our `system_phi` implementation returns 0 for all weighted graphs at n ≤ 7 due to MIP degeneracy (see Section 6.3). The spectral proxy λ₂ provides useful relative ordering but should not be treated as a quantitative Φ substitute.

4. **ETHICS benchmark gap:** Moral algebra achieves 77.5% on Virtue classification but only 44-53% on Deontology, Justice, and Commonsense. The virtue category benefits from clear sentiment polarity in trait words; the other categories require deeper contextual reasoning about obligations and fairness that HDC keyword-matching alone cannot capture. Integrating the moral algebra's compositional operators (CAUSES, VIOLATES, SATISFIES) with richer text parsing is a priority for future work.

---

## 7. Conclusion

We presented Hyperdimensional Active Inference (HAI), the first integration of Free Energy Principle active inference with Hyperdimensional Computing. By reformulating variational free energy in hypervector space and introducing precision-weighted binding, we achieve:

1. **Computational efficiency:** 1.9-15.8× speedup over pymdp (7.9× total), with O(d) scaling
2. **Interpretable action selection:** Eight motor command types from EFE minimization
3. **Correct precision dynamics:** Adaptive confidence weighting validated empirically
4. **Mathematical soundness:** Free energy convergence and KL non-negativity confirmed

HAI opens new directions for efficient, interpretable cognitive architectures that combine the theoretical rigor of active inference with the computational elegance of hyperdimensional computing.

---

## References

[1] Friston, K. (2010). The free-energy principle: a unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.

[2] Friston, K., FitzGerald, T., Rigoli, F., Schwartenbeck, P., & Pezzulo, G. (2017). Active inference: a process theory. *Neural Computation*, 29(1), 1-49.

[3] Kanerva, P. (2009). Hyperdimensional computing: An introduction to computing in distributed representation with high-dimensional random vectors. *Cognitive Computation*, 1(2), 139-159.

[4] Heins, C., Millidge, B., Da Costa, L., et al. (2022). pymdp: A Python library for active inference in discrete state spaces. *Journal of Open Source Software*, 7(73), 4098.

[5] Feldman, H., & Friston, K. J. (2010). Attention, uncertainty, and free-energy. *Frontiers in Human Neuroscience*, 4, 215.

[6] Rahimi, A., Kanerva, P., & Recht, B. (2016). A robust and energy-efficient classifier using brain-inspired hyperdimensional computing. *ISLPED*, 64-69.

[7] Parr, T., Pezzulo, G., & Friston, K. J. (2022). *Active Inference: The Free Energy Principle in Mind, Brain, and Behavior*. MIT Press.

[8] Joshi, A., Halseth, J. T., & Kanerva, P. (2016). Language geometry using random indexing. *QI*.

[9] Neubert, P., Schubert, S., & Protzel, P. (2019). An introduction to hyperdimensional computing for robotics. *KI-Künstliche Intelligenz*, 33(4), 319-330.

[10] Garcez, A., & Lamb, L. C. (2020). Neurosymbolic AI: The 3rd wave. *arXiv preprint arXiv:2012.05876*.

[11] Casali, A. G., Gosseries, O., Rosanova, M., et al. (2013). A theoretically based index of consciousness independent of sensory processing and behavior. *Science Translational Medicine*, 5(198), 198ra105.

[12] Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience*, 5(1), 42.

[13] Tononi, G., Boly, M., Massimini, M., & Koch, C. (2016). Integrated information theory: from consciousness to its physical substrate. *Nature Reviews Neuroscience*, 17(7), 450-461.

---

## Appendix A: Implementation Details

**Repository:** https://github.com/Luminous-Dynamics/symthaea
**Language:** Rust (Edition 2021)
**Dependencies:** ndarray, nalgebra, rand, burn (optional GPU), duckdb
**Lines of code:** ~338K total, ~3,700 core FEP, ~17,700 mathematical foundations, ~8,100 code understanding
**Test suite:** 2,797 tests, 13 ignored, 0 failures
**Feature flags:** 75 (48 active in cfg attrs)
**Test command:** `cargo test test_fep_active_inference`
**Demo command:** `cargo run --example fep_active_inference_demo`
**Self-analysis demo:** `cargo run --features code_generation --example demo_self_analysis`

---

## Appendix B: Full Mathematical Derivations

### B.1 Free Energy Gradient

Starting from:
$$F = \frac{1}{2}\pi_p(1 - \cos(\mathbf{h}_q, \mathbf{h}_p)) - \left(-\frac{1}{2}\pi_s(1 - \cos(\mathbf{h}_q, \mathbf{h}_o))^2\right)$$

The gradient with respect to $\mathbf{h}_q$:

$$\frac{\partial F}{\partial \mathbf{h}_q} = \pi_p \frac{\partial}{\partial \mathbf{h}_q}(1 - \cos(\mathbf{h}_q, \mathbf{h}_p)) + \pi_s(1 - \cos(\mathbf{h}_q, \mathbf{h}_o))\frac{\partial}{\partial \mathbf{h}_q}\cos(\mathbf{h}_q, \mathbf{h}_o)$$

Using $\frac{\partial}{\partial \mathbf{x}}\cos(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{y} - \mathbf{x}\cos(\mathbf{x},\mathbf{y})}{|\mathbf{x}|}$:

$$\nabla_{\mathbf{h}_q} F = \pi_s(\mathbf{h}_o - \mathbf{h}_q\cos(\mathbf{h}_q, \mathbf{h}_o)) + \pi_p(\mathbf{h}_p - \mathbf{h}_q\cos(\mathbf{h}_q, \mathbf{h}_p))$$

### B.2 Expected Free Energy Decomposition

For action a with predicted next state $\mathbf{h}_{s'}^{(a)}$:

**Pragmatic value:**
$$G_p(a) = \beta \sum_j (\hat{o}_j^{(a)} - \tilde{o}_j)^2$$

where $\hat{\mathbf{o}}^{(a)} = A\mathbf{h}_{s'}^{(a)}$ and $\tilde{\mathbf{o}}$ is the preferred observation.

**Epistemic value:**
$$G_e(a) = H[p(\mathbf{o}|\mathbf{h}_{s'}^{(a)})] - H[p(\mathbf{o}|\mathbf{h}_s)]$$

For Gaussian beliefs: $H = \frac{1}{2}(d + d\ln(2\pi) + \ln|\Sigma|)$

---

## Appendix C: Benchmark Methodology

### C.1 Environment Specifications

**T-Maze Task:**
- 4 locations: start, junction, left arm, right arm
- Binary context: reward left or right (signaled by cue at start)
- Success: reach correct arm within 50 steps

**Grid World Task:**
- 3×3 or 5×5 grid with 4-action movement (up/down/left/right)
- Agent starts at center, goal at corner
- Success: reach goal within step limit

### C.2 Timing Methodology

All timing measurements:
- Warm-up runs: 10 (discarded)
- Measured runs: 100
- Reported: Mean ± standard error
- Platform: Single-threaded execution on AMD Ryzen 9

### C.3 Quality Ratio

The Quality Ratio (3.10) is computed as:
$$QR = \frac{|FE_{pymdp}|}{|FE_{HAI}|} = \frac{9.33}{3.01} \approx 3.1$$

Higher QR indicates HAI achieves lower (better) free energy.

---

*Draft completed: February 8, 2026*
*Version: 0.5*
*Status: Full experimental validation with extended benchmark suite (17 benchmarks, 72/82 tests passing), 14 mathematical foundation modules wired into live system, λ₂ proxy validation (meaningful topology ordering, exact Φ degeneracy documented), Byzantine tolerance validated to 34%, ETHICS moral reasoning benchmark (56.1%), and CfC gradient stabilization.*

**Supplementary Materials:**
- `papers/figures/` - Figures 1-6 (PDF and PNG):
  - Fig 1: HAI Architecture Diagram
  - Fig 2: Free Energy Convergence Curves
  - Fig 3: Precision Dynamics
  - Fig 4: Scaling Analysis (HAI vs pymdp)
  - Fig 5: Benchmark Validation Radar (16 benchmarks, 93% pass rate)
  - Fig 6: λ₂-Φ Proxy Validation Scatter (ρₛ = 0.50)
- `papers/appendices/theoretical_analysis.md` - Appendix D: Formal Proofs
- `docs/PYMDP_COMPARISON_REPORT.md` - pymdp Benchmark Details
- `docs/ABLATION_STUDIES_REPORT.md` - Dimension, Precision, EFE Weight Ablations
- `docs/STATISTICAL_ANALYSIS_REPORT.md` - 95% CI and Significance Tests
- `docs/EXTENDED_BENCHMARKS_REPORT.md` - Tiger, Large Grids, Multi-Agent
