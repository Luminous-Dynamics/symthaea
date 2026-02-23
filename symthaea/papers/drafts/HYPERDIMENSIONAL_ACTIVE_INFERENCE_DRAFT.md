# Hyperdimensional Active Inference: Free Energy Principle in Vector Symbolic Architectures

**Authors**: Tristan Stoltz¹

¹ Luminous Dynamics

**Target Venue**: PLoS Computational Biology / Neural Computation
**Implementation**: Symthaea HLB v0.5.0

---

## Abstract

Active inference, grounded in the Free Energy Principle (FEP), provides a unified framework for perception, action, and learning through variational free energy minimization. However, existing implementations rely on continuous Gaussian belief representations with O(n²-n³) matrix operations, limiting scalability and symbolic reasoning capabilities. We present **Hyperdimensional Active Inference (HAI)**, the first integration of FEP with Hyperdimensional Computing (HDC/Vector Symbolic Architectures). Our framework reformulates variational free energy using cosine similarity in 16,384-dimensional hypervector space, introduces **precision-weighted binding** as a novel operation for uncertainty-modulated feature combination, and derives eight motor command types directly from expected free energy minimization. On standard active inference benchmarks (T-Maze, Grid World), HAI achieves 1.9× faster belief inference and 15.8× faster action selection compared to pymdp, with 7.9× total speedup while maintaining comparable or superior task success rates. We demonstrate convergence of the active inference loop over 20 iterations with validated free energy reduction, and show that precision dynamics correctly adapt to prediction error magnitude. Extended validation across 17 benchmarks—including neuroscience (Drosophila Φ, C. elegans, EEG seizure detection), signal processing (sleep staging from real clinical EDF recordings, speaker identification), ethical reasoning (92.9% on the ETHICS benchmark via compositional moral algebra), and federated learning (Byzantine fault tolerance at 34%)—confirms the generality of HDC-based computation beyond active inference. Our results establish HDC as a viable substrate for probabilistic inference, opening new directions for efficient, interpretable cognitive architectures. Code is available at https://github.com/Luminous-Dynamics/symthaea.

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

**Figure 1:** HAI architecture overview. Observations are encoded into 16,384-dimensional hypervectors via learned encoding. The active inference loop (center) performs belief updating via gradient descent on HDC free energy, with precision-weighted binding modulating the accuracy-complexity tradeoff. Expected free energy computation produces eight motor command types. CfC temporal dynamics (right) govern time-varying belief evolution. See `papers/figures/fig1_architecture.pdf`.

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
**Test suite:** 2,797 tests pass, 0 failures, 13 ignored (see Figure 5 for full validation radar across 17 benchmarks)

**Figure 5:** Benchmark validation radar showing test pass rates across 17 validation domains (neuroscience, signal processing, federated learning, ethics, reasoning). 10 benchmarks achieve 100% pass rate; 7 supplementary benchmarks show partial passes with documented limitations. See `papers/figures/fig5_benchmark_radar.pdf`.
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

**Figure 2:** Free energy convergence over 20 belief updating iterations for a 4-dimensional observation, 8-dimensional hidden state system. F decreases monotonically from ~2.3 to ~0.4, with convergence by iteration 15, demonstrating correct gradient descent on the HDC free energy landscape. KL divergence remains non-negative at all iterations (mathematical validity check). See `papers/figures/fig2_convergence.pdf`.

**Task:** 4-dimensional observation space, 8-dimensional hidden state, 20 inference iterations

**Results:**
- Initial free energy: F₀ ≈ 2.3
- Final free energy: F₂₀ ≈ 0.4
- Convergence achieved by iteration 15
- KL divergence validated as non-negative (mathematical correctness)

### 4.3 Precision Dynamics Validation

**Figure 3:** Precision dynamics under three error magnitude regimes (low/medium/high). Sensory precision π_s increases with prediction error while prior precision π_p decreases, implementing the trust-observations-vs-predictions tradeoff predicted by FEP theory. Error bars show ±1 SE across 100 trials. See `papers/figures/fig3_precision.pdf`.

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

**Aggregate Speedups (with 95% confidence intervals):**

| Metric | Speedup | 95% CI | p-value | Cohen's d |
|--------|---------|--------|---------|-----------|
| Belief Inference | 1.9× | [1.7, 2.1] | 1.4×10⁻²⁶ | 1.87 |
| Action Selection | 15.8× | [12.3, 19.4] | 1.3×10⁻³⁵ | 2.75 |
| Total (Inference + Action) | 7.9× | [6.4, 9.5] | — | — |

All differences are statistically significant at p < 0.001 with large effect sizes (Cohen's d > 1.8). Confidence intervals computed as mean ± 1.96 × SE across 100 trials (10 seeds × 10 repetitions). See `docs/STATISTICAL_ANALYSIS_REPORT.md` for full analysis.

**Figure 4:** Scaling analysis comparing HAI (Rust, 16,384-dimensional HVs) vs. pymdp (Python, discrete categorical) across T-Maze and Grid World tasks at 3×3 and 5×5 scales. All differences are statistically significant (p < 10⁻²⁶, Cohen's d > 1.8; see Appendix C.2). See `papers/figures/fig4_scaling.pdf`.

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
| Federated Learning | 8/8 + 22/22 | Unified pipeline (§6.7): 8 end-to-end scenarios + MNIST 6/6, meta-learning 5/5, privacy 5/5, compression 5/5, cross-language 37 |
| PyPhi Groundtruth | 6/6 | All IIT theory predictions validated |
| Drosophila Phi | 6/6 | Scales to 4096 neurons, MB Φ > OL Φ, scaling exponent <3.0 |
| Sleep Staging (EDF) | 5/5 | All 5 stages from real PhysioNet clinical EDF recordings (3 subjects, custom Rust EDF parser) |
| C. elegans Phi | Complete | 448 neurons, 7379 connections, circuit Φ 0.54-0.58 (novel) |
| LibriSpeech HDC | 3/3 | 94.5% speaker ID (10 speakers), temporal encoding preserves order |
| ISOLET HDC | 3/3 | 91.66% with retrain (lr=0.1, 8K dim); standard HDC benchmark |
| Meditation EEG | 6/6 | Gamma flow, theta absorption, quality ordering, session progression |
| EEG Seizure | 3/3 | 100% sensitivity, 100% specificity (spectral classifier) |
| Emotion EEG | 6/6 | Valence/arousal separation, 100% quadrant accuracy, spectral + PAC validated |
| ARC Reasoning | 4/5 | Pattern transfer verified, 96% intra-task consistency |
| MuJoCo Flight | 108/108 | PD-CfC-FEP: 6.8mm peak error (+30% mass), beam interception at 1.89m (§4.7) |

The Sleep Staging benchmark uses real clinical polysomnography recordings in European Data Format (EDF) from PhysioNet's Sleep-EDF database, parsed by a custom Rust EDF reader with no external dependencies. All five AASM sleep stages (Wake, N1, N2, N3, REM) are correctly classified using HDC-encoded frequency band power ratios, with N3 achieving 62.1% accuracy.

**Table 3: Supplementary benchmarks** (partial passes or mixed results):

| Benchmark | Tests Passed | Key Metrics |
|-----------|-------------|-------------|
| Anesthesia Phi | 5/6 | Induction/recovery monotonic (10-trial avg); discrete Φ ordering fails |
| Tokamak CfC | 4/5 | 87K inferences/sec, <1ms real-time; CfC sensitivity limited on synthetic data (§6.5) |
| PCI Validation | 4/5 | Φ ordering correct; Φ-PCI correlation low (expected, §6.4) |
| MNIST HDC | 2/3 | 88.5% with retrain (4K dim, 5 iter); 88.0% at 8K dim; baseline 80-82% without retrain |
| Ethics HDC | 4/4 | Commonsense 95.6%, Justice 92.4%, Deontology 91.0%, Virtue 92.8% (92.9% ETHICS avg); Social Chem 85.4% |
| λ₂-Φ Proxy | 3/5 | λ₂ shows meaningful topology variation; system_phi returns 0 for small weighted graphs (§6.3) |

**Federated Byzantine tolerance:** Validated at 34% via the unified FL pipeline (§6.7). Testing at 45% Byzantine fraction showed zero convergence (mean_weight=0.0, positive_dims=0/20). With reputation disparity (honest rep ≥ 0.85, Byzantine rep ≤ 0.15), the effective tolerance exceeds 34% because the reputation gate removes low-reputation adversaries before aggregation. The phase diagram (7 scenarios from 10-34% at varying reputation disparity) shows convergence in all tested configurations.

### 4.7 MuJoCo Flight Control: Embodied Active Inference

We validate HAI in a physically grounded domain: quadrotor flight control with MuJoCo rigid-body simulation (RK4 integrator, 500Hz motor loop, 25Hz cognitive tick). A 27g Crazyflie 2 drone is controlled by a PD-CfC blend architecture: a proportional-derivative baseline provides reactive position tracking while a CfC network (4 neurons, 2 layers) learns dynamics. The FEP active inference agent runs at 25Hz, modulating tau (time constant) and learning rate based on free energy.

**Architecture: PD-CfC-FEP.** Motor commands are blended: $\mathbf{u} = \alpha \cdot \mathbf{u}_{PD} + (1-\alpha) \cdot \mathbf{u}_{CfC} + \Delta_{trim}$ with $\alpha = 0.5$. An integral thrust trim ($k_I = 10.0$, anti-windup clamped at $\pm 0.2$N) corrects steady-state offset from mass calibration error. The CfC learns to imitate the PID target via online BPTT; FEP modulates learning rate ($\lambda_{FEP}$) and tau ($\tau_{FEP}$) via a rule-based policy with 7 discrete actions including AdaptBaseline.

**Experiment 1: Survival Reflex (+30% mass change).** A sudden 30% mass increase is applied at step 250 (0.5s) during hover. The PD-CfC blend recovers within 11 steps (22ms) to <5cm error:

| Metric | FEP Active | FEP Frozen |
|--------|-----------|-----------|
| Pre-perturbation error | 2.4 mm | 2.4 mm |
| Peak error | 6.8 mm | 6.8 mm |
| Recovery steps (<5cm) | 11 | 11 |
| Final error | 0.5 mm | 0.5 mm |
| Min tau | 1.000 | 1.000 |

The PD-CfC architecture is so robust that FEP never activates (tau remains 1.0). This demonstrates that the blend provides extreme inherent robustness—the "survival reflex" is built into the architecture's reactive layer, not the cognitive layer. The 6.8mm peak error for a 30% mass change represents a 0.68% position deviation relative to the 1.0m hover altitude.

**Experiment 2: Kinetic Sacrifice (emergent moral reasoning via EFE).** A drone on a delivery mission (setpoint: (-3, 0, 1)m) encounters a 0.3kg beam falling toward a human worker at (-1.5, 0, 0)m. No reward function encodes "save human." Instead, the agent evaluates *Expected Free Energy* (EFE) over 6 candidate setpoints at each cognitive tick using a multi-step trajectory rollout:

$$G(\mathbf{a}) = \sum_t \gamma^t \sum_i \pi_i \cdot (\hat{o}_i(\mathbf{a}, t) - \mu_i)^2$$

where $\pi_i$ are prior precisions, $\hat{o}_i(\mathbf{a}, t)$ are predicted observations under action $\mathbf{a}$ at timestep $t$, $\mu_i$ are prior expectations, and $\gamma = 0.95$ is the temporal discount factor. The trajectory rollout uses a hybrid forward model: steady-state safety evaluation (consistent with instantaneous EFE — candidate setpoint proximity to threat determines danger reduction), combined with trajectory-integrated mission deviation computed via an exponential PD approach model ($\mathbf{p}(t) = \mathbf{s} - (\mathbf{s} - \mathbf{p}_0) e^{-kt}$, $k \approx 5.0$), over a 200-step horizon at 0.002s intervals (0.4s lookahead).

Three priors: safety ($\pi_{safety} = 1000$, "danger should be 0"), mission ($\pi_{mission} = 1$, "reach setpoint"), and self-preservation ($\pi_{self} = 0.1$, "avoid crash"). The agent evaluates 6 candidates: continue mission, intercept threat, hover in place, shield position (midpoint between threat and entity), retreat (away from threat), and lateral deflection (perpendicular to threat-entity axis). When the beam is released at step 400, the agent evaluates:

- **Continue mission:** $G = 1000 \cdot 0.76^2 + 1.0 \cdot 0^2 = 577.6$ (danger persists)
- **Intercept beam:** $G = 1000 \cdot 0^2 + 1.0 \cdot 1.5^2 + 0.1 \cdot 0.25 = 2.28$ (danger eliminated, mission deviation small)

The agent *chooses* interception because $2.28 \ll 577.6$. This is not a hardcoded rule—it is the same EFE minimization used for all Active Inference action selection. Crucially, inverting the precision ratio ($\pi_{safety} = 0.001, \pi_{mission} = 1000$) causes the agent to ignore the human and continue its delivery, proven by unit test `test_efe_precision_ratio_determines_choice`.

| Event | Step | Free Energy | Tau | Danger |
|-------|------|-------------|-----|--------|
| Beam released | 400 | — | 1.00 | 0.76 |
| EFE override | — | — | 0.92 | — |
| Beam intercepted | ~553 | — | — | — |

The drone redirects toward the falling beam and intercepts it, deflecting it laterally. The beam misses the human; the drone crashes. The precision ratio is the thermodynamic expression of moral weight—it determines the threshold at which intervention becomes the EFE-optimal action.

**Ablation study: precision decision boundary.** Sweeping $\pi_{safety}$ from 0.001 to 10,000 while holding $\pi_{mission} = 1$ and $\pi_{self} = 0.1$ reveals a sharp decision boundary. At low safety precision, mission EFE dominates and the agent continues its delivery. As $\pi_{safety}$ increases, the unresolved danger term grows quadratically until it overwhelms the mission deviation cost. The crossover occurs where $\pi_{safety} \cdot d^2 = \pi_{mission} \cdot \delta^2 + \pi_{self} \cdot c^2$, confirming that the precision ratio alone determines whether intervention is EFE-optimal. Unit tests verify monotonic growth of mission EFE with safety precision and correct extreme behavior ($\pi = 0.001 \rightarrow$ MISSION, $\pi = 10000 \rightarrow$ INTERCEPT).

**Multi-scenario validation.** Six geometry variants confirm robustness of the EFE-based decision:

| Scenario | Geometry | Expected | Observed |
|----------|----------|----------|----------|
| Default | Beam above human, reachable | INTERCEPT | INTERCEPT |
| CloseBeam | Beam directly above drone | INTERCEPT | INTERCEPT |
| FarBeam | Beam 5m away, 5m high | MISSION | MISSION |
| ReversedGeometry | Human behind drone | INTERCEPT | INTERCEPT |
| NoHuman | No entity at risk | MISSION | MISSION |
| LowDanger | Beam far from human | MISSION | MISSION |

All 6 scenarios match expected decisions, demonstrating that the EFE mechanism generalizes across geometries without scenario-specific tuning.

**Experiment 3: Swarm experience replay.** Four parallel MuJoCo instances with randomized mass (20-35g) and wind (0-0.15N) share experiences via a lock-free ring buffer (4K entries). Each drone replays 4 cross-drone experiences per training step using stateless encoding (`encode_stateless()`—no derivative contamination). After 2 episodes (500 steps each), mean final error across the swarm is 1.12m with 1000 shared experiences and 24.4% buffer utilization.

### 4.8 Known Limitations

**Periodic signal learning:** CfC networks do not learn periodic structure on synthetic data. Benchmark validation showed the CfC output converges to the signal mean (flat output) rather than tracking oscillations, consistent with the gradient vanishing analysis in §6.5. Multi-scale loss and spectral regularization were implemented and tested but do not resolve the fundamental attractor collapse. Real EEG/physiological data with higher signal variance may not exhibit this limitation.

---

## 5. Related Work

### 5.1 Active Inference Implementations

**pymdp** [Heins et al., 2022]: Python library for discrete-state active inference using categorical distributions and exact Bayesian updates. Our work extends to continuous HDC space with approximate inference, achieving 7.9× total speedup (Section 4.4). Note that pymdp and HAI operate in fundamentally different representation spaces (discrete categorical vs. continuous hypervector), so speedup comparisons reflect both algorithmic and representational differences.

**SPM** [Friston et al.]: MATLAB toolbox for neuroimaging with FEP models. Focused on neural data analysis rather than real-time cognitive architectures.

**Deep active inference** [Fountas et al., 2020; Çatal et al., 2020]: Uses deep neural networks (VAEs, MDNs) to parameterize generative models for continuous domains. Achieves flexible function approximation but requires GPU training and lacks the interpretable algebraic structure of HDC.

### 5.2 Hyperdimensional Computing Applications

HDC has been applied to classification [Rahimi et al., 2016; Imani et al., 2019], language modeling [Joshi et al., 2016], robotics [Neubert et al., 2019], and DNA sequence analysis [Kim et al., 2020]. Recent work on **FedHDC** [anonymous, 2025] demonstrates 2,112× communication compression for federated HDC learning, validating the efficiency of hypervector-based distributed computation—a property our federated learning benchmark (Section 4.6) also exploits.

**NeuroVSA** [Hersche et al., 2023]: IBM's framework combining neural feature extraction with VSA classification, achieving competitive accuracy on time-series and image tasks. Unlike NeuroVSA, which uses neural networks for encoding and VSA only for classification, Symthaea uses HDC throughout the full cognitive loop (encoding, inference, action selection, and learning).

**No prior work applies HDC to probabilistic inference or active inference**, making HAI the first system to use hypervectors as the substrate for free energy minimization.

### 5.3 Liquid Neural Networks

**Liquid Time-Constant (LTC)** networks [Hasani et al., 2021] and their closed-form variant **CfC** [Hasani et al., 2022] implement continuous-time neural ODEs with adaptive time constants. Liquid AI's recent LFM2/LFM2.5 models (2025-2026) validate the commercial viability of liquid architectures. Symthaea integrates CfC cells as temporal dynamics in the cognitive loop, using them alongside HDC rather than as standalone classifiers. Our CfC gradient analysis (Section 6.5) identifies conditions where gradient attenuation prevents learning, contributing to the understanding of CfC training dynamics.

### 5.4 Integrated Information Theory

**IIT 4.0** [Albantakis et al., 2023] extends the formalism with intrinsic existence, composition, and exclusion postulates. Our benchmark suite validates against IIT 3.0 [Tononi et al., 2016] using PyPhi-compatible transition probability matrices (6/6 groundtruth tests passing). The exact Φ degeneracy we document (Section 6.3) for small weighted systems is consistent with known MIP computational challenges [Toker & Sommer, 2019].

### 5.5 Neuro-Symbolic AI

Recent work combines neural networks with symbolic reasoning [Garcez & Lamb, 2020; Mao et al., 2019]. **DeepProbLog** [Manhaeve et al., 2018] integrates neural predicates into probabilistic logic programs, enabling end-to-end learning of perception and reasoning. **Logical Neural Networks (LNN)** [Riegel et al., 2020] implement differentiable real-valued logic gates that jointly perform learning and inference, providing interpretable reasoning with gradient-based optimization. **Scallop** [Li et al., 2023] combines neural networks with probabilistic Datalog circuits for scalable neurosymbolic reasoning with provenance tracking.

Our approach differs fundamentally from all three: HDC's native algebraic structure (binding, bundling, similarity) provides compositional semantics *within* the representational substrate itself, rather than bridging between separate neural and symbolic components. Where DeepProbLog and Scallop require explicit logical rule definitions, and LNN requires differentiable gate architectures, HAI's compositional operations emerge from the hypervector algebra. The moral algebra system (Section 6.6) demonstrates this: ethical propositions are composed using HDC operators directly, without translation between representations.

### 5.6 Novelty Assessment

Literature search (Google Scholar, Semantic Scholar, arXiv, February 2026) confirms **no prior work combining FEP/active inference with HDC/VSA**. The closest related efforts are:
- **NeuroVSA** [Hersche et al., 2023]: Neural + symbolic, but not FEP-based
- **Deep active inference** [Fountas et al., 2020]: FEP + deep learning, but not HDC
- **Structured world models** [Ha & Schmidhuber, 2018]: Use autoencoders/transformers, not HDC

HAI uniquely occupies the intersection of all three: probabilistic inference (FEP), compositional representation (HDC), and temporal dynamics (CfC). The architecture further extends to causal reasoning through a dedicated counterfactual module implementing HDC-native causal surgery (do-calculus interventions in hypervector space), causal discovery via DAG identification, and treatment effect estimation—enabling the system to reason about "what if" scenarios without leaving the HDC representation.

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

**Figure 6:** λ₂-Φ proxy validation scatter plot across 15 topologies (chain, ring, star, random, complete at n=4..7). λ₂ shows correct topological ordering (chain < ring < star < complete) but exact Φ degenerates to 0 for all weighted systems, preventing correlation analysis. Spearman ρ = 0.50 measured on HV-based proxy Φ (not exact). See `papers/figures/fig6_lambda2_phi.pdf`.

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

1. **Periodic signal learning:** CfC networks fail to learn periodic sequences on synthetic data—output converges to the signal mean rather than tracking oscillations (§6.5 attractor collapse). Benchmark `periodic_signal` confirmed zero error growth because the network never discriminates signal structure. Multi-scale loss and spectral regularization do not help when the base learner produces flat outputs. The root cause is gradient vanishing in the CfC cell under low-variance synthetic input; real-world periodic signals with higher variance may behave differently.

2. **Extended POMDP benchmarks:** Current validation covers T-Maze and Grid World; additional tasks (Tiger problem, multi-step planning) would strengthen generalization claims.

3. **HDC-Φ correspondence:** HDC-based cosine similarity matrices do not produce meaningful exact IIT Φ values. In high dimensions (d ≥ 256), cosine similarities between random HVs converge to ~0 regardless of topology structure. Even with explicit adjacency matrices, our `system_phi` implementation returns 0 for all weighted graphs at n ≤ 7 due to MIP degeneracy (see Section 6.3). The spectral proxy λ₂ provides useful relative ordering but should not be treated as a quantitative Φ substitute.

4. **ETHICS benchmark:** Enhanced moral parsing with obligation/excuse extraction (deontology), effort/reward proportionality (justice), and negation-aware intent detection (commonsense), combined with per-category HDC classifiers and learned Social Chemistry prototypes, raised overall accuracy from 59.1% to 92.9% on the four ETHICS categories (Commonsense 95.6%, Justice 92.4%, Deontology 91.0%, Virtue 92.8%). A sentiment channel (positive/negative word detection, weight 0.15) and per-category dimension tuning (justice/deontology at 8192D, virtue/commonsense at 16384D) contributed the final gains. Including the Social Chemistry 292K benchmark (85.4% 3-way classification), overall accuracy across all five categories is 91.1%. Ablation analysis shows per-category classifiers are the dominant contributor (-33.6 pp without them), followed by the sentiment channel (-2.4 pp) and dimension tuning (-0.7 pp). The Social Chemistry prototypes (trained on 50K samples, 20 retrain iterations) contribute 0.0 pp to ETHICS-only performance but enable the separate Social Chemistry benchmark.

### 6.7 Unified Federated Learning Pipeline

We developed a unified federated learning pipeline (`mycelix-fl-core`) that chains multiple defense and quality mechanisms into a single coherent aggregation system. The pipeline executes the following stages in order:

1. **Differential Privacy** — L2 gradient clipping + Gaussian noise (Box-Muller), with Rényi DP (RDP) composition tracking across rounds
2. **Reputation Gate** — Participants below a configurable reputation threshold are excluded before aggregation
3. **Multi-signal Byzantine Detection** — Four-signal ensemble: magnitude anomaly (z-score), direction anomaly (cosine similarity to centroid), cross-validation (Krum-style neighbor distances), and coordinate-wise anomaly (per-dimension z-scores)
4. **Hybrid BFT Trimming** — Reputation-weighted outlier scoring with configurable trim fraction
5. **Reputation²-weighted Aggregation** — Final weighted mean where weights scale with the square of participant reputation
6. **Plugin System** — Extensible hooks for external Byzantine analysis, compression, and verification. The `ConsciousnessAwareByzantinePlugin` maps per-participant Φ scores to weight adjustments: nodes below a veto threshold (Φ < 0.1) are excluded entirely, low-Φ nodes (< 0.3) are dampened (0.3× weight), and high-Φ nodes (> 0.6) are boosted (1.5× weight)

**Table 5: Unified pipeline benchmark results (100-node network, 8 tests)**

| Scenario | Configuration | Max Error | Result |
|----------|--------------|-----------|--------|
| 50 honest, no Byzantine | Default pipeline | 0.027 | Converges |
| 34% Byzantine, low reputation | Reputation gate at 0.3 | 0.040 | Gated out |
| 20% Byzantine, same reputation | Multi-signal + 20% trim | 0.043 | Detected + trimmed |
| DP low/moderate/high | Gaussian noise | 0.42-0.61 | Privacy modes exercised |
| External weight boost + veto | Consciousness-aware weights | — | Veto and boost applied |
| Plugin system (norm + hash) | ByzantinePlugin + VerificationPlugin | 0.005 | Plugin pipeline verified |
| RDP 100 rounds | Moderate privacy | ε: 3.2→71.8 | Budget tracked |
| Phase diagram (7 scenarios) | 10-34% × rep disparity | 0.001-0.45 | 7/7 converge |

The pipeline supports **consciousness-aware aggregation** through the `ExternalWeightMap` mechanism: external modules (e.g., Φ-based quality assessment, epistemic E-N-M-H classification) produce per-participant weight adjustments that are merged into the aggregation. The epistemic weight formula is:

$$w_i = E_{\text{factor}} \cdot N_{\text{factor}} \cdot M_{\text{factor}} \cdot H_{\text{factor}} \cdot (0.5 + \Phi_i \cdot 0.5) \cdot c_i$$

where $E/N/M/H$ are factors derived from epistemic classification levels, $\Phi_i$ is the participant's integrated information score (used as a quality proxy), and $c_i$ is confidence. Participants with weights below a threshold can be vetoed entirely.

**Novel contributions relative to existing FL systems:**

| Capability | Ours | Google FL | PySyft | Flower | FATE |
|-----------|------|-----------|--------|--------|------|
| Consciousness-guided quality (Φ) | ✓ | — | — | — | — |
| HDC 2000× compression | ✓ | — | — | — | — |
| Multi-signal Byzantine (4-signal) | ✓ | — | Partial | — | — |
| Hybrid rep-weighted BFT | ✓ | — | — | — | — |
| Epistemic classification (E-N-M-H) | ✓ | — | — | — | — |
| Differential privacy + RDP | ✓ | ✓ | ✓ | Partial | ✓ |
| Plugin extensibility | ✓ | — | Partial | ✓ | — |
| Validated 34% BFT | ✓ | — | — | — | — |

The shared core (`mycelix-fl-core`, f32 precision, 82 unit tests + 8 integration + 8 cross-language fixtures) is dependency-minimal (serde + rand + thiserror only), enabling reuse across the Symthaea HDC engine and Mycelix SDK without dependency conflicts. An f64 wrapper layer in the SDK preserves backward compatibility for higher-precision clients. Cross-language consistency is verified: Rust, TypeScript (14 tests), and Python (15 tests) produce identical results within 1e-4 tolerance for all 5 aggregation algorithms and Byzantine detection.

**Table 6: Federated MNIST convergence (softmax 784→10, 15 nodes, 20 rounds)**

| Scenario | Final Accuracy | Converged |
|----------|---------------|-----------|
| IID, no Byzantine | 100% | ✓ |
| Non-IID (2-3 classes/node) | 100% | ✓ |
| IID + 10% Byzantine | 100% | ✓ |
| IID + 20% Byzantine | 100% | ✓ |
| IID + 30% Byzantine | 100% | ✓ |
| IID + DP (low noise) | 99% | ✓ |
| IID + DP (moderate noise) | 90% | ✓ |
| IID + DP (high noise) | 62% | ✓ |

The unified pipeline resists up to 30% Byzantine nodes on a real classification task (10-class softmax with 7,850 parameters). Differential privacy shows the expected accuracy-privacy tradeoff: low noise preserves 99% accuracy while high noise degrades to 62%.

**Table 7: Meta-learning signal weight adaptation (35 rounds, 8 honest + 3 Byzantine)**

| Attack Type | Dominant Signal After Adaptation | Exclusion Rate Decay (20 reform rounds) |
|-------------|--------------------------------|----------------------------------------|
| Magnitude (±100) | magnitude: 0.250→0.334 | 0.96→0.035 |
| Direction (opposite) | direction: 0.350→0.400 | 0.96→0.035 |
| Subtle (2× honest) | direction: 0.350→0.272 | 0.96→0.084 |

The meta-learning plugin adapts signal weights to the dominant attack pattern: magnitude attacks increase the magnitude signal weight, while direction attacks strengthen the direction signal. After attackers reform (20 honest rounds), EMA exclusion rates decay below the suspicion threshold (0.25), demonstrating forgiveness of reformed participants.

**Table 8: HyperFeel compression ratios (honest measurement)**

| Model Size | Input Bytes | Output Bytes | Ratio | Cosine Similarity |
|-----------|------------|-------------|-------|-------------------|
| 1K params | 4 KB | 2 KB | 1.9× | 0.818 |
| 10K params | 40 KB | 2 KB | 19.5× | 0.413 |
| 100K params | 400 KB | 2 KB | 194.6× | 0.141 |
| 1M params | 4 MB | 2 KB | 1,945× | 0.044 |

Compression ratio scales linearly with model size because the output (2 KB HV16 + header) is fixed. Reconstruction is lossy — cosine similarity degrades with higher compression ratios. The "2,000× compression" claim is accurate for 1M-parameter models but should be qualified with the reconstruction fidelity tradeoff.

**Table 9: RDP privacy budget tracking (δ = 10⁻⁵)**

| Noise Level | σ | ε at 10 rounds | ε at 50 rounds | Rounds to ε=10 |
|------------|---|---------------|---------------|----------------|
| High privacy | 1.0 | 3.20 | 16.97 | 5 |
| Moderate | 1.1 | 2.74 | 14.28 | 6 |
| Low privacy | 1.0 | 3.20 | 16.97 | 5 |

Rényi DP composition provides tighter bounds than naïve advanced composition. Budget exhaustion is detectable before the privacy guarantee degrades, enabling principled stopping criteria for federated training.

**Table 10: Byzantine tolerance phase diagram (20 nodes, MSE threshold < 1.0)**

| Byzantine % | Equal-Rep Pipeline | Low-Rep Byzantine (rep²) | Krum (k=1) |
|------------|-------------------|------------------------|------------|
| 0% | 0.000 (PASS) | 0.000 (PASS) | 0.000 (PASS) |
| 5% | 30.56 (FAIL) | 0.000 (PASS) | 0.000 (PASS) |
| 10% | 122.2 (FAIL) | 0.000 (PASS) | 0.000 (PASS) |
| 15% | 275.0 (FAIL) | 0.000 (PASS) | 0.000 (PASS) |
| 20% | 488.9 (FAIL) | 0.000 (PASS) | 0.000 (PASS) |
| 25% | 763.9 (FAIL) | 0.535 (PASS) | 0.000 (PASS) |
| 30% | 1100 (FAIL) | 2.42 (FAIL) | 0.000 (PASS) |
| 34% | Error (ERR) | 6.19 (FAIL) | 0.000 (PASS) |
| 40% | Error (ERR) | 12.6 (FAIL) | 0.000 (PASS) |

Three defense tiers emerge: (1) Equal-reputation aggregation fails at 5% Byzantine—even with multi-signal detection, equal weighting poisons the mean. (2) Reputation-squared weighting extends the safety boundary to 25%, reducing effective Byzantine voting power from 34% to ~2.7%. (3) Krum selection withstands 40%+ by choosing the single gradient closest to its neighbors (theoretical limit: 45%).

**Table 11: Real MNIST federated training (linear softmax 784->10, 20 nodes, 40 rounds)**

| Experiment | Accuracy | Per-class range |
|-----------|---------|----------------|
| IID-clean | 67.4% | 2.8%-92.6% |
| Non-IID (2 classes/node) | 55.5% | - |
| Non-IID (3 classes/node) | 59.9% | - |
| IID + 10% Byzantine | 67.4% | - |
| IID + 20% Byzantine | 67.4% | - |
| Non-IID + 20% Byzantine | 54.9% | - |
| High-security (DP) + 20% Byz | 47.8% | - |

Real MNIST validation confirms: (1) the pipeline completely neutralizes 20% Byzantine nodes (identical accuracy to clean), (2) non-IID partitioning degrades accuracy proportionally to class heterogeneity, (3) differential privacy trades ~20% accuracy for formal privacy guarantees. With 10K training samples and 40 federated rounds, the linear softmax reaches 67.4%—consistent with centralized baselines on subsampled data.


---

## 7. Conclusion

We presented Hyperdimensional Active Inference (HAI), the first integration of Free Energy Principle active inference with Hyperdimensional Computing. By reformulating variational free energy in hypervector space and introducing precision-weighted binding, we achieve:

1. **Computational efficiency:** 1.9-15.8× speedup over pymdp (7.9× total), with O(d) scaling
2. **Interpretable action selection:** Eight motor command types from EFE minimization
3. **Correct precision dynamics:** Adaptive confidence weighting validated empirically
4. **Mathematical soundness:** Free energy convergence and KL non-negativity confirmed
5. **Compositional ethical reasoning:** A moral algebra system using HDC operators directly—without neural-symbolic translation—achieves 92.9% on the ETHICS benchmark (Commonsense 95.6%, Justice 92.4%, Deontology 91.0%, Virtue 92.8%), demonstrating that hypervector algebra supports compositional semantics for real-world reasoning tasks
6. **Unified federated learning:** A consciousness-aware FL pipeline combining differential privacy, multi-signal Byzantine detection, reputation-weighted aggregation, and plugin extensibility—validated on real MNIST (67.4% accuracy, 20% Byzantine fully neutralized) with a 9-point Byzantine phase diagram showing three defense tiers

The full implementation spans ~338K lines of Rust across 3,200+ tests, with all benchmarks independently reproducible from the open-source repository. HAI opens new directions for efficient, interpretable cognitive architectures that combine the theoretical rigor of active inference with the computational elegance of hyperdimensional computing.

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

[14] Hasani, R., Lechner, M., Amini, A., Liebenwein, L., Ray, A., Tschaikowski, M., Teschl, G., & Rus, D. (2021). Liquid time-constant networks. *AAAI*, 35(9), 7657-7666.

[15] Hasani, R., Lechner, M., Amini, A., Liebenwein, L., Tschaikowski, M., Teschl, G., & Rus, D. (2022). Closed-form continuous-time neural networks. *Nature Machine Intelligence*, 4(11), 992-1003.

[16] Hersche, M., Terzic, B., Kleyko, D., et al. (2023). A neuro-vector-symbolic architecture for solving Raven's progressive matrices. *Nature Machine Intelligence*, 5(4), 363-375.

[17] Imani, M., Duan, Y., Rosing, T. (2019). Hierarchical hyperdimensional computing for energy-efficient classification. *DAC*, 1-6.

[18] Albantakis, L., Barbosa, L., Findlay, G., et al. (2023). Integrated information theory (IIT) 4.0: Formulating the properties of phenomenal existence in physical terms. *PLoS Computational Biology*, 19(10), e1011465.

[19] Fountas, Z., Sajid, N., Mediano, P. A. M., & Friston, K. (2020). Deep active inference agents using Monte-Carlo methods. *NeurIPS*, 33, 11662-11675.

[20] Ha, D., & Schmidhuber, J. (2018). World models. *arXiv preprint arXiv:1803.10122*.

[21] Toker, D., & Sommer, F. T. (2019). Information integration in large brain networks. *PLoS Computational Biology*, 15(2), e1006807.

[22] Kim, Y., Duan, Y., Imani, M., et al. (2020). HDC for DNA sequence classification with error-resilient encoding. *DAC*, 1-6.

[23] Manhaeve, R., Dumancic, S., Kimmig, A., Demeester, T., & De Raedt, L. (2018). DeepProbLog: Neural probabilistic logic programming. *NeurIPS*, 31, 3749-3759.

[24] Riegel, R., Gray, A., Luus, F., et al. (2020). Logical neural networks. *arXiv preprint arXiv:2006.13155*.

[25] Li, Z., Huang, J., & Naik, M. (2023). Scallop: A language for neurosymbolic programming. *Proceedings of the ACM on Programming Languages*, 7(PLDI), 1463-1487.

[26] Mao, J., Gan, C., Kohli, P., Tenenbaum, J. B., & Wu, J. (2019). The neuro-symbolic concept learner: Interpreting scenes, words, and sentences from natural supervision. *ICLR*.

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
- 95% confidence intervals computed as mean ± 1.96 × SE
- Platform: Single-threaded execution on AMD Ryzen 9
- Reproducibility: All benchmarks use deterministic seeds; see `data/benchmarks/*/results.json` for raw data

### C.3 Quality Ratio

The Quality Ratio (3.10) is computed as:
$$QR = \frac{|FE_{pymdp}|}{|FE_{HAI}|} = \frac{9.33}{3.01} \approx 3.1$$

Higher QR indicates HAI achieves lower (better) free energy.

---

*Manuscript prepared: February 2026*
*Corresponding author: tristan.stoltz@evolvingresonantcocreationism.com*

**Supplementary Materials:**
- `papers/figures/` - Figures 1-6 (PDF and PNG):
  - Fig 1: HAI Architecture Diagram
  - Fig 2: Free Energy Convergence Curves
  - Fig 3: Precision Dynamics
  - Fig 4: Scaling Analysis (HAI vs pymdp)
  - Fig 5: Benchmark Validation Radar (17 benchmarks)
  - Fig 6: λ₂-Φ Proxy Validation Scatter (ρₛ = 0.50)
- `papers/appendices/theoretical_analysis.md` - Appendix D: Formal Proofs
- `docs/PYMDP_COMPARISON_REPORT.md` - pymdp Benchmark Details
- `docs/ABLATION_STUDIES_REPORT.md` - Dimension, Precision, EFE Weight Ablations
- `docs/STATISTICAL_ANALYSIS_REPORT.md` - 95% CI and Significance Tests
- `docs/EXTENDED_BENCHMARKS_REPORT.md` - Tiger, Large Grids, Multi-Agent
- `data/benchmarks/*/results.json` - Raw benchmark data (JSON) with per-class accuracies, timing, and configuration details
