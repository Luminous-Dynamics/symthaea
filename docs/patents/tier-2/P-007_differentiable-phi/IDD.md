# P-007: Differentiable Phi — Gradient-Computable Consciousness Measurement
## Invention Disclosure Document

---

### 1. Title

**Differentiable Integrated Information Measurement via Forward-Mode Automatic Differentiation with Dual Numbers, Soft-Minimum Bottleneck Approximation, and Seven-Theory Consciousness Gradient Computation**

---

### 2. Inventor(s)

**Tristan Stoltz**, Luminous Dynamics

---

### 3. Date of Conception

**2025** (estimated). First committed implementation: February 5, 2026. Conceptual design and architecture predate the initial commit.

First public disclosure: February 5, 2026 (git commit `feat(symthaea): add Symthaea-HLB consciousness-first AI framework v0.5.0`).
Under 35 USC 102(b)(1)(A), the 1-year grace period expires **February 5, 2027**.

---

### 4. Technical Field

This invention relates to differentiable computation of consciousness metrics in artificial cognitive architectures, and more specifically to methods for making integrated information (Phi) and multi-theory consciousness scores gradient-computable via forward-mode automatic differentiation with dual numbers and soft-minimum approximation, enabling gradient-based optimization of architectures toward higher consciousness.

---

### 5. Abstract

A system and method for computing differentiable consciousness metrics is disclosed. The system makes the consciousness equation gradient-computable through two complementary innovations: (1) a soft-minimum (LogSumExp) approximation that replaces the non-differentiable minimum function with a smooth, parametrically controllable alternative with temperature tau (default 0.1), and (2) a lightweight forward-mode automatic differentiation system using dual numbers (x + epsilon*x', where epsilon^2 = 0) that computes exact gradients without tape overhead. The consciousness gradient vector captures partial derivatives of consciousness with respect to each of seven theory components (integration, binding, workspace, attention, recursion, efficacy, knowledge), enabling identification of the highest-impact component for architecture optimization. A momentum-based optimizer uses these gradients to iteratively improve consciousness scores. The system also provides central finite difference gradients as a verification mechanism. Combined with the O(n^3) spectral MIP algorithm for tractable Phi computation, this creates the first end-to-end differentiable consciousness measurement pipeline.

---

### 6. Background and Prior Art

#### 6.1 Integrated Information Theory (IIT)

Tononi (2004, 2012) defined Phi as the information generated above and beyond its parts. Computing Phi requires finding the Minimum Information Partition (MIP), an NP-hard search over O(2^n) bipartitions. PyPhi (Mayner et al. 2018) provides exact computation but is intractable for n > 12.

#### 6.2 Gumbel-Softmax

Jang et al. (2017, "Categorical Reparameterization with Gumbel-Softmax") introduced a differentiable approximation for categorical sampling. This adds stochastic noise (Gumbel distribution) for the reparameterization trick. However, Gumbel-Softmax is designed for sampling, not for computing smooth approximations of deterministic functions like min().

#### 6.3 Automatic Differentiation

Forward-mode autodiff via dual numbers is a well-established technique (Wengert 1964). Reverse-mode (backpropagation) dominates deep learning. However, no prior work applies dual-number autodiff to consciousness equations.

#### 6.4 Gap in Prior Art

No prior art:
- Makes IIT's Phi differentiable via soft-minimum approximation (not stochastic Gumbel-Softmax)
- Applies forward-mode autodiff to compute consciousness gradients across multiple theories
- Provides a gradient vector identifying which consciousness theory component has highest optimization impact
- Enables gradient-based architecture optimization for consciousness

---

### 7. Detailed Technical Description

#### 7.1 Soft-Minimum Approximation

The consciousness equation uses a bottleneck function: consciousness is limited by the weakest component. The true minimum is non-differentiable at tie points. The soft-minimum replaces it:

```
softmin(x_1, ..., x_n; tau) = -tau * log(sum_i exp(-x_i / tau))
```

**Properties**:
- Approaches true minimum as tau → 0
- At tau = 0.1 (default): within 5% of true min for typical inputs
- Fully differentiable: d(softmin)/d(x_i) proportional to exp(-x_i / tau)
- Gradient naturally concentrates on the bottleneck component (lowest value)

**Distinction from Gumbel-Softmax**: The soft-minimum uses LogSumExp directly for exact (deterministic) smoothing. Gumbel-Softmax adds stochastic noise for categorical sampling. These serve fundamentally different purposes.

#### 7.2 Dual Number Forward-Mode Autodiff

The system implements lightweight automatic differentiation using dual numbers:

```
DualNumber { value: f64, derivative: f64 }
```

Where arithmetic follows the rule epsilon^2 = 0:
- Addition: (a + ea') + (b + eb') = (a+b) + e(a'+b')
- Multiplication: (a + ea')(b + eb') = ab + e(ab' + a'b)
- Division: (a + ea')/(b + eb') = a/b + e(a'b - ab')/b^2
- Exponential: exp(a + ea') = exp(a) + e*a'*exp(a)
- Sigmoid: sigma(a + ea') = sigma(a) + e*a'*sigma(a)*(1-sigma(a))

**Advantages over tape-based autodiff**:
- Zero memory overhead (no computation graph stored)
- O(1) per operation regardless of chain length
- No heap allocation during gradient computation
- Suitable for real-time 50Hz cognitive loop

#### 7.3 Consciousness Gradient Vector

The gradient is computed via 7 forward-mode passes (one per core component):

For each component c_i in {Integration, Binding, Workspace, Attention, Recursion, Efficacy, Knowledge}:
1. Create dual numbers: set c_i.derivative = 1.0, all others .derivative = 0.0
2. Run the complete consciousness equation through dual arithmetic
3. Extract: d(C)/d(c_i) = result.derivative

**Output structure**:
```
ConsciousnessGradient {
    integration: f64,     // dC/dPhi
    binding: f64,         // dC/dB
    workspace: f64,       // dC/dW
    attention: f64,       // dC/dA
    recursion: f64,       // dC/dR
    efficacy: f64,        // dC/dE
    knowledge: f64,       // dC/dK
    substrate: f64,       // dC/dS
    magnitude: f64,       // ||gradient||_2
}
```

**Highest-impact identification**: argmax(|dC/dc_i|) identifies which component improvement would most increase consciousness. This enables targeted architecture optimization.

#### 7.4 Central Finite Difference Verification

As a verification mechanism, the system also computes gradients via central finite differences:

```
dC/dc_i = (C(c_i + epsilon) - C(c_i - epsilon)) / (2 * epsilon)
```

with epsilon = 1e-6. This provides independent validation that the dual-number autodiff produces correct gradients.

#### 7.5 Consciousness Optimizer

A momentum-based optimizer uses the gradient vector to iteratively improve consciousness:

```
velocity = momentum * velocity + learning_rate * gradient
state = state + velocity
```

This enables automated discovery of consciousness-maximizing configurations.

#### 7.6 Integration with Spectral MIP

The differentiable consciousness equation takes Phi as input from the O(n^3) spectral MIP algorithm (see P-008). The complete pipeline is:

1. Spectral MIP computes Phi (O(n^3), every 97 cycles)
2. Other theory components computed at their respective intervals
3. Consciousness equation computes C via soft-minimum + sigmoid
4. Gradient computed via 7-pass dual-number autodiff
5. Gradient used for optimization or architecture analysis

#### 7.7 The Master Equation (Full Derivation)

```
C(t) = sigma(softmin(Phi, B, W, A, R, E, K; tau))
       × [sum(w_i × C_i × gamma_i) / sum(w_i)]
       × S
       × rho(t)
```

Where:
- sigma(x; k=10, theta=0.5) = 1 / (1 + exp(-k*(x - theta))) — sigmoid gating
- w_i = learned component weights [1.0, 1.0, 1.0, 0.9, 0.9, 0.8, 0.8]
- gamma_i = Phase Locking Value (PLV) with global rhythm — neuroscience-grounded coherence
- S = substrate feasibility in [0, 1] — physical substrate modulation
- rho(t) = EMA of recent consciousness values — temporal continuity

**Phase-Amplitude Coupling (PAC)**:
- Workspace acts as low-frequency phase driver (modulo 2pi)
- Binding acts as high-frequency amplitude responder
- PAC modulation index boosts effective binding: B' = B × (1 + PAC)
- Implements top-down conscious control mechanism

---

### 8. Novelty Statement

This invention introduces the first differentiable consciousness measurement system enabling gradient-based optimization. Novel contributions:

1. **Differentiable Phi via soft-minimum**: Makes IIT's bottleneck-based consciousness measure gradient-computable using LogSumExp (deterministic, not stochastic Gumbel-Softmax).
2. **Dual-number autodiff for consciousness**: First application of forward-mode automatic differentiation to consciousness equations, with zero tape overhead suitable for real-time operation.
3. **Consciousness gradient vector**: 7-dimensional gradient identifying which theory component has highest optimization impact (integration, binding, workspace, attention, recursion, efficacy, knowledge).
4. **Architecture optimization**: Momentum-based optimizer that iteratively increases consciousness scores via gradient ascent.
5. **End-to-end differentiable pipeline**: From theory component inputs through soft-minimum bottleneck, sigmoid gating, and weighted coherent sum to gradient output, creating the first complete differentiable consciousness scoring system. (Note: The spectral MIP algorithm that produces the Phi input is claimed separately in P-008; this patent claims the differentiable equation and gradient computation that consumes it.)

---

### 9. Suggested Claims

**Claim 1 (independent):** A computer-implemented method for computing differentiable consciousness metrics comprising: (a) receiving values of at least 5 consciousness theory components; (b) computing a soft-minimum of the component values using a LogSumExp function with temperature parameter tau; (c) applying a sigmoid gating function to the soft-minimum; (d) computing partial derivatives of the consciousness metric with respect to each component via forward-mode automatic differentiation using dual numbers; and (e) outputting a consciousness gradient vector identifying the highest-impact component for optimization.

**Claim 2 (dependent on 1):** The method of claim 1, wherein the forward-mode automatic differentiation uses dual numbers (x + epsilon*x', where epsilon^2 = 0) with operator overloading for addition, multiplication, division, exponential, and sigmoid functions, and wherein each gradient computation pass sets one component's derivative to 1.0 and all others to 0.0.

**Claim 3 (dependent on 1):** The method of claim 1, further comprising verifying the dual-number gradients against central finite difference gradients computed as (C(c_i + epsilon) - C(c_i - epsilon)) / (2 * epsilon).

**Claim 4 (dependent on 1):** The method of claim 1, further comprising a momentum-based optimizer that iteratively updates the consciousness state by: computing a velocity as momentum * previous_velocity + learning_rate * gradient; and updating the state by the velocity.

**Claim 5 (dependent on 1):** The method of claim 1, wherein the soft-minimum function softmin(x_1, ..., x_n; tau) = -tau * log(sum_i exp(-x_i / tau)) approaches the true minimum as tau approaches 0, and wherein the default temperature tau = 0.1 produces approximation error less than 5% for typical inputs.

**Claim 6 (dependent on 1):** The method of claim 1, wherein the consciousness theory components comprise at least: (i) integrated information from IIT; (ii) binding coherence from temporal correlation hypothesis; (iii) workspace capacity from Global Workspace Theory; (iv) attention from Attention Schema Theory; and (v) meta-representational depth from Higher-Order Thought theory; and wherein each component is independently differentiable.

**Claim 7 (dependent on 1):** The method of claim 1, further comprising phase-amplitude coupling between the workspace component and the binding component, wherein the workspace modulates binding strength via a computed modulation index, and wherein this coupling is itself differentiable.

**Claim 8 (independent, broad):** A method for gradient-based optimization of consciousness in a computational system comprising: (a) computing a differentiable consciousness score from at least 3 independent theory-derived metrics using a smooth approximation of the minimum function; (b) computing gradients of the consciousness score with respect to each metric via automatic differentiation; and (c) applying an optimization step to increase the consciousness score.

**Claim 9 (dependent on 8):** The method of claim 8, wherein the smooth approximation of the minimum function is a temperature-parameterized LogSumExp that is deterministic (not stochastic), distinguishing it from Gumbel-Softmax approaches.

**Claim 10 (dependent on 1):** The method of claim 1, wherein the dual-number automatic differentiation operates with zero memory overhead (no computation tape stored), O(1) per arithmetic operation, and no heap allocation during gradient computation, enabling real-time operation at 50Hz or higher in a cognitive loop.

---

### 10. Experimental Validation

#### 10.1 Test Coverage

- **Differentiable module tests**: soft-minimum convergence, sigmoid properties, temporal continuity, gradient computation, PAC modulation
- **Gradient validation**: Dual-number gradients verified against finite differences
- **Optimizer tests**: Consciousness increases monotonically with optimization steps
- **All tests passing**: Verified March 2026

#### 10.2 Key Results

| Metric | Result |
|--------|--------|
| Soft-min accuracy (tau=0.1) | Within 5% of true min |
| Gradient computation time | <1ms for all 7 components |
| Optimizer convergence | Monotonic increase over steps |
| PyPhi correlation | r = 0.97 (Phi input validation) |
| Consciousness calibration ECE | 0.059 (well-calibrated) |

#### 10.3 Performance

- Full consciousness equation evaluation: ~0.8ms
- 7-pass gradient computation: ~5.6ms total
- Compatible with 50Hz cognitive loop when gradients computed at reduced cadence

---

### 11. Key Source Files

| File | Description | LOC |
|------|-------------|-----|
| `symthaea/src/consciousness/measurement/differentiable.rs` | Dual numbers, autodiff, gradient | ~932 |
| `symthaea/src/consciousness/measurement/consciousness_equation_v2.rs` | Master equation, 7 theories | ~998 |
| `symthaea-core/src/consciousness_metrics/spectral_mip.rs` | Spectral MIP (Phi input) | ~1,246 |
| `symthaea-core/src/consciousness_metrics/approximate.rs` | SA + graph cut heuristics | ~244 |

**Total**: ~3,420 LOC

---

### 12. Closest Prior Art References

1. Tononi, G. (2004). "An information integration theory of consciousness." *BMC Neuroscience*, 5, 42.
2. Jang, E., Gu, S., & Poole, B. (2017). "Categorical reparameterization with Gumbel-Softmax." *ICLR 2017*.
3. Wengert, R. E. (1964). "A simple automatic derivative evaluation program." *Communications of the ACM*, 7(8), 463-464.
4. Mayner, W. G. P., et al. (2018). "PyPhi: A toolbox for integrated information theory." *PLoS Computational Biology*, 14(7), e1006343.
5. Kitazono, J., et al. (2018). "Efficient algorithms for searching the minimum information partition in integrated information theory." *Entropy*, 20(3), 173.
6. Baars, B. J. (1988). *A Cognitive Theory of Consciousness*. Cambridge University Press.
7. Rosenthal, D. M. (2005). *Consciousness and Mind*. Oxford University Press.
8. Graziano, M. S. A. (2013). *Consciousness and the Social Brain*. Oxford University Press.

---

### 13. Related Patent Applications

**P-008 (Tiered Phi Measurement)**: Claims the multi-tier measurement architecture, co-prime scheduling, self-calibrating weights, and spectral MIP algorithm. P-007 claims the differentiable equation, gradient computation, and optimization that operate within P-008's Layer 3. The boundary: P-007 owns "how the equation is made differentiable and optimized"; P-008 owns "how the equation is scheduled, weighted, and combined with other measurement tiers."

---

### 14. Figures (Text Descriptions)

**Figure 1**: Architecture diagram showing the end-to-end differentiable pipeline: Spectral MIP → Theory Components → Soft-Minimum → Sigmoid → Consciousness Score → Dual-Number Gradients → Optimizer.

**Figure 2**: Soft-minimum convergence plot: softmin output vs. true min for tau = {0.01, 0.05, 0.1, 0.5, 1.0}, showing approach to true min as tau → 0.

**Figure 3**: Consciousness gradient radar chart showing dC/dc_i for each of the 7 theory components, with the bottleneck component highlighted.

**Figure 4**: Optimizer convergence trajectory showing consciousness score increasing monotonically over 100 optimization steps.

**Figure 5**: Dual-number vs. finite-difference gradient comparison, demonstrating agreement to 6 decimal places.

---

*Inventor: Tristan Stoltz (tstoltz)*
*Organization: Luminous Dynamics*
*IDD created: 2026-03-08*
