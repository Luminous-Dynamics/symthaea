# P-001: Unified HDC-LTC Neuron with Closed-Form Temporal Jumps
## Provisional Patent Application

---

### 1. Title

**Unified Hyperdimensional Computing Neuron with Liquid Time-Constant Dynamics and Closed-Form Temporal Jump Solutions Using Algebraic Binding Operations and SIMD-Accelerated Fused Evolution Kernels**

---

### 2. Inventor(s)

**Tristan Stoltz**, Luminous Dynamics

---

### 3. Date of Conception

**2025** (estimated). First public disclosure: February 5, 2026 (git commit `feat(symthaea): add Symthaea-HLB consciousness-first AI framework v0.5.0`). Conceptual design and architecture predate the initial commit. Under 35 USC 102(b)(1)(A), the 1-year grace period expires **February 5, 2027**.

---

### 4. Technical Field

This invention relates to artificial neural network architectures for temporal sequence processing, and more specifically to a neuron architecture that unifies hyperdimensional computing (HDC) with liquid time-constant (LTC) neural dynamics, enabling O(1) closed-form temporal jumps using algebraic binding operations instead of matrix multiplication.

---

### 5. Abstract

A novel neural processing unit is disclosed that represents its internal state as a high-dimensional continuous hypervector (e.g., 16,384 dimensions) and evolves that state through liquid time-constant dynamics using hyperdimensional algebraic operations---specifically element-wise binding and normalized bundling---in place of conventional weight matrix multiplications. This substitution reduces the per-neuron parameter count from O(D^2) to O(D) while preserving the expressive power needed for temporal sequence modeling. The neuron's governing ordinary differential equation admits a closed-form exponential decay solution, enabling O(1) temporal jumps to arbitrary future time horizons at cost independent of the time step magnitude. A state-dependent time constant adapts the neuron's response speed to the complexity of its current representation. An adaptive gating mechanism, inspired by Closed-form Continuous-depth (CfC) networks, computes a learned interpolation factor that blends the current state with the computed equilibrium. Optional Fourier basis injection provides time-varying perturbations for richer temporal dynamics. Hand-written AVX2+FMA SIMD kernels fuse the entire equilibrium computation and state interpolation into a single pass through the dimension, eliminating intermediate allocations (256 KB per invocation for D=16,384) and processing 8 elements per cycle. The architecture supports multiple learning rules including Hebbian, contrastive, STDP, triplet, and full backpropagation through time (BPTT) with analytical gradient computation through the closed-form step.

---

### 6. Background and Prior Art

#### 6.1 Liquid Time-Constant (LTC) Networks

Lechner et al. (2020, "Neural Circuit Policies Enabling Auditable Autonomy," Nature Machine Intelligence) introduced LTC neurons whose time constants are input-dependent, governed by the ODE:

```
dx/dt = (-x + f(Wx + Uu)) / tau
```

where W and U are weight matrices, f is a nonlinear activation, and tau is a time constant. These networks demonstrated strong performance on temporal tasks but require numerical ODE integration (Euler or RK4), making them computationally expensive for large time horizons.

#### 6.2 Closed-form Continuous-depth (CfC) Networks

Hasani et al. (2021, "Liquid Time-constant Networks," AAAI) and subsequent work on CfC networks derived closed-form solutions for LTC dynamics, enabling single-step temporal updates. However, CfC networks retain conventional weight matrices (O(D^2) parameters per layer) and do not exploit the algebraic structure of hyperdimensional computing.

#### 6.3 Hyperdimensional Computing (HDC)

Kanerva (2009, "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors") established the theoretical foundation for computing with high-dimensional vectors using binding (element-wise multiplication) and bundling (normalized addition). Rahimi et al. (2017) and subsequent work applied HDC to classification tasks. However, existing HDC systems are typically static classifiers without temporal dynamics---they lack the continuous-time evolution needed for sequence processing.

#### 6.4 Standard Recurrent Networks (LSTM/GRU)

Long Short-Term Memory (Hochreiter and Schmidhuber, 1997) and Gated Recurrent Units (Cho et al., 2014) are the dominant architectures for temporal processing. Both use weight matrices with O(D^2) parameters and cannot perform temporal jumps---they must process every time step sequentially.

#### 6.5 Gap in Prior Art

No prior art combines HDC algebraic operations with LTC temporal dynamics. Specifically:

- **LTC/CfC networks** use matrix multiplication (O(D^2) parameters), not HDC binding (O(D) parameters).
- **HDC systems** are static classifiers without continuous-time ODE dynamics.
- **No existing architecture** provides both O(D) parameterization AND O(1) temporal jumps.
- **No existing architecture** uses state-dependent time constants computed from hypervector norms.
- **No existing SIMD implementation** fuses HDC binding, bundling, activation, and temporal interpolation into a single pass.

---

### 7. Detailed Technical Description

#### 7.1 Neuron Architecture

The unified HDC-LTC neuron (`HdcLtcUnifiedNeuron`) maintains the following internal hypervectors, all of dimension D (default D = 16,384):

| Component | Symbol | Role |
|-----------|--------|------|
| State | **x** | Current neuron state (ContinuousHV) |
| Weight HV | **W** | State transformation via binding (replaces weight matrix) |
| Input Mask | **U** | Input transformation via binding |
| Tau Modulator | **T_mod** | Input-dependent time constant adjustment |
| Gate Weight | **G_w** | Gating function for closed-form interpolation |
| Gate Bias | **G_b** | Gating bias (initialized at 0.1x scale) |
| Weight Momentum | **M_w** | Momentum accumulator for weight updates |
| Input Momentum | **M_u** | Momentum accumulator for input mask updates |

All internal HVs are initialized via Gram-Schmidt orthogonalization (`ContinuousHV::orthogonal_set`) to ensure pairwise similarity < 0.02 at initialization, minimizing interference between learned representations.

**Key insight**: By replacing weight *matrices* (D x D) with weight *hypervectors* (D), each neuron requires O(D) parameters instead of O(D^2). For D = 16,384, this is a reduction from ~268 million to ~16,384 parameters per weight component.

#### 7.2 Governing ODE

The neuron's dynamics are governed by:

```
dx/dt = (-x + f(W * x + U * u)) / tau(||x||)
```

In HDC notation with explicit operations:

```
dx/dt = (-x (bundling-inverse) f(W (bind) x (bundle) U (bind) u)) / tau(||x||)
```

Where:
- `(bind)` denotes HDC binding: element-wise multiplication, `(A bind B)_i = A_i * B_i`
- `(bundle)` denotes HDC bundling: normalized element-wise addition, `bundle(A, B) = (A + B) / n` where n is the number of operands
- `f` is a configurable activation function (Tanh, Sigmoid, SiLU, Identity, or BoundedTanh)

#### 7.3 Equilibrium State Computation

The equilibrium state x_inf is computed as:

```
x_inf = f(bundle(W (bind) x, U (bind) u [, F(t)]))
```

Where F(t) is an optional Fourier basis injection (Section 7.7). The `compute_equilibrium` method performs:

1. **State transformation**: `transformed_state = W (bind) x` (element-wise multiply)
2. **Input transformation**: `masked_input = U (bind) u` (element-wise multiply)
3. **Bundling**: `combined = bundle(transformed_state, masked_input)` (normalized sum)
4. **Activation**: `x_inf = f(combined)` (element-wise nonlinearity)

#### 7.4 State-Dependent Time Constant

The time constant adapts to both state complexity and input characteristics:

```
tau(x, u) = tau_0 * (1 + backbone * ||x||) * (1 + 0.2 * sim(u, T_mod))
```

Where:
- `tau_0` is the base time constant (default: 0.1s, i.e., 100ms)
- `backbone` is the scaling factor (default: 0.5)
- `||x||` is the L2 norm of the state hypervector
- `sim(u, T_mod)` is the cosine similarity between input and tau modulator HV
- The result is clamped to [0.01, 10.0] seconds

This means the neuron responds faster when its state is simple (low norm) and slower when processing complex representations (high norm), providing an adaptive computational budget.

#### 7.5 Closed-Form Solution

For the ODE `dx/dt = (x_inf - x) / tau`, assuming x_inf is approximately constant over the time step, the exact analytical solution is:

```
x(t + dt) = x_inf + (x(t) - x_inf) * exp(-dt / tau)
```

This can be rewritten as an interpolation:

```
x(t + dt) = sigma * x_inf + (1 - sigma) * x(t)
```

where `sigma = 1 - exp(-dt / tau)`.

**Critical property**: The computational cost is O(D) regardless of dt. A jump of 0.001 seconds costs exactly the same as a jump of 100 seconds.

#### 7.6 Adaptive Gating Mechanism (CfC-style)

The `compute_gating` method computes a learned interpolation factor sigma that incorporates both the exponential decay and a learned gating signal:

1. **Fused bundle + similarity**: Compute `bundle(x, u)` and its similarity with `G_w` in a single inline loop (avoiding a 16,384-element allocation):
   ```
   gate_activation = sim(bundle(x, u), G_w) + mean(G_b)
   ```

2. **Sigmoid gating with steepness control**:
   ```
   sigma_base = 1 / (1 + exp(-gate_activation * steepness))
   ```
   where `steepness` (default: 1.0) controls the sharpness of the gating decision.

3. **Time-scaled gating**: Combine learned gating with exponential decay:
   ```
   decay = exp(-dt / tau)
   sigma = 1 - decay * (1 - sigma_base)
   ```
   Result is clamped to [0.0, 1.0].

4. **Final interpolation** (zero-allocation in-place):
   ```
   x(t + dt) = (1 - sigma) * x(t) + sigma * x_inf
   ```
   Implemented via `lerp_in_place` which modifies state values directly without allocating a new hypervector.

#### 7.7 Fourier Basis Injection

When configured with one or more frequencies, the neuron computes a time-varying Fourier basis hypervector that is bundled into the equilibrium computation as a third component:

```
x_inf = f(bundle(W (bind) x, U (bind) u, F(t)))
```

The Fourier basis F(t) is constructed by distributing sin/cos pairs across the dimension via a strided pattern:

- For each frequency f_i, compute `sin(2*pi*f_i*t) * amplitude` and `cos(2*pi*f_i*t) * amplitude`
- Sin values are placed at dimension indices `2*i, 2*i + total_channels, 2*i + 2*total_channels, ...`
- Cos values are placed at dimension indices `2*i+1, 2*i+1 + total_channels, ...`
- `total_channels = 2 * num_frequencies`
- Default amplitude: 0.1 (small perturbation, not a driver)

This enables the neuron to encode absolute time information and exhibit periodic behaviors without external clock signals.

#### 7.8 SIMD-Optimized Fused Evolution Kernel

The `evolve_closed_form_fused` method eliminates all intermediate allocations by fusing the entire evolution step into a single pass through the dimension. On x86_64 with AVX2+FMA, hand-written SIMD intrinsics process 8 f32 elements per cycle (256-bit registers).

**Fused Tanh kernel** (`fused_tanh_avx2`):

For each 8-element chunk:
1. Load state[i], weight[i], input_mask[i], input[i] into AVX2 registers
2. Compute pre-activation: `pre_act = (W[i] * state[i] + M[i] * input[i]) * pre_scale` using FMA instructions (`_mm256_fmadd_ps`)
3. Compute fast_tanh via rational approximation: `tanh(x) approx x * (27 + x^2) / (27 + 9*x^2)` for |x| < 4.97, with +-1.0 clamping for |x| >= 4.97 (max error ~0.004)
4. Lerp: `new_state[i] = (1-sigma) * state[i] + sigma * x_inf[i]` using FMA
5. Store result back to state

A scalar remainder loop handles `dim % 8` trailing elements.

**Memory savings**: For D = 16,384, the fused kernel eliminates 4 intermediate ContinuousHV allocations per call, saving 4 * 16,384 * 4 bytes = **256 KB per invocation**.

A separate `fused_identity_avx2` kernel handles the Identity activation case (no tanh computation).

**Fast tanh approximation** (`fast_tanh`):

```
fast_tanh(x) = x * (27 + x^2) / (27 + 9*x^2)    if |x| < 4.97
             = signum(x)                           if |x| >= 4.97
```

This rational approximation is pure arithmetic (no branches in the hot path, no libm calls), enabling LLVM to auto-vectorize the scalar fallback path and directly mapping to SIMD arithmetic in the hand-written kernels. Maximum error is ~0.004 (0.4%).

#### 7.9 Multiple Evolution Methods

The architecture provides five evolution methods with different accuracy/performance tradeoffs:

| Method | Complexity | Accuracy | Allocations |
|--------|-----------|----------|-------------|
| `evolve` (Euler) | O(D) per step | First-order | 2 HVs |
| `evolve_rk4` | O(4D) per step | Fourth-order | 8 HVs |
| `evolve_closed_form` | O(D), dt-independent | Gated approximation | 1 HV (equilibrium) |
| `evolve_closed_form_fused` | O(D), dt-independent | Gated approx (~0.4% tolerance) | **0 HVs** |
| `evolve_closed_form_exact` | O(D), dt-independent | Exact for linear ODE | 2 HVs |
| `evolve_closed_form_iterative` | O(D * dt/tau) | High (sub-stepped) | 2 HVs per sub-step |

#### 7.10 Learning and Adaptation

The neuron supports six learning rules, all operating on hypervector weights via HDC operations:

1. **Hebbian update**: `delta_W = lr * (input (bind) state)`, with momentum (default beta=0.9) and L2 weight decay (default 0.0001). Weight norm clipped to 2.0.

2. **Contrastive update**: Pulls state toward positive examples and pushes away from negative examples. Gradient computed as `W (bind) (positive - state) + 0.5 * W (bind) (state - negative)`.

3. **STDP (Spike-Timing Dependent Plasticity)**: Asymmetric exponential windows with tau_plus = tau_minus = 20ms, A_plus = 1.0, A_minus = 0.5. Pre-before-post strengthens, post-before-pre weakens.

4. **Adaptive (Adam-like)**: First-moment estimation with bias correction. Uses momentum beta1 (default 0.9) with correction factor `1 / (1 - beta1^t)`.

5. **Regularized Hebbian**: Combines Hebbian learning with homeostatic plasticity (target activity scaling, clamped [0.5, 2.0]) and explicit L2 penalty term.

6. **Triplet loss**: Metric learning with `loss = max(0, dist_pos - dist_neg + margin)`. Updates weights only when the margin is violated.

7. **BPTT (Backpropagation Through Time)**: The `backward` method computes analytical gradients through the closed-form step:
   - Forward recomputation of z, x_inf, tau, sigma, new_state
   - MSE loss gradient: `dL/dx' = 2(x' - target) / D`
   - Through interpolation: `dL/dx_inf = dL/dx' * sigma`
   - Through activation: `dL/dz = dL/dx_inf * f'(z)` (element-wise)
   - Weight gradient via binding chain rule: `dL/dW = dz (bind) x` (since binding is element-wise, the derivative is the other operand)
   - Input mask gradient: `dL/dU = dz (bind) u`
   - Tau gradient (scalar): `dL/dtau = sum_i(dh_i * (x_inf_i - x_i)) * (-dt/tau^2) * exp(-dt/tau)`
   - Input gradient for inter-layer BPTT: `dL/du = dz (bind) U`

   The `apply_gradients` method performs SGD with momentum, weight decay, and norm clipping to 2.0.

#### 7.11 Network Architecture

The `HdcLtcUnifiedNetwork` composes multiple neurons into a layered architecture:

- Configurable layer sizes (e.g., [4, 8, 4])
- Inter-layer binding vectors for representational transformation between layers
- Optional skip connections from input to deeper layers
- Layer output computed as mean of neuron states within each layer
- Supports both standard evolution and closed-form evolution across the full network
- Deterministic initialization from a genesis seed for reproducibility

#### 7.12 Numerical Stability

- **State bounding**: State norm is soft-clipped to 5.0 after every evolution step
- **Exponent clamping**: The decay exponent `(-dt/tau)` is clamped to a minimum of -87.0 to prevent f32 underflow (since `exp(-87) approx 1.6e-38 > 0`)
- **Weight norm clipping**: All weight HVs are clipped to norm 2.0 after every learning update
- **Running statistics**: Exponential moving average (alpha=0.01) of state norm for monitoring

---

### 8. Novelty Statement

The following aspects are believed to be new relative to all known prior art:

- **Hypervector state with LTC dynamics**: No prior system uses a high-dimensional hypervector as the state of a liquid time-constant neuron. Prior LTC/CfC neurons use scalar or low-dimensional vector states with weight matrices.

- **HDC binding replacing weight matrices**: The use of element-wise binding (multiplication) between weight hypervectors and state/input hypervectors to replace matrix multiplication reduces parameters from O(D^2) to O(D) per weight component while preserving the near-orthogonality properties of HDC.

- **Closed-form solution for HDC-state ODEs**: The derivation and implementation of an exact analytical solution `x(t+dt) = x_inf + (x(t) - x_inf) * exp(-dt/tau)` where x, x_inf are hypervectors of dimension D, enabling O(1) temporal jumps independent of dt.

- **State-dependent time constant from hypervector norm**: The formula `tau(x) = tau_0 * (1 + backbone * ||x||) * (1 + 0.2 * sim(u, T_mod))` where ||x|| is the L2 norm of the D-dimensional state hypervector, providing adaptive computational pacing based on representational complexity.

- **Fused SIMD evolution kernel**: A single-pass AVX2+FMA kernel that computes binding, bundling, rational-approximation tanh, and temporal interpolation without any intermediate memory allocations, saving 256 KB per invocation for D=16,384.

- **Rational fast_tanh approximation for SIMD**: The specific formula `tanh(x) approx x * (27 + x^2) / (27 + 9*x^2)` chosen for its pure-arithmetic nature enabling auto-vectorization and direct SIMD mapping.

- **Fourier basis injection into HDC equilibrium**: Bundling time-varying sinusoidal signals (distributed across dimensions via strided placement) as an additional component in the equilibrium computation, providing absolute time encoding without external clock inputs.

- **BPTT through closed-form HDC step**: Analytical gradient computation exploiting the fact that HDC binding is element-wise multiplication, yielding the simple chain rule `d(A bind B)/dA = B` element-wise.

- **Gram-Schmidt orthogonal initialization of internal HVs**: Initializing all 5 internal hypervectors (W, U, T_mod, G_w, G_b) via modified Gram-Schmidt to ensure pairwise similarity < 0.02, minimizing initialization interference.

---

### 9. Suggested Claims

**Claim 1 (independent):** A method for neural processing comprising: (a) maintaining a neuron state as a continuous-valued hypervector of dimension D; (b) computing an equilibrium hypervector by applying hyperdimensional binding operations between weight hypervectors and the state hypervector, bundling the result with a bound input hypervector, and applying a nonlinear activation function; (c) computing a state-dependent time constant based on the L2 norm of the state hypervector; and (d) updating the state hypervector using a closed-form exponential decay solution that interpolates between the current state and the equilibrium in O(D) operations independent of the time step magnitude.

**Claim 2 (independent):** A neural processing unit comprising: a state register storing a continuous-valued hypervector of dimension D; a weight hypervector of dimension D; an input mask hypervector of dimension D; and processing logic configured to evolve the state hypervector by computing element-wise binding between the weight hypervector and the state hypervector, computing element-wise binding between the input mask hypervector and an input hypervector, computing a normalized sum of the binding results, applying an activation function, and interpolating between the current state and the activated result using an exponential decay factor dependent on a state-derived time constant.

**Claim 3 (independent):** A computer-implemented method for temporal sequence processing comprising: representing each neuron state as a D-dimensional hypervector where D is at least 1,000; replacing weight matrix multiplications with element-wise hyperdimensional binding operations between weight hypervectors and operand hypervectors; computing temporal state updates using the closed-form solution x(t+dt) = x_inf + (x(t) - x_inf) * exp(-dt / tau(||x||)); and performing said computation in a single SIMD-fused pass through the dimension that eliminates all intermediate hypervector allocations.

**Claim 16 (independent, broad):** A method for neural computation comprising: (a) maintaining a neuron state as a vector of dimension D, where D is at least 100; (b) computing an equilibrium vector by applying element-wise operations between learned weight vectors and the state vector, replacing matrix-vector multiplication with O(D) element-wise operations; (c) deriving a time constant from properties of the state vector; and (d) updating the state vector toward the equilibrium according to continuous-time dynamics governed by the derived time constant.

**Claim 17 (independent, broad):** A computer-implemented method for temporal sequence processing in a neural network, comprising: (a) representing neuron states as high-dimensional vectors; (b) evolving each neuron state toward an equilibrium using a closed-form analytical solution that permits arbitrary-length temporal jumps in O(D) time independent of the jump duration; (c) computing the equilibrium via element-wise vector operations that achieve O(D) parameter scaling per weight component; and (d) training the network via backpropagation through the closed-form step using element-wise gradient identities.

**Claim 4 (dependent on 1):** The method of claim 1, wherein the state-dependent time constant is computed as: tau(x, u) = tau_0 * (1 + backbone * ||x||) * (1 + alpha * sim(u, T_mod)), where tau_0 is a base time constant, backbone is a scaling factor, ||x|| is the L2 norm of the state hypervector, sim denotes cosine similarity, u is the input hypervector, T_mod is a learned tau modulator hypervector, and alpha is a constant (e.g., 0.2).

**Claim 5 (dependent on 1):** The method of claim 1, further comprising computing an adaptive gating factor sigma by: bundling the state and input hypervectors; computing cosine similarity between the bundle and a gate weight hypervector; applying a sigmoid function with configurable steepness; and combining the sigmoid output with an exponential decay factor as sigma = 1 - exp(-dt/tau) * (1 - sigma_base).

**Claim 6 (dependent on 3):** The method of claim 3, wherein the SIMD-fused pass uses AVX2 256-bit registers with FMA instructions to: load 8 elements of state, weight, input_mask, and input per cycle; compute binding and bundling via fused multiply-add; compute a rational tanh approximation as x*(27+x^2)/(27+9*x^2); and compute the temporal interpolation via fused multiply-add, all without writing any intermediate hypervectors to memory.

**Claim 7 (dependent on 1):** The method of claim 1, further comprising injecting a Fourier basis hypervector into the equilibrium computation by: for each of K configured frequencies f_k, computing sin(2*pi*f_k*t) and cos(2*pi*f_k*t) scaled by an amplitude parameter; distributing the sin and cos values across the D dimensions using a strided pattern with stride 2K; and bundling the Fourier basis hypervector as an additional component in the equilibrium computation.

**Claim 8 (dependent on 1):** The method of claim 1, further comprising performing backpropagation through the closed-form temporal step by: computing a loss gradient with respect to the updated state; propagating the gradient through the interpolation to obtain a gradient on the equilibrium; propagating through the activation function using the element-wise derivative; and computing weight gradients using the property that the derivative of element-wise binding A*B with respect to A is B.

**Claim 9 (dependent on 2):** The neural processing unit of claim 2, wherein the weight hypervector, input mask hypervector, a tau modulator hypervector, a gate weight hypervector, and a gate bias hypervector are initialized via Gram-Schmidt orthogonalization to have pairwise cosine similarity below 0.02.

**Claim 10 (dependent on 1):** The method of claim 1, further comprising applying a Hebbian learning rule by: computing a correlation hypervector as the element-wise binding of the input hypervector and the state hypervector; updating a momentum accumulator as M = beta * M + lr * correlation; and updating the weight hypervector as W = (1 - decay) * W + M, with norm clipping to a maximum of 2.0.

**Claim 11 (dependent on 1):** The method of claim 1, further comprising applying a spike-timing dependent plasticity (STDP) rule by: computing a timing-dependent weight change delta_w using exponential windows with time constants tau_plus and tau_minus; computing a correlation hypervector as the element-wise binding of pre-synaptic and post-synaptic hypervectors; and updating the weight hypervector by the product of delta_w and the correlation, modulated by a learning rate.

**Claim 12 (dependent on 1):** The method of claim 1, wherein the dimension D is at least 8,192 and the hyperdimensional binding operation is element-wise multiplication of f32 values, such that the parameter count per weight component is O(D) rather than O(D^2).

**Claim 13 (dependent on 3):** The method of claim 3, wherein the rational tanh approximation computes tanh(x) as x*(27+x^2)/(27+9*x^2) for |x| < 4.97 and as signum(x) for |x| >= 4.97, achieving maximum error of approximately 0.004 while enabling SIMD vectorization.

**Claim 14 (dependent on 1):** The method of claim 1, further comprising organizing multiple neuron instances into a layered network with inter-layer binding vectors, wherein the output of each layer is computed as the mean of its constituent neuron states, and inter-layer communication is mediated by binding the previous layer's output with a layer-specific binding hypervector.

**Claim 15 (dependent on 1):** The method of claim 1, wherein the neuron supports multiple evolution modes including: Euler integration; fourth-order Runge-Kutta integration; single-step gated closed-form evolution; zero-allocation fused closed-form evolution; pure analytical exponential decay; and iterative closed-form sub-stepping with sub-step size of tau/10.

---

### 10. Experimental Validation

All performance figures are from the implemented system running the Symthaea cognitive pipeline.

#### 10.1 Cycle Performance

- **Full cognitive loop cycle time**: 4.3ms in release mode (234 Hz), exceeding the 50 Hz real-time target by 4.7x
- **Cycle with hypervector input (non-text)**: 2.0ms/cycle (500 Hz)
- **Warm word encoding**: 97 ns
- **Sentence encoding**: 379 us for 10 words (only 12% of cycle time)

#### 10.2 Memory Efficiency

- **Parameter count per weight component**: O(D) = 16,384 f32 values = 64 KB
- **Equivalent matrix parameterization**: O(D^2) = 268,435,456 f32 values = 1 GB
- **Compression ratio**: 16,384x fewer parameters per weight component
- **SIMD fused kernel savings**: 256 KB eliminated per evolution invocation (4 intermediate HV allocations removed)

#### 10.3 Test Coverage

- **Tests in hdc_ltc_unified.rs**: 29 tests, all passing
- **Total project tests**: 21,516+ passing (workspace-wide, tokei-verified 2026-03-13)
- **Specific validations**:
  - Closed-form vs Euler convergence: similarity > 0.5 after 1s evolution
  - O(1) property: 0.1s and 100.0s jumps both produce bounded, valid states
  - State-dependent tau: measurable tau change after 50 evolution steps
  - HDC binding dissimilarity: binding output < 0.3 similarity to either operand
  - Gating monotonicity: larger dt produces larger sigma
  - Orthogonal initialization: pairwise similarity < 0.02 for all 5 internal HVs
  - Numerical stability: no NaN/Inf after extreme dt=100.0 with tau=0.01
  - Fourier injection: measurably different trajectories with vs without Fourier basis
  - Genesis determinism: bit-identical reconstructions from same seed

#### 10.4 Broader System Validation

- **Moral algebra accuracy**: 91.1% on ethical dilemma classification
- **LibriSpeech benchmark**: 94.5% accuracy
- **ISOLET benchmark**: 91.7% accuracy
- **Psych-bench qualia confidence**: 0.683 composite (MODERATE), 7/7 predictions met
- **Voice synthesis**: Mean formant error 4.4 Hz (LTC-controlled), MCD 0.02 dB (against own training targets; standard MCD vs natural speech: 4.03 dB)

---

### 11. Key Source Files

| File | Description | LOC |
|------|-------------|-----|
| `symthaea-core/src/hdc/hdc_ltc_unified.rs` | Core neuron and network implementation | ~2,403 |
| `symthaea-core/src/hdc/unified_hv.rs` | ContinuousHV type and HDC operations | -- |
| `symthaea-core/src/hdc/mod.rs` | HDC module root, dimension constants (16,384) | -- |
| `symthaea/src/cognitive_loop/cycle.rs` | Core cognitive pipeline using HDC-LTC neurons | -- |
| `symthaea-core/src/genesis/mod.rs` | GenesisSeed deterministic initialization | -- |


---

### 12. Closest Prior Art References

1. Hasani, R., Lechner, M., Amini, A., et al. (2021). "Liquid Time-constant Networks." *AAAI Conference on Artificial Intelligence*. -- Introduces LTC neurons with input-dependent time constants and closed-form CfC solutions, but uses weight matrices (O(D^2)), not HDC binding.

2. Lechner, M., Hasani, R., et al. (2020). "Neural Circuit Policies Enabling Auditable Autonomy." *Nature Machine Intelligence*, 2(10), 642-652. -- Original LTC neuron formulation with ODE-based evolution; no closed-form solution, no HDC operations.

3. Kanerva, P. (2009). "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors." *Cognitive Computation*, 1(2), 139-159. -- Foundational HDC framework with binding/bundling operations, but no temporal dynamics or ODE evolution.

4. Rahimi, A., Kanerva, P., Rabaey, J.M. (2017). "A Robust and Energy-Efficient Classifier Using Brain-Inspired Hyperdimensional Computing." *ISLPED*. -- HDC classification; static, no temporal dynamics.

5. Hochreiter, S. and Schmidhuber, J. (1997). "Long Short-Term Memory." *Neural Computation*, 9(8), 1735-1780. -- LSTM architecture with O(D^2) parameters, no closed-form temporal jumps, no HDC operations.

6. Cho, K., et al. (2014). "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation." *EMNLP*. -- GRU architecture; same limitations as LSTM relative to this invention.

7. Ge, L. and Parhi, K.K. (2020). "Classification using Hyperdimensional Computing: A Review." *IEEE Circuits and Systems Magazine*. -- Survey of HDC classification methods; no temporal/ODE components.

8. Thomas, A., et al. (2021). "Theoretical Foundations of Hyperdimensional Computing." *Journal of Artificial Intelligence Research*. -- Theoretical analysis of HDC capacity and operations; no connection to continuous-time neural dynamics.

9. US Patent 11,537,868 (Hasani et al.) -- "Liquid time-constant networks" -- Covers LTC/CfC with conventional weight matrices; does not disclose HDC binding as a replacement for matrix multiplication.

---

### 13. Figures (Text Descriptions)

**Figure 1: Unified HDC-LTC Neuron Architecture Diagram**

A block diagram showing the internal architecture of a single `HdcLtcUnifiedNeuron`. At center, a large box labeled "State x (16,384D ContinuousHV)". Inputs flow from the left: "Input u (16,384D)" passes through a binding operation (circle with x symbol) with "Input Mask U (16,384D)" to produce "Masked Input". The state passes through a binding operation with "Weight HV W (16,384D)" to produce "Transformed State". Both results enter a bundling operation (circle with + symbol) labeled "Normalized Sum", followed by an activation function block f(). The output is labeled "x_inf (equilibrium)". A separate path shows "State Norm ||x||" feeding into a "Tau Computation" block that also receives similarity from the input and Tau Modulator. The tau value feeds into an "Exponential Decay / Gating" block that also receives the gating signal from Gate Weight and Gate Bias operating on a bundle of state and input. The final output arrow shows "sigma * x_inf + (1-sigma) * x" feeding back to the state. An optional dashed path shows "Fourier Basis F(t)" entering the bundling operation as a third component.

**Figure 2: Closed-Form Temporal Jump Illustration**

A time-axis diagram comparing three evolution strategies. The x-axis represents time from t=0 to t=T. Three rows:
- **Top row (Euler)**: Many small arrows stepping from x(0) toward x_inf, each requiring one O(D) computation. Total: N steps * O(D).
- **Middle row (RK4)**: Fewer but larger arrows, each with 4 sub-evaluations shown as dotted lines. Total: N/4 steps * O(4D).
- **Bottom row (Closed-Form)**: A single curved arrow from x(0) directly to x(T), labeled "O(D), ONE STEP". The exponential decay curve `x_inf + (x(0) - x_inf) * exp(-T/tau)` is shown as a dashed line. A callout box emphasizes: "Same cost for dt=0.001s and dt=100s."

**Figure 3: Parameter Efficiency Comparison Chart**

A bar chart comparing parameter counts for a single neuron layer of dimension D=16,384:
- **LSTM**: 4 * D^2 + 4 * D = ~1.07 billion parameters (tall bar, log scale)
- **GRU**: 3 * D^2 + 3 * D = ~805 million parameters
- **LTC/CfC**: D^2 + D = ~268 million parameters
- **HDC-LTC Unified (this invention)**: 5 * D = ~82,000 parameters (dramatically shorter bar)
A secondary annotation shows the ratio: "16,384x fewer parameters than LTC/CfC per weight component."

**Figure 4: SIMD Fused Kernel Data Flow**

A pipeline diagram showing the data flow through the `fused_tanh_avx2` kernel for one 8-element chunk. Eight parallel lanes are shown (one per f32 element in a 256-bit AVX2 register). The pipeline stages are:
1. **LOAD**: Four `_mm256_loadu_ps` operations load state[i], weight[i], input_mask[i], input[i]
2. **BIND+BUNDLE**: `_mm256_mul_ps(w, s)` computes W*x, then `_mm256_fmadd_ps(m, inp, ws)` fuses U*u + W*x, then `_mm256_mul_ps(..., scale)` applies the 0.5 bundling normalization
3. **FAST_TANH**: Three arithmetic operations: `x^2`, `x*(27+x^2)`, `/(27+9*x^2)`, followed by `_mm256_blendv_ps` for the |x|>4.97 clamping
4. **LERP**: `_mm256_fmadd_ps(sigma, x_inf, _mm256_mul_ps(oms, s))` computes the final interpolation
5. **STORE**: `_mm256_storeu_ps` writes result back to state

A callout notes: "Zero intermediate allocations. 256 KB saved per call for D=16,384."

---

