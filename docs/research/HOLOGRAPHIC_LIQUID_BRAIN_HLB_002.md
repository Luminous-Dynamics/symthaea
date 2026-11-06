# HLB-002: Holographic Liquid Brain — HDC/CfC Neuron Audit and Research Programme

**Status:** Architecture audit, canonical dilation fix, an in-branch HDC-LTC one-step gradient correction, and a typed HDC-LTC parameter-introspection/edit API on branch research/hlb-002-holographic-liquid-neuron.  
**Base:** main.  
**Scope:** HDC/CfC neuron mathematics, holographic binding and memory, multi-neuron liquid networks, numerical integration quality, and evidence-qualified HLB evaluation.  
**Evidence status:** repository code audit and published-literature review; the new Rust regression test has been added but has not been run in this environment.

---

## 1. Executive assessment

The HDC/CfC neuron is worth developing. It combines a real representational primitive (continuous hypervectors with binding, bundling, similarity, and permutation) with solver-free, time-dependent state evolution. It is a useful experimental architecture and has a distinct design space from ordinary dense recurrent networks.

However, the current implementation should not be treated as a validated replacement for general recurrent or continuous-time neural networks yet. Three distinctions matter:

1. **HDC binding is not arbitrary recurrent mixing.** In ContinuousHV, bind is element-wise/Hadamard multiplication. A single learned weight hypervector therefore performs a diagonal coordinate-wise transformation. It is O(D), but it does not mix arbitrary coordinates. Cross-coordinate interaction comes mainly from global reductions (norms, cosine similarities, scalar gates), explicit permutations, and the layer structure.
2. **The implemented closed-form step is an approximation.** It computes an equilibrium under the current state/input, calculates a scalar interpolation factor, and applies one O(D) state update. Its work is independent of the elapsed time interval, but it is still O(D) in hypervector dimension. It is not an exact closed-form solution of a fully nonlinear state-dependent ODE whose equilibrium changes continuously during the interval.
3. **The current backward pass does not match the current forward pass.** Forward evolution uses a learned sigmoid gate, state/input-dependent time constant, and optional Fourier forcing. backward() reconstructs a simpler ungated step, omits the Fourier term, and returns no gradients for the learned gate vectors. As a result, the previous analytic gradients were not the derivatives of the actual forward operation. A candidate correction has now been added on this branch; it must pass the finite-difference tests below before it can be considered validated.

The recommended direction is not to discard HDC/CfC. It is to build a more expressive, order-aware, numerically characterized **HoloLiquid v2** with correct gradients, structured cross-coordinate mixing, adaptive input interpolation, stable multi-timescale dynamics, and a reproducible benchmark suite.

## 2. What exists in the repository today

### 2.1 Unified HDC-LTC neuron

Primary file: crates/core/symthaea-core/src/hdc/hdc_ltc_unified.rs

The neuron state and internal weight-like parameters are continuous hypervectors:
- state
- weight_hv
- input_mask
- tau_modulator
- gate_weight
- gate_bias

The current equilibrium is computed from:
- weight_hv.bind(state);
- input_mask.bind(input);
- optional Fourier basis injection;
- element-wise activation.

The effective time constant is:
\[
\tau = \operatorname{clip}\left(
\tau_0(1 + \beta\|x\|)(1 + 0.2\,\operatorname{sim}(u,m_\tau)),
0.01, 10
\right).
\]

The gate computes a scalar cosine similarity between the state/input bundle and gate_weight, adds the mean of gate_bias, passes it through a sigmoid, and mixes it with the time-decay term.

The state update is one vector interpolation:
\[
x' = (1-\sigma)x + \sigma x_\infty.
\]

This costs O(D) per update and has no numerical ODE substeps as dt grows. **The O(1) claim applies to the number of temporal integration steps, not to total work: the implementation still scans or vectorizes the D-dimensional state.**

### 2.2 HDC algebra: use the characterized semantics

Primary file: crates/core/symthaea-core/src/hdc/unified_hv.rs  
Audit record: docs/BINDING_ALGEBRA_CHARACTERIZATION_REPORT.md and the binding algebra tests.

The repository already corrected important earlier claims:
- ContinuousHV.bind is commutative Hadamard multiplication.
- For real random components distributed uniformly over [-1, 1], binding is not self-inverse.
- Actual unbinding requires inverse(), which has a documented numerical-stability problem because reciprocal components near zero can become extremely large.
- bind_temporal() adds a permutation before binding, making the operation order-sensitive.
- BinaryHV offers a different algebra (XOR) when true self-inverse binding is required.

This is good scientific practice: HLB work should build on the measured algebra, not textbook assumptions from bipolar vectors that do not match ContinuousHV.

### 2.3 Complex CfC neuron

Primary file: crates/core/symthaea-core/src/hdc/complex_cfc_neuron.rs

The complex variant represents 16,384 real values as 8,192 complex pairs. Each complex coordinate has a learned decay eigenvalue and oscillatory frequency. This provides useful oscillatory modes and stable exponential decay when the real eigenvalues remain negative.

It is still block-diagonal at the complex-pair level: each pair evolves independently apart from any global input/state interactions. It is a useful baseline for HoloLiquid v2, not yet a general cross-coordinate holographic operator.

### 2.4 Multi-neuron network

HdcLtcUnifiedNetwork stores layers of neurons and layer-binding hypervectors. Within a layer, neurons are evolved independently on a shared layer input; the layer output is the element-wise mean of their states. Deeper layers receive the previous layer's mean, optionally bound with a layer vector and optionally skip-connected with the original input.

This is a parallel ensemble/mean-pooling architecture with layer-wise transforms, not a fully connected recurrent graph. That may be an intentional efficiency trade-off, but its consequences for capacity, specialization, and cross-neuron communication must be measured.

### 2.5 LiquidHolocell

Primary file: crates/core/symthaea-core/src/hdc/liquid_holocell.rs

LiquidHolocell is a simpler stateful cell with a scalar tau and an exponential interpolation update. It is not the same model as HdcLtcUnifiedNeuron: it does not have the latter's learned state/input transforms, gate, or adaptive tau modulator.

It should either remain explicitly a lightweight memory/decay primitive or eventually delegate to a shared liquid-state contract. Avoid treating the two classes as interchangeable merely because both contain an HDC state and tau.

**Correctness fix on this branch:** LiquidHolocell::dilate() previously implemented a second, subtly different folding algorithm. It did not inverse-permute chunks during downsampling. It now delegates to ContinuousHV::dilate(), the canonical implementation that undoes the rotations before folding. The downsampling test now checks exact recovery after an upsample/downsample round-trip, not just the output dimension. This test has been added but not executed here.

---

## 3. P0 mathematical audit: make backward equal the forward operation

### 3.1 Mismatch found

Forward evolution uses:
- equilibrium with two or three bundled terms (state, input, optional Fourier basis);
- a state/input-dependent time constant;
- a sigmoid gate from gate_weight and gate_bias;
- the time-scaled interpolation factor;
- optional state norm clipping.

The current backward():
- recomputes only the two-term equilibrium;
- omits the Fourier basis;
- computes sigma = 1 - exp(-dt/tau) and does not reproduce the sigmoid gate;
- does not include the correct derivative of the gate in the tau derivative;
- returns gradients for weight_hv, input_mask, a scalar tau term, and input, but not for gate_weight or gate_bias;
- uses a comment that says bundling's 1/n factor is absorbed into the learning rate, while the method does not adjust that learning rate.

Thus, even with a correct element-wise derivative for binding, the returned gradient is not generally the gradient of evolve_closed_form().

### 3.2 Required correction

Factor the real forward computation into reusable components so forward and backward cannot silently diverge:
- equilibrium pre-activation and number of bundled terms;
- raw and clamped tau;
- gate activation and sigmoid value;
- decay and final sigma;
- state-bounds status.

Differentiate the actual update, including:
- the bundle factor (1/2 or 1/3);
- gate weight and gate bias;
- tau modulator through the actual cosine similarity;
- state and input contributions to both tau and the gate;
- Fourier forcing as a constant term in the equilibrium;
- the chain rule through the clamp in regimes where tau is clamped (zero derivative through the clipped branch).

Add d_gate_weight, d_gate_bias, and a vector d_tau_modulator to the gradient object. Preserve dtau_scalar for diagnostics only if a scalar summary is useful. Add d_state if temporal credit assignment is intended, and clearly separate one-step gradients from sequence-level BPTT.

The initial finite-difference test must compare analytic and central-difference gradients for at least one coordinate in every parameter group:
- weight hypervector;
- input mask;
- tau modulator;
- gate weight;
- gate bias.

Test both with Fourier forcing disabled and enabled. Keep state norms below the clipping threshold during the gradient check, and test the clipped regime separately with an explicitly defined subgradient policy.

**In-branch correction (not yet validated):** the updated backward() now recomputes the actual 2-/3-term equilibrium, the full-resolution input/state cosines, the sigmoid gate, time-scaled interpolation, tau clamp, optional Fourier basis, and the state-norm projection. The gradient object now contains d_tau_modulator, d_gate_weight, d_gate_bias, d_input, and d_state in addition to dW/dU and the scalar dL/dtau diagnostic. apply_gradients_inner() now applies the previously omitted tau/gate parameter gradients and bounds those groups. The updated full-resolution tau cosine also prevents an unrelated global selective-scan stride from changing the gradient's forward semantics.

Finite-difference regression tests have been added for seven directions (W, U, tau modulator, gate weight, gate bias, state, input), both with Fourier forcing disabled and enabled, plus a separate check through the radial state-bound projection. **These tests have not been executed.** They are not evidence of a passing gradient audit until the exact branch head is compiled and the tests pass.

Acceptance:
- relative directional-gradient error below 1e-3 for small, well-conditioned test cases (target; report actual distribution);
- relative error below 1e-2 across a randomized multi-seed suite after tolerance calibration;
- zero unexplained discrepancy between the forward implementation and the forward recomputation used by backward;
- no silently omitted trainable parameter groups.

### 3.3 Distinguish local credit assignment from full temporal BPTT

A one-step backward pass is not, by itself, full BPTT through a temporal sequence. The current repository has additional training/controller logic that converts layer outputs and d_input into local targets. That can be a useful local/teacher-forced learning rule, but papers and API names should distinguish:
- exact gradient of one closed-form transition;
- cross-layer propagated input gradients;
- approximate local target propagation;
- full sequence-level BPTT with stored/recomputed state history.

Do not call these equivalent.

---

## 4. Paradigm shifts for HoloLiquid v2

### Shift A — from diagonal binding to structured holographic operators

Current W ⊙ x and U ⊙ u are diagonal transforms. A richer HDC liquid neuron should retain O(D) or O(D log D) compute while enabling cross-coordinate interaction.

Candidate operator family:
1. **Permutation mixtures:** A(x) = Σ_j α_j P_j(W_j ⊙ x), using a small number of fixed or learned permutations.
2. **Circulant/FFT operators:** use Fourier-domain diagonal operators for O(D log D) structured mixing.
3. **Unitary or orthogonal transforms:** preserve energy/norm in the conservative component and support invertible role/filler transformations.
4. **Generalized Holographic Reduced Representations (GHRR):** test non-commutative binding for order-sensitive sequences and directed relations, with structured/vector-space implementations rather than naive dense matrices.

Do not replace the existing operation in place. Make the operator family configurable so the benchmark can compare:
- current Hadamard binding;
- bind_temporal;
- permutation mixtures;
- complex-pair block operators;
- structured unitary/circulant mixing;
- GHRR-style non-commutative binding.

The 2026 DATE paper on scalable symbolic reasoning with matrix-based brain-inspired representations is directly relevant: it argues that commutative binding needs permutation logic for order/direction and studies non-commutative GHRR with a vector-space execution strategy.

### Shift B — from zero-order input hold to interpolation-aware continuous time

The September 2026 IEEE TNNLS paper Rederived Closed-Form Continuous-Time Neural Networks identifies error in the standard zero-order input approximation for high-resolution and irregularly sampled data. It proposes Lagrange interpolation-based integration, derives error bounds, and evaluates DFA-CfN / PRDFA-CfN variants on six sequence tasks.

HLB should directly benchmark this issue. Add an input interpolation mode:
- zero-order hold (current baseline);
- linear interpolation between consecutive inputs;
- optional higher-order interpolation only if its numerical benefit survives complexity analysis.

For constant tau and linearly interpolated equilibrium x_inf(t) = a + b t, the linear first-order state equation has the analytic solution:
\[
x(t+\Delta t) = e^{-\Delta t/\tau}x(t)
+ a(1-e^{-\Delta t/\tau})
+ b\left[\Delta t - \tau(1-e^{-\Delta t/\tau})\right].
\]
This provides a useful structured reference case. A nonlinear equilibrium still requires approximation or additional structure; do not claim the formula solves the fully nonlinear problem exactly.

### Shift C — dissipative plus conservative liquid dynamics

Investigate an operator generator of the form:
\[
A = -D + S,
\]
where D is positive semidefinite (damping) and S is skew-symmetric / skew-Hermitian (energy-preserving rotation).

This separates:
- forgetting/decay;
- reversible temporal transformation;
- memory retention;
- learned input forcing.

A structured complex-pair or Fourier/circulant realization may retain a solver-free transition while allowing richer dynamics. Compare stability and expressivity against the existing per-coordinate complex CfC neuron.

### Shift D — explicit multi-timescale memory

Use separate fast, medium, and slow dynamical channels with distinct, positive time constants. Bind their states with role tags or a learned readout rather than averaging them indiscriminately.

Candidate decomposition:
- fast channel: event/sensory transients;
- medium channel: working context and task state;
- slow channel: stable semantic/episodic context;
- associative memory: content-addressable, explicitly capacity-tested HDC store.

Measure interference, retrieval, forgetting, adaptation speed, and power/compute cost. Do not equate repeated/tiled coordinates with extra independent memory capacity.

### Shift E — from static layer means to typed communication

Replace implicit mean pooling with an explicit, benchmarkable communication strategy:
- mean pooling baseline;
- learned scalar/vector gates;
- sparse top-k messages;
- attention or associative retrieval;
- role/filler binding and unbinding;
- event-triggered updates.

The goal is not a giant dense matrix. The goal is selective, interpretable communication with controllable O(D), O(kD), or O(D log D) cost.

### Shift F — event-driven and resource-aware evolution

Investigate whether low-surprise, low-change state regions can skip some computation while preserving error bounds. A resource-aware policy should be driven by measured state change, uncertainty, and the required downstream precision—not by an unvalidated consciousness proxy.

---

## 5. Current SOTA and relevant comparisons

### Continuous-time models
- Hasani et al., Closed-form continuous-time neural networks, Nature Machine Intelligence (2022): establishes CfC's solver-free approximate temporal update and strong efficiency trade-offs.
- Liang et al., Rederived Closed-Form Continuous-Time Neural Networks, IEEE TNNLS 37(9), 4381–4395 (2026): NIALIM/Lagrange interpolation addresses the zero-order input approximation in high-resolution and irregular sampling; DFA-CfN and PRDFA-CfN report results across six task families.
- Qian et al., A liquid neural network with physical evolution for variable continuous time series prediction, Journal of Computational Science 94 (2026): HCFC combines synaptic dynamics and Hamiltonian evolution and reports irregular-time-series comparisons.
- Banerjee, Xu, and Gupta, *Efficient Continuous-Depth Modeling with GRU Equivalents*, ICML 2026, PMLR 306, 6194–6209: CoDA uses Mori–Zwanzig/Koopman ideas to replace repeated nonlinear ODE layers with a compact GRU, one low-dimensional linear ODE, and a dense layer. The paper reports 6.7× training and 1.8× inference speedups without accuracy loss on LFM-1.2B. HLB should include continuous-depth compression as a baseline, not assume a liquid neuron is automatically more efficient.
- Standard baselines should include LTC (with high-accuracy ODE solver), CfC, ODE-RNN, GRU-D, Neural CDE, and strong state-space sequence models. Use matched parameter and compute budgets.

### HDC and holographic memory
- Salik et al., Hyper-dimensional computing architectures for AI applications: a comprehensive survey of principles, performance, and future prospects, Results in Engineering (March 2026).
- Zhou et al., SynapseHD: A unified training framework for bridging spiking neural networks and hyperdimensional computing, Neurocomputing 659 (2026): combines SNN feature extraction with HDC memory and an unbalanced co-training strategy for few-shot class-incremental learning.
- Scalable Symbolic Reasoning with Matrix-Based Brain-Inspired Representations and Vector-Space Acceleration, DATE 2026: GHRR-style non-commutative binding for sequence/directed-structure encoding and efficient vector-space execution.
- Existing HDC/VSA baselines: binary XOR binding, real-valued Hadamard binding, permutation binding, complex HRR, and GHRR.

These results are not directly comparable on a single accuracy leaderboard. The benchmark must compare like-for-like datasets, splits, memory budgets, compute, and online-update rules.

---

## 6. HLB benchmark suite

### A. Algebra and holographic memory
- binding/unbinding round-trip similarity;
- self-inverse correctness for binary vs continuous variants;
- order sensitivity and directionality;
- sequence and graph retrieval accuracy;
- capacity vs number of superposed items;
- crosstalk and cleanup success under controlled noise;
- graceful degradation vs dimension, sequence length, and interference;
- norm stability and inverse amplification.

### B. Continuous-time numerical fidelity
- one-step local error vs high-accuracy ODE reference;
- accumulated trajectory error across long horizons;
- error vs dt, including sparse and irregular intervals;
- zero-order vs linear/NIALIM input interpolation;
- stiffness and abrupt input transitions;
- stability under very large dt;
- derivative/gradient agreement with finite differences.

### C. Temporal learning
- long-range recall and order reconstruction;
- variable-duration event prediction;
- irregularly sampled sequence classification;
- online few-shot adaptation;
- class-incremental learning and catastrophic forgetting;
- time extrapolation beyond the training interval distribution;
- transfer to new sensors/modalities.

### D. Structured mixing
- diagonal vs permutation mixture vs block-circulant vs complex vs GHRR;
- accuracy and retrieval vs parameter count;
- quality vs O(D), O(kD), O(D log D) cost;
- sensitivity to coordinate permutation and basis changes;
- robustness to noise and partial erasure.

### E. HLB system-level tasks
- memory-grounded multi-step reasoning;
- belief updating under delayed/conflicting evidence;
- temporal planning under variable-duration actions;
- cross-modal binding and sequence recall;
- persistent memory with retrieval precision/recall;
- calibration and abstention;
- task success and resource use.

Do not use Φ, consciousness scores, or self-reported experience as the main performance target. If they are logged, treat them as exploratory measurements separate from capability and safety evaluation.

### F. Efficiency and deployment
- latency p50/p95/p99;
- throughput at D=1K, 4K, 16K, 64K;
- memory allocation / bytes moved;
- FLOPs or hardware counters where available;
- energy per transition on CPU, GPU, and embedded target;
- quantized and reduced-precision robustness;
- scaling with dt separately from scaling with D.

---

## 7. Preregistered initial targets

These are engineering targets, not results:
- finite-difference gradient relative error: <1e-3 in well-conditioned unit tests, and <1e-2 across randomized characterization cases;
- stable trajectories across a preregistered dt grid with no unbounded state growth;
- exact dilation round-trip for canonical divisible dimensions where the underlying embedding/folding contract defines an inverse;
- explicit error curves for zero-order vs linear input interpolation;
- sequence retrieval and capacity curves, not one cherry-picked memory example;
- report accuracy/quality at matched parameter, latency, and memory budgets;
- any claimed performance advantage must have paired seeds, confidence intervals, and independent reruns.

No SOTA claim until a common benchmark, strong baselines, and reproducible artifacts exist.

---

## 8. Integration with SWI

The HDC/CfC neuron now has an initial first-class SWI parameter surface on this branch: `HdcLtcParameterManifest`, typed group IDs and roles, exact flattened parameter values, BLAKE3 architecture/manifest/parameter/model fingerprints, digest-bound typed group replacement, and transactional batch application. The setter validates exact length, finite values, destination-vector dimensions, and the per-group norm bound before mutation; successful edits reset stale W/U momentum.

This is a first introspection/edit boundary, not a complete proof-carrying evolution system. The tests for manifest layout, fingerprinting, group edits, stale digests, invalid norm, wrong lengths, duplicate targets, batch rollback, optimizer updates across all five groups, and finite-difference gradients have been added but have not been run. The neuron should only become a trusted SWI substrate once the actual gradient and algebraic-invariant tests pass on the exact head.

The HDC-specific SWI manifest records:
- typed group identity and mathematical role;
- dimension, shape, flat offset, and length;
- norm constraints;
- architecture/configuration digest;
- exact little-endian parameter-value digest;
- combined model digest.

The next HDC-LTC manifest version should additionally encode operator family and algebra semantics, binding/unbinding contracts, role/filler or permutation structure, interpolation mode, and any future frequency/eigenvalue ranges.

A generic flat-float edit must not be allowed to violate an algebraic or dynamical invariant. HDC-aware edits should propose operations such as:
- bounded change to a role vector;
- unitary/orthogonal-preserving perturbation;
- stable eigenfrequency or damping adjustment;
- bounded low-rank/circulant operator update;
- tau change within configured limits;
- normalization-preserving vector edit.

The verifier must test both task behavior and algebraic contracts. A more accurate task result is not acceptable if it breaks unbinding, temporal order, numerical stability, or previously stored associations.

---

## 9. Immediate implementation order

### P0 — correctness
1. Execute the canonical holographic dilation round-trip tests.
2. Execute the in-branch HDC-LTC backward/forward finite-difference tests and fix any remaining discrepancies.
3. Expand gradient checks to clamped tau, state-bound projection, and multi-seed directional derivatives.
4. Make the one-step-gradient vs sequence-BPTT distinction explicit in APIs/docs.
5. Add input/tau/dt finite and range checks at the public boundary.

### P1 — characterization
1. Add reproducible algebra tests for continuous, binary, complex, and temporal binding.
2. Add closed-form-vs-reference ODE integration tests.
3. Add irregular-interval and input-interpolation benchmark harness.
4. Add sequence-retrieval/capacity benchmark with noise and crosstalk sweeps.

### P2 — HoloLiquid v2 prototypes
1. Add a configurable permutation-mixture operator.
2. Add a structured complex/circulant mixing baseline.
3. Add dissipative-plus-skew dynamics prototype.
4. Add fast/medium/slow state channels with typed readout.
5. Compare GHRR-style non-commutative binding against existing bind_temporal.

### P3 — SWI and HLB integration
1. Add an HDC-LTC typed parameter manifest.
2. Add digest-bound, invariant-preserving parameter edits.
3. Add independent repair benchmarks.
4. Add replayable receipts and rollback.
5. Only then allow offline multi-round substrate redesign.

---

## 10. References

1. Hasani et al. (2022). Closed-form continuous-time neural networks. Nature Machine Intelligence. https://www.nature.com/articles/s42256-022-00556-7
2. Banerjee, A., Xu, B., and Gupta, S. (ICML 2026). Efficient Continuous-Depth Modeling with GRU Equivalents. PMLR 306, 6194–6209. https://proceedings.mlr.press/v306/banerjee26b.html
3. Liang et al. (2026). Rederived Closed-Form Continuous-Time Neural Networks. IEEE Transactions on Neural Networks and Learning Systems 37(9), 4381–4395. https://doi.org/10.1109/TNNLS.2026.3669345
4. Qian et al. (2026). A liquid neural network with physical evolution for variable continuous time series prediction. Journal of Computational Science 94, 102757. https://doi.org/10.1016/j.jocs.2025.102757
5. Zhou et al. (2026). SynapseHD: A unified training framework for bridging spiking neural networks and hyperdimensional computing. Neurocomputing 659, 131757. https://doi.org/10.1016/j.neucom.2025.131757
6. Salik et al. (2026). Hyper-dimensional computing architectures for AI applications: A comprehensive survey of principles, performance, and future prospects. Results in Engineering 29, 109589. https://doi.org/10.1016/j.rineng.2026.109589
7. (DATE 2026). Scalable Symbolic Reasoning with Matrix-Based Brain-Inspired Representations and Vector-Space Acceleration. https://doi.org/10.23919/DATE69613.2026.11539261

---

**Decision:** preserve and deepen the HDC/CfC direction, but make mathematical fidelity, algebraic correctness, and honest compute/complexity measurements the next milestone. The most promising differentiators are order-aware non-commutative holographic operators, interpolation-aware continuous-time evolution, stable dissipative/conservative dynamics, and multi-timescale associative memory—not simply increasing the hypervector dimension.
