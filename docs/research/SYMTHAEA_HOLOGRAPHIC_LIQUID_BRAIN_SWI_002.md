# SWI-002: Holographic Liquid Brain — Architecture and Benchmark Programme

**Status:** Research specification with first HDC-LTC implementation increments committed on `research/swi-001-weight-space-introspection`.  
**Related work:** [SWI-001 Weight-Space Intelligence](./SYMTHAEA_WEIGHT_SPACE_INTELLIGENCE_SWI_001.md)  
**Purpose:** Turn the HDC/CfC neuron and Holographic Liquid Brain into a mathematically explicit, testable, learnable, and benchmarked computational architecture.

> **Claim boundary:** “Holographic Liquid Brain” is the name of a research architecture combining vector-symbolic representations, continuous-time state dynamics, associative memory, and learned cross-dimensional coupling. It is not a claim that the current implementation reproduces a biological brain, realizes a physical hologram, or demonstrates consciousness.

---

## 1. Executive assessment

The HDC/CfC combination is worth pursuing. It offers a distinctive computational decomposition:

- high-dimensional distributed state and compositional representations;
- state-dependent continuous-time dynamics;
- multiple temporal scales;
- local and associative learning;
- potential structured transformations that are cheaper or more inspectable than dense recurrent matrices.

However, three architectural limitations should define the next research cycle:

1. **The repository contains two HDC-LTC implementations with different temporal parameterizations.** The unified core neuron uses a scalar adaptive time constant and scalar interpolation gate; the standalone `symthaea-hdc-ltc` neuron has per-coordinate time constants and gates. They should be treated as separate variants until a shared specification and equivalence tests exist.
2. **The core neuron’s current HDC transformations are predominantly coordinate-wise.** Hadamard binding and elementwise activations do not, by themselves, create rich cross-coordinate interactions. Global norms, cosine gates, bundling and layer operations couple the system only through a limited set of aggregate signals.
3. **The one-step closed-form update is an approximation for a state-dependent nonlinear equilibrium.** When state, gate, or input changes materially over a time interval, freezing the equilibrium/gate for one interpolation step creates numerical error. The core validation code already measures this; the next step is error-controlled integration, not declaring the formula exact.

The recommendation is to evolve the system into a **Holographic Liquid Substrate (HLS)** with distinct representation, dynamical, mixing, memory and learning layers. The Holographic Liquid Brain is the larger architecture composed from these components.

## 2. What currently exists in Symthaea

### 2.1 Core unified HDC-LTC neuron

File: `crates/core/symthaea-core/src/hdc/hdc_ltc_unified.rs`

The current core neuron stores:
- `state`: a continuous hypervector, normally 16,384 dimensions;
- `weight_hv`: a Hadamard state-transform vector;
- `input_mask`: a Hadamard input-transform vector;
- `tau_modulator`: a vector that modulates the effective scalar time constant through cosine similarity;
- `gate_weight` and `gate_bias`: vectors controlling an adaptive scalar interpolation gate;
- momentum, running statistics and an evolution clock.

Its equilibrium is composed from:
`bundle(weight_hv ⊙ state, input_mask ⊙ input, optional Fourier basis)`, followed by an elementwise activation.

The scalar time constant depends on state norm and input similarity:
`tau = clamp(tau_base * (1 + backbone_tau * ||state||) * (1 + 0.2 * cosine(input, tau_modulator)), 0.01, 10.0)`.

The adaptive gate depends on cosine similarity between the mean-bundled state/input vector and `gate_weight`, plus the mean of `gate_bias`. **SWI-001 initially used an additive gate floor in the interpolation formula, but the zero-time audit found that this made `sigma(dt=0)=gate` rather than zero.** That is incompatible with a continuous-time flow.

The current branch defines the sigmoid gate as a positive rate multiplier:
`gate_rate = sigmoid(gating_steepness * gate_activation)`
`sigma = 1 - exp(-dt * gate_rate / tau)`
`state_next = (1 - sigma) * state + sigma * equilibrium`.

This satisfies the essential identity `F(0, x) = x`. The learned gate changes the effective rate rather than injecting an instantaneous state jump. The frozen-equilibrium step remains an approximation to the nonlinear, time-varying ODE; RK4 and iterative substeps now evaluate the same gate-scaled vector field and Fourier phase.

### 2.2 Standalone HDC-LTC neuron

File: `crates/core/symthaea-hdc-ltc/src/neuron.rs`

The standalone implementation uses per-coordinate temporal parameters and gates. It is more expressive in one sense: each coordinate can have its own effective timescale and gate. But it still updates coordinates independently apart from any mixing performed outside the neuron.

It should be benchmarked as a separate model family:
- scalar-tau / scalar-gate unified core neuron;
- per-coordinate-tau / per-coordinate-gate standalone neuron;
- blockwise or low-rank coupled HDC liquid neuron (proposed).

Do not conflate their equations, parameter counts, or accuracy results.

### 2.3 Unified network composition

File: `crates/core/symthaea-core/src/hdc/hdc_ltc_unified.rs`

The current network:
- groups neurons into layers;
- evolves each neuron in a layer;
- averages neuron states into a layer output;
- optionally binds deeper-layer inputs with a layer-binding vector;
- optionally bundles a skip connection from the original input;
- caches the resulting layer output.

This yields a layered, high-dimensional liquid representation. A separate HRR module now supplies unitary role binding, correlation unbinding, and a confidence-aware nearest-codeword cleanup codebook. It is the first associative-memory primitive, not yet a complete multi-factor resonator or a demonstrated capacity guarantee.

### 2.4 Current algebra is not all of HRR

File: `crates/core/symthaea-core/src/hdc/unified_hv.rs`

The current `ContinuousHV::bind` is Hadamard/elementwise multiplication. The code’s binding-algebra audit correctly documents that for continuous random components in `[-1, 1]`, this operation is commutative but not self-inverse: `A ⊙ A` does not recover the all-ones identity. A previous bipolar-VSA assumption would be incorrect for this continuous distribution.

This is not a defect in Hadamard binding itself. Different VSAs have different algebraic contracts. The branch now includes a separate FFT-based HRR implementation (`crates/core/symthaea-core/src/hdc/hrr.rs`) with deterministic unitary roles: spectral coefficients are phase-normalized and mirrored to preserve a real time-domain vector, so circular correlation exactly unbinds (up to floating-point error). It is not substituted for the Hadamard operator. The HRR role and both backends remain explicit and independently testable. Different VSAs have different algebraic contracts:

| Algebra | Typical binding | Unbinding contract |
|---|---|---|
| Binary Spatter / MAP | XOR or bipolar Hadamard | Self-inverse for bipolar vectors |
| Real-valued MAP-like | Hadamard | Requires nonzero coordinate-wise inverse or a defined approximation |
| HRR | Circular convolution | Approximate inverse via circular correlation / pseudo-inverse |
| Tensor product | Outer product | Exact unbinding at higher-dimensional cost |

Symthaea should expose the algebra as an explicit choice and test it with the assumptions each algebra requires. Do not call all binding operations interchangeable “holographic binding.”

## 3. Findings that change the engineering priorities

### 3.1 Genesis and seeded initialization were on different scales

Before this branch’s change:
- `new()` used modified Gram–Schmidt and returned unit-norm orthogonal vectors;
- `from_genesis()` used raw GenesisSeed hypervectors with components in approximately `[-1, 1]`, without normalizing or orthogonalizing them.

For a D-dimensional independent uniform vector, expected L2 norm is approximately `sqrt(D / 3)`. At D=16,384, that is approximately 73.9, compared with norm 1 for the `new()` path. These are materially different initial dynamical regimes.

**Branch change:** `from_genesis()` now draws domain-separated GenesisSeed vectors and applies modified Gram–Schmidt, so both constructors use unit-norm, low-interference parameter vectors. The gate-bias vector retains the 0.1 scale. A regression test checks norms and pairwise orthogonality.

This is an intentional initialization-contract correction. Existing trained checkpoints are not rewritten by this change; checkpoint compatibility and migration remain an explicit release concern.

### 3.2 Forward/backward consistency and local learning

The old core `backward()` did not differentiate the actual adaptive forward map:
- it omitted the optional Fourier term;
- it used a pure exponential interpolation factor rather than the adaptive gate;
- it omitted the exact 1/2 or 1/3 bundle derivative;
- it used a tau derivative for a different equation.

The phoneme trainer also evolved the network before calling `backward()`, then optimized a further hypothetical step while reporting similarity for the already-evolved state. This muddied the loss/gradient relationship.

**Branch changes:**
- shared pre-activation construction between forward and backward;
- shared adaptive-gate definitions;
- exact full-dimensional cosine functions for tau and gate derivatives, independent of the global similarity stride;
- local gradients for the equilibrium vectors, tau-modulator, gate weight, gate bias, input and pre-evolution state;
- zero tau derivative at the clamp boundaries;
- `HdcLtcUnifiedNetwork::train_local_target_step()`, which calculates local gradients before evolving each layer and makes the layer-local target semantics explicit;
- `PhonemeHdcLtc::train_step()` now uses this pre-evolution local target update.

These are **one-step local gradients**, not temporal BPTT. The layer-local trainer is not full chain-rule backpropagation through depth. Temporal credit assignment remains a separate research problem; the repository’s temporal-credit design recommends validating the temporal task first and investigating eligibility traces or e-prop before adding a large activation tape.

The long-standing end-to-end phoneme “training changes output” test remains ignored. It must be re-run with per-step loss, parameter-norm, saturation, and held-out phoneme accuracy measurements before being promoted to a passing acceptance test.

### 3.3 The liquid approximation must be error-aware

The core source includes a numerical validation file:
`crates/core/symthaea-core/src/hdc/hdc_ltc_unified_validation.rs`.

That validation compares the one-step update with a higher-accuracy numerical reference and reports sizeable errors in some tested regimes (approximately 27–30% relative error for the adaptive one-step path in its tested scenarios). Iterative sub-stepping is below 1% in those particular tests, but costs O(dt / tau) state updates rather than a single O(D) update.

These numbers are repository-specific measurements, not universal error bounds. They show that:
- a single interpolation step is fast but can be inaccurate for changing equilibria;
- iterative sub-stepping is a useful reference/accuracy baseline;
- the system should report its error/cost tradeoff and not claim an exact closed-form solution.

A September 2026 IEEE TNNLS paper, *Rederived Closed-Form Continuous-Time Neural Networks*, proposes NIALIM using Lagrange interpolation of the input-dependent integral, derives a bound on approximation error, and introduces DFA-CfN and PRDFA-CfN variants evaluated across six tasks. This is directly relevant to Symthaea’s next integrator experiment. Compare against the paper’s actual datasets, splits, metrics, and implementation—not the headline “up to 69.2% improvement” across heterogeneous tasks.

### 3.4 Continuous-time identity correction

A deeper audit found that the previous gate formula,
`sigma = 1 - exp(-dt / tau) * (1 - gate)`,
did not satisfy the identity map at zero elapsed time: `sigma(0) = gate`. That allowed a nonzero state jump at `dt = 0`.

The branch now makes the learned sigmoid gate a positive rate multiplier:
`sigma = 1 - exp(-dt * gate_rate / tau)`.
The exact and iterative frozen-coefficient paths use this same rate. Euler and RK4 use the corresponding vector field:
`dx/dt = gate_rate(x,u) * (equilibrium(x,u,t) - x) / tau(x,u)`.

The RK4 stages now evaluate the Fourier basis at their own stage times. The iterative path recomputes Fourier phase, equilibrium, tau and gate on every substep. Non-finite and negative dt values are normalized to zero at the neuron integration boundary. Tests assert zero-time identity across all integrators and finite-difference agreement against the post-forward retained context.

This correction makes the model a better-defined continuous-time system. It does not make the frozen-coefficient update an exact solution of the nonlinear ODE; numerical fidelity still needs to be measured against RK4 and a higher-accuracy reference.

### 3.4 Existing resonator audit and repair

The repository already had `crates/core/symthaea-core/src/hdc/resonator.rs`; the HLB programme should extend it rather than fork a second unrelated resonator.

The audit found:
- `solve_system()` returned `converged=true` for every variable even when the iteration loop exhausted its limit;
- unconstrained unknowns could be left at random initial estimates yet be counted as converged;
- multi-system results could inherit stale single-system energy history;
- the solver used process-global randomness for initialization and exploration, preventing exact replay;
- the continuous-valued Hadamard update used elementwise multiplication as an approximate correlation signal. The source's own tests already acknowledged that this is not an exact inverse for general real-valued vectors.

**Branch changes:**
- validate dimensions, finite values, positive finite weights, unique/non-empty unknown names, declared symbol/variable references, and the requirement that each unknown has a solving constraint;
- report the actual number of iterations and the actual system-wide convergence state; iteration-limit exhaustion remains `converged=false`;
- compute and retain weighted multi-constraint residual energy on every sweep;
- add `ResonatorOutcome::{CorrectConvergence, SpuriousConvergence, NonConvergence}`, classified against known ground truth separately from the solver's fixed-point flag;
- remove the artificial confidence floor and report no symbol confidence when no codebook exists;
- add a configurable seed and a local deterministic PRNG for initialization and exploration noise, making identical call sequences replayable;
- normalize and validate codebook vectors at insertion.

**Important algebra boundary:** this repairs reporting and run integrity; it does not turn the existing continuous-Hadamard correlation heuristic into an exact inverse. For exact HRR role inversion, use the separate unitary HRR backend. The next integration task is to benchmark the existing resonator algorithm with matched bipolar/MAP and HRR variants, not to relabel its current float path as exact.

### 3.4 SWI branch implementation increments

The research branch now contains the following HDC/CfC work:

- `from_genesis()` uses domain-separated vectors followed by modified Gram–Schmidt, matching the unit-norm initialization contract of `new()`.
- Forward pre-activation and adaptive-gate helpers are shared by the core local-gradient implementation.
- One-step local gradients include equilibrium, Fourier bundling, adaptive gate, cosine-based tau modulation, gate weight/bias, input and pre-evolution state paths.
- The tau derivative is zero at the effective tau clamp boundaries.
- The phoneme trainer applies local target updates before evolving the frame. It is not described as temporal BPTT.
- HDC-LTC now exposes typed parameter group IDs, profile statistics, bounded read-only windows, and dimension/finite/norm-validated replacement.
- The existing `resonator.rs` no longer misreports iteration-limit exhaustion as convergence; it exposes correct/spurious/non-convergence classification against ground truth, validates constraints, tracks system residuals and uses replayable RNG seeds. Its float-Hadamard unbinding remains a heuristic.
- The `structured_mixer.rs` prototype provides identity, cyclic permutation and normalized Walsh–Hadamard transforms with exact inverse methods.
- The `hrr.rs` prototype implements O(D log D) circular convolution/correlation using a radix-2 FFT and unitary HRR roles, with exact unbinding for unitary roles.
- `HdcHrrCodebook` now performs nearest-codeword cleanup with both minimum-similarity and best-versus-runner-up margin thresholds; ambiguous retrieval is represented as abstention.
- HRR and structured-mixer tests specify round-trip reconstruction, norm preservation, one-hot spreading, role-slot retrieval, loaded-record top-1 retrieval, confidence-based abstention, commutativity, and invalid-shape rejection.
- The Criterion suite `hlb_algebra_dynamics` now times Hadamard, Walsh–Hadamard, permutation, HRR bind/unbind, HRR cleanup, and liquid integrators across D=256–16,384; the 65,536 and retrieval-load sweeps remain follow-on work.

**Important:** the structured mixer is not yet wired into the default HDC-LTC neuron. Its tests establish operator correctness, not a task-accuracy or efficiency improvement. The new Rust tests have not run in this environment.

## 4. Paradigm shifts for the Holographic Liquid Brain

### Shift A — Separate representation, dynamics, mixing, memory and learning

Do not make one neuron responsible for every cognitive function. Define five cooperating primitives:

1. **Representation algebra:** encodes symbols, roles, fillers, percepts and relationships in named HDC/VSA spaces.
2. **Liquid temporal state:** evolves context and latent state over continuous time, with scalar, per-coordinate, or blockwise time constants.
3. **Structured mixer:** creates controlled interactions across dimensions.
4. **Associative memory and cleanup:** binds, bundles, retrieves, resolves interference, and returns known items or calibrated uncertainty.
5. **Learning/credit assignment:** updates parameters with a defined objective and explicit temporal semantics.

This decomposition lets us improve one capability without silently changing the mathematical assumptions of another.

### Shift B — Introduce genuine cross-dimensional coupling

The current core neuron’s state/input transforms are Hadamard products, followed by elementwise activation. Most coordinate interactions are therefore separable; cross-coordinate coupling occurs mainly through global scalar norms, cosine similarities, bundling, and the network’s layer structure.

That is not yet a richly coupled “liquid brain.”

A first independent mixer primitive is now implemented in `crates/core/symthaea-core/src/hdc/structured_mixer.rs`. It is not enabled in the default neuron. Research and compare three structured mixer families:

- **Permutation / orthogonal transform:** fixed signed permutations or fast Walsh–Hadamard transforms; inexpensive, norm-preserving, and easy to ablate.
- **Circular-convolution mixer:** HRR-like interaction with O(D log D) FFT implementation and an explicit correlation-based unbinding test.
- **Low-rank learned mixer:** `x + A(Bx)` with rank k, costing O(Dk) and allowing learned interactions without a full D×D matrix.

Compare them to the current diagonal/Hadamard baseline and a dense recurrent matrix upper bound. The objective is not “more mixing” for its own sake: establish whether it improves nonlinear temporal prediction, associative retrieval under interference, or generalization per unit compute.

### Shift C — Error-controlled liquid integration

Build an integrator ladder:

1. Current adaptive single-step CfC-like update (fast baseline).
2. Iterative sub-stepping / RK4 (accuracy reference and current robustness baseline).
3. Piecewise-linear input interpolation with a Lagrange/NIALIM-like update.
4. An embedded error estimate that selects one-step, two-stage, or sub-stepped evolution based on dt and local curvature.
5. Event-triggered multi-rate updates where fast state groups update frequently and slow memory groups update only when evidence changes.

The key paradigm shift is **adaptive fidelity**: do cheap work when the update is easy, spend more computation when the model predicts its frozen-equilibrium approximation will be inaccurate. Every adaptive decision must be measurable and deterministic under replay.

### Shift D — Make “holographic memory” a measured memory system

Add a separate memory module with:
- role/filler encoding;
- multiple binding backends (MAP, bipolar/XOR, HRR);
- bundling and interference management;
- cleanup memory or resonator retrieval;
- temporal order and chunk/sequence coding;
- explicit capacity and false-retrieval confidence.

This is a better-defined holographic component than simply labeling a 16K liquid state as holographic. The liquid state can remain the dynamical context; the associative memory can provide structured long-term storage and retrieval.

### Shift E — Treat time scales as a learned hierarchy

Compare:
- one scalar tau per neuron;
- one tau per coordinate;
- grouped tau per learned subspace or block;
- a mixture of fast/medium/slow liquid channels;
- event-triggered long-term memory writes.

Measure whether different channels specialize in distinct temporal horizons. Use a fixed total parameter and compute budget so per-coordinate tau does not win simply by adding more degrees of freedom.

### Shift F — Temporal credit assignment without pretending local learning is BPTT

The network’s current `d_input` and `d_state` are local one-step derivatives. They do not carry credit across a sequence.

Investigate:
- eligibility traces / e-prop with O(1) per-neuron state;
- local predictive objectives (predict next input/state);
- short truncated BPTT as an accuracy baseline;
- hybrid local traces plus occasional short unrolls.

Validate the task before adding a temporal-learning mechanism. Use nonperiodic sequences, swap-sensitivity controls, delayed cues, and strong EMA/linear baselines. A periodic benchmark can be solved with no memory and must not be used to claim sequence learning.

### Shift G — Weight-space programming for HDC-LTC itself

SWI now has a CfC parameter manifest and typed edit interface. HDC-LTC now has a corresponding first introspection surface:
- typed `HdcLtcParameterGroupId`;
- `HdcLtcNeuronParameterProfile`;
- group norm, mean, variance, min/max, non-finite and saturation counts;
- bounded read-only windows into named hypervectors;
- validated replacement with exact dimension, finite-value, and norm checks.

Future HDC-native edit operators should understand the algebra:
- rotate/permute a vector;
- orthogonalize an initialization set;
- apply a tangent-space update to a cosine-scored vector;
- adjust a time-constant field;
- adjust a gate vector or scalar gate bias;
- change the mixer rank or structured transform.

A generic “scale this vector” edit is insufficient for cosine-controlled parameters because positive scaling leaves cosine similarity unchanged.

## 5. Benchmark programme

Create a benchmark family, not a single “brain score.”

### 5.1 HLB-DYN: dynamical fidelity

Compare the model output against RK4 and a high-accuracy adaptive solver over:
- fixed and irregular dt;
- piecewise-constant and piecewise-linear input;
- smooth, discontinuous, and high-curvature input;
- small and large dt/tau ratios;
- long-horizon rollout and regime changes.

Metrics:
- one-step and multi-step relative/absolute error;
- error growth over horizon;
- boundedness, finite-state rate, and stability violations;
- runtime and memory at equal error;
- number of adaptive substeps / interpolation stages.

Baselines:
- current core one-step;
- core iterative update;
- standalone per-coordinate liquid neuron;
- standard CfC / LTC;
- NIALIM / DFA-CfN / PRDFA-CfN (2026 IEEE TNNLS) where a reproducible implementation and matched protocol are available;
- RK4 / adaptive reference solver.

### 5.2 HLB-HDC: algebra and associative memory

Test each binding backend over:
- Hadamard continuous binding (current default);
- bipolar MAP/Hadamard and BSC/XOR (comparison backends still to implement);
- HRR circular convolution with unitary and non-unitary roles (unitary HRR operator now implemented);
- FHRR complex phase binding as a separate comparator;
- D = 256, 1,024, 4,096, 16,384, 65,536;
- bundle load (number of stored items);
- comparison/query count;
- retrieval noise and missing components;
- nested role/filler structure;
- sequence and order retrieval.

Metrics:
- exact and top-k retrieval accuracy;
- crosstalk and false-association rate;
- unbinding reconstruction similarity;
- calibration / abstention when retrieval is ambiguous;
- capacity versus dimension, bundle load, and number of comparisons;
- memory and energy per stored/query item.

For multi-factor factorization, report three terminal states separately: correct convergence, spurious convergence, and non-convergence. Also report updates-to-correct-convergence conditioned on success. A recent 2026 resonator study demonstrates that cleanup rules (sign, softmax, ReLU, polynomial) can trade correct convergence against spurious attractors and timeout; the fastest successful subset is not the same as the most reliable algorithm. Its matched-dimension bipolar/FHRR analysis should be a baseline for HLB-HDC.

Do not treat capacity as a function of bundle load alone: query/comparison count changes the probability of false retrieval.

### 5.3 HLB-SEQ: temporal memory and prediction

Use non-periodic, information-validated tasks:
- delayed impulse recall;
- temporal order and swap sensitivity;
- variable-duration context;
- change-point/regime detection;
- compositional sequence recall;
- irregularly sampled sensor dynamics;
- delayed phoneme/context discrimination.

Baselines:
- static HDC / ridge;
- EMA bank (strong low-complexity baseline);
- AR / linear state-space models;
- GRU/LSTM;
- standard LTC/CfC and NIALIM-style CfC variants;
- Neural CDE / ODE-RNN where appropriate;
- SSMs (S4/Mamba-style) where their discrete-time assumptions are reasonably matched;
- low-rank RNN / neural-engineering-framework models, including online FORCE-style teacher-trained approaches.

### 5.4 HLB-LEARN: actual learning ability

Test whether learning changes the intended behavior:
- gradient finite-difference agreement;
- per-step loss and held-out accuracy;
- weight and gate/tau parameter movement;
- saturation and gradient norm distributions;
- class-balanced confusion matrices;
- calibration and uncertainty;
- performance versus data and compute budget.

For phoneme recognition, report PER/WER on an identified dataset and split. Preserve the current honest repository status: the existing phoneme HDC-LTC test is ignored; it is not a solved ASR system.

### 5.5 HLB-MIX: structured coupling ablations

Compare:
- no mixer / current coordinate-wise baseline;
- fixed permutation;
- signed permutation;
- fast orthogonal transform;
- circular convolution;
- low-rank learned mixer at several ranks;
- dense recurrent mixer.

Keep dimensions, parameter budgets, initialization, training tokens, optimizer steps, and compute budgets visible. Evaluate accuracy, dynamical fidelity, retrieval, robustness and cost.

### 5.6 HLB-OPS: operational efficiency

Report:
- CPU latency at D=1K/4K/16K/64K;
- GPU latency when supported;
- allocations and peak memory;
- throughput per watt when measurement hardware is available;
- exact numerical error at each performance point;
- SIMD versus scalar path difference;
- checkpoint size and serialization cost.

A faster one-step method that is 30% away from its reference is not “better” than a slower method without an explicit application-level error budget.

## 6. Initial research gates (targets, not results)

| Gate | Proposed target |
|---|---:|
| Finite-difference agreement for each one-step parameter group | relative error ≤1e-2 on seeded test cases, with absolute tolerance near zero |
| State/input gradient agreement | relative error ≤1e-2 on non-clamped cases |
| Genesis vs seeded constructor vector norms | within 1e-3 of unit norm (gate bias norm 0.1) |
| Initialization pairwise cosine | absolute cosine <0.02 on tested groups |
| Non-finite parameter edits | 100% rejected before mutation |
| HDC read-window bounds | 100% safe for overflow/out-of-range offsets |
| Dynamical accuracy | publish error/runtime Pareto curve; target <1% relative error on preregistered reference cases |
| Sequence learning | statistically supported improvement over EMA and static HDC on hidden nonperiodic tasks |
| Cross-dimensional mixing | improvement at equal compute/parameter budget, not just relative to a strawman |
| Phoneme end-to-end learning | unignore only after held-out accuracy and per-step learning curves meet a preregistered threshold |

These are engineering gates. They are not claims of current performance or SOTA.

## 6.1 Industry and SOTA comparison matrix

The comparison needs to be task-specific. HDC is not a universal classifier winner: the 2025 systematic comparison uses seven established HDC datasets plus 121 UCI datasets and reports that conventional off-the-shelf classifiers can surpass HDC methods on many tabular classification tasks. It also identifies key-value encoding and iterative adaptive training as strong HDC directions. Therefore, include conventional classifiers in any classification claim, and make compositional memory, temporal irregularity, robustness and deployable efficiency the areas where HLS must prove a distinct advantage.

| Capability | Strong external comparators | What Symthaea must match or beat |
|---|---|---|
| HDC classification | TorchHD baselines: OnlineHD, AdaptHD, SparseHD, QuantHD, NeuralHD, CompHD, LeHDC, SemiHD; plus SVM, LDA, Ridge, GLVQ and RVFL | Accuracy/F1, few-shot curves, calibration, latency, memory, and robustness on the seven-dataset HDC benchmark and the 121-UCI benchmark |
| Continuous-time dynamics | LTC, CfC, Neural CDE, ODE-RNN, adaptive RK methods; NIALIM / DFA-CfN / PRDFA-CfN (2026) | Matched one-step and rollout error, irregular-sampling robustness, accuracy/runtime Pareto curves, and a declared error bound |
| Structured recurrent coupling | Dense RNN, GRU/LSTM, low-rank RNN, NEF/FORCE-style teacher-trained recurrent systems, OMP-minimal dynamics fits | Same-task accuracy at matched parameter count, FLOPs and training data; report whether structured mixing helps |
| Associative retrieval | HRR/FHRR, bipolar MAP/BSC, modern Hopfield memory, Symthaea's existing continuous-Hadamard resonator, and confidence-aware HRR cleanup | Exact/top-k retrieval, capacity, false recall, confidence calibration, correct convergence, spurious convergence, non-convergence, and updates to correct convergence |
| On-device / edge efficiency | TorchHD / optimized SIMD HDC and domain-specific HDC ASIC/FPGA systems | Latency, peak memory, bytes moved, energy per correct retrieval/prediction, and numerical error on identical hardware |

The 2026 resonator comparison is especially important for HLB-HDC. It evaluates sign, softmax, ReLU and polynomial cleanup; it separates correct convergence from spurious convergence and non-convergence; and it finds that cleanup nonlinearities change both factorization capacity and failure mode. On matched-dimensionality FHRR experiments, the relative ranking also changes by representation. HLS should reproduce this protocol before claiming a better associative memory. For dense bipolar F=3, D=1,000 factorization, that study reports roughly 4.2 / 13.8 mean recurrent updates for sign cleanup at search spaces near 4.9e3 / 6.4e4, but those iteration counts are conditional on correct convergence; they are meaningless without the terminal-outcome proportions.

The 2026 NIALIM paper is the direct continuous-time comparator. It challenges the frozen-input / zero-order approximation on high-resolution and irregularly sampled inputs, derives a Lagrange-interpolation approximation with an error bound, and evaluates DFA-CfN and PRDFA-CfN across six sequence tasks. Symthaea should reproduce selected task protocols rather than infer superiority from one local ODE test.

The NeurIPS 2025 low-rank RNN work motivates a parameter-matched low-rank baseline, including teacher-training, OMP-based minimality and active selection of informative activity patterns. A fixed Walsh–Hadamard transform is only a useful mixer if it gives measurable performance per unit compute over simpler permutations and learned low-rank alternatives.

## 6.2 Quantitative research gates

These targets are preregistration goals, not measured results.

| Experiment | Initial target to beat |
|---|---|
| Zero-time identity | Exact state identity to floating-point tolerance for every integrator at dt=0 |
| One-step gradient correctness | Relative finite-difference error ≤1e-3 on well-conditioned coordinates; explicitly report absolute error near zero and clamp-boundary cases |
| Liquid numerical fidelity | ≤1% normalized rollout error on preregistered smooth and piecewise-linear tasks, while reporting latency and work per step |
| Irregular time sampling | No material accuracy collapse under held-out irregular dt schedules versus the matched NIALIM / CfC baseline |
| One-factor HRR cleanup | ≥99% top-1 retrieval at declared dimension/load, with separately reported abstention, false-accept and false-reject rates |
| Multi-factor resonator | Beat sign/softmax/ReLU/polynomial baselines on correct-convergence rate at matched D, codebook size, restart count and iteration budget; never optimize mean iterations alone |
| Temporal task learning | Statistically significant improvement over EMA, AR/linear state-space, GRU/LSTM, CfC/LTC and low-rank RNN at matched data/compute |
| Efficiency | Show a Pareto improvement in task quality versus latency/memory/energy on fixed hardware; do not compare CPU microbench timings to GPU papers |

For multi-factor memory, preregister D, number of factors F, codebook size n per factor, search space M=n^F, random seeds, initialization, maximum iterations and restart budget. Record:
- correct convergence;
- spurious convergence;
- non-convergence;
- exact factor recovery;
- per-factor similarity and top-1 margin;
- mean/median/worst-case updates *conditioned on correct convergence*;
- wall time and memory per trial.

For HDC classification, reproduce the published split and preprocessing rules; use the 7-task benchmark and 121-UCI set; include conventional SVM/LDA/Ridge/GLVQ/RVFL controls; and report both training and inference times. A benchmark-only win on one selected dataset is not a SOTA claim.

## 7. What would make this different from the field?

The most credible differentiation is not simply “HDC plus CfC.” Both HDC/VSA and liquid-network research are active fields. A 2025 systematic HDC/VSA benchmark review covers seven established tasks plus 121 UCI datasets and reports that HDC methods can still be surpassed by off-the-shelf classifiers, despite improvements in few-shot learning and robustness. The HLB programme must therefore include well-tuned conventional baselines, not just HDC variants. In continuous-time modelling, the 2026 NIALIM/DFA-CfN paper is a direct comparator for the current zero-order/frozen-input CfC path. For structured coupling, recent NeurIPS low-rank RNN results motivate a parameter-matched low-rank baseline, not just dense RNNs.

The potentially distinctive contribution is a **verified, algebra-aware, error-controlled hyperdimensional liquid substrate** with:
- multiple explicit VSA algebras and tested inverse contracts;
- structured low-cost cross-dimensional coupling;
- temporal integration with an empirical error budget;
- separate associative memory and dynamical context;
- locally correct gradients and a clear temporal-credit model;
- architecture-aware weight-space diagnosis and safe edits.

This is a research hypothesis. To claim superiority, Symthaea must reproduce selected baseline tasks and beat matched baselines on accuracy, robustness, compute, or energy. A new architecture diagram is not a benchmark result.

## 8. Immediate implementation sequence

### P0 — branch changes already made
- Align Genesis initialization with unit-norm orthogonal initialization.
- Share forward equilibrium and gate helpers with the local-gradient path.
- Differentiate adaptive gate, tau, tau-modulator, gate-weight, gate-bias, input, and recurrent-state paths.
- Stop using global stride-dependent similarity in the tau derivative.
- Respect tau-clamp derivative boundaries.
- Add finite-difference tests for equilibrium, Fourier, gate, tau, input, and state gradients.
- Correct the phoneme local trainer to update before its frame evolution.
- Add typed HDC-LTC parameter profiles, safe group replacement and bounded read-only windows.
- Add an independent identity/permutation/Walsh–Hadamard mixer prototype with inverse and norm-preservation tests.
- Add a separate unitary-HRR circular-convolution/correlation backend with O(D log D) radix-2 FFT and unitary-role tests.
- Add HDC-LTC parameter profiles, typed replacement, and bounded immutable parameter windows.
- Keep the end-to-end phoneme training regression ignored until it has a measured acceptance threshold.

### P1 — next
- Run exact-head targeted tests and compile checks.
- Re-audit all HDC-LTC backward callsites and specify whether each objective is current-frame or one-step-ahead.
- Add a single mathematical specification shared by core and standalone HDC-LTC, with explicit scalar and per-coordinate variants.
- Implement an independent integration comparison harness against RK4 / an adaptive solver.

### P2 — architecture experiments
- Implement a small HDC-LTC structured-mixer prototype behind a feature/configuration gate.
- Add HRR circular-convolution as a separate binding backend and compare to current Hadamard/MAP.
- Extend the current confidence-aware one-factor HRR cleanup into multi-factor resonator factorization, with explicit correct/spurious/non-convergence outcomes.
- Benchmark role/filler HRR record retrieval across load and dimension, not just the exact one-role inverse.
- Add FHRR as a matched-effective-dimension comparator; the current backend is real-valued HRR and must not be presented as FHRR.

### P3 — learning and full HLB
- Add eligibility traces after the temporal benchmark validator is run.
- Evaluate learned fast/slow/block timescale specialization.
- Compose the HDC memory, liquid dynamics, and structured mixer as separate documented modules.

## 9. Current validation status

The code changes in this document are committed to the SWI research branch and included in the draft SWI PR. **The Rust test suite has not been executed in this environment**: this container does not have `rustc` or `cargo`, and the relevant CI workflows have previously reported `skipped`, not passed. The new tests are therefore regression specifications awaiting exact-head execution, not verified passing results.

The ignored phoneme training test remains ignored. No new phoneme accuracy, numerical superiority, or SOTA claim is made.

## 10. References

1. Liang, X. et al. (2026). *Rederived Closed-Form Continuous-Time Neural Networks.* IEEE Transactions on Neural Networks and Learning Systems, 37(9), 4381–4395. DOI: https://doi.org/10.1109/TNNLS.2026.3669345. PubMed: https://pubmed.ncbi.nlm.nih.gov/41790819/
2. Vergés, P. et al. (2025). *Classification using hyperdimensional computing: a review with comparative analysis.* Artificial Intelligence Review 58, 173. https://doi.org/10.1007/s10462-025-11181-2
3. Arora, A. and Pillow, J. (2025). *Efficient Training of Minimal and Maximal Low-Rank Recurrent Neural Networks.* NeurIPS 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/902244a9e85ebc8535ce111cf96f3e04-Abstract-Conference.html
4. Yeung, C., Poduval, P. and Imani, M. (2026). *A comparative study of nonlinear cleanup rules in resonator networks.* Frontiers in Artificial Intelligence, 9, 1793314. https://doi.org/10.3389/frai.2026.1793314
4. Salik, A. Y. et al. (2026). *Hyper-dimensional computing architectures for AI applications: A comprehensive survey of principles, performance, and future prospects.* Results in Engineering, 29, 109589. DOI: https://doi.org/10.1016/j.rineng.2026.109589
5. IEEE CIS Task Force on Hyperdimensional Computing and Vector Symbolic Architectures: https://cis.taskforce.ieee.org/hdcvsa/
6. *A comparison of vector symbolic architectures.* Artificial Intelligence Review (2021). https://link.springer.com/article/10.1007/s10462-021-10110-3
7. *Vector symbolic algebras for the Abstraction and Reasoning Corpus.* npj Unconventional Computing (2026). https://www.nature.com/articles/s44335-026-00100-5
8. *Vector Symbolic Architectures as a Computing Framework for Emerging Hardware.* https://pmc.ncbi.nlm.nih.gov/articles/PMC10588678/
9. NIALIM / DFA-CfN implementation release: https://github.com/nuist-xf/DFA-CfNs
10. *Scalable Symbolic Reasoning with Matrix-Based Brain-Inspired Representations and Vector-Space Acceleration.* DATE 2026. DOI: https://doi.org/10.23919/DATE69613.2026.11539261
11. Yeung, C., Poduval, P. and Imani, M. (2026). *A comparative study of nonlinear cleanup rules in resonator networks.* Frontiers in Artificial Intelligence, 9, 1793314. DOI: https://doi.org/10.3389/frai.2026.1793314.

---

**Decision:** proceed with the Holographic Liquid Brain as a compositional research architecture. The next decisive experiment is not adding more dimensions or neurons: it is demonstrating that the algebra is correct, the gradients match the forward dynamics, temporal integration has a measured error bound, and structured coupling or associative memory adds measurable capability over strong baselines.
