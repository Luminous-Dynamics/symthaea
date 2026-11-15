# HLB HDC–LTC Neuron Qualification
## Source audit and next benchmark contract

**Recorded:** 2026-11-15  
**Source revision audited:** \`42446eccd8205c1acf5bff8c48a96a43d8fd2776\` (\`main\`)  
**Status:** source-level audit + proposed qualification protocol; no new runtime result is claimed.  
**Scope:** the HDC/LTC neuron, its cognitive-loop adapter, temporal prediction, HDC algebra, and the next measurable research steps.

This document extends the existing temporal benchmark v2 and HLB/FEP/WSP research programme. It does not replace their frozen protocols, change production defaults, or claim a new capability.

---

## 1. Executive finding

Symthaea has a real HDC–LTC implementation in its active cognitive loop, but three distinctions must be made before describing it as a general-purpose "Holographic Liquid Brain":

1. **The active implementation and the extracted crate are different models.** The cognitive loop imports the in-core \`symthaea_core::hdc::hdc_ltc_unified\` implementation through \`HdcLtcBridge\`. The standalone \`symthaea-hdc-ltc\` crate is marked archived in \`docs/crate-status.toml\`, has one workspace dependent, and has a materially different per-coordinate gate/time-constant formulation. It is not the implementation currently used by the cognitive-loop bridge.
2. **The active neuron's recurrent operator is coordinate-wise, not a learned dense transformation.** Each coordinate has its own state and weight-mask values, but the in-core neuron computes one scalar equilibrium gate and one scalar time constant for the whole hypervector. Dense random projections exist at the bridge input/output boundaries; they do not turn the recurrent update itself into a dense \(D\times D\) transformation.
3. **A one-shot update is not an exact solution of the full nonlinear recurrent trajectory.** The equilibrium is computed from the current state and then held fixed for the interpolation step. The source's own \`evolve_closed_form_exact\` documentation states that this is exact only when the equilibrium is constant; because it depends on state, large-step error is possible. The iterative method recomputes the equilibrium through substeps and therefore gives up the horizon-independent step count.

The opportunity is not to add another layer of terminology. It is to determine, with matched controls, whether HDC composition, liquid dynamics, cross-coordinate interaction, and memory each add a measurable capability—and to select the smallest configuration that does.

## 2. What the current cognitive loop actually executes

### 2.1 Runtime path

The active path is:

\`\`\`text
CognitiveLoopConfig
  -> TemporalBackend::HdcLtcUnified
  -> CognitiveLoopService::new
  -> TemporalNetwork::HdcLtc(HdcLtcBridge)
  -> symthaea_core::hdc::hdc_ltc_unified::HdcLtcUnifiedNetwork
\`\`\`

Evidence-bearing source locations:

- \`src/cognitive_loop/config/temporal.rs\`: \`TemporalBackend\` enum.
- \`src/cognitive_loop/config/mod.rs\`: \`CognitiveLoopConfig::default()\`.
- \`src/cognitive_loop/constructor.rs\`: backend construction.
- \`src/cognitive_loop/temporal_network.rs\`: backend delegation.
- \`src/hdc_ltc_bridge.rs\`: input projection, neuron network, output projection, training and pure prediction snapshots.
- \`crates/core/symthaea-core/src/hdc/hdc_ltc_unified.rs\`: active neuron/network definition.

**Default-configuration detail:** \`TemporalBackend\` defaults to \`HdcLtcUnified\`, and \`CognitiveLoopConfig::default()\` uses \`TemporalBackend::default()\`. That cognitive-loop default config also sets the bridge's initial HDC dimension to **2,048**, with adaptive dimension configuration enabled. The 16,384 dimension is the core hypervector constant/default for the underlying HDC types; it must not be presented as the cognitive loop's unconditional initial runtime dimension. The REPL's own persisted setting defaults to the \`cfc\` string, and \`symthaea-watch\` explicitly selects classic CfC. Executable entry points therefore need to be reported separately from the enum/config default.

### 2.2 Active in-core neuron

The active \`HdcLtcUnifiedNeuron\` stores a continuous hypervector state, weight hypervector, input mask, time-constant modulator, gate weight and gate bias. The equilibrium path is conceptually:

\[
x_\infty = f\!\left(\frac{W\odot x + U\odot u}{2}\right)
\]

where \(\odot\) is coordinate-wise multiplication, and \(f\) is the selected activation. Binding and bundling are useful representational operations, but in this implementation the recurrent transform is coordinate-wise.

The effective time constant is a **single scalar per neuron**:

\[
\tau = \operatorname{clamp}\left[
\tau_0(1+b\lVert x\rVert)
(1+0.2\,\operatorname{sim}(u,T)),\ 0.01,\ 10
\right]
\]

The interpolation gate is also a **single scalar per neuron**, computed from a similarity-like quantity over the bundled state/input and the gate hypervector. The resulting update is a shared interpolation coefficient applied across all coordinates.

Consequences:

- The vector dimension carries distributed state, but coordinates do not have independently learned recurrent transition matrices.
- Time-scale diversity within one neuron is not currently the same thing as the per-coordinate \(\tau_i\) advertised by the archived standalone crate.
- The bridge's dense input/output projection can mix dimensions at the interface, while the neuron's internal recurrent update remains coordinate-wise.
- Inter-layer HDC binding, bundling, and skip connections are a separate source of interaction from recurrent mixing within a neuron.

### 2.3 The archived standalone implementation

\`crates/core/symthaea-hdc-ltc/src/neuron.rs\` defines a different model:

- per-coordinate gates \(g_i\);
- per-coordinate time constants \(\tau_i\);
- per-coordinate interpolation values \(\sigma_i\);
- a distinct \`NeuronConfig\` and \`ContinuousHV\` API.

The extraction analysis and crate registry record that this crate was copied out but not adopted as the canonical implementation. The in-core definition continued to evolve, and the two APIs have diverged. This is not a harmless packaging difference: it changes the dynamics being tested.

**Decision:** do not migrate or merge the implementations by name. Treat them as separate experimental arms. Any eventual consolidation must identify which measured behaviour is being retained, port tests and consumers, remove the duplicate definition, and only then update the crate disposition.

### 2.4 The cognitive-loop trainer is not full temporal BPTT

The adapter's training names also need a precise boundary:

- `TemporalNetwork::train_step_bptt()` dispatches HdcLtc to `HdcLtcBridge::train_step()`.
- `HdcLtcBridge::train_step_impl()` runs one scratch forward transition, then calls `neuron.backward(...)` once for each neuron in each layer.
- The `HdcLtcGradients` type contains `d_input` for inter-layer propagation, but this bridge loop does not consume that field or pass an accumulated error to the preceding layer. Each neuron is instead updated against the same projected target hypervector.
- The call uses a historical evolution snapshot for the (encoding at t−1 → encoding at t) pair once the rolling snapshot queue is warm. This corrects the training start state; it is not an unrolled activation tape or backpropagation through multiple time steps.
- The default cognitive-loop config enables asynchronous training. In that path the sample is queued and the cycle returns no synchronous training-loss value, so qualification harnesses should pin `async_training = false` when they need a directly observed per-update result.

This does not mean the local gradient is useless: the neuron computes a derivative through its single interpolation, activation, binding and scalar-τ update. It means the name "BPTT" must not be read as proof of temporal credit assignment or full cross-layer chain-rule propagation in the active HdcLtc bridge.

**Immediate research consequence:** qualify three independent mechanisms rather than one ambiguous "BPTT" switch:

1. single-transition local gradient correctness;
2. cross-layer error propagation (including use of `d_input`);
3. temporal credit assignment across earlier activations (eligibility traces/e-prop or a bounded activation tape).

Use the existing temporal benchmark v2 to decide whether either of the latter two adds value. Keep any change to the bridge's local target rule separately ablated.

### 2.5 Multi-horizon prediction is a policy choice, not a single calibrated forecast

`get_multi_scale_prediction()` requests predictions from several horizons from the same restored starting state and averages them. It also retains the raw predictions and the shortest-horizon forecast. The compression programme correctly identifies the raw `delta_t` forecast as the appropriate object for one-step predictive-bits comparisons.

Averaging a (t+1), (t+5), and (t+10) forecast generally creates a new vector that is not itself a forecast at any one of those horizons. It may be a useful feature or ensemble statistic, but it should not be treated as a calibrated one-step prediction without a separate validation. Report horizon-specific error/calibration, and score the aggregate only against a separately declared aggregate target or downstream decision.

A related perception issue remains visible in `detect_semantic_patterns()`: it uses substring tests for words such as `if`, `do`, and `act`. The 2026-07-18 zero-vector failure from unmatched phantom primitives was fixed by changing the attention-weight averaging rule, but substring-triggered false positives (for example, `if` inside `life`) still deserve a word-boundary regression battery. Do not interpret the old zero-vector fix as a complete semantic-pattern parser.

### 2.6 What "O(1) temporal jump" means—and does not mean

The per-neuron interpolation update is \(O(D)\) in hypervector dimension \(D\), while a single update does not require a number of ODE solver substeps proportional to the elapsed interval. It is therefore reasonable to describe the interpolation step as **horizon-independent in step count**, not as constant total work independent of dimension.

The complete bridge has additional costs. It projects the compressed input into HDC space and projects the network output back. The current implementation stores dense projection arrays and performs row accumulation across those dimensions. Its end-to-end complexity therefore includes approximately:

\[
O(d_{\mathrm{in}}D + d_{\mathrm{out}}D + N_{\mathrm{neurons}}D)
\]

per forward step, before accounting for training and snapshot operations. A neuron-only microbenchmark cannot substantiate an end-to-end cognitive-cycle speed claim.

The source's pure analytical method also documents the nonlinear-equilibrium caveat. For a state-dependent equilibrium, one large interpolation and multiple smaller re-evaluated steps need not agree. This is exactly why trajectory error and composition error must be measured alongside throughput.

## 3. Research landscape that changes the plan

### 3.1 CfC is an approximation family, not a free exact solver

Hasani et al. introduced CfC as a fast closed-form *approximation* to LTC dynamics, motivated by the cost of numerical ODE solvers. Their formulation derives a bounded approximation under stated assumptions; it does not make arbitrary nonlinear recurrent dynamics exactly integrable.

A relevant 2026 comparison is Liang et al., *Rederived Closed-Form Continuous-Time Neural Networks* (IEEE TNNLS, 2026; DOI: 10.1109/TNNLS.2026.3669345). It introduces NIALIM, using Lagrange interpolation and elapsed-interval factors, and evaluates DFA-CfN/PRDFA-CfN variants on six sequence tasks including physical dynamics, human activity, PhysioNet, sentiment, remote sensing and energy forecasting. The reported improvements are claims from that paper's tasks and baselines—not transferable Symthaea results. The useful implication is that Symthaea's time-step approximation should be compared against a stronger contemporary closed-form family, not only its older scalar-gate variant.

### 3.2 HDC is not automatically more accurate than conventional ML

Vergés et al., *Classification using hyperdimensional computing: a review with comparative analysis* (Artificial Intelligence Review, 2025; DOI: 10.1007/s10462-025-11181-2), provide a standardized comparison across seven commonly used HDC datasets and 121 UCI datasets. Their analysis finds meaningful HDC strengths in efficiency and limited-data settings, while also reporting that off-the-shelf methods can surpass HDC classifiers. Key-value encodings performed well for their tabular tasks, and iterative adaptive methods often improved performance.

Therefore, Symthaea should compete on the domains where HDC has a testable structural advantage—compositional retrieval, corruption tolerance, few-shot adaptation, compact state, and interpretable role/filler operations—while retaining strong conventional baselines.

## 4. The paradigm shift to test

The next architecture hypothesis is a **factorized liquid memory**, not an ever-larger vector:

- **HDC/VSA layer:** explicit role/filler binding, relational composition, superposition and associative cleanup.
- **Interaction layer:** structured cross-coordinate mixing so a coordinate can respond to other coordinates when the task requires it.
- **Liquid layer:** elapsed-time-aware state evolution with a chosen granularity of time constants.
- **Episodic layer:** event records and source attribution, not a blind average injected into the forecast.
- **Epistemic layer:** uncertainty over hidden state, parameters and model structure; diagnostic actions selected by prospective information gain.
- **Verifier:** independent task outcomes and counterfactual controls decide whether any component deserves to remain enabled.

The current production model provides a useful coordinate-wise reference. The archived per-coordinate-tau model is a useful second reference. A new structured-mixing model is only a proposal until implemented and measured.

### 4.1 Orthogonal dynamics factors

Run a small factorial study instead of changing multiple mechanisms simultaneously.

**Factor A — recurrent interaction operator**

| Arm | Operator | Purpose |
|---|---|---|
| A0 | Current coordinate-wise HDC binding/update | Production reference |
| A1 | Fixed structured orthogonal mixing (e.g. signed permutation / normalized Hadamard) around the coordinate update | Test whether cross-coordinate interaction helps without learned dense weights |
| A2 | Low-rank learned mixing \(UV^\top\), with rank and parameter budget declared | Test economical learned coupling |
| A3 | Parameter-matched dense recurrent baseline | Conventional interaction control |

The fixed transform must be invertible or have an explicitly stated cleanup method where it participates in HDC role/filler algebra. Do not silently treat continuous Hadamard binding as a self-inverse bipolar MAP operation.

**Factor B — time-constant granularity**

| Arm | Time constants |
|---|---|
| B0 | One scalar \(\tau\) per neuron (current in-core implementation) |
| B1 | Grouped \(\tau\) (e.g. 8 or 16 declared timescale groups) |
| B2 | Per-coordinate \(\tau_i\) (standalone crate formulation) |

Group-level dynamics are a potentially better engineering point than either one global time constant or 16K independently adapted values: they can represent a few meaningful time scales without multiplying parameter count and optimizer state unnecessarily. This is a hypothesis, not a result.

**Factor C — integration/update strategy**

- one-shot frozen-equilibrium interpolation;
- iterative re-evaluated closed-form substeps;
- high-accuracy RK4 or adaptive Dormand–Prince reference;
- NIALIM/DFA-CfN/PRDFA-CfN comparator where its formulation and task assumptions match.

The test must report both output quality and elapsed compute. A slower method is not a better production method solely because its trajectory error is lower; a faster method is not better if it destroys the prediction signal.

Temporal credit assignment (local update, eligibility traces/e-prop, truncated BPTT) remains a separate factor under the existing Step 4 design. Do not mix it into the first dynamics comparison.

## 5. Qualification battery

Use the existing \`TEMPORAL_BENCHMARK_V2_PREREGISTRATION_2026-07-31.md\` as the sequence-task gate. This section defines the implementation-level and representation-level tests needed before interpreting its task scores.

### T1 — dynamical fidelity

Generate deterministic systems with known numerical reference trajectories, including:

- constant-input and piecewise-constant-input systems;
- linearly interpolated and high-curvature inputs;
- regular and irregular observation intervals;
- gaps spanning several time constants;
- controlled regime changes and missing observations.

Report:

- relative trajectory error against a converged high-accuracy numerical reference;
- error stratified by \(\Delta t/\tau\) and input-curvature regime;
- semigroup/composition error \(F(t_1+t_2,x)\) versus \(F(t_2,F(t_1,x))\);
- zero-duration identity, boundedness, state-norm and finite-value checks;
- one-shot versus iterative versus NIALIM/CfC/ODE-solver accuracy;
- inference latency, training latency, allocations, peak memory, parameters and energy when the device measurement is valid.

The implementation must not hide long-gap error behind an average across easy short intervals.

### T2 — compositional HDC

Test role/filler and temporal composition with held-out combinations:

- single-level and nested role/filler bindings;
- variable binding order and role swaps;
- partial cues, missing roles and corrupted fillers;
- superposition load sweeps as a function of dimension and number of stored items;
- retrieval under controlled noise, interference and bit flips;
- temporal sequence order and position-sensitive retrieval.

Report precision, recall, false-retrieval rate, held-out composition accuracy, capacity-versus-load curves, calibration, memory bytes and operations/query.

Required baselines: MAP/bipolar XOR, HRR/circular convolution, continuous Hadamard with its declared inverse/cleanup contract, fixed random projection, OnlineHD or an equivalent online HDC classifier, and a conventional classifier on the same encoded features.

### T3 — temporal learning and regime adaptation

Reuse the existing preregistered temporal ladder, including its mechanically guaranteed \`Static\` negative control and tuned \`EmaBank\` baseline. Preserve the existing arms:

- \`Static\`;
- \`EmaBank\`;
- \`PermutationVsa\`;
- \`HdcLtc\`;
- \`HdcLtc+Traces\`.

Add the dynamics factors only as separately identified arms. Measure ambiguous-point accuracy, NLL, Brier score, retention versus delay, order-swap sensitivity, recovery steps after regime shift, sample efficiency, and compute per correct ambiguous prediction.

Do not use overall token accuracy as the primary outcome when most positions are solvable without history.

### T4 — matched external task families

Where the I/O contract is comparable, include:

- original LTC/CfC;
- parameter-matched GRU and LSTM;
- Neural CDE or ODE-RNN for irregular-time input;
- NIALIM, DFA-CfN and PRDFA-CfN;
- a state-space baseline for long sequence tasks;
- tuned EMA bank and simple leaky-state controls;
- a memoryless model.

Use a task-specific strongest baseline. Do not compare a 16K-dimensional HDC model to a large transformer on an unrelated task and call the result an architecture comparison.

## 6. Acceptance rules

The temporal benchmark v2 preregistration already fixes the following sequence-task gates; keep them unchanged:

- the primary endpoint is accuracy only at validator-identified history-dependent decision points;
- minimum meaningful effect is +0.10 absolute accuracy over the best non-\`Static\` baseline;
- paired bootstrap 95% confidence interval for the per-seed difference must exclude zero;
- at least eight disjoint development-independent seeds per arm/task family;
- more than 3× the EMA compute per correct ambiguous prediction is an experimental result, not a production win.

For T1 and T2, register task-specific smallest effects of interest and all analysis choices before the first model run. Always report paired seeds, effect sizes, confidence intervals, the full per-seed distribution and all failed/invalid runs.

A result is not a win if it depends on:

- unequal training data, update counts or optimization budget;
- a weaker or untuned comparator;
- test-set tuning or unsealed holdout access;
- a self-reported internal confidence score rather than external outcomes;
- an aggregate metric that hides the ambiguous/hard subset;
- an unverified benchmark adapter or an unexercised ablation.

## 7. Proposed research targets

These are qualification targets, not claims about current performance.

| Capability | Target for a research win | Required comparison |
|---|---|---|
| Continuous-time fidelity | Lower trajectory error at matched latency, or equal error at lower compute, across regular and irregular gaps | RK4/DP reference, original CfC, NIALIM family |
| Temporal prediction | Clear the existing +0.10 ambiguous-point gate with 95% CI excluding zero | Tuned EMA, positional VSA, GRU/LSTM, static and memoryless |
| Compositional transfer | Improve held-out nested role/filler and order-swap accuracy without increasing false retrieval | MAP, HRR, continuous Hadamard, OnlineHD, matched conventional features |
| Robustness | Graceful degradation across load/noise sweeps with calibrated false-retrieval rates | Same dimension and memory budget baselines |
| Efficiency | Better task quality per joule/byte/latency, measured on the same hardware and release configuration | Complete bridge and end-to-end loop, not neuron-only microbenchmarks |
| Adaptation | Faster recovery after regime shift with bounded forgetting of previous regimes | EMA bank, GRU/LSTM, memoryless and continual-learning baselines |

For classification-oriented HDC claims, include the 2025 review's standardized seven-dataset and 121-UCI comparison where licensing and task formats allow. The review explicitly warns that conventional off-the-shelf methods can outperform HDC on some datasets.

## 8. Execution sequence that does not depend on CI runners

1. **P0 — source and manifest audit.** Pin the active core implementation, the archived standalone implementation, all bridge projections, dimensions, training path and consuming call sites. Record the exact source SHA in every output.
2. **P1 — mathematical contract tests.** Add tests for one-shot vs. iterative evolution, semigroup error, irregular-\(\Delta t\) trajectories, large-gap stability and equivalence of scalar/SIMD paths. These can run on a developer machine; queued CI is not evidence of a pass.
3. **P2 — implementation matrix.** Build benchmark-only arms for scalar, grouped and per-coordinate \(\tau\), plus diagonal and structured mixing. Keep production behavior unchanged.
4. **P3 — integrate with the frozen temporal benchmark v2.** Run validator gates and negative controls first. If any gate fails, stop and repair the harness before comparing models.
5. **P4 — external baselines.** Reproduce the NIALIM/DFA-CfN family on matched irregular-time tasks and report differences in model capacity, interpolation, data and compute budgets.
6. **P5 — decide implementation fate.** Only after measured evidence choose whether the active in-core model stays canonical, the archived per-coordinate model is revived, or a third explicitly named version is adopted. Migrate one definition and delete the other; do not leave two silently diverging canonical implementations.

No step here requires waiting for a hosted runner to do source inspection, task validation, local property tests or benchmark harness work. A CI result is recorded separately as pending until the exact head is tested.

## 9. Relationship to existing Symthaea work

This protocol deliberately builds on, rather than duplicates:

- \`docs/TEMPORAL_BENCHMARK_V2_PREREGISTRATION_2026-07-31.md\` — frozen sequence-task arms, negative control, threshold and compute gate;
- \`docs/TEMPORAL_CREDIT_ASSIGNMENT_STEP4_DESIGN_2026-07-31.md\` — time-credit hypothesis and eligibility-trace design;
- \`docs/research/HLB_FEP_GUIDED_WEIGHT_SPACE_PROGRAMMING.md\` — Stage 3 requirements for structured cross-dimensional mixing, scalar versus per-coordinate \(\tau\), and temporal credit assignment;
- \`docs/SYMTHAEA_CORE_EXTRACTION_ANALYSIS_2026-07-29.md\` — measured dependency graph and orphaned extraction history;
- \`docs/CONSOLIDATION_FAILURE_AUDIT_2026-07-29.md\` and \`docs/crate-status.toml\` — duplicate-definition failure mechanism and archived crate disposition;
- \`docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md\` — counterfactual controls and the requirement to separate correlation from causal contribution.

## 10. Claim ceiling

Until these tests run and replicate, do not claim that:

- HDC-LTC is superior to CfC, GRU/LSTM, EMA, Neural CDE or NIALIM;
- per-coordinate time constants improve sequence learning;
- HDC binding inside this neuron guarantees compositional generalization;
- the bridge is 8.5× faster end-to-end than a conventional CfC model;
- the cognitive loop has human-level or brain-equivalent temporal cognition;
- a high-dimensional state or integrated-information score demonstrates consciousness.

The defensible claim today is narrower: Symthaea has an active hypervector-state recurrent model with HDC-style coordinate-wise binding, state-dependent scalar continuous-time interpolation, HDC bridge projections, and an experimentally open question about whether those design choices earn their compute and complexity.

---

## References

- Hasani et al. (2022), [Closed-form continuous-time neural networks](https://doi.org/10.1038/s42256-022-00556-7), *Nature Machine Intelligence*.
- Liang et al. (2026), [Rederived Closed-Form Continuous-Time Neural Networks](https://doi.org/10.1109/TNNLS.2026.3669345), *IEEE Transactions on Neural Networks and Learning Systems*.
- Vergés et al. (2025), [Classification using hyperdimensional computing: a review with comparative analysis](https://doi.org/10.1007/s10462-025-11181-2), *Artificial Intelligence Review*.
