# Complexity, Chaos and Multiscale Dynamics — Knowledge and Validation Programme

**Status:** Research programme and implementation audit, 2026-11-16. This document does not claim new scientific results. It defines what Symthaea should know, how that knowledge should be qualified, and how existing implementations should be tested before their outputs are used for cognitive interpretation or engineering decisions.

## Executive decision

Yes: Symthaea should become materially stronger in complexity science, nonlinear dynamics, chaos theory, information dynamics, multiscale organization, reservoir computing and artificial-life theory.

Do not solve this by indiscriminately adding more papers, equations or domain labels to a knowledge base. Build a **validated scientific knowledge capability**: definitions, assumptions, source provenance, executable reference cases, uncertainty, counterexamples, and explicit limits on what each measurement can support.

The immediate opportunity is not to claim that Symthaea is beyond the state of the art. It is to turn scattered domain modules into a coherent, falsifiable dynamical-systems laboratory whose predictions and measurements can be independently checked.

## Current inventory and scientific debt

The current public tree already includes:

- `crates/core/symthaea-core/src/physics/chaos_dynamics.rs`: Lyapunov estimates/spectra, time-delay embedding, correlation-dimension estimation, Kaplan–Yorke dimension, bifurcation diagrams and classical Lorenz/Rössler/Hénon/logistic systems.
- `src/dynamics/stability_analysis.rs`: Jacobian/eigenvalue analysis, Lyapunov analysis, fixed points, bifurcation classification and contraction analysis.
- `crates/core/symthaea-core/src/hdc/reservoir.rs`: an echo-state/reservoir implementation with ridge-style readout.
- `crates/core/symthaea-core/src/hdc/fractal_consciousness.rs` and `crates/domains/symthaea-spore/src/fractal.rs`: hierarchical/fractal-inspired topology and recursive cognitive layers.
- `src/hdc/information_theory.rs` and `crates/domains/symthaea-information-theory/`: information-theoretic primitives.
- Existing stability and dynamics benches, including `benches/dynamics_stability.rs` and `benches/dynamics_ode.rs`.

This inventory means the next step is validation, integration and improved epistemic structure—not another disconnected implementation of Lorenz or a second generic chaos detector.

### Audit items to resolve before trusting interpretations

1. In `chaos_dynamics.rs`, the maximal-Lyapunov path advances the base trajectory with RK4 but advances the tangent perturbation with a first-order Euler update using a numerically differentiated Jacobian. The perturbation is initialized along one coordinate. Compare this implementation with a consistently integrated Benettin/Wolf-style tangent evolution and an independently implemented QR spectrum.
2. The correlation-dimension estimator constructs and sorts all pairwise distances, giving quadratic memory/work pressure. It chooses a fixed quantile-derived scale window and does not apply a Theiler window to exclude temporally adjacent pairs. Add scale-window diagnostics, temporal exclusion, uncertainty and a bounded alternative for larger series.
3. Bifurcation classification currently relies on a tolerance-based count of distinct attractor samples and uses a high-period threshold as a proxy for chaotic onset. This can confuse long transients, noise, quasiperiodicity and high-period orbits. Treat it as an exploratory diagram heuristic until cross-validated against continuation, Floquet multipliers or independent diagnostics appropriate to each system.
4. In `symthaea-spore/src/fractal.rs`, `aggregate_fractal_dim = 1 + normalized_error_entropy` is an error-distribution complexity index, not a geometric or dynamical fractal dimension. Rename it in a versioned API or clearly deprecate the misleading label while preserving compatibility.
5. The `strange_loop_active` rule in that module is a threshold heuristic over top-level error and bottom-level integration. It does not demonstrate causal closure, recursive constraint production or an experimentally established strange loop. Label it as a candidate indicator, then design intervention tests.
6. In `stability_analysis.rs`, a positive finite-sample maximum Lyapunov estimate alone must not be treated as proof that a bounded system is chaotic. Nor does a negative estimate establish a stable percept or any consciousness property. Require stationarity/transient checks, boundedness, estimator convergence, appropriate surrogate tests and a scoped interpretation.
7. Any use of a universal “edge-of-chaos” target (including a fixed Langton lambda) must be treated as a model-specific hypothesis. There is no universal critical parameter or single criticality optimum across different architectures and tasks.

These are audit findings from source inspection, not claims that every downstream experiment is invalid. Preserve useful measurements but tighten names, assumptions and claim scope.

## Knowledge model Symthaea should maintain

Represent each knowledge item as a claim with explicit epistemic metadata:

- Stable claim identifier and exact proposition.
- Claim class: definition, theorem, empirical result, model assumption, engineering heuristic, conjecture or open question.
- Source type: peer-reviewed paper, standard, preprint, implementation, benchmark report or internal measurement.
- Full bibliographic metadata, DOI/URL, publication/update date and access date.
- System/domain, parameter regime, units, data provenance and assumptions.
- Evidence level, independent replication status and known counterexamples.
- Executable fixture or derivation when one exists.
- Confidence and scope; explicit statements of what the item does **not** imply.
- Review/expiry date for fast-moving work.

Do not turn a preprint abstract into a settled fact. Keep internal estimates distinct from published reference values and third-party replication.

## Core subject map

### 1. Nonlinear dynamical systems

Symthaea should distinguish:

- Discrete maps, continuous ODEs, delay differential equations, PDEs, hybrid systems, stochastic systems and non-autonomous systems.
- Local linear stability, global asymptotic stability, Lyapunov stability, contraction and input-to-state stability.
- Fixed points, periodic orbits, quasiperiodicity, transient chaos, attractors, riddled basins, multistability and bifurcations.
- Deterministic chaos from stochasticity, measurement noise, nonstationarity and finite-sample artefacts.
- State observability/reconstructability from prediction accuracy.

### 2. Chaos and predictability

Treat chaos as a set of testable properties, not a visual impression:

- Maximal and full Lyapunov spectra, finite-time Lyapunov exponents and convergence across windows/initial conditions.
- Covariant Lyapunov vectors where interpretation of expanding/contracting directions matters.
- Kolmogorov–Sinai entropy bounds/estimates where their assumptions apply; do not substitute Shannon entropy of a one-step output.
- Recurrence plots and recurrence quantification analysis.
- Delay embedding, false-nearest-neighbour analysis and mutual-information delay selection.
- Attractor geometry, correlation dimension and Kaplan–Yorke dimension as different estimands; never assume they are identical.
- Shadowing, numerical integration error, finite precision, and forecast horizon measured in Lyapunov times.
- Surrogate-data tests against linear stochastic and phase-randomized alternatives.

A robust chaos classifier should return **inconclusive** when data length, stationarity, embedding adequacy or estimator convergence is insufficient.

### 3. Complexity and information dynamics

Extend beyond static entropy:

- Shannon entropy, conditional entropy, mutual information and transfer entropy with estimator-bias controls.
- Active information storage, information transfer and information modification.
- Partial information decomposition with explicit redundancy/synergy definitions and estimator sensitivity.
- Statistical complexity and predictive-state representations.
- Effective network inference and causal intervention analysis.
- Multiscale entropy and complexity profiles, retaining scale-dependent results rather than collapsing them into one “complexity score.”
- Distinguish algorithmic complexity (generally not computable exactly), Lempel–Ziv estimators, statistical complexity, geometric complexity and computational cost.

The December 2025 *Physics Reports* review “Information theory for complex systems scientists: What, why, and how” is a key entry point for this map. It covers information dynamics, statistical complexity, information decomposition and effective network inference.

### 4. Criticality, tipping points and adaptive regimes

Symthaea should study bifurcation-, noise- and rate-induced transitions separately.

Early-warning signals—such as rising variance, autocorrelation or critical slowing down—are conditional indicators, not universal predictors. Their failure modes include nonstationarity, changing external forcing, false positives, insufficient sampling and transitions without classical critical slowing down.

Use multiple independent indicators and calibrated false-alarm/missed-detection rates. Measure lead time and warning reliability under held-out forcing/noise regimes. Compare against simple trend, persistence and domain baselines.

Do not assume “edge of chaos” maximizes intelligence, learning, integration or resilience. Estimate task-conditioned trade-off curves and identify when a claimed optimum disappears under ablation or distribution shift.

### 5. Reservoir computing and continuous-time learning

Compare current CfC/LTC and reservoir approaches with:

- Echo-state networks, deep reservoirs and next-generation reservoir computing.
- Neural ODE/CDE and state-space models where the problem warrants them.
- Physics-informed learning and data assimilation where governing equations or partial observations are available.
- Simple linear, autoregressive and persistence baselines that expose spurious advantage.

Measure autonomous rollout stability and long-horizon distributional fidelity, not only teacher-forced one-step RMSE. Compare memory capacity, computational cost, energy, robustness to sampling-rate changes and online adaptation.

### 6. Multiscale geometry and artificial life

Separate at least four concepts:

- A recursively nested architecture.
- A graph with scale-free or heavy-tailed degree distribution.
- A geometric fractal with a measurable scaling regime.
- A dynamical system exhibiting scale-invariant statistics.

These are not interchangeable. A recursive tree is not automatically a fractal; a power law on a short range is not proof of scale invariance.

For artificial-life research, operationalize autopoiesis as causal organization in which a network of processes regenerates components and boundary conditions that sustain that network. Resilience after perturbation is useful but is not by itself autopoietic closure. Study material/energy budgets, boundary maintenance, repair, viability, reproduction, variation and evolution separately. Do not infer biological life or subjective experience from simulated self-organization.

## Recent literature and standards watch (checked 2026-11-16)

The following items should be imported as qualified knowledge, not promoted to universal truths:

1. **Information theory for complex systems scientists: What, why, and how** — *Physics Reports* 1148, 2025-12-08, DOI: [10.1016/j.physrep.2025.09.007](https://doi.org/10.1016/j.physrep.2025.09.007). A broad tutorial/review on information dynamics, statistical complexity, information decomposition and effective network inference.
2. **Classification using hyperdimensional computing: a review with comparative analysis** — *Artificial Intelligence Review*, published 2025-03-17, DOI: [10.1007/s10462-025-11181-2](https://doi.org/10.1007/s10462-025-11181-2). Important because it highlights inconsistent benchmark practices in HDC and the need for standardized comparisons.
3. **IEEE CIS Task Force on HDC/VSA** — [official overview](https://cis.taskforce.ieee.org/hdcvsa/). Useful map of compositional representations, tensor-product representations, holographic reduced representations, hardware, neuroscience and theory connections.
4. **Quantum hyperdimensional computing: a foundational paradigm for quantum neuromorphic architectures** — *Nature* journal article, 2026; [DOI/article](https://www.nature.com/articles/s44335-026-00064-6). Treat as a proposed quantum-HDC architecture, not as demonstrated quantum advantage and not as post-quantum cryptography. Quantum computing and cryptography designed to resist quantum attacks are different topics.
5. **Scalable Symbolic Reasoning with Matrix-Based Brain-Inspired Representations and Vector-Space Acceleration** — DATE 2026, DOI: [10.23919/DATE69613.2026.11539261](https://doi.org/10.23919/DATE69613.2026.11539261). Reports generalized holographic reduced representations with non-commutative matrix binding and a vector-space acceleration path. Its reported throughput/energy comparison is a paper-specific result; reproduce on matched hardware and workloads before using it as an engineering baseline.
6. **Emerging opportunities and challenges for the future of reservoir computing** — *Nature Communications*, 2024, DOI: [10.1038/s41467-024-45187-1](https://doi.org/10.1038/s41467-024-45187-1). A useful theory/algorithm/experimental agenda for reservoirs; prediction of chaotic trajectories is not equivalent to long-term exact trajectory prediction.
7. **Tipping points and cascading transitions: methods, principles, and evidence** — 2026 review, DOI: [10.1093/nsr/nwag478](https://doi.org/10.1093/nsr/nwag478). Distinguishes bifurcation-, noise- and rate-induced transitions and reviews limitations of classical early-warning signals.
8. **Experimental probes of autopoietic self-maintenance** — *BioSystems* 268, October 2026, DOI: [10.1016/j.biosystems.2026.105928](https://doi.org/10.1016/j.biosystems.2026.105928). Particularly relevant for distinguishing causal autopoietic closure from a measured resilience score.
9. **NIST IR 8547, Transition to Post-Quantum Cryptography Standards** — final, 2026-10-27: [NIST publication](https://csrc.nist.gov/pubs/ir/8547/final). Use it to inform actual deployment/migration planning; experimental HHLF ideas must not delay established PQC adoption.
10. **NIST IR 8610** — final, 2026-05-14: [NIST publication](https://csrc.nist.gov/pubs/ir/8610/final). Documents nine additional-signature candidates advanced to the third round: FAEST, HAWK, MAYO, MQOM, QR-UOV, SDitH, SNOVA, SQIsign and UOV.

This list is a dated snapshot. Review primary sources and later revisions before each research release.

## Benchmark programme

### A. Ground-truth dynamical systems

Create a small, deterministic, CPU-only reference suite with published parameter regimes and explicit integration conventions:

- Stable linear systems with known eigenvalues.
- Limit-cycle oscillator (e.g. van der Pol) with transient and parameter sweeps.
- Logistic map across fixed-point, periodic and chaotic regimes.
- Hénon map.
- Lorenz-63 and Rössler continuous flows.
- Quasiperiodic torus signal.
- Stochastic coloured-noise and linear autoregressive negative controls.
- Coupled-map lattices / Kuramoto oscillators for collective dynamics.
- One controlled nonstationary or forced system for regime-shift tests.

For every fixture, freeze equations, parameters, initial conditions, precision, integrator, step size, burn-in, observation interval and expected reference bands with citations. Store the raw trajectory and a digest. Never tune tolerances after seeing the candidate result.

### B. Estimator qualification

| Measurement | Qualification requirement |
|---|---|
| Leading Lyapunov exponent | Agreement with an independent method/reference band; convergence across dt, transient, window and perturbation |
| Full Lyapunov spectrum | Stable ordering, QR/reorthogonalization checks, volume-contraction/sum rules where applicable |
| Correlation dimension | Explicit scale-region fit, Theiler exclusion, sensitivity profile and uncertainty |
| Kaplan–Yorke dimension | Derived from a qualified spectrum; report assumptions and finite-sample uncertainty |
| Bifurcation locations | Parameter continuation/reference comparison; classify ambiguity rather than forcing a label |
| Chaos detection | Positive controls + stable/periodic/stochastic/nonstationary negatives; controlled false-positive rate |
| Critical-transition warnings | Lead time, precision/recall, false-alarm rate and missed transitions across forcing/noise classes |
| Information dynamics | Bias-controlled estimators, nulls, sample-size curves and estimator sensitivity |
| Fractal scaling | Scaling-range diagnostics, finite-size effects, competing-model tests and confidence intervals |

Initial acceptance targets (not current results):
- Zero false “chaos” classifications on the frozen stable/periodic negative-control suite at the declared operating point.
- At least 95% detection on a pre-registered positive-control suite where the reference regime is unambiguous.
- Qualified Lyapunov estimates within pre-registered reference bands on every eligible system.
- Every dimension estimate ships with scale-window, fit residual, sample count and uncertainty diagnostics.
- No “complete” result after budget exhaustion, numerical overflow or inadequate data.
- Any estimator whose classification changes under a small reasonable configuration perturbation is marked fragile/inconclusive.

### C. Forecasting and control

Measure:
- One-step error and autonomous rollout error separately.
- Forecast horizon in Lyapunov times, not just raw steps.
- Attractor/statistical fidelity: invariant-distribution distance, power spectrum, recurrence statistics and boundedness.
- Calibration/coverage for probabilistic forecasts.
- Response to perturbations, unseen parameters, sampling-rate changes and missing observations.
- Control success, control effort and safety constraint violations where control is evaluated.

Baselines must include persistence, linear/AR models, a tuned ESN/NG-RC, a conventional recurrent model and a strong task-specific state-space/neural baseline where feasible. Compare only under a frozen data and compute budget. Report seed distributions and statistical uncertainty.

### D. Multiscale/HDC ablations

For identical problems compare:
1. Flat HDC.
2. Hierarchical HDC without cross-scale feedback.
3. Fractal-inspired topology without HDC binding.
4. HDC + continuous-time dynamics.
5. HDC + continuous dynamics + measured multiscale graph.
6. Full proposed HHLF candidate.

Ablate dimension, precision, graph depth, branching factor, coupling, update rate and noise. Measure retrieval, compositional generalization, erasure robustness, adaptation, stability, memory, energy and compute. A full system wins only if its incremental benefit survives these ablations.

## Paradigm shifts worth testing

1. **From scores to trajectories:** preserve state evolution and uncertainty; do not collapse rich dynamics into one “consciousness/complexity” scalar.
2. **From one global regime to local regime maps:** different subsystems may be stable, oscillatory, critical or chaotic at the same time.
3. **From static attractors to viability landscapes:** study which states can be maintained under finite resource and perturbation budgets.
4. **From prediction alone to intervention:** test whether the model predicts the effect of controlled perturbations, not merely correlations in a trace.
5. **From one scale to cross-scale causal transfer:** identify which scale leads, constrains or destabilizes another through interventions and lag-aware information measures.
6. **From a fixed reservoir to controlled adaptive reservoirs:** change coupling/readout only under bounded update rules, with rollback and held-out evaluation.
7. **From vague emergence to operational autopoiesis:** test regenerative causal closure, material/energy constraints and lineage.
8. **From static HDC vectors to structured, non-commutative composition:** evaluate GHRR/matrix binding for sequences, directed graphs and hierarchy against ordinary HRR/FHRR/TPR, while accounting for matrix cost and acceleration.
9. **From “edge of chaos” slogans to task-conditioned Pareto surfaces:** optimize forecast horizon, learning, robustness, energy and stability jointly; do not assume one universal optimum.
10. **From model claims to evidence-carrying knowledge:** every important scientific conclusion links to raw fixtures, executable derivations, source revisions and explicit counterexamples.

## Work sequence that does not depend on CI runners

### W0 — Knowledge and naming repair
- Land this programme as the scientific map.
- Identify every field/API that uses “fractal dimension,” “chaos,” “criticality,” “strange loop,” “attractor” or “consciousness” as a label.
- Assign definitions, scope, source and evidence-level metadata.
- Do not silently change serialized/public API semantics; use versioned names/deprecations.

### W1 — Numerical reference harness
- Add fixed deterministic fixture generators and frozen reference metadata.
- Implement an independent, high-accuracy offline oracle for low-dimensional canonical systems.
- Compare existing Lyapunov, dimension and bifurcation estimators against it.
- Add adversarial controls and uncertainty reporting.

### W2 — Multiscale information and causal tests
- Add controlled synthetic networks and coupled dynamical systems.
- Compare static MI, transfer entropy, active information storage and intervention results.
- Quantify finite-sample bias and avoid causal claims from observational transfer entropy alone.

### W3 — HDC/continuous-time integration
- Build a narrow research adapter between HDC representations and dynamical state models.
- Run the six-way ablation above with matched compute and data.
- Keep the experimental adapter out of the default cognitive path until it demonstrates reliable benefit.

### W4 — Artificial-life and HHLF bridge
- Define operational viability, repair, boundary maintenance and reproduction tests.
- Couple a multiscale HDC state to a bounded dynamical graph.
- Verify self-maintenance under component loss and constrained resources; distinguish designed recovery from endogenous regeneration.

### W5 — Release qualification
- Publish environment, source SHA, config digest, seeds, raw JSONL, analysis tool digest and uncertainty.
- Independent replication before any SOTA claim.
- Maintain a public claim/retraction ledger.

## Comparison policy

Do not claim to beat “the industry” in complexity science as a single category. Compare on named tasks and exact regimes:

- Nonlinear dynamics: numerical integrators and established Lyapunov/attractor toolkits.
- Chaotic forecasting: persistence/linear, ESN/NG-RC, tuned recurrent networks and state-space baselines.
- HDC: HRR, FHRR, MAP/BSC/TPR/GHRR and matched neural/symbolic baselines.
- Complex-systems analysis: established information-theoretic, recurrence, network and statistical-physics methods.
- Artificial life: autopoietic/ALife models with explicit resource, boundary and lineage protocols.

A result can be better in latency or energy but worse in forecast horizon or statistical fidelity. Publish a Pareto comparison and all failed/unsupported runs. A matched SOTA claim requires the same task, data access, train/test split, compute/hardware budget, evaluation protocol and uncertainty treatment.

## Security and consciousness boundary

- Complexity, chaos, fractal geometry, unpredictability and high-dimensionality are not cryptographic hardness assumptions.
- HHLF outputs must never be used as a key generator, entropy source, nonce source, KEM, signature or integrity primitive.
- Continue using independently reviewed, standardized cryptography for actual security. PQC migration must follow current NIST guidance independently of HHLF research.
- Neither high complexity nor simulated self-maintenance proves sentience or consciousness. Any consciousness-related interpretation requires independent construct validation and careful non-claims.

## Selected primary references

- IEEE CIS Task Force on HDC/VSA: https://cis.taskforce.org/hdcvsa/ (official task-force overview: https://cis.taskforce.ieee.org/hdcvsa/)
- Vergés et al., “Classification using hyperdimensional computing: a review with comparative analysis” (2025): https://doi.org/10.1007/s10462-025-11181-2
- “Information theory for complex systems scientists: What, why, and how,” *Physics Reports* 1148 (2025): https://doi.org/10.1016/j.physrep.2025.09.007
- “Emerging opportunities and challenges for the future of reservoir computing,” *Nature Communications* (2024): https://doi.org/10.1038/s41467-024-45187-1
- “Quantum hyperdimensional computing: a foundational paradigm for quantum neuromorphic architectures” (2026): https://www.nature.com/articles/s44335-026-00064-6
- “Scalable Symbolic Reasoning with Matrix-Based Brain-Inspired Representations and Vector-Space Acceleration,” DATE 2026: https://doi.org/10.23919/DATE69613.2026.11539261
- “Tipping points and cascading transitions: methods, principles, and evidence” (2026): https://doi.org/10.1093/nsr/nwag478
- “Experimental probes of autopoietic self-maintenance,” *BioSystems* 268 (2026): https://doi.org/10.1016/j.biosystems.2026.105928
- NIST IR 8547, final 2026-10-27: https://csrc.nist.gov/pubs/ir/8547/final
- NIST IR 8610, final 2026-05-14: https://csrc.nist.gov/pubs/ir/8610/final

---

**Decision:** build Symthaea's complexity/chaos knowledge as an evidence-qualified, executable capability. The first technical win is to make existing estimators trustworthy and their outputs scientifically interpretable; the next win is to test whether HDC + continuous dynamics + measured multiscale organization adds a reproducible capability that simpler models do not.



## Frontier literature addendum — verified 2026-11-16

The following work changes Symthaea's research priorities. The findings are paper-specific, not general performance guarantees.

### 1. Generalized holographic reduced representations (GHRR), DATE 2026

The DATE 2026 paper *Scalable Symbolic Reasoning with Matrix-Based Brain-Inspired Representations and Vector-Space Acceleration* reports non-commutative matrix binding for GHRR, intended to encode order, directionality, sequences, graphs and hierarchies without permutation logic. It then proposes vector-space flattening and a custom complex-valued ASIC datapath. The paper reports 1.36x–1.56x throughput and 16.2x–18.6x energy-efficiency improvements against its PyTorch/RTX 4090 comparison.

**Implication for Symthaea:** build an isolated GHRR benchmark adapter and test whether non-commutative binding improves temporal/graph compositionality over current HRR/FHRR/TPR/MAP/BSC representations. Compare memory, dimension, numerical stability, expressivity, retrieval, training and energy. Reproduce the paper-specific acceleration claim only with equivalent hardware/workloads; it is not a blanket HDC speed claim.

Reference: https://doi.org/10.23919/DATE69613.2026.11539261

### 2. Quantum hyperdimensional computing (QHDC), 2026

The 2026 QHDC paper reports a quantum-native mapping of hypervector operations and a physical implementation on a 156-qubit IBM Heron r3 processor for symbolic analogical reasoning and supervised classification. It reports a near-500x simulation speedup against selected variational quantum circuit / quantum SVM baselines, while explicitly distinguishing current quantum utility from demonstrated quantum advantage over the best classical algorithms. It also identifies LCU/OAA bundling circuit depth and gate count as important scaling constraints.

**Implication for Symthaea:** this is a quantum-computing architecture direction, not post-quantum cryptography. Use it to expand the representation/hardware research map, but do not claim quantum advantage, deploy a quantum dependency, or infer PQC security. Replication must compare against strong classical HDC and conventional ML, not only selected QML baselines.

Reference: https://www.nature.com/articles/s44335-026-00064-6

### 3. HDC benchmarking warning, 2025 comparative review

The 2025 HDC classification review compares multiple HD/VSA methods over seven commonly used datasets and a 121-dataset UCI subset. It notes that inconsistent benchmarks had hindered comparison, and reports that off-the-shelf methods can surpass HDC approaches in some classification settings.

**Implication for Symthaea:** avoid comparing only against weaker or poorly tuned baselines. Use matched splits, multiple seeds, strong conventional ML baselines, equal hyperparameter budgets, and report quality and compute/energy. Hyperdimensional robustness or few-shot behavior may be the differentiator, but must be measured for the task.

Reference: https://doi.org/10.1007/s10462-025-11181-2

### 4. Information dynamics for complex systems, Physics Reports 2025

The 2025 review covers Shannon and relative entropy, mutual information, active information storage, transfer entropy, information decomposition, statistical complexity and effective network inference. It also discusses limitations: data requirements can be substantial, estimation assumptions matter, and algorithmic/Kolmogorov complexity is generally not computable exactly; Lempel–Ziv is a heuristic rather than a universal complexity meter.

**Implication for Symthaea:** extend from static entropy features to time-resolved information flow, storage, modification and multivariate structure. Use bias-corrected estimators, surrogate controls, sample-size curves, confidence intervals and interventions before causal claims. Do not turn a single entropy value into a universal complexity or consciousness score.

Reference: https://doi.org/10.1016/j.physrep.2025.09.007

### 5. Critical transitions and tipping cascades, National Science Review 2026

The 2026 review distinguishes bifurcation-induced, noise-induced and rate-induced tipping and reviews early-warning indicators based on critical slowing down alongside their limitations. It emphasizes interactions and cascades between coupled subsystems, while noting data quantity, heterogeneity and methodological robustness as continuing challenges.

**Implication for Symthaea:** represent local subsystem regimes and cross-system coupling explicitly. Benchmark early-warning lead time, false alarms and missed transitions under different forcing/noise regimes. A single “distance from edge of chaos” scalar is not an adequate phase map.

Reference: https://doi.org/10.1093/nsr/nwag478

### 6. Autopoietic self-maintenance, BioSystems 2026

The 2026 article *Experimental probes of autopoietic self-maintenance* emphasizes the difference between causal autopoietic closure and its formal/experimental proxies; its reaction-network example distinguishes causal closure from measured resilience.

**Implication for Symthaea/Symtropy/HHLF:** survival after perturbation, repair scripts, homeostasis scores or an animation continuing to run are not enough. Artificial-life experiments need to test whether a network of processes regenerates the components and boundary conditions that sustain the network, under explicit resource and intervention constraints.

Reference: https://doi.org/10.1016/j.biosystems.2026.105928

### 7. Reservoir computing research agenda, Nature Communications 2024

The 2024 perspective surveys theory, algorithms and experimental realization of reservoir computing, and describes unresolved challenges for large-scale adoption. It recognizes chaotic time-series prediction as a promising application while calling for more work on precision, speed, interpretability and practical systems.

**Implication for Symthaea:** compare current ESN/reservoir and CfC/LTC systems across matched chaotic and non-chaotic regimes. Measure autonomous rollout and distributional fidelity, not teacher-forced one-step error alone.

Reference: https://doi.org/10.1038/s41467-024-45187-1

### 8. PQC migration is an independent engineering track

NIST IR 8547 was finalized on 2026-10-27 as transition guidance for moving from quantum-vulnerable cryptographic algorithms to post-quantum standards. NIST IR 8610 (2026-05-14) reports nine additional signature candidates advancing to a third evaluation round.

**Implication:** continue PQC migration on standardized, reviewed algorithms independently of HHLF/QHDC research. Quantum-computing research and cryptography designed to resist quantum adversaries are different disciplines.

References:
- https://csrc.nist.gov/pubs/ir/8547/final
- https://csrc.nist.gov/pubs/ir/8610/final

### Revised research priorities

1. Validate existing estimators before expanding feature count.
2. Add GHRR/non-commutative binding as a narrow comparative representation experiment.
3. Add a qualified information-dynamics layer with estimator-bias and surrogate controls.
4. Build task-conditioned phase maps for stability, criticality, chaos, resource use and robustness.
5. Test cross-scale causal influence with interventions, not correlational graphs alone.
6. Test autopoietic closure as an independent artificial-life hypothesis.
7. Keep QHDC, HHLF cryptography and standardized PQC deployment in separate categories.

**Status:** literature watch and research implications only. No result in this addendum has been reproduced by Symthaea.



## Additional frontier findings — verified 2026-11-16

These papers materially refine the benchmark baselines and acceptance criteria. Their results are claims from the cited studies, not reproduced Symthaea results.

### A. Full Lyapunov-spectrum reliability — November 2026

**A Framework for the Assessment of Lyapunov Spectrum Computation Methods for Nonlinear Systems of Increasing Dynamical Complexity**, *Communications in Nonlinear Science and Numerical Simulation* 163 (November 2026), DOI: [10.1016/j.cnsns.2026.110493](https://doi.org/10.1016/j.cnsns.2026.110493).

The study proposes a reliability-oriented assessment using full-spectrum error, trace/divergence consistency error and determinant-consistency error. It classifies methods as reliable, conditionally reliable or diagnostic-only, and warns that agreement on the largest exponent may still coexist with a poor reconstruction of the full spectrum.

**Immediate change to Symthaea's qualification contract:**

- Compare every estimated exponent in the spectrum, not only the maximum.
- For systems with a known Jacobian, compare the sum of exponents against the long-time average divergence (trace of the Jacobian) under matching conventions.
- Check determinant/volume-evolution consistency for the tangent propagator where a suitable reference is available.
- Run QR/reorthogonalization sensitivity tests and report transient, timestep, window and initial-condition sensitivity.
- Do not compute or publish Kaplan–Yorke dimension as qualified when the spectrum is only diagnostic.
- Label results Reliable, ConditionallyReliable, DiagnosticOnly or Inconclusive, with machine-readable reasons.

This is a particularly high-value addition to the existing chaos-dynamics audit.

### B. Next-generation reservoir computing — January 2026

Cestnik and Martens, **Next-generation reservoir computing for dynamical inference**, *Chaos* 36(1), 013115 (2026), DOI: [10.1063/5.0302319](https://doi.org/10.1063/5.0302319).

The paper introduces a scalable NGRC implementation using pseudorandom nonlinear projections of time-delay embedded inputs. It reports attractor reconstruction and bifurcation-diagram estimation with partial/noisy measurements, and reports that small amounts of training noise can improve long-horizon autonomous stability.

**Benchmark implication:** NGRC must become a mandatory baseline for Symthaea's CfC/LTC and reservoir experiments. Compare exact feature budgets, trainable parameter count, training data, inference cost, autonomous rollout stability and distributional fidelity. Test whether the noise-regularization benefit persists on held-out parameter regimes rather than only the authors' reported cases.

### C. Complex Ginzburg–Landau reservoir — October 2026

Chong, Sun and Xu, **Reservoir computing with complex Ginzburg–Landau lattice dynamics for chaotic time-series prediction**, *Chaos, Solitons & Fractals* 211 (October 2026), article 118873, DOI: [10.1016/j.chaos.2026.118873](https://doi.org/10.1016/j.chaos.2026.118873).

The study reports a 16-complex-node CGLE lattice reservoir with R-squared greater than 0.999 on Lorenz next-step prediction. It compares with ESN, NGRC and linear AR, and notes that the Lorenz next-step task is near-linear. Its own ablations report task-dependent benefit from CGLE nonlinearity and a memory–responsiveness trade-off controlled by dissipation.

**Benchmark implication:** add CGLE-reservoir results where reproducible implementation details are available, but do not let next-step Lorenz R-squared stand in for chaotic forecasting capability. Require autonomous rollout, forecast horizon in Lyapunov times, attractor/statistical fidelity, parameter extrapolation and non-Lorenz systems.

### D. VSA for object-centric program synthesis — October 2026

Joffe and Eliasmith, **Vector symbolic algebras for the Abstraction and Reasoning Corpus**, *npj Unconventional Computing* 3, article 54 (published 2026-10-01), [article](https://www.nature.com/articles/s44335-026-00100-5).

The paper uses VSAs for object-centric program synthesis and reports 10.8% ARC-AGI-1 Train and 3.0% ARC-AGI-1 Eval, as well as 94.5% Sort-of-ARC, 83.1% 1D-ARC and 20.5% ConceptARC. The authors explicitly describe the ARC-AGI results as preliminary and far from state of the art; DSL coverage is a material limitation.

**Benchmark implication:** this is a useful architectural comparator for Symthaea's HDC + symbolic/program reasoning, not an SOTA result to copy. Add object-centric scene decomposition, explicit transformation DSL coverage, compositional generalization, sample efficiency, search cost and held-out operator composition. Report the trade-off between interpretable symbolic programs and raw task coverage.

### E. Hardware-aware sparse HDC — ISLPED 2026

Oh et al., **HyperSPACE: Sparse-Adder-Compatible Encoding for Efficient Hyperdimensional Computing on Digital CIM Arrays**, ISLPED 2026, DOI: [10.1145/3816440.3818512](https://doi.org/10.1145/3816440.3818512).

The reported design uses AND-binding, sparsity-tuned projection vectors and adaptive binary quantization on 128×64 digital SRAM compute-in-memory arrays. The authors report up to 13.03× lower encoding energy and up to +3.82% classification accuracy across four benchmarks.

**Benchmark implication:** add sparse/quantized binding as an HDC hardware-efficiency axis. Compare recall/accuracy and robustness at equal memory and energy budgets; do not treat the paper's encoding-energy result as a general end-to-end energy improvement or as a software CPU result.

### F. Generalized holographic reduced representations — distinguish the related papers

Two closely related 2026 research outputs should be tracked separately:

1. **Scalable Symbolic Reasoning with Matrix-Based Brain-Inspired Representations and Vector-Space Acceleration**, DATE 2026, DOI: [10.23919/DATE69613.2026.11539261](https://doi.org/10.23919/DATE69613.2026.11539261).
2. Chung et al., **Vector-Space Projection and Unified Complex-Valued Acceleration for Scaling Matrix-Based Brain-Inspired Representations**, ISLPED 2026, DOI: [10.1145/3816440.3818598](https://doi.org/10.1145/3816440.3818598). An arXiv version was posted 2026-11-12: [arXiv:2611.16363](https://arxiv.org/abs/2611.16363).

Both explore non-commutative matrix binding for ordered/graph/hierarchical structures and vector-space/hardware acceleration approaches. Keep their venues, claims and baselines distinct when citing or reproducing them.

### G. Quantum HDC — physical feasibility is not quantum advantage

Cumbo et al., **Quantum hyperdimensional computing: a foundational paradigm for quantum neuromorphic architectures**, *npj Unconventional Computing* 3, 21 (published 2026-05-04), DOI: [10.1038/s44335-026-00064-6](https://doi.org/10.1038/s44335-026-00064-6).

The work reports a QHDC implementation tested on a 156-qubit IBM Heron r3 processor. Its own hardware-noise discussion is important: at dimension 128, the physical execution averaged 54.75% weighted F1 with a transpiled Hadamard-test depth of 510; a hardware-aware dimension-32 configuration achieved 68.59% weighted F1 with depth 126, but still required about 52 minutes of QPU time per reported execution. The authors explicitly state that the primary result is physical feasibility of the mapping, not an immediate performance advantage over mature classical HDC.

**Benchmark implication:** treat QHDC as a separate hardware research axis. Compare it with strong classical HDC and conventional ML on identical tasks, including total wall time, QPU time, circuit depth, shots, error mitigation, energy and result quality. Do not infer PQC security or general quantum advantage.

### Updated baseline matrix

| Symthaea research task | Required comparison set |
|---|---|
| Full Lyapunov spectrum | Independent Benettin/QR, variational equations, trace/divergence and determinant/volume checks |
| Chaotic forecasting | Persistence, AR, ESN, NGRC, CGLE reservoir where reproducible, current CfC/LTC, strong recurrent/state-space models |
| HDC representation | HRR, FHRR, MAP/BSC/TPR, GHRR, sparse/quantized HyperSPACE-like encoding, conventional learned embeddings |
| Symbolic / object-centric reasoning | Explicit DSL/program synthesis, VSA-based scene/object representations, strong conventional neurosymbolic and task-specific baselines |
| Quantum HDC | Classical HDC and ML, plus hardware-noise and resource accounting; never quantum baselines alone |
| Critical transitions | Persistence/trend, CSD-only, nonlinear diagnostics, network cascade models and ensemble warning approaches |

**Updated principle:** compare the strongest available baseline under a matched experimental budget. A new architecture should be retained only when its contribution survives ablation, held-out regimes, uncertainty analysis and independent reproduction.



## Applied multiscale systems: architected materials and modular structural grammars

Symthaea's complexity knowledge should include engineered multiscale structures, not only cognitive/physical dynamics. This is a concrete application domain for HDC/VSA compositional reasoning and generative design.

### Scientific framing

Distinguish:
- **Hierarchical architecture:** components nested across scales; no claim of scale invariance.
- **Graded architecture:** density, topology, material or geometry varies spatially to meet local requirements.
- **Modular metamaterial:** global mechanical response depends on module interfaces, contact and constraints as well as each cell's properties.
- **Fractal geometry:** a statistically supported scaling regime over a stated scale interval; recursive construction alone is insufficient.
- **Free-form modular structure:** a global continuous/irregular shape assembled from a bounded, repeatable interface vocabulary.

The 2025 Nature Materials perspective identifies cross-scale design and characterization over multiple length and timescales as an important frontier. The 2026 generative-lattice review calls for process-aware, human-in-the-loop search combining physics-based constraints and data-driven acceleration. A September 2026 modular-metamaterial review emphasizes that inter-module interaction governs global mechanical response.

### Symthaea research role

- Encode module catalogs, connector families, orientation, allowed interfaces, assembly sequence and service constraints compositionally.
- Benchmark HRR/FHRR/MAP/BSC/TPR and GHRR/non-commutative binding for assembly graphs, sequence constraints and held-out module compositions.
- Explore hierarchical HDC for cell/panel/module/bay/system levels, retaining scale-specific state rather than collapsing it into one complexity score.
- Propose diverse design candidates and explain constraint conflicts.
- Use uncertainty-aware active learning to recommend the next Symtropy simulation or physical coupon experiment.
- Keep human-authored expressive infill and accessibility preferences as explicit requirements; do not optimize inhabitants toward one aesthetic.

### Required structural-design baselines

- Hand-authored parametric grammars.
- Conventional graph search and evolutionary optimization.
- Topology optimization and robust multiobjective design.
- Bayesian/surrogate-assisted optimization.
- Conventional structural embeddings or graph-learning models where appropriate.
- Standard connector/spaceframe and voxel assembly grammars.

### Evidence boundary

Symthaea may generate and rank candidate geometry, but it does not certify strength or safety. Symtropy must supply qualified physical/structural analysis, and physical coupons plus applicable inspection/qualification evidence are needed before real-world claims. Graph connectivity, cycle rank, HDC similarity and visual fractality are not substitutes for stress, buckling, fatigue, fracture, joint and manufacturing analysis.

References:
- 3D architected materials across scales (Nature Materials, 2025): https://www.nature.com/articles/s41563-025-02119-8
- Generative lattice design for AM (Materials & Design, 2026): https://doi.org/10.1016/j.matdes.2026.115431
- Modular discrete mechanical metamaterials (IJMS, 2026): https://doi.org/10.1016/j.ijmecsci.2026.111915
- NASA ARMADAS: https://www.nasa.gov/automated-reconfigurable-mission-adaptive-digital-assembly-systems-armadas/
- NASA modular in-space truss technology: https://technology.nasa.gov/patent/LAR-TOPS-359



## Applied multiscale systems: free-flow modular structural grammars — 2026-11-16

Symthaea should understand FFMFS as an applied multiscale-structure research domain. It should not treat generated geometry or HDC similarity as mechanical evidence.

### Design representation

Represent a candidate structurally as a typed, hierarchical graph:
- module/bay identity and generation level;
- member role, length class, orientation and section/material tag;
- connector family, port orientation, tolerance and interface fit;
- triangular face / multi-point attachment topology;
- assembly parent, sequence, tool/grasp access and service ports;
- environmental/pressure boundary, utility and occupant-authored infill as separate layers;
- evidence links to geometry digest, solver assumptions, manufacturing process, inspection and coupon tests.

HDC/VSA may encode and retrieve these structures, but the authoritative representation remains the typed graph and its verified constraints. Explore GHRR/non-commutative binding for ordered assembly and graph composition only as a comparative representation experiment.

### Scientific guardrails

- Connected graph, high cycle rank or visually complex morphology does not prove kinematic rigidity.
- Face-attached tetrahedral cells are a candidate ideal pin-jointed rigid-module grammar: child cells attach through three shared non-collinear vertices. This removes a one-point hinge in the ideal bar model, but not real joint compliance, local stress, buckling, fatigue or manufacturing defects.
- Structural claims must come from Symtropy's qualified analysis and physical evidence, not from HDC retrieval, fractal dimension or shape aesthetics.
- Fractal geometry requires a statistically supported scaling regime over a stated interval. Recursive generation alone is not enough.
- Preserve the distinction between primary structure, services, pressure/environmental boundary, secondary support and expressive infill.

### Required competitive baselines

Commercial tools already cover sophisticated implicit modeling, lattice generation, topology optimization and analysis:
- nTop: implicit geometry, graph/TPMS/stochastic lattices, multi-objective topology optimization, structural/thermal analysis and CAD/CAE/AM integration.
- Autodesk Fusion: generative design with manufacturing constraints such as additive, milling, 2-axis cutting and casting.
- Altair Inspire: structural topology/topography/gauge optimization, FEA, motion and additive manufacturing analysis.
- Materialise 3-matic/Magics: lattice and conformal mesh design, data optimization and AM preparation.
- NASA ARMADAS and JPL modular truss projects: reusable voxel and self-aligning truss module assembly, including robot construction.

Symthaea should not claim that its HDC generator beats these systems at generic lattice generation. The research hypothesis is whether a typed multiscale grammar plus interface reasoning, assembly planning, evidence links and occupant-controlled infill gives better system-level results.

### Symthaea tasks

1. Compare HRR/FHRR/MAP/BSC/TPR and GHRR for module/connector graph composition and ordered assembly reasoning.
2. Compare parametric grammars, conventional graph search/evolutionary methods, topology optimization and Bayesian/surrogate-assisted design.
3. Generate diverse Pareto candidates and expose constraint conflicts rather than returning one opaque optimum.
4. Use uncertainty-aware active learning to select Symtropy simulations and physical coupons.
5. Hold out entire topology families, connector families, load-case families and manufacturing process families to prevent leakage.
6. Keep accessibility, egress, service access, acoustic/sensory choices, privacy, cultural expression and occupant agency as explicit design requirements.

### Current research/engineering links

- Symtropy programme: https://github.com/Luminous-Dynamics/symtropy/blob/main/docs/FREEFLOW_MODULAR_FRACTAL_STRUCTURES.md
- Geometry and module grammar: https://github.com/Luminous-Dynamics/symtropy/tree/research/freeflow-modular-fractal-v0/research/freeflow-structure-lab
- Linear-elastic truss reference evaluator: https://github.com/Luminous-Dynamics/symtropy/tree/research/freeflow-modular-fractal-v0/research/truss-fem-lab
- Symthaea research issue: https://github.com/Luminous-Dynamics/symthaea/issues/12519

**Evidence boundary:** Symthaea proposes and reasons over candidate grammars; Symtropy evaluates declared physical/structural models; physical coupons and applicable independent engineering qualification determine whether a result can be used in real structures.
