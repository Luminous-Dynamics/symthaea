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
