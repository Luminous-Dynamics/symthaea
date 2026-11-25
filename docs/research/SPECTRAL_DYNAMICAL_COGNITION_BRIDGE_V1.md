# Spectral–Dynamical Cognition Bridge (SDCB) v1

**Status:** design-only research protocol  
**Issue:** [#13672](https://github.com/Luminous-Dynamics/symthaea/issues/13672)  
**Scope:** falsifiable tests across analytic number theory, random-matrix statistics, statistical physics, dynamical systems, neural representations, and proof/search systems.

> **Central claim boundary.** This programme does not assume that the Riemann hypothesis (RH) explains brains, language models, intelligence, or consciousness. It asks whether specific mathematical structures that occur in number theory and physics can yield predictive or causal improvements in computational models—and whether any such result is specific to zeta/prime structure or is explained by generic spectral/dynamical effects.

## 1. Why this bridge is worth testing

There are real, but distinct, connections:

1. **Number theory ↔ spectral statistics.** The unfolded spacings of high Riemann-zeta zeros are associated with Gaussian-unitary-ensemble (GUE) statistics. This links analytic number theory to random-matrix theory and quantum chaos, but a statistical resemblance is not an identified physical Hamiltonian.
2. **Random matrices ↔ neural networks.** Random-matrix baselines have been used to distinguish bulk-like random directions from informative spectral deviations in trained weights. A 2025 NeurIPS study reports that both large and small singular-value deviations can matter in pretrained transformers; this is a concrete, directly testable precedent.
3. **Statistical physics ↔ brains.** Criticality, marginal stability, scale-free dynamics, and nonequilibrium models are active neuroscience hypotheses. A 2025 *Neuron* review/meta-analysis examined 140 datasets from 2003–2024 and argues for criticality as a possible computational setpoint; this remains a hypothesis to test, not a universal law to assume.
4. **Dynamical systems ↔ recurrent cognition.** Stability, attractors, Lyapunov spectra, Koopman operators, observability, controllability, and system identification can be measured in HDC/CfC state evolution and compared with sequence models.
5. **Physics-informed ML ↔ generalization.** Encoding symmetries, conservation laws, and governing dynamics can improve data efficiency and physical plausibility in suitable tasks. The question is which constraints transfer to cognitive workloads, not whether every physical analogy belongs in a neural architecture.

The RH itself concerns zeros of the zeta function and the distribution of primes. Its explicit-formula connections give a precise, number-theory-specific test bed. It does not state that neural systems implement zeta zeros, and the imaginary parts of zeros must not be casually relabelled as brain frequencies.

## 2. Research questions and preregistered hypotheses

The pilot is exploratory. A later confirmatory phase must freeze the corpus, analysis plan, software, and model versions before any hidden-set evaluation.

| ID | Hypothesis | Positive evidence would require | Falsifier / negative result |
|---|---|---|---|
| H0 | Zeta-derived features have no special advantage beyond matched generic spectral features. | No incremental performance after matched capacity, basis, and compute controls. | Not rejected by a simple correlation or in-sample fit. |
| H1 | Spectral descriptors of model state transitions predict transfer, robustness, or memory behaviour. | Incremental held-out prediction beyond task, size, and compute covariates; survives across seeds and families. | Predictive signal disappears under held-out families or matched random-matrix controls. |
| H2 | Near-critical dynamics improve some computational trade-offs. | A reproducible regime-dependent improvement in a preregistered outcome, with stability and perturbation costs reported. | No universal optimum; effects vary or disappear under finite-size and control tests. This outcome is informative. |
| H3 | Both tails of learned singular spectra can carry task-relevant information. | Targeted low- and high-tail interventions impair task performance more than matched bulk interventions, with controls for rank and conditioning. | Tail effects vanish after matched perturbations or fail to reproduce across architectures. |
| H4 | Koopman/operator features improve prediction or intervention fidelity for recurrent dynamics. | Better held-out multi-step prediction or causal intervention prediction at matched compute, not only a lower training error. | Gain disappears under parameter/compute matching or out-of-regime tests. |
| H5 | Prime/logarithmic or zeta-ordinate bases help tasks whose targets genuinely encode prime structure. | A reproducible gain over Fourier, wavelet, random-feature, shuffled-prime, and random-ordinate controls on held-out number-theoretic ranges. | No advantage over matched generic bases, or gains fail outside the discovery interval. |
| H6 | Spectral-mode interventions provide causal explanatory value, not just descriptive similarity. | Removing/replacing a named mode yields a preregistered, directionally predicted change, rescued by a matched restoration. | Spectral descriptors correlate with task outcomes but interventions do not support the proposed mechanism. |

No hypothesis is permitted to infer consciousness from spectral fits, criticality, integrated information, or model similarity.

## 3. Experimental tracks

### A. Zeta / prime explicit-formula track

Use a pinned, independently sourced finite set of zeta-zero ordinates and high-precision prime-counting or Chebyshev-function targets. The question is narrow: do zeta-derived bases help a learner predict prime-related oscillatory structure better than matched alternatives?

Controls:
- shuffle zero ordinates while preserving count and range;
- substitute GUE, GOE, Poisson, and moment-matched synthetic spectra;
- shuffle prime labels or use held-out arithmetic-progression moduli;
- compare the same number of Fourier, wavelet, random Fourier, and learned basis functions;
- vary truncation depth and numerical precision;
- keep the explicit-formula truncation error and source provenance visible.

The benchmark must not use an RH assumption to manufacture labels. Record whether each zero is numerically verified, what precision is certified, the exact data release, and how targets are computed. A model that memorizes known ordinates has not discovered a new theorem.

### B. Spectral anatomy of learned models

For HDC, CfC/LTC, HDC+CfC, reservoirs, and open transformer checkpoints, inspect:
- singular-value distributions and deviations from appropriate random-matrix nulls;
- left/right singular-vector localization and overlap with activation covariance;
- low-tail and high-tail importance;
- recurrent Jacobian eigenvalues, stability margins, and effective rank;
- how these measurements change across training, tasks, and interventions.

Use the 2025 transformer result as a direct comparator, not a number to borrow. Match checkpoint, layer, tokenizer, data, precision, and intervention budget. Compare tail ablations with matched random subspace ablation, rank-matched truncation, condition-number-matched perturbations, and restoration controls.

### C. Criticality and operating-regime sweep

Sweep recurrent gain/coupling through subcritical, near-critical, and supercritical regimes. Do not classify criticality using a single power-law fit. Use multiple diagnostics where applicable:
- finite-size scaling and susceptibility;
- autocorrelation and correlation length/time;
- branching or avalanche statistics with estimator sensitivity;
- response to perturbation and recovery;
- Lyapunov/stability indicators and dynamical range.

Measure memory capacity, delayed recall, sequence prediction, noise tolerance, adaptation, calibration, and energy/latency. The expected result may be a task-dependent Pareto frontier rather than one universal critical point.

### D. Koopman and control track

Fit linear latent evolution / Koopman approximations to nonlinear HDC/CfC trajectories. Compare with:
- direct CfC/LTC state prediction;
- reservoir computing / echo-state models;
- autoregressive and state-space baselines;
- small transformer sequence models;
- physics-informed and sparse system-identification baselines.

Evaluate multi-step prediction, regime-shift generalization, intervention prediction, observability, controllability, and uncertainty. Training reconstruction error alone is not a success criterion.

### E. Memory, proof, and compositional reasoning

Use controlled HDC binding/bundling and interference tasks, plus ECPCB-style proof-plan hypergraphs. The proof track must preserve the existing assurance boundary:
- graph accounting is not proof validity;
- a discharged edge is not a kernel certificate;
- spectral resemblance is not checker independence;
- theorem identity and trust closure remain separate.

Test whether spectral or dynamical diagnostics predict retrieval failure, unresolved obligations, route instability, or recovery after distractor injection. Compare against exact symbolic oracles, graph-aware retrieval, and LLM+Lean baselines.

### F. Biological neural data (later gate only)

Do not use human or EEG data in the v1 synthetic pilot. A later study requires source licensing, participant consent, ethics review, acquisition metadata, preprocessing provenance, and a preregistered analysis. Do not infer an individual’s mental state from spectral measurements.

## 4. Baseline matrix and fair comparison

Every result must report at least three comparison views:
1. **Capacity-matched:** comparable parameter/storage budget and representation size.
2. **Compute-matched:** comparable measured operations or accelerator-seconds, with precision and hardware recorded.
3. **Deployment-matched:** the same latency, memory, and energy budget where relevant.

Baselines:
- exact symbolic/oracle solver where available (upper-bound/reference, not a learned competitor);
- simple heuristic, linear/ridge, and MLP baselines;
- Fourier, wavelet, random Fourier, prime/log, and random spectral bases;
- reservoir/echo-state and Koopman/DMD models;
- HDC-only, CfC/LTC-only, coupled HDC+CfC;
- graph-aware retrieval and LLM+retrieval+Lean for proof tasks;
- frozen small open transformer checkpoints;
- random/untrained and spectrum-matched controls.

Do not compare Symthaea against a different task split, data budget, context length, precision, or hardware and call the result SOTA. Reproduce the comparator first; cite its exact version and report reproduction status.

## 5. Outcome vector (no compensating scalar score)

Report each dimension independently:
- exact task accuracy / normalized error / exact-match;
- OOD transfer by family and parameter regime;
- calibration, selective risk, and abstention quality;
- perturbation, adversarial, metamorphic, and causal-intervention robustness;
- memory capacity, interference, retention, and recovery time;
- temporal prediction horizon and uncertainty calibration;
- proof-plan consistency and unresolved-obligation rate (with proof validity separately reported);
- latency, peak memory, measured energy, and cost per accepted result;
- reproducibility across seeds, machines, and independent implementations.

For spectral claims, report the null model, fit range, estimator, finite-size sensitivity, uncertainty interval, and all attempted diagnostics. Do not choose the spectral statistic after seeing the test result.

## 6. Negative controls and anti-self-deception tests

Mandatory controls:
1. random orthogonal rotations preserving singular values;
2. spectrum-preserving eigenvector shuffles and matched random matrices;
3. phase-randomized inputs preserving power spectrum;
4. shuffled prime labels and shifted/wrong zero-ordinate sets;
5. GUE / GOE / Poisson / moment-matched synthetic spectra;
6. random Fourier and wavelet bases with identical feature count;
7. random subspace ablations matched for rank, norm, and conditioning;
8. time reversal, sequence permutation, and distractor injection;
9. no-op and known-correct / known-incorrect symbolic controls;
10. parameter-count, compute, and training-data matched baselines.

A negative control that fails to behave as specified invalidates the affected comparison; it is not silently discarded.

## 7. Statistical plan and provisional targets

Exploratory pilot: at least 5 independent seeds for each model/configuration cell, with deterministic generator seeds separately recorded. Confirmatory sample sizes are determined by a prospective power analysis, not selected after observing outcomes. Bootstrap at the task-family level; report per-family outcomes and confidence intervals; correct confirmatory multiple comparisons (for example, Holm). Separate tuning, validation, and hidden test data by task family or arithmetic/dynamical parameter ranges.

Provisional targets (not achieved results or universal standards):
- 100% exact recovery of known analytic controls and their expected dispositions;
- 100% detection of deliberately injected spectrum/identity/label mutations in the frozen control corpus;
- spectral diagnostic test–retest reliability of at least 0.90 on the same frozen input and environment;
- a claimed mechanism must replicate across at least 3 model seeds and 2 task families before being promoted beyond exploratory;
- a zeta-specific benefit must beat every matched basis/control, have a confidence interval excluding zero on the preregistered primary measure, and survive held-out arithmetic ranges;
- a criticality claim must show a measured trade-off frontier and finite-size sensitivity, not just a power-law-looking histogram;
- report energy only with calibrated hardware telemetry; otherwise report it as unavailable.

A target miss is a result, not permission to weaken the gate. Prospective targets may be revised only in a versioned protocol amendment made before the confirmatory run.

## 8. Source status and current RH context

The RH is a statement about the non-trivial zeros of the Riemann zeta function and prime-distribution error. The spectral-statistics connection to random matrices is well established as a research phenomenon; the Hilbert–Pólya spectral-operator proposal remains a conjectural route to understanding it.

As of 25 November 2026, the Clay Mathematics Institute lists RH as **Active**. A very large Lean formalization claiming RH and GRH for Dirichlet (L)-functions was posted on 20 November. Public reporting dated 24 November describes extensive machine checking, including kernel and independent-checker results, while noting that the final end-to-end comparator stage was still running, and that no readable human proof or peer review was available. The Lean community mirror is actively cleaning and modularizing the formalization. Therefore:
- do not label RH as officially settled;
- do not dismiss the machine-checking work;
- preserve exact theorem statement, source snapshot, checker diversity, environment, and run receipts;
- distinguish kernel acceptance, end-to-end statement comparison, human mathematical understanding, peer review, and institutional recognition.

The 15-root axiom-audit prototype in this repository is a small, separate fixture. It is not a verification of the large RH formalization.

## 9. Physics concepts: adoption priority

| Concept | Potential engineering use | Priority / guardrail |
|---|---|---|
| Nonequilibrium statistical mechanics and criticality | Regime sweeps, memory/response trade-offs, noise resilience | High; do not presume criticality is always optimal. |
| Dynamical systems, control, Koopman operators | Stability, prediction, interventions, recovery, system identification | High; validate under distribution and regime shifts. |
| Random-matrix theory and quantum-chaos statistics | Spectral nulls, weight/state diagnostics, common-mode controls | High for measurement; low for claims of shared mechanism. |
| Information theory and thermodynamics | Information retention, rate–distortion, energy per useful computation, dissipation | High; energy claims need physical measurement. |
| Symmetry, geometry, and conservation laws | Equivariant representations, invariance, conservation-preserving transitions | High when task symmetries are real and known. |
| Renormalization / multiscale methods | Predictive tests of how features and dynamics change across scale | Medium-high; require scale-held-out validation. |
| Topological data analysis | State-space and transition-graph structure, loop persistence | Medium; compare with simpler graph/spectral statistics. |
| Quantum mechanics / quantum information | Formal operator tools and genuinely quantum tasks | Low for ordinary cognition; no quantum-consciousness leap. |
| Relativity / gravitation | Coordinate-invariant causal modelling as an analogy | Low unless a concrete task requires it. |

## 10. No-runner execution sequence

1. Freeze and parse the JSON manifest; verify IDs, enum values, split disjointness, and required negative controls.
2. Hand-check analytic cases: prime labels, simple matrix spectra, linear stability examples, known finite-state memory transitions, and proof-graph accounting.
3. Implement generator and metric definitions in Rust only after the manifest stabilizes. Keep synthetic reference oracles small and deterministic.
4. Add unit tests for no-op, known-positive, known-negative, and mutation controls.
5. Reproduce one baseline per family before expanding the model matrix.
6. Execute the exploratory pilot and publish raw per-seed results, then preregister hidden/confirmatory runs.
7. Seek independent reproduction on a second machine and independent review of the statistical analysis.

**Current evidence state:** protocol, 34-task design manifest, and a dependency-free Rust analytic reference harness with 32 authored unit tests are present. A standalone receipt runner defines 29 analytic control cases and a JSON receipt schema. Neither Rust source file has been compiled or executed. No dataset generation, HDC/CfC or transformer model run, EEG analysis, RH proof verification, SOTA reproduction, or performance result is claimed by this artifact.


## 11. References and evidence boundaries

1. **Riemann zeros and random-matrix theory:** N. C. Snaith, *Riemann zeros and random matrix theory* (2009), [author manuscript](https://people.maths.bris.ac.uk/~mancs/papers/SnaithRiemann.pdf). Supports the statistical connection between zeta-zero spacings and random-matrix ensembles; does not identify a neural mechanism or a physical Hamiltonian.
2. **Numerical zero-spacing evidence:** A. M. Odlyzko, *On the Distribution of Spacings Between Zeros of the Zeta Function*, *Mathematics of Computation* 48 (1987), [DOI / JSTOR](https://doi.org/10.2307/2007890). Supports the numerical comparison with random-Hermitian-matrix statistics.
3. **A physical zeta-zero analogue:** *Riemann zeros from Floquet engineering a trapped-ion qubit* (2021), [Nature Scientific Reports](https://www.nature.com/articles/s41534-021-00446-7). Demonstrates an engineered trapped-ion experiment whose driven dynamics encode low zeta zeros. This is a designed laboratory analogue, not evidence that biological brains or LLMs implement zeta arithmetic.
4. **Neural-network random-matrix baseline:** M. Thamm, M. Staats, and B. Rosenow, *Random matrix analysis of deep neural network weight matrices*, *Physical Review E* 106, 054124 (2022), [DOI](https://doi.org/10.1103/PhysRevE.106.054124). Supports use of random-matrix null models for weight spectra.
5. **Transformer spectral comparator:** M. Staats, M. Thamm, and B. Rosenow, *Small Singular Values Matter: A Random Matrix Analysis of Transformer Models*, *NeurIPS 2025*, [proceedings abstract](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e1a8f3e897b25df1f82ebfc79991f5b4-Abstract-Conference.html). Reports learned deviations at both high and low singular-value tails and task impact under interventions. It is not a result for Symthaea.
6. **Brain criticality review/meta-analysis:** *Is criticality a unified setpoint of brain function?*, *Neuron* (2025), [PubMed record](https://pubmed.ncbi.nlm.nih.gov/40555236/) and [full-text record](https://pmc.ncbi.nlm.nih.gov/articles/PMC12374783/). Reports a meta-analysis of 140 datasets from 2003–2024 and argues for criticality as a possible computational setpoint; the protocol still tests alternative regimes and finite-size effects.
7. **Criticality methods caution:** *Theoretical foundations of studying criticality in the brain* (2024), [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11117095/). Discusses methodological/theoretical foundations, including nonequilibrium and percolation models.
8. **Physics-informed ML survey:** C. Meng et al., *When physics meets machine learning: a survey of physics-informed machine learning* (2025), [DOI / full text](https://doi.org/10.1007/s44379-025-00016-0). Surveys how physical priors can be incorporated through data, architectures, and optimization, and discusses limitations.
9. **Current institutional RH status:** Clay Mathematics Institute, [Riemann Hypothesis problem page](https://www.claymath.org/millennium/Riemann-Hypothesis/). The live page currently labels the problem **Active**.
10. **Public formalization and comparator setup:** [leanprover-community/riemann-hypothesis](https://github.com/leanprover-community/riemann-hypothesis). The community project documents theorem statements, comparator configuration, and its cleanup/readability goals.
11. **Dated report on the 20 November 2026 Lean submission:** [Cairn Commons status guide, updated 24 November 2026](https://cairn-commons.com/guides/riemann-hypothesis-proof-claim-november-2026). Used only for a dated summary of reported machine-check milestones and remaining human-understanding/peer-review work; it is not itself a formal verification receipt or institutional ruling.

When a source changes, preserve an archived copy, capture date, content digest, and the exact claim it supports. The live Clay status and the RH formalization status must be rechecked before future publication.

## Additional references and competitive comparators

- Staats, Thamm & Rosenow, “Small Singular Values Matter: A Random Matrix Analysis of Transformer Models,” NeurIPS 2025: https://proceedings.neurips.cc/paper_files/paper/2025/hash/e1a8f3e897b25df1f82ebfc79991f5b4-Abstract-Conference.html
- “Is criticality a unified setpoint of brain function?”, *Neuron* (2025): https://pubmed.ncbi.nlm.nih.gov/40555236/
- CoDBench (2024): https://pubs.rsc.org/en/content/articlehtml/2024/dd/d4dd00028e
- “An overview of Koopman-based control: From error bounds to closed-loop guarantees” (2025): https://doi.org/10.1016/j.arcontrol.2025.101035
- “Adversarial dynamical systems characterize when data-driven learning succeeds or fails” (2026): https://pubmed.ncbi.nlm.nih.gov/42448691/
- LeanGraphSearch live results and methodology: https://pluslabnlp.github.io/LeanGraphSearch/
- Goedel-Architect preliminary paper: https://openreview.net/pdf/23d889a527e56c9e2b43596e463b086dfe16d5a3.pdf
- Exgentic Open Agent Leaderboard: https://www.exgentic.ai/
- Akhunzada et al., “From benchmarks to deployment: a comprehensive review of agentic AI evaluation” (2026): https://doi.org/10.1007/s10462-026-11571-0


## 12. Extended benchmark governance and implementation links

The following supporting artifacts extend this protocol:

- [Industry and research benchmark landscape snapshot](SDCB_INDUSTRY_BENCHMARK_LANDSCAPE_2026_11_25.md): current external baselines, reported metrics, comparison caveats, and SDCB promotion gates.
- [Hidden evaluation and reproducibility protocol](SDCB_HIDDEN_EVALUATION_AND_REPRODUCIBILITY_V1.md): sealed holdouts, salted commitments, anti-leakage, exact system identity, independent evaluator and delayed reveal.
- [Dependency-free Rust analytic reference harness](../../scripts/research/sdcb_reference.rs) with [run instructions and limitations](../../scripts/research/SDCB_REFERENCE_HARNESS_README.md).
- [Machine-readable reference receipt runner](../../scripts/research/sdcb_reference_receipt.rs) and [receipt schema](SDCB_REFERENCE_CONTROL_RECEIPT_V1.schema.json).
- [Scientific-discovery benchmark interoperability](SDCB_SCIENTIFIC_DISCOVERY_INTEROPERABILITY_V1.md).

A key refinement from current Koopman/system-identification work is to evaluate *identifiability*, not just prediction accuracy. Each dynamical experiment must state persistent excitation, observability, sampling, noise, stationarity or regime assumptions, sequence length, and approximation error. Include adversarial non-identifiable trajectories and allow the correct result to be “insufficient evidence to identify the dynamics.”


### Identifiability and abstention track (added 25 November 2026)

This track upgrades system identification from a pure prediction task to a **know-when-not-identifiable** task. It is represented by the seventh pilot family, `identifiability_abstention`, with six task specifications:

- two distinct hidden dynamics that yield identical measurements under an unobservable sensor;
- rank-deficient input regressors that cannot identify a two-feature model;
- an observable positive control using independently varied initial states;
- finite/noisy/partially observed dynamics where structural observability does not guarantee practical identifiability at the available sample size;
- active sensor selection that distinguishes currently observationally equivalent hidden models;
- active input-probe selection that raises rank-deficient excitation to a full-rank design.

The standard-library Rust reference harness includes 2x2 observability-rank, input-regressor-rank and hidden-mode indistinguishability controls. It should return an explicit insufficient-evidence disposition on the unobservable/rank-deficient cases, not invent a unique model. A full-rank observability or excitation matrix is necessary for the relevant linear identification problem, but not sufficient to guarantee good finite-data estimation under noise or model mismatch.

This adopts a critical lesson from current Koopman work: report the assumptions and conditions under which a spectral/dynamical approximation is learnable, and include adversarial systems where the same observations are consistent with different hidden dynamics. Identifiability is task-relative: the observed output may be predictable even when the hidden mechanism is not identifiable. The system should preserve that limited prediction while refusing the stronger mechanistic claim, then propose a sensor or excitation intervention that would distinguish the alternatives. A model that knows what it does not know—and how to reduce that uncertainty—is more scientifically useful than one that always emits a unique model.

The pilot manifest now contains 34 design-only task specifications across eight families, 20 negative controls and 16 outcome metrics. No generated trajectory, fitted model or model performance has been measured.


## 13. Sequential scientific discovery and benchmark interoperability

SDCB now includes a sequential experiment-design family with four design-only tasks: budgeted multi-arm measurement, noisy/costed response-surface search, prior-misspecification recovery, and ranking of never-measured designs under held-out family shift. The pilot now contains 34 tasks across eight families, 20 negative controls and 16 outcome metrics.

Do not create an isolated scientific-agent benchmark and claim leadership. Compose with existing suites using a non-invasive parallel report:
- Autonomous Science Bench (ASB) for lab-like experiment selection and learning about unmeasured designs;
- FIRE-Bench for independently rediscovering verifiable published findings;
- SciAgentArena for interactive, stepwise-verified scientific research;
- AstaBench for broad discovery-workflow coverage and cost;
- SorryDB for real evolving Lean proof obligations;
- Goedel-Prover-V2 for high-capability formal proving;
- Open Agent Leaderboard / Exgentic for full-system capability and cost.

Keep upstream scores untouched. Report SDCB's parallel dimensions: hypothesis discrimination, experiment value, cumulative regret, surprise detection/recovery, unmeasured-candidate ranking/calibration, identifiability/abstention, provenance, replayability and total cost. Discovery and learning are separate gates; no single aggregate score can hide failure of either.

See [SDCB scientific-discovery benchmark interoperability](SDCB_SCIENTIFIC_DISCOVERY_INTEROPERABILITY_V1.md) for comparator scope, baseline policy and proposed integration sequence.
