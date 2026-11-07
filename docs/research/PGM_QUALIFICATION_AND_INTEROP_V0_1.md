# Probabilistic Graphical Model Qualification and Cross-Project Interoperability v0.1

**Status:** Research specification; not an implementation or qualification claim  
**Date:** 2026-11-07  
**Related:** Symthaea #10889; Mycelix #5910, #5601, #5632, #5648; Symtropy #3973, #4017, #4026, #4027, #3018; Sol-Atlas #582, #583

## 1. Executive decision

Do not create a second generic PGM crate.

The ecosystem already contains useful but specialized pieces:

- **symthaea-factor-graph:** discrete factor tables with sum-product and max-product message passing.
- **symthaea-fep:** action-conditioned hierarchical information gain over finite joint hypotheses.
- **Symthaea ETH-015 / ETH-016:** exact factor elimination over selected output algebras and an independent certificate direction. This is related to PGM inference but is not automatically a normalized probabilistic posterior engine.
- **Symthaea probabilistic HDC:** diagonal-Gaussian uncertainty attached to hypervectors. This is a representation/approximation layer, not a general graphical model.
- **Mycelix Terra Preta:** finite sequential Bayesian experimental design, with planned covariance-aware likelihoods, model discrepancy, robust design and simulation-based calibration (SBC).
- **Symtropy:** specialized uncertainty, correlated-sensor, coupled-fault and finite-horizon diagnostic work.
- **Sol-Atlas:** explicit event-factor composition and continuous spatial detection-likelihood work.

The priority is to **qualify existing engines, define interoperable typed contracts, and connect them through narrow adapters**. Do not replace deterministic physical simulators, provenance graphs, domain-specific observation operators, or decision policies with a universal probabilistic abstraction.

## 2. Research hypothesis

A useful next-generation inference capability is:

> Incremental, provenance-aware, dependence-conscious probabilistic inference that combines discrete hypotheses, continuous state estimates, temporal transitions, physical constraints, and explicit model discrepancy—while reporting exactly which parts of an answer are exact, approximate, uncalibrated, or unsupported.

The novelty is not using Bayesian networks, factor graphs, Kalman filters, message passing, variable elimination, particle filters, or probabilistic circuits. Those are established methods.

The research opportunity is to join:
1. **Evidence lineage:** source ancestry and measurement dependence remain inspectable.
2. **Inference correctness:** small cases have independent exact oracles and invalid inputs fail closed.
3. **Model adequacy:** parameter uncertainty is not confused with structural error or omitted mechanisms.
4. **Decision relevance:** information gain is a diagnostic; expected value of sample information, regret, safety and cost determine whether a probe is worthwhile.
5. **Incrementality:** local evidence/model changes trigger bounded recomputation without stale posterior reuse.
6. **Independent verification:** important inference outputs become replayable artifacts, not opaque scores.

## 3. Separation of concerns

These objects must not be conflated:

| Object | Meaning | Does not imply |
|---|---|---|
| Evidence artifact | Sourced observation, document, measurement, event or derived result | Independence, authenticity or truth |
| Provenance/dependence graph | How artifacts share sources, processing, instruments, people, batches or selection | A causal model |
| Statistical factor graph | Factorization of a joint probability distribution | Causal direction or validated physics |
| Causal graph | Model of causal assumptions and interventions | That observational associations identify causal effects |
| Physical model | Equations, constraints and solver state | That its parameterization is correct |
| Posterior | Conditional uncertainty under a specified model and evidence | Model adequacy or social legitimacy |
| Decision policy | Objective, utility/loss, constraints and decision authority | That the posterior alone determines what ought to be done |

A distinct hash, file, model name, sensor, modality, witness, or attestation is not sufficient evidence of statistical independence. Unknown dependence remains unknown. Provenance edges may constrain a statistical model, but provenance itself is not a probability distribution.

Causal interventions require explicit causal semantics; conditioning on a variable is not the same as applying do(variable = value).

## 4. Common model and evidence contract

A cross-project interface should describe the mathematical problem without forcing one storage or execution backend.

### 4.1 Typed variables

Minimum metadata:
- stable variable ID and semantic name;
- domain: finite categorical, integer, real-valued, vector, or versioned extension;
- cardinality or shape where applicable;
- units and coordinate frame for physical values;
- temporal index or support interval;
- optional physical/logical constraints;
- model version and source references.

No implicit conversion between categories, physical units, coordinate frames or time bases.

### 4.2 Typed factors

Each factor declares:
- stable factor ID and ordered variable scope;
- factor family and parameterization;
- domain and units;
- whether it is a normalized conditional likelihood, unnormalized potential, transition kernel, physical constraint, or prior;
- treatment of parameter and model-structure uncertainty;
- source, calibration and provenance references;
- dependence group and declared dependence assumptions;
- implementation/version digest;
- validity domain and known exclusions.

A factor potential is not automatically a normalized conditional distribution. A joint density cannot silently be consumed as a CPT.

### 4.3 Evidence inputs

Observation records bind:
- immutable evidence ID and content digest;
- observed variable(s) and value/distribution;
- event time, acquisition time, arrival time and clock basis;
- instrument, calibration and observation-operator identity;
- spatial, temporal and sample support;
- source ancestry and shared-dependence groups;
- missingness, censoring, saturation and quality status;
- units and coordinate frame;
- admissibility status for the specific inference task.

Missing observations are absent or explicitly missing; they are never encoded as zero-valued measurements by default.

### 4.4 Inference receipt

Every qualified result should emit a versioned receipt containing:
- query and target variables;
- canonical model digest and factor identities;
- evidence IDs/digests and provenance-closure digest;
- prior/model-family references;
- algorithm, configuration, build and numeric backend;
- exactness class: EXACT, APPROXIMATE, HEURISTIC, UNSUPPORTED;
- posterior or requested estimate and normalization/evidence mass when valid;
- solver status, convergence residual, iterations, tolerance and approximation diagnostics;
- Monte Carlo seed/RNG identity and effective sample size where relevant;
- numerical error estimates and known failure modes;
- calibration scope, held-out split and calibration status;
- sensitivity to priors, nuisance assumptions and model discrepancy;
- explicit claim ceiling and abstention reason.

A content digest binds identity; it does not prove the inference result. Independent replay or a sound checker is needed to validate the computation. A valid computation does not authenticate the evidence.

## 5. Solver portfolio and algorithm-selection rules

No single inference algorithm is appropriate for every model.

| Problem shape | Preferred first method | Qualification rule |
|---|---|---|
| Small finite discrete graph | Exhaustive enumeration / exact variable elimination | Exact known-answer differential tests |
| Acyclic factor graph | Sum-product / max-product tree message passing | Must match exact enumeration |
| Small treewidth discrete graph | Variable elimination / junction tree | Report induced width and intermediate table size |
| Loopy discrete graph | Damped or residual-scheduled loopy BP | Approximate only; disclose convergence and compare to exact small cases |
| Linear-Gaussian factors | Kalman / information-form Gaussian elimination | Covariance consistency and exact linear-Gaussian oracle |
| Nonlinear continuous factors | Linearization / smoothing; GTSAM is a strong comparator | Report linearization, initialization, local minima and covariance assumptions |
| Nonlinear/non-Gaussian temporal state | Particle filtering / SMC or explicitly qualified alternatives | ESS, degeneracy, Monte Carlo error and likelihood quality |
| Large differentiable discrete graph | Variational / differentiable BP methods | Bound approximation and calibration; compare exact subsets |
| Tractable generative distributions | Probabilistic circuits where structure supports the query | Verify circuit assumptions and query guarantees |

Algorithm selection must be explicit and report why a requested output is unsupported. A solver must never silently downgrade an exact request to an approximation.

### 5.1 Semantic requirements

- Zero evidence mass is an inconsistency/undefined-posterior state, not a uniform posterior.
- Exact zeros in likelihoods and potentials are allowed.
- Negative and non-finite probability potentials are rejected unless a distinct signed algebra explicitly permits them.
- Products of many small probabilities require log-domain or scale-tracked arithmetic.
- Independent per-variable max-product belief argmax is not a general joint-MAP guarantee. On a tree with tied max-marginals, local tie-breaking can select an assignment that violates a factor. A three-variable chain with A=B and B!=C has two valid MAP assignments, (0,0,1) and (1,1,0); selecting the same tied state at every variable yields (0,0,0) or (1,1,1), both with zero joint potential. Loopy BP adds approximation on top.
- A factor belief is computed from the factor and incoming messages; it is not generally the product of its variable marginals.
- Loopy BP convergence is not proof that its marginals are accurate.
- Bethe free energy is an approximation under declared assumptions, not a generic exact log-partition function.
- A parameter posterior cannot identify a missing causal mechanism without explicit model comparison/discrepancy work.
- Model-structure, parameter, latent-state, observation-noise and model-discrepancy uncertainty remain distinct unless a joint posterior justifies their combination.

## 6. Paradigm-shift research programme

### 6.1 Dependence-aware evidence fusion

Move from counting evidence records to modeling independent information pathways.

Benchmark shared source images, common sensors, shared calibration, common training data, annotator exposure, repeated sampling, batch effects and copied reports. Unknown ancestry must not default to independence. Evaluate false corroboration, claim-promotion risk, sensitivity to missing dependence edges, calibration and abstention.

Connect Mycelix evidence provenance with Symthaea evidence-dependence work (#10404). Do not invent an independence probability from a hash or a distinct identity.

### 6.2 Joint physical-state and sensor-state inference

Use a coherent transition-observation kernel such as:

P(s_next, e_next, y, hazard | model, s, e, action)

where s is physical state and e is sensor/instrument state.

This is stronger than independently updating a physical model and a sensor-health estimate. Persistent sensor drift can imitate a physical regime change; physical carryover can imitate measurement persistence. Preserve the joint posterior until the relevant dependence is resolved.

Start with the exact finite-state oracle in Symtropy's stochastic physical-state/sensor-state programme (#3018), then compare an explicitly named SMC/particle approximation. Keep deterministic physics replay as a separate source of physical-state trajectories.

### 6.3 Model discrepancy as a first-class hypothesis

Separate:
- process/physical parameter uncertainty;
- measurement noise and shared nuisance;
- observation-operator uncertainty;
- structural/model-family uncertainty;
- omitted process/discrepancy.

When all candidate models predict poorly, the answer should be MODEL_INADEQUATE, NOT_IDENTIFIABLE or ABSTAIN—not a concentrated posterior over the least-wrong available model.

Use prior/posterior predictive checks, held-out mechanism families, counterexample-guided model expansion, and independent discrepancy controls.

### 6.4 Decision-aware active inference

Expected information gain is not expected decision value.

Compare:
- nominal EIG;
- conditional EIG;
- expected value of sample information (EVSI);
- robust/lower-tail EVSI;
- minimax regret;
- explicit STOP / HUMAN_REVIEW / ABSTAIN;
- random, fixed, cost-only and domain-expert baselines.

A high-information measurement that cannot change a decision can be wasteful. A low-information measurement can be valuable if it prevents a high-cost wrong decision. The utility owner, budget, risk, authorization and reversible/irreversible effects remain outside the generic inference engine.

### 6.5 Incremental, content-addressed inference

Treat exact intermediate messages as immutable artifacts bound to:
- canonical input factor digests;
- variable domains and scopes;
- algebra/model semantics version;
- elimination order/schedule;
- numeric backend;
- checker identity.

A changed local factor should invalidate dependent results only. Reuse is allowed only after exact identity and checker-receipt validation. This is not a claim that caching or incremental inference is novel; compare against existing incremental Bayesian and factor-graph systems.

### 6.6 Tractable graph distributions

Investigate probabilistic circuits and graph circuits for models where exact conditional queries over structured graph subsets matter. The 2025 UAI probabilistic graph-circuit work is a research comparator, not a drop-in general solution: graph invariance, expressiveness and efficiency trade-offs must be tested.

### 6.7 HDC as a complementary representation

Keep probabilistic HDC's diagonal-Gaussian assumption explicit. HDC may be useful for associative indexing, candidate retrieval and compressed representations. It does not provide a full covariance model, a calibrated posterior over arbitrary graph variables, or a replacement for exact factor inference. Shared-variable/correlated-coordinate operations need explicit dependence handling; independent-Gaussian product formulas are not valid when operands are the same or statistically dependent.

## 7. Cross-project ownership

### Symthaea

Owns:
- canonical discrete factor graph and inference semantics;
- factor-graph correctness qualification;
- hierarchical state/parameter/model information gain;
- inference/decision separation and active information acquisition;
- HDC-to-PGM retrieval/representation adapters;
- shared inference receipts and exact small-model differential tests.

Do not claim PGM calibration from HDC similarity or a novelty score. Do not treat EIG as a complete expected-free-energy policy.

### Mycelix

Owns:
- evidence identity, ancestry, dependence and admissibility;
- distributed provenance and evidence exchange;
- domain-specific Bayesian model specifications for scientific claims;
- sequential experimental-design and posterior-calibration programmes;
- governance boundaries between posterior support and legitimate decisions.

The existing mycelix-desci Bayesian module needs separate source remediation (#5910). Do not use it as a qualified backend until its sampling, message passing, evidence likelihood, graph and d-separation semantics pass independent tests.

### Symtropy

Owns:
- deterministic physical solvers and replay;
- physical observation operators and engineering constraints;
- domain-specific correlated sensing and measurement uncertainty;
- fault diagnosis and sequential passive measurement decisions;
- calibrated coupling between physical state, sensor state and model discrepancy.

Probabilistic inference is a sidecar or decision-support layer. It must not silently introduce nondeterminism into the deterministic physics hot path or authorize physical actuation.

### Sol-Atlas

Owns:
- infrastructure/observation domain models and spatial/temporal support;
- event and spatial likelihood composition;
- integration with infrastructure data services;
- user-facing maps and visualizations of posterior distributions, uncertainty, provenance and unsupported regions.

The Leptos/WebGL renderer is not the inference engine. It should not turn uncertainty into a categorical certainty or hide unsupported geographic coverage.

## 8. External industry and research comparators

Use matched semantics, workloads, data and budgets. These projects solve different problem classes; a universal speed comparison is meaningless.

| Comparator | Strength / role | What to compare |
|---|---|---|
| pgmpy 1.1.x | Discrete Bayesian networks, structure/parameter learning and causal graph utilities | BN semantics, exact posterior, structure-learning quality, causal assumptions and reference datasets |
| GTSAM 4.3.x | Continuous nonlinear factor graphs, smoothing/mapping, incremental estimation and robotics | Matched continuous-state/sensor fusion tasks, batch vs incremental cost, covariance consistency and robustness |
| Exact enumeration / variable elimination | Independent small-instance oracle | Correctness, not runtime leadership |
| Probabilistic circuits / graph circuits | Tractable generative distributions for supported queries | Query tractability, circuit size, structural assumptions and incremental learning |
| Conventional Kalman / EKF / UKF / particle filters | Continuous-state and temporal estimation baselines | Calibration, consistency, latency, robustness to nonlinearity/non-Gaussianity |
| Domain-specific fault/reliability systems | Industrial diagnostic baseline | Fault isolation, false alarms, missed detection, cost and subgroup/domain transfer |

Pin exact package releases, code commits, options, model parameterizations, stopping criteria, CPU/GPU resources and datasets in every comparison. GTSAM's continuous-estimation workload must not be compared directly against a discrete Bayesian-network benchmark.

### Benchmark corpora and reference families

- Discrete BN: Asia, Sachs, Alarm, Insurance, Hailfinder, Win95pts and larger public networks where licensing permits.
- Exact factor-graph tests: enumerated trees, cycles, disconnected components, mixed cardinalities, zero likelihoods and high dynamic range.
- Continuous estimation: standard synthetic linear-Gaussian and nonlinear state-estimation cases; add public robotics datasets only after source/license and split audits.
- Sequential inference: independent-noise, correlated-noise, persistent sensor drift, physical-state carryover and out-of-family process shifts.
- Evidence dependence: shared source, copied reports, shared calibration, shared batch, shared preprocessing and blinded independent observations.
- Spatial inference: analytic infinite-strip and finite-domain detection integrals, irregular paths and support mismatch.

## 9. Scorecard and advancement targets

All numbers below are proposed qualification targets, not achieved results.

| Dimension | Metric | Initial gate |
|---|---|---|
| Exact posterior correctness | TV distance / max absolute error vs exhaustive oracle | Every accepted exact fixture within preregistered tolerance (e.g. 1e-10 for stable f64 fixtures) |
| Evidence mass | Partition function / log evidence error | Exact agreement on small reference cases; explicit zero-mass state |
| MAP correctness | Joint assignment and joint score | 100% match to brute-force joint maximum on bounded tests |
| Tree BP | Marginal error | Match exact enumeration for all tree fixtures |
| Loopy BP | Convergence, residual, posterior error | Always labelled approximate; publish non-convergence and worst error |
| Numerical robustness | Underflow/overflow, dynamic range, invalid inputs | No silent uniform fallback; invalid values reject |
| Calibration | Prior/posterior SBC and held-out proper scores | No systematic pathology across preregistered families; subgroup results shown |
| Predictive performance | Log score, Brier, CRPS/interval score as appropriate | Improve over strongest matched baseline on at least two independent mechanism families before superiority claim |
| Decision quality | Paired regret, EVSI prediction error, wrong-decision rate | Target at least 10% lower held-out 90th-percentile regret at matched cost/risk, with paired uncertainty interval excluding zero |
| Robustness | Worst-case/lower-tail risk and abstention | No hard-budget/eligibility violation; disclose ambiguity set |
| Incrementality | Recompute operations, cache validity, stale-message rejection | Local changes recompute only affected dependents; exact cold-run parity |
| Operational efficiency | Latency, memory, throughput, energy | Publish Pareto frontier vs exact and approximate baselines; do not optimize away correctness |

SBC must specify prior draws, data-generation process, test quantities, independent replications and multiplicity. Prior SBC checks inference implementation under the prior-predictive model; posterior SBC or conditional calibration addresses selected observed-data regions. Neither establishes real-world model adequacy.

Use proper scoring rules (log score, Brier, CRPS where appropriate), reliability diagrams and interval coverage. Expected calibration error alone is insufficient. Separate calibration from discrimination and sharpness.

## 10. Implementation sequence

### Phase 0 — immediate safety and inventory
- Track Mycelix legacy module remediation (#5910).
- Inventory call sites and public exports for all current probability modules.
- Pin source revisions and mark exact/approximate/experimental semantics in docs.
- Do not remove exports until downstream usage is checked.

### Phase 1 — qualify the Symthaea factor-graph core
- Fallible constructors and frozen/validated compiled model.
- Explicit zero-mass, finite-value, non-negative-potential and scope contracts.
- Exact enumeration/VE oracle for small finite graphs.
- Tree-BP differential tests and adversarial marginal-MAP vs joint-MAP controls.
- Correct factor beliefs and Bethe calculation, or remove unqualified output.
- Log-domain or scale-tracked messages.
- Distinct exact and approximate result types.

**First executable slice:** draft PR https://github.com/Luminous-Dynamics/symthaea/pull/10900 adds a bounded exhaustive joint-MAP reference under `symthaea-factor-graph::exact_map`. It provides an oracle for small assignment spaces and a concrete joint-MAP tie counterexample. It does not yet provide exact posterior marginals, log-partition evidence, a scalable solver, or arbitrary-precision arithmetic; compile/test and exact-head CI remain unverified.

### Phase 2 — bridge FEP and robust decisions
- Feed action-conditioned likelihoods into hypothetical posterior calculations.
- Preserve the existing hierarchical MI chain-rule identity.
- Extend to EVSI/regret and robust scenario uncertainty through the existing REA line; do not create a competing active-inference policy stack.
- Keep candidate posterior updates isolated from live belief/evidence state.

### Phase 3 — connect domain work
- Mycelix: source-bound dependence graph and calibrated posterior contract.
- Symtropy: exact joint physical/sensor-state reference, then approximate temporal inference.
- Sol-Atlas: factor-scope-safe event/spatial likelihood adapters and display layer.
- Cross-project adapters preserve units, time, scope, provenance and claim ceilings.

### Phase 4 — scalability and research frontier
- Incremental exact message reuse with checked content-addressed artifacts.
- Correlated continuous observations and state-space models.
- Explicit discrepancy and model-family uncertainty.
- Probabilistic circuits for supported large structured queries.
- GPU, SIMD and hardware acceleration only after correctness parity and profiling.

## 11. Release gates and evidence policy

- A source-level audit is not a test pass.
- Authored tests are not executed tests.
- A queued or stale CI run is not a pass; record the exact tested commit SHA.
- Benchmark fixtures must be frozen, versioned and split by independent mechanism/source groups.
- Preserve null and negative results.
- Exact and approximate algorithms must be compared on the same objective and data.
- Any model change invalidates receipts derived from the previous digest.
- No posterior is elevated to truth, causal proof, consciousness, physical qualification or governance legitimacy without separate evidence and authority.

## 12. Key external references

- Kschischang, Frey & Loeliger (2001), *Factor Graphs and the Sum-Product Algorithm*, IEEE Transactions on Information Theory.
- GTSAM: https://gtsam.org/ (current 4.3.0 release and documentation; continuous factor-graph/robotics comparator).
- pgmpy: https://pgmpy.org/ (1.1.x; discrete BN and causal modeling comparator).
- Probabilistic Graph Circuits, UAI 2025: https://proceedings.mlr.press/v286/papez25a.html.
- Rethinking Probabilistic Circuit Parameter Learning, AISTATS 2026: https://proceedings.mlr.press/v306/liu26a.html.
- Simulation-based calibration: Talts et al. (2018), https://arxiv.org/abs/1804.06788; Modrák et al. (2025), https://doi.org/10.1214/23-BA1404.
- NIST Sequential Bayesian Experiment Design: https://www.nist.gov/programs-projects/sequential-bayesian-experiment-design.

---

**Decision:** qualify what exists first. Build no new generic PGM package until factor-graph semantics, evidence/dependence contracts, and cross-project interoperability gaps are demonstrated by exact-oracle tests.


## 13. 2026 research-frontier update (2026-11-07)

The benchmark matrix should explicitly incorporate the following current comparisons rather than treating EIG or generic BP as the whole frontier.

### Constrained multi-step Bayesian experimental design

Guo et al., *Constrained Bayesian Experimental Design via Online Planning* (ICML 2026, PMLR 306), combine an amortized policy and posterior network with online multi-step lookahead using scenario trees. The benchmark reports information efficiency under dynamic budgets, varying costs and physical constraints.

Required comparator:
- nominal one-step EIG / EVSI;
- exact finite-horizon oracle on small cases;
- fixed-sequence and random eligible policies;
- amortized policy without online lookahead;
- scenario-tree online planning;
- robust / minimax-regret extension only where implemented faithfully.

Measure decision regret, information acquired, experiment cost, budget feasibility, tree expansion/search cost, posterior approximation error and out-of-family performance. Do not treat the paper's reported performance as a portable score without reproducing its tasks and protocol.

Reference: https://proceedings.mlr.press/v306/guo26g.html

### Partially observed dynamical systems and nested particle inference

Perez-Vieites et al., *Online Bayesian Experimental Design for Partially Observed Dynamical Systems* (ICML 2026, PMLR 306), estimate EIG and its gradient while marginalizing latent states using nested particle filters for online state/parameter inference.

Required comparator for stateful Symtropy and Mycelix tasks:
- exact finite joint-state oracle for tiny cases;
- particle filter for state estimation plus a clearly defined parameter update;
- nested particle-filter / SMC²-style online design;
- one-step EIG and finite-horizon scenario-tree planning;
- model-misspecification and persistent sensor-state controls.

Measure posterior error, EIG bias/variance, effective sample size, resampling/degeneracy, predictive log score, decision regret, memory, latency and total compute. A particle method does not become calibrated merely because it reports ESS.

Reference: https://proceedings.mlr.press/v306/perez-vieites26a.html

### Two-axis misspecification: model error and inference error

Callahan & Catanach, *On the Misinformation in a Statistical Experiment* (AISTATS 2026, PMLR 300), show that experiment rankings and classical information-based utility properties can fail under data-generating-model and inference-approximation misspecification. Tang, Sloman & Kaski, *Representative, Informative, and De-Amplifying* (AISTATS 2026), develop R-IDeA to address representativeness, information and error amplification.

The benchmark must keep separate:
1. correct generator + correct inference;
2. correct generator + approximate/biased inference;
3. shifted generator + exact inference under the wrong assumed model;
4. both model and inference error;
5. all supplied candidate models inadequate.

Primary outcomes are held-out decision loss, confidence calibration and false-decision rate—not EIG alone. This connects to Symtropy SIM-REL-030 (#3189) and counterexample-guided model-family refinement (#3171). Reproduce published objectives faithfully; label local analogues as inspired comparators.

References:
- https://proceedings.mlr.press/v300/callahan26a.html
- https://proceedings.mlr.press/v300/tang26d.html
- https://github.com/Luminous-Dynamics/symtropy/issues/3189
- https://github.com/Luminous-Dynamics/symtropy/issues/3171

### Industry release pinning

As of this audit:
- GTSAM 4.3.0 is the stable tagged release announced on 2026-09-18. The release notes describe faster inference, parallel/multifrontal elimination, expanded filtering and hybrid inference, and experimental CUDA optimization. Compare against this pinned release for reproducibility.
- A 4.3.2 version appears in the ROS package index on 2026-11-06, but that listing follows the repository development branch. Treat it as a development snapshot until an official stable tag/release is established; do not silently call 4.3.2 the stable release.
- pgmpy v1.1.2 is the latest listed stable release in the GitHub release list, dated 2026-04-30.

References:
- https://github.com/borglab/gtsam/releases
- https://index.ros.org/p/gtsam/
- https://github.com/pgmpy/pgmpy/releases

### Changed programme priority

The first executable shared-engine priority remains exact finite reference inference, not a new particle implementation. The next qualification layer should then be:

1. exact posterior/MAP reference;
2. tree and bounded-treewidth differential tests;
3. explicit independent-vs-correlated observation history;
4. separate generator-shift and inference-approximation error controls;
5. particle and amortized/scenario-tree baselines on the same tasks and budgets;
6. decision-regret and subgroup calibration under sealed mechanism-family shifts.

This sequence avoids tuning an approximate policy against an unqualified inference kernel.


The companion draft PR https://github.com/Luminous-Dynamics/symthaea/pull/10907 adds bounded exhaustive posterior marginals and a log-partition reference, with a direct tree sum-product differential test. It is stacked on #10900. Both reference paths use f64 arithmetic and explicit state-space bounds; neither is a scalable inference algorithm.

## 14. Exact rational probability reference and external benchmark protocol (2026-11-07)

### New exact reference path

Draft PR https://github.com/Luminous-Dynamics/symthaea/pull/10934 adds a rational coefficient domain to the exact semiring engine in `symthaea-complexity-lab`:

- `CheckedRational`: canonical reduced numerator / positive denominator; checked i128 arithmetic; cross-cancelled multiplication and division; explicit overflow and zero-divisor errors.
- `rational_probability::posterior_marginals`: exact partition function and per-variable posterior marginals through conditioned bucket variable elimination.
- Rejects negative probability potentials, treats exact zeros as impossible support, rejects zero partition/evidence mass, and verifies each returned marginal sums to exactly one.

This is exact relative to the finite factor tables and representable i128 intermediates. It is not arbitrary precision. The module is a reference and qualification mechanism, not a high-throughput inference algorithm. Its current posterior method performs one partition-function elimination and one additional elimination per variable state.

### Matched external baselines

| Track | External baseline | Compare on | Do not claim |
|---|---|---|---|
| Discrete Bayesian networks | pgmpy 1.1.2; published bnlearn example networks | CPD validation, posterior marginals, MAP, variable elimination and junction-tree queries | pgmpy does not represent continuous robotics factors like GTSAM |
| General discrete PGM inference | UAI 2014 / UAI 2022 competition problem sets | PR (partition/evidence), MAR (marginals), MPE/MAP, MMAP; preserve original problem/evidence/query files | Small synthetic tests do not establish public benchmark SOTA |
| Continuous, hybrid and incremental robotics | GTSAM 4.3.0 stable | Gaussian/nonlinear factor graphs, incremental updates, hybrid discrete-continuous inference, trajectory/sensor tasks | Rational discrete VE is not a like-for-like replacement for GTSAM's Lie-group and nonlinear optimization |
| Tractable exact repeated-query models | Probabilistic circuits and UAI-2025 Probabilistic Graph Circuits | Query latency, exactness, circuit size, graph-permutation invariance, model expressivity and training cost | Circuit tractability depends on structural restrictions; exact inference is not cost-free |
| Stateful partial-observation design | Symtropy #3018; ICML-2026 nested-particle online BED | Joint latent state/parameter inference, EIG bias/variance, ESS, decision regret, cost and model shift | A static factor graph cannot substitute for transition/sensor-state dynamics |
| Correlated observations and misspecification | Mycelix #5632/#5648; Symtropy #3189/#3171 | Joint likelihood correctness, SBC, discrepancy, calibration, worst-case decision regret | Good posterior algebra does not prove the likelihood model is adequate |

Pinned external sources:
- pgmpy v1.1.2 release (2026-04-30): https://github.com/pgmpy/pgmpy/releases
- pgmpy example models: https://pgmpy.org/documentation and https://github.com/pgmpy/example_models
- UAI 2014 task definitions (PR, MAR, MAP, MMAP): https://www.auai.org/uai2014/competition.shtml
- UAI 2022 competition benchmark sets: https://uaicompetition.github.io/uci-2022/results/benchmarks/
- GTSAM 4.3.0 stable release: https://gtsam.org/2026/09/23/gtsam-4-3-release.html
- UAI 2025 Probabilistic Graph Circuits: https://proceedings.mlr.press/v286/papez25a.html
- ICML 2026 online BED with nested particle filters: https://proceedings.mlr.press/v306/perez-vieites26a.html
- AAAI 2026 weighted model-counting variance: https://ojs.aaai.org/index.php/AAAI/article/view/39007

### Benchmark tiers and target gates

**Tier A — mathematical oracle (tiny finite instances)**
- Full enumeration, exact rational VE, and a separate independently coded checker.
- 100% exact equality for partition functions and every marginal on sealed cases.
- 100% exact normalization and nonnegative marginal support.
- Zero fabricated posterior on inconsistent evidence or invalid negative potentials.
- Zero unchecked overflow; every arithmetic/resource error is explicit.
- Tree BP must match exact rational reference after conversion to f64 within a preregistered absolute tolerance (initial target 1e-10).

**Tier B — public discrete networks**
- Start with Asia, Cancer, Earthquake, Sachs, Survey, Child, Alarm, Insurance, Water, Barley, Hailfinder, Hepar2, Win95pts, Pathfinder, Andes, Diabetes, Pigs, Link and Munin variants where formats / parameter sizes permit.
- Use small networks for exact rational references; use float/double exact algorithms and independent cross-checks on large networks if i128 rational bit growth is prohibitive.
- Freeze evidence/query sets. Report success rate, exactness, log-evidence error, TV distance / marginal L1, MAP score/assignment, runtime, peak memory, induced width, largest intermediate table and arithmetic bit lengths.
- Run tree, poly-tree, low-treewidth, high-treewidth, sparse/dense, deterministic, near-deterministic and zero-support families.

**Tier C — approximate and learned inference**
- Compare tree BP, loopy BP, junction tree, variable elimination, mini-bucket, importance/likelihood-weighted sampling, MCMC, particle filtering, variational inference and tractable probabilistic circuits only on matched query semantics.
- Report error against exact references: total variation, KL when finite and meaningful, log evidence, MAP regret, convergence rate, ESS/degeneracy, calibration and decision loss.
- Distinguish inference approximation error from model misspecification with a 2x2 factorial evaluation. Never score a method only by EIG if downstream decision quality can worsen.

**Tier D — domain transfer**
- Symtropy: joint stochastic physical state + persistent sensor state, exact finite-horizon oracle versus nested particles / SMC².
- Mycelix: batch/instrument/temporal/spatial covariance, discrepancy and prior/posterior SBC.
- Sol-Atlas: event composition, spatial/temporal evidence dependence, risk and resource decisions.
- Symthaea: active model/parameter/state inference and model-adequacy diagnostics.

Every published result should carry a frozen model/data/query manifest, seed(s), code SHA, compiler/toolchain, command, exactness level, timeout/memory budget, raw per-case outputs and a replayable result digest.

### What would count as an actual advance over SOTA?

1. **Correctness frontier:** outperform existing libraries on neither correctness nor scale by assertion; show exact parity on public semantics and zero false exact claims.
2. **Efficiency frontier:** for each workload family, produce a Pareto curve of latency, memory, error and model width. A target is at least 2x faster or 2x lower memory at matched error on a preregistered, representative slice; report regressions elsewhere.
3. **Calibration frontier:** improve posterior predictive / SBC calibration and robust decision regret under dependence and model-family shifts, with paired uncertainty intervals excluding zero.
4. **Anytime frontier:** return certified bounds or a clearly approximate posterior with calibrated error/abstention at any compute budget; never silently blur exact, approximate or unsupported.
5. **Evidence frontier:** an independent replay checker verifies exact small-instance results and each approximate method reports estimator/solver gap, convergence diagnostics and model-adequacy status.

These are proposed qualification gates, not current results. A 2x speed target is meaningful only after identical models, hardware, compiler modes, query sets, precision, warm-up policy and memory accounting are frozen.

### Research implications

Do not build a universal solver that tries to beat every specialized library. Build a typed inference compiler that:
- preserves probability, evidence, objective, and query semantics;
- compiles low-treewidth exact models to exact elimination;
- uses junction-tree / sparse elimination where structure permits;
- selects loopy/particle/variational/circuit methods only with explicit approximation and calibration contracts;
- routes continuous manifolds and incremental nonlinear problems to GTSAM-class backends;
- uses active experimentation to reduce uncertainty that matters to a decision;
- records evidence lineage, dependence, model discrepancy and abstention reasons.

The moat is not another implementation of sum-product. It is end-to-end, evidence-qualified inference and decisions across changing models, observations, sources and physical processes.


### Proof-carrying follow-up

Issue https://github.com/Luminous-Dynamics/symthaea/issues/10935 proposes exact rational posterior certificates by reusing the independent elimination replay framework in ETH-017 (#10203). This is a design/issue only; no rational posterior certificate or independent checker has been implemented yet.


## 15. Evidence-qualified factor and result contracts

A factor graph is only as trustworthy as its factor semantics. A shared engine must not accept an untyped list of arrays and imply that it knows whether entries are probabilities, likelihoods, constraints, utilities or duplicated observations.

### Typed factor contract

The eventual versioned interop schema should separate:

- **Identity:** factor ID, schema version, variable IDs and sorted scope.
- **Mathematics:** cardinalities, table ordering, coefficient domain, support/zero semantics, normalization claim (if any), and whether values are probabilities, likelihood potentials, deterministic constraints or utilities.
- **Conditioning:** explicit conditioning variables and event scope; no implicit assumptions of independence.
- **Provenance:** source/event IDs, artifact digests, producer/version, calibration receipt and rights/authority boundary.
- **Dependence:** dependence-group IDs, shared latent-cause references, batch/instrument/site/temporal covariance or explicit joint-kernel ID.
- **Time and validity:** event time, model version, validity interval, temporal ordering and source snapshot.
- **Model adequacy:** declared mechanism family, discrepancy hypotheses, OOD / unsupported-domain state and abstention conditions.

A conditional factor `P(B|A)` is not interchangeable with a joint factor `P(A,B)`. A composite process likelihood is not to be multiplied again by its already included component likelihoods. If a factor is only a likelihood up to proportionality, do not report it as a normalized probability without a partition calculation.

### Typed inference-result contract

Every result should declare:
- query semantics: PR / MAR / MPE / MAP / MMAP / conditional / intervention / EIG / EVSI / decision;
- method and coefficient domain;
- exactness level: exhaustive finite, exact elimination, exact tree-message passing, numerical exact under stated tolerance, or approximate;
- approximation diagnostics: residual, non-convergence, particle ESS/degeneracy/MCSE, variational bound/gap, or missing diagnostic;
- normalization/evidence mass and support status;
- model adequacy and discrepancy status;
- provenance/input digest, code/build identity and replay/checker receipt;
- abstention or unsupported-domain reason.

A method that has no valid diagnostic for a required field must return `not_available` or abstain. It must not fill missing uncertainty with zero or a confidence-like scalar.

### Cross-project ownership

- **Symthaea:** mathematical inference engines, exact/approximate algorithm registry, exactness types, bounded references, numerical diagnostics and algorithm compatibility.
- **Mycelix:** signed source/evidence provenance, actor/organization lineage, dependence metadata, uncertainty decomposition, covariance, discrepancy, SBC and calibration evidence.
- **Symtropy:** dynamic physical state, sensor state, transition kernels, risk/cost/horizon, intervention and policy evaluation.
- **Sol-Atlas:** spatial/event observation domains, CRS and geometry, factor conditioning scopes, event-graph likelihood composition, hazard/resource decisions and operational evidence.

Use small versioned adapters with explicit information loss. The frontend/rendering layer must not run inference. No project should copy a second Bayesian engine or reinterpret another project's factor tables silently.

### Consequence for benchmark design

Benchmark not only the numeric posterior but also semantic integrity:
- source/factor provenance completeness;
- dependence-group and conditioning-scope preservation;
- double-counting negative controls;
- model-misspecification and inference-approximation factorial controls;
- time-order and stale-evidence rejection;
- exact/approximate/unsupported classification correctness;
- replayable input/result digests and abstention.

This extends standard PGM numerical benchmarks into evidence-qualified decision benchmarks, without claiming that a correct posterior proves true source data or an adequate causal/physical model.
