# PHYS-CHAIN-002 Addendum: Relational-Historical Dynamics Benchmark v1

**Status:** research design / protocol addendum. No physical experiment or Chain Theory derivation is claimed.  
**Parent:** PHYS-CHAIN-002 / Symthaea #8923  
**Related owners:** #8919 (shared generative grammar), #8941 (order/time/thermodynamics), #8964 (discrete time crystals), #9858 (similarity-weighted historical learning), #8967/#8969 (concealed-target protocol/synthetic harness), #9926 (gaze-specific localization), Symtropy #2870 / PR #2871 (SIM-REL).

## 1. Purpose

Build one falsifiable comparison programme for three ideas that are easy to conflate:

1. **Generative relational grammar:** reusable typed production rules from which multiple worlds/models can be constructed.
2. **Construction/read history:** the order in which a particular network or model is built and observed.
3. **Cross-system historical influence:** the separate hypothesis that prior occurrences in similar, otherwise isolated systems alter later systems.

The programme tests whether each mechanism contributes predictive value beyond conventional explanations. It does not assume that Alex Wissner-Gross's Chain Theory entails Sheldrake's morphic resonance, or that a shared grammar is a literal ontology of the universe.

## 2. Source-grounded boundaries

### 2.1 Chain Theory

Alex Wissner-Gross's 24 October 2026 announcement states the axiom:

> Nothing exists but finitely many maximally entangled pairs of finite-dimensional systems, some shared, known only by distilling them.

The announcement defines a distillation as joining two systems into a maximally entangled pair in one step. It describes construction/read order as accumulating because each distillation builds on preceding ones.

Source: https://theinnermostloop.substack.com/p/chain-theory

For this benchmark, Chain Theory is a source-bounded candidate formal model. Only exact source-defined objects, moves, and order variables may be represented. A classical graph edge is not an entanglement witness; a shared graph vertex is not a shared quantum subsystem; a graph rewrite is not automatically a distillation; a construction count is not automatically physical time or thermodynamic entropy.

Keep formal results separate from physical identifications. In particular:
- the fine-structure algebraic expression is distinct from its conjectural identification with physical alpha(0);
- construction order is distinct from an operational clock;
- count monotonicity is distinct from a derivation of physical entropy production;
- entanglement is not a controllable signalling channel by itself.

### 2.2 Luminous relational-grammar hypothesis

Symthaea #8919 and Symtropy #2870 ask whether a compact typed relational grammar can be recovered across multiple worlds and transfer to unseen compositions better than equation-by-equation fitting or conventional graph/state models.

This is an operational computational hypothesis. It is not a claim already established by Chain Theory.

### 2.3 Morphic resonance

Symthaea #9858 operationalizes one Sheldrake-style prediction: prior pattern learning by one population may affect learning by later, similar, independently isolated populations. That is a separate empirical hypothesis. It requires an explicit similarity function, dose/lag prediction, leakage controls, cluster-aware inference, and independent replication.

A positive result in #9858 would not validate Chain Theory. A Chain Theory construction-order theorem would not be evidence for morphic resonance.

## 3. Candidate model families

All candidates receive matched observations, interventions, data access, compute budgets, stopping rules, and held-out worlds.

| ID | Model family | Where history lives |
|---|---|---|
| M0 | Independent equation/symbolic-regression model per world | No shared history |
| M1 | Pooled multi-world predictor without explicit grammar | Learned parameters |
| M2 | Shared typed generative grammar | Reusable production rules |
| M3 | Shared grammar plus conventional local state-space / recurrent memory | Explicit local latent state |
| M4 | Source-bounded Chain Theory construction/read representation | Exact network construction and read history |
| M5 | Shared grammar plus similarity-weighted cross-system history kernel | Proposed cross-system historical term |
| M6 | Strong graph-neural, state-space or recurrent baseline with matched compute | Learned relational/state representation |

M5 is a speculative model family for comparison, not a Chain Theory consequence or accepted physical mechanism. If its physical mechanism and lawful information boundary cannot be specified, M5 remains a synthetic/statistical comparator only.

## 4. Formal model family for the historical-kernel ablation

Let each world have local state x(w,t), inferred or known grammar G, world-specific parameters theta(w), and conventional local latent state z(w,t).

Conventional local dynamics:

    x(w,t+1) = f_G(x(w,t), u(w,t); theta(w), z(w,t)) + noise(w,t)

A separate cross-system history term may be studied as:

    h(w,t) = sum over v != w and j < t of
             exp(-(t-t_j)/tau) * K_psi(G_v,G_w) * q(v,j) * phi(o(v,j))

    x(w,t+1) = f_G(x(w,t), u(w,t); theta(w), z(w,t), lambda*h(w,t))
               + noise(w,t)

Where:
- K_psi is a frozen, predeclared structural-similarity kernel;
- q is a declared prior exposure/dose;
- tau is a predeclared temporal-lag scale;
- phi maps a prior observation to typed features;
- lambda = 0 is the no-cross-system-history null.

This is a candidate statistical model, not a physical law. Its kernel, dose function, lag, and outcome must be selected on development data and frozen before confirmatory evaluation. The system may not select a new similarity metric after seeing results.

An observed lambda != 0 would first indicate that the declared conventional model was incomplete. It would not establish nonlocal influence until common causes, hidden local state, information leakage, apparatus memory, selection effects, and ordinary channels are independently ruled out.

## 5. Factorial synthetic benchmark

Generate synthetic worlds with known ground-truth grammars and dynamics. Cross these factors:

- shared vs heterogeneous grammars;
- known vs unknown grammar family;
- seen vs held-out rule compositions;
- shallow vs deep composition;
- full vs partial observation;
- low vs high measurement noise;
- same vs different local histories;
- local memory present vs explicitly reset;
- structurally similar vs dissimilar prior worlds;
- surface-similar vs surface-dissimilar forms, varied independently;
- exposure dose and temporal lag;
- interventions available vs observational-only data;
- identifiable vs observationally equivalent candidate models.

Hold out entire grammar families, topologies, compositions, and task seeds. Random row splits are insufficient.

### Required negative controls

1. **Heterogeneous-world control:** no common grammar exists. A candidate must reject forced unification.
2. **Observational-equivalence control:** several models explain current data. Correct output is a set, a discriminating intervention, or unresolved—not an unjustified singleton.
3. **Surface-similarity control:** visually similar but structurally unrelated tasks.
4. **Structural-similarity control:** structurally similar tasks with different surface encodings.
5. **Future-history placebo:** future booster activity cannot predict earlier test outcomes.
6. **Label/manifest permutation:** incorrect task, target, or history linkage is detected.
7. **Ordinary local-memory control:** standard state-space/recurrent memory explains history effects when relevant local state is available.
8. **Common-cause control:** shared environmental drift, device bias, batch/seed effects, or hidden context must not be misclassified as cross-system memory.
9. **Data-leakage control:** target labels, timestamps, identifiers, file metadata, network requests and operator knowledge cannot reveal held-out truth.
10. **No-history null:** prior similar-system exposure has no effect by construction.
11. **Positive planted-kernel control:** a known synthetic kernel is recovered at the expected direction/magnitude within declared tolerance. This tests analysis plumbing only.
12. **Transformation control:** node/edge relabeling, basis/coordinate changes, unit changes, and valid observation-order permutations preserve quantities they are supposed to preserve.

## 6. Path-equivalence test for Chain Theory's order coordinate

This is the highest-priority formal discriminator to run against exact source semantics.

For all pairs of valid construction sequences pi and pi-prime that are permitted by the theory and reach the same final network/configuration:

1. enumerate both complete paths;
2. bind the exact final state/network identity;
3. calculate source-defined construction/read order values and boundary counts;
4. apply source-declared basis/gauge/label transformations;
5. compare proposed physical observables and predictions;
6. preserve every path and discrepancy.

Questions to resolve:

- Do valid reorderings of independent construction operations reach the same final structure?
- If the final structure is the same but construction count/history differs, is that difference a physical observable or provenance?
- Can two distinct final structures have the same order count?
- Does the theory supply a mapping from its order coordinate to an operational physical observable before data comparison?
- Are physical predictions invariant under allowed basis changes and relabelings?

If formal move semantics admit only one order, document why; do not invent alternative paths. If a mapping is unavailable, report UNMAPPED rather than choosing one after seeing data.

### Required controls

- An acyclic construction order paired with a reversible physical process.
- A recurrent/periodic observable with monotone bookkeeping.
- Multiple valid topological orders of independent operations, if allowed.
- Same count with different graph structure.
- Same final graph with different construction histories, if allowed.
- Reverse moves only where the exact formalism permits them.

This tests whether the order coordinate is well-defined and invariant. It does not by itself test a discrete time crystal.

## 7. Temporal-order and time-crystal comparator

Reuse #8964. In a periodically driven many-body system, an observable can exhibit robust subharmonic order (e.g. period 2T) while the microscopic construction/provenance count continues to increase.

Report separately:
- subharmonic order parameter and autocorrelation;
- period-doubled spectral peak and lifetime;
- robustness to detuning, interactions, and noise;
- system-size/observation-window scaling;
- drive work, heat, and independently defined entropy production;
- construction-order mapping, if source-derived.

A monotone register can coexist with a periodic coarse-grained observable, but the observable map must be fixed in advance. Do not identify the register with physical time or entropy by naming alone.

## 8. Baseline and state-of-the-art comparison

Reproduce or adapt compatible task subsets from:

- **NewtonBench:** 324 interactive law-discovery tasks across 12 physics domains, including altered-law/metaphysical-shift environments. https://arxiv.org/abs/2510.07172
- **Science-Gym:** published 7 January 2026; requires autonomous data collection, experimental design and equation discovery across seven simulation domains. https://doi.org/10.1007/s10994-025-06914-x
- **LLM-SRBench:** 239 problems in chemistry, biology, physics and materials; its ICML 2025 paper reports 31.5% symbolic accuracy for the best tested system under that benchmark's protocol. https://proceedings.mlr.press/v267/shojaee25a.html
- **ERBench:** public/secret evaluation split and symbolic-regression leaderboard. https://equation-discovery.ti2.fmi.uni-jena.de/
- AI Feynman / PySR / conventional symbolic regression and system-identification methods where exact parity adapters are feasible.
- Conventional graph, recurrent, state-space and neural system-identification baselines.

Published headline scores must not be compared as if task protocols match. Reproduce compatible subsets locally with exact model versions, data, split, tools, compute, intervention budget, and scorer.

## 9. Metrics and acceptance gates

Report each dimension independently:

- held-out prediction log score and trajectory error in physical units;
- production-rule/equivalence-class recovery;
- held-out composition and topology transfer;
- grammar description length / MDL vs predictive quality;
- intervention/query count to resolve candidate models;
- correct abstention and calibration under non-identifiability;
- false-shared-grammar rate on heterogeneous worlds;
- false-history-effect rate under no-history and common-cause controls;
- recovery error and uncertainty for planted synthetic history kernels;
- invariance under permitted relabeling, basis and observation-order transformations;
- exact count of valid/invalid/unsupported/numerically failed cases;
- latency, memory, energy, model size and total data/compute budget.

A useful advancement requires:
1. exact known-answer evaluator controls;
2. matched budget comparisons;
3. paired uncertainty intervals on whole held-out worlds/families;
4. no increase in confident wrong grammar or false history claims;
5. successful heterogeneous and observational-equivalence controls;
6. reproducible source/configuration/data hashes;
7. independent reproduction before claiming a general advantage.

An aspirational development target is at least +5 percentage points on the frozen held-out composition metric or at least 20% fewer interventions at matched error and calibration. These are goals only; set the final smallest effect of interest after pilot baselines and before confirmatory evaluation.

## 10. Evidence and claim separation

Maintain independent evidence states for:
- formal Chain Theory result;
- source-to-code reproduction;
- classical graph-grammar representation performance;
- physical time/thermodynamics correspondence;
- time-crystal model comparison;
- conventional local memory;
- cross-system historical-effect experiment;
- concealed-target / scopaesthesia experiments.

No positive result may be promoted across these boundaries without a separate derivation or experiment.

## 11. Execution sequence without waiting for existing runners

1. Source audit and path-equivalence specification using only exact Chain Theory move semantics.
2. Synthetic generator/evaluator design using existing SIM-REL and #8919 contracts.
3. Baseline reproduction for a tiny known-answer domain.
4. Historical-kernel and common-cause synthetic controls.
5. Freeze a confirmatory design and sealed whole-world splits.
6. Only then implement an isolated grammar learner or begin human/physical experimentation under the relevant research and ethics gate.

No new general simulator, PhysicsIR, CI workflow, or competing evidence ledger is required.

## 12. Claim ceiling

This addendum proposes a model-comparison and falsification programme. It does not show that Chain Theory is true or false; does not establish that reality literally has a grammar; does not establish a physical historical-memory field; and does not infer morphic resonance, ESP, scopaesthesia, or nonlocal signalling from entanglement, temporal order, or a benchmark result.
