# SCI-SYMGRAM-001 — Proof-Qualified Symmetry Quotients for Shared Physical-Law Discovery

**Date:** 2026-11-18  
**Status:** research protocol draft; no benchmark execution or performance result claimed  
**Owner:** Symthaea issue [#12781](https://github.com/Luminous-Dynamics/symthaea/issues/12781)  
**Branch:** `research/symmetry-quotient-shared-grammar-v0-2026-11-18`

## 1. Executive question

Can a scientific discovery system use **independently qualified symmetry** to construct a smaller, semantics-preserving search space, then learn a reusable grammar of physical concepts and laws across multiple systems?

The proposed pipeline is:

```
raw multi-world observations
        |
        v
discover candidate concepts and group actions
        |
        v
qualify the mathematical claim at its correct scope
        |
        v
compile only admissible actions into a quotient / equivariant representation
        |
        v
induce a shared grammar across training worlds
        |
        v
discover laws for held-out compositions and interventions
        |
        v
lift predictions back to physical coordinates
        |
        v
independent mathematical + numerical + physical qualification
```

This is a controlled test of a specific paradigm hypothesis, not a claim that the universe is literally a grammar or that all physical laws arise from symmetry.

**Why this matters:** AI-Newton demonstrates concept-driven law generalization across 46 mechanics experiments, and explicitly identifies representing symmetry as an explicit concept as future work. Unsatz shifts evaluation from coefficient-fitting with a supplied model form to constructing an ansatz, including hidden variables, interactions, and measurement channels. Our opportunity is to test whether mathematically qualified symmetry can become a safe representation primitive that improves cross-world concept and law discovery.

References:
- Fang et al., *AI-Newton: A concept-driven physical law discovery system without prior physical knowledge*, Physical Review Research 8, 043129 (published 2026-11-04): https://doi.org/10.1103/dvvc-4rsr
- Ji et al., *Unsatz: Evaluating Scientific Discovery Without a Given Ansatz* (2026-10-31): https://arxiv.org/abs/2611.03713
- Imai Aldeia et al., *Call for Action: towards the next generation of symbolic regression benchmark* (SRBench, 2025): https://arxiv.org/abs/2505.03977

## 2. What is new here — and what is not

This study deliberately composes existing capabilities rather than creating parallel infrastructure.

| Existing owner | Reused responsibility |
|---|---|
| Symthaea #11336 / PR #12711 | Symmetry evidence classes, exact fixtures, calibration protocol, proof-carrying quotient hypothesis |
| Symthaea #11334 / #11416 | Typed symmetry reports, action/metric semantics, representation covariance |
| Symthaea #8919 | Shared generative grammar across physical worlds |
| Symthaea #8705 | Invariant mechanism transport across environments and interventions |
| Symthaea #6862 | Cold discovery, held-out trajectories, symbolic verification and novelty boundary |
| Symthaea #6879 / #6868 | Canonical ScientificModel / PhysicsIR and physical typing |
| Symtropy #3749 / #460 | Analytic oracles, solver-level V&V, step-map and physical qualification |
| Symtropy #6027 / #2886 | Candidate-independent proposals, counterexample evaluation and bounded grammar induction |

The unowned increment is a **factorial interaction study** of symmetry treatment × discovery organization, with a single protocol for task pairing, costs, held-out evaluation and result interpretation.

This is not:
- a new symmetry learner;
- a second generative-grammar learner;
- an alternative PhysicsIR or model AST;
- a new world generator, generic experiment selector or evidence ledger;
- an AI-Newton or Unsatz reproduction;
- evidence of a new physical law.

## 3. Mathematical admissibility: a quotient is a theorem obligation

Let a group (G) act on a state space (X). A quotient representation identifies states on the same orbit:

[
[x] = {g cdot x mid g in G}, qquad X/G.
]

Replacing (x) by ([x]) is safe only when the prediction target and task semantics retain the information needed after quotienting.

For a scalar observable (h), invariance requires:

[
h(g cdot x) = h(x)
]

for all admitted (g) and all (x) in the declared domain.

For a dynamical system (dot{x}=F(x)), an action (ho(g)) is equivariant only when:

[
F(ho(g)x) = Dho(g)_x F(x)
]

under the declared smoothness and domain assumptions. For linear actions, this reduces to the corresponding matrix relation.

These are different claims. In particular:

1. Scalar-expression invariance is not vector-field equivariance.
2. Vector-field equivariance is not automatically invariance of initial/boundary conditions.
3. Continuous-time equivariance does not automatically imply the chosen discrete integrator is equivariant.
4. A simulator symmetry is not automatically a symmetry of the physical system.
5. A finite-sample low residual is not an exact identity.
6. A symmetry of the equations may not preserve the observation, intervention, boundary or cost model.

A claim-bearing quotient compiler must therefore consume the strongest *applicable* existing evidence type and the matching physical-type judgment. It must preserve:
- the action and coordinate/chart identity;
- the metric and units where relevant;
- the declared domain and admissible transformations;
- the projection into quotient/equivariant coordinates;
- a lift/reconstruction map where predictions or controls need original coordinates;
- the exact proof or sampled-evidence receipt and its scope;
- any assumptions about initial conditions, boundaries, sensors, interventions and numerical steps.

If any required condition is absent, unsupported, ill-conditioned, or merely sampled, the original search space remains available. Sampled symmetry may rank proposals; it must never silently become a hard pruning rule.

## 4. Falsifiable hypotheses

**H1 — safe search reduction.** On eligible exact-symmetry systems, a proof-qualified quotient reduces candidate search work while preserving semantic law recovery and held-out predictions.

**H2 — compositional transfer.** Explicit shared-grammar discovery transfers better to unseen world compositions than independent per-world discovery and pooled discovery without reusable grammar rules.

**H3 — positive interaction.** Applying a qualified quotient before shared-grammar induction has a measurable positive interaction beyond the isolated gains of quotienting and shared-grammar learning.

**H4 — epistemic discipline.** The system rejects, qualifies, or abstains on broken/near symmetries, heterogeneous mechanisms, non-identifiable worlds and out-of-grammar alternatives. It does not manufacture a common grammar when the worlds do not share one.

All hypotheses may fail. A reduction in computation accompanied by more false laws is a negative result, not an efficiency win.

## 5. Experimental design

### 5.1 Primary factorial arms

The primary experiment is a (2 	imes 3) factorial:

| Symmetry treatment | Independent per-world discovery | Pooled fit, no reusable grammar | Explicit shared-grammar induction |
|---|---|---|---|
| No quotient | A | B | C |
| Exact, proof-qualified quotient | D | E | F |

The principal interaction contrast is:

[
Delta_{mathrm{interaction}} = (F-C) - (E-B)
]

with the sign and primary outcome direction preregistered separately for each metric. The study also reports D−A, E−B and F−C. This is not a single aggregate leaderboard score.

### 5.2 Diagnostic arms (not included in the factorial estimate)

- **Soft sampled-symmetry hint:** sampled candidates may affect proposal ordering only; no candidate may be removed from the unquotiented search.
- **False-hard-quotient control:** deliberately mark a near-symmetry as exact. The evaluator should expose the invalid reduction, false law or loss of held-out prediction.
- **Tiny exact oracle:** exhaustive group/grammar search on tractable instances; diagnostic upper bound only.
- **No-transfer control:** start a fresh discovery run for each held-out world.
- **Oracle-structure fit ceiling:** supply the true action/grammar to isolate parameter-fitting and implementation failures from discovery failures.

### 5.3 Matched budgets

Every comparable arm receives identical:
- world identity and paired seed;
- initial observations and permitted interventions;
- discovery/calibration/confirmation split;
- candidate-evaluation and experiment budgets;
- model, prompt, tools and memory access;
- wall-time, CPU/GPU, memory and energy ceilings where metered;
- stopping, abstention and safe-abort semantics.

Record both total cost and the cost of each stage. Do not compare a planner against a full research agent or a model-free arm against a model-informed arm and call the difference algorithmic superiority.

## 6. World and task families

The first confirmatory corpus should contain at least six analytically checkable families. Candidate starting families:

1. Coupled oscillators and normal-mode transformations.
2. Central-force motion and rotational symmetries.
3. Spring–mass networks with graph automorphisms and parameter heterogeneity.
4. Rigid-body rotations and frame transformations.
5. Constrained mechanical systems with explicit boundary/constraint conditions.
6. A non-mechanics family (for example, a simple reaction–diffusion or transport system) only after its variables, units and action semantics have independent physical types.

Each family should include positive, negative and ambiguity controls:

| Challenge | Hidden truth | Correct disposition |
|---|---|---|
| Exact symmetry | Declared action truly preserves the target | Admit at exact/formal scope after proof |
| Broken symmetry | A term or boundary condition breaks the action | Reject the exact quotient |
| Near-symmetry | Small perturbation produces a graded residual | Sampled/approximate evidence only; no hard pruning |
| Representation change | Same geometry under a valid coordinate transform | Preserve classification and transformed predictions |
| Heterogeneous worlds | No sufficient shared grammar exists | Reject common grammar or return scoped partial grammar |
| Non-identifiability | Multiple mechanisms remain observationally equivalent | Return equivalence class, abstain or request a distinguishing intervention |
| Out-of-grammar mechanism | Truth is outside the candidate grammar | Report misspecification; do not force the nearest in-grammar law |
| Measurement confounding | Sensor transform mimics a physical change | Distinguish measurement-model error from law change |

Split by generating grammar, world configuration, composition family and transformation family. Randomly splitting rows from the same trajectory or program is not sufficient.

Use separate discovery, calibration and sealed confirmation partitions. Target at least 30 paired seeds per family in the confirmatory phase; increase the sample size when the preregistered precision/power analysis requires it. The seed is not the only independent unit: report task/world clustering and avoid treating many observations from one world as independent scientific replications.

## 7. Baselines and industry comparison

Use only compatible baselines on matched task families:

| Baseline | Proper role | Do not claim |
|---|---|---|
| Rational CAS / exact group oracle | Exact checks on bounded algebraic fixtures | General physical discovery |
| Existing Symthaea unquotiented discovery | Internal ablation/reference | External SOTA |
| PySR / Operon / SRBench-compatible methods | Symbolic expression recovery | Full causal or experimental autonomy |
| PySINDy / SINDy-PI / weak-form SINDy | ODE/system identification where assumptions hold | Direct performance on arbitrary hidden-structure tasks |
| Shared-structure regression | Cross-world shared terms with world-specific parameters | Explicit symbolic grammar recovery unless that is what it implements |
| AI-Newton | Concept-driven multi-experiment law discovery | Direct score comparison unless the task, data, budgets and scoring match |
| CoSR-style progressive symbolic discovery | Reusable intermediate concepts, if implementation and interface are reproducible | Shared-grammar superiority from citation alone |
| LieFlow / SymFP / other symmetry methods | Candidate action or subgroup proposal on native symmetry tasks | Physical-law discovery without downstream qualification |
| Exact tiny-world grammar oracle | Diagnostic ceiling for finite worlds | Deployable performance baseline |

AI-Newton is the closest concept-discovery comparator. Its published controlled mechanics study uses 46 predefined experiments and simulated data generators; with up to 1,200 trials and an average 48-hour run, the authors report roughly 90 concepts and 50 general laws on average across ten test cases. The authors state that symmetry is implicit in the discovered laws and identify explicit symmetry concepts as future work. This gives us a concrete baseline for concept inventory, cross-experiment generalization, trials and wall time—but not a matched performance result for our factorial. https://doi.org/10.1103/dvvc-4rsr

CoSR (Chain of Symbolic Regression) is a second strong comparator for progressive knowledge composition. Its November 2, 2026 Physical Review Research paper describes chaining semantically meaningful knowledge units, reproducing the Kepler-to-gravitation discovery path and applying the method to turbulent Rayleigh–Bénard convection, viscous circular-pipe flow, laser–metal interaction and aerodynamic coefficient scaling. Our shared-grammar arm must distinguish genuinely reusable semantic rules from simply chaining fitted intermediate expressions. https://doi.org/10.1103/f2kp-pp4q

Unsatz is the key ansatz-free structural-discovery comparator: its authors report 21–60% solve rates across 80 systems / 20 domains for nine models. They also report that classical equation-discovery methods rarely solve 44 systems even where parameter fitting succeeds once the correct structure is supplied. Neither its published results nor AI-Newton's are matched results for this experiment. https://arxiv.org/abs/2611.03713

Scientist World is a broader paper-derived autonomous-research benchmark (1,000 executable environments claimed by its authors). Its reported results concern paper-derived research tasks, not this controlled physics/grammar task, so it belongs on a separate full-cycle research scorecard rather than this leaderboard.

**Whole-system scientific autonomy is another separate scorecard.** Autonomous Science Bench 0.1 evaluates six pilot lab campaigns and requires both discovery quality among selected/measured designs and learning quality on designs never measured. Its public leaderboard reports GPT-6 Astra + Codex at 72.2% and Opus 5.5 + Claude Code at 66.7%; the maintainers note only three trials per task, so small gaps are not meaningful. It also reports all-in campaign cost and agent time. These are not directly comparable to the SCI-SYMGRAM mechanics factorial and must never be pooled with its law-recovery metrics. Reuse Symthaea #10921 for the pinned replay/twin adapter and #12461 for the six-campaign scorecard expansion rather than creating another ASB adapter. References: https://asb.open-ssi.org/ and https://arxiv.org/abs/2610.18449.

SRBench 2025 is the symbolic-expression comparison floor: 25 methods in a Docker-based setup, hyperparameter tuning, and 30 independent runs per dataset across 24 PMLB datasets plus first-principles problems. Use that reproducibility discipline for the PySR/Operon-compatible lane; do not reinterpret it as end-to-end scientific-agent performance. Reference: https://cavalab.org/srbench/.

External baseline status must be explicit:
- **Reproduced:** same source/task, configuration, budget and scorer independently executed.
- **Matched adapter:** a common interface and paired task subset, with all divergences recorded.
- **Related work only:** implementation, licence, task semantics or resources prevent a defensible comparison.

Never fill missing baseline results with published scores or estimates. Pin the upstream commit, model/config, operator set, tuning budget, hardware, task manifest, scorer and all exclusions for every executed baseline.

## 8. Metrics and statistical analysis

Do not collapse the metrics below into one number.

### 8.1 Symmetry correctness

- Group/action or observable-equivalence-class recovery.
- False exact-admission rate on broken and near-symmetry controls.
- Sampled-lane precision, recall, risk–coverage and abstention.
- Representation-covariance consistency for paired coordinate transformations.
- Correct separation of scalar invariance, vector-field equivariance, initial/boundary invariance and step-map symmetry.

### 8.2 Shared concept and law discovery

- Production-rule recovery by semantic equivalence, not token overlap.
- Variable, latent-state and interaction recovery.
- World-level law equivalence class recovery.
- Held-out trajectory prediction, intervention-effect prediction and longer-horizon prediction.
- Held-out composition-depth transfer.
- False-shared-grammar and false confident-singleton rate.
- Correct unresolved / set-valued / no-compatible-model outcomes.

### 8.3 Search efficiency and resources

- Candidate ASTs, grammar expansions and candidate-model evaluations.
- Experiments/interventions and acquisition cost.
- Wall time, CPU/GPU time, peak memory and energy (only where actually metered).
- Cost to first independently confirmed transferable law.
- Search-cost distributions and worst-family costs, not only means.

### 8.4 Statistics

- Paired treatment contrasts by family and world.
- 95% paired confidence intervals with clustering at the world/task level.
- Preregistered primary outcomes and multiple-comparison correction for confirmatory secondary outcomes.
- Calibration and false-claim rates on separate negative-control sets.
- Complete denominators: invalid, failed, unsupported, timed-out, abstained and underpowered runs remain visible.

Publish a Pareto frontier for semantic correctness, held-out transfer, false-confidence risk and resource cost.

## 9. Proposed acceptance targets (not achieved results)

1. **Exact mathematical fixtures:** zero false exact admissions and 100% expected classification within the explicitly supported fragment.
2. **Approximate fixtures:** thresholds fixed on calibration data; sampled evidence never promoted to exact/formal status.
3. **Quotient auditability:** every reduction edge binds its source representation, action, assumptions, proof/evidence scope, projection and any lift.
4. **Correctness non-inferiority:** the exact-quotient arms must remain within a preregistered, practically meaningful non-inferiority margin for semantic recovery and held-out prediction.
5. **Efficiency hypothesis:** target at least 20% lower median candidate-evaluation cost on eligible exact-symmetry families, with a paired interval excluding zero and no correctness-gate regression. If not met, report the measured Pareto result; do not weaken the gate.
6. **Transfer:** shared-grammar induction must improve a preregistered held-out composition outcome versus the strongest compatible independent/pooled baseline, without increasing false-shared or false-singleton outcomes.
7. **Negative controls:** no false exact quotient admissions; heterogeneous and non-identifiable worlds must reject, return a set, propose an informative action or remain unresolved.
8. **Replication:** an independently run confirmation on a later frozen corpus or separate implementation before broad generalization claims.

These are research targets; none is currently measured.

## 10. Implementation plan that does not wait for broad CI

### Slice 0 — owner and artifact audit

- Fetch the exact current contents of #11336 / PR #12711 and #8919.
- Map existing fixture IDs, schemas, evidence types, generators, splits, and claim scopes.
- Audit #8671 progressive discovery and #8487 recovery protocol so shared-grammar metrics are not duplicated.
- Identify the existing quotient compiler boundary, if any; do not infer that the benchmark protocol PR already implements a production quotient compiler.

**Exit:** a table of reusable artifacts and missing cross-track pieces.

### Slice 1 — factorial contract only

- Add a machine-readable factorial manifest that references existing owner IDs and artifact digests.
- Define arm identity, matching fields, interaction contrast, denominators and result fields.
- Add contract checks that reject a mixed source SHA, grammar, action profile, task split, budget or physical claim scope.
- Keep source-visible development fixtures clearly marked non-blind.

**Exit:** deterministic contract validation; no physical or SOTA claim.

### Slice 2 — tiny exact pilot

- Use one exact symmetry-positive family and one broken/near-symmetry family.
- Use a tiny shared grammar with exhaustive ground truth.
- Compare all six primary arms and diagnostic controls on paired seeds.
- Verify quotient/lift round trips and predictions against the original unquotiented world.
- Preserve a counterexample when a quotient or grammar-transfer assumption fails.

**Exit:** a reproducible pilot with complete per-arm outputs and a failure analysis.

### Slice 3 — cross-family development campaign

- Extend to six or more families.
- Add coordinate changes, interventions, held-out compositions and sensor confounding.
- Add compatible external methods and exact resource metering.
- Use development failures to refine the protocol; never tune on confirmation.

**Exit:** a frozen, independently checked candidate for confirmation.

### Slice 4 — sealed confirmation and physical bridge

- Freeze task families, splits, code, baseline versions and score rules.
- Run paired seeds under identical budgets.
- Have Symtropy independently check analytic solutions, numerical convergence, invariants and step-map behavior where applicable.
- Preserve physical measurements and calibration uncertainty separately from simulator evidence.

**Exit:** a scoped comparative result, whether positive, negative or inconclusive.

## 11. Evidence and claim boundaries

The benchmark receipt should reuse the existing task/result/evidence infrastructure and bind at least:
- source commits and file/artifact digests;
- symmetry report and evidence class;
- canonical model/grammar and version;
- world/seed/split manifest;
- arm and resource budget identity;
- baseline model/tool/config and scorer identity;
- raw outputs, failure dispositions and analysis revision.

A digest establishes byte identity, not authenticity or physical truth. A proof establishes a formal consequence of stated assumptions, not the truth of the physical model. A simulation establishes behavior of that implementation, not the external world. An external measurement only supports claims within its calibration and applicability scope.

## 12. Current capability gap: researcher adapter

The existing `src/scientific_method.rs` module should not be treated as the production researcher for this programme. Its current `predict()` returns the hypothesis posterior, and `test_hypothesis()` uses fixed Normal–Normal update assumptions; it does not own structured physical models or intervention selection.

Symthaea #11171 remains the owner of identifying and connecting an actual research-policy callable to the blind episode protocol. The finite reference policy in #11060 is not the learned researcher. This study can validate the representation/grammar interaction independently while the policy adapter work proceeds; it must not invent an autonomy claim.

## 13. Decision rules

- If exact quotienting lowers search cost and preserves held-out law recovery, extend to more group actions and world families.
- If it lowers training search cost but harms out-of-domain prediction, reject that quotient compiler or narrow its applicability predicate.
- If shared grammar helps only some worlds, represent applicability conditions and partial grammars rather than force one universal grammar.
- If approximate symmetry hints help proposal ordering but hard quotienting fails, retain the soft hint as a heuristic only.
- If no arm beats strong baselines under matched budgets, publish the negative result and investigate grammar support, identifiability, measurement models or experiment allocation.

**Success is not that the quotient always wins. Success is that Symthaea can determine when a representation reduction is justified, quantify what it buys, and preserve the evidence when it is not.**
