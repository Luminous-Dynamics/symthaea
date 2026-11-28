# MORPHOSTASIS: competitive research crosswalk and benchmark targets

**Status:** research specification / design only  
**As-of date:** 2026-11-28  
**Base audited:** `main` at `625e3947be602c5e866b12cba767f90401d2e454`  
**Primary repository:** [Luminous-Dynamics/symthaea](https://github.com/Luminous-Dynamics/symthaea)  
**Program name:** MORPHOSTASIS — Morphological Homeostasis, Identity, Adaptation and Distributed Intelligence

> This document specifies experiments and promotion gates. It reports no new Rust implementation, model execution, benchmark score, wet-lab result, or state-of-the-art win.

## 1. Executive thesis

The opportunity is to move beyond "make a damaged shape look like its target" toward **maintain a system's intended identity and critical capabilities while its components, topology, observations and environment change**.

That reframes morphogenesis-inspired computation as a general control and inference problem:

- The *current state* is an estimate of what exists now.
- The *target identity* is a persistent, provenance-bearing hypothesis about what the system is meant to remain or become.
- The *competence set* describes what the system must still be able to do.
- The *repair policy* selects changes that improve the target while preserving constraints and respecting uncertainty.
- The *morphodynamic memory* is distributed across local state, recurrent dynamics and longer-lived structural/relational state; it is not assumed to be one global label or one target bitmap.
- The *interaction topology* is part of the adaptive system, not necessarily a fixed wiring diagram.

The key research question is therefore:

> At matched information, compute, memory and communication budgets, can a distributed controller preserve intended identity and functional constraints, recover from damage, and adapt its representation and interaction topology more robustly than strong local NCA, graph, and centralized baselines?

Michael Levin's work is a major scientific inspiration for testable questions about pattern regulation and competency across scales. It is not a licence to treat software state as membrane voltage, graph edges as gap junctions, or synthetic recovery as biological regeneration.

## 2. Paradigm shifts worth testing

### 2.1 Target morphology as an attractor region, not a pixel-perfect template

A single binary target mask is too brittle for many systems. Multiple geometries can instantiate the same design intent; some geometric deviations are harmless, while others destroy function. Represent a target as a **declared admissible set** or equivalence class with explicit invariants:

- identity constraints (what object/system is this?);
- topology and connectivity constraints;
- functional constraints (what must it still do?);
- safety/structural constraints;
- permitted variation and repair cost;
- uncertainty over the target when history is incomplete.

Evaluate the distance to the admissible set, not merely pixel equality. Keep exact-shape IoU/Chamfer metrics as subordinate geometric measures. Do not infer the target from labels hidden in the evaluator.

**Falsifier:** the admissible-set representation adds no held-out robustness or functional preservation over a matched template controller, or it creates a larger false-acceptance rate.

### 2.2 Separate present morphology, remembered intent and repair action

The Smart Cellular Bricks paper makes a useful distinction: current-shape classification can change after damage, and the chair-to-table ambiguity exposes why current appearance does not necessarily recover original target identity. The existing #8401 fixture captures this identifiability problem.

The controller must emit separate outputs:

1. current-state estimate;
2. target-identity belief and provenance;
3. feasible repair options;
4. confidence / ambiguity / abstention;
5. executed repair and observed outcome.

**Falsifier:** target memory does not improve correct restoration under controlled, fair information access, or a calibrated no-memory system can perform equally well without false confidence.

### 2.3 Co-evolve the interaction graph, not only edge weights

A new 2026 mathematical morphogenesis framework combines a bistable electrical state, synthetic regulatory modules, adaptive gap-junction conductance and slow tissue memory; however, its graph is fixed between cell divisions, and the authors identify dynamic topology as a natural extension. This leaves a clear computational question: can a controller learn when to rewire, split, merge, create or retire communication paths while preserving task identity?

Test at least three levels separately:

- fixed topology / fixed weights;
- fixed topology / adaptive weights;
- adaptive topology plus adaptive weights.

Add degree- and message-budget-matched random rewiring controls. A changing graph is not automatically better; rewiring can destroy reliable local competence.

**Falsifier:** topology adaptation does not improve held-out recovery/generalization after matching message count, degree distribution, update count and memory, or it improves score only by leaking target/damage information.

### 2.4 Use measured timescale separation, not a generic global latent

The 2026 FLIM study reports non-neural bioelectric dynamics on seconds, minutes and hours scales. This motivates a *temporal architecture experiment*, not a direct software analogue of those measurements.

Test independently:
- fast local state updates;
- intermediate coordination / adaptive edge state;
- slow distributed identity or tissue-history state;
- optional low-bandwidth global summary;
- adaptive invocation/gating of each channel.

Match total node updates, communication bytes and wall-clock/energy measurement boundary. Include channel-shuffle, phase-scramble and equal-cost noise controls. The value of slow state should disappear on memory-irrelevant control tasks.

**Falsifier:** any gain is explained by extra updates, parameters, message volume or simple smoothing, or disappears when the temporal channels are independently ablated.

### 2.5 Make sensing and repair decisions value-of-information aware

Do not assume the controller should always act. When target identity is ambiguous or repair is irreversible, it should choose among:
- act;
- request or perform another observation;
- run a low-cost diagnostic;
- preserve current state;
- abstain / escalate.

Evaluate expected utility of information acquisition against actual downstream decision quality, observation cost and false-repair risk. This connects morphostasis to Symthaea's active-inference/EVSI work without assuming that a novelty score is information gain.

**Falsifier:** active sensing does not reduce wrong repairs or total resource cost under partial observability, compared with an equal-observation-budget baseline.

### 2.6 Test competence, not just shape

Morphological similarity can be high while a system is nonfunctional. Give each benchmark family task-native invariants: connectivity, load path, reachability, flow capacity, collision-free motion, transport continuity or another explicitly defined function. The evaluator must be independent and must not reveal hidden target labels or future damage.

**Falsifier:** functional constraints do not add explanatory or predictive value beyond geometry, or the chosen function oracle is itself leaky or invalid.

## 3. Literature and competitive landscape

The research plan is grounded in the following work. These sources motivate tests; none establishes a Symthaea result.

| Work | Contribution relevant to this programme | Consequence for our comparison |
|---|---|---|
| Hartl, Levin & Pio-Lopez (2026), [Neural cellular automata: Applications to biology and beyond classical AI](https://doi.org/10.1016/j.plrev.2025.11.010) | Reviews NCA as trainable local update systems with applications to biology, robotics and AI; highlights interpretability, scaling and biological-data integration as open challenges. | Local NCA must be a first-class baseline; test beyond it on scaling, interpretability, robust adaptation and cost. |
| Pio-Lopez, Hartl & Levin (2026), [BraiNCA preprint](https://arxiv.org/abs/2604.01932) | Reports attention and explicit long-range connections as useful additions to vanilla local NCA on selected tasks. | Compare local NCA, BraiNCA-style long-range attention and multi-rate adaptive routing. Only call it a reproduction if its architecture, artifacts and evaluation are matched. |
| Moreno et al. (2026), [Smart cellular bricks](https://doi.org/10.1038/s41467-026-75166-7) | Distributed 3D modules classify shape and detect damage; paper reports 98.97% simulated NCA shape-classification accuracy and a 100% distributed voxel graph Weisfeiler–Lehman result. Physical modular hardware is demonstrated, while active physical repair remains distinct from simulated regrowth. | Reproduce per-claim, not by chasing one headline accuracy. Include graph-WL and physical communication constraints. Existing issue #8423 tracks this. |
| Cortés-Poza (2026), [A hybrid mathematical framework for morphogenesis and regeneration](https://doi.org/10.1007/s00285-026-02459-2) | Combines bistability, adaptive coupling weights, a synthetic GRN, slow tissue memory and attractor-based target morphology; develops reduced-system analysis and numerical studies. | Include a clean-room mathematical comparator or explicitly labelled conceptual baseline. Extend tests to dynamic topology, richer target identity and independent functional evaluation; do not treat the paper's synthetic model as a validated biological simulator. |
| McMillen & Levin (2026), [Lifetime imaging of long-distance non-neural bioelectric patterns](https://doi.org/10.1016/j.ydbio.2026.08.001) | FLIM observations over 17+ hours report distinct seconds-, minutes- and hours-scale components in migrating neural crest explants; minutes-scale voltage dynamics were largely distinct from calcium dynamics. | Justifies multiscale temporal hypotheses; does not identify a software latent-variable implementation. Measurement modality and uncertainty must be preserved. |
| DynamicAtlas (2025/2026), [Drosophila morphodynamic atlas](https://doi.org/10.1038/s41592-025-02897-8) | Live and static datasets from 500 embryos, including 18 mutants, aligned to a common developmental timeline; describes morphodynamic modules. | Candidate observational trajectory dataset for a later external-validation lane, conditional on license, access and representation compatibility. Observational fit is not causal mechanism validation. |
| Levin & Resnik (2026), [Mind Everywhere, Part II](https://doi.org/10.1007/s13752-025-00524-5) | Calls for operational, testable system-specific protocols when discussing goal-directedness and cognition across scales. | Keep competence and agency claims task-bounded and operational; do not infer consciousness. |

### 3.1 What current evidence does and does not establish

- A strong local NCA is not a weak baseline; the 2026 Smart Cellular Bricks work includes a non-neural graph baseline that reportedly reaches 100% on its simulated classification task. Beating an NCA alone is insufficient.
- The 2026 hybrid mathematical framework is a meaningful conceptual comparator, not a broad empirical SOTA score. Its own discussion notes fixed graph topology between divisions and suggests topology dynamics as future work.
- The FLIM findings are empirical observations in a specific biological preparation. Their timescales are a reason to ask better questions, not target numbers for a software simulator.
- DynamicAtlas offers a promising external data lane, but observational trajectory prediction cannot establish causal bioelectric mechanisms.
- Physical decentralized hardware is an important separate comparison lane. Simulation-only success must never be described as physical self-repair.

### 3.2 A separate fungal-network lane: demand-responsive morphology and transport

Symthaea already has a more data-grounded mycelial programme that should be treated as a **complementary substrate-specific research lane**, not duplicated by this crosswalk and not conflated with Levin's animal somatic-cell bioelectric research.

Van Iersel et al. (2026), [*The mycelial network organization adapts to facilitate new substrate colonization and fruiting in Agaricus bisporus*](https://doi.org/10.1016/j.funbio.2026.101799), report that new substrate colonization is associated with directional growth and cord formation, while cords in routes supplying developing fruiting bodies can widen. Their measures include optical grayscale as a colonization proxy and cord area/width metrics. Crucial constraints: the longitudinal fructification series has only three independent cultivation cycles with substantial between-cycle timing variation; cord abundance was not significantly correlated with total mushroom yield; the directional-extension assay is a different cohort with six biological replicates per treatment. These are useful dynamics/observational data, not a large leaderboard or evidence that more cords always means higher yield. <Cite refs={["turn514855search0","turn515208search0"]}/>

The current repository owners already split the problem correctly:

- [#13450](https://github.com/Luminous-Dynamics/symthaea/issues/13450) asks whether demand/source/hydration context predicts future cordogenesis, cord width and network restructuring; it explicitly retains the null cord-abundance/yield result.
- [#13423](https://github.com/Luminous-Dynamics/symthaea/issues/13423) separates structural connectivity, intrahyphal forward/reverse cytoplasmic streams, extracellular hydraulic flow, isotope-measured source-to-sink transfer and microbial dispersal. Those measurements must not be collapsed into one graph edge weight or one generic flow variable.
- [#13460](https://github.com/Luminous-Dynamics/symthaea/issues/13460) owns biological-replicate-disjoint and forward-time splits; for the three-cycle cohort, leave-one-cycle-out estimates must show their wide uncertainty and cannot support broad SOTA claims.
- [#13425](https://github.com/Luminous-Dynamics/symthaea/issues/13425) owns data rights, provenance and file-level registry; do not mix the cordogenesis and isotope datasets without verified paired sample IDs.
- [#13144](https://github.com/Luminous-Dynamics/symthaea/issues/13144) already has a structural graph failure-resilience evaluator and independent BFS oracle. Its current structural/geometric outcomes are not measured transport or active self-repair.

The useful connection to MORPHOSTASIS is a shared question at the level of **demand-responsive structural adaptation**: can changing topology/geometry improve a measured function under constrained resources? But the biological observation models, mechanism and claim ceiling remain separate. A fungal hyphal cord is not an animal tissue gap junction; the dataset does not test somatic-cell bioelectric pattern regulation. Conversely, a synthetic graph controller score does not explain fungal cordogenesis.

### 3.3 Endogenous graph adaptation as a missing algorithmic benchmark

The new [#14123 TOPO-ADAPT-001](https://github.com/Luminous-Dynamics/symthaea/issues/14123) isolates an algorithmic gap not owned by the biological transport programmes: **a controller that is allowed to change its own interaction graph**. #13144 removes edges/nodes to measure structural vulnerability; #13423/#13450 study measured biological network remodeling; #8372/#8388 study remapping and communication schedules. #14123 instead compares no edits, weight-only adaptation, local greedy repair, distributed fixed-topology routing, adaptive rewiring and degree-matched random rewiring under the same service-demand shifts and costs.

The first deterministic slice is deliberately small (12–20 nodes) and uses independent exact max-flow/min-cut references where tractable. Its candidate-sweep oracle evaluates no edit plus each single-edge candidate and selects maximum served flow, then minimum declared edit/maintenance cost; it is a full-visibility upper bound, not a distributed peer. The evaluator reports edge-edit and maintenance costs; message/update costs remain uncharged until a controller emits those traces. This can become the strongest computational test of whether topology co-evolution adds value beyond remapping, multi-rate communication or a fixed graph—without claiming it reproduces a living network.

### 3.4 Stronger 2026 topology-reconfiguration comparators

The new #14123 tournament should not stop at local greedy versus random rewiring.

**Graph neural network rewiring.** Attali et al., *Graph Rewiring in GNNs to Mitigate Over-Squashing and Over-Smoothing: A Survey* (IJCAI 2026), organizes current rewiring methods by structural edit type, local/global signal, computational overhead and hyperparameter sensitivity. Its surveyed approaches include Delaunay and dynamic triangulation, spectral-gap/FOSR and expander-style methods, curvature-based rewiring, spectrum-preserving sparsification, virtual nodes and learned/signal-guided edits. This work is about improving message propagation in GNNs, not capacity-feasible service restoration; it should supply structural communication baselines, not replace flow/service evaluation. [IJCAI 2026 proceedings](https://doi.org/10.24963/ijcai.2026/861). <Cite refs={["turn788955view1"]}/>

**Resilience-based network design.** An online-available early-access study by Zaman, Sharmin & Mahmud, *Resilience-based network design optimization under aleatory uncertainty* (DOI: 10.1016/j.cie.2026.112478), is listed for *Computers & Industrial Engineering* volume 223 (January 2027). It jointly selects topology and integer link capacities, models repeated disruption/recovery dynamics, and imposes chance-constrained recovered-flow requirements. It reports a MILP-based probabilistic solution-discovery algorithm (PSDA) against a genetic algorithm, with applications including 7-, 13-, 20- and 74-node networks. This is a closer functional comparator to #14123 than generic graph rewiring because it optimizes delivered flow and recovery under uncertainty. Its exact objective, uncertainty model, constraints, artifact and protocol still need reconciliation before claiming a faithful reproduction. [Paper record](https://www.sciencedirect.com/science/article/abs/pii/S0360835226006790). <Cite refs={["turn540123search7"]}/>

The revised SOTA ladder should therefore contain two distinct families:

- **Communication-topology baselines:** Delaunay/geometric augmentation when coordinates are legitimately known; spectral-gap/FOSR or expander-inspired rewiring; curvature/effective-resistance-ranked edits; spectrum-preserving sparsification; virtual-node/hub shortcuts; dynamic triangulation; degree-matched random rewiring.
- **Capacity/service-design baselines:** exact max-flow/min-cost and MILP design oracles on small cases; PSDA-style stochastic topology/capacity design and GA when paper semantics can be matched; fixed topology, weight-only, local greedy and task-directed adaptive editing.

A graph with a larger spectral gap or shorter diameter is not necessarily a better service network. The controller must optimize **task-specific, capacity-constrained, uncertainty-aware service and recovery**, while explicitly paying for edits, maintenance, communication, latency and downtime. Add repeated/correlated disruptions, recovery horizon, chance-constrained service and no-edit stationary controls. The differentiator to pursue is a Pareto improvement in recovered service, latency and resilience per unit reconfiguration/communication cost—not connectivity alone.

## 4. Existing Symthaea programme ownership — reuse, do not fork

| Owner | Existing question | MORPHOSTASIS contribution |
|---|---|---|
| [#8372 DIVERSE-INTEL-001](https://github.com/Luminous-Dynamics/symthaea/issues/8372) | Representation remapping and navigation under damage/topology shift | Parent benchmark, held-out remapping and target navigation |
| [#8388 MULTIRATE-COORD-001](https://github.com/Luminous-Dynamics/symthaea/issues/8388) | Fast-local / slow-nonlocal coordination | Temporal-channel and adaptive-routing factorial |
| [#14123 TOPO-ADAPT-001](https://github.com/Luminous-Dynamics/symthaea/issues/14123) | Endogenous topology edits under damage and demand shift | Dedicated synthetic graph-control tournament; distinguishes graph edits from edge-weight adaptation and communication scheduling |
| [PR #14125](https://github.com/Luminous-Dynamics/symthaea/pull/14125) | TOPO-ADAPT v0.1 frozen reference fixtures | Three 12-node cases with independently checked max-flow/min-cut values; declarative fixtures only, no controller implementation |
| [PR #14127](https://github.com/Luminous-Dynamics/symthaea/pull/14127) | TOPO-ADAPT Rust reference evaluator | Draft Rust max-flow/min-cut evaluator, input validation, edit/maintenance cost reporting, an exact one-step candidate-sweep upper-bound oracle and 11 authored tests; exact-head Rust execution not yet qualified |
| [#8401 MORPH-ID-001](https://github.com/Luminous-Dynamics/symthaea/issues/8401) | Current morphology vs target identity | Parent contract for identity, ambiguity and abstention |
| [PR #8405](https://github.com/Luminous-Dynamics/symthaea/pull/8405) | Dependency-free Rust morphology continuity oracle | Existing chair/backrest identifiability fixture, exhaustive four-cell deletion enumeration and deterministic receipts; exact-head Rust qualification must be checked |
| [PR #8468](https://github.com/Luminous-Dynamics/symthaea/pull/8468) | Active evidence selection under morphology ambiguity | Existing action-selection, decision-value/EVSI, noisy evidence, misspecification and robust calibration fixtures; treat provenance flags as scenario inputs, not cryptographic verification |
| [#8409 GNCA-DYNAMICS-001](https://github.com/Luminous-Dynamics/symthaea/issues/8409) | Structured microfluctuations vs matched controls | Mechanism-specific intervention arm |
| [#13144](https://github.com/Luminous-Dynamics/symthaea/issues/13144) | Mycelial graph structural failure resilience | Existing deletion/attack-family evaluator and independent BFS oracle; structural metrics only |
| [#13423](https://github.com/Luminous-Dynamics/symthaea/issues/13423) | Multiplex dynamic mycelial transport | Separate structural, cytoplasmic, hydraulic, tracer/source–sink and microbial-dispersal lanes |
| [#13450](https://github.com/Luminous-Dynamics/symthaea/issues/13450) | Demand-responsive mycelial morphogenesis | Observational cordogenesis/width and source-sink forecasts; does not assume cord abundance predicts yield |
| [#13460](https://github.com/Luminous-Dynamics/symthaea/issues/13460) | Replicate-disjoint forward-time splits | Guard against frame/timepoint/cultivation-cycle leakage |
| [#13425](https://github.com/Luminous-Dynamics/symthaea/issues/13425) | Rights-aware mycelial dataset registry | File-level licenses, raw artifact lineage and sample/replicate IDs |
| [#13066](https://github.com/Luminous-Dynamics/symthaea/issues/13066) | Evidence-driven Mycelial Presence morphogenesis and CognitiveTopology v0 | Separates semantic cognitive topology from its spatial/morphological realization; no renderer-created cognitive claims |
| [#13094](https://github.com/Luminous-Dynamics/symthaea/issues/13094) | Event-sourced CognitiveTopology reducer and deterministic topology deltas | Semantic event provenance, replay and graph-state consistency; not a biological tissue controller |
| [#13079](https://github.com/Luminous-Dynamics/symthaea/issues/13079) | Irregular bud-fate, space-colonization and Physarum-inspired visual morphogenesis | Deterministic geometry/fusion/network-structure comparators; explicitly not a fungal nutrient/biophysics simulator |
| [#8423 SCB-REPRO-001](https://github.com/Luminous-Dynamics/symthaea/issues/8423) | Independent reproduction of Smart Cellular Bricks | Strong published NCA and graph baseline admission |
| [#14054 NEUROALIGN-001](https://github.com/Luminous-Dynamics/symthaea/issues/14054) | Compare actual Symthaea activations/dynamics to public neural data | Separate real-neural-data lane; not a substitute for morphogenesis validation |
| [Symtropy #2685 MORPH-FUNCTION-001](https://github.com/Luminous-Dynamics/symtropy/issues/2685) | Function-preserving repair beyond geometric similarity | Mechanical/load-path qualification; reuse its truss solver and receipt owners rather than add another solver |

No new parallel issue or crate is proposed here. In particular, #8405 and #8468 already cover identity and evidence-selection oracles, while Symtropy #2685 owns the function-preserving structural lane. Before implementation, inspect the active owners and code paths at the exact intended base commit; reuse shared HDC, temporal dynamics, memory, evidence and benchmark infrastructure only where the interfaces actually fit.

### 4.2 Three topology layers must remain distinct

The repository now has three related but non-interchangeable notions of topology:

1. **Semantic cognitive topology** — #13066/#13094: event-sourced, typed, provenance-bearing relations emitted by owning runtime subsystems. Rendering, camera, animation, procedural growth or geometry cannot create semantic edges or cognitive evidence.
2. **Morphological/spatial topology** — #13079 and the Mycelial Presence visual network: filaments, branches, contacts, fusion candidates, geometry and deterministic spatial realization. A geometric crossing is not an accepted biological fusion or semantic relation unless the corresponding source contract says so.
3. **Dynamical interaction/transport topology** — #13423/#13450 for observed fungal structure and distinct transport evidence; #14123 for a synthetic controller that actively edits an artificial graph under service-demand constraints.

This separation is a key architectural safeguard. Levin's biological work motivates operational questions about living pattern regulation. It does not validate a renderer, a semantic memory graph, or an arbitrary software topology as a biological analogue. Every layer needs its own source of authority, observables, intervention semantics and claim ceiling.

### 4.1 Existing executable slices and current code-status caution

The inspected `symthaea-alife` crate manifest and public `src/lib.rs` describe an artificial-life agent/population system centered on foraging/rest, energy, encounters and ecological interactions. That crate is not, by itself, an NCA or morphogenesis controller. A fetch of the guessed `crates/domains/symthaea-alife/src/gnca.rs` path returned 404; this is a narrow path check, not an exhaustive repository-wide inventory.

There are existing finite reference slices in open PRs: #8405 supplies the dependency-free chair/table identifiability oracle and exhaustive four-cell backrest deletion fixture; #8468 stacks active evidence-selection and decision-value fixtures on it. These are evaluator/oracle work, not a learned NCA, production repair controller or biological model. Separately, #13144 reports a structural mycelial failure-resilience evaluator with a BFS cross-check, but explicitly does not simulate transport or biological regrowth; #13079's network topology/visual morphogenesis work is likewise a separate synthetic geometry lane. The new #14123 is a proposed algorithmic graph-control benchmark, not a claim that a currently merged adaptive-topology controller exists. The #8405 PR body says exact-head Rust tests, formatting, Clippy and receipt workflow remain unverified. The latest retrieved #8468 comment records head `2e40571fa2a1186bfa13bf95a04f4b99e3c775fb` and queued checks at that time; a status query during this audit returned no status records for that head. Neither should be promoted to a Rust/CI pass without fresh exact-head evidence.

## 5. Benchmark architecture

### 5.1 Task families

**A. Identity under destructive ambiguity (P0).**  
Use the table/chair-style identifiability fixture: distinct undamaged histories produce the same damaged observation. Test a trusted target memory, absent memory, corrupted memory, conflicting memory, stale memory and malicious target substitution. The observation-only model must be allowed to abstain.

**B. Local repair and recovery (P0).**  
2D and 3D targets, local and distributed lesions, graded severity, repeated damage, observation noise and failed nodes. Measure full recovery trajectories rather than final snapshots only.

**C. Topology and scale generalization (P1).**  
Train on selected graph families and sizes; hold out graph families, node relabelings, size ranges, edge dropout, rewiring, fragmentation and reattachment. Prevent hidden position and generator identifiers from leaking.

**C2. Endogenous topology adaptation (P1; #14123).**  
Compare fixed graph, weight-only adaptation, local repair, distributed routing, adaptive edge edits and degree-matched random rewiring under demand shifts and failures. Charge graph-edit, message, latency and maintenance costs; include stationary-demand controls to penalize gratuitous rewiring.

**D. Multi-rate coordinated repair (P1).**  
Fast local / intermediate / slow distributed channels, with and without conditional invocation. Match total work and bytes. Include equal-budget random routing, delayed channels, phase-shuffled channels and one-rate controls.

**E. Functional morphostasis (P1).**  
A shape must preserve specified capabilities after repair, not just visual resemblance. Examples include graph reachability, load-path connectivity, transport continuity and collision-free navigation. Each task needs a separately validated oracle and explicit constraints.

**F. Endogenous target adaptation (P2).**  
The system begins with a distributed stable target state, experiences a declared context change, and must distinguish a legitimate target update from damage, adversarial target substitution or drift. Evaluate both stability and authorized change. This is where an attractor-based identity hypothesis is tested against external-target memory.

**G. Observation / intervention value (P2).**  
Partial observation, uncertain damage and costly diagnostics. The controller can act, sense, defer or abstain. Measure whether information acquisition changes decisions and reduces expected downstream loss.

### 5.2 Minimum baseline ladder

1. Current-state nearest-template classifier and nearest-template repair.
2. Local hand-coded cellular rule / reaction-diffusion-style baseline appropriate to each task.
3. Vanilla local NCA with pinned architecture, recipe, weights or faithful retraining protocol.
4. Smart Cellular Bricks reference model, admitted only after issue #8423 resolves artifact and licensing gates.
5. Distributed graph Weisfeiler–Lehman baseline for shape classification, plus appropriate graph baselines for task extensions.
6. BraiNCA-style attention/sparse long-range NCA, clearly distinguished from a faithful reproduction unless exact semantics are pinned.
7. Clean-room multiscale attractor model inspired by the 2026 hybrid framework, with fixed and adaptive edge weights; a dynamic-topology extension is a separate experimental arm.
8. Symthaea candidates: HDC only; temporal dynamics only; HDC × CfC/LTC; adaptive graph/communication only where a real implementation exists; full combination only after individual arms qualify.
9. Centralized MPC/optimizer or exact solver as an explicitly labelled upper bound, not a peer distributed baseline.
10. Random, no-memory, no-remapping, no-slow-channel, fixed-topology, shuffled-channel and target-blind controls.
11. TOPO-ADAPT tournament: fixed topology, weight-only adaptation, local greedy repair, fixed-topology distributed routing, adaptive edge editing and degree-matched random rewiring.
12. GNN communication rewiring: Delaunay/dynamic triangulation, spectral-gap/FOSR or expander-inspired augmentation, curvature/effective-resistance ranking, spectrum-preserving sparsification and virtual-node/hub shortcuts. These are message-propagation baselines, not capacity-aware service policies.
13. Resilience-based topology/capacity design: exact MILP or max-flow/min-cost upper bound on small cases, plus PSDA-style stochastic design and GA where the online-available early-access study's objective/constraints and artifacts can be reconciled.
14. One-step candidate-sweep oracle: full-visibility evaluation of no edit and all allowed single-edge edits, maximizing served flow then minimizing edit/maintenance cost. Upper bound only, not a decentralized or learned controller.

Every baseline receives the same observable inputs and target-memory privileges. Report both parameter-matched and work/communication-matched comparisons when both cannot be satisfied simultaneously.

### 5.3 Common episode and receipt contract

Every episode binds:
- source commit and source-tree digest;
- benchmark schema and evaluator version;
- task family, topology generator and immutable task/target digest;
- split identity and whether the episode is confirmatory;
- seed and deterministic replay inputs;
- observation channels and visibility rules;
- damage/perturbation schedule;
- model/checkpoint/configuration identity;
- parameter/state bytes, update count, message count/bytes and work proxy;
- full state and outcome trajectory;
- target-identity posterior, calibration and abstention;
- terminal outcome, failure reason and explicit missing/invalid status.

Freeze these fields and the primary decision rule before opening held-out evaluation. Missing measurements are typed missing/blocker states, never numeric zeroes. All attempted episodes must remain in the ledger.

## 6. Metrics: publish a dashboard, not one blended score

### Morphology and target identity
- Exact-mask IoU / precision / recall where exact geometry is meaningful.
- Surface Chamfer or Hausdorff distance for continuous/3D geometry.
- Target-relative missing and extra elements.
- Current-shape classification vs historical target-identity accuracy, scored separately.
- Calibration, selective risk, coverage and correct abstention under non-identifiability.

### Function and safety constraints
- Task-native function success and margin.
- Connectivity/reachability or load/transport invariant preservation.
- False-positive repair rate and severity-weighted constraint violations.
- Unnecessary modification of undamaged regions.

### Dynamics and adaptation
- Recovery success and recovery steps/time.
- Full target-error trajectory, overshoot, oscillation and post-recovery stability.
- Repeated-damage and delayed-failure performance.
- Adaptation to held-out target/topology changes.
- Worst-group and per-condition performance, not only average.

### Resource and deployment
- Total node updates and actual inference work.
- Messages and bytes communicated, including coordination overhead.
- Per-node state bytes, total memory and model parameters.
- p50/p95/p99 latency and throughput.
- Energy only where measured with a declared hardware and measurement boundary.
- Quality-versus-compute, quality-versus-communication and quality-versus-energy curves.

Keep these values separate. A single composite may be a secondary decision aid only if weights are frozen before testing and all raw metrics remain visible.

## 7. Competitive targets and SOTA claim rules

The numerical goals below are **proposed engineering targets**, not achieved scores or forecasts. Final margins and test sizes must be preregistered after the published baselines are reproduced.

| Gate | Proposed target | What it would support |
|---|---|---|
| Published baseline fidelity | Reproduce each selected reference result within 1 percentage point where the paper reports a directly comparable metric, or explain the exact discrepancy | Baseline reproduction for that task/metric only |
| Identity safety | At least 99% correct abstention on deliberately non-identifiable/no-trusted-memory cases; wrong confident repairs at or below 1% on the declared ambiguity suite | Reliable handling of the tested ambiguity cases |
| Severe damage | At least +5 percentage points absolute recovery success over the strongest matched decentralized baseline on preregistered severe/OOD conditions, with paired 95% CI excluding zero | Task-bounded performance advantage |
| Efficiency frontier | At equal task quality within a preregistered 1-point non-inferiority margin, at least 2× lower communication bytes or measured inference work | A bounded efficiency advantage, not universal superiority |
| Topology transfer | Retain at least 90% of in-distribution task performance on held-out graph families and 2× size extrapolation, without hidden topology identifiers | Generalization over the declared graph family |
| Calibration | Better selective risk/coverage than baseline, with no material regression in worst-group error | Better uncertainty handling |
| Ablation attribution | Each claimed mechanism loses its advantage under its corresponding removal/shuffle intervention; negative controls do not reproduce the win | Evidence that the mechanism matters rather than merely model size or compute |
| Independent reproduction | Independent rerun reproduces the key effect within the frozen tolerance and emits compatible receipts | Reproducible claim for that benchmark |

**SOTA is not a single paper score.** A superiority claim requires (i) the strongest compatible baseline, (ii) task and information parity, (iii) matched or fully disclosed compute/communication, (iv) preregistered held-out tests, (v) uncertainty and worst-group reporting, and (vi) independent reproduction. If a simpler system wins, keep it as the selected system.

### 7.1 What would count as a genuinely differentiated result?

The most valuable result is not a small gain in shape accuracy. It is a measured **quality–resilience–cost frontier**:

- same or better function after more severe damage;
- correct identity retention when current morphology is misleading;
- calibrated abstention when identity is not recoverable from available information;
- graceful performance as communication is delayed, dropped or rewired;
- lower communication/work per successful recovery;
- transfer to new topology and scale without re-training on the test family.

That combination is a credible engineering differentiator over local NCA and centralized repair, if the comparisons survive audit.


## 7.2 Separate industrial validation lane: engineering operations and repair

This is a transfer/utility programme, not part of the biological or morphogenesis score. The same architecture hypotheses become useful to industry only when they improve a real decision under operational constraints.

### Industry comparators and entry points

- **NIST Super Intelligence for Manufacturing:** NIST describes planned work on evaluation and verification/validation/uncertainty quantification for industrial AI and digital twins, including Tennessee Eastman process fault detection/diagnosis and preventive-maintenance testbeds. Its public plan is a roadmap for evaluation work, not a completed leaderboard or an industry-wide production benchmark: https://www.nist.gov/programs-projects/super-intelligence-si-manufacturing

The more useful comparison is NIST's **TIDAL** reference pattern: Teamed, Integrated, Digital-twin-assured and Layered Intelligence. It organizes functions across asset, site/area and enterprise tiers; combines human-machine teaming, contextualized OT data (SCADA/MES/ERP/streaming sensors), knowledge graphs and digital twins; and proposes an integration test bench with architecture alternatives such as UNS/MQTT, I3X and OPC UA. The initial described test case is Tennessee Eastman, with fault detection/diagnosis, preventive maintenance and digital-twin VVUQ tasks. NIST lists accuracy, latency, cost and grounding; its fault-diagnosis description prioritizes detection recall ahead of precision and includes progressive context ablation. These are planned research outputs, not a finished downloadable production leaderboard or benchmark harness. <Cite refs={["turn534376search0","turn534376search1"]}/>

For Luminous, a possible systems mapping is: Symthaea proposes and evaluates bounded topology/configuration actions; Mycelix carries authority, provenance and decision receipts; Symtropy checks the relevant physical/engineering constraints; Sol-Atlas maintains spatial/topological context. This is an integration hypothesis, not a current validated end-to-end industrial system. Human operators/asset owners retain decision authority; begin with offline and shadow-mode evaluations, never direct live actuation from the synthetic benchmark.
- **Industrial datasets:** use task-native datasets only after license, split, sensor semantics and train/test contamination review. Relevant families include NASA C-MAPSS / N-CMAPSS, PHM2012/PRONOSTIA, IMS and Paderborn bearing data, CWRU bearing data, Tennessee Eastman Process, and MVTec AD. An IEEE IES dataset catalogue is a discovery index, not the authoritative license holder: https://ieee-ies-industrial-ai-lab.github.io/industrial-ai-hub/datasets/. Review common dataset usage and task limitations before selecting an evaluation: https://doi.org/10.1016/j.icte.2025.11.013
- **Industrial robotic repair:** current industrial demonstrations use conventional robot integration—RGB-D/force sensing, CAD/device graphs, ROS 2, behavior trees, teleoperation and digital-twin support. This is a relevant incumbent architecture to compare against, not evidence that morphogenetic control has already displaced it: https://www.maxrobotics.com/en/news/a-modular-dual-arm-robotic-cell-for-disassembly-and-repair-of-industrial-control-electronics
- **Reconfigurable production digital twins:** compare against deployed engineering patterns such as digital twins, Asset Administration Shell representations and ROS 2-based plug-and-produce workflows; treat a research demonstration as a comparator, not an industry standard: https://doi.org/10.1016/j.procir.2025.02.246

### Industrial tasks

1. **Condition monitoring / fault diagnosis:** identify faults under changing load, sensor drift, missing channels, correlated failures and domain shifts.
2. **Maintenance decision support:** choose inspect, continue operation, derate, repair, replace or abstain; quantify the cost of false alarms and missed faults.
3. **Repair/configuration provenance:** maintain the intended configuration across component substitutions, repairs and firmware/software revisions; reject stale or untrusted asset identity.
4. **Function-preserving recovery:** assess whether the repaired asset restores declared service capacity, structural integrity or process constraints—not just whether it resembles its CAD/BOM configuration.
5. **Reconfigurable cells:** adapt the communication and task graph after equipment removal, module replacement or changed process routing while maintaining safety constraints.

### Industrial transfer candidate: distribution-grid reconfiguration

A 2026 survey of decentralised electricity-distribution architectures argues for hybrid, multi-timescale coordination: local fast control, slower inter-cell/federated coordination, and a self-healing overlay during faults/cyber events. Suggested measures include scalability versus DER count, recovery time, load served under contingencies, communication complexity, data governance/attack surface, and economic/fairness outcomes. DOI: https://doi.org/10.1016/j.compeleceng.2026.111279. <Cite refs={["turn540123search3"]}/>

This is a high-value second industry lane for topology adaptation, but it is **not** the generic max-flow benchmark. Distribution-grid evaluation must include domain-qualified power-flow semantics, voltage/thermal constraints, radiality/islanding, protection coordination, switching limits, DER uncertainty and operator authority. Candidate baselines include established distribution-system reconfiguration and OPF solvers, MILP/MIQP/MISOCP formulations, branch-exchange heuristics, GA/PSO, and safety-constrained learned graph/DRL policies where their operating semantics are reproducible.

Metrics should include load served by criticality, unserved energy, voltage/thermal violations, switching count/wear, restoration time, losses, feasibility, communication overhead and fairness across affected loads/customers. Begin in offline digital-twin simulation and shadow mode. The synthetic TOPO-ADAPT result cannot establish grid safety, and no live switching is authorized or proposed by this benchmark.

### Strong baseline ladder

- Conventional condition-monitoring/statistical process-control baselines appropriate to the sensor and process.
- Task-native classical ML such as logistic/linear models, random forests, gradient-boosted trees and calibrated one-class/anomaly models.
- Strong sequence/time-series baselines where the task warrants them (e.g. temporal CNN, GRU/LSTM, state-space model), with matched data exposure and tuning.
- Physics/process-model and digital-twin baselines where calibrated models and valid operating envelopes exist.
- Existing CMMS / rule / threshold / maintenance-planning workflows, when access allows a fair prospective comparison.
- Industrial robot orchestration baseline: CAD/device graph + perception/force sensing + ROS 2 / behavior-tree / human-in-the-loop recovery.
- Symthaea candidate only when connected to validated data/actuation adapters and evaluated in shadow mode before any control authority.

### Metrics that determine industrial value

- Fault detection recall by severity (NIST's current project description weights recall ahead of precision for fault diagnosis), precision, false alarms per asset-hour and missed-critical-fault rate.
- Grounding: whether the system selects the correct OT data/context, ties each diagnosis to source evidence and preserves asset/process context.
- Context ablation: compare minimal tags, richer structured context and digital-twin context under the same task and resource budget; measure the effect on reliability, grounding, latency and cost.
- Detection lead time and remaining-useful-life error/calibration only when the dataset supports those labels.
- Diagnosis top-k accuracy, calibrated uncertainty and correct abstention on unknown faults.
- Expected maintenance loss / decision regret under a published cost matrix; avoided downtime and service-level impact.
- Repair success, recurrence, mean time to repair, rework and warranty/quality escape rate.
- Constraint/safety violations and unsafe action rate; human override and escalation rate.
- p50/p95/p99 end-to-end latency, data/compute cost, communication overhead and measured energy boundary.
- Operator workload and auditability: evidence needed to justify the action, reproducibility of the work order, and traceable state/configuration lineage.

### Promotion protocol

1. Start in retrospective, frozen-split evaluation; then a shadow-mode prospective period that cannot issue commands.
2. Compare on the same assets, sensor windows, fault horizons, labels and information availability. Use chronological and asset-level splits to prevent leakage.
3. Evaluate distribution shift explicitly: unseen assets, operating regimes, fault families, sensor replacements, missing data and maintenance interventions.
4. Report costs with the asset owner, including the cost of missed faults, unnecessary interventions, downtime, data collection and integration.
5. Only move to recommendation mode after calibration and human-factors review; any control/actuation mode needs separate safety engineering, operational authorization and fail-safe design.
6. Claim industrial superiority only on a named application, dataset/asset fleet and deployment horizon with measured business outcomes. Do not use an academic dataset score as proof of factory ROI.

**Critical distinction:** industrial predictive maintenance, industrial robotic repair and biological morphogenesis are separate problem domains. A strong score in one cannot be transferred as evidence in another. NIST's VVUQ and grounding work is a useful methodological comparator; operational ROI and reliability are the acceptance criteria for actual industry adoption.

## 8. Statistical and adversarial discipline

- Split by target family, graph family, topology generator and perturbation family; no near-duplicate leakage.
- Use paired seeds and multiple independent training seeds; report sample counts and confidence intervals.
- Train/tune only on train/validation partitions. Do not select thresholds, checkpoints, metrics or stopping points on held-out outcomes.
- Include class imbalance, adversarial target substitution, impossible targets, stale memory, duplicated/stale messages, non-finite state and evaluator-leakage tests.
- Run graph-isomorphism and node-relabeling metamorphic controls.
- Match parameter count, update count, total message bytes, observations and action budgets; report unmatched comparisons as exploratory.
- Mutation-test the evaluator and each critical gate. An unexpected compile failure is not a killed mutant.
- Publish nulls, failures, invalid episodes and terminal dispositions.

## 9. Execution sequence — no runner dependency

### Gate 0 — Source, ownership and rights
- Keep #8423 as the source-admission authority for Smart Cellular Bricks.
- Pin paper version, repository commit, artifact metadata, data terms, model provenance and digests.
- Do not redistribute or adapt code/weights while license/permission is unresolved.
- If legally permissible reproduction is unavailable, create a clean-room implementation from the paper and label it accordingly.

### Gate 1 — Qualify existing identity and evidence oracles
- Review and exact-head qualify existing #8405; do not recreate its chair/table fixture.
- Review and exact-head qualify stacked #8468; distinguish Brier information value from downstream EVSI and robust policy loss.
- Verify provenance inputs are scenario assumptions, not cryptographic validation.
- Freeze target digest, split, damage schedule and receipt contract; mutation-test the evaluator.

### Gate 2 — Qualify function-aware recovery
- Reuse Symtropy #2685 and its existing planar-truss/receipt owners; PR #2363 is the auditable planar-truss reference solver.
- Prove the geometry-only baseline accepts at least one visually close but mechanically invalid structure and distinguish a functionally valid alternative repair.
- Keep mechanical solver limits and claim ceiling intact.

### Gate 3 — Reproduce strong simple baselines
- Implement/reference the nearest-template and deterministic local rule.
- Reproduce the distributed graph-WL result on the task where it applies.
- Admit the Smart Cellular Bricks NCA only when #8423's source/data/rights gate passes.
- Reproduce or clearly label any BraiNCA and hybrid-attractor implementation.

### Gate 4 — Qualify endogenous topology adaptation
- Follow #14123: start with hand-checkable 12–20-node max-flow fixtures; [PR #14125](https://github.com/Luminous-Dynamics/symthaea/pull/14125) provides v0.1 stationary-control, new-sink and bridge-failure fixtures, while [PR #14127](https://github.com/Luminous-Dynamics/symthaea/pull/14127) adds the draft Rust evaluator. The evaluator has not yet been compiled or run.
- Reproduce fixed-topology, weight-only, local greedy, distributed routing and exact small-instance oracle baselines.
- Add adaptive edge edits only after exact budget, feasibility and deterministic receipt tests pass.
- Include degree-matched random rewiring, stationary-demand false-edit controls and held-out topology/demand families.
- Keep this synthetic controller benchmark separate from #13144 structural failure scores and #13423/#13450 biological observations.

### Gate 5 — Mechanism factorial
Test one change at a time before the combined candidate:
- representation remapping (#8372);
- fixed long-range vs adaptive routing;
- multi-rate channels (#8388);
- target memory and identity (#8401);
- structured vs shuffled/noise micro-updates (#8409);
- fixed graph vs adaptive weights vs adaptive topology;
- external target vs distributed attractor memory;
- active observation vs fixed observation budget.

### Gate 6 — Symthaea integration
Audit exact code ownership and public interfaces before implementation. Add only a narrow adapter if the existing HDC/LTC/memory systems can be called without coupling a benchmark evaluator to production behavior. Do not add a new general-purpose crate or duplicate a shared memory/evidence substrate without proving a gap.

### Gate 7 — External validation lanes
- DynamicAtlas: observational developmental trajectories, subject to licensing/access and data-contract review.
- Public neural/bioelectric imaging: only where raw data, modality, calibration, timestamps and data license are sufficiently documented.
- NEUROALIGN (#14054): keep actual neural predictivity separate from synthetic morphology-controller results.
- Physical modular hardware: a later, separately funded qualification; do not describe simulated regrowth as physical repair.

## 10. Claim ladder

| Evidence obtained | Permitted claim | Not permitted |
|---|---|---|
| Deterministic fixtures pass | Evaluator behaves as specified on its fixtures | Learned regeneration or biological analogy |
| Synthetic baseline comparison passes | Candidate is better on the named synthetic task and split | General SOTA, biological intelligence or tissue regeneration |
| Matched-cost held-out win | Bounded performance/efficiency advantage on tested conditions | Universal superiority |
| Independent software reproduction | Reproducible result for the declared implementation/task | Biological mechanism |
| Fit to public biological trajectories | Predictive correspondence on the declared dataset | Causality or mechanism |
| Controlled independent biological study | Only the specific biological result supported by that study | Consciousness or general claims beyond its design |

No synthetic or observational benchmark here establishes consciousness, sentience, biological equivalence, medical benefit or permission to intervene in living systems.

## 11. Immediate dispositions

1. **Proceed now:** exact-head review and qualification of existing #8405 and #8468; do not rebuild their identity/evidence-selection fixtures.
2. **Execute #14123 P0 slice:** review [PR #14125](https://github.com/Luminous-Dynamics/symthaea/pull/14125) and [PR #14127](https://github.com/Luminous-Dynamics/symthaea/pull/14127); compile/test the evaluator at its exact head, then qualify fixed-topology, weight-only, greedy and exact-oracle baselines before introducing adaptive rewiring.
3. **Parallel:** qualify Symtropy #2685 and reuse the existing planar-truss solver and receipt owners; do not add a second mechanics engine.
4. **Parallel:** complete #8423's Smart Cellular Bricks artifact/rights audit without importing code or weights.
5. **Biological lane:** keep #13423/#13450/#13460/#13425 independent: validate data rights, replicate/time splits and separate transport observables before modeling.
6. **Then:** build the common task harness and reproduce strong local, graph-WL and NCA baselines; only afterwards admit Symthaea arms.
7. **Do not wait for runners:** source, task definitions, receipt semantics, baseline contracts and mutation fixtures can be audited independently of full-workspace CI.
8. **Do not claim a win yet:** no new controller, model run, benchmark score or physical test is reported by this document.

## References

- Hartl, B., Levin, M., & Pio-Lopez, L. (2026). *Neural cellular automata: Applications to biology and beyond classical AI*. Physics of Life Reviews. https://doi.org/10.1016/j.plrev.2025.11.010
- Pio-Lopez, L., Hartl, B., & Levin, M. (2026). *BraiNCA: brain-inspired neural cellular automata and applications to morphogenesis and motor control*. https://arxiv.org/abs/2604.01932
- Moreno, R. et al. (2026). *Smart cellular bricks for decentralized shape classification and damage recovery*. Nature Communications. https://doi.org/10.1038/s41467-026-75166-7
- Cortés-Poza, Y. (2026). *A hybrid mathematical framework for morphogenesis and regeneration*. Journal of Mathematical Biology. https://doi.org/10.1007/s00285-026-02459-2
- McMillen, P. & Levin, M. (2026). *Lifetime imaging reveals long-distance non-neural bioelectric patterns on timescales from seconds to hours*. Developmental Biology. https://doi.org/10.1016/j.ydbio.2026.08.001
- DynamicAtlas (2025/2026). *A morphodynamic atlas for Drosophila development*. Nature Methods. https://doi.org/10.1038/s41592-025-02897-8
- Levin, M. & Resnik, D. (2026). *Mind Everywhere: A Framework for Conceptualizing Goal-Directedness in Biology and Other Domains, Part II*. Biological Theory. https://doi.org/10.1007/s13752-025-00524-5
