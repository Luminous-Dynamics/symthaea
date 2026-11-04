# INVENTOR-LOOP-001 — Evidence-Driven Invention and Human Capability Program

**Status:** Research and integration proposal v0.1  
**Tracking:** [Issue #10442](https://github.com/Luminous-Dynamics/symthaea/issues/10442)  
**Date:** 2026-11-04  
**Claim ceiling:** This document specifies a programme and prospective tests. It does not establish an autonomous engineering capability, a validated physical design, or a benchmark result.

## 1. Thesis

The valuable unit of progress is not an AI-generated answer, image, mesh, CAD file or simulation run. It is a **human-understandable, editable, reproducible engineering result** whose requirements, model, assumptions, analyses, experiments, physical observations and limitations remain connected.

The intended loop is:

```text
human intent
  -> needs and requirements
  -> explicit hypotheses and unknowns
  -> typed scientific / engineering model
  -> candidate architectures and design artifacts
  -> analysis at appropriate fidelity
  -> discriminating experiment
  -> physical measurement
  -> model/design revision
  -> independent review and reproducible record
  -> human learning and shared discovery
```

The programme asks two separate questions:

1. Can the system help produce an artifact that survives independent engineering evaluation?
2. Does the human become more capable of understanding and solving a new problem without the system?

A successful project must answer both. A system that builds the artifact while making the learner dependent has failed the second objective.

## 2. Architecture: compose existing owners

Do not add a second systems-engineering authority layer, physics ontology, generic task schema, scheduler, run-receipt format, evidence graph or solver.

| Concern | Existing owner / contract | Role in this programme |
|---|---|---|
| Requirements, architecture, lifecycle, authority and assurance | [Systems Engineering #3677](https://github.com/Luminous-Dynamics/symthaea/issues/3677), ETK | Need-to-requirement traceability, hazards, verification obligations, authority/currentness |
| Solver-independent model | [PhysicsIR #6879](https://github.com/Luminous-Dynamics/symthaea/issues/6879) | Units, dimensions, frames, equations, assumptions, conditions, parameters, uncertainty, solver intent and explicit non-claims |
| Retrieval, repository and engineering-agent task evaluation | [CEB v0 #10115](https://github.com/Luminous-Dynamics/symthaea/pull/10115), [CEB intelligence #10419](https://github.com/Luminous-Dynamics/symthaea/issues/10419) | Task contract, comparison arms, source context, task/run identity and scored agent behaviour when qualified |
| Evaluator isolation and custody | [CEB task custody #10436](https://github.com/Luminous-Dynamics/symthaea/issues/10436), [execution isolation #10064](https://github.com/Luminous-Dynamics/symthaea/issues/10064) | Keep hidden oracle, credentials and grader outside the agent trust domain |
| Hardware/device composition | [ENG-DEVICE-000 #5670](https://github.com/Luminous-Dynamics/symthaea/issues/5670) | Reusable component, power, thermal, material, sensor and interface profiles |
| Reverse engineering | [REA benchmark #9969](https://github.com/Luminous-Dynamics/symthaea/issues/9969), [causal diagnosis #9911](https://github.com/Luminous-Dynamics/symthaea/issues/9911) | Observation, localization, identifiability, hypothesis testing and next-experiment selection |
| Structural mechanics integration target | Symtropy [PHYS-TRUSS-001 #2361](https://github.com/Luminous-Dynamics/symtropy/issues/2361), core solver [open PR #2363](https://github.com/Luminous-Dynamics/symtropy/pull/2363), and open/draft follow-ons: mutation [#2462](https://github.com/Luminous-Dynamics/symtropy/pull/2462), symmetric receipt [#2466](https://github.com/Luminous-Dynamics/symtropy/pull/2466), sweep [#2472](https://github.com/Luminous-Dynamics/symtropy/pull/2472), energy [#2476](https://github.com/Luminous-Dynamics/symtropy/pull/2476), asymmetric oracle [#2480](https://github.com/Luminous-Dynamics/symtropy/pull/2480), asymmetric receipt [#2483](https://github.com/Luminous-Dynamics/symtropy/pull/2483), conditioning [#2498](https://github.com/Luminous-Dynamics/symtropy/pull/2498), bounded truss intervals [#2501](https://github.com/Luminous-Dynamics/symtropy/pull/2501), [interval decision #2502](https://github.com/Luminous-Dynamics/symtropy/issues/2502), [benchmark manifest #2358](https://github.com/Luminous-Dynamics/symtropy/issues/2358) | Intended digital mechanics integration target, not a merged mainline API: PR #2363 is open/unmerged and the cited follow-ons are open/draft. Confirm owner-approved exact head/interface and qualification before use. Do not create a parallel solver/oracle/receipt. These checks are mathematical/software verification, not physical validation. |
| Physical trials and metrology | Symtropy [trial evidence #3522](https://github.com/Luminous-Dynamics/symtropy/pull/3522), [round-robin protocol #3592](https://github.com/Luminous-Dynamics/symtropy/pull/3592), [study #3591](https://github.com/Luminous-Dynamics/symtropy/issues/3591) | As-built identity, calibrated measurements, ODD, uncertainty, attempted-run accounting and reproducibility |
| Learner experience | Mycelix Praxis [planner/learning benchmark #5286](https://github.com/Luminous-Dynamics/mycelix/issues/5286), [non-harm protocol #5461](https://github.com/Luminous-Dynamics/mycelix/issues/5461), [history-grounded teaching #5455](https://github.com/Luminous-Dynamics/mycelix/issues/5455), [support router #5467](https://github.com/Luminous-Dynamics/mycelix/issues/5467), [bounded Symthaea advisor #5472](https://github.com/Luminous-Dynamics/mycelix/issues/5472), [research PR #5446](https://github.com/Luminous-Dynamics/mycelix/pull/5446) | Age-appropriate explanation, learner prediction, progressive scaffolding, support selection, non-harm, reflection and independently measured transfer. Reuse existing planner, BKT/SRS and assessment boundaries. |
| Shared invention and contribution history | Mycelix | Rights, opt-in sharing, provenance, attribution, reproducibility, collaboration and durable negative knowledge |

Mycelix's current README reports Praxis as live but also says its flagship source does not compile at the current HEAD and that the live site is served from an older pinned IPFS snapshot; treat this as a repository self-report requiring a separate exact-head reconciliation, not an independently verified build result. The advisor boundary in Mycelix #5472 is the integration path; do not create a competing tutor, learner-state model or credential authority. The present monorepo Praxis scaffold must not be represented as a finished engineering-learning product.

## 3. Paradigm shifts

### 3.1 From answer generation to experimental closure

Maintain competing hypotheses and identify the observation or intervention that best distinguishes them. Choose the next experiment by expected information gain or decision value relative to cost, time and risk. State a prediction before a trial; update from independently observed data; retain negative results.

An experiment-selection policy should be evaluated against random, fixed-sequence, expert-authored and uncertainty-only baselines. The system must not grade its own hypotheses using evidence it generated itself.

### 3.2 From a CAD object to a complete engineering subject

A render is not a design, a mesh is not a manufacturable part, and a valid STEP solid is not a verified product. Bind the candidate to editable geometry, topology, interfaces, tolerances, material/process assumptions, BOM, configuration, analysis inputs and verification obligations.

Use qualified existing CAD/CAE/CFD/circuit/structural tools through adapters. Keep the native Symtropy physics research path distinct from specialist solver backends where they serve different purposes.

### 3.3 From one solver answer to a model-error budget

Use an evidence-appropriate hierarchy:

1. dimensional checks and analytic bounds;
2. reduced-order or coarse numerical analysis;
3. higher-fidelity specialist simulation;
4. sensitivity and uncertainty analysis;
5. physical measurement;
6. prediction-residual calibration and model revision.

Keep parameter uncertainty, measurement uncertainty, numerical/discretisation error and model-form discrepancy distinct. A surrogate is an accelerator, not an authority: define its training domain, calibration, out-of-distribution detection and high-fidelity fallback.

### 3.4 From visual reverse engineering to identifiability-aware system identification

Infer candidate geometry and structure from scans, drawings, teardown, signals and controlled observations. Distinguish directly observed facts from inferred quantities. When multiple internal models explain the available evidence, preserve the equivalence class and choose a safe discriminating measurement. Never infer material, heat treatment, fatigue history or hidden function from appearance alone.

### 3.5 From AI completion to human capability growth

Praxis should make the learner perform meaningful cognitive work: predict, sketch, estimate, derive, test, explain and revise. Scaffolding should reduce as competence grows. Track whether help was requested, which hint was needed and whether a later task can be solved without assistance.

A child should be allowed to own the question and the design. AI assistance must not silently become AI substitution.

### 3.6 From isolated sessions to cumulative invention commons

Mycelix can preserve design lineage, authorship, licensed inputs, contributor roles, failed approaches, measurement provenance and reproducibility notes. Sharing is opt-in. Children's personal data, precise location, school identity and private work remain protected by default. No token or public-disclosure requirement should be imposed on learning.

## 4. First vertical slice: tabletop bridge / truss

Start with a small, supervised, non-human-load-bearing educational project. The initial learner-facing, design-only brief is [INVENTOR-LOOP-001 Bridge / Truss Design Challenge v0.1](INVENTOR_LOOP_001_BRIDGE_TRUSS_LEARNER_PACKET_V0_1.md). Its intended digital mechanics target is Symtropy's open PHYS-TRUSS PR series (#2361/#2363 and the follow-ons in the owner table); #2363 is still open/unmerged, and the follow-on PRs are open/draft. Until the Symtropy owner confirms an exact reviewed head, supported request interface and qualification state, keep the activity to paper/hand calculations and do not claim solver integration. Once approved, consume that work rather than implementing a second truss kernel, analytic oracle or receipt format. The novel contribution is the full human-to-verified iteration and teaching loop. The PR-described truss tests are mathematical/software verification of a limited static model—not experimental physical validation. A guarded physical classroom fixture remains a distinct, future metrology study under the existing trial/evidence owners.

**Problem packet**

- A declared span and envelope.
- A small set of available, characterized material and joint options.
- A mass or material budget.
- A low-load test envelope chosen after fixture and risk review.
- Required displacement/strength or load-retention criteria, with units and decision margins.
- A permitted design/manufacturing method and an explicit list of prohibited assumptions.

**Mathematical path**

- geometry, vectors and trigonometry;
- force balance and free-body diagrams;
- moments and equilibrium;
- stiffness and deformation;
- material properties and dimensional consistency;
- uncertainty, sensitivity and constrained optimisation.

**Engineering path**

- generate multiple truss topologies;
- derive first-order analytical bounds;
- build an editable design and complete member/joint inventory;
- run a pinned structural solver as a separate analysis result;
- identify the most informative and safest next measurement;
- build a guarded low-energy test article after review;
- compare measured deflection/failure observations against predeclared predictions;
- preserve all runs, failures and revisions.

**Learning path**

The learner predicts the critical member and expected deformation before simulation and test, explains the force path afterwards, and completes a structurally different transfer task later without step-by-step assistance.

The task is not a real bridge design and cannot be used for people, buildings, public infrastructure or other safety-critical applications. A pilot must determine materials, load range, measurement reference, calibration, sample size and stopping rules before confirmatory trials.

## 5. Benchmark programme

Use existing external benchmarks as *separate diagnostic instruments*, not one blended leaderboard:

- **Physical-Engineering's Last Exam (PLE):** long-horizon, multi-file professional engineering deliverables and hard constraints. Mercor's live leaderboard reports 48.3% on its 56-task hidden set; the PLE paper abstract separately reports a 50.6% strongest configuration. These are distinct published evaluation summaries and must not be treated as interchangeable. Mercor reports that only 41% of designs cleared every hard check in its reported analysis. The leading run averages about 27 million tokens and 63 minutes per task, making cost and latency essential comparison dimensions. These scores are comparable only under the exact official evaluator conditions.
- **EngiBench:** foundational retrieval, contextual reasoning and open-ended modelling; its authors report declining performance under increased complexity and perturbation.
- **EngDesign:** 101 structured tasks across nine domains, with 473 gradable items. Its public historical leaderboard reports o3 at 34.38% average pass rate; iterative refinement approached 60% after ten attempts on that older evaluation. Treat this as a historical benchmark result, not a current frontier-model baseline.
- **CADGenBench:** valid STEP/BREP generation and editing, with validity, shape, interface and topology metrics.
- **CADBench:** 18,000 samples over six benchmark families and five input modalities, with geometric-fidelity, executability and program-compactness measures. Its paper reports that specialized mesh-to-CAD systems can outperform general VLMs under ideal inputs, while modality shift and geometric complexity still cause brittleness. It is distinct from CADGenBench.
- **REA/SRE-Bench:** authorized reverse-engineering comparisons; use the existing REA owner rather than cloning private test assets.

### Evaluation axes

| Axis | Primary measures | Non-compensable rule |
|---|---|---|
| Requirements | Coverage, traceability, ambiguity detection, assumption errors, abstention | Missing critical requirement cannot be marked satisfied |
| CAD/geometry | STEP/BREP validity, editability, surface/volume deviation, topology, interface match | Invalid solid fails the validity gate |
| Analysis | Analytic reference error, convergence, invariant residuals, solver agreement, uncertainty coverage | Convergence alone is not physical truth |
| Design | Hard feasibility, robustness, Pareto quality, mass/cost/performance, repairability, material use | Critical constraint failure cannot be averaged away |
| Reverse engineering | Localization, parameter error on identifiable variables, causal precision, next-experiment utility | Ambiguous/unobservable cases require abstention |
| Physical evidence | Calibration, reference uncertainty, repeatability, inter-lab reproducibility, all-run accounting | Safety/ODD and performance remain separate |
| Learning | Delayed retention, transfer, independent solution quality, learner agency, educator burden | Project completion is not proof of learning |
| Efficiency | Cost/time per independently accepted artifact, rework, energy, tokens, latency | Report failures and all denominators |

### Matched comparison arms

1. Human novice and qualified-expert cohorts, separately.
2. Strongest available model with no specialist tools.
3. Same model with a commercial/open CAD/CAE toolchain.
4. Same model with Symthaea orchestration, with experimental Symthaea components ablated.
5. Symthaea + specialist solvers + Symtropy physical evidence loop.
6. Standard project-based learning compared with Praxis-supported project-based learning, using separate education protocols.

Keep model revision, prompts, task inputs, tools, runtime budgets and grading fixed for paired comparisons. Use blinded/independent adjudication. Report cost, latency, failure, refusal and abstention in addition to quality.

### Prospective competitive gates

These are programme targets, not existing Luminous results:

- Establish a reproducible external baseline before claiming a lift.
- Target ≥5 percentage-point improvement over the strongest matched agent/tool baseline on a held-out digital engineering task set, without worse critical-constraint performance.
- PLE stretch target: ≥55% under the same official task/evaluation conditions, compared with the published 48.3% reference; do not claim improvement without a matched reproduction.
- Reverse-engineering target: reuse #9911/#9969's prospective ≥20% relative top-3 localization lift, no increase in false causal diagnoses, and correct abstention on ambiguity.
- Reject every unit/dimension/frame invalid control in the initial PhysicsIR corpus; require zero false-pass on known-bad and boundary controls.
- Educational pilot target: ≥0.30 standard-deviation improvement in delayed transfer against an active project-based comparison, with independent problem solving non-inferior. Preregister duration, primary endpoint and sample-size rationale.
- Choose physical reproducibility margins and sample size after a pilot estimates variance and cross-laboratory confounding. Do not set arbitrary universal tolerances.

Never produce a single score that hides a critical safety, security, code, material or regulatory failure.

## 6. Industry positioning

We should not attempt to replace mature CAD or solver companies with an immature in-house clone.

- **Autodesk Fusion** already offers AI-assisted AutoTimeline reconstruction of editable design history from imported models, plus System Modeler workflow automation. The October 2026 release reports reconstruction of extrudes, revolves, chamfers, fillets, holes and basic sweeps. The correct Luminous strategy is interoperability and an evidence/learning loop around specialist CAD, not competing on feature-history reconstruction alone.
- **NVIDIA PhysicsNeMo** is an open-source physics-ML stack for neural operators, GNNs, transformers and physics-informed learning. Use qualified models as acceleration/surrogate candidates, with Symtropy or independent solvers and physical data retaining the verification role.
- **Commercial CAE platforms** offer powerful solvers and cloud execution. The distinctive opportunity is connecting requirements, multiple solvers, uncertainty, experiments, measured outcomes, learning and public provenance without tying the entire workflow to one vendor. PhysicsNeMo is a capable open physics-ML stack for neural operators, GNNs, transformers and PINNs; treat its learned models as optional, versioned surrogate candidates, not replacements for qualified solvers or a validation authority.
- SimScale's 2026 survey of 350 senior engineering leaders reports agent use in 19% of simulation/CAE workflows and 11% of CAD workflows; the survey is vendor-produced and self-reported, not a controlled performance benchmark. It suggests adoption is still early, but it does not prove any specific technical gap.

### How Luminous can compete rather than imitate

Do not attempt a broad "all engineering" leaderboard sprint. Use a wedge strategy:

1. **Win a narrow workflow end to end.** Start with the paper-first bridge/truss learner journey, then a single authorized reverse-engineering task. Prove the complete requirements → model → design → analysis → experiment-selection → measured revision loop before adding more domains.
2. **Exploit model and solver complementarity.** Benchmark strong frontier-model-only and model-plus-commercial-tool baselines first. Symthaea's potential value is context continuity, contradiction detection, typed scientific models, uncertainty-aware reasoning and selecting the next useful observation; Symtropy should call suitable specialist solvers rather than recreate their mature numerics.
3. **Optimize all-gates success, not partial-credit theatre.** PLE's 91% average individual-constraint adherence but 41% all-hard-check clearance is a warning: near-correct is still not acceptable in physical engineering. Report hard-gate pass probability, failure severity and justified abstention before cost/weight optimization.
4. **Make experiment choice a first-class capability.** Compare random, fixed-sequence, expert-selected and model-selected experiments under an identical budget. Score information gain only when it changes a decision or reduces uncertainty in a decision-relevant quantity.
5. **Make the human a co-inventor, not a prompt operator.** Demonstrate both artifact quality and unaided transfer. Offer support-mode choice and progressive hints; preserve the ability to work offline, ask a teacher/peer, decline AI or keep a project private.
6. **Publish failure as well as success.** Release permitted fixtures, exact versions, input/output digests, mutation controls, exclusions, cost and failure analyses. Independent reproduction is part of the product, not a marketing afterthought.

A result should be called better than SOTA only after the incumbent baseline is replicated on the same held-out tasks, exact grader, tools and resource budget, with uncertainty intervals and independent review. Otherwise report it as a promising local result, not a leaderboard win.

Our differentiation should be **decision quality per independently accepted artifact**, with a human who learns, and an evidence record that an external reviewer can inspect.

## 7. Learning research basis

Two 2026 syntheses provide useful starting evidence:

- A preregistered meta-analysis of 22 GenAI-in-mathematics studies (46 samples, N=5,232) reported an average effect of g=0.534. Effects varied with how GenAI was integrated into instruction.
- A 34-study meta-analysis of physically embodied educational robots (N=3,665) found r=0.31 overall; effects were stronger when robots played active collaborative/teaching roles, programmable kits were used and programmes lasted more than four weeks.

These results support testing a sustained, active, scaffolded learning design, but do not establish the efficacy of Praxis. A crucial counterpoint is the October 2026 randomized university trial (2,379 undergraduates, 30 instructors) in which course-integrated tutor access reduced final grades by 0.37 SD and LMS participation by 0.90 SD in the studied setting. That study is not universal evidence against AI tutoring; it shows why support displacement, independent effort, accessibility, subgroup effects and delayed transfer must be measured as first-class outcomes. Praxis #5461 owns the non-harm protocol. Source: [Liu et al., IZA Discussion Paper 18989](https://iza.org/publications/dp/18989/the-effects-of-course-integrated-ai-tutoring-on-student-performance-and-engagement-a-randomized-university-trial).

## 8. Delivery sequence and gates

| Phase | Runner-independent output | Gate |
|---|---|---|
| P0: contract | task packet, model/requirement map, benchmark/licence registry, learner protocol, adapter inventory | independent review; no sealed answer leakage |
| P1: digital pilot | analytic reference problems, STEP tasks, solver baselines, source-only agent views | CEB isolation and custody qualification before untrusted execution |
| P2: tool loop | matched model/solver ablations, uncertainty and experiment-selection comparison | exact-version replayable run evidence |
| P3: physical pilot | guarded fixture, metrology pilot, pre-registered confirmatory plan | professional safety review and calibrated reference chain |
| P4: education pilot | multi-week classroom/maker-space protocol, delayed transfer assessment | ethics/guardian/educator approval and privacy protections |
| P5: commons | opt-in invention records, licenses, provenance and replication instructions | rights review, privacy review and independent reproducibility |

## 9. Runner-independent next tasks

1. Prepare a source-visible bridge/truss challenge packet with requirements and a learner-facing project brief; keep evaluator labels and gold criteria out of the agent view.
2. Map the model and evidence fields to PhysicsIR, SE/ETK, CEB and Symtropy types; list only genuine integration gaps.
3. Pin external benchmark versions, licences, access mode, evaluator revision and comparable task cohorts.
4. Define adapter contracts for STEP/BREP, one structural solver and one physical measurement instrument.
5. Preregister the Praxis learning protocol and independent transfer measure.
6. Make a small Mycelix contribution/provenance profile proposal; preserve private learner work unless sharing is explicitly chosen.

No repository-controlled build, test or candidate-code execution is authorized until CEB execution isolation #10064 is qualified. No physical actuation or testing is authorized by this document.

## 10. Claim discipline

Use explicit status labels: **PROPOSED**, **DESIGN_ONLY**, **SIMULATED**, **MEASURED**, **INDEPENDENTLY_REPRODUCED**, **QUALIFIED**. A result may advance only when its required evidence exists. Formal proof, solver convergence, receipt validity, model confidence and human approval are different claims and must stay different.

---

## References and external comparators

- PLE leaderboard: https://www.mercor.com/apex/partner-benchmarks/ple-bench-leaderboard/
- PLE paper: https://arxiv.org/abs/2610.30537
- EngiBench ACL Findings 2026: https://aclanthology.org/2026.findings-acl.1810/
- EngDesign: https://github.com/AGI4Engineering/EngDesign
- CADGenBench: https://github.com/huggingface/cadgenbench/
- CADGenBench metrics: https://github.com/huggingface/cadgenbench/blob/main/docs/metrics.md
- CADBench: https://arxiv.org/abs/2605.10873
- Autodesk Fusion October 2026: https://www.autodesk.com/products/fusion-360/blog/october-2026-major-product-update-whats-new/
- Autodesk AutoTimeline research (2026-11-03): https://www.research.autodesk.com/blog/rebuilding-design-history-with-autotimeline/
- NVIDIA PhysicsNeMo: https://docs.nvidia.com/physicsnemo/latest/overview.html
- SimScale State of Engineering AI 2026: https://explore.simscale.com/hubfs/resources/reports/state-of-engineering-ai-2026.pdf
- GenAI mathematics meta-analysis: https://www.mdpi.com/2227-7102/16/1/140
- Educational robotics meta-analysis: https://www.sciencedirect.com/science/article/pii/S000169182600377X
