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

Mycelix's current README reports Praxis as live but also says its flagship source does not compile at the current HEAD and that the live site is served from an older pinned IPFS snapshot; treat this as a repository self-report, not an independently verified build result. The source/artifact/deployment reconciliation is now tracked separately in [Mycelix #5795](https://github.com/Luminous-Dynamics/mycelix/issues/5795) and should precede new live Praxis product surfaces. The advisor boundary in Mycelix #5472 is the integration path; do not create a competing tutor, learner-state model or credential authority. The present monorepo Praxis scaffold must not be represented as a finished engineering-learning product.

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

Start with the [paper-first Bridge / Truss Design Challenge v0.1](INVENTOR_LOOP_001_BRIDGE_TRUSS_LEARNER_PACKET_V0_1.md) and its public [instructor analytical key](INVENTOR_LOOP_001_BRIDGE_TRUSS_INSTRUCTOR_KEY_V0_1.md). The initial deliverable is a learner's explained hand calculation, comparison of at least two concepts, prediction, and reflection—not a fabricated bridge or an executed solver run.

The paper exercise is numerically specified:
- A=(0,0) m, B=(4,0) m, C=(2,3) m.
- Two ideal pin-jointed axial members with E=200 GPa and A=10 mm².
- Both base nodes restrained in x and y; apex load (0,−100) N; self-weight excluded.
- The instructor key independently derives the symmetric stiffness, displacement, member compression, stresses/strains, support reactions, equilibrium and energy/work consistency.

**Digital phase — blocked pending owner review.** Its intended target is Symtropy's open PHYS-TRUSS PR series (#2361/#2363 and follow-ons). Core PR #2363 is still open/unmerged, and the follow-on PRs are open/draft. Until the Symtropy owner confirms an exact reviewed head, supported request interface and qualification state, do not call that branch or claim digital solver integration. Once approved, consume the existing solver/oracles/receipts rather than implementing a second truss kernel, analytic oracle or receipt format. These PR-described tests are mathematical/software verification of a limited static model, not experimental physical validation.

**Physical phase — separate future study, not authorized by this programme.** Any prototype would need competent adult design/review, a guarded non-human-load-bearing fixture, characterized materials and joints, calibrated force/displacement measurement, predeclared stopping criteria, raw data retention, and the existing Symtropy physical-trial/metrology review. The paper task is not a test-load specification for a real structure.

The eventual learning loop can extend from:
- predicting tension/compression and displacement before seeing the answer;
- explaining load paths and model assumptions;
- comparing different topologies under matched requirements;
- requesting the most decision-relevant next observation;
- revising a design after evidence;
- transferring the reasoning to a structurally different task without step-by-step assistance.

The task is not a real bridge design and cannot be used for people, buildings, public infrastructure or other safety-critical applications. A separate physical pilot must determine materials, load range, measurement reference, calibration, sample size and stopping rules before any confirmatory trial.
## 5. Benchmark programme

Use existing external benchmarks as *separate diagnostic instruments*, not one blended leaderboard:

- **Physical-Engineering's Last Exam (PLE):** long-horizon, multi-file professional engineering deliverables and hard constraints. Mercor's live leaderboard reports 48.3% on its 56-task hidden set; the PLE paper abstract separately reports a 50.6% strongest configuration. These are distinct published evaluation summaries and must not be treated as interchangeable. Google Cloud / Mercor reports that only 41.3% of deliverables in the co-authored task cohort cleared every hard safety and feasibility check. The leading run averages about 27 million tokens and 63 minutes per task, making cost and latency essential comparison dimensions. These scores are comparable only under the exact official evaluator conditions. The public 29-task development leaderboard reports three-run mean results, while the competition-entry dev score grades committed artifacts and live-checks the harness on a sample; hidden competition scores average three harness runs. Do not compare a committed-artifact dev score with the three-run dev leaderboard mean without stating the scoring pathway. The public dev task set is access-gated under CC-BY-4.0, while the official runner repository is Apache-2.0.
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
- Educational pilot target: demonstrate a preregistered, domain-specific superiority or non-inferiority result on delayed unaided transfer against an active project-based comparison, while showing no meaningful displacement of protected human instruction or independent work. Set the minimum meaningful effect/margin and sample size from pilot variance and educator review; do not impose one universal effect-size target across contexts.
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
2. **Exploit model and solver complementarity.** Benchmark strong frontier-model-only and model-plus-commercial-tool baselines first. Google's PLE launch analysis reports that model choice produced up to a 50-point score spread across tested configurations, while harness choice for the same model produced a median 3-point spread. This is an important warning: do not attribute a stronger foundation model's gain to Symthaea orchestration. Use matched-model ablations. Symthaea's potential value is context continuity, contradiction detection, typed scientific models, uncertainty-aware reasoning and selecting the next useful observation; Symtropy should call suitable specialist solvers rather than recreate their mature numerics.
3. **Optimize all-gates success, not partial-credit theatre.** PLE's 91% average individual-constraint adherence but 41% all-hard-check clearance is a warning: near-correct is still not acceptable in physical engineering. Report hard-gate pass probability, failure severity and justified abstention before cost/weight optimization.
4. **Make experiment choice a first-class capability.** Compare random, fixed-sequence, expert-selected and model-selected experiments under an identical budget. Score information gain only when it changes a decision or reduces uncertainty in a decision-relevant quantity.
5. **Make the human a co-inventor, not a prompt operator.** Demonstrate both artifact quality and unaided transfer. Offer support-mode choice and progressive hints; preserve the ability to work offline, ask a teacher/peer, decline AI or keep a project private.
6. **Publish failure as well as success.** Release permitted fixtures, exact versions, input/output digests, mutation controls, exclusions, cost and failure analyses. Independent reproduction is part of the product, not a marketing afterthought.

A result should be called better than SOTA only after the incumbent baseline is replicated on the same held-out tasks, exact grader, tools and resource budget, with uncertainty intervals and independent review. Otherwise report it as a promising local result, not a leaderboard win.

Our differentiation should be **decision quality per independently accepted artifact**, with a human who learns, and an evidence record that an external reviewer can inspect.

## 7. Learning research basis and benchmark strategy

The current evidence does not support a universal claim that AI tutoring helps or harms. Four recent studies point to different mechanisms and contexts:

- **StudentBench (September 2026; adult GRE):** 2,383 participants were assigned to AI tutoring, expert human tutoring or no tutoring. The authors report statistical equivalence between pooled AI and human tutoring on the studied GRE gains (p=.015), with substantial variation among tutor configurations and separate measurements for lesson plans, practice problems, pedagogy, cost and engagement. This is a useful adult-learning outcome comparator, not direct evidence for school-age learners or engineering education. [Paper](https://arxiv.org/abs/2609.28470) · [Reproduction code/data](https://github.com/Handshake-AI-Research/studentbench).
- **Khanmigo two-year school trial (NBER 35620, August 2026):** a cluster-randomized trial across 18 Tennessee middle schools reports about 0.06–0.08 SD math-achievement improvement over a school year, similar to Khan Academy practice without AI. AI uptake was low: the median student used it on only about one third of practice days, and on a minority of mistake sessions. This shows that access does not equal meaningful use. [NBER paper](https://www.nber.org/papers/w35620).
- **NUMI mastery-based math experiment (NBER 35621, August 2026):** a randomized experiment with more than 6,000 middle-school students found that AI support improved correctness on the next attempt after mistakes, but students progressed more slowly and attempted fewer questions. Delayed gains were mixed and concentrated in some practiced material; mastery progression alone did not guarantee transfer. This supports testing well-timed, mistake-responsive scaffolding rather than default answer provision. [NBER paper](https://www.nber.org/papers/w35621).
- **Course-integrated tutor access university trial (IZA Discussion Paper 18989, October 2026):** 2,379 undergraduates across 30 instructors saw reduced final grades (−0.37 SD) and LMS participation (−0.90 SD) in the studied access condition, with potentially greater losses among first-generation students. This is a setting-specific warning about displacement of existing learning activity, not a universal conclusion about AI tutors. [IZA paper](https://iza.org/publications/dp/18989/the-effects-of-course-integrated-ai-tutoring-on-student-performance-and-engagement-a-randomized-university-trial).

LongTutor (ACL 2026) and MMTutorBench (ACL 2026) provide additional diagnostic benchmarks: LongTutor evaluates historical evidence acquisition, knowledge-state diagnosis and adaptive teaching action, and identifies a gap between acquiring learner history and using it to choose effective teaching actions. MMTutorBench tests step-level multimodal mathematical tutoring across 770 problems and six rubric dimensions. Neither substitutes for measuring actual human learning gains. [LongTutor](https://aclanthology.org/2026.acl-long.1371/) · [MMTutorBench](https://aclanthology.org/2026.acl-long.1068/).

### Praxis strategy implied by the evidence

Praxis should not optimize "AI answer correctness" alone. It should choose a support mode and time of intervention:

1. Preserve learner-chosen goals and normal educator/peer/offline resources.
2. Where pedagogically appropriate, invite an unaided attempt or prediction first; do not block accessibility accommodations or learner requests.
3. Trigger help at meaningful decision points (e.g. after a misconception or error), using a small hint or question before a full explanation where appropriate.
4. Ask the learner to reconstruct the reasoning, practise a structurally different case, and later retrieve it without assistance.
5. Fade scaffolding as independent capability improves; allow the learner to decline AI or request a human.
6. Measure whether this policy actually improves learning and does not displace more effective teaching or practice.

Reuse Mycelix's existing planner/BKT/SRS and assessment contracts (#5286, #5403, #5455), support router (#5467), non-harm/displacement protocol (#5461), bounded Symthaea advisory boundary (#5472) and curriculum research (#5446). Do not create a second learner-state model, tutor, assessment authority or credential path.

A controlled Praxis pilot should preregister:
- independent, unaided transfer on unfamiliar tasks as the primary outcome;
- delayed retention at a domain-appropriate interval;
- attempts, error recovery, task progression and time-on-task;
- AI availability, invocation, dose and assistance type separately;
- educator/peer engagement and displaced course activity;
- accessibility and subgroup effects where consent, study design and sample size permit;
- attrition, missing follow-ups and adverse outcomes.

Use active comparators and randomization where feasible and ethical. Choose a domain-specific minimum meaningful effect or non-inferiority margin with educators from pilot variance and power calculations; **do not impose a universal effect-size target such as +0.30 SD across unrelated ages, topics and interventions.** A study without enough power cannot claim equivalence. Keep adult GRE, middle-school maths and engineering-project cohorts separate. Synthetic benchmark success is not evidence of real learning efficacy.

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
- Google Cloud / Mercor PLE launch report (3 Nov 2026; model vs. harness spread and 41.3% co-authored all-hard-check clearance): https://cloud.google.com/blog/products/ai-machine-learning/introducing-physical-engineerings-last-exam-for-agents/
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


---

## 2026-11-05 research update: external baselines and parametric-history frontier

This addendum records a dated snapshot. It does not retroactively change the scoring pathway, reproduce an external benchmark, or qualify a Luminous system.

### PLE hidden leaderboard snapshot

The official Mercor page retrieved 2026-11-05 reports means on its 56-task hidden test set:

| Model | Native harness | Mean | Displayed ± |
|---|---|---:|---:|
| Claude Opus 5.5 | Claude Code | 48.3% | ±1.4% |
| GPT-6 Astra | Codex | 45.8% | ±0.6% |
| Claude Sonnet 5.5 | Claude Code | 43.5% | ±1.4% |
| Gemini 4 Argon | Antigravity | 40.6% | ±1.3% |
| Claude Opus 5 | Claude Code | 40.0% | ±1.6% |

The stated protocol is each model at highest reasoning effort in its native harness, three runs per task, isolated container without internet, and six hours per task. The leaderboard page does not define the statistical meaning of its displayed ± values; preserve them as displayed, not as confidence intervals or standard deviations. This is hidden-test context only—not a Luminous baseline, not the public 29-task dev mean, and not the separate committed-artifact competition dev score. See the [dated leaderboard](https://www.mercor.com/apex/partner-benchmarks/ple-bench-leaderboard/) and [metadata registry snapshot PR #10458](https://github.com/Luminous-Dynamics/symthaea/pull/10458).

### Parametric CAD Bench V3

The publisher's 2026-09-24 V3 snapshot reports 100 tasks, with overall mean continuous reward and failed/unscored outputs counted as zero. Its top rows include Opus 5.5 + Claude Code at 61.03% (95% CI ±7.53), Opus 5.5 + mini-swe-agent at 57.96% (±7.82), and GPT-6 Astra + Codex at 56.87% (±6.85). The page states that the first four confidence intervals overlap. In a paired same-model comparison, Claude Code leads mini-swe-agent by 3.08 points, but the paired 95% interval ranges from −1.15 to +7.31; this is not decisive evidence of an agent ranking. See the [publisher's V3 analysis](https://cadbench.ai/news/cad-bench-v3-opus-5-5).

These values are not STEP/BREP geometric-equivalence rates and must not be pooled with PLE, CADGenBench or CADBench. Benchmark task, scorer, model/harness and cost definitions differ.

### AutoTimeline and the next CAD paradigm

Autodesk Research's [AutoTimeline work (2026-11-03)](https://www.research.autodesk.com/blog/rebuilding-design-history-with-autotimeline/) reconstructs multiple candidate editable feature timelines from a history-less solid, rebuilds each candidate and checks its geometry against the source. The key research implication is that an original feature sequence is generally not identifiable from final geometry alone.

The next frontier is therefore not exact original-history recovery. It is **edit-preserving parametric equivalence**: can a candidate history rebuild within a declared geometry tolerance, expose meaningful editable parameters, preserve required interfaces under specified edits, and represent ambiguity honestly? This programme now tracks that question in [CAD-HISTORY-001 #10510](https://github.com/Luminous-Dynamics/symthaea/issues/10510), composing existing [FAB-GEOM-001 #4927](https://github.com/Luminous-Dynamics/symthaea/issues/4927) rather than adding a new geometry authority or CAD kernel.

### Numerical assurance feedback from PHYS-TRUSS

An independent review of Symtropy's latest advertised solver head found a scale-invariance defect in force/moment residual normalization: fixed 1 N and 1 N·m denominator floors can make a 100% imbalance appear numerically tiny for sub-unit problems. The reviewer independently recomputed the published load-case displacement vectors (maximum discrepancy below (5.5\times10^{-20}\,m)); this checks those fixture values only, not Rust execution. The issue and inline review are tracked in [Symtropy #3676](https://github.com/Luminous-Dynamics/symtropy/issues/3676) and [solver PR #2363](https://github.com/Luminous-Dynamics/symtropy/pull/2363). This illustrates why benchmark gates themselves need dimensional and scale-invariance audits.



### CAD-HISTORY-001 synthetic fixture advancement (2026-11-05)

The initial CAD reconstruction benchmark has now advanced from a taxonomy into a synthetic contract corpus in [draft PR #10529](https://github.com/Luminous-Dynamics/symthaea/pull/10529), linked from [CAD-HISTORY-001 #10510](https://github.com/Luminous-Dynamics/symthaea/issues/10510). It specifies six positive editable-part cases, four negative/abstention controls and ten metamorphic probes. The evaluator accepts alternative native feature histories when they satisfy the observable geometry and semantic-edit contract; it does not reward guessing a private original history.

The corpus is deliberately an extension to, not a replacement for, Parametric CAD Bench V3. The V3 paper (2026-11-02) uses pinned FreeCAD and a separate verifier for native CAD artifacts, but notes that geometry/specification checks do not establish arbitrary future editability or manufacturing readiness. Its 2026-09-24 leaderboard snapshot shows a significant workflow difference: Opus 5.5 / Claude Code 84.66% image-to-CAD versus 39.69% create-and-edit; GPT-6 Astra / Codex 69.70% versus 44.37%. The proposed extension tests held-out semantic interventions and preservation of unrelated invariants after reconstruction. It has not yet generated CAD fixtures or run a CAD kernel.

The PHYS-TRUSS follow-up [Symtropy #3676](https://github.com/Luminous-Dynamics/symtropy/issues/3676) now includes explicit dimensionless force, moment and normwise free-DOF residual formulas, zero-denominator disposition, and scale-metamorphic test requirements. The original inline review remains unresolved; the formulas are implementation guidance, not a code fix or numerical qualification.

