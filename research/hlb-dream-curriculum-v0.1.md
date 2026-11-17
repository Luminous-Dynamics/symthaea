# HLB Dream Curriculum v0.1
## An evidence-directed, cross-domain dream-to-discovery programme

**Status:** research contract and prospective preregistration draft  
**Date:** 2026-11-17  
**Owner:** Symthaea research orchestration; composed with Symtropy and Mycelix  
**Claim boundary:** proposed protocol; no benchmark result or implementation completion is asserted here

---

## 1. Executive thesis

The research objective is not to make Symthaea generate the most visually convincing imagined worlds. It is to determine whether HLB-directed imagination can produce **more independently verified, transferable capability per unit of experimental budget** than matched direct-learning, world-model, random-search and classical experiment-design baselines.

The proposed loop is:

> identify a capability gap → state competing hypotheses → design a discriminating world/intervention → execute through Symtropy → seal and compare predictions with independently held outcomes → preserve provenance through Mycelix → qualify a scoped learning candidate → test transfer and retention.

This is a stronger proposition than “dreaming improves prediction.” A dream is useful only when it changes what the system can reliably do or know, under a declared budget, and the gain survives an independent holdout.

### 1.1 The primary falsifiable question

Given the same starting model, allowed observations, task families, experiment budget, compute budget and safety/authority boundary:

**Does an HLB-guided dream curriculum produce a statistically supported improvement in held-out decision quality, fault diagnosis, recovery or transferable skill acquisition—without unacceptable regressions in protected capabilities—compared with the strongest matched non-HLB curriculum?**

Secondary questions:

1. Does HLB choose more informative experiments than random, uncertainty-only, novelty-only or expected-improvement schedulers?
2. Do Symtropy-executed worlds provide more useful counterfactual evidence than heuristic action perturbations?
3. Does model-based imagination add decision value after inference latency, energy, memory and rollout cost are counted?
4. Does a learned mechanism transfer to a different generator family or domain, or is the apparent gain merely simulator/task memorization?
5. Can the system detect when the dream world/model is inadequate and abstain instead of manufacturing a confident answer?

### 1.2 What success is not

A visually coherent dream, a high HLB activation, an increase in a Phi proxy, a lower training loss, or a self-authored “wisdom” record is not evidence of improved intelligence or capability.

Do not collapse:
- environment appearance into physical validity;
- one-step prediction into long-horizon causal adequacy;
- prediction accuracy into decision utility;
- repeated seeds into independent replication;
- simulation success into real-world competence;
- signed provenance into scientific truth;
- confidence into calibration;
- a successful branch into a complete experiment.

---

## 2. Source audit: what exists today

### 2.1 Symthaea dream engine

The current symthaea-dream crate is a real but limited counterfactual-dreaming prototype:
- records high-surprise state/action/outcome events;
- chooses the most surprising remembered event for a dream cycle;
- creates alternatives by perturbing the original action;
- stores action-fingerprint/state/outcome associations in a small transition-memory structure;
- estimates an outcome with a heuristic action prediction, optionally blended with a matching remembered outcome;
- labels alternatives “wisdom” when a vector-magnitude Phi proxy or variance-based EI proxy improves;
- includes a motor-trajectory dreamable action representation.

Important limitations:
- the present transition memory is not an executable Symtropy world model;
- the current dream cycle does not ask Symtropy to run physical branches;
- the present Phi estimate is mean absolute vector magnitude, not an IIT Phi calculation;
- the present EI estimate is an inverse-variance-like proxy, not a qualified causal Effective Information calculation;
- confidence is derived from those heuristic scores, not calibrated against held-out outcomes;
- no sealed independent evaluator, hidden challenge set, cross-generator transfer or protected-skill retention gate is established by this crate.

This implementation is a useful seed for an adapter and experiment harness. It is not yet the dream-to-discovery system proposed here.

### 2.2 Existing owners to compose rather than duplicate

**Symthaea**
- [#11426 — HLB-VSSI-001](https://github.com/Luminous-Dynamics/symthaea/issues/11426): unfamiliar-fault investigation and independently verified model repair.
- [#9010 — CMAQ-004](https://github.com/Luminous-Dynamics/symthaea/issues/9010): world-model value-add versus matched direct policies.
- [#12723 — HUM-HLB-ABLATION-001](https://github.com/Luminous-Dynamics/symthaea/issues/12723): HLB incremental value for humanoid cognition.
- [#12368 — HLB-DYNAMICS-ARBITRATION-001](https://github.com/Luminous-Dynamics/symthaea/issues/12368): HDC-LTC update laws and active model discrimination.
- [#10844 — SYM-RSI-STATE-ROUTING-001](https://github.com/Luminous-Dynamics/symthaea/issues/10844): scoped research-state routing.
- Also compose existing multi-timescale metareasoning, active-sensing, recurrent snapshot/restore and protected-skill owners. Do not create parallel owners for them.

**Symtropy**
- [#4687 — ACTION-CONDITIONED-DECISION-UTILITY-001](https://github.com/Luminous-Dynamics/symtropy/issues/4687): world models scored by action value rather than video fidelity.
- [#1387 — WORLD-START-00](https://github.com/Luminous-Dynamics/symtropy/issues/1387): scenario/world bootstrap and exact starting identity.
- [#1403 — WORLD-VALIDATION-00](https://github.com/Luminous-Dynamics/symtropy/issues/1403): staged scenario validation.
- [#4581 — deadline-aware anytime imagination allocator](https://github.com/Luminous-Dynamics/symtropy/pull/4581): candidate rollout allocation mechanism. Its PR description explicitly records that tests were not yet run in its originating environment and CI was pending; do not treat it as qualified.

**Mycelix**
- [#5900 — reproducible algorithm-experiment attestations](https://github.com/Luminous-Dynamics/mycelix/issues/5900): experiment lineage and independent reproduction references.
- [#3563 — PRAX-AI-001](https://github.com/Luminous-Dynamics/mycelix/issues/3563): AI observation, model/runtime provenance and prospective evaluation.
- [#5709 — CIVOS-EVIDENCE-001](https://github.com/Luminous-Dynamics/mycelix/issues/5709): rights-aware claims and provenance.

No second simulator, world authority, generic memory engine, evidence ledger, verifier or authorization engine is proposed.

---

## 3. Paradigm shifts

### Shift 1 — From dream quality to epistemic value per resource

Optimize for the reduction in decision-relevant uncertainty or regret per unit of compute, elapsed time, energy, experiment cost and human review—not number of generated worlds.

A candidate dream is valuable when it distinguishes between plausible hypotheses or changes a decision. A hundred beautiful but redundant scenes can be less useful than one carefully chosen intervention.

### Shift 2 — From a single world model to a portfolio of executable worlds

Use the cheapest model that can answer the question, and escalate fidelity when uncertainty or consequence warrants it:

1. **Analytic toy world:** known equations and exact outcomes; tests logic and information accounting.
2. **Reduced-order model:** cheap screening over broad parameter ranges.
3. **Symtropy physical/agent simulation:** explicit constraints, interactions, branch execution and replay.
4. **Hardware-in-the-loop:** only when the interface and test profile are qualified.
5. **Supervised physical evaluation:** governed by the existing robot safety/actuation authority, never by the dream engine.

Learned generative models can provide observations, appearance priors, scene proposals or candidate trajectories. They must not silently replace the executable physics or evaluator.

### Shift 3 — From passive replay to active experiment design

Dreaming should ask: “Which experiment most sharply separates these hypotheses?” The selector should compare:
- expected information gain;
- expected reduction in action regret;
- expected risk reduction;
- novelty/coverage contribution;
- compute, simulation, energy and wall-clock cost;
- probability that the experiment is valid and executable.

Use the existing value-of-information and metareasoning owners. An information-gain estimate is a selection heuristic until its realized value is measured prospectively.

### Shift 4 — From one learner to a learner–world–adversary–evaluator system

The learner proposes a scenario. A separately configured scenario adversary tries to expose unsupported assumptions. The simulator executes. The independent evaluator holds the hidden truth and scores the result. The learner never controls the evaluator’s answer key or promotion gate.

This reduces self-confirmation and makes “dreaming” a falsifiable scientific process rather than a closed self-approval loop.

### Shift 5 — From task memorization to mechanism transfer

Hold out complete generator families, causal mechanisms, embodiments, sensor profiles, network topologies and task compositions—not merely random seeds.

A skill transfers only when its causal preconditions and limits are carried with it. Similar HDC representations or shared words are not sufficient evidence that two domains share a mechanism.

### Shift 6 — From silent continual updates to reversible, typed knowledge promotion

Separate:
- observation and episode capture;
- scoped memory/retrieval changes;
- world-model state updates;
- candidate skill/policy changes;
- core parameter updates;
- changes to action authority.

Each class has a different promotion bar. Most online learning should update a bounded episode/semantic/procedural memory or candidate world model first. Parameter and policy changes remain candidate artifacts until independently evaluated. Authority changes are outside this research loop.

### Shift 7 — From sim-to-real as a one-way pipeline to a qualified reality/model feedback loop

Physical observations can challenge the simulation model; the corrected model can generate new test worlds; newly learned skills must still pass held-out simulation and supervised physical evaluation. Preserve the distinction between observed reality, estimated state, simulated outcome and counterfactual prediction.

### Shift 8 — From a monolithic “brain” claim to a compositional architecture portfolio

HLB must earn its place on matched tasks against MLP, GRU, CfC, transformer/VLA, compact latent world models, model-free policies and classical experiment design. A hybrid may win even when an individual HLB component does not. Publish ablations, including null and negative results.

---

## 4. Proposed cross-system contracts

This section specifies the minimum handoff. It does not prescribe a duplicate storage or execution subsystem.

### 4.1 DreamProposalV1 — produced by Symthaea

Each proposal binds:

- schema/version and canonical proposal digest;
- research question and capability-gap identifier;
- scoped memory/evidence references, status and provenance;
- competing hypotheses, predicted discriminating observations and expected outcomes;
- target domain, task family and generator family;
- exact scenario/world profile and version requested from Symtropy;
- intervention set and allowed action/observation interface;
- seed derivation identity and exogenous-randomness policy;
- budget: rollouts, transitions, simulation time, wall time, memory, accelerator time/energy and any human review;
- proposed stopping rule and maximum horizon;
- expected information/decision value and its uncertainty;
- permitted tools and capability scopes;
- explicit non-goals, unsupported assumptions and abstention conditions.

A proposal cannot include hidden evaluator labels, future outcomes, privileged simulator state or sealed challenge-set content.

### 4.2 ScenarioAdmissionV1 — owned by Symtropy

Before execution, Symtropy returns one of:
- admitted;
- rejected with reason;
- unsupported;
- not executed / resource bound.

Admission binds the exact scenario package, simulator/physics profile, generator revision, initial-state digest, environment assumptions, timestep, solver/backend, task/actor profiles, seed policy and executable capability set.

A structurally valid world is not automatically physically valid, solvable, realistic or representative. Compose WORLD-VALIDATION-00’s validation ladder; do not create a rival validator.

### 4.3 PredictionSealV1 — produced before reveal

Before the evaluator reveals outcomes, the model seals:
- model/checkpoint/runtime identity;
- HLB state and memory snapshot digest;
- observation history digest;
- predicted outcome distribution and horizon;
- action ranking or selected action;
- uncertainty distribution and abstention disposition;
- forecast time and compute cost.

Prediction sealing is not evidence that the prediction is correct. It prevents post-outcome rewriting.

### 4.4 DreamExecutionReceiptV1 — produced by Symtropy

For every expected branch, retain:
- proposal and scenario-admission identities;
- parent snapshot and intervention/action identity;
- exact simulation/model/build/profile identities;
- branch identity and exogenous event-stream identity;
- actor policy, observation and action lineage;
- initial and final state commitments;
- per-step or checkpointed trajectory artifact references;
- completion, failure, incomplete, unsupported or resource-bound status;
- replay disposition and reason;
- metric values with definitions, units and evaluator/profile identity.

Branch identity must be unique. Expected-but-unexecuted branches remain in the denominator. Do not turn timeout, missing output or unsupported physics into “infeasible” or “failed task.”

### 4.5 IndependentEvaluationV1 — produced by an evaluator outside the learner

The evaluator binds:
- sealed prediction;
- exact expected-branch manifest;
- hidden reference outcomes or analytic known truth;
- outcome and metric definitions;
- calibration and decision-value calculations;
- control/negative-control outcomes;
- contamination and privilege checks;
- completeness and uncertainty;
- evaluator implementation/version and independence profile.

A simulator oracle is for evaluation only. Its hidden state must not leak into the policy, HLB memory or scenario selector.

### 4.6 ExperimentAttestationV1 — Mycelix-compatible export

Reuse #5900, #3563 and #5709. The experiment record should reference:
- source revision and binary/model/checkpoint digests;
- proposal, scenario, protocol and benchmark manifest digests;
- seed and resource profile;
- input/output/trajectory artifact digests and content-addressed URIs;
- independent evaluator/checker identity and verdict;
- data license, rights, consent and usage scope where relevant;
- AI-generated/AI-tuned component disclosure;
- replication, correction, dispute, retraction and supersession links.

Mycelix attests who published which evidence references and validation claims. It does not make those claims scientifically true. Store large trajectories, videos, checkpoints and raw data in appropriate content-addressed object storage, not as bulk Holochain DHT payloads.

---

## 5. Dream-world curriculum

World classes are selected by an identified uncertainty or capability gap. A proposal should state why the chosen class is informative.

| Dream class | Purpose | Example |
|---|---|---|
| Replay / reconstruction | Reproduce an experience and isolate contributing conditions | Re-run a failed grasp from the exact initial state |
| Parameter counterfactual | Change one declared physical or operational parameter | Alter friction, mass, latency or packet loss |
| Structural counterfactual | Change a causal relationship or topology | Remove a dependency, change network topology, swap contact geometry |
| Regime-shift | Test whether the model detects changed dynamics | New material, changed payload, new fault distribution |
| Adversarial | Find weaknesses and shortcut policies | Misleading observation, decoy tool, partition, common-cause fault |
| OOD / open-world | Test abstention and model adequacy | Unsupported contact model or unseen process family |
| Transfer | Test a mechanism in a different generator family | Transfer recovery logic from manipulator contact to a new tool/task family |
| Self-architecture | Compare candidate cognitive/learning components | HDC-LTC versus CfC/GRU/MLP under matched dynamics |

The initial release should not attempt to generate arbitrary societies, planets, physics or biology from an unconstrained prompt. Begin with versioned scenario families and known executable semantics. Broader creative world generation can be an upstream proposal interface; Symtropy still validates what is actually executable.

### 5.1 First two domains

**Domain A — robotics / contact-rich manipulation**
- object/tool selection, grasp, insertion, pushing, contact transitions and recovery;
- friction, payload, geometry, sensor occlusion, timing, compliance and tool-decoy variations;
- unseen object instances, task compositions and at least one held-out generator family;
- compose Symthaea’s HUM-HLB ablation and Symtropy’s action-conditioned decision-utility protocol.

**Domain B — distributed systems / Mycelix resilience**
- node churn, network partitions, Byzantine behavior, stale identities, common-cause failures and recovery;
- distinguish model-visible observations from evaluator-only ground truth;
- measure diagnosis, intervention ranking, time-to-recovery, false positive/negative decisions and resource use;
- keep all scenarios sandboxed; no live production-network mutation.

Do not pool robotics and distributed-systems scores into one average. Cross-domain transfer is a separate experiment with an explicit common-mechanism hypothesis.

---

## 6. Benchmark: Dream-to-Discovery (D2D) v0.1

### 6.1 Benchmark objective

D2D measures whether an agent can **choose informative executable experiments, learn from the results, and transfer a capability**—not merely whether it can predict a fixed dataset or generate a plausible scene.

The first release should include:
- analytic toy tasks with known ground truth;
- one bounded Symtropy robotics scenario family;
- one bounded distributed-systems scenario family;
- train/dev/hidden evaluation split by generator family and mechanism;
- a prospective campaign budget;
- immutable expected-branch manifests;
- fresh-process replay fixtures;
- a separate human review layer for interpretability and usefulness.

### 6.2 Matched experimental arms

Run at least these arms with the same initial policy/model, permitted data, task/observation interface and inclusive budget:

A. **Direct policy:** no learned world model or dream curriculum.  
B. **Direct policy + same memory:** isolates memory from imagination.  
C. **Current dream prototype:** action perturbation and heuristic outcome estimator, reported as a prototype baseline only.  
D. **Random curriculum:** randomly selects eligible scenarios under the same budget.  
E. **Classical experiment design:** tuned Bayesian optimization / expected improvement / value-of-information baseline where task assumptions support it.  
F. **Conventional learned world model:** compact action-conditioned latent transition model plus planner.  
G. **HLB predictor without Symtropy dream execution:** isolates representation/predictive contribution.  
H. **HLB + Symtropy + active dream selection:** full proposed loop.  
I. **Oracle evaluator:** hidden upper bound only; never eligible as a learner arm.

Add frontier video-world-model or VLA baselines only where the released code, weights, license, action interface, compute profile and task protocol can be reproduced. Record incompatibilities rather than comparing non-equivalent scores.

### 6.3 Primary metrics — report separately

**Discovery and decision value**
- validated discoveries per fixed experiment/compute budget;
- expected versus realized information gain;
- action-ranking regret and top-k action agreement;
- closed-loop task success and lower-tail success;
- risk-sensitive regret and false-safe decisions;
- reduction in human interventions or recovery burden.

**World-model quality**
- action-conditioned transition error by state variable and horizon;
- intervention sensitivity under paired counterfactual branches;
- rollout-validity horizon and error-versus-horizon curve;
- calibration: Brier score, log score/CRPS where suitable, reliability curves, coverage and risk-coverage;
- explicit OOD detection and correct abstention.

**Learning and transfer**
- held-out generator-family success;
- transfer to unseen task compositions, tool/object instances, embodiments or topology;
- protected-skill retention and catastrophic-forgetting measure;
- repeated-failure recurrence after memory consolidation;
- false memory promotion and poisoned-memory rejection;
- adaptation sample efficiency and recovery time after regime shift.

**Cost and operational behavior**
- total simulator transitions and rollouts;
- training and inference FLOPs/GPU-hours where measurable;
- CPU/GPU/accelerator energy where instrumented;
- peak memory and storage;
- p50/p95/p99 decision latency;
- wall-clock time and human review/correction burden;
- valid discovery per unit of compute, energy, experiment cost and human time.

**Assurance and reproducibility**
- expected/observed branch-denominator completeness;
- prediction-before-reveal compliance;
- hidden-state / evaluator / teacher privilege leaks;
- exact replay and fresh-process reconstruction;
- source/model/data/seed lineage completeness;
- negative-control detection and independence of replication.

Do not invent a universal weighted “dream score.” Publish a Pareto frontier and metric-level results. If a composite score is later proposed, preregister its rationale, sensitivity and failure modes, and retain the underlying measurements.

### 6.4 Prospective target gates

These are targets to test, not achieved results or universal SOTA claims:

1. **Completeness:** 100% of declared branches represented as completed, failed, incomplete, unsupported or resource-bound; no silent denominator loss.
2. **Privilege integrity:** zero accepted hidden evaluator, future-outcome or oracle-state leaks; all planted leakage negative controls detected.
3. **Decision value:** the full HLB+Symtropy arm should achieve either at least a 10% relative reduction in action-ranking regret or a pre-registered task-appropriate absolute success improvement over the strongest matched baseline, with a paired confidence interval excluding zero.
4. **Transfer:** a prospective first gate of at least +5 percentage points on a genuinely held-out generator family, without a statistically credible loss on protected tasks. Calibrate sample size and minimum effect per task before the confirmatory run.
5. **Uncertainty:** declared interval coverage should be statistically compatible with nominal coverage on held-out strata; report calibration by regime and OOD status rather than relying on one ECE number.
6. **Retention:** no accepted corrupted memory and no material protected-skill regression; define the non-inferiority margin before training.
7. **Efficiency:** show positive decision-value-per-compute after charging model queries, simulation rollouts, training and human review. A more expensive model must justify its total cost.
8. **Replication:** at least one independent fresh-process rerun of the frozen campaign; independence is judged from generator/model/corpus ancestry, not merely a different seed.

If the effect is absent, negative, or too imprecise, report the null result and keep the simpler arm. Do not weaken gates after observing the hidden set.

### 6.5 Statistical and leakage controls

- preregister task families, primary outcomes, sample size/power assumptions, exclusions, stopping rules and analysis code;
- group uncertainty by independent generator family, not only rollout count;
- paired seeds are allowed only under a justified exogenous-randomness contract;
- distinguish independent experimental units from correlated branches of one episode;
- seal hidden test manifests and evaluator labels;
- use no-tune-after-reveal;
- include shuffled-world-model, stale-model, wrong-scope memory, poisoned-memory, missing-branch and privilege-leak negative controls;
- preserve all failed/partial executions;
- report confidence intervals and per-family breakdowns;
- use independent evaluation code where feasible.

---

## 7. Industry and research comparison

The programme should compare itself against strong, task-compatible systems—not make a single apples-to-oranges leaderboard claim.

| Comparator | What it demonstrates | What D2D should learn from it |
|---|---|---|
| DreamerV3 | Model-based reinforcement learning from imagined latent trajectories across a broad task portfolio | Include a conventional imagined-rollout baseline and charge all model/rollout costs |
| Google DeepMind Genie 4 | Real-time interactive, multimodal world simulation and natural-language world editing; announced as a research preview on 27 Oct 2026 | Treat generative worlds as high-value scenario/observation tools, but independently test long-horizon object permanence, physical interactions and spatial consistency |
| NVIDIA Cosmos Policy | Video foundation model adapted for visuomotor planning; NVIDIA reports 98.5% average LIBERO and 67.1% RoboCasa simulation success | Use strong video-model/world-model policies as comparators where compatible; compare matched data, robot, task, compute and evaluation conditions |
| Autonomous Science Bench 0.1 | Whole-agent autonomous scientific workflows on six pilot lab tasks; published leaderboard reports 72.2% for GPT-6 Astra+Codex and 66.7% for Opus 5.5+Claude Code | Benchmark experiment choice and learning over time, include cost and laboratory-workflow success, not just answer quality |
| Bayesian optimization / active learning | Strong low-complexity baselines for structured experiment spaces | HLB must earn its inference cost by discovering better experiments or transferring better |
| MLP / GRU / CfC / transformers / VLA | Alternative representation and control families | Require capacity/data/compute-matched ablations before attributing gains to HLB |

**Source caveats:** reported vendor/model results are source-reported comparators, not Luminous reproductions. Genie 4 is a research preview and its official materials note active limitations in long-session consistency and complex physical interaction. Autonomous Science Bench 0.1 is a small pilot with six tasks; its leaderboard is not a general ranking of scientific intelligence. Benchmark task protocols differ; scores must not be pooled across them.

### 7.1 Current field gaps worth attacking

Recent surveys and benchmark audits point to a consistent opportunity:
- world models are often evaluated for prediction or visual quality without measuring whether they improve executed actions;
- long-horizon error accumulation and planner exploitation remain important problems;
- uncertainty calibration, physical constraints and rollout management are central;
- robotics benchmark scores can be inflated by shortcut solvability, weak statistical significance, overfitting and data-source dependence.

The most defensible competitive wedge is therefore not “our HLB is better at imagining.” It is:

**Our system selects better experiments, makes better decisions under intervention, knows when its imagined world is unreliable, transfers verified mechanisms to unseen worlds, and exposes the full evidence trail at a lower total resource cost.**

---

## 8. Continuous online learning and HLB integration

### 8.1 Separate learning clocks

1. **Fast episode capture:** record observation/action/outcome and prediction error with exact time/source lineage.
2. **Frequent retrieval and memory routing:** retrieve only scope-compatible experiences; retain counterexamples and uncertainty.
3. **Dream consolidation:** select high-value unresolved experiences and create bounded scenario proposals.
4. **World-model adaptation:** update candidate dynamics models, not the canonical evaluator or simulator.
5. **Skill/parameter adaptation:** create candidate artifacts and run retention, regression, OOD and independent verification.
6. **Authority changes:** not in scope; require the established authorization and safety systems independently.

Online updates should be versioned and reversible. Never let untrusted web content, a generated scenario or a self-evaluated result directly modify protected parameters or physical action authority.

### 8.2 HLB role

HLB is a candidate persistent predictive state and compositional memory system:
- bind task, body/tool, environment, event and constraint relations;
- maintain fast/slow temporal state where qualified;
- predict action-conditioned futures with uncertainty;
- route memories to their appropriate task/domain scope;
- represent competing hypotheses and unresolved evidence;
- suggest the next discriminating observation or intervention.

Do not assume high-dimensional encoding or liquid dynamics are superior. Keep the HLB ablation ladder and baseline families. Any HLB contribution must be measured over the strongest comparable non-HLB model.

### 8.3 Dream world representations

Use a layered representation:
- **canonical executable state:** typed Symtropy world/scenario entities and physical variables;
- **dynamics:** explicit solver for qualified physics and agent mechanics;
- **learned residual/model:** HLB or alternative learned dynamics for uncertain/unmodeled effects;
- **observations:** sensor/rendered/audio/modal channels available to the policy;
- **evaluator truth:** hidden state and reference outcomes, isolated from the learner;
- **provenance:** manifests, digests and rights metadata.

A rendered world is not a physically validated world. A plausible video is not a trajectory receipt. A learned residual must expose its domain and uncertainty.

---

## 9. Security, safety, rights and research integrity

- Dream execution runs in a sandbox with no ambient credentials, network access or repository write access.
- Tool use is explicit, scoped, revocable, time-bounded and logged.
- The dream generator cannot call physical actuators or issue Mycelix governance/payment actions.
- Symtropy simulation outputs cannot promote themselves into real-world safety claims.
- Human/agent, private/public and licensed/unlicensed data remain separately scoped.
- External web or repository material is untrusted data, never instructions to the research agent.
- Dataset, prompt, scenario and model poisoning are first-class negative controls.
- Memory correction, retraction and supersession preserve history.
- Independent evaluator and learner have separate code/configuration and challenge-set permissions.
- Any real robot deployment retains the current deterministic local control, watchdog, e-stop, feasibility and independent safety boundary; late/stale HLB output must not become actuation authority.

---

## 10. Delivery plan

### Phase 0 — source truth and contract freeze
- audit current dream-engine functions and claim ceilings;
- freeze DreamProposalV1, PredictionSealV1, ScenarioAdmissionV1 and receipt references;
- map to existing Symtropy and Mycelix owners;
- no new world manager, memory engine, ledger or verifier.

### Phase 1 — analytic executable slice
- implement one tiny known-truth environment and branch manifest;
- add deterministic scenario proposal, prediction seal and evaluator;
- prove failure/timeout/unknown denominator semantics;
- use a fresh-process replay fixture.

### Phase 2 — one Symtropy adapter
- select one existing manipulation scenario family with a qualified simulator profile;
- execute paired interventions and emit DreamExecutionReceiptV1;
- compare the current heuristic dream prototype, random selection and matched model-based arms;
- do not introduce a universal procedural world generator.

### Phase 3 — HLB-directed curriculum
- integrate uncertainty and value-of-information selection through existing Symthaea owners;
- ablate HLB representation/dynamics/memory independently;
- validate candidate updates through VSSI and protected-skill tests.

### Phase 4 — second domain and mechanism transfer
- add a bounded Mycelix/distributed-resilience scenario family;
- hold out topology/failure mechanisms;
- test whether any transfer occurs, with negative controls for lexical similarity and wrong-scope memory.

### Phase 5 — independent campaign and reality boundary
- freeze hidden generator families and evaluator;
- run fresh-process independent replication;
- only after simulation evidence, consider hardware-in-the-loop under separate qualified robotics authority.

---

## 11. Immediate work items

1. **Symthaea:** this programme owns the DreamProposal/prediction-seal orchestration and the D2D benchmark protocol; it composes #11426, #9010, #12723 and #10844.
2. **Symtropy:** adapt existing scenario admission, world validation, action-conditioned branch execution and replay; extend #4687 only where D2D exposes a concrete missing contract.
3. **Mycelix:** use #5900/#3563/#5709 for experiment provenance, rights and independent-replication references; do not add bulk artifacts to Holochain.
4. **Research:** preregister the baseline matrix and first robot task family before confirmatory runs.
5. **Qualification:** CI pending, skipped, queued or not run is not a pass. Report source inspection, structural checks, local command results, CI results and physical evidence as separate states.

---

## 12. References

- [Google DeepMind, Genie 4: A new generation of world simulation (27 Oct 2026)](https://deepmind.google/blog/genie-4-a-new-generation-of-world-simulation/)
- [NVIDIA Cosmos Policy](https://research.nvidia.com/labs/cosmos-lab/cosmos-policy/)
- [Autonomous Science Bench 0.1](https://asb.open-ssi.org/)
- [A survey of world models for physical AI with uncertainty representation and control (2026)](https://link.springer.com/article/10.1007/s44163-026-02122-1)
- [Do World Models Make Better Robots? A Survey of Predictive Embodied Intelligence (2026)](https://arxiv.org/abs/2609.29669)
- [What Are We Actually Benchmarking in Robot Manipulation? (2026)](https://arxiv.org/abs/2606.04233)
- [DreamerV3](https://danijar.com/project/dreamerv3/)

