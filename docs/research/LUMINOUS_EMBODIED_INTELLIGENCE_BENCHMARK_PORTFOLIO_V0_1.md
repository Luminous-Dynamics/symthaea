# Luminous embodied-intelligence benchmark portfolio v0.1

**Status:** research and protocol proposal. Every target below is prospective. Published external scores are references, not Luminous results. No cross-benchmark composite score is defined.

## 1. Core proposition

The objective is not simply a robot that achieves the highest task-success rate on one leaderboard. It is a robot that can acquire the right evidence, preserve task-relevant memory, execute within bounded authority, recover from deviations, and produce independently reproducible evidence that the physical task was completed.

The platform should optimize a constrained, multi-dimensional frontier:

`verified task completion × robustness × recovery × resource efficiency × evidence completeness`

Do not multiply these quantities into a single headline score. A high average success rate must not hide unsafe events, unresolved runs, excessive human intervention, high energy consumption, or missing physical evidence.

## 2. External benchmark portfolio

| Research dimension | External benchmark / method | Primary question | Required comparison |
|---|---|---|---|
| Active perception | [ActiveArena](https://leeibo.github.io/ActiveArena/) | Does the agent seek useful evidence and use it under ID/OOD shift? | No memory, FIFO/recency, uniform sampling, chunk writing, uncertainty-triggered, planner-guided sparse memory, Luminous selector |
| Environment/process memory | [RoboCasa-Mem](https://arxiv.org/abs/2611.06475) | Does the agent remember hidden attributes, spatial relations and unfinished obligations? | Reactive, current-state-only, FIFO, task-state history, object-state trajectory, combined memory, matched HMAP |
| Distribution-shift robustness | [RoboCasa-Shift](https://github.com/HaochengWang23/RoboCasa-Shift) | Which individual outcomes change under controlled shifts, and where does execution fail? | Same policy/checkpoint/executor on paired clean and shifted initial states |
| Long-horizon/physical task execution | [RoboDojo](https://github.com/RoboDojo-Benchmark/RoboDojo) and RoboDojo-RC | Does the robot complete, recover and produce more accepted work under real constraints? | Same embodiment, task, evaluator, resource budget and intervention policy |
| Contact-aware uncertainty | [CaPTURe](https://proceedings.mlr.press/v329/marques26a.html) | Are future-configuration uncertainty regions valid in contact and contactless regimes? | Nominal Gaussian/ellipsoidal baselines, unstratified calibration, contact-stratified conformal/particle sets |
| Runtime risk control | [Calibrated VLA safety gating](https://www.mdpi.com/1424-8220/26/10/3140) and conformal selective-risk methods | Can the policy abstain or re-observe with bounded accepted risk without excessive intervention? | No gate, point threshold, calibrated threshold, fixed selective-risk certificate, conformal/fixed-sequence baseline |
| Physical measurement | NIST-compatible task methods and independent instrumentation | Did the physical task actually happen with traceable measurement? | Independent judge, instrument and calibration lineage; no policy-authored success claims |

### External reference findings

**ActiveArena.** The benchmark contains 35 tasks across five categories, with multi-round information acquisition and ID/OOD scene protocols. Published task-macro results include ActiveArena-OFT 62.97% ID / 47.60% OOD and ActiveArena-Plan 60.97% / 45.54%. Its ablations indicate uniform memory sampling, reliable memory capacity, proprioception and subtask supervision can help OOD performance; planner-guided sparse memory can approach the strongest result with less memory. This is a direct warning against assuming the most elaborate memory selector wins. Source: https://arxiv.org/abs/2609.24124 and https://leeibo.github.io/ActiveArena/

**RoboCasa-Mem.** The paper introduces nine mobile-manipulation tasks and 1,800 successful demonstrations. Its HMAP baseline reports average SR/SCR 24.2%/47%, compared with π0.5 16.2%/41%. The authors report that process-memory improvements remain inconsistent and neither interleaved scheduling task is completed by the evaluated policies. Its physical evaluation is one task with ten trials per method, so it is not broad real-world validation. Source: https://arxiv.org/abs/2611.06475

**RoboCasa-Shift.** Its formal release contains 1,040 rollouts: four clean-qualified tasks, 20 matched base episodes per task, one clean plus 12 controlled single-factor shifts. One camera condition has ΔSR=0 but eight of 80 matched episode outcomes flip. Equal ΔSR values can also hide different stage-failure depths. Source: https://arxiv.org/abs/2610.27861 and https://github.com/HaochengWang23/RoboCasa-Shift

**CaPTURe.** This 2026 PMLR paper constructs geometry-aware conformal configuration regions using particle-based predictors. It explicitly treats contact-rich and contactless transitions separately because contact can produce multimodal, disjoint or lower-dimensional future configuration distributions. It reports coverage in both regimes on two simulated tasks (marble-maze control and peg-in-hole insertion), with up to a 30-percentage-point success improvement over the best baseline. Those are narrow simulator results, not a general manipulation result. Source: https://proceedings.mlr.press/v329/marques26a.html

**Calibrated VLA gating.** A 2026 Sensors paper reports two Isaac Sim household tasks under lighting, texture, occlusion, sensor and combined shifts; its calibrated gate improves mean shifted success from 57.5% to 77.2% and lowers aggregate ECE from 0.303 to 0.100, while an aggressive uncalibrated threshold requires nearly twice as many interventions per shifted episode. This is a valuable methodological comparator, not a matched Luminous or industry-wide result. Source: https://www.mdpi.com/1424-8220/26/10/3140

## 3. Matched experiment rules

Every primary comparison must freeze:
- policy/checkpoint and forecast/post-processing pipeline;
- action executor, embodiment and robot configuration;
- task/evaluator version and official success definition;
- scene/initial-state/seed manifests;
- resource budgets, action horizons and intervention/fallback rules;
- training, calibration, validation and held-out test splits.

Use paired episodes wherever possible. Bootstrap at the independent scene/task-family level, not at the frame or action-step level. Preserve all attempted runs. Report invalid, timed-out, interrupted and unscored runs; never turn missing outcomes into zero cost or silently delete them.

Ablations should change one dimension at a time. For memory:
1. no memory;
2. current-state-only;
3. FIFO/recency;
4. uniform reservoir/chunk writing;
5. task-state history;
6. object-state trajectory;
7. combined relational/process memory;
8. planner-guided sparse memory;
9. full Symthaea memory/value-of-information selector.

For sensing:
1. no extra sensing;
2. fixed-rate sensing;
3. uncertainty-triggered sensing;
4. value-of-information sensing;
5. placebo-cost control.

For runtime assurance:
1. no gate;
2. fixed point-probability threshold;
3. calibrated point threshold;
4. finite-sample selective-risk certificate;
5. conformal/other risk-control baseline.

For contact:
1. free-motion-only calibration;
2. pooled contact/contactless calibration;
3. contact-regime-stratified calibration;
4. particle/geometry-aware prediction regions;
5. risk-gated runtime planner using those regions.

## 4. Proposed Luminous research targets

These are ambitious targets to preregister, not achieved results.

| Dimension | Proposed target | Guardrail |
|---|---|---|
| ActiveArena OOD | Exceed the 47.60% OFT reference by at least 5 percentage points under the same official protocol | ID success must not fall by more than 2 pp; matched policy/backbone/executor and memory/compute budget |
| RoboCasa-Mem | Reach ≥34% average full-task SR and ≥57% stage completion | At least one successful completion on each currently unsolved interleaved scheduling task; report all nine tasks |
| Memory value | ≥15 pp gain on a preregistered memory-required subset versus matched no-memory and simple uniform/FIFO baselines | Irrelevant-history perturbations must not alter decisions; removal of a necessary fact should cause a predictable change or abstention |
| Active sensing | ≥10 pp OOD success gain at equal or lower total episode resource budget | Report sensor-only and total energy/latency, requests versus acquisitions, and intervention rate |
| RoboCasa-Shift | ≥30% relative reduction in harmful Success→Failure transitions under held-out moderate shifts | Clean success within 2 pp; no increase in unresolved attempts; stage depth and Failure→Success transitions retained |
| Contact-aware uncertainty | ≥90% empirical region coverage in each preregistered motion regime | Coverage must be reported separately for contactless, contact-onset, sustained-contact, constrained-insertion and release; report region volume/path inflation |
| Selective risk | Upper confidence bound ≤5% for both accepted task failure and accepted unsafe events | Target ≥70% accepted coverage; same state-regime context; no missing labels; enough accepted samples for the declared bound |
| Production/field operation | ≥20% more accepted output per shift at lower intervention/rework burden | Measure energy, repair time, downtime, replacement cost and independent acceptance criteria |

These targets should be revised only before the relevant held-out evaluation is opened. Do not retune them after viewing the test results.

## 5. Risk-certificate sample-size implications

The selective-risk contract uses two one-sided exact binomial bounds and allocates `delta/2` to each. At family-wise `delta=0.05`, each bound uses `alpha=0.025`. With zero observed events:

| Accepted episodes | Upper bound for each event rate |
|---:|---:|
| 50 | 7.11% |
| 72 | 4.99% |
| 100 | 3.62% |
| 200 | 1.83% |
| 368 | 0.99% |

Therefore 50 accepted episodes cannot certify a 5% risk ceiling, even with zero observed events. The total planned episodes must be larger than the required accepted count because some forecasts will be rejected by the frozen score rule. If separate certificates are issued for multiple tasks, shifts or motion regimes, preregister a multiplicity plan or allocate the overall error budget across them.

## 6. A new Luminous meta-benchmark: evidence-backed embodied intelligence

The long-term contribution should be a reproducible *evaluation protocol*, not a new leaderboard that merges unrelated success rates.

For each episode, create one immutable evidence envelope linking:
1. task intent and evaluator version;
2. scene, geometry, coordinate frames and robot configuration;
3. policy, forecast pipeline, memory snapshot and sensing decisions;
4. controller admission, request/ticket identity and action chunks;
5. raw observations, instrument identities, calibration and synchronization uncertainty;
6. physical outcome, stage/milestone receipts, failure/abstention and recovery;
7. exact artifact digests and reproducible analysis.

The envelope is not proof merely because it is signed or hashed. The underlying measurements, evaluator independence, calibration currentness and instrument provenance must still be qualified.

Do not define a composite SOTA number. Publish a dashboard of task success, stage progress, paired outcome transitions, accepted coverage, selective-risk bounds, calibration, intervention rate, energy, latency, unresolved outcomes, recovery and physical-evidence completeness.

## 7. Criteria for a defensible SOTA claim

A Luminous result may be described as better than a published baseline only when:
- the public task/evaluator protocol is the same or differences are explicitly reconciled;
- policy, checkpoint, training data access, action interface, hardware/simulator and seed split are pinned;
- the baseline is reproduced or its original raw records are independently reanalysed;
- the improvement is statistically supported with matched/clustered analysis;
- clean competence, OOD degradation, operational faults, unresolved outcomes, intervention and total cost are all reported;
- independent reviewers can rerun the analysis from the raw episode records.

Until then, use phrases such as "source-audited", "adapter implemented", "synthetic contract tests pass", "simulator smoke passed" or "physical qualification pending" rather than "SOTA" or "industry-beating".

## 8. Next execution sequence

1. Qualify the exact-head Rust contract workflows; do not wait on them to start separate research.
2. Finish the current exact-context selective-risk and process-memory manifest corrections.
3. Build the first native ActiveArena adapter with the upstream seed/task/action mapping pinned.
4. Resolve RoboCasa-Mem current wheel, simulator asset coverage and data rights before native baseline claims.
5. Add a task-specific StageTracker/milestone receipt adapter for RoboCasa-Shift; do not infer stage states from policy text.
6. Add contact-regime manifests and geometry-aware prediction-region evaluation for peg-in-hole / assembly scenarios.
7. Run matched policy ablations with the same backbones and resource budgets.
8. Only after simulator qualification, move selected tasks to C0/HIL and supervised physical trials.

The strategy is to outperform by being more reliable, more resource-efficient and more independently verifiable—not by maximizing one score while ignoring everything it fails to measure.
