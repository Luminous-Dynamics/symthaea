# ROB-FEP-THERMO-001 — competitive landscape and benchmark targets

Date: 10 November 2026  
Status: research synthesis and proposed evaluation plan; no comparative result claimed.

## Executive position

Do not compete on "entropy minimization" or on one scalar energy-efficiency leaderboard.

The competitive target is **thermodynamically constrained, observation-aware mission recoverability**:

> Under matched embodiment, plant, payload, sensing, action, training and compute budgets, can a belief-aware mission executive preserve more useful future mobility and terminal recovery options across sequential faults and persistent thermal/energy state than the strongest matched MPC/HQP, robust-control and thermal-aware learning baselines?

The contribution is the composition and measurement contract, not a claim that FEP is inherently superior.

## 1. Current reference landscape

The following results are distinct strata. They must not be pooled into one leaderboard.

| Research line | Reported result / scope | What it establishes | What it does not establish |
|---|---|---|---|
| RAIBO2 marathon, Lee et al., Nature 2026 | 42.195 km; 4 h 19 min 52 s; total CoT 0.25; one battery charge | A compelling whole-system hardware/drive/policy energy-efficiency reference | Thermal-aware active diagnosis or sequential-fault recovery superiority |
| A1 thermal-aware locomotion, Qian et al. 2026 | >27 min versus ~7 min baseline under that study's fixed 3 kg payload | Thermal state can materially change sustainable locomotion duration | Cross-platform endurance or mission recovery topology |
| A1 residual thermal policy, Wan et al. 2026 | >13 min versus ~5 min nominal policy in a separate 3 kg payload experiment | A residual policy can improve thermal endurance in its declared experiment | A second measurement of Qian et al.'s experiment |
| Keep It Cool, De Benedittis et al., IEEE TCST 2026 | Thermal-aware HQP/MPC; redundant planar-manipulator hardware validation | Temperature constraints can be integrated into task-prioritized redundant control | Whole-body quadruped sequential-fault results |
| PACE / systematic sim-to-real transfer, Bjelonic et al., IJRR 2026 | Physics-grounded PMSM energy model; 32% reduction in ANYmal full CoT; tested on three primary platforms and deployed on ten additional robots | Bottom-up actuator loss modeling and transfer can deliver substantial energy gains | Observation-constrained recoverability or FEP value-add |
| STEP-CBF+, Scientific Reports 2026 | In the paper's randomized reduced-order simulation, reported 4.0% mean falls versus 16.7–18.7% for comparators; ~13.6 ms mean selection time versus ~30.5 ms for reduced-order convex MPC | Foothold/posture screening and contact-feasibility integration can improve its tested terrain cases | Hardware validation, whole-body invariance, or a direct comparison with Luminous |
| Impact-aware robust convex MPC, IEEE RA-L 2026 | Go2 simulation and hardware; min–max MPC compared with conventional/adaptive MPC under impact conditions | Robust MPC is a serious physical-control comparator under uncertain terrain impacts | Thermal mission reserve or value-of-information superiority |
| Viability-preserving passive torque control, ICRA 2026 | 7-DoF Franka simulation and hardware | Viability sets can constrain passive torque control while retaining useful performance | Legged thermal/fault mission results |
| GenAIF, arXiv:2610.05692, October 2026 | MuJoCo manipulation; diffusion, autoregressive, CVAE and flow-matching trajectory models; diffusion strongest control in tested dynamics, CVAE comparable short-horizon prediction with faster inference | Active-inference systems need useful action proposals, action-conditioned prediction and observation likelihoods; inference cost matters | A thermally constrained physical quadruped or a complete safe execution architecture |
| GLean, Wiley 2026 | Hierarchical latent-variable active inference and humanoid research line | A current active-inference planning comparator worth tracking | Do not count simulated trajectory evaluations as hardware locomotion validation; real-time compute remains a relevant limitation |
| Test-time physical diagnosis, arXiv:2609.22299 v2 (26 Oct 2026) | Identifies six ordered gates: meaningful reference, identifiability, evidence use, decision value, selection value and safe realization; reports transfer/OOD failures in its studied panels | Evidence acquisition alone does not prove evidence is used or improves action | A universal diagnosis failure rate; this is one preprint's controlled/public-task evaluation |

Primary references:

- RAIBO2: https://doi.org/10.1038/s41586-026-11102-5
- A1 thermal-aware locomotion: https://arxiv.org/abs/2603.01631
- A1 residual thermal policy: https://arxiv.org/abs/2605.27046
- Keep It Cool: https://doi.org/10.1109/TCST.2026.3721130
- PACE sim-to-real energy model: https://doi.org/10.1177/02783649261459628
- STEP-CBF+: https://www.nature.com/articles/s41598-026-72507-w
- Impact-aware robust MPC: https://doi.org/10.1109/LRA.2026.3692086
- Viability-preserving passive torque control: https://doi.org/10.1109/ICRA57385.2026.11696479
- GenAIF: https://arxiv.org/abs/2610.05692
- GLean: https://doi.org/10.1111/tops.70081
- Physical diagnosis: https://doi.org/10.48550/arXiv.2609.22299

## 2. What Luminous should do differently

### Shift A — from lowest cost of transport to capability retained per mission

CoT remains essential, but it is not a complete reliability metric. Add sequential-mission outcomes:

- useful work completed before a safe-stop or mission abort;
- number and severity of disturbances survived without resetting battery, temperature, wear or uncertainty;
- time and resource expenditure to recover;
- number of observation-constrained common continuations remaining after each event;
- independent recovery paths after accounting for common-cause failure domains;
- probability/coverage of mission completion conditional on an explicitly defined fault sequence.

The mission, not the time step, is the statistical unit.

### Shift B — from thermal thresholds to a coupled, uncertain physical state

Use the embodiment-specific thermal network and actuator loss evidence. Predict upper-bound node temperatures and derating, not only mean temperature. Carry the complete state through gait changes, probes, cooldown, recovery and successive faults. Never reset thermal history between events.

Consume #4044, #4179, #4206 and #4610; do not create another energy ledger, actuator map or thermal model.

### Shift C — from information gain to decision-value-qualified sensing

A robot should not probe simply because a probe is expected to reduce entropy.

For each proposed observation/intervention, test the ordered chain:

1. Is there a meaningful reference/current measurement?
2. Is the relevant fault/mechanism identifiable?
3. Does the delivered evidence actually change the belief/predictive distribution?
4. Does that evidence change the downstream optimal/admissible decision?
5. Is the selected observation better than a fixed or cheaper alternative at matched budget?
6. Can the resulting action be independently admitted and safely realized?

Report EIG, decision-focused EVSI, realized posterior risk reduction, model adequacy and evidence use separately. Preserve no-probe as an explicit option.

### Shift D — from clairvoyant option counts to observation-constrained continuation

A robot cannot choose a different continuation for hidden states that generate the same delivered observation. Group physically possible successor states by their actual observation equivalence and intersect independently admitted continuations for each group. Treat this as the deployment metric; retain per-state clairvoyant option count only as an optimistic upper bound.

Compose Symtropy #4511, #4549/#4555 and #4588.

### Shift E — from a monolithic FEP claim to a compute-matched architecture portfolio

Compare rules, Bayesian/set-membership inference, robust EVSI, corrected action-conditioned FEP, thermal-aware MPC/HQP, robust MPC and thermal-aware RL. Add GenAIF-style trajectory-model comparators where feasible.

HDC, CfC/LTC and FEP must earn their place through held-out value at equal observations, compute, memory and latency. Report p50/p95/p99/worst latency, deadline misses, measured compute energy and modelled compute energy separately.

## 3. Benchmark programme

### B0 — analytical correctness

- 100% exact agreement on the finite known-answer corpus for equations, units, interval corners and dispositions.
- Reject non-finite values, unit/boundary mismatch, invalid temperature, reversed declared heat direction, missing heat ports and stale predecessor identity.
- Keep belief entropy, information gain, thermodynamic entropy and exergy in separate typed/reporting domains.
- No physical entropy/exergy result when boundary, environment reference or source evidence is incomplete.

This is correctness of the declared finite model, not physical validation.

### B1 — FEP/decision-value correctness

- Exact agreement with independent enumeration on bounded Bayesian EVSI fixtures.
- The current action-invariant epistemic score remains a mandatory negative control.
- A high-EIG/zero-EVSI probe is declined when costly or risky.
- A lower-EIG/high-EVSI probe can win when it changes the optimal downstream decision and remains admissible.
- Out-of-distribution or unidentifiable evidence causes an explicit Unknown/abstention, not false confidence.
- Report decision regret, Brier/log score, calibration, EIG, EVSI and realized decision lift separately.

### B2 — thermal model and actuator response

- Reproduce analytical RC-network and actuator-response references before using a simulator as truth.
- On held-out full trajectories and held-out units, report per-node temperature bias/MAE/RMSE, interval coverage/width, peak upper-bound error near derating, time-to-derating error and cooldown/requalification delay.
- Proposed target: nominal 90% and 95% interval coverage should be calibrated by node and operating stratum, not merely in pooled aggregate.
- Proposed target: upper-bound temperature error should consume no more than 20% of the declared remaining thermal safety margin on the held-out qualification set; freeze the tolerance only after sensor metrology and baseline variance are known.
- Zero acceptance of stale thermal predecessor digests, incomplete node manifests or average-temperature masking of an over-limit node.

### B3 — sequential degraded missions

Minimum scenario sequence:

1. traversal under a declared payload and ambient condition;
2. heating-induced actuator derating;
3. ambiguous sensor evidence (real heat versus sensor bias/lag);
4. candidate diagnostic probe with measured/upper-bound energy, time and thermal cost;
5. a second fault before complete cooldown;
6. terrain/contact or communication degradation;
7. recovery action whose success changes the remaining battery/thermal/mechanical reserve;
8. terminal safe-stop or retreat under observation constraints.

Preserve all physical and evidence state across the sequence. Include reversed event order and physically possible zero-probability branches.

Compare at equal hardware/plant, observation, action, training and compute budgets:

- rules / finite-state mission executive;
- Bayesian/set-membership + EVSI;
- current audited FEP (negative control);
- corrected action-conditioned FEP + EVSI;
- thermal-aware MPC/HQP or NMPC;
- robust min–max MPC / viability-filtered control;
- thermal-aware RL / residual policy;
- evaluator-only clairvoyant upper bound (never deployable).

### B4 — sim-to-real and physical qualification

1. Reproduce a public reference in its exact declared domain (A1 thermal endurance or PACE actuator-loss/CoT transfer) before claiming a comparison.
2. Use an independently qualified external plant model while Symtropy's relevant dynamics remain unqualified.
3. Instrument one actuator/joint before a whole robot. Synchronize voltage/current, torque, encoder, temperature, ambient and timestamps.
4. Run shadow mode before any authority change. FEP remains proposal-only; RTA/HAL retains independent control authority.
5. Repeat on hardware with held-out actuator unit, mission order, payload, ambient and sensor faults.

## 4. Proposed success targets (aspirational, not achieved)

- **Safety/evidence:** zero false acceptance in finite adversarial suites for missing/stale state, hidden over-limit nodes, incomplete successor branches, lost safe-stop or mismatched physical action identity.
- **Inference:** exact finite EVSI agreement; calibrated uncertainty; no OOD confidence escalation.
- **Thermal:** match or exceed the declared A1 thermal reference under the same platform/payload/environment before claiming locomotion-endurance progress; do not compare its 27-minute result with another platform.
- **Energy:** reproduce the matched PACE/CoT baseline first. Any Luminous improvement must preserve raw battery-inclusive energy boundaries and report compute/cooling/standby separately.
- **Mission viability:** proposed stretch gate is at least a 10% relative reduction in unrecoverable mission failures versus the strongest matched controller on held-out sequential-fault missions, with the paired 95% confidence interval excluding no improvement, while service completion, hard constraints and adverse-event categories are non-inferior. Revise the effect-size threshold only before confirmatory evaluation, based on pilot variance and task importance.
- **Real-time:** p99 planning/diagnostic latency at or below 50% of the declared planning deadline on the target hardware, with zero missed hard deadlines in the qualification set. This is an engineering margin target, not a published industry standard.
- **Architecture value:** remove HDC/CfC/FEP components that do not improve the held-out service/viability frontier at matched compute and observation budgets.

## 5. The result that would constitute a real contribution

The strongest result would not be "our robot has lower entropy".

It would be:

> On held-out sequentially degraded missions, a thermodynamically grounded, evidence-qualified mission executive preserves more feasible and observable recovery continuations than the strongest matched thermal-aware MPC/HQP, robust-control and RL baselines, without exceeding hard physical limits or buying its advantage with extra energy, sensing, training data or compute.

Entropy/exergy measurements can then help explain physical irreversibility or quantify lost work potential. They remain secondary until they demonstrably add decision value over measured energy, actuator-loss, thermal and mission-state evidence.

## 6. Evidence and claim boundaries

- External reported results remain within their paper's embodiment, payload, task, environment and simulator/hardware strata.
- Simulation-only results are not hardware results.
- Synthetic arithmetic is not thermal-model calibration.
- A calibrated thermal predictor is not a continuous-time safety proof.
- A correct information metric is not a causal intervention model.
- A viable route under hidden state is not a route the robot can choose without observation.
- No FEP score, entropy score, Phi value or model confidence grants actuator authority.
- Report null results and worst-case strata; do not let averages conceal unsafe subpopulations.


## 7. Additional current comparators and public benchmark context

### Stronger quadruped control comparators

- **SSZT-MPC on LeoQuad (IEEE/ASME Transactions on Mechatronics, August 2026):** simultaneous centroidal state and input-derived ZMP tracking with velocity-error adaptation; full-size quadruped hardware trials include disturbances, uneven terrain, stairs and acceleration/deceleration. This should be a serious balance-control comparator where the platform and task are matched. DOI: https://doi.org/10.1109/TMECH.2026.3714419
- **Variable-bandwidth nonlinear extended-state-observer whole-body disturbance rejection (Control Engineering Practice, November 2026):** combines lumped disturbance estimation, adaptive foothold planning and constrained whole-body control. Add as a recent observer/adaptation comparator; confirm its platform, code and hardware evidence before assigning a leaderboard tier. DOI: https://doi.org/10.1016/j.conengprac.2026.107167
- **Terrain-adaptive MPC + impedance/admittance + SINDy corrections (McConnellogue et al., 14 August 2026):** reports 40% slip reduction, 40–60% force-tracking improvement and 80% yaw-tracking improvement in PyBullet. Importantly, the paper explicitly describes its validation as simulation and identifies physical PAWS validation as future work. Treat it as a simulation-only controller ablation comparator, not a hardware SOTA result. DOI: https://doi.org/10.1002/rnc.70700

### Industrial digital-twin / predictive-maintenance comparator

Aasa et al. (16 June 2026) report a physics-informed digital twin with edge computing, hardware-in-the-loop on a UR10e and Jetson AGX Orin, and claimed 98.2% simulation-response similarity, 99.7% synchronization accuracy and 34% lower unplanned downtime in their study. This is an industrial arm/HiL comparison, not a directly matched mobile robot or independent replication. It reinforces that Luminous must measure end-to-end synchronization, fault-detection sensitivity, compute cost and downtime/service impact—not only internal model prediction. Source: https://link.springer.com/article/10.1007/s44291-026-00241-8

### Cross-domain embodied benchmark context

Stanford HAI's **AI Index 2026** gives a useful warning about benchmark saturation and task-horizon effects:
- RLBench 18-task subset: EquAct 89.4% average task success as of January 2026, in controlled short-horizon evaluation.
- BEHAVIOR-1K 2025 challenge: top reported Q-score 25.99%, but full-task success 12.4%.
- ResponsibleRobotBench: reported best safe-success score 0.64 across 23 multi-stage hazard tasks.

These are not thermal locomotion baselines and must not be pooled with quadruped results. They motivate a complementary long-horizon manipulation / household / hazard lane in which success requires the complete task *and* safety. Source: https://hai.stanford.edu/assets/files/ai_index_report_2026.pdf (section 2.7, pp. 117–120).

### Additional research implications

1. Report the first failing stage in the diagnostic chain, not just final recovery accuracy.
2. Separate physical hardware, hardware-in-the-loop, full-physics simulation, reduced-order simulation and synthetic arithmetic in every comparison label.
3. Include state/ZMP tracking, disturbance observers and impedance/admittance adaptation in the locomotion baseline suite where the platform permits.
4. Extend beyond locomotion only after the quadruped and single-joint tracks have a repeatable protocol; preserve separate manipulation and hazard benchmarks.


## 8. Real-time world-evolves-during-inference benchmarks

The competitive programme now adds a distinct real-time lane, tracked by [Symthaea #11561](https://github.com/Luminous-Dynamics/symthaea/issues/11561):
- **RoboDyna** (NeurIPS 2026 workshop) evaluates dynamic manipulation in which the world does not pause during inference. It directly motivates charging model/planning time against state freshness and task outcome.
- **SafeLab** (ICML 2026) has 64 tasks in nine lab manipulation categories, reports safe-success separately from ordinary task completion, and includes 50 open-loop physical replays with 86% simulation/observed safety-outcome agreement.
- **AM-Bench** (2026) makes policy × low-level-control × embodiment a configurable comparison rather than treating task success as a policy-only property.
- **Dual control / simultaneous SysID-MPC** is a required alternative to explicit diagnostic probing: it learns while performing task actions. The IEEE TRO 2025 method includes hardware and physics-based simulation.
- **Long-horizon system identification** on a KUKA LWR iiwa reports measured-input rollouts up to 1000 steps and ID/OOD splits; it is a reminder that one-step fit is not enough for iterative planning.

These lanes are not pooled with quadruped endurance or thermal manipulation metrics. They add an orthogonal comparison: whether more computation, more observations or a richer model improve the action *at the time it is actually applied*, after the environment has evolved and the physical resources have changed.


## 9. New real-time benchmark federation (11 November 2026)

Three public benchmarks now provide complementary, maintained reference lanes:

### RT-SAFE — navigation safety while reasoning

RT-SAFE's October 2026 preprint uses 5 city maps, 36 navigation routes, 8 VLMs and 16 available actions. In its matched static-versus-real-time comparison, task completion changes from 91.3% to 94.1%, while safe completion falls from 19.8% to 0.7%; collision count rises by 12.3×. The study reports that the simulated world advances during inference and separates collisions during inference from collisions during action.

Use it to test safe arrival, passive/inference-window collisions, active/action collisions, route compliance and real-time reasoning cost. Important limitation: the public snapshot notes exploratory route selection, single-seed route conditions, provider-specific settings and exclusions of some incomplete/output-truncated/model-identity-mismatched episodes; it is a preprint, not a physical mobile-robot benchmark. Keep its route/task metrics separate from locomotion.

Sources: https://arxiv.org/abs/2610.09294 ; https://xtrose29.github.io/RT-SAFE/

### LIBERO-Dynamic — latency-aware manipulation

LIBERO-Dynamic covers all 40 LIBERO tasks with three target-motion families, three speed tiers and a shared 1,800-episode population. Model inference latency is calibrated and frozen while the physical simulation continues. It supports synchronous hold, asynchronous queueing, RTC and a delay-aware VLASH reference, and reports four real-robot conditions at two action horizons.

Released π0.5 PyTorch execution-strategy results are 43.1% success for synchronous hold (K=10), 48.4% for async queueing (K=6), and 37.8% for RTC (K=16); separately trained VLASH reaches 55.4%, so it is not a controller-only ablation. The JAX π0.5 asynchronous result is 57.8% over the 1,800-episode population. These are specific model/training/horizon strata, not universal rankings.

Use it for action-horizon sweeps, queue underflow, observation staleness, asynchronous chunk alignment, and physical transfer. It is the best immediate external manipulation lane for our timing protocol.

Source: https://github.com/renshaojie233/LIBERO-Dynamic

### RoboDyna — contact-rich dynamic manipulation

RoboDyna supplies 20 conceptual tasks plus 10 household tasks, 3,500 demonstrations and paired sync/async evaluation. In its 5.4K episode model evaluation, all tested policies achieved under 20% success; latency flipped 5–12% of episode outcomes while average success changed at most 3%, and human baselines outperformed the tested policies under matched conditions.

Use it for causal physical interaction, moving-object prediction, partial completion and human reference comparisons.

Source: https://robodyna.github.io/robodyna/

### Common cross-lane temporal contract

Use a shared event record across these lanes:
- observation cutoff and clock-domain identity;
- observation timestamp interval / uncertainty;
- model and policy digest;
- proposal generated time and action-chunk identity;
- executed-prefix digest and next action index;
- clock-map offset/drift interval;
- projection/RTA admission time and result;
- controller dispatch time and matching execution outcome;
- action application time and physical-state snapshot.

The cross-lane invariant is that an action's source observation, model, executed prefix and physical application time remain traceably aligned. The exact platform safety controller remains independent.

Do not pool scores across RT-SAFE navigation, LIBERO-Dynamic manipulation, RoboDyna dynamic interaction and Symtropy thermodynamic locomotion. Standardize event semantics and compare within each lane.

The action-chunk known-answer corpus and contract are in [action_chunk_temporal_alignment.md](./action_chunk_temporal_alignment.md) and [fixtures/action_chunk_alignment_v0_1.json](./fixtures/action_chunk_alignment_v0_1.json).


## 10. Distributed inference placement: onboard vs edge vs cloud

Microsoft Research's September 2026 study measures physical-AI inference across onboard, edge and cloud GPU placements for mobile manipulation. It reports that offloading can improve performance and battery life, while introducing trade-offs in network latency, bandwidth and available remote GPU capacity. The March 2026 Microsoft measurement-report abstract further notes that smaller onboard GPUs may not support the full workload, large onboard GPUs can shorten battery life, and network latency can erode task accuracy.

This is an industry comparator and an important systems direction. It is not direct evidence that cloud FEP is beneficial for a quadruped, nor proof of whole-system energy savings. Luminous should compare:
- onboard-only / onboard-lightweight;
- nearby edge / remote cloud;
- hybrid local safety plus remote deliberation;
- asynchronous chunked offload;
- fleet-shared edge with contention;
- adaptive placement under deadlines and resource reserves.

Measure complete capture-to-admission latency and robot, network, edge and cloud energy separately. Report robot battery endurance and whole-system energy independently. A cloud result is never actuation authority; local RTA/HAL and safe-stop must remain available under network partition.

Sources:
- https://www.microsoft.com/en-us/research/blog/offloaded-inference-for-real-world-physical-ai-robotics/
- https://www.microsoft.com/en-us/research/publication/offload-or-overload-a-platform-measurement-study-of-mobile-robotic-manipulation-workloads/

Implementation/evaluation owner: [Symthaea #11573](https://github.com/Luminous-Dynamics/symthaea/issues/11573).


### Distributed inference claim boundary refinement

The distributed-inference fixture corpus now requires an explicit whole-system energy-boundary completeness flag and an onboard baseline for a whole-system comparison. It also adds an unauthenticated-endpoint negative control. A remote result may be eligible for local projection while its whole-system energy benefit remains Unknown; endpoint identity, freshness and physical action admission are separate gates.


## 11. Action-horizon baselines: latest 2026 WAM execution research

The new action commitment benchmark (#11581) now compares against four distinct advances:

- **TempoWAM** (arXiv:2608.09492): progress-monitoring-driven replanning; reports 26.9% fewer WAM inferences on easy real-world tasks while maintaining success, and +13.3 percentage points on difficult tasks. https://arxiv.org/abs/2608.09492
- **AHA-WAM** (arXiv:2606.09811): slow world planner / fast action expert with observation-guided context routing. Crucially, the authors state that "horizon-adaptive" describes decoupled world and action horizons, not online action-prefix-length selection. It reports 92.80% RoboTwin success, 78.3% over four real-world tasks and 24.17 Hz in its stated setting. https://arxiv.org/abs/2606.09811
- **Streaming-WAM** (arXiv:2609.28927): conditions future visual prediction on actions already committed during asynchronous inference; reports 98.35% LIBERO average success, 2.93× lower episode time than Fast-WAM, and a Stamp Paper real-world task reduction from 90 s synchronous to 38 s asynchronous. https://arxiv.org/abs/2609.28927
- **Reuse-then-Refine** (ICML 2026): latent-space high-frequency action chunks plus asynchronous refinement; evaluates three real-world contact-rich tasks and targets temporal/spatial consistency at up to 60 Hz. https://proceedings.mlr.press/v306/wang26lj.html

These results are task/training/hardware-specific. They should not be combined into a cross-paper ranking. The Luminous research question is narrower: do physical hard bounds and observation-constrained viability give a better committed-prefix horizon than fixed K, task-progress monitoring, or model-architecture-specific chunking on a matched platform/task?

The reference formula is a minimum over qualified integer caps—not a composite score. Any Unknown/stale/mismatched required cap blocks new-prefix eligibility; the output never grants actuation authority.


### Adaptive horizon composition conditions

The #11581 minimum-of-bounds reference is only sound when all qualified caps refer to the same state/action/model/uncertainty/clock/control-period context and each cap is prefix-closed over every intermediate tick. The 17-case corpus contains explicit mixed-context and non-prefix-closed negative controls. These prevent a superficially safe minimum from combining incompatible evidence or hiding an intermediate constraint violation.


## 12. Additional action-horizon comparators: PACE, ChunkTrust and REMAC

The action-horizon lane now includes three additional public baselines:

- **PACE — Phase-Aware Chunk Execution** (arXiv:2606.00537, 30 May 2026): selects execution/replanning boundaries using low-speed transitions in the predicted action chunk. It reports RoboTwin 2.0 average success increasing from 57.8% to 64.2% over 50 tasks; on ALOHA and Franka real-robot tasks it reports average success 50.7% to 70.4% and task score 60.7 to 77.7 in its own evaluation. This is a test-time phase selector, not a physical safety bound. https://arxiv.org/abs/2606.00537
- **ChunkTrust** (arXiv:2609.39754, 30 September 2026): infers horizon reliability from action-expert spectral stability, inter-chunk continuity, and an online Beta posterior. It reports +6.80 percentage points for π0.5 on RoboTwin 2.0, +9.67 points for Qwen3GR00T on RoboCasa, and four real-world household tasks with normalized process score 50.4% to 57.5% in its own task strata. This is an action-evidence horizon proposal, not a qualified thermal/viability cap. https://arxiv.org/abs/2609.39754
- **REMAC** (ICLR 2026): masked action chunking and prefix-preserved sampling target intra-chunk perception/execution mismatch as well as inter-chunk continuity. This adds an important training-time robustness arm beyond post-hoc chunk blending. https://proceedings.iclr.cc/paper_files/paper/2026/hash/c35834443b7881e782e52b3519fe27c7-Abstract-Conference.html

**Fair comparison design:** evaluate the selector first as a candidate-horizon proposer, then apply the same Luminous physical bottleneck envelope to each proposer. The envelope must not improve a selector's model prediction; it only clips commitment to a common qualified admissible prefix. Compare proposal quality, clipped horizon, safe-success, replanning overhead and physical outcomes separately. Do not pool platform/task results across papers.


## 13. OpenWAM and counterfactual action-outcome identification

OpenWAM (6 October 2026) is a particularly useful comparator because it exposes video/action interaction programs under a common model family: video-then-action, action-then-video, joint and decoupled generation, plus separate local inverse/forward dynamics. Its reported results include 98.6% mean VTA success across four LIBERO suites and about 92% mean success on three bimanual tasks. Its counterfactual supervision reduces RGB prediction error by 34.5% and improves outcome identification among 16 same-state action outcomes from 21.1% to 71.3% in its evaluation. https://openwam.stanford.edu/

The important benchmark implication is that visual prediction, action-outcome identification, action ranking and decision value are four separate gates. Symthaea should consume #598/#664 for action-conditioned physical/causal evidence and #10625 for repeated held-out continuation evaluation. Do not build a new world model or a second selectable-headroom benchmark here.

See [counterfactual_action_value_extension.md](./counterfactual_action_value_extension.md).


## 11. New asynchronous world-action model comparators (11 November 2026)

The action-chunk temporal benchmark must distinguish three separate recent methods that are easy to conflate by name:

### Streaming-WAM — action-conditioned prefix continuation (Huang et al., arXiv:2609.28927, 24 September 2026)

This method conditions future visual prediction on the actions already committed for execution, then generates the remaining chunk. Its project page reports:
- LIBERO: 98.35% average success and 41.0 ms chunk time, with a 12× chunk-time reduction versus FastWAM.
- RoboTwin 2.0: 91.74% success versus 87.0% for FastWAM-Joint, with 54.4 ms chunk time.
- Real Stamp Paper task: 27/30 successes (90%) and 38 s mean time versus 26/30 (86.67%) and 90 s for Joint-WAM.

These are the authors' declared evaluation strata and hardware; they are not direct comparisons with Symthaea, and the H100-based throughput must not be assumed to transfer to the user's target robot.

Paper: https://arxiv.org/abs/2609.28927  
Project/code: https://github.com/SJTU-DENG-Lab/Streaming-WAM

### AHA-WAM — slow world context, fast action expert (Cai et al., arXiv:2606.09811, 8 June 2026)

A dual-DiT architecture runs a low-frequency world planner with rolling latent context, while a high-frequency action expert reuses that context and adapts to the latest observation. The paper reports 92.80% average success on RoboTwin, 78.3% across four real-world tasks, and 24.17 Hz closed-loop control (4.59× Fast-WAM in the authors' comparison).

This gives Luminous a testable architectural hypothesis: Symthaea's slower deliberative/world-model operations should not block a faster local control/continuation process; the fast process must nevertheless bind to the current observation and committed action prefix.

Paper: https://arxiv.org/abs/2606.09811

### StreamingWAM — chunk-wise staircase flow matching (Ren et al., arXiv:2610.15585, 11 October 2026)

A separate work, despite the similar name. It combines a noise ladder, prefix rollout and dense video-history memory. The abstract reports steady-state replanning latency of 72.9 ms in simulation and 57.3 ms on a real robot, with smoother transitions and reduced sensitivity to action horizon K.

Paper: https://arxiv.org/abs/2610.15585

### New required ablations

For any Symthaea/HDC/CfC/FEP contribution, compare:
1. Single-timescale world/action model.
2. Slow world model + fast local action proposal.
3. Slow world model + fast action proposal conditioned on the *executed/committed prefix*.
4. Prefix-conditioned proposal plus independent temporal eligibility, action projection and runtime assurance.
5. Same architecture with a stale or shuffled world-context negative control.
6. Direct reactive policy with no world-model access.

Report chunk latency, closed-loop frequency, queue-underflow, boundary discontinuity, task success, safe-success, action regret at application time, observation age, compute/battery/thermal cost and recovery viability. Do not optimize success alone by lengthening chunks or allowing stale commands.

The contribution we should aim to establish is not "another WAM". It is whether this asynchronous split-timescale pattern can be made evidence-qualified and thermodynamically viable across different embodiments and persistent degradation—not whether it wins a LIBERO score against a different training recipe.


## 12. Thermodynamic asynchronous locomotion benchmark

The physical quadruped lane now has a dedicated experiment owner: [Symtropy #4689 — LOCO-ASYNC-THERMAL-001](https://github.com/Luminous-Dynamics/symtropy/issues/4689).

It composes:
- #4610 qualified per-node thermal transitions;
- #4179/#4206 actuator loss and command-to-motion dynamics;
- #4549/#4555 persistent mission reserve and event order;
- #4588 observation-constrained future mobility;
- #4117 timing/certificate validity and #4382 dynamics projection;
- Symthaea #11561 real-time cognition and #11573 inference placement.

The matched experiment asks whether asynchronous split-timescale cognition, committed-prefix continuation and independent temporal eligibility improve **sequential legged missions with persistent heat, actuator degradation and recovery state**, against thermal-aware MPC/HQP, robust MPC, thermal-aware RL and direct policies.

WAM methods (Streaming-WAM / AHA-WAM / StreamingWAM) remain manipulation-domain architectural comparators until ported to a platform and evaluated under a matched locomotion protocol. Do not reuse their LIBERO/RoboTwin numbers as quadruped claims.


## 13. Combined legged execution × thermodynamic mission lane

The combined quadruped/legged experiment is now owned by [Symtropy #4689 — LOCO-ASYNC-THERMAL-001](https://github.com/Luminous-Dynamics/symtropy/issues/4689), with the first 12-case composition fixture and dedicated validator in [draft PR #4694](https://github.com/Luminous-Dynamics/symtropy/pull/4694).

This lane tests the asynchronous split-timescale / committed-prefix idea under persistent per-node temperature, actuator authority, event order and observation-constrained recovery. It compares to thermal-aware MPC/HQP, robust MPC, thermal-aware RL and direct high-rate policies. WAM methods remain manipulation references until a matched quadruped implementation exists.

The fixtures qualify gate composition only; #4610 remains the physical thermal model owner and #4549/#4555/#4588 retain mission carryover and future-option measurement.


## 14. Viability-bounded adaptive action horizon

A new child workstream [Symtropy #4696](https://github.com/Luminous-Dynamics/symtropy/issues/4696) treats action horizon K as a joint evidence/physical constraint rather than a fixed model parameter. It composes observation freshness, clock mapping, chunk expiry, deadline, certificate validity, contact-model horizon, actuator-response validity, per-node thermal-to-limit time and terminal safe-stop/recovery reserve.

For queued open-loop execution, the measured p99 total proposal path defines K_min; the earliest qualified physical/evidence horizon defines K_max. If K_min > K_max, queued execution cannot safely cover deliberation. A continuously operating split-timescale local expert is evaluated as a distinct architecture: it does not need to span the slow context latency, but its own cycle must meet the local deadline and its chunks remain bounded by K_max.

Initial 12-case corpus and dedicated validator: [Symtropy draft PR #4704](https://github.com/Luminous-Dynamics/symtropy/pull/4704). This is an arithmetic/evaluation contract, not a new scheduler, certificate validator or controller.


## 11. Decision-sensitive actuation freshness

A new benchmark extension is in [decision_sensitive_freshness.md](./decision_sensitive_freshness.md). It adds actuation-side Age of Information (AoI), distinct from receipt latency and result age, and separates AoI from EVSI / stale-decision regret. It also adds external baselines:
- EdgePace for memory-bandwidth-aware LLM scheduling around foreground sensor deadlines;
- Jeon et al. (Frontiers 4 Nov 2026) for a physical robot with actuator-side AoI and a payload/freshness saturation result;
- FASTER, PACT-VLA and REMAC for compatible asynchronous action-chunk / fast reaction comparisons.

Do not pool manipulation and locomotion metrics. The same temporal-event semantics should be preserved across the distinct platform lanes.


## 12. Multi-robot serving and deadline-aware inference scheduling

### Armory: action-chunk lookahead serving

Armory (Bansal et al., arXiv:2608.00337) is a direct industry-relevant comparator for remote robot inference serving. It maintains a server-side mirror of each robot's action queue, in-flight chunks, execution horizon and communication delay. It compares Round Robin, EDF and a Lookahead policy that simulates candidate batch effects. On a real fleet of ten AgileX PiPER arms served by a single cloud L40S GPU, the authors report up to 18% overall throughput improvement in their heterogeneous workload. The dynamic fast task is more starvation-sensitive than static sorting. This is a preprint/benchmark reference; don't present its numbers as Luminous results or assume quadruped transfer.

- Project / method / real-robot demo: https://actionchunkscheduling.github.io/
- Paper: https://arxiv.org/abs/2608.00337

### RED: real-time inference DAG scheduling

RED (Li et al., arXiv:2605.24044) proposes adaptive real-time DAG scheduling for robotic multi-task inference under environmental dynamics, with intermediate sub-deadlines and workload-graph reconstruction. The authors report evaluations on NVIDIA Jetson platforms and Apple M-series hardware. Treat its claimed end-to-end timing guarantees as conditional on the scheduler's modeling assumptions; it is a scheduling baseline, not a certified runtime-assurance authority.

- Paper: https://arxiv.org/abs/2605.24044

### Luminous comparison

For fleet-shared inference, require:
1. Round Robin;
2. EDF;
3. Armory Lookahead;
4. RED-style DAG/sub-deadline scheduling where workloads match;
5. Luminous constrained mission-aware admission / batching.

The candidate Luminous scheduler must consume existing qualified action-buffer, actuation-AoI, clock/freshness, EVSI, thermal, actuator and recovery evidence. It may schedule optional deliberation, not safety control. Compare matched workloads and model quality, and disclose model/hardware differences.

Required metrics: throughput and safe-success by robot tier; per-robot starvation/control-step no-action rate; p50/p95/p99 request-to-actuator latency; actuation deadline miss probability and miss cost; fairness/service-floor violations; queue wait; useful result rate (fresh, decision-positive, locally admissible); energy by robot/network/edge/cloud; and recovery degradation. A fleet average cannot conceal a single robot that starves or loses its safe-stop path.


## New actuation and fleet-serving evidence

- **Age of Actuation / Probability of Missing Actuation (PoMA):** Nikkhah, Ephremides and Pappas, IEEE Transactions on Communications (2026), DOI 10.1109/TCOMM.2026.3663227. The paper studies timely use of status updates, AoA, violation probability and missed-actuation cost in a wireless-power-transfer model. This is a conceptual comparator; robot semantics require adaptation.
- **Armory:** Bansal et al., arXiv:2608.00337. A cloud policy-serving system mirrors robot action queues and compares Round Robin, EDF and Lookahead. Its 10-PiPER / single-L40S heterogeneous fleet reports up to 18% higher total throughput in its tested workload.
- **RED:** Li et al., arXiv:2605.24044. Adaptive deadline-aware DAG scheduling for multi-task robotic inference, evaluated on Jetson and Apple M-series platforms under the authors' scheduling assumptions.

The Luminous target is to compare these methods under matched robot/task/model/network/compute conditions and go further by incorporating qualified action-buffer depletion, actuation-side AoI, PoMA-style deadline misses, EVSI, thermal/actuator state and recovery constraints. Keep hard local safety and safe-stop outside the shared scheduler.


The mission-aware fleet-serving child [#11627](https://github.com/Luminous-Dynamics/symthaea/issues/11627) / [draft PR #11630](https://github.com/Luminous-Dynamics/symthaea/pull/11630) is now included as a distinct constrained admission benchmark.


### Decision-sufficiency horizon as a new control bound

A fixed freshness threshold is useful but does not tell us whether the current information is still decision-sufficient. The new [decision-sufficiency protocol](./decision_sufficiency_horizon.md) adds a calibrated upper bound on stale-action regret and a robust common-admissibility requirement. It is a separate bound in the adaptive commitment horizon: stable old information may remain sufficient, while very recent information may already be inadequate after a fast state transition.
