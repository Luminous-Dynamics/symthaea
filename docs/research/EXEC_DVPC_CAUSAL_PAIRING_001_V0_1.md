# DVPC Causal Pairing Qualification (v0.1)

**Tracking:** Symthaea #13375 — EXEC-DVPC-CAUSAL-PAIRING-001  
**Paired decision-regret evaluator:** #13281  
**Tournament protocol:** #13282 / #13216  
**Generic causal replay contract:** #8764  
**Driver-owned information/replay:** #8732  
**Execution/replay owner:** Symtropy #3387  
**Evaluator isolation:** #9437  
**Status:** finite Rust reference authored; no physical simulator or external benchmark result is claimed.

## 1. Research question

When compute and observation treatments change the agent's actions, do the four paired benchmark arms remain valid counterfactual comparisons after their resulting world states and future observations diverge?

The pairing must hold constant what is legitimately shared: the episode's initial state, registered dynamics and simulator build, timebase, task/loss contract, resource envelope, and exogenous disturbance schedule. It must not force action-dependent state trajectories to remain identical.

This is a missing bridge between the 13-field identity check in #13281 and the dynamic-world application-time regret estimand in #13282. It is not a new simulator, causal-inference engine, hidden-world generator, physics model or policy.

## 2. Three evaluation regimes

The regime is explicit for every four-arm experiment and is never inferred from matching digests.

### OPEN_LOOP_ACTION_INVARIANT

Policy actions do not affect the source world stream. A shared prerecorded physical trajectory is valid. The four arms must have equal world-tick coverage and matching pre/post state and exogenous-input commitments at each tick. The observation-delivery transcript may differ only as declared by the sensing treatment.

This regime is appropriate for perception or prediction tasks where the policy is not coupled back into the process that generates the future stream.

### CLOSED_LOOP_ACTION_COUPLED

Policy actions can affect future world state, actor response, observation availability or reward. Every arm must have an arm-local driver execution and an independent replay-audit receipt. The arms share the initial state, dynamics/simulator/timebase roots, exogenous forcing and randomness-domain commitments; their endogenous states and observation transcripts may diverge.

A common prerecorded future trajectory is rejected. Common random numbers are keyed to stable event/tick/channel identity (or an independently frozen equivalent schedule). Sequential PRNG consumption is insufficient because branch-specific extra draws can shift all later disturbances.

Equal trajectories are allowed in a genuine null/action-invariant subcase. Divergence is not itself evidence of causal correctness.

### OFFLINE_LOGGED_POLICY_EVALUATION

This is a separate estimand. The simulation trajectory validator refuses to label logged-policy estimates as closed-loop execution or action-invariant replay. Offline evaluation needs its own support/positivity, effective-sample-size, missingness and identification assumptions.

## 3. Contract and certificate

The Rust reference is in:

`research/decision-value-interaction-001/src/causal_pairing.rs`

The common identity binds:
- the existing 13-field `PairedRunIdentity`;
- dynamics-model and simulator-build digests;
- exogenous-schedule and clock-profile digests;
- initial world tick;
- distinct world-disturbance, policy-RNG, sensor-noise and evaluator-RNG domain commitments.

Each arm binds:
- one of the four treatment arms;
- declared causal regime and trajectory origin;
- a contiguous sequence of integer-tick transitions;
- pre-state, applied-action, exogenous-input, post-state and observation-delivery digests per tick;
- decision time and action-application time;
- delivered evidence event IDs with availability and delivery times;
- the evidence IDs consumed by each decision;
- an explicit scored-outcome binding;
- a driver-execution receipt and an independent replay-audit receipt;
- explicit complete/incomplete and replay-audit disposition.

Validation enforces:
1. exactly one receipt per treatment arm;
2. matching shared roots and the original 13 paired identities;
3. distinct, non-empty RNG-domain commitments;
4. common initial state and contiguous tick/state-chain commitments;
5. common exogenous event-bundle commitments on all overlapping ticks;
6. no conflicting post-state for an identical tick/pre-state/action/exogenous input tuple;
7. a shared physical trajectory for open-loop action-invariant arms;
8. arm-local driver execution for closed-loop action-coupled arms;
9. decision-time evidence filtration: every consumed ID was delivered to that arm by the decision time and was already available at delivery;
10. monotonic decision clocks and action application no earlier than the associated decision;
11. exact binding between the causal trajectory and the decision-regret outcome.

### Binding the causal trajectory to the scored outcome

A valid trajectory and a valid regret vector are not sufficient if they refer to different actions. The `ScoredOutcomeBinding` closes this join:

- `Applied`: binds a world tick, the exact action digest and its application timestamp. All three must match a step in that arm's trajectory.
- `NotApplied`: binds the task-defined terminal timestamp and explicit reason. It must not precede the final decision time.
- `validate_causal_pairing_with_interaction`: evaluates the existing #13281 four-arm regret outcomes, validates the four causal trajectories, verifies the complete 13-field identity, and matches every arm's applied/terminal disposition to the corresponding trajectory binding.

It returns a combined `CausalDecisionValueCertificate` containing both the causal pairing certificate and the decision-value interaction certificate. This is a join adapter, not a new regret formula or causal inference system.


## 4. Decision-time information filtration

Trajectory lineage is not sufficient if a policy can consume an event before the driver actually made it available. The evaluator must distinguish source availability, driver delivery and decision time.

Each trajectory step therefore includes:
- `decision_at_us`: the driver-clock time when the policy committed the decision/proposal;
- `action_applied_at_us`: the time the consumer applied that proposal;
- `delivered_evidence`: stable event IDs with source-availability and arm-delivery times;
- `consumed_evidence_ids`: the event IDs the policy claims to have used for that decision.

For each arm and decision time (t), the required invariant is:

\[
\operatorname{Consumed}(a,t)\subseteq
\operatorname{DeliveredTo}(a,\leq t)
\]

and for every delivered event (e):

\[
t_{available}(e)\leq t_{delivered}(e)\leq t_{decision}.
\]

The validator also requires monotonic decision-clock time, action application no earlier than its decision, non-empty event identities, no duplicate event deliveries within one arm, and no duplicate evidence references within a single decision. Evidence delivered at an earlier decision may legitimately be reused later; each use remains explicit.

This catches a simple but dangerous failure mode: an arm may be paired to the correct initial world and exogenous schedule, yet use a future observation or label to choose an earlier action. The four arms may have different delivered event sets because the observation treatment changes what each arm acquires; each arm is checked against its own delivered filtration.

The event IDs and timestamp claims are still supplied by the driver. Canonical typed event bytes, content hashes, hash-chain construction, oracle isolation and signed receipts remain with the generic receipt/security owners (#9437 and #8715). This module checks the decision-time semantics of those references. A valid event ID alone is not proof that its event payload was honestly acquired.

The reference reports the number of validated decision steps and consumed evidence references across all four arms. It does not execute candidate policies. Driver-level future-prefix invariance still requires running the same policy on two worlds with identical prefixes but different future suffixes, then proving that actions up to the prefix horizon are identical.

## 5. Why common random numbers need event identity

Counterfactual variance reduction should align exogenous disturbances across arms without synchronizing endogenous branches. Consider two arms where one policy requests an extra sensor read or performs more internal steps. If both consume a single sequential random-number stream, the extra draw shifts all subsequent noise. The same "seed" then fails to represent the same exogenous disturbance at a later world tick.

The contract therefore binds separate seed domains and a common exogenous schedule, with the per-tick exogenous input committed independently. The execution driver must assign disturbances by stable event/tick/channel identifiers (or a frozen indexed schedule) so different branch consumption cannot shift unrelated events. Sensor-noise domains remain separate from world disturbances; a treatment can legitimately request a different set of sensor events.

The validator checks the commitments presented by the driver. It cannot prove the driver keyed its random stream correctly unless the independent replay auditor actually reconstructs that mapping from the retained event tape.

## 6. Falsification matrix

| Control | Expected result |
|---|---|
| Closed-loop arms share initial state and exogenous schedule but have action-conditioned state divergence | Accept if each arm has its own complete execution and replay audit |
| Closed-loop arms all have identical trajectories in an action-invariant/null case | Accept |
| Closed-loop arm cites only a shared prerecorded future trajectory | Reject |
| Open-loop action-invariant arms share the physical world trajectory | Accept |
| Open-loop arms have different physical state trajectories | Reject |
| Any overlapping tick has a different exogenous event bundle | Reject |
| Same tick/pre-state/action/exogenous input produces different next state | Reject |
| A future tick is missing or a state-chain link is broken | Reject |
| Initial state, dynamics, simulator build, clock or budget identity differs | Reject |
| Policy/world/sensor/evaluator RNG domain commitments are reused | Reject |
| An arm lacks a complete driver receipt or passing independent replay audit | Reject |
| Offline logged-policy estimates are submitted as simulator trajectory receipts | Reject |

The source test suite contains positive and negative controls for these cases, plus ten decision-time filtration controls covering prior-evidence reuse, future-event consumption, late delivery, delivery-before-availability, duplicate delivery/consumption, clock regression, action-before-decision and empty event identities. The driver must separately test future-action prefix invariance and hidden-state isolation; a static receipt validator cannot establish those dynamic properties by itself.

## 7. Integration and ownership

- #8764 owns the general open-loop/action-coupled causal replay distinction.
- #8732 owns driver-owned clocks, per-tick delivered evidence and decision-time information views.
- Symtropy #3387 owns scenario execution/replay; #3471/#3502 own action-conditioned actor response and its execution adapter where relevant.
- #13281 owns four-arm regret arithmetic and the cluster adapter.
- #13282 owns the compute×observation experiment design.
- #8578 owns independent-cluster inference.
- #9437 remains the process/OS evaluator-isolation boundary.

No new transition simulator, actor policy, resource ledger, factorial decomposition, VOI engine, freshness horizon, statistical engine or action authority is introduced.

## 9. Benchmark sequence and claim ceiling

First run the finite controls, then integrate with the qualified RoboDyna driver. Reproduce the official π0.5, X-VLA and FastWAM baselines under the benchmark's own splits, checkpoints, timing modes and resource protocol before adding Symthaea as a separately labelled decision-layer arm. Keep the open-loop DEEP perception reproduction separate from action-coupled RoboDyna results.

Report separately:
- application-time regret and signed compute×observation interaction;
- task success and progress;
- first trajectory divergence and recovery cost;
- paired sync→async success/failure flips;
- p50/p95/p99 end-to-end and stage latency;
- measured energy and the non-fungible resource vector;
- deadline, stale-action, missing-output and hard-budget failures;
- per-scene/sequence cluster uncertainty, worst-family and tail outcomes.

A passing Rust validator establishes only finite structural pairing and receipt consistency over supplied commitments. It does not validate the physics, prove that a stochastic schedule is complete, identify unsupported counterfactuals from logged data, demonstrate a better policy, establish physical safety, or show a SOTA improvement. Those require independently replayed driver traces and held-out benchmark results.

## 8. Competitive baseline crosswalk and experiment order

The most important new gap is prediction-to-action advantage, not prediction quality in isolation. A 2026 survey of 160 benchmarks reports only 11 with an explicit VLA-versus-world-model contrast and only four that take prediction through to executed action. This is a reported survey classification, not a universal census; it nevertheless identifies a concrete measurement gap. The campaign should therefore compare a direct reactive/VLA policy and a predictive/world-action policy in the same closed-loop environment, then measure whether prediction produces a positive incremental task/decision advantage at matched resources.

### Two complementary primary testbeds

| Testbed | What it contributes | Baselines and reported context | Symthaea's experiment |
|---|---|---|---|
| RoboDyna (NeurIPS 2026 Workshop) | Broad physical dynamics: 30 tasks, five capabilities, compositional difficulty, household scenes, paired sync/async execution and human baselines. | π0.5 18%, X-VLA 11%, FastWAM 6% overall success in the official summary. Async timing flips 5–12% of episode outcomes while changing average success by at most 3 points. | Reproduce official baselines; compare direct VLA, FastWAM/world-action, and each frozen policy with the Symthaea decision layer. |
| LIBERO-Dynamic (2026) | Controlled latency-aware extension of all 40 LIBERO tasks, three motion families, three target speeds, 1,800 fixed episodes/configuration and calibrated inference delay while the scene moves. | Async best-measured-horizon pretrained policies: π0.5 (JAX) 57.83%, FastWAM-Joint 40.61%, VLA-JEPA 35.72%, FastWAM 32.06%, π0 (JAX) 31.50%, OpenVLA-OFT 27.28%. Strategy table: Async Queue 48.39%; measured-latency Sync Hold 43.06%; RTC 37.83%; Sync Zero Latency 50.39% (optimistic diagnostic only); separately trained VLASH 55.44% (system reference, not controller ablation). | Reproduce official checkpoints, fixed 1,800 episode population, latency calibration, action-horizon sweeps, queue underflow and the full five-row strategy comparison without altering the official baseline protocol. |

RoboDyna and LIBERO-Dynamic are not one leaderboard: task sets, action interfaces, motion distributions, checkpoints, training budgets and metrics differ. Report each separately. RoboDyna is the broader physical-competence stress test; LIBERO-Dynamic provides a more controlled latency/motion experiment. Their human baselines and real-robot sub-studies must be retained where compatible. In LIBERO-Dynamic, the zero-latency sync row is an optimistic diagnostic control, not a deployable policy; measured-latency Sync Hold must remain a separate row. VLASH is separately trained and cannot isolate a controller-only effect.

The actual benchmark output schemas, seed/step-limit files, launcher revisions and official result CSV blob hashes are pinned in [PREDICTION-ADVANTAGE-UPSTREAM-PINS-v0.1](PREDICTION-ADVANTAGE-UPSTREAM-PINS-V0_1.md) and the accompanying [machine-readable manifest](../../research/decision-value-interaction-001/fixtures/prediction_advantage_upstream_pins_v0.1.json). The formal LIBERO runtime source is separately declared in the upstream reproducibility notes but could not be fetched from the connected GitHub source API; it remains an unresolved reproduction dependency.

**Fairness boundary:** a direct VLA and a world-action model from different architectures/checkpoints are an end-to-end system comparison, not a causal ablation proving that prediction itself caused an advantage. To claim incremental predictive value, require a matched same-backbone/same-data/same-training-budget ablation with the predictive rollout/planner enabled versus disabled, or label the comparison descriptive. To measure the Symthaea increment, hold the underlying policy checkpoint fixed and compare P with P+Symthaea under the same seed, observation stream, action domain, timing and resource envelope.

### Outer and inner pairing are different contracts

The benchmark has two levels of pairing, and they must not be conflated.

**Outer policy-family pairing** compares a direct VLA/reactive policy with a predictive/world-action policy, or compares a fixed policy (P) with (P+\text{Symthaea}). Both sides must share a benchmark episode/world key: benchmark revision, task/condition, split, seed/episode, initial state, dynamics/simulator build, clock profile, exogenous disturbance schedule, compatible observation/action semantics, runtime profile and resource/deadline envelope. Policy family, checkpoint, model configuration and augmentation identity are explicit contrast factors and are expected to differ.

The outer key does not force identical future state trajectories. In an action-coupled world, the two policy variants run their own actions and create their own endogenous futures under a common exogenous forcing commitment. Each gets its own arm-local trajectory and evidence filtration under #13375/#13377. If action domains, observations, clocks or resource ceilings cannot be matched, mark the contrast NOT_COMPARABLE.

**Inner compute×observation pairing** happens separately inside each fixed policy variant. Its four arms must share the complete thirteen-field #13281 identity—including candidate policy and model-configuration digest—because only the compute and observation treatments change. It is invalid to use the full inner 13-field equality as the outer policy-contrast key, since the policy digest is intentionally different between outer arms.

A comparison between released π0.5 and FastWAM checkpoints is a useful **end-to-end system comparison**, but not a causal ablation that isolates prediction: architecture, data, training recipe, checkpoint and execution horizon may all differ. To make a predictive-component claim, use the same backbone/checkpoint, data and training budget with a predictive module enabled versus disabled, or label the result descriptive. To measure Symthaea's incremental decision-layer value, hold the underlying policy fixed and compare (P) versus (P+\text{Symthaea}) at the same seed, world key, observations, action domain, timing and resource envelope.

Tracking work package: [#13389 — matched direct-policy vs world-model tournament](https://github.com/Luminous-Dynamics/symthaea/issues/13389).

### Matched prediction-advantage design

Within each benchmark, run these policy families with frozen checkpoints and matched input/action interfaces:

1. **Direct policy:** reactive VLA / direct policy without a predictive rollout.
2. **Predictive policy:** world-action model or world-model-plus-planner.
3. **Decision-layer direct:** the same direct policy with Symthaea's evidence/compute/observation allocation layer.
4. **Decision-layer predictive:** the same predictive policy with the same Symthaea decision layer.
5. **Execution controls:** fixed horizon, official asynchronous queue, RTC, progress-triggered execution (TempoWAM-style), and the benchmark's official delay-aware reference where available.
6. **Evaluator-only oracle:** finite or hindsight diagnostic only; never a deployable ranking baseline.

Do not train the direct and predictive arms on different data unless the comparison is explicitly labelled as an end-to-end system comparison. Prefer a frozen-backbone comparison first. Keep the official benchmark's own released checkpoints and scoring untouched; the Symthaea layer is an additional adapter/arm, not a silent modification to the baseline.

The key within-task comparison is:

\[
\Delta_{\text{prediction}}=U(\text{predictive policy})-U(\text{direct policy})
\]

and the decision-layer increment is:

\[
\Delta_{\text{Symthaea}|P}=U(P+\text{Symthaea})-U(P)
\]

where \(P\) denotes the same frozen underlying policy family. These are task-utility contrasts, not a new scalar model score. Report task success, progress, regret, latency, queue underflow, resources and failures separately. A predictive model can win on open-loop prediction error and still lose this closed-loop contrast.

For the compute×observation study, run the four #13281 arms separately within each frozen policy family. This preserves the existing paired arithmetic and avoids expanding the benchmark into a new, unreviewed eight-arm or multi-factor attribution engine. #13176 owns its existing resource/exposure factorial decomposition; #8578 owns independent-cluster inference.

### LIBERO-Dynamic reproduction caveats

- Use its official 1,800-episode population per configuration and official task/motion/speed assignments.
- Reproduce calibrated, model-specific inference latency; do not replace it with nominal FLOPs or a guessed average.
- Preserve action horizon \(K\), startup underflow, steady queue underflow, and actual executed action count.
- Separate official π0.5, FastWAM-Joint, VLA-JEPA, FastWAM, π0 and OpenVLA-OFT checkpoint rows.
- Keep the VLASH comparison separately labelled because it is a separately trained delay-aware system, not a pure controller-only comparison.
- Treat reported best measured \(K\) as descriptive unless the same validation-selection procedure is reproduced. Do not choose \(K\) on confirmatory episodes.
- Keep physical-robot results separate from simulation results; report the embodiment, task, control horizon and hardware conditions.

### RoboDyna reproduction caveats

- Pin the official repository commit, runtime assets, SAPIEN environment, checkpoints, language instructions, evaluation seeds and task step limits.
- Reproduce the official \`base\`, \`var1\`, \`var2\` and held-out \`var1+2\` task conditions.
- Retain paired sync and async outcomes by seed, plus official task success and progress score.
- Report sync-success→async-failure, sync-failure→async-success and unchanged episode counts separately; mean success differences do not substitute for episode transition counts.
- Preserve household and human-baseline slices; do not pool unlike slices into one unsupported claim.

### New field-level metrics

Report an **advantage-of-prediction** table by task capability:
- closed-loop utility gain over the matched direct policy;
- action regret at actual application time;
- prediction calibration / error on the same held-out transitions;
- rate of harmful predictive interventions (prediction changed the action and increased loss);
- counterfactual evidence-consistency rate under matched interventions;
- recovery quality after model error or execution deviation;
- full latency, energy and memory costs of predicting, planning, sensing and verifying.

This ties prediction fidelity to executed utility without assuming the two correlate. It also makes model exploitation visible: test the policy against deliberately perturbed or adversarially selected world-model rollouts, then verify in the authoritative environment rather than allowing the learned model to grade its own policy.


### Additional prior-art comparison

The closest public results measure different parts of the problem. They should be treated as distinct baselines, not collapsed into one leaderboard score.

| Prior work / benchmark | What it already establishes | Symthaea comparison that remains open |
|---|---|---|
| DEEP, *The Decision Value of Perception Compute* (2026-09-28) | Perception escalation can worsen downstream decisions; perception-quality gains need not produce decision-value gains. | Reproduce the official perception-cost / downstream-loss comparison, then add paired observation-only, compute-only and joint arms with application-time regret. |
| Online policy-wise state abstraction (IEEE TNNLS, 2026-09-30) | Online task-relevant abstraction with parallel learners and free-energy-guided selection is prior art. | Test temporal decision sufficiency of compressed recurrent state across future observations and regimes; measure actual memory traffic and future regret, not just abstraction quality. |
| *How Can Driving World Models Do Counterfactual Prediction?* (2026-08-12) | Direct action-conditioned prediction can miss factual continuation evidence; the paper formalizes abduction–action–prediction in a short-horizon setup where other agents are not action-coupled. | Require arm-local action-coupled trajectories when the environment responds to actions, and separately audit factual evidence transfer and counterfactual replay. |
| TempoWAM (2026-08-10) | Progress-monitored adaptive action-chunk execution is prior art; its paper reports 26.9% fewer WAM inferences on easy real-world tasks while maintaining success and a 13.3-point success increase on difficult real-world tasks. | Include fixed-horizon, progress-triggered and TempoWAM-style execution baselines. Symthaea must show added value from joint observation/compute/verification and temporal evidence, not merely replan timing. |
| RoboDyna (NeurIPS 2026 Workshop; arXiv 2026-11-20) | 30 physically grounded dynamic tasks, synchronous/asynchronous paired execution, task success and progress. Official results report π0.5 18%, X-VLA 11%, and FastWAM 6% overall success; sync-to-async execution flips 5–12% of episode outcomes while average success changes by at most 3 percentage points. | Reproduce the exact checkpoints, seeds, task splits and sync/async protocol first. Then add Symthaea as a separately labelled decision layer and report per-seed flips, progress and application-time regret. |
| LIBERO-Dynamic (2026) | Latency-aware evaluation across all 40 LIBERO tasks, with contact-driven motion, calibrated inference delay and asynchronous queue execution. | Official fixed-1,800-episode async results range from π0.5 (JAX) 57.8% to OpenVLA-OFT 27.3%; the reported best measured Async Queue strategy is 48.4%, Sync Hold 43.1%, RTC 37.8%, and separately trained VLASH 55.4%. | Use as the controlled latency/motion counterpart to RoboDyna. Reproduce its checkpoints and queue semantics, then evaluate the Symthaea layer on the same episode population. Keep VLASH as a separately trained system reference. |
| RoboRecover (2026-09-24) | Initial-state policy performance does not determine recovery after execution-induced deviations; its dataset contains 2,000 scenarios across RoboTwin and LIBERO. | Add a recovery arm after controlled deviations. Measure recovery success, time-to-recovery, extra actions, damage/regret and tail failures at matched deviation states. |

### Recommended campaign

1. **Reproduction before augmentation.** Reproduce DEEP on its official KITTI, nuScenes and open-loop nuPlan settings; reproduce RoboDyna's π0.5, X-VLA and FastWAM results; and reproduce LIBERO-Dynamic's released pretrained-policy and execution-strategy baselines. Freeze checkpoints, versions, seeds, task splits, clock mode, latency calibration and resource ceilings.
2. **Decision-layer isolation.** Keep each foundation policy frozen. On each compatible benchmark, compare the base policy, fixed-budget escalation, random allocation, confidence/entropy routing, DEEP-style decision-value routing, TempoWAM-style progress routing, and Symthaea. This isolates the value of the decision layer from model-scale or task-specific fine-tuning effects.
3. **Factorial attribution.** Use #13281 for compute×observation regret; use #13176 for its existing resource/exposure decomposition; use #8578 for independent-cluster uncertainty. Add memory and verification only through their existing owners after the four-arm campaign is stable.
4. **Closed-loop branch correctness.** On action-coupled tasks, the four arms share the initial state and indexed exogenous forcing, not a forced common future state trajectory. Use arm-local execution, action-conditioned actor response and independent replay. Keep open-loop DEEP results separate from closed-loop RoboDyna/RoboRecover results.
5. **Recovery and temporal integrity.** After baseline reproduction, inject controlled execution deviations and evaluate recovery. Separately compare full state, identity, PCA, random projection, quantization and decision-aware recurrent-state compression across future decision epochs.

### Pre-registered success criteria

- **Decision-value track:** retain the existing #8572 targets: at least 5% relative downstream-loss reduction with a 95% interval excluding zero, harmful-escalation upper confidence bound no greater than 5%, and zero hard budget violations. Treat the existing 20% lower measured resource-cost objective at a fixed non-inferior regret margin as a target, not an observed result.
- **Embodied track:** first reproduce official baseline intervals. Then require a positive held-out decision-regret or progress improvement over the strongest matched baseline, with independent-cluster uncertainty excluding zero; also reduce sync-to-async success-to-failure flips without increasing hard safety/budget failures.
- **Recovery track:** improve recovery completion and time-to-recovery over matched deviation-state baselines, and report the worst task families and tail outcomes rather than only the mean.
- **Compression track:** zero false preservation certificates on exhaustive finite controls, one held-out workload with measured memory reduction and non-inferior future decision regret, and one negative control where lossy compression is rejected.

Do not use a composite score to hide trade-offs. Report decision regret, success, progress, recovery, first-divergence time, p50/p95/p99 latency, measured energy, memory footprint, sensor cost, and hard-budget violations as separate dimensions. Five clusters are only the mechanics floor in the current analyzer; a real campaign needs a preregistered power/MDE analysis and enough independently sampled scenes or sessions.

## References

- [Symthaea #13375](https://github.com/Luminous-Dynamics/symthaea/issues/13375)
- [Symthaea #8764](https://github.com/Luminous-Dynamics/symthaea/issues/8764)
- [Symthaea #8732](https://github.com/Luminous-Dynamics/symthaea/issues/8732)
- [Symthaea PR #13281](https://github.com/Luminous-Dynamics/symthaea/pull/13281)
- [Symthaea PR #13282](https://github.com/Luminous-Dynamics/symthaea/pull/13282)
- [RoboDyna](https://robodyna.github.io/robodyna/)
- [LIBERO-Dynamic paper](https://arxiv.org/abs/2610.24579)
- [LIBERO-Dynamic code and evaluation protocol](https://github.com/renshaojie233/LIBERO-Dynamic)
- [Do World Models Make Better Robots?](https://arxiv.org/abs/2609.29669)
- [DEEP](https://arxiv.org/abs/2609.35910)
- [How Can Driving World Models Do Counterfactual Prediction?](https://arxiv.org/abs/2608.11601)
- [TempoWAM: Rethink Before You Execute](https://arxiv.org/abs/2608.09492)
- [RoboRecover](https://arxiv.org/abs/2609.28952)
- [RoboDyna paper](https://arxiv.org/abs/2611.26969)
- [Online Policy-Wise State Abstraction for Reinforcement Learning Agents Leveraging Free Energy](https://pubmed.ncbi.nlm.nih.gov/?term=Online+Policy-Wise+State+Abstraction+for+Reinforcement+Learning+Agents+Leveraging+Free+Energy)
