# Decision-Value Test-Time Computation Benchmark (v0.1)

**Tracking:** #11561 — real-time cognition under dynamic embodied tasks  
**Temporal execution mechanism:** #13097 / PR #13098 — transactional solver continuation  
**Work and freshness evidence:** #13061/#13062 and #13091/#13093, joined by #13142  
**Decision oracle:** #13152; mixed-strategy extension #13165  
**Active sensing:** #9563; expected observation value #12567/#12551; robust support bound #13202  
**Joint resource allocation:** #9666/#9882/#13176  
**Probability calibration:** #11367  
**Evaluator isolation:** #9437  
**Status:** protocol and synthetic known-answer manifest authored; no executable candidate controller or benchmark result is claimed.

## 1. Research question

When should a system spend another unit of computation, and when should it stop, act, yield, request an observation, switch method, or abstain?

The benchmark evaluates whether a candidate can allocate additional computation according to its **predicted future decision improvement**, rather than treating current confidence, entropy, answer stability, forecast accuracy, or a fixed inference depth as sufficient stopping criteria.

The measured outcome is the decision made at the state and time where the action is actually applied, after charging elapsed physical time, observation age, actuation/consumer delay, compute and memory resources. Improving an internal prediction is not a win if that improvement cannot change the action, misses its deadline, or makes the action stale.

This is a benchmark/protocol adapter under #11561. It does not create a second real-time scheduler, solver, continuation token, action authority, confidence calibrator, work ledger, or decision-loss oracle.

## 2. Paradigm shift: computation is an action with opportunity cost

A computation step is not free introspection. In a dynamic environment it consumes physical time and resources while the plant continues to evolve.

At a candidate-visible checkpoint \(c_t\), the system can choose among:
- publish the current admissible action;
- spend another bounded compute block;
- switch to a different qualified solver or reasoning path;
- yield and resume later at a qualified continuation safe point;
- request new evidence through the existing active-sensing interface;
- defer, safe-stop or abstain under the task contract.

The benchmark must distinguish four outcomes that are often conflated:

1. **Current answer confidence:** what the candidate believes now.
2. **Future value of computation:** whether another compute block is likely to improve the qualified decision.
3. **Physical opportunity cost:** what changes in the world, validity window and available action set while the block runs.
4. **Execution mechanism:** early exit, operator preemption, temporal solver yield/resume, and cancel/restart are different mechanisms and receive separate labels.

The target is not minimum compute. It is the best qualified decision under a fixed resource envelope.

## 3. Decision-value definition

Let \(W_t\) be the evaluator's finite set of plausible worlds at checkpoint \(t\), \(A_t(w)\) the actions admissible in world \(w\) at actual action-application time, and \(L(w,a,t_a)\) the evaluator-owned loss at application time \(t_a\).

For a candidate-visible information set \(I_t\), define the evaluator's decision regret floor:

\[
R(I_t,t_a)=\min_{a\in A(I_t,t_a)}
\max_{w\in W(I_t)}\left[
L(w,a,t_a)-\min_{a'\in A_t(w)}L(w,a',t_a)
\right].
\]

The exact finite-world calculation composes #13152; randomized policies compose #13165. The candidate does not receive \(W_t\), hidden losses, future checkpoints, or evaluator bounds.

For compute allocation, compare two policies over the same initial episode commitment:

- \(a_0\): the action produced by the current checkpoint and applied through the real end-to-end path;
- \(a_c\): the action produced after an additional compute allocation \(c\), applied through its real end-to-end path.

The evaluator reports the regret contrast at the respective application states:

\[
\Delta R(c)=R_{realized}(a_0,t_{a0})-R_{realized}(a_c,t_{ac}).
\]

This is an episode-level outcome contrast, not a candidate-visible oracle signal. The candidate may predict \(\widehat{\Delta R}(c)\) from current authorized information, but it must not see the realized future value before selecting the compute action.

The benchmark keeps the following in separate dimensions:
- realized decision regret / task loss;
- current and post-compute information-set regret bounds;
- probability-calibrated expected decision improvement, when available from #11367/#12551;
- elapsed wall time and deadline reserve;
- measured compute energy, memory traffic and checkpoint overhead;
- observation freshness and action-application age;
- thermal / battery / physical reserve and hard safety feasibility.

Do not add unlike resource units to one arbitrary scalar. Hard deadlines, admissibility, safety, authority and viability are non-compensatory. A utility-weighted objective is permitted only when its weights and conversion units are declared by the application before evaluation.

## 4. Critical research basis and baseline boundary

Recent work already studies adaptive test-time compute. Symthaea should reproduce compatible baselines and position its contribution accurately.

### AERA — Adaptive Evidence Residual Allocation (2026)

AERA argues that current confidence or answer stability can be misleading because checkpoint correctness can change non-monotonically. It learns whether another generation block is likely to recover a better answer from checkpoint-observable evidence; future correctness is used only for offline supervision, not at inference. Its frozen-threshold incremental-generation experiment reports 92.61% accuracy versus 93.01% for 128 responses on 300 untouched GSM8K questions, with 95.99% fewer completion tokens. These are paper-reported results in that specific language-reasoning protocol, not a Symthaea baseline result or physical-control claim.

- Paper: https://arxiv.org/abs/2608.27964
- Required comparator: reproduce its published checkpoint features/training protocol where compatible; otherwise mark NOT_COMPARABLE and implement a clearly labelled adaptation.

### Constrained test-time compute allocation (2026)

The April 2026 constrained-policy-optimization work frames compute allocation under an average compute budget, derives a Lagrangian oracle and trains a light policy to imitate oracle allocations. It reports up to 12.8% relative accuracy improvement on MATH under matched compute in its experimental setting. This is a useful comparator for budget-constrained compute policy learning, but task accuracy is not the same as dynamic decision regret at action application.

- Paper: https://arxiv.org/abs/2604.14853

### CATTS and agentic meta-reasoning

CATTS uses confidence-aware allocation for web agents; its Berkeley 2026 report compares adaptive allocation against uniform scaling and reports improvements up to 11.8% on the evaluated web-agent tasks. “Thinking Before Thinking” (September 2026) makes controller-level choices—reuse, restart, delegate, stop and budget allocation—an explicit agentic meta-reasoning process. These are relevant design baselines for agent workloads, not direct physical-control scores.

- CATTS technical report: https://www2.eecs.berkeley.edu/Pubs/TechRpts/2026/EECS-2026-75.pdf
- Meta-reasoning paper: https://arxiv.org/abs/2609.38147
- ACL 2026 Pareto test-time scaling: https://aclanthology.org/2026.acl-srw.1/

### Existing temporal and physical execution baselines

- #13097 / PR #13098 owns transactional solver continuation at committed substep safe points.
- #11561 owns world-keeps-moving, deadline, compute contention and action-application consequence evaluation.
- #13062/#13140 owns source-bound HDC-LTC work instrumentation.
- #13061 owns the temporal work ledger; #13091/#13093 own correctness-at-consumption and freshness; #13142 joins the receipts.
- #13032 owns deterministic irregular input reference traces and numerical work sweeps.
- #9563 owns active sensing; do not implement another sensing policy here.
- #13152/#13165 own deterministic/mixed decision-observability and regret bounds.
- #12567/#12551 own information-channel integrity and expected posterior Bayes-risk/VOI.
- #13202 adds the complementary robust support-regret and epsilon-information-set-sufficiency score; it does not measure compute value by itself.

## 5. Candidate policy arms

All candidate arms receive the same initial observation transcript, model/task contract, resource ceilings and action deadline. The candidate may see its own checkpoints, but never future labels, the hidden trajectory, evaluator oracle outputs, held-out continuation outcomes or the evaluator's exact regret.

Required arms:

1. **Fixed compute:** fixed 4/8/16/32 solver/RHS work units where the method has a comparable work unit, plus fixed wall-clock budgets.
2. **Fixed-depth reasoning:** a frozen number of reasoning/solver iterations.
3. **Uniform continuation:** spend the available compute budget in equal blocks independent of task/checkpoint.
4. **Confidence stop:** stop on a frozen confidence threshold.
5. **Entropy / margin stop:** stop on answer/action entropy or top-1/top-2 margin thresholds.
6. **Stability stop:** stop when consecutive checkpoints agree for a preregistered count.
7. **AERA-style adaptive compute:** reproduce or clearly adapt checkpoint-evidence-based residual allocation.
8. **Constrained budget allocation:** reproduce the compatible Lagrangian / solve-then-learn policy.
9. **Candidate value-of-computation policy:** predicts decision improvement from authorized checkpoint evidence and allocates compute under the vector budget.
10. **Direct fast policy:** no additional world-model/reasoning query; an essential lower-cost comparator.
11. **Evaluator-only finite continuation oracle:** upper-bound/reference only, never candidate-visible.

For agentic workloads, CATTS and meta-reasoning controllers may be included where their task interfaces genuinely match. Do not report a language-agent result as a robotics or temporal-solver comparison.

## 6. Execution mechanisms: do not conflate them

The benchmark records the scheduler decision and the mechanism separately.

| Mechanism | Meaning | Required cost/accounting |
|---|---|---|
| Early exit | Publish an intermediate result and stop | Output quality and action freshness at publication/application |
| Operator-boundary preemption | Suspend after a completed model operator | Coverage, widest blocking region, dispatch and resume overhead |
| Sub-operator/full preemption | Interrupt within an operator where supported | Exact runtime/kernel coverage, state-save cost, output parity |
| Temporal solver continuation | Yield after a committed numerical substep and resume | Token/snapshot bytes, capture/restore time, discarded/recomputed work |
| Cancel/restart | Discard in-flight work and start from a committed state | Wasted work and replay cost |
| Alternate solver | Switch numerical/reasoning method | State transfer, method-specific error and work profile |

A system does not receive credit for a large number of yield points without reporting actual execution coverage and the widest uninterruptible region. Solver continuation must consume #13097's token contract and #1269's snapshot payload; do not create another snapshot format.

## 7. Timing protocol

Use the exact same episodes in these separate strata:

1. **Frozen-world oracle:** evaluator-only upper bound; never reported as deployed performance.
2. **Wall-clock dynamic plant:** the world continues evolving through sensing, inference, projection, transport and actuation.
3. **Hard-deadline:** action is useful only if it is actually consumed/applied before the frozen deadline.
4. **CPU/GPU contention:** offered background load, DVFS, throttling and scheduling are declared.
5. **Memory-traffic contention:** measure DRAM/EMC bandwidth, stalls/backpressure where the target exposes them; do not infer physical traffic from logical bytes touched.
6. **Delayed/revised information:** stale observations and input revisions invalidate any continuation/evidence that no longer applies.
7. **Actuation-side freshness:** source observation → inference → projection → command dispatch → actual application. Report age at the consumer/actuator, not just model completion.

The deadline budget must reserve command transport and actuator response. Finishing inference before the deadline is insufficient if the action enters effect after expiry.

## 8. Required synthetic known-answer cases

The fixture file defines these scenario families:

- **High-confidence trap:** current checkpoint is confidently wrong; a later compute block can correct the action before its deadline.
- **Low-confidence but decision-sufficient:** current confidence is low, but every plausible checkpoint outcome retains an epsilon-sufficient common action; extra computation has no decision value for this decision.
- **Non-monotonic checkpoint quality:** correctness/action quality improves, deteriorates, then recovers; a present confidence or stability signal can stop at the wrong point.
- **Overthinking / missed opportunity:** more computation improves a prediction but the improved action arrives after its effective deadline; a faster robust action is better at application time.
- **Prediction-only improvement:** compute lowers trajectory/forecast error but does not change the action or regret envelope.
- **Dynamic-world reversal:** an initially correct action becomes wrong/stale while computation runs because the physical state changes.
- **Safe anytime proposal:** a valid intermediate proposal is applied; a later refinement may replace it only prospectively after independent admission, never retroactively.
- **Contention and invalidation:** contention or revised input makes the remaining compute value negative or invalidates a continuation; the policy must fall back or restart under the owning contract.
- **Future-leakage negative control:** checkpoint identifiers/order/timing or evaluator artifacts correlate with hidden outcomes; candidate actions must remain invariant under nuisance relabelling and future-suffix changes.
- **Redundant cognition:** additional HDC/LTC/retrieval/verification work changes representation details but leaves the epsilon action envelope invariant.

For each known-answer case, report the candidate's decision, compute allocation, actual action-application time, available information prefix, realized regret, resource use and whether the evaluator classifies the checkpoint as epsilon-sufficient. Retain every failed, timed-out, invalid, abstained and unserved episode.

## 9. Metrics — no composite score

### Decision and quality
- realized regret at actual action application;
- action/oracle agreement under the task's loss and admissibility contract;
- safe-success / viable-success (task completion plus all independent hard gates);
- wrong-belief, abstention and missing-output temporal exposures;
- prediction/trajectory error as a separate diagnostic;
- action-set changes and epsilon-information-set sufficiency.

### Compute and time
- compute work units/RHS evaluations/tokens/iterations, each in native declared units;
- p50/p95/p99 and worst-case wall-clock by inference, continuation, projection, transport and actuation stage;
- time-to-first-admissible-action and time-to-final-qualified-action;
- unused budget, wasted/discarded/recomputed work;
- checkpoint capture/restore overhead, continuation bytes and operator/preemption coverage;
- deadline misses, expired proposals and consumer-side stale rejections.

### Physical and system resource cost
- measured electrical joules by declared system boundary;
- CPU/GPU utilization and memory traffic/stalls where available;
- memory footprint and retained byte-time;
- temperatures, clocks/DVFS, throttling and cooling/battery reserve;
- sensing/communication/actuation energy where measured.

Operation counts and FLOPs are not joules. Simulated logical time is not wall time. If the hardware instrumentation does not exist, physical energy is UNKNOWN, not inferred.

### Adaptive policy quality
- predicted vs realized marginal decision benefit;
- calibration of predicted compute value;
- false-stop rate (stop while a future admissible compute block would materially improve the decision);
- false-continue rate (continue when no admissible compute block can change the epsilon action envelope);
- missed-deadline due to continued reasoning;
- regret reduction per native compute unit and per measured joule, as separate ratios;
- policy regret versus evaluator-only compute-allocation oracle.

## 10. Statistical design

- Freeze scenario families, task/loss/action contracts, seeds, compute blocks, deadlines, hardware, software and splits before confirmatory runs.
- Split by environment family, dynamics parameters, source family and initial-condition family; do not randomly split adjacent checkpoints.
- Use paired episode replays and the same offered workloads across policy arms.
- The mission/episode, not each checkpoint, is the independent statistical unit.
- Preregister primary axes, non-inferiority margins, multiplicity family and stop rules.
- Report paired uncertainty intervals and source-family holdouts.
- Use #9881's randomized intervention audit where its assignment assumptions and design are satisfied; otherwise identify the alternative design and its estimand explicitly.
- Keep oracle-generated labels and future checkpoint quality in offline training only. Freeze the candidate before held-out evaluation.
- Preserve invalid runs, timeouts and missing receipts in the denominator.

## 11. Prospective qualification gates (not results)

1. Zero false stops on exhaustive finite synthetic controls.
2. Zero accepted future-event/checkpoint-label leakage on the frozen attack suite.
3. Zero stale/expired action applications in the deadline-bound known-answer suite.
4. No hard safety/authority/viability gate may be bypassed by predicted compute value.
5. At matched decision quality and hard deadlines, the adaptive compute policy should improve at least one preregistered resource dimension without materially regressing the others.
6. On held-out families, seek at least 20% lower measured compute cost at a preregistered non-inferior regret margin; this is an initial research target, not a promised result.
7. Demonstrate a workload where adaptive allocation beats uniform/fixed compute and a workload where the direct fast policy is superior. Both are necessary to show the policy knows when not to use the expensive architecture.
8. Publish the full decision-quality/resource Pareto frontier, not just one winning scalar.

## 12. External benchmark mapping

- **RoboDyna (NeurIPS 2026 workshop):** asynchronous physical manipulation and environment evolution during inference. Use its synchronous protocol only as a paired reference; asynchronous outcomes matter for deployment. https://robodyna.github.io/robodyna/
- **MLPerf Inference v6.1 Agentic Edge:** compatible edge-agent deployment/latency/accuracy workload, not a substitute for physical control regret. https://docs.mlcommons.org/inference/index_gh/
- **NeuroBench 2.4.0:** native neuromorphic algorithm/system tasks and metrics; add continuous-time decision/freshness/resource measures as a labelled extension, not a replacement. https://neurobench.readthedocs.io/en/2.4.0/readme.html
- **BITS/TIME:** irregular temporal prediction tracks; prediction quality remains separate from decision quality. https://arxiv.org/abs/2609.33303 ; https://proceedings.mlr.press/v306/qiao26g.html
- **AERA / constrained compute / CATTS / agentic meta-reasoning:** algorithmic baselines where interfaces match; keep published results tied to their original tasks and protocols.

## 13. Claim boundary

This protocol can compare decision-aware compute allocation in declared task families, hardware, timing strata and action contracts. It does not prove general optimal stopping, a universal value of computation, real-time safety, or cross-platform energy efficiency.

A finite evaluator oracle is an upper bound for its enumerated worlds, actions and compute continuations only. Expected value depends on independently qualified probability models. Mixed-policy robust bounds use #13165. Candidate-visible predictions must be calibrated and never derived from evaluator-only future outcomes. Confirmatory evaluation requires #9437 isolation and #13061/#13091/#13142 evidence binding.

No SOTA, hardware, energy or deployment advantage is claimed by this protocol.
