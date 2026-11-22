# Joint Decision-Value Tournament for Computation and Observation (v0.1)

**Tracking:** #13216 — decision-value test-time computation; #11561 — real-time cognition  
**Observation-policy owner:** #9563  
**Expected posterior value / information-channel owner:** #12567 / PR #12551  
**Robust observation-support bound:** #13202  
**Joint resource and causal allocation owner:** #9666 / #9882  
**Exact factorial attribution:** #13176  
**Decision-regret authority:** #13152 / #13165  
**Evaluator isolation:** #9437  
**Status:** protocol and machine-readable fixture authored; no candidate controller, benchmark result or physical measurement is claimed.

## 1. Research question

When does additional computation become useful only when paired with another observation, and when does an observation become useful only when the system can afford to interpret it?

The benchmark compares four paired interventions on the same episode cohort:

1. no additional compute and no additional observation;
2. additional compute only;
3. additional observation only;
4. additional compute and additional observation together.

The treatment is evaluated at the actual time the action is consumed/applied. The world continues evolving through sensor acquisition, transport, inference, projection and actuation. The currently qualified fallback remains active while additional work is performed; its exposure and opportunity cost are not free.

This is a benchmark adapter under #13216 and the existing joint-resource programme #9666/#9882. It does not add another scheduler, sensing policy, VOI estimator, regret oracle, resource ledger, action authority or evaluator.

## 2. Why this is a new experimental question

Adaptive computation and decision-aware perception are established directions.

The September 28, 2026 preprint **The Decision Value of Perception Compute (DEEP)** is a particularly close comparator. It defines the downstream decision value of escalating perception from a cheap to an expensive mode, with an explicit budget-constrained decision oracle. Its reported KITTI/nuScenes experiments find that 34–54% of escalations that change downstream loss make the decision worse; its allocator must pay for its own computation, and only its lightest routers beat random under the reported 20% latency budget. These results are paper-reported for the authors' perception/planning workloads, not Symthaea results.

- DEEP: https://arxiv.org/abs/2609.35910

The narrower question here is interaction across two resource actions: additional computation and additional observation. A better observation may be unusable without interpretation; more computation may be valueless or harmful without fresh evidence. The combination can be synergistic, redundant, antagonistic or too late.

The benchmark must be capable of showing all four outcomes. It must not assume that additional sensing or compute is beneficial.

## 3. Factorial treatment contract

Let compute treatment \(c\in\{0,1\}\) indicate whether the declared additional compute block is executed, and observation treatment \(s\in\{0,1\}\) indicate whether the declared additional observation is acquired.

For every paired episode and hidden-world commitment, run:

| Arm | \(c\) | \(s\) | Treatment |
|---|---:|---:|---|
| Baseline | 0 | 0 | neither additional compute nor observation |
| Compute only | 1 | 0 | additional compute using the existing authorized evidence prefix |
| Observation only | 0 | 1 | acquire one permitted observation; do not add the treatment compute block |
| Joint | 1 | 1 | acquire the observation and execute the additional compute block |

The task contract freezes the compute block, observation option, resource ceilings, deadline, fallback action, model/configuration, candidate policy identity and episode seed before confirmatory evaluation. If treatment order or concurrency varies, it is a separately declared timing stratum; it must not be silently mixed within one estimate.

Every arm receives the same initial authorized evidence. The candidate never sees hidden-world identity, future outcomes, action-loss oracle values, or counterfactual treatment outcomes.

The observation treatment includes acquisition, sensor exposure, queueing, transport and preprocessing. The compute treatment includes the actual selected operator/solver/refinement path, dispatch, memory traffic and any checkpoint/continuation overhead. A sensor event timestamp is not the time at which the complete usable result becomes available.

## 4. Decision-value interaction

Let \(R_{cs}(w)\) be evaluator-owned realized decision regret for hidden world \(w\), measured at the actual application/consumption time of the arm's selected action:

\[
R_{cs}(w)=L(w(t_{cs}),a_{cs},t_{cs})-
\min_{a'\in A(w,t_{cs})}L(w(t_{cs}),a',t_{cs}).
\]

The per-world marginal benefit of compute alone and observation alone is:

\[
B_C(w)=R_{00}(w)-R_{10}(w),
\qquad
B_S(w)=R_{00}(w)-R_{01}(w).
\]

The joint treatment benefit is:

\[
B_{CS}(w)=R_{00}(w)-R_{11}(w).
\]

The compute-observation interaction is:

\[
I_{CS}(w)=B_{CS}(w)-B_C(w)-B_S(w)
=R_{10}(w)+R_{01}(w)-R_{00}(w)-R_{11}(w).
\]

Interpretation:

- \(I_{CS}>0\): complementarity; the joint treatment reduces more regret than the sum of isolated benefits predicts.
- \(I_{CS}=0\): no interaction on the declared regret scale.
- \(I_{CS}<0\): redundancy or antagonism; the joint treatment is less useful than the additive prediction.

Use paired per-world contrasts and preregistered episode-level summaries. Do not infer synergy from unpaired averages or different scenario cohorts. Negative treatment value is retained; harmful escalation must not be clipped to zero.

The evaluator must also report:
- action regret at application time;
- the fallback action's cumulative wrong-belief, abstention and missing-output exposure while additional work is pending;
- actual decision/application time and whether the deadline was met;
- stale/expired/rejected actions and all failed or unserved episodes.

The cumulative exposure and application-time regret are different endpoints. Do not add them together unless the task has a frozen, dimensionally valid utility that explicitly defines that conversion.

## 5. Attribution and existing evaluator ownership

The exact factorial treatment matrix is four rows for the two binary factors. Reuse #13176's exact subset/factorial machinery for its supported outcome dimensions:

- wrong-belief exposure benefit;
- abstention exposure benefit;
- missing-output exposure benefit;
- the existing twelve measured resource-cost dimensions.

#13176 also supplies higher-order interaction and Shapley reporting for those declared measured outcomes. This benchmark must not create a second factorial transform or attribution implementation.

Decision regret is reported as the separate paired per-world contrast above, using the #13216 / #13152 decision-evaluation contract. Do not claim that the current #13176 implementation automatically attributes decision regret: its current factorial outcome schema covers exposure and resource-cost dimensions. Any future extension to additional response dimensions must be made through the existing owner and versioned contract.

Observation outcomes and robust support remain with #9563, #12567/#12551 and #13202. This benchmark must not implement another expected posterior Bayes-risk calculation or another robust support oracle.

The twelve non-fungible resource coordinates remain those owned by #9882:
persistent bytes; retained byte-microseconds; read bytes; write bytes; compute microseconds; refresh operations; observations; sensor-active microseconds; end-to-end latency; measured energy nanojoules; verification operations; and receipt bytes. Unknown cost is not zero. No arbitrary scalar price can override hard ceilings.

## 6. Candidate policies and baselines

Where interfaces and task assumptions match, compare:

1. direct fast policy / no additional query;
2. fixed compute and fixed observation quotas;
3. compute-only adaptive policies from #13216: confidence, entropy/margin, stability, AERA-style and constrained-budget allocation;
4. observation-only policies from #9563: periodic, event/threshold, novelty, uncertainty, residual and candidate-estimated VOI;
5. expected-posterior VOI from #12567/#12551;
6. robust outcome-support regret from #13202;
7. DEEP-style cheap-to-expensive perception escalation, faithfully reproduced on perception/planning inputs where compatible;
8. joint complete-plan and multi-resource candidate policies under #9666/#9882;
9. evaluator-only exact four-arm factorial oracle for finite synthetic cases, never candidate-visible.

Keep official external benchmark reproduction separate from a Symthaea-specific extension. Mark a baseline NOT_COMPARABLE when input information, action domain, hardware, timing, or budget semantics cannot be matched.

## 7. Adversarial scenario families

At minimum:

1. **Complementary evidence and compute:** the observation alone is ambiguous; compute alone cannot resolve it; together they identify a decision-relevant state.
2. **Redundant observation:** more sensing does not change the epsilon-qualified action, with or without extra compute.
3. **Irrelevant compute:** extra compute refines an internal prediction but cannot change the action or regret envelope.
4. **Harmful escalation:** higher-fidelity perception changes the proposal but increases downstream regret.
5. **Late evidence:** the observation is informative but its complete result arrives after the decision window.
6. **Interpretation bottleneck:** observation arrives on time, but the candidate lacks sufficient compute to interpret it before the deadline.
7. **Fallback exposure:** the current qualified fallback is better than waiting for the joint treatment; charge its exposure while work continues.
8. **Dynamic-world reversal:** the additional compute/observation changes the actual action-application state and invalidates an initially good proposal.
9. **Overlapping execution:** compute and sensor acquisition overlap; compare with a separately frozen serial execution stratum, charging all concurrent resources.
10. **Contention and energy limit:** extra compute or sensing causes a latency, thermal, battery or hard-budget failure.
11. **Future leakage:** event IDs, row order, hidden-world labels, timestamps or treatment assignment leak evaluator truth.
12. **Negative interaction:** compute and observation individually help, but together cause delay or redundant work and perform worse than expected.

Include null tasks in which additional compute and sensing have no decision value. Preserve failures, timeouts, abstentions, invalid inputs and missing receipts in denominators.

## 8. Timing and safety semantics

For every arm, record the complete chain:

\[
t_{obs,event}\rightarrow t_{obs,available}\rightarrow
t_{compute,start}\rightarrow t_{compute,finish}\rightarrow
t_{action,dispatch}\rightarrow t_{action,apply}.
\]

The exact ordering may be asynchronous, but the timing contract is frozen and every stage is measured. The environment does not pause during computation or observation. The fallback remains subject to its own freshness/admissibility contract. A new proposal may supersede it only prospectively after independent admission; it cannot retroactively erase the fallback exposure already incurred.

Deadline reserve must include projection, transport and consumer/actuator delay. Finishing inference before a deadline is not enough if the action is consumed after expiry. No predicted value of information or computation may bypass authority, viability, physical safety or hard resource gates.

## 9. Metrics — no composite score

### Decision quality
- realized action regret at actual application time;
- paired per-world compute benefit, observation benefit and compute-observation interaction;
- action agreement and epsilon-sufficiency;
- candidate value prediction versus realized marginal benefit;
- worst-world and mean episode regret, with the declared statistical design.

### Temporal exposure and fallback
- wrong-belief, abstention and missing-output exposure separately;
- exposure accumulated while fallback is active;
- time to first admissible action and time to final qualified action;
- observation age and action age at the consumer;
- late observations, missed deadlines and stale-action rejection.

### Resource and physical cost
- all twelve #9882 resource dimensions;
- actual sensor exposure and acquisition time;
- compute work in native units;
- p50/p95/p99 stage and end-to-end latency;
- measured electrical energy with explicit measurement boundary;
- contention, thermal and battery reserve.

### Adaptive-policy quality
- false-stop and false-continue rates;
- harmful-escalation rate;
- observation/compute opportunity missed by each policy;
- predicted versus realized interaction;
- regret against the evaluator-only feasible four-arm oracle.

No one weighted score should hide an increase in one exposure dimension, hard budget overrun, or deadline failure.

## 10. Statistical and causal design

- Freeze treatment definitions, action/loss contracts, resource limits, deadlines, model/configuration, split manifests, seeds and timing strata before confirmatory evaluation.
- Use paired initial states and hidden-world commitments across all four arms.
- Randomize treatment order or execution order where carry-over is possible; retain order/period in the analysis.
- Use independent episodes/mission windows as statistical units, not correlated checkpoints.
- Block by environment family, sensor mode, dynamics regime and compute contention.
- Report paired uncertainty intervals, worst-family results and all failures.
- Use exact exhaustive four-row treatment coverage on small fixtures; use preregistered factorial/covering designs for larger resource combinations.
- Keep oracle labels, future trajectories and unobserved treatment outcomes evaluator-only.
- Separate exploratory policy tuning from held-out confirmatory episodes.
- Bind result receipts to exact source/evaluator/model/config/seed/sensor/hardware identities.

## 11. Prospective targets (not achievements)

1. 100% exact agreement with finite four-arm golden interaction fixtures.
2. Zero future-information leaks and zero hidden-world/treatment-cohort mismatch.
3. Zero false decision-value credit for prediction-only improvement without downstream decision benefit.
4. Zero stale/expired action applications in exhaustive deadline controls.
5. Zero hard resource overruns and no authority/safety gate bypasses.
6. At matched task and budget, demonstrate a held-out workload with positive joint interaction and a workload with negative or zero interaction.
7. Demonstrate at least one workload where direct fast or one-factor policy beats the joint treatment; the more complex system must know when it is unnecessary.
8. Seek at least 20% lower measured resource cost at a preregistered non-inferior regret margin on a held-out workload; this remains a research target, not a promised result.
9. Publish the full Pareto frontier and null/adverse regimes, not only a selected winning arm.

## 12. External benchmark mapping

- **DEEP (arXiv:2609.35910):** closest comparator for decision-value perception escalation under compute/latency/energy budgets. Reproduce its cheap/expensive perception arms and task-specific decision oracle where possible; keep the four-arm compute×observation interaction as an additional Symthaea extension. https://arxiv.org/abs/2609.35910
- **RoboDyna (NeurIPS 2026 workshop):** paired synchronous/asynchronous physical manipulation, dynamic world and action outcomes. Use its async mode for temporal consequence; preserve its own task success/progress metrics. https://robodyna.github.io/robodyna/
- **MLPerf Inference v6.1:** compatible agentic-edge/end-to-end inference system measurements. Use official workload accuracy/latency semantics; add Symthaea decision-regret and sensing interaction as separately labelled extensions. https://mlcommons.org/2026/09/mlperf-inference-v6-1-results/
- **NeuroBench:** native neuromorphic algorithm/system metrics remain separate from temporal decision quality and wall-power measurements. https://neurobench.readthedocs.io/

External published results are not a shared leaderboard. No cross-task SOTA claim follows from the four-arm protocol.

## 13. Claim boundary

This protocol defines an evaluator-side paired factorial benchmark. It does not implement a candidate joint scheduler, perception model, sensing policy, compute allocator, calibrated VOI estimator, action authority or physical safety system.

It can measure finite treatment effects and decision-regret contrasts under the declared episodes, task losses, timing and resource boundaries. It does not prove global optimality, arbitrary-distribution guarantees, universal synergy, measured hardware advantage or deployment readiness.

Confirmatory evaluation must reuse the isolated evaluator and source-bound receipt requirements in #9437, the decision/observation evaluators in #13152/#13202/#12567, and the existing joint resource/interaction owners #9666/#9882/#13176.


## 14. Executable paired decision-regret reference

The narrow signed paired 2×2 decision-regret contrast is implemented in the standalone crate:

research/decision-value-interaction-001/

Tracking issue: #13279. The crate requires exactly one baseline, compute-only, observation-only and joint result for the same frozen episode/hidden-world/action-loss scope. It binds the common task/loss, evaluator, model/configuration, policy and seed identities; preserves the distinct actual application timestamps; rejects missing/duplicate arms, identity mismatches, empty digests and out-of-range regret; and retains negative marginal benefits and negative interactions.

Eleven Rust unit tests are authored for positive complementarity (+230,000 PPM), zero/additive interaction, negative antagonism (−170,000 PPM), harmful compute escalation, row-order independence, distinct application times, missing/duplicate arms and identity/range failures.

This reference does not replace #13176. #13176 remains responsible for exposure and resource factorial attribution; the new crate supplies only the paired decision-regret contrast, which is not in the current #13176 outcome schema. Rust execution and CI qualification remain pending.

## 15. November 2026 research update: DEEP v2 and long-horizon embodied evaluation

The latest DEEP revision is v2, last revised November 16, 2026. Its central contribution is already decision-aligned perception-compute allocation: the value of escalating from cheap to expensive perception is the signed change in downstream loss, and the allocator is charged for its own computation against a budget-constrained decision oracle. The reported KITTI, nuScenes and open-loop nuPlan results include harmful escalation and cases where improved perception disagrees in sign with downstream decision value. We must treat signed decision value, negative escalation value, allocator-cost accounting and budgeted-oracle comparison as established prior art, not as Symthaea inventions.

- DEEP v2: https://arxiv.org/abs/2609.35910
- The paper's fixed-history track conventions make per-input escalation value additive for its cache/oracle evaluation. That enables efficient re-evaluation of pre-escalation allocators against cached outcomes, but it is not a complete model of a recurrent agent whose actions, observations and environment evolve together.
- This tournament's narrower extension is an explicitly paired compute × observation factorial contrast, with each arm's action evaluated at its own actual application time and with fallback exposure, all resource dimensions and treatment interaction reported separately. It extends DEEP's decision-aligned accounting; it does not supersede or replace DEEP's benchmark.

The open-access survey **Long-horizon embodied manipulation: a survey of methods, trends, and challenges** (published November 16, 2026) argues that binary task success can obscure partial completion, error propagation, recovery cost and horizon-dependent degradation. It emphasizes shared task state, progress-aware monitoring and failure-conditioned recovery. The tournament should therefore preserve verified partial progress, first divergence, recovery cost and horizon-specific results as separate outcomes rather than treating a late correct action as equivalent to uninterrupted successful execution.

- Survey: https://link.springer.com/article/10.1007/s44336-026-00048-x

### Reproduction priority

1. Reproduce DEEP v2 on its declared perception/planning tracks with its fixed models, official splits, costs and decision oracle. Use its public cached outcomes where the release provides them; do not substitute Symthaea-specific labels or loss weights.
2. Report official DEEP allocator/oracle metrics unchanged.
3. Run the Symthaea compute × observation extension as a separate paired experiment on compatible dynamic tasks, using the same initial episode and exogenous hidden-world stream across the four arms.
4. Add memory and verification as further factors only through #9666/#9882/#13176 after the four-arm compute × observation comparison is stable.
5. Evaluate partial completion, failure propagation and recovery cost under the long-horizon survey's recommendations; do not collapse those into a weighted aggregate.

## 16. Industry comparison and claim discipline

MLPerf Inference v6.1 (September 16, 2026) adds Edge Agentic Inference and end-to-end RAG alongside established datacenter/edge system workloads. Its official results evaluate declared workloads, quality targets and latency/throughput scenarios. Symthaea should reproduce any compatible official test exactly, then publish decision regret, observation interactions and temporal freshness as separate extension metrics. Do not compare an HDC/LTC task result to an MLPerf token/query throughput figure as if they measured the same capability.

- MLCommons v6.1 results: https://mlcommons.org/2026/09/mlperf-inference-v6-1-results/
- MLPerf Edge suite and scenario rules: https://mlcommons.org/benchmarks/inference-edge/

NeuroBench 2.4.0 supplies algorithm workloads such as few-shot class-incremental learning, event-camera object detection, non-human-primate motor prediction and chaotic-function prediction, with system metrics such as parameter footprint, activation sparsity and synaptic operations. Symthaea should enter the official track only where its model/task interface matches, retaining NeuroBench's native metrics and reporting measured wall-power/latency separately from theoretical operation counts.

- NeuroBench 2.4.0: https://neurobench.readthedocs.io/en/2.4.0/readme.html

The defensible competitive claim is not that Symthaea is more efficient or more intelligent today. It is that this programme is setting up a reproducible comparison of decision quality, temporal application cost, observation-compute interaction and non-fungible resources, while retaining official benchmark metrics where applicable.