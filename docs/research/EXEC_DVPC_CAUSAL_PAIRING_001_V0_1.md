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
9. complete receipts and successful independent replay audits.

The returned certificate reports the regime, episode/world identity, number of ticks common to all arms, each arm's terminal state digest and trajectory origin. It does not compute regret; #13281 remains responsible for application-time regret and signed compute×observation interaction. #8578 remains responsible for downstream cluster-level inference.

## 4. Why common random numbers need event identity

Counterfactual variance reduction should align exogenous disturbances across arms without synchronizing endogenous branches. Consider two arms where one policy requests an extra sensor read or performs more internal steps. If both consume a single sequential random-number stream, the extra draw shifts all subsequent noise. The same "seed" then fails to represent the same exogenous disturbance at a later world tick.

The contract therefore binds separate seed domains and a common exogenous schedule, with the per-tick exogenous input committed independently. The execution driver must assign disturbances by stable event/tick/channel identifiers (or a frozen indexed schedule) so different branch consumption cannot shift unrelated events. Sensor-noise domains remain separate from world disturbances; a treatment can legitimately request a different set of sensor events.

The validator checks the commitments presented by the driver. It cannot prove the driver keyed its random stream correctly unless the independent replay auditor actually reconstructs that mapping from the retained event tape.

## 5. Falsification matrix

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

The source test suite contains positive and negative controls for these cases. The driver must separately test future-action prefix invariance and hidden-state isolation; a static receipt validator cannot establish those dynamic properties by itself.

## 6. Integration and ownership

- #8764 owns the general open-loop/action-coupled causal replay distinction.
- #8732 owns driver-owned clocks, per-tick delivered evidence and decision-time information views.
- Symtropy #3387 owns scenario execution/replay; #3471/#3502 own action-conditioned actor response and its execution adapter where relevant.
- #13281 owns four-arm regret arithmetic and the cluster adapter.
- #13282 owns the compute×observation experiment design.
- #8578 owns independent-cluster inference.
- #9437 remains the process/OS evaluator-isolation boundary.

No new transition simulator, actor policy, resource ledger, factorial decomposition, VOI engine, freshness horizon, statistical engine or action authority is introduced.

## 7. Benchmark sequence and claim ceiling

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

## References

- [Symthaea #13375](https://github.com/Luminous-Dynamics/symthaea/issues/13375)
- [Symthaea #8764](https://github.com/Luminous-Dynamics/symthaea/issues/8764)
- [Symthaea #8732](https://github.com/Luminous-Dynamics/symthaea/issues/8732)
- [Symthaea PR #13281](https://github.com/Luminous-Dynamics/symthaea/pull/13281)
- [Symthaea PR #13282](https://github.com/Luminous-Dynamics/symthaea/pull/13282)
- [RoboDyna](https://robodyna.github.io/robodyna/)
- [DEEP](https://arxiv.org/abs/2609.35910)
