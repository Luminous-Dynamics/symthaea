# TOPO-ADAPT trajectory replay: sequential, partially observed, budgeted topology adaptation

**Contract:** \`topo_adapt_trajectory_replay_v0_1\`  
**Status:** research/evaluation RFC; proposed, not implemented or qualified  
**Date:** 2026-11-28  
**Owner:** TOPO-ADAPT-001, Symthaea issue #14123  
**Related:** evaluator PR #14127; fixture PR #14125; MORPHOSTASIS crosswalk PR #14128; Symtropy MORPH-FUNCTION-001 #2685

## Executive thesis

The next benchmark must evaluate a **policy acting over a persistent trajectory**, not a sequence of independent one-step graph snapshots.

The question is no longer just whether a candidate edge can improve a flow calculation. It is whether a distributed controller can preserve service through a sequence of shocks, make locally justified changes with incomplete and delayed observations, retain useful changes across time, avoid destabilizing churn, and stay inside cumulative resource and authority budgets.

This is the most direct way to turn TOPO-ADAPT into a meaningful benchmark for endogenous structural adaptation.

The proposed paradigm is **viability-oriented morphostasis**:

> Maintain a set of admissible operating states from which the system can continue to satisfy its essential service constraints under the declared disturbance model, while preserving the option to adapt when evidence justifies it.

The target is therefore not one graph, one morphology, or one maximum-flow score. It is a trajectory of states that preserves declared capabilities at measured lifecycle cost.

## 1. Why v0.1 needs a trajectory successor

This section records a static source audit of the open v0.1 evaluator, not a test result.

Audited inputs:
- PR #14127, head commit \`871215ca00eee3555e7a19ccec67ba0c33db8bed\`.
- \`tools/topo-adapt-benchmark/src/lib.rs\`, fetched from branch \`research/topo-adapt-evaluator-v0-1\`.
- \`tools/topo-adapt-benchmark/src/main.rs\`, fetched from the same branch.
- Main was observed at \`57d871b0102aea96c43d27e2259b7ce3edd19ad6\` during this audit.

The v0.1 evaluator is valuable as a small deterministic reference, but it does not yet implement the complete trajectory semantics already required by #14123:

1. **No persistent policy state.** Each call to \`evaluate_episode\` clones the original \`active_edges\`, applies interventions up to the requested time, and then adds only the candidate IDs passed to that call. An accepted edit is not automatically carried into the next call.
2. **The CLI enumerates independent variants.** It scores no-edit and each single-edge candidate at each listed time. It does not execute an online controller whose previous actions change later observations and decisions.
3. **Declared budgets are not measured traces.** Message-byte and controller-update budgets are present in the fixture metadata, but the evaluator does not charge actual messages or controller updates. The README already discloses this limitation.
4. **Latency is input metadata, not yet an outcome.** Edge latency is parsed and preserved, but the current flow result does not calculate end-to-end path latency, deadline-constrained service, or recovery delay.
5. **Intervention ordering is underspecified.** Interventions are applied in stored vector order. Validation does not impose a strict time order or define simultaneous conflicting events. Two capacity changes to the same edge can therefore produce different results if their stored order is reversed.
6. **The candidate-sweep selector is an omniscient one-step bound.** It sees every candidate edge and evaluates each alternative exactly. It is not a distributed, partial-information, learned or temporal policy.

These are not reasons to discard v0.1. They define exactly what the next evaluator must add. Keep v0.1 as the static max-flow/min-cut reference and promote the trajectory evaluator as a separately versioned contract.

## 2. Paradigm shift: from graph repair to viability preservation

A fixed target graph is too restrictive. A system can preserve its function while using a different route, and can retain nearly all its edges while losing an essential capability.

Define:

- \(x_t\): the complete physical/network state at time \(t\);
- \(o_t\): observations actually available to the controller at time \(t\);
- \(a_t\): the controller's proposed action;
- \(w_t\): an exogenous disturbance;
- \(C(x_t)\): declared service and safety constraints;
- \(B_t\): remaining cumulative resource and intervention budget.

The closed-loop system is:

\[
x_{t+1}=F(x_t,a_t,w_t)
\]

\[
o_t=H(x_t,\eta_t)
\]

where \(\eta_t\) describes sensing loss, noise, delay and partial observability.

The admissible state set is:

\[
\mathcal{A}=\{x: C(x)\ \text{is satisfied}\}
\]

The research problem is to preserve or recover membership in \(\mathcal{A}\), or to maximize the time spent in an explicitly graded service region, while respecting \(B_t\). The *viability kernel* is the subset of admissible states from which a policy can keep the system admissible under the declared disturbance set. An exact viability computation is only expected on tiny fixtures; the larger benchmark tests whether practical policies approximate useful parts of it.

Make the state decomposition explicit:

```text
z_t = (i_t, G_t, q_t, b_t, r_t)
i_t  authorized target / intent version
G_t  current interaction and transport topology
q_t  measured service and functional capabilities
b_t  evidence, uncertainty and observation freshness
r_t  remaining authority and lifecycle resource budgets
```

The target specification (i_t) must not be inferred from the current shape or silently changed by the policy. It remains fixed within an episode unless an authorized target-change event is admitted. The admissible set is therefore target-relative:

[
\mathcal{A}(i_t)=\{z_t: C_j(q_t,G_t;i_t)\leq 0,\ j=1,\ldots,m\}.
]

This separates four questions that a single morphology score would confound: what the system is meant to preserve, what its structure currently is, what it can actually do, and what the available evidence justifies believing.

This reframes topology adaptation as a controlled capability problem:

- keep a functioning network unchanged when no edit is justified;
- retain reserve options when their lifecycle cost is justified;
- adapt when the existing topology cannot satisfy a persistent new requirement;
- avoid edits based on stale or ambiguous observations;
- recover after repeated disturbances without resetting the episode;
- explicitly abstain when no permitted action can meet the constraints.

### New hypothesis: topology reserve has option value

A dormant candidate connection can have value even before it carries flow because it may reduce future recovery time. But keeping it available can also incur maintenance, monitoring, ownership and failure-surface costs.

Test three distinct strategies:

1. **Build on demand:** install an edge only after service has degraded.
2. **Maintain reserve:** keep a qualified dormant edge or pre-authorized route available.
3. **Hybrid option policy:** pay a small reserve cost only when estimated disruption risk and activation delay justify it.

This tests whether the controller learns *when to preserve an option*, not merely when to maximize present throughput.

## 3. Normative trajectory semantics

The evaluator must own the environment state. A controller may propose actions; it must not write the evaluator's state directly.

### 3.1 Decision-tick order

Every decision tick has exactly this causal order:

1. **Exogenous transition:** apply the tick's declared failures, degradation, demand changes and communication faults to the authoritative state.
2. **Observation generation:** create the observation snapshot available at this tick, with source IDs, timestamps, age, missingness and sensor costs.
3. **Policy decision:** expose only the observations and action authority actually available to this policy.
4. **Action validation:** validate the action against its pre-state digest, authority, local action set and remaining budgets.
5. **Action execution:** apply the accepted action, or record a rejected/failed/unknown outcome.
6. **Receipt and state snapshot:** emit the post-state digest, complete resource delta, service result and terminal status.

An action at time \(t\) may use only evidence whose availability time is at or before the decision time. Future demand, future failures, hidden global topology and evaluator-only labels are prohibited unless a comparator receives the exact same information.

### 3.2 Persistent state

Accepted topology changes persist until an explicit subsequent action or declared exogenous event changes them. The evaluator must not silently reconstruct the initial graph at every tick.

At minimum, the state contains:
- node and edge identities, capacities, status and declared types;
- active, degraded, failed, dormant and removed edge states;
- current supply and sink-specific demand;
- pending actions, if delayed actuation is modeled;
- observation freshness and communication reachability;
- remaining per-episode and per-tick budgets;
- cumulative installation, removal, maintenance, sensing, communication and compute costs;
- the event, action and evidence ledger.

The benchmark must distinguish an accepted command from a confirmed physical/network state change. For simulated actions, the environment can return a deterministic actuation result. For any future hardware lane, an unknown execution outcome must block further mutations until the state is reconciled against observations.

### 3.3 Deterministic event ordering

- Episode ticks are strictly increasing.
- Exogenous events at the same tick are applied as one declared batch.
- Conflicting events that mutate the same resource at the same tick must either have an explicitly defined composition rule or cause fixture validation to fail.
- The result must not depend on the order of JSON array entries when event semantics say the events are simultaneous.
- Controller actions occur after that tick's exogenous batch and observation snapshot.
- Duplicate action IDs are idempotently rejected or reconciled; they must never charge or apply the same accepted mutation twice.

This removes vector-order dependence from the environment model.

## 4. Action and evidence receipts

Each policy decision emits one structured receipt, including for no-op, defer and abstain decisions.

Required fields:

- episode ID and tick;
- stable action ID and policy/build digest;
- pre-state digest and observation-set digest;
- observation IDs actually consumed by the policy;
- action type and target resource IDs;
- the action's declared reason code;
- predicted service change and uncertainty, when the policy exposes them;
- resource reservation and action cost estimate;
- accepted/rejected/failed/unknown execution status;
- post-state digest when the outcome is known;
- exact evaluator and fixture version.

Initial action vocabulary:

- \`no_op\`: preserve the present topology;
- \`add_edge\`: add a permitted connection;
- \`remove_edge\`: remove a permitted connection;
- \`set_capacity\`: change a declared capacity or weight;
- \`request_observation\`: request a permitted additional measurement;
- \`defer\`: take no irreversible action while waiting for a named condition;
- \`abstain\`: report that the current evidence or action set is insufficient.

Rewiring should be represented as explicit removal and addition operations, not as an invisible replacement. The evaluator must charge both sides and preserve the intermediate state if an operation is not atomic.

## 4A. Governed adaptation loop: proposal is not permission

The controller must not be the sole author, executor and judge of its own adaptation.

The proposed closed loop is:

1. **Capability contract:** identify the required capabilities, admissible alternatives, operating envelope, and conditions that require abstention.
2. **Observation and belief:** construct a timestamped state estimate with explicit missingness, staleness and uncertainty.
3. **Policy proposal:** generate a bounded action and predicted service/cost consequences.
4. **Authority check:** verify that the actor is permitted to change the named resource or objective. Authority is separate from model confidence.
5. **Independent feasibility shield:** check topology, capacity, budget, service and domain-specific constraints with a verifier that does not simply reuse the policy's prediction.
6. **Commit and execute:** apply the accepted action with an idempotent action ID and pre-state digest.
7. **Postcondition verification:** observe the result independently and compare it with the predicted transition.
8. **Reconcile or stop:** if execution is failed or unknown, block further irreversible mutations until the state is reconciled.

The Luminous responsibility mapping is intentionally non-overlapping:

- **Symthaea** proposes observations and bounded topology actions.
- **Mycelix** provides identity, permission, provenance and durable decision receipts; it does not certify physical safety.
- **Symtropy** checks the engineering constraints supported by its qualified model classes; the policy may not grade its own repair.
- **Sol-Atlas** supplies spatial/topological context with geometry and coordinate provenance.
- **The independent evaluator** remains authoritative for simulated state transitions, service outcomes and resource charges.

This creates a **proposal–authority–feasibility–execution–verification** boundary. It is more valuable than attaching an undifferentiated safety score to a controller because each failure mode has a named owner and an auditable receipt.

## 5. Resource accounting must be cumulative

The current fixture's declared message and update budgets are a useful start. The trajectory evaluator must charge actual policy traces against them.

| Resource | Required accounting |
|---|---|
| Topology edits | Additions, removals, rewires, capacity changes, reversals and rejected proposals |
| Edge lifecycle | Installation cost, maintenance per edge-time, dormant reserve cost and removal cost |
| Communication | Actual message count and bytes, including headers, retries, routing and coordination |
| Computation | Node updates, controller updates, inference operations, wall time and memory/state bytes |
| Sensing | Observation count, bytes, latency, acquisition cost and value-of-information action |
| Service latency | Per-path propagation/queue/processing delay, p50/p95/p99 and missed deadlines |
| Recovery | Downtime, time to recovery threshold and service deficit accumulated while recovering |
| Energy | Metered energy where available; otherwise a clearly named and calibrated proxy |
| Complexity | Active edges, maximum degree, topology churn, retained state and memory growth |

Budgets are cumulative over the entire episode unless a fixture explicitly declares a replenishment rule. A controller must not receive a fresh edit/message/compute budget simply because the evaluator advanced to the next tick.

Publish both:
- **capacity-matched comparisons**, where the available physical service capacity is held constant; and
- **work/communication-matched comparisons**, where actual controller work and communication are held constant.

Do not hide an unmatched budget behind one composite score.

## 6. Service, resilience and uncertainty metrics

The benchmark must keep the following outcome families separate.

### Service
- demand-weighted served fraction at each tick;
- cumulative delivered service over the full horizon;
- unmet demand and maximum consecutive unmet interval;
- deadline-constrained delivered flow;
- minimum service across critical sinks and worst operating group.

### Recovery
- time from disturbance to the declared recovery threshold;
- service-deficit area over the recovery interval;
- fraction of events recovered before the declared horizon;
- repeated-event recovery degradation;
- recurrence of the same failure after an apparent repair.

### Structural adaptation
- edit count, edit type and edit reversals;
- edge-time maintenance and reserve cost;
- churn per unit of additional service;
- maximum degree and graph growth;
- unnecessary edits during stationary periods;
- transient-demand overreaction and oscillation.

### Risk and epistemic performance
- worst-group service;
- lower-tail service and predeclared CVaR, with its tail convention stated;
- probability of unsafe or infeasible action;
- false-confidence rate on impossible or underdetermined cases;
- calibration of predicted service improvement and failure probability;
- selective risk versus abstention coverage;
- decision regret relative to a clearly labeled information-matched oracle.

### Efficiency
- actual message bytes per delivered service unit;
- controller work and state memory per delivered service unit;
- sensing cost per avoided service deficit;
- lifecycle cost per unit of cumulative service;
- p95/p99 decision and actuation latency.

A single scalar utility may be reported only as an additional, preregistered decision metric. It must never replace the component outcomes.

## 7. P0 temporal fixture suite

Build small graphs with independently checked max-flow/min-cut, time-expanded or min-cost references where appropriate. Every action and disturbance is deterministic in the initial suite.

| Fixture | Required behavior | Primary negative control |
|---|---|---|
| Stationary demand | Preserve the initial graph | Any unnecessary edit consumes budget without service gain |
| Transient demand pulse | Wait, sense or temporarily route without permanent overreaction | Always-rewire policy churns |
| Persistent new demand | Create and retain a useful route | Weight-only policy cannot create a missing path |
| First bridge failure | Restore service through an alternate route | Dead-end and random additions |
| Repeated disruption | Preserve prior edits, then adapt again after a second failure | Per-tick reset falsely erases earlier state |
| Shared bottleneck | Allocate limited capacity across competing sinks | Greedy policy improves one sink by silently starving another |
| Stale/partitioned observations | Defer, seek evidence or act only within safe local authority | Policy reads unavailable or future state |
| Conflicting same-tick events | Reject ambiguous fixture or apply its explicit batch rule | Input-array ordering changes the result |
| Exhausted budget | Abstain or preserve service without exceeding budget | Controller spends more than the remaining budget |
| Impossible demand | Report infeasible/abstain | Unbounded graph growth pretends the demand is feasible |
| Held-out graph family | Transfer to unseen topology generators | Memorized node IDs or candidate ordering |
| Scale transfer | Transfer to larger graphs and different degree distributions | Hidden dependence on training graph size |

### Mandatory repeated-disruption trace
### Concrete machine-readable trace

A deterministic three-tick persistence example is committed at
`docs/research/topo-adapt-repeated-disruption-trace-example-v0-1.json`.
It starts with route A, fails route A and adds route B, then fails the *runtime-added*
route B and adds route C. The expected flow is 4 before and after each recovery;
accepted edits are cumulative (0, 1, 2), not reset at each tick. The JSON marks
its actions as contract-demonstration scripts only, not evidence of controller
intelligence. A state-reset evaluator must fail this trace because it cannot
find the runtime-created route B when the second disruption arrives.


A minimal demonstration should include:

1. Initial service through route A.
2. Failure of route A.
3. A controller addition of route B.
4. A later independent failure of route B.
5. A second controller action that adds or activates route C.
6. A final interval in which both the surviving prior topology and the second repair remain present.

At each step, the receipt must show the same authoritative episode ID, a changing state digest, persistent accepted edits, and monotonically decreasing cumulative budgets. This single trace catches the current evaluator's most important missing temporal behavior.

### P1 extensions after deterministic P0 qualification

Do not mix these harder tasks into the first conformance gate. Add them only after persistent state, causal observations, event ordering, receipts and cumulative accounting are qualified.

| P1 extension | Scientific question | Strong negative control |
|---|---|---|
| Reserve-route option value | When is it worth paying to maintain a dormant route before failure? | Always-on reserve and never-reserve policies under identical disturbance priors |
| Active sensing / EVSI | Should the controller inspect, wait or reconfigure when persistent demand and a transient pulse are observationally ambiguous? | Free-oracle observation, no-sensing and random-sensing controls |
| Legitimate target change | Can the system distinguish an authorized change in required function from damage, noise or adversarial target substitution? | Target-blind policy and unauthorized target replacement |
| Partitioned local authority | Can local components preserve critical service during communication partitions without exceeding their local permissions? | Central controller with future information and a policy with unrestricted authority |
| Correlated shock sequences | Does the policy retain service under clustered failures and recovery-window overlap? | Independent-IID shock generator with same marginal failure rate |
| Topology hysteresis | Does the controller avoid repeated add/remove oscillations around a noisy decision threshold? | No-hysteresis threshold policy and deliberately perturbed observation sequence |

For reserve-route tests, predeclare the disruption distribution and reserve lifetime. Compare expected cumulative service deficit plus reserve installation/maintenance cost; do not reward an idle backup edge merely because it exists. Include a no-future-information online policy and a clairvoyant reserve oracle as separate arms.

For active sensing, observation requests must have latency, cost, missingness and a declared effect on the information set. Compute decision-sensitive value of information from the realized action and service outcome, not from entropy reduction alone. The same observation opportunity and budget must be offered to each policy.

For legitimate target change, the target specification and the authorization to change it are separate inputs. Mycelix may establish authority/provenance, while the independent task evaluator determines whether the authorized new target's functional constraints are met. A controller must not silently redefine its own success condition.

For partitioned authority, local actions must be checked against local authority and local evidence. Reconnection must reconcile divergent event histories and unknown outcomes before further irreversible edits. Centralized upper bounds must be labeled with their additional information and authority.

## 7A. Multi-rate coordination is a factorial, not a global-field toggle

Existing Symthaea owner #8388 is the implementation/benchmark owner for multi-rate coordination. This RFC evaluates that existing owner inside persistent topology adaptation; it does not define a competing channel architecture.

Use the channel vocabulary and semantics already in #8388:

- **Fast local updates:** immediate local state transitions and short-horizon responses using current neighbor observations.
- **Delayed / low-frequency local aggregate:** a slower local state signal with explicit aggregation window, age and update cadence.
- **Sparse nonlocal messages:** bounded longer-range coordination with explicit message latency, loss, routing overhead and byte/update budgets.
- **Optional field-like latent:** a separate experimental arm, not silently folded into any of the three channels above.

Keep persistent target/capability memory as a distinct state variable. Do not equate it with a global bioelectric field or with the slow local aggregate. Target memory is tested through the identity/authority owners (#8401 and Mycelix); the temporal channel mechanism is tested through #8388.

Each channel needs its own state, update cadence, message budget and timestamp semantics. Cross-channel messages must identify their source channel, event time, state version, confidence/uncertainty and expiry. Stale delayed aggregates must not overwrite a fresh local fault observation; a fast transient must not automatically trigger an expensive permanent topology rebuild.

The benchmark factorial should include:

1. all three #8388 channels active;
2. fast local channel removed;
3. delayed / low-frequency aggregate removed;
4. sparse nonlocal message channel removed;
5. all channels collapsed to one update rate;
6. channel timing shuffled or phase-randomized;
7. channel contents preserved but update cadence swapped;
8. optional field-like latent added as a separately reported arm;
9. target/capability memory independently removed or corrupted.

Match total controller updates, state bytes and communication budget across comparisons where feasible, and report unavoidable mismatch. A useful multi-rate result must demonstrate a functional or cost advantage attributable to temporal separation, not merely to more state, larger parameter count or more computation.

McMillen and Levin's Xenopus FLIM paper was published online on 5 August 2026 and appears in the November 2026 issue of *Developmental Biology*. It motivates asking whether genuinely distinct seconds-, minutes- and hours-scale components matter in a biological observation lane. It does **not** prescribe the controller's clock rates, imply that those tissue dynamics map onto the software channels, or prove that a three-channel controller is biologically equivalent. Choose controller cadences from the target system's declared time constants and measure them independently.

## 8. Comparator ladder: peer policies versus upper bounds

Do not compare different kinds of graph optimization as though they solve the same problem.

### Decentralized peer baselines
1. Static topology, no edits.
2. Fixed topology with dynamic routing or edge-weight adaptation.
3. Deterministic local greedy repair.
4. Distributed shortest-path/augmenting-path policy.
5. Degree- and edge-budget-matched random rewiring.
6. Distributed adaptive-topology heuristic with an explicit edit-cost threshold.
7. Multi-rate policy: fast local reaction plus slower coordination.
8. Learned policy, including candidate Symthaea HDC/CfC/LTC variants only after the common evaluator is qualified.

### Structural rewiring baselines
Delaunay/geometric augmentation, spectral-gap or expander-inspired rewiring, effective-resistance/curvature ranking, sparsification and virtual-node shortcuts are relevant where the task is **information propagation**. The IJCAI 2026 survey covers these GNN rewiring families and their over-squashing/over-smoothing trade-offs. They are not substitutes for capacity-feasible network-design baselines.

### Centralized and mathematical comparators
1. Rolling-horizon centralized optimizer with the same current information and action constraints.
2. Capacity-aware MILP or min-cost network-design optimizer on small instances.
3. Robust/stochastic network-design methods for repeated disturbances and uncertain recovery.
4. Genetic algorithm or another documented heuristic under the same objective and tuning budget.
5. Exact small-instance oracle.
6. Offline clairvoyant trajectory oracle with future disturbance knowledge, labeled strictly as an upper bound.

The one-step candidate sweep remains useful, but its label must stay **one-step full-visibility upper bound**. It cannot be used as evidence of an online decentralized controller's performance.

The early-access resilience-based network-design study by Zaman, Sharmin and Mahmud is a particularly relevant comparator because it jointly optimizes topology and integer capacities under repeated disruptions and chance-constrained recovery requirements. It evaluates a 7-node/12-arc benchmark, a 13-node/31-arc Bangladesh road corridor, a 20-node/30-arc network and a 74-node/258-arc Eastern Massachusetts network. Its MILP-based PSDA is compared with a genetic algorithm; this is a centralized network-design comparator, not a decentralized online peer. Use the published cases as comparator targets only after reconciling source data, objective, uncertainty model, constraints and runtime protocol. If the full published instances cannot be admitted, use independently generated 7/13/20/74-node stress families and label them as synthetic scale tests, not reproductions.

## 9. Experimental design and promotion gates

Use common random numbers: every policy receives the same disturbance sequence for each paired seed. Freeze scenario generation and data splits before tuning. Seeds and episodes—not individual nodes or timepoints—are the independent statistical units for the main comparison.

Required promotion sequence:

1. **Contract conformance:** parser, schema, event order, action vocabulary, budget semantics and digest rules pass deterministic tests.
2. **State persistence:** accepted actions persist; failures and removals mutate the same authoritative state.
3. **Causal observability:** no policy receives future, hidden or stale information outside its declared observation model.
4. **Independent reference:** small-instance service and objective values agree with an independent solver or exhaustive oracle.
5. **Resource enforcement:** no episode can exceed its declared cumulative budgets; rejected and unknown actions are accounted for.
6. **Trajectory reproducibility:** identical fixture, seed, policy digest and evaluator digest produce identical replay receipts.
7. **Baseline reproduction:** published and industrial comparators are pinned to exact versions and matched protocols.
8. **Held-out tournament:** graph generator, topology, scale, demand, disturbance and communication patterns are held out from tuning.
9. **Independent replication:** reproduce the key effect in a second implementation or independently maintained evaluation.

### Proposed aspirational gates

These are preregistration candidates, not achieved results:

- 100% agreement with the deterministic small-instance reference fixtures.
- Zero accepted actions that exceed a declared authority or cumulative budget.
- Zero future-information violations in the audit suite.
- 100% no-edit decisions on the frozen stationary deterministic controls when no permitted edit changes service.
- At least a 5-percentage-point absolute improvement in cumulative served-demand ratio over the strongest information- and resource-matched decentralized baseline on the designated damage/demand-shift suite, with paired 95% confidence interval excluding zero.
- No statistically supported degradation in any preregistered critical subgroup or the lower-tail service measure.
- At least 90% of in-distribution cumulative service retained on held-out graph families and a 2x graph-size transfer test.
- A separate efficiency result: at non-inferior cumulative service, demonstrate at least 2x lower communication bytes or a predeclared reduction in measured lifecycle cost. Do not claim both from a single proxy.

A result is a win only on the metric and operating envelope it actually qualifies. A better mean with worse tail risk is not an unqualified improvement.

## 9A. SOTA tournament protocol and statistical discipline

The benchmark should be designed so that a result is difficult to obtain by leakage, tuning, selective failure reporting or a favorable resource definition.

### Split design

- Split by graph generator and topology family, not by randomly sampled states from the same base graph.
- Hold out demand patterns, disturbance sequences, spatial layouts, communication faults and graph sizes.
- Keep all timepoints and perturbations from one base episode/seed in the same split.
- Publish train/dev/test manifests and their digests before final evaluation.
- Do not adapt the held-out test after inspecting candidate results; new fixtures become a new version with a new holdout.

### Paired evaluation

- Use common random numbers: every policy receives the same exogenous sequence for each paired episode.
- Treat the complete episode or independent disturbance sequence—not individual nodes, edges, frames or ticks—as the primary independent statistical unit.
- Report paired differences and confidence intervals, with cluster/bootstrap resampling at the graph-family and episode levels when appropriate.
- Choose the number of independent episodes from a predeclared power/precision analysis; do not select it after seeing which policy wins.
- Correct for multiple comparisons when promoting more than one primary claim. Publish the uncorrected and corrected result.
- Preserve every failed, timed-out, infeasible, abstained or unknown-outcome episode in the census.

### Budget equivalence

Every learned policy must disclose training data, training compute, tuning trials, hyperparameter search budget, parameter/state size and inference cost. Every baseline must receive the same task information and the same declared resource allowances. Include:
- capacity-matched results;
- communication/work-matched results;
- lifecycle-cost results;
- the remaining unmatched quantities.

### Fixed-corpus versus generalized performance

Report two separate estimands:
- **Benchmark performance:** performance on the exact frozen graph, disturbance, demand and seed manifest.
- **Generalized performance:** expected performance over the declared population of graph families, operating regimes and disturbances, including held-out generators.

The second question cannot be answered by treating thousands of correlated ticks from one graph as thousands of independent samples. Use a predeclared hierarchical or generalized linear mixed-effects model when its assumptions fit the data, with topology/graph family and episode/seed clustering represented explicitly. For time-to-recovery, service success, lower-tail service and resource outcomes, select outcome-appropriate models rather than forcing every metric into a single accuracy estimator. Publish raw paired episode results and cluster/bootstrap intervals alongside model estimates; state assumptions and diagnostics, and report a simpler robust estimate if the model is misspecified or underpowered.

This follows the NIST AI 800-3 measurement distinction between benchmark and generalized performance and its finding that statistical models can quantify uncertainty more efficiently while exposing benchmark-composition effects. It is a reporting discipline, not a claim that a GLMM automatically fixes a biased or unrepresentative test set.

Do not compare a learned decentralized policy's inference budget while omitting its training/tuning cost, and do not charge a baseline for communication while ignoring the proposed policy's coordination overhead.

### Reporting

Freeze one primary outcome for each stated superiority claim. Publish:
- per-episode distributions, paired effect sizes and confidence intervals;
- mean, median, lower-tail service and predeclared CVaR;
- complete failure, abstention, timeout and infeasible counts;
- calibration and selective-risk curves;
- Pareto frontiers over cumulative service, recovery time, communication, compute, edit churn and lifecycle cost;
- exact software versions, hardware, compiler/runtime settings, policy digests, fixture digests and commands.

A result should be called **SOTA on a named task and protocol** only after the strongest directly comparable baseline is reproduced or its deviation explained, the held-out comparison is preregistered, all resource differences are visible, and the key result is independently replicated. Otherwise report it as an internal result, a partial reproduction or an unmatched comparison.

## 9B. Topology is not one operation: three distinct adaptive regimes

The newest literature and existing repository owners require a sharper separation. A system can change its graph in at least three fundamentally different ways:

1. **Reconfiguration of an existing graph:** add/remove/rewire edges and alter capacities while node identity and the supplied node set persist. This is the primary task of #14123.
2. **Developmental graph growth:** generate new nodes and edges from a seed through local developmental rules. This is the separate computational task in [TOPO-DEV-GROW-001 #14173](https://github.com/Luminous-Dynamics/symthaea/issues/14173).
3. **Strategic co-evolution:** a defender changes the graph while an adversary observes and changes its strategy in response. This is the separate game task in [TOPO-ADAPT-ADV-001 #14172](https://github.com/Luminous-Dynamics/symthaea/issues/14172).

They must not be pooled into one leaderboard. Their state spaces, action sets, information assumptions, resource models and upper bounds differ. Shared infrastructure may be reused—episode replay, digest/receipt contracts, budget metering and statistical protocol—but each regime needs separate fixtures and outcome contracts.

### A. Strategic co-evolution: add an adaptive opponent

Zhao, Gao and Su's Nature Communications paper (published online 31 July 2026) models repeated strategic interaction between attackers and defenders, with both sides updating from prior outcomes. The authors report a heterogeneity-dependent asymmetry and attack strategies concentrating near a percolation threshold in their tested synthetic and empirical networks. Treat these as results under that paper's model and network families, not universal laws.

This exposes a limitation in a benchmark with fixed random or targeted failure scripts: a topology repair policy can overfit to the attack order. The next tournament therefore needs cross-play:
- static random/degree/betweenness/min-cut attack;
- dynamically reranked attacks;
- attacker best-response or bounded adaptive attack;
- defender local repair, flow-aware repair, reserve-route and adaptive-topology policies;
- exact small-game minimax/dynamic-programming bounds where tractable.

The attacker's and defender's observation sets must be separately declared. Neither receives hidden future actions, seeds or private state except in a clearly labeled perfect-information upper bound. Use cumulative bounded budgets for both sides and measure worst-case service, recovery, policy exploitability/regret where computable, false-positive defense costs in benign episodes, and lower-tail outcomes. Reuse #13203 for intervention re-ranking/delivery uncertainty and #12632 for adaptive attack/recovery assurance; #14172 owns only the co-evolving game.

### B. Developmental growth: grow from a seed, do not just rewire candidates

Barandiaran and Stovold's *Growing Echo State Networks through Graph-Based Morphogenesis with Developmental Graph Cellular Automata* (*Artificial Life*, online 4 September 2026; DOI 10.1162/ARTL.a.486) reports deterministic and stochastic DGCA that grow directed reservoir graphs from single-node seeds. Their abstract distinguishes task-driven NARMA-family targets from task-independent reservoir-metric targets; it reports specialized deterministic-grown structures and greater robustness/regularity in the stochastic-grown reservoirs in the studied comparisons.

This is not a baseline for ordinary edge rewiring: nodes as well as edges are generated through developmental rules. The benchmark is therefore owned separately by #14173 and should compare:
- deterministic and stochastic DGCA, if the exact published methods/artifacts and rights are admitted;
- a clean-room implementation, labeled separately from a faithful reproduction;
- conventional fixed-size Echo State Network reservoirs;
- sparse random directed reservoirs;
- fixed-topology optimized reservoirs;
- task-blind local growth and random feasible growth;
- subsequently, edge-only rewiring and developmental regrowth after damage.

Use the paper's exact NARMA task definitions, task-independent metrics, update rules and resource protocol where recoverable. Do not guess missing task order, split, training, or reservoir-stability settings. Report held-out sequence seeds, signal families, graph sizes, construction steps, state/parameter bytes, total node updates, compute, memory, task error and robustness. Construction, task adaptation and post-damage recovery are different phases and must have separate results.

Cortés-Poza's 31 August 2026 hybrid morphogenesis model couples bistable local dynamics, a synthetic regulatory layer, adaptive gap-junction conductances and slow tissue memory, but fixes graph topology between cell divisions. Its stated future extension is to let the graph itself grow and change neighbors. This supports the research gap, but the synthetic reservoir task is not biological validation and the biological model is not a directly comparable NARMA baseline.

### C. Open populations: agents join, leave and are replaced

A further industry-relevant case is an open network whose node set changes over time. This is distinct from a fixed node set with edge failures.

Peng et al. (IEEE Transactions on Cybernetics, volume 56 issue 9, 2026; DOI 10.1109/TCYB.2026.3690612) combine distributed spectral estimation, algebraic connectivity and the Kirchhoff index for topology reconfiguration under agent entry/exit. This is a strong structural baseline for open multiagent systems, but it still needs function/capacity-matched evaluation on our task.

A second IEEE early-access paper, *Resilient Consensus of Open Multiagent Systems: A Dynamic Topology Reconfiguration Approach* (published 22 October 2026; DOI 10.1109/TIE.2026.3743690), studies dynamic populations and f-local attack models, with a fully distributed circulant-digraph reconfiguration approach and Raspberry Pi hardware-in-the-loop experiments. Include its assumptions as a specialized consensus baseline when consensus is the task; do not treat it as a general flow/network-design solver.

The open-population suite should measure:
- agent entry, departure, replacement and identity/authority transfer;
- quorum and service continuity during churn;
- consensus residual/convergence time only on consensus tasks;
- capacity-feasible service and lower-tail performance on transport tasks;
- f-local fault assumptions and adversary capability, exactly as declared;
- reconfiguration overhead, communication, convergence and state reconciliation.

A graph that preserves algebraic connectivity but loses critical service is not a functional success. Conversely, service can remain admissible even if a structural metric worsens. Structural and functional results remain distinct.

### F. Event-triggered communication switching with safety certificates

A 16 November 2026 arXiv preprint, Wei et al., *A Scalable Decentralized Event-Triggered Topology-Switching Protocol for Verifiable Safety in Multi-Agent Systems* (arXiv:2611.21513), gives a different but highly relevant form of topology adaptation: agents choose when to disconnect from and reconnect to the communication network while a robust barrier certificate preserves a declared plant-state safe set under bounded estimation error.

The paper proves forward invariance and non-Zeno switching under its assumptions. It reports reducing topology-dependent per-agent calculations from O(2^N) to O(N), then evaluates four interconnected microgrids. The physical power-line topology is **fixed**; only communication links are switched. Its four-agent simulation reports 22.76% less average connected time and all four agents simultaneously disconnected for 22.57% of the run while phase/frequency remain in the specified safe region. These are preprint results for that plant/model, not a real grid validation or a universal guarantee.

The important research opportunity is **certified selective connectivity**: do not maximize network connectivity, and do not minimize message count blindly. Communicate only while required by the certificate, and reconnect before the declared estimation/safety envelope can be breached.

This is now a separate owner: [TOPO-CERT-SWITCH-001 #14247](https://github.com/Luminous-Dynamics/symthaea/issues/14247). Its benchmark distinguishes physical plant topology from communication topology, compares always-connected, periodic, threshold-triggered, centralized-barrier and decentralized local-certificate strategies, and measures false-safe disconnection, barrier margin, estimation-error bounds, non-Zeno behavior, connected time, bytes, reconnect latency and worst-agent control quality.

The certificate is assumption-bound and independently checked. Missing, stale, invalid, model-mismatched or expired certificates block disconnection. Simulation is not proof: analytical barrier conditions, finite numerical checks and runtime event semantics are separate promotion gates.

### G. Runtime capability/dependability certificates

A separate 2026 IEEE CASE contribution, Fröhler et al., *Runtime Dependability Certificates for Modular Multi-Agent Production Systems* (DOI 10.1109/CASE69030.2026.11704118), proposes machine-interpretable capability contracts that are continuously enforced and automatically revoked when certified assumptions stop holding. The abstract describes identity gates for process-input binding, resource identity attestation and continuous monitoring, with a laboratory additive-manufacturing/robot-handling case.

This supports a useful separation in Luminous: a topology edit is not permanently safe because it passed one check. The certificate must bind to the exact resource/model/operating envelope and be revoked when identity or preconditions change. Mycelix can bind identity/authority/provenance; an independent verifier still owns physical or mathematical validity.

Use this paper as a runtime-certificate architecture comparator, not as evidence that our current Rust runner already has cryptographic authority, continuous revocation or industrial certification.

### H. Viability and recoverability are different from current service

Current flow, connectivity and spectral robustness answer questions about the present graph. They do not answer whether an admissible future trajectory exists after the next disturbance.

The new owner [TOPO-VIABILITY-001 #14249](https://github.com/Luminous-Dynamics/symthaea/issues/14249) defines an exact finite-horizon robust viability kernel on small graph-control state spaces. It asks whether there exists a policy, under the declared information pattern and cumulative budget, that keeps constraints satisfied for every admissible disturbance. It separately measures recoverability to a target capability envelope before a deadline.

This changes the optimization target from a single graph score to **future feasible option space**. A graph can have full current flow but no viable response to the next outage; a temporarily degraded graph can still be viable when repair routes, time and budget remain. Exact dynamic programming is a small-instance oracle; approximate/learned viability shields come only after oracle conformance and must report false-viable errors.

A 2026 critical-network maintenance paper already couples post-disruption maintenance and routing across preparedness, response, recovery and durability. This supports evaluating the full intervention/recovery trajectory rather than only immediate restoration. The new kernel benchmark extends that idea with explicit disturbance quantification, action information and robust future feasibility.

### E. OSPF transition safety: exact routing verification between feasible endpoints

Bhattacharya and Samanta's October 2026 GNN-PMU study is a particularly useful industry-facing comparator because it does not evaluate only the initial and final routing graphs. It sequences link-metric updates and recomputes shortest paths and ECMP load at every intermediate transition, with a conservative serialized min-margin greedy fallback.

The paper reports direct learned-schedule acceptance of 90.5–96.9% and final rejection of 0.4–2.4% on its evaluated scenarios; accepted schedules satisfy link-capacity constraints under its exact routing model. It studies stale/corrupted telemetry, localized and asymmetric traffic spikes, update-batch distribution shift, forwarding-table churn and end-to-end OSPF reconfiguration time. Its simulations extend topology transfer through 100 nodes. These numbers are paper-specific results, not generic deployment guarantees.

This motivates a separate domain-specific successor, [TOPO-ADAPT-OSPF-001 #14206](https://github.com/Luminous-Dynamics/symthaea/issues/14206). It must not reuse maximum flow as an OSPF safety oracle. OSPF shortest paths, ECMP splitting, per-link utilization and every intermediate metric-update state require their own independent exact operator.

The target is:
- zero false-safe accepted sequences under the exact admitted routing model;
- direct safe acceptance and final rejection reported separately from fallback and false-infeasibility;
- scheduler time, exact validator time, OSPF convergence and orchestration measured separately;
- exact small-instance feasibility oracle for fallback completeness;
- held-out routing topology and demand transfer, beyond the comparator's 100-node simulation scope only after matched reproduction.

As aspirational same-protocol targets, attempt to exceed the paper's 96.9% upper direct-acceptance report and reduce its 0.4% lower final-rejection report, while preserving zero false-safe acceptance and paired confidence intervals. If exact protocol, training split or resource matching cannot be reproduced, call the work an adjacent benchmark, not a SOTA comparison.

### D. Topological metrics are diagnostics, not objectives by themselves

The 2026 distributed spectral/Kirchhoff work strengthens our baseline ladder: add algebraic-connectivity and Kirchhoff-index-informed reconfiguration, plus the 2024 Distributed Topology Reconfiguration method as older online-topology context. Compare their structural outputs with the independent service evaluator. Do not optimize a spectral metric alone where the operational objective is capacity, deadlines, safety or recovery.

Li, Zhang and Li's 30 October 2026 review of active topology and morphology in living systems emphasizes non-equilibrium activity, topological defects, multiscale modeling and the experimental/engineering challenges of active morphogenesis. This broadens the scientific context, but its biological observations remain a separate evidence lane from artificial topology controllers.

## 10. External validation lanes must stay separate

### A. Smart Cellular Bricks: embodied decentralized comparator

Moreno et al. (Nature Communications, 2026) demonstrate local NCA shape classification in large simulated assemblies and almost 200 physical bricks. The paper reports 98.97% simulated classification accuracy and 94.8% mean damage-direction detection accuracy in the simulated damage task. Its hardware demonstration transfers the classifier to physical assemblies (four tested shape classes); do not describe that as a physical damage-localization or repair demonstration. Damage localization and recovery/regrowth are evaluated in simulation; physical self-repair requires actuation and is identified as future work. The paper also reports a distributed voxel-graph Weisfeiler–Lehman classifier at 100% simulated accuracy and mean damaged-shape classification of 72.17 ± 35.68 for NCA versus 67.96 ± 44.11 for WL. These task-specific values are not a general leaderboard result.

Use this work as a strong **physical local-computation and communication-fault comparator**, not as a claim that its modules physically repair themselves.

The paper links code at \`https://github.com/rmorenoga/cube3D\` and study data at Figshare. Existing rights owner #8423 records that the code repository has no top-level LICENSE and that code, data and model-weight terms require separate admission. Keep this gate closed until the relevant permissions are documented. A clean-room implementation from the paper is a different artifact and must be labeled accordingly.

### B. DynamicAtlas: observed developmental trajectories

The DynamicAtlas resource reports live and static data from 500 Drosophila embryos, including wild type and 18 mutant genotypes, aligned to a common morphological timeline. The experimental data are publicly available through Dryad.

This can support a separate observational task lane for temporal alignment, tissue-flow trajectories, perturbation comparisons and held-out developmental stages/genotypes. It does not establish a causal morphogenesis mechanism or validate an artificial network controller. Start with metadata and a bounded subset; the public archive is large, so do not pull the entire dataset into the source repository.

### C. Multiscale bioelectric dynamics

The November 2026 FLIM study reports optically estimated membrane-potential patterns in Xenopus neural-crest explants over more than 17 hours, with seconds-, minutes- and hours-scale components. The authors distinguish these optically estimated measurements from direct electrophysiological measurements and report three independent clutch replicates.

This motivates a *factorial multirate ablation*:
- fast channel removed;
- intermediate channel removed;
- slow channel removed;
- all channels present;
- channels shuffled or time-scrambled;
- one generic global-field control.

It does not prescribe a software architecture or demonstrate that an engineered three-channel controller is biologically equivalent. Keep this lane observational and separate from synthetic graph-control scores.

### D. NIST manufacturing evaluation

NIST's 2026 Super Intelligence for Manufacturing initiative organizes evaluation around trustworthy system architecture, human-machine teaming, contextualized industrial data, integration test benches and digital-twin VVUQ. Its TIDAL framing layers functions across asset, site/area and enterprise levels. The Tennessee Eastman Process is identified as an initial test case, with planned work on fault diagnosis, preventive maintenance and VVUQ.

The industry lane should therefore progress from **offline benchmark → shadow-mode advisory → human-approved intervention**, with no autonomous actuation claim at the initial gate.

Measure missed critical faults, false alarms per asset-hour, detection lead time, root-cause grounding, downtime, time-to-recovery, recurrence, p95/p99 latency, lifecycle cost, operator workload and integration effort. Compare against the installed condition-monitoring, process-model, maintenance and digital-twin workflow—not just another neural network.

Distribution-grid reconfiguration remains a separate future qualification lane. It requires power-flow, voltage, thermal, radiality, protection and switching constraints; generic max-flow results cannot establish grid safety.

## 11. Proposed Luminous architecture boundary

The benchmark should keep authority and evidence distinct:

- **Symthaea:** proposes whether to observe, defer, abstain or adapt; emits a bounded action and policy receipt.
- **Mycelix:** supplies identity, authority, permission, provenance and durable decision receipts. It does not infer that a graph edit is physically safe.
- **Symtropy:** independently qualifies domain-specific physical constraints and function, using the existing solver where its claim ceiling permits.
- **Sol-Atlas:** supplies spatial/topological context and geometry provenance where location and physical structure matter.
- **Independent evaluator:** owns the authoritative state transition, resource ledger and outcome labels.

The controller must not grade its own repair. The proposal, authorization, execution, observation and independent verification records remain separate.

## 12. Immediate execution order

1. Preserve v0.1 as the static reference evaluator; do not present it as a temporal controller.
2. Add deterministic tests for intervention-order validation and conflicting same-tick event handling.
3. Define a stateful episode runner with persistent graph state and cumulative budgets.
4. Add the mandatory repeated-disruption trace before learning any policy.
5. Add deadline-aware service and explicit latency measurement.
6. Implement fixed, greedy, randomized, multi-rate and rolling-horizon baselines under the common policy interface.
7. Add partial observation, delayed messages, active sensing and EVSI only after the transition and accounting contract is stable.
8. Reproduce Smart Cellular Bricks only after rights admission; build DynamicAtlas and FLIM adapters as independent observational lanes.
9. Integrate Symthaea after the evaluator and baseline ladder are independently qualified.

## References

- Hartl, Levin & Pio-Lopez (2026), “Neural cellular automata: Applications to biology and beyond classical AI,” *Physics of Life Reviews*. https://doi.org/10.1016/j.plrev.2025.11.010
- Moreno et al. (2026), “Smart cellular bricks for decentralized shape classification and damage recovery,” *Nature Communications*. https://doi.org/10.1038/s41467-026-75166-7
- Attali et al. (2026), “Graph Rewiring in GNNs to Mitigate Over-Squashing and Over-Smoothing: A Survey,” IJCAI 2026. https://doi.org/10.24963/ijcai.2026/861
- “Lifetime imaging reveals long-distance non-neural bioelectric patterns on timescales from seconds to hours” (November 2026), *Developmental Biology*. https://www.sciencedirect.com/science/article/pii/S0012160626001661
- “DynamicAtlas: a morphodynamic atlas for Drosophila development” (published 2025; *Nature Methods* volume 23, 2026). https://doi.org/10.1038/s41592-025-02897-8
- Zaman, Sharmin & Mahmud, “Resilience-based network design optimization under aleatory uncertainty,” early-access article, *Computers & Industrial Engineering*, issue listed for January 2027. https://doi.org/10.1016/j.cie.2026.112478
- NIST, “Super Intelligence (SI) for Manufacturing.” https://www.nist.gov/programs-projects/super-intelligence-si-manufacturing

## Claim boundary

This RFC defines a proposed evaluation contract. It adds no controller, no measured benchmark result, no biological mechanism, no physical repair capability and no industrial qualification. The first evidence-bearing implementation must pass the deterministic contract and trajectory tests before any SOTA comparison is made.
