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

A minimal demonstration should include:

1. Initial service through route A.
2. Failure of route A.
3. A controller addition of route B.
4. A later independent failure of route B.
5. A second controller action that adds or activates route C.
6. A final interval in which both the surviving prior topology and the second repair remain present.

At each step, the receipt must show the same authoritative episode ID, a changing state digest, persistent accepted edits, and monotonically decreasing cumulative budgets. This single trace catches the current evaluator's most important missing temporal behavior.

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

The early-access resilience-based network-design study by Zaman, Sharmin and Mahmud is a particularly relevant comparator lead because it jointly optimizes topology and integer capacities under repeated disruptions and chance-constrained recovery requirements. Its paper compares a MILP-based PSDA with a genetic algorithm. Reproduction still requires matching its uncertainty model, constraints, objective, data and runtime protocol.

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

## 10. External validation lanes must stay separate

### A. Smart Cellular Bricks: embodied decentralized comparator

Moreno et al. (Nature Communications, 2026) demonstrate local NCA shape classification and damage localization across large simulated assemblies and almost 200 physical bricks. The paper reports 98.97% simulated classification accuracy and 94.8% mean damage-detection accuracy in its described setting. Its physical demonstration supports decentralized shape recognition and fault localization. The recovery/regrowth experiments are simulation results; physical self-repair requires actuation and is identified as future work.

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
