# TOPO-ADAPT: exact robust viability and recoverability oracle v0.1

**Status:** research implementation slice; exact finite-state synthetic reference only  
**Owner:** TOPO-VIABILITY-001 (#14249)  
**Parent:** TOPO-ADAPT-001 (#14123)  
**Composition:** sequential trajectory contract (#14155) and persistent replay substrate (#14205)

## 1. Why this is a different benchmark objective

Current flow, connectivity, cut and spectral metrics describe the current graph. They do not say whether the system retains a policy that can keep it admissible through every disturbance in a declared future horizon, given the resources that remain.

The reference problem therefore distinguishes:
1. **Current function:** how much service is delivered now.
2. **Immediate admissibility:** whether current state and resource constraints hold.
3. **Robust viability:** whether a policy exists that preserves admissibility for every disturbance in the declared finite support through the horizon.
4. **Recoverability:** whether a policy can reach a target capability envelope by a deadline while preserving intermediate admissibility.
5. **Reserve-option value:** the change in viable state/budget pairs caused by preserving an option, net of holding and maintenance costs.

A network can have identical current max-flow to another and still have a smaller robust viability set. Conversely, a currently degraded state can remain recoverable. The scalar resilience score is deliberately rejected: report service, viability, recovery, uncertainty and resource vectors plus their Pareto frontier.

## 2. Exact finite-state game

A game contains a finite set of physical states, a finite action catalogue, an explicit observation partition, and a finite set of declared disturbance outcomes for every enabled state/action pair.

For state x, action a, and disturbance w:
- F(x,a,w) is the named successor state;
- c_hold(x) is the per-step maintenance/holding debit;
- c_action(a) is the action debit;
- c_disturbance(x,a,w) is any outcome-specific resource debit;
- remaining budget is an integer b;
- C(x) is current admissibility;
- C_H(x) is terminal admissibility;
- b_min is the required terminal reserve.

The transition is infeasible when any debit exceeds the remaining budget. All arithmetic is checked; overflow is an invalid model, not an infinite budget.

The exact terminal set is:

**K_H = {(x,b) | C(x) and C_H(x) and b ≥ b_min}.**

At a decision tick, the controller sees only the observation class and remaining budget. One action must therefore be valid for every hidden state in that observation class. The robust predecessor is:

**K_t = {(x,b) | C(x), exists one action a for observation O(x), and for every hidden x′ in O(x), for every w in W(x′,a), the successor (F(x′,a,w), b − c_hold(x′) − c_action(a) − c_disturbance(x′,a,w)) lies in K_(t+1)}.**

The quantifier order is **controller action first, disturbance second**. A future disturbance identity is not available to the controller. The witness action is selected deterministically by lexicographic action ID when multiple actions preserve viability.

Observation classes are fixed per tick and remaining budget is observable. This v0.1 contract is a conservative memoryless observation-policy model: it does not exploit hidden-state inference from longer observation histories. A later finite-memory/belief-state extension must be a separate version with its own exact oracle and tests.

For recoverability, the target envelope must be observation-consistent. Layer zero contains already-recovered states; layer k contains states from which the controller can reach the target in at most k actions, robustly against every declared disturbance, without violating safety or the cumulative resource budget. Unreachable targets return None for minimum recovery time; no arbitrary finite penalty is assigned.

## 3. Implemented API in this slice

The viability.rs module adds:
- typed FiniteHorizonGame, GameState, Action, StateAction and Outcome inputs;
- explicit disturbance support, cross-checked against the complete outcome set supplied for each state/action;
- strict uniqueness and reference validation for state, action and disturbance IDs;
- checked resource arithmetic and bounded exact-solver limits;
- solve_robust_viability() with time-indexed exact kernels and deterministic witness actions;
- solve_recoverability(deadline) with at-most-deadline semantics, minimum-step query, minimum starting-budget query and first-action witness;
- observation grouping that requires one common action for indistinguishable physical states;
- canonical state/action ordering so input vector permutations do not change the report.

The solver bounds are explicit: at most 64 states, 64 actions, 32 horizon steps, 128 integer resource units and 16,384 total outcome rows. These are small-instance reference limits, not claims about scalability to industrial network sizes.

## 4. Tests authored

Fifteen deterministic unit tests cover:
- one unsafe disturbance invalidating a nominally successful action;
- robust acceptance only when every disturbance successor remains viable;
- one common action for indistinguishable states;
- rejection when hidden states require mutually exclusive actions;
- cumulative holding, action and disturbance debits;
- non-spendable terminal reserve;
- minimum recovery steps, minimum starting budget and first-action witness;
- unreachable recovery returning None rather than an invented finite time;
- state/action permutation invariance and lexical tie breaking;
- exact disturbance-support/outcome matching;
- observation consistency of target membership;
- unknown successor, duplicate disturbance and arithmetic-overflow rejection;
- independent recursive minimax cross-check on small full-information games;
- current inadmissibility blocking an otherwise feasible future path.

These are authored tests, not an executed test pass. The editing environment does not provide cargo, rustc or rustfmt; no local compile, formatting, Clippy or CI pass is claimed.

## 5. Benchmark tournament

The exact oracle is a label/reference lane, not a peer controller. Every method must use the same state, action, disturbance, information and resource model.

**Reference and diagnostic baselines**
- present max-flow/min-cut only;
- graph connectivity, edge/node connectivity, algebraic-connectivity and Kirchhoff-index policies;
- greedy immediate-service restoration;
- greedy minimum-repair-cost;
- reserve-route, always-maintain and on-demand-maintenance policies;
- finite-horizon MPC with published horizon, solver time and information limits;
- robust MILP/scenario design where formulations can be matched;
- exact robust DP on small instances;
- clairvoyant disturbance-aware planning as a separately labelled information-rich upper bound.

Learned/approximate methods enter only after the oracle and split protocol are frozen. They must report false-viable and false-unviable errors, selective risk/coverage, calibration, abstention, out-of-support detection and decision regret. A false-viable classification is safety-critical and must not be hidden inside average accuracy.

**Mandatory episode families:** same-current-service/different-future-viability; recoverable current service loss; intact but unrecoverable service; cumulative budget exhaustion; reserve-option value; shared-risk/common-cause failures; partial observation; strategic post-action disturbance; deadline pressure; model uncertainty; node churn; held-out graph-family/scale transfer.

**Report separate outcomes:** current and worst-case service; robust viability and kernel membership; minimum recovery time and cost; disturbance margin and terminal reserve; false-viable/false-unviable rates and oracle coverage; service-deficit area and deadline misses; lower-tail and worst-group performance; action churn and reversals; lifecycle holding, repair, communication, sensing, memory and compute costs; solver and decision latency. Kernel counts are not comparable across different state spaces without a declared common state measure.

## 6. External research and industry positioning

Three comparator classes must remain separate:

1. **Resilience-based network design:** Zaman, Sharmin and Mahmud's 2026 online/January-2027 issue study jointly chooses topology and integer link capacities, models repeated recover-mode disruptions and chance-constrained recovered flow, and compares a MILP-based PSDA with a genetic algorithm. It is a strong stochastic design-time baseline, not an online partially observed robust controller.
2. **Runtime safety and distributed topology switching:** a November 2026 event-triggered switching preprint uses local robust barrier certificates and non-Zeno conditions for communication-link switching; an IEEE CASE 2026 study describes continuously enforced runtime dependability certificates for modular production agents. These motivate assumption-bound certificates, revocation when assumptions fail and an independent checking core. They do not establish authority authentication or physical certification in this software.
3. **Industrial evidence:** NIST's manufacturing initiative emphasizes trustworthy architecture, human-machine teaming, contextualized industrial data, integration test benches and digital-twin VVUQ. A July 2026 microgrid study reports real-time OPAL-RT hardware-in-the-loop validation for communication-efficient distributed secondary control under switching communication topologies. Synthetic exactness alone is not industrial qualification.

For industrial transfer, use offline replay → calibrated digital-twin/VVUQ → software-in-the-loop → hardware-in-the-loop → human-approved shadow/advisory → bounded pilot. Each stage needs an independent domain verifier. Power-grid use requires power-flow, voltage, thermal, radiality, protection and switching checks; max-flow is not a grid safety oracle. Manufacturing use needs real asset identity, process limits, operator override, runtime monitoring and outcome evidence.

## 7. Research hypotheses and falsifiers

- **H1:** current flow/connectivity can disagree with robust future viability under finite budgets, common-cause failures and hidden state.
- **H2:** preserving a reserve option increases viable state/budget coverage in some held-out disturbance families, but only when lifecycle holding cost is included.
- **H3:** a viability-aware policy improves worst-case service/recovery at matched resources over myopic greedy repair and finite-horizon MPC.
- **H4:** exact small-instance labels can calibrate an approximate viability shield with a useful coverage frontier while holding false-viable error below a pre-registered tolerance.
- **H0:** after matching action horizon, information and resources, viability-aware methods do not outperform existing flow-aware robust-design/MPC baselines.

Any claim must name graph generator, state distribution, action set, disturbance support, observation partition, horizon, resource units, baseline tuning budget and verifier. No universal SOTA claim follows from passing the synthetic oracle.

## 8. Promotion gates

1. Exact DP matches hand-solvable 2–6 node transition fixtures.
2. An independent reference implementation or exhaustive game-tree enumerator agrees on every P0 fixture.
3. Disturbance-after-action order, information partition, terminal reserve, cumulative costs and deadlines are mutation-tested.
4. Current function, robust viability and bounded recoverability disagree in the intended fixtures and are reported separately.
5. Greedy, reserve-route and MPC baselines use identical action/disturbance sets.
6. Approximate methods expose false-viable rates and selective coverage on held-out graph families.
7. Industrial claims require domain-specific SIL/HIL/field evidence; software-level proofs do not cross that boundary.

## 9. Proposed numerical promotion targets

These are ambitious preregistration targets for the next benchmark version, not achieved results or universal certification thresholds.

| Gate | Proposed target | Interpretation |
|---|---:|---|
| Exact-oracle agreement | 100% | Match a separately implemented recursive/game-tree reference on every enumerated state, time and budget in the admitted small-instance suite. |
| Exact false-viable labels | 0 | No state is marked viable when the independent exact reference says non-viable. Mutation tests must demonstrate that broken safety, budget, reserve and disturbance gates are detected. |
| Approximate-shield false-viable risk | One-sided 95% upper confidence bound ≤ 10⁻⁴ | Require at least 29,956 independent held-out decisions certified as viable with zero false-viable outcomes; use the exact binomial bound, not only a point estimate. Cluster dependence by graph/episode must be handled explicitly and cannot be hidden by counting correlated ticks as independent. |
| Approximate-shield coverage | ≥ 80% | Among the predeclared in-support held-out decisions, at least 80% should receive a non-abstaining viability decision while meeting the false-viable bound. Always-abstain is safe but not useful. Report the risk/coverage curve, not only this operating point. |
| Matched-cost policy value | ≥ 10% improvement target | Target at least 10% lower worst-decile service-deficit area or higher deadline recovery success than the strongest non-clairvoyant baseline at matched full lifecycle resource cost. Require paired 95% uncertainty intervals and the improvement to survive at least three held-out graph families and two unseen scales. This is a minimum-practical-effect target to preregister, not a prediction. |
| Operational safety | 0 false-safe committed actions | Every accepted action is checked against the exact domain operator and current certificate assumptions; stale, incomplete, model-mismatched or revoked evidence means abstain / no mutation. |
| Measurement completeness | 100% episode accounting | Every started episode terminates in a declared outcome: success, failure, infeasible, abstain, timeout or censored/unknown. Missing, queued and skipped runs are not successes. |

For zero events among n independent Bernoulli trials, the one-sided 95% exact upper bound is 1 − 0.05^(1/n). At n = 29,956 it is just below 10⁻⁴. This statistical result applies only to the sampled decision distribution and its independence assumptions; it is not a proof for arbitrary future systems. If events are clustered by graph family, simulator seed, or episode, increase the design effect / sample size or use a cluster-aware analysis. Count independent episode-level decisions, not every repeated tick as an independent sample.

## 10. Matched SOTA tournament and transfer

The strongest comparators are not interchangeable, so run three separate tournaments:

1. **Design-time topology and capacity:** reproduce the RBNDO PSDA/MILP and GA comparison on its recover-mode and chance-constrained design objective. Compare topology, installed capacity, stochastic recovered flow, lifecycle cost and solve time. Where aleatory distributions are unavailable or misspecified, add interval/robust uncertainty as a distinct treatment; do not relabel it as the paper's stochastic protocol.
2. **Joint operations recovery:** reproduce the 2026 manufacturing O&M HMA-PPO setting against MAPPO, ACO, SA and ALNS, and exact enumeration on small instances. Preserve the coupling between network reconfiguration and downstream resource scheduling. The paper reports exact optima on three small instances and approximately 0.24 s post-training online decisions in its setting; those are reproduction targets, not results transferable to our benchmark.
3. **Routing/communication reconfiguration:** for OSPF metric sequencing, compare against GNN-PMU and its serialized min-margin greedy fallback using exact Dijkstra/ECMP on every intermediate route state. For safety-certified communication switching, compare always-connected, periodic, threshold-only, centralized barrier and decentralized certificate strategies on the same plant and disturbance support. Keep switched communication topology separate from fixed physical plant topology.

The proposed Luminous advantage is a composable pipeline: Symthaea proposes and can abstain; Mycelix binds principal, grant, epoch, scope and provenance; a domain-specific independent verifier checks the state transition and certificate assumptions; the runtime commits only a verified action and reconciles the postcondition. This requires explicit integration with existing authorization/revocation owners and does not exist merely because this oracle computes a viability set.

Use two fairness views when necessary: capacity-matched and lifecycle-work/communication-matched. Publish both where one cannot satisfy the other. Include training/tuning budgets, inference latency, verification latency, communication, memory, sensing and maintenance. Compare fixed-manifest performance separately from graph-family generalization. A learned proposal can be faster than an optimizer while still being rejected by exact safety checks; report proposal quality, validator rejection, fallback, and end-to-end outcome separately.

## References

- Zaman, Sharmin & Mahmud, “Resilience-based network design optimization under aleatory uncertainty,” Computers & Industrial Engineering, article 112478, DOI 10.1016/j.cie.2026.112478.
- Wei et al., “A Scalable Decentralized Event-Triggered Topology-Switching Protocol for Verifiable Safety in Multi-Agent Systems,” arXiv:2611.21513 (16 Nov 2026).
- Fröhler et al., “Runtime Dependability Certificates for Modular Multi-Agent Production Systems,” IEEE CASE 2026, DOI 10.1109/CASE69030.2026.11704118.
- NIST, “Super Intelligence for Manufacturing.”
- Lorzadeh et al., “Communication-efficient distributed secondary control for accurate reactive and harmonic power sharing in stand-alone AC microgrids under switching topologies,” Scientific Reports (31 Jul 2026), DOI 10.1038/s41598-026-64236-x.

## Claim boundary

This is an exact finite synthetic game oracle with a deliberately small state/budget/horizon cap. It does not implement a learned controller, stochastic probability model, arbitrary-memory policy, physical network/plant, cryptographic grant check, runtime certificate issuer, actuator or industrial control interface. The disturbance support and transition model remain assumptions that must be independently qualified for any external use.
