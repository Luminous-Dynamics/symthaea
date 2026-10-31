# AR-5B4 — matched scheduler round semantics v0.1

**Status:** typed round/action/cost/receipt contract implemented; benchmark drivers and live verifier integration remain future work.  
**Tracking issue:** [#9386](https://github.com/Luminous-Dynamics/symthaea/issues/9386)  
**Implementation:** \`crates/domains/symthaea-architecture-engineering/src/round_campaign.rs\`  
**Module export:** \`crates/domains/symthaea-architecture-engineering/src/lib.rs\`

## Why this tranche exists

A scheduler comparison is invalid when its action unit changes between policies. Coral Discovery is a four-agent allocation policy with round-level feedback and claim caps. The current Symthaea DirectionBoard synthetic campaign counts a single action slot as either a direction evaluation or a NEW-direction proposal. Those two action types are not necessarily the same work, and four concurrent agent seats are not one evaluation.

AR-5B4 defines a shared execution and evidence contract before adding Coral and other policies to a confirmatory head-to-head benchmark.

## Three separate comparison tracks

These tracks must not be collapsed into one leaderboard.

### Track A — mechanism-matched scheduler comparison

Question: which allocation rule best distributes a fixed evaluation budget over an already registered, verifier-labelled set of directions?

- Every policy receives the same pre-registered direction set, bootstrap evidence, four-seat round cadence, and total evaluation budget.
- The same claim-cap profile applies to all policies.
- Plans are generated from the same frozen pre-round snapshot.
- Every seat is one candidate evaluation. There are no direction-admission proposals in this track.
- Coral's direction-yield/credit rule can be evaluated as a **directions-only ablation**. Do not label this ablation as the complete Coral Discovery system because its NEW arm is deliberately excluded.
- Compare fixed round-robin, seeded random, global-score UCB, Symthaea local-record DirectionBoard, Coral directions-only, discounted UCB and Thompson sampling.
- CVT-MAP-Elites/MAP-Elites belong in this track only when behavior descriptors are frozen, measured independently and common across policies.

This isolates allocator quality from the difficulty and cost of finding genuinely new directions.

### Track B — end-to-end direction discovery

Question: does a search system discover useful directions and then convert them into verified capability at a better total cost?

- A NEW action is a proposal, not an admitted direction.
- The independent verifier decides admitted, duplicate, rejected or inconclusive.
- A newly admitted direction must receive a later, separately charged evaluation before it can earn a score.
- Proposal cost, candidate-evaluation cost, duplicate work, failed verification, retries and timeouts all remain in the cost ledger.
- Compare complete Symthaea DirectionBoard behavior, the faithful Coral scheduler with NEW enabled, UCB/Thompson variants, MAP-Elites where appropriate, and an oracle only as a synthetic upper bound.

The proposal/evaluation separation is intentional. It avoids granting a free experiment to a direction merely because the proposal itself was admitted.

### Track C — best-practical-system comparison

Question: which full system produces the most independently verified capability under a real operational budget?

Pin or report model and version, prompts, context limits, tool access, agent count, orchestration, candidate sandbox, verifier, task suite, retry policy, wall time, CPU/GPU time, memory, energy, token use and evaluator calls. Include setup and orchestration overhead. A per-evaluation scheduler advantage must not be misrepresented as an end-to-end agent-system advantage.

## Implemented contract

The Rust contract accepts any positive seat count so it can support different system profiles. The matched benchmark protocol pins four seats per round; that experimental choice is deliberately separate from the generic validator.

The v0.1 Rust module defines:

- \`RoundPlan\`: immutable policy ID, seed, round index, parent-close digest, pre-round snapshot digest, cost-profile digest, required cost dimensions, claim caps, seat list and plan digest.
- \`RoundSeat\`: a unique action ID, immutable work-item digest and exactly one typed action.
- \`RoundAction\`: \`EvaluateDirection\` or \`ProposeDirection\`. Proposal and evaluation are separate charged actions.
- \`ActionResultReceipt\`: result and verifier receipt digests, action kind, disposition, normalized score where verified, resolved direction and the exact pre-round snapshot observed.
- \`ActionCostReceipt\`: campaign budget units and a typed resource-cost map.
- \`RoundCloseReceipt\`: exactly one result and one cost receipt per planned seat.
- \`RoundLedger\`: contiguous round order, parent close chain, action count and charged budget totals.

Plan and close digests are domain-separated SHA-256 content commitments. They are not digital signatures. This module does not authenticate verifier identity, verify the verifier's independence, execute candidate code, or prove a score truthful.

## Fail-closed invariants

1. Every round has at least one seat and unique action IDs.
2. Every scheduled work item is digest-bound before execution.
3. The parent-close chain is absent only for round zero; later rounds must extend the ledger tip.
4. A round's results must all name the same pre-round snapshot. A result generated from an updated within-round snapshot is rejected.
5. Every seat has exactly one terminal result and exactly one cost receipt.
6. Result kind must match the planned action kind.
7. Verified scores are finite and normalized to [0,1]. Failed, rejected, inconclusive and timed-out evaluations cannot carry a score.
8. Admitted proposals require a new resolved direction; duplicates must resolve to an existing direction; rejected/inconclusive/failed/timed-out proposals cannot claim admission.
9. Every required resource-cost dimension must be present. Missing means unknown, not zero.
10. Every executed seat charges at least one budget unit. Failures/timeouts cannot be dropped from the denominator.
11. Common claim caps are checked at plan creation, before any work executes.
12. A closed round can be appended only once and in strict index order.

The ledger currently validates a caller-supplied receipt bundle. It is not a distributed consensus ledger, durable store, cryptographic verifier service or atomic multi-worker reservation system.

## Authored tests

Fourteen unit tests are authored for:

- plan digest and seat validation;
- duplicate action IDs;
- per-direction claim-cap overflow;
- duplicate proposal IDs;
- missing result/cost receipts;
- unknown required resource dimensions;
- stale snapshot / within-round feedback leakage;
- result/action-kind mismatch;
- failure cost accounting;
- contiguous round and parent-close chain, including reconstruction of ledger totals and tamper rejection;
- order-independent close digests for asynchronous result arrival;
- distinct duplicate-proposal accounting.
- zero action charge rejection.

These tests are authored but have **not** been compiled or executed in this authoring environment. Do not report them as passing.

## Next benchmark implementation

1. Add a four-seat round runner with a common bootstrap and frozen pre-round snapshots.
2. Add Track A policies: round-robin, seeded random, global UCB, DirectionBoard, Coral directions-only, discounted UCB and Thompson.
3. Add Track B policies: DirectionBoard NEW proposal flow, faithful Coral NEW flow, verifier-controlled direction identity, proposal/evaluation separation and all rejected/duplicate costs.
4. Add MAP-Elites/CVT-MAP-Elites only after a stable, independent behavior descriptor exists. Do not invent a descriptor after seeing policy outcomes.
5. Add independent synthetic landscape generators: delayed breakthrough, saturated leader, rank reversal, misleading proxy, multiple resource optima, genuinely new operator family, no-new null, adversarial duplicate refactors and multi-objective regressions.
6. Add real task replications only after synthetic construct validity: fixed task versions, held-out evaluators and independent verification.
7. Add atomic multi-worker budget reservations before parallel deployment. The current resource Pareto allocator is single-worker and cannot be used as a concurrency-safe global budget authority.

## Confirmatory design and metrics

Use at least 100 paired landscape/payoff instances across at least six independent landscape families. Freeze seeds, policies, metrics, score transforms, claims, cost dimensions, stopping rules and analysis before the confirmatory run. Cluster confidence intervals at the independent landscape/payoff-template level, not at the individual evaluation level.

Primary metrics:

- verified best-so-far capability and simple regret versus total charged cost;
- target-hit probability as a function of budget;
- wall-clock and resource cost to first independently verified improvement;
- delayed-breakthrough recovery and worst-family performance;
- NEW admission precision, duplicate rate and wasted proposal cost;
- distinct verifier-admitted direction coverage;
- false discovery rate on no-new and adversarial duplicate controls;
- allocation entropy / Herfindahl concentration;
- Pareto frontier across wall time, CPU/GPU, memory, energy, tokens, tools and evaluator calls;
- transfer to held-out task families and unseen operator combinations.

Report confidence intervals, per-family breakdowns and all negative controls. Show both mean and worst-family performance. Do not use the oracle arm in a deployable-policy average.

## Industry comparison

Relevant families include Coral Discovery's direction-allocation scheduler, ShinkaEvolve's evolutionary parent/model selection, CodeEvolve's island and CVT-MAP-Elites search, and multi-agent scientific research benchmarks such as SciAgentArena and FIRE-Bench. These systems target different parts of the problem: scheduler allocation, candidate evolution, persistent multi-agent research, or full-cycle scientific reliability.

The useful differentiator for Symthaea is not to copy one allocator. It is to connect verified direction identity, evidence-bound candidate lineage, explicit proposal/evaluation costs, multi-resource scheduling and held-out transfer under a common experimental contract. The comparison must preserve each baseline's native policy semantics, disclose adaptations and avoid claiming equivalence where action units differ.

## Claim ceiling

This PR establishes a typed evidence contract and its authored tests. It does not establish a functioning round scheduler, a Coral replication, benchmark execution, direction-discovery capability, SOTA performance, superiority, autonomous science or production readiness. Those claims require subsequent drivers, independent verifier integration and exact-head measured receipts.
