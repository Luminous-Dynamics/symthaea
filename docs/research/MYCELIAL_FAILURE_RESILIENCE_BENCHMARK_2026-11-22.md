# Mycelial Failure Resilience Benchmark

**Date:** 2026-11-22  
**Status:** First controlled junction/segment deletion evaluator committed; exact-head validation pending.  
**Code:** `crates/domains/symthaea-canvas/src/mycelial_failure_resilience.rs`

## 1. The shift: from structural redundancy to functional disruption

A high cycle rank or visually dense network does not prove that important routes survive damage. The decisive question is whether declared network demand remains connected and how much longer its paths become after a controlled intervention.

This evaluator works on the junction–hypha graph, not the filament-adjacency graph. It removes explicitly named junctions and/or hypha segments from a derived graph view and measures the surviving topology and demand-pair paths. The source graph and evidence are immutable.

## 2. Input contract

### Demand pairs

Each `MycelialDemandPair` specifies:
- stable demand ID;
- source and target junction IDs;
- positive integer demand weight.

Demand pairs must be sorted by unique ID. The total demand weight is bounded to (2^{53}), so the integer weights can be represented exactly during IEEE-754 weighted aggregation. No normalization or inferred demand distribution is applied.

### Failure scenarios

A `MycelialFailureScenario` contains:
- a stable scenario ID;
- sorted unique failed hypha segment IDs;
- sorted unique failed junction IDs.

Removing a junction also removes all its incident segments. Combined scenarios can remove both junctions and segments. Unknown IDs, duplicates, unsorted failure inventories and empty non-baseline scenarios fail closed.

### Reproducibility

The profile binds:
- source junction–hypha graph digest;
- complete sorted demand inventory and demand digest;
- exact failed segment/junction IDs for every scenario;
- a per-demand outcome record (direct distance and surviving network path, or both explicitly absent);
- baseline and intervention summaries;
- canonical profile digest.

Profile validation recomputes each scenario's reachable/unreachable pair counts, weights, weighted reachability, directness and path-length aggregates from the per-demand outcomes. It rejects a graph path shorter than the Euclidean endpoint separation instead of omitting that demand from the summary.

## 3. Metrics

For each baseline or failure scenario:

### Structural topology

- remaining junction count;
- remaining hypha segment count;
- connected components;
- cycle rank:

[
eta_1 = E - V + C
]

### Demand functionality

For demand pair (q) with positive weight (w_q), the result stores its direct Euclidean distance (d_E(q)) and surviving graph shortest-path length (d_G(q)), or two explicit absent values if unreachable.

Weighted reachability:

[
R_w = rac{sum_{qin Q_{mathrm{reachable}}} w_q}
{sum_{qin Q} w_q}
]

Demand-weighted geometric directness:

[
D_w = rac{
sum_{qin Q_{mathrm{reachable}}}
w_q,d_E(q)/d_G(q)}
{sum_{qin Q_{mathrm{reachable}}}w_q}
]

Demand-weighted network path length:

[
L_w = rac{
sum_{qin Q_{mathrm{reachable}}}w_q,d_G(q)}
{sum_{qin Q_{mathrm{reachable}}}w_q}
]

Unreachable pair count and unreachable demand weight are reported separately. Directness and path length are undefined when no demand weight remains reachable; they are not set to zero.

The implementation rejects an impossible path where graph path length is shorter than Euclidean endpoint separation.

### What this means

- (R_w) measures structural reachability under the declared demand set.
- (D_w) measures geometric path directness among the reachable demand.
- (L_w) measures the weighted shortest-path distance among the reachable demand.
- (eta_1) describes graph cycle rank, not functional survival.

These are geometric graph quantities in the source graph's declared length unit. For the current generated Mycelial Presence model that unit is an organism-local model unit. No physical transport, conductance, material flow or biological viability is inferred.

## 4. Implemented operations

- `build_mycelial_failure_resilience_profile`: evaluate caller-supplied sorted controlled failure scenarios.
- `build_single_failure_resilience_profile`: generate one scenario per source segment and one per source junction, then evaluate the exhaustive single-failure sweep.
- Dijkstra is computed once per distinct demand source for each scenario.
- The graph is indexed in stable junction/segment ID order.
- Demand outcomes are stored individually and profile validation recomputes their counts, weights, reachability and directness aggregates.
- The exact failed identity lists are part of the receipt digest.

## 5. Reproducible attack families

The module now generates versioned attack plans from a frozen source graph and demand inventory. A plan binds the source graph digest, demand digest, strategy, seed, maximum failure budget, full ranked entity order and exact nested scenarios.

Implemented attack strategies:

| Strategy | Target | Ranking policy | Interpretation |
|---|---|---|---|
| Random segments | Hypha segments | SplitMix64 shuffle of stable IDs | Unstructured segment disruption |
| Random junctions | Junctions | SplitMix64 shuffle of stable IDs | Unstructured junction disruption |
| Highest-degree junctions | Junctions | Descending incident-segment count on original graph; stable ID tie-break | Static hub-targeted attack |
| Adaptive highest-degree junctions | Junctions | Recompute degree on residual graph after every prior removal; stable ID tie-break | Adaptive hub-targeted attack with per-step rankings |
| Lowest-degree junctions | Junctions | Ascending incident-segment count on original graph; stable ID tie-break | Peripheral/leaf-targeted negative comparator |
| Demand-path-load segments | Hypha segments | Descending declared-demand weight on deterministic shortest paths; stable ID tie-break | Demand-conditioned edge load, not generic all-pairs betweenness |
| Demand-path-load junctions | Junctions | Weighted internal transit-node use on deterministic shortest paths | Demand-conditioned node load, not generic all-pairs betweenness |
| Spatial-cluster junctions | Junctions | Ascending Euclidean distance to a declared seed junction; stable ID tie-break | Localized correlated structural-damage proxy |

For each strategy, the plan contains budgets 1 through k, each budget being a nested prefix of the selected attack order. Scenario IDs encode the policy version, seed and budget. The plan retains both the selected sequence and the complete candidate ranking at every step. The plan validator re-computes the rankings from the source graph and demand set, then checks every scenario is exactly the matching prefix. A changed graph, demand, seed, per-step ranking or failed-identity list invalidates the plan.

### Static versus adaptive degree

The static highest-degree policy computes degree once on the original graph and never changes the order. The adaptive highest-degree policy removes previously selected junctions in a derived residual graph, removes their incident hypha edges, and recomputes degree before selecting the next junction. It records all remaining candidates and their deterministic order at every step, not just the winner.

This separates two questions:
- Does the original graph's hub ranking identify critical junctions?
- Does an adaptive attacker find a different vulnerability after earlier hubs have been removed?

The current dynamic policy adapts degree only. Dynamic demand-path load, recalculated betweenness, articulation/min-cut recomputation and partial-capacity attacks remain separate future strategies. The current adaptive intervention is still a sequence of fully enacted structural removals; probability-of-delivery is a separate optional simulation profile.

The demand-path-load strategy computes weighted shortest routes only for the supplied demand pairs. It must not be called global betweenness: it is explicitly demand-conditioned. Equal-cost route ties are resolved by stable node/edge ordering. The spatial-cluster policy uses source graph model coordinates and is currently a local geometric stress proxy; coordinates are not calibrated physical distances.

### Research comparator and extension boundary

Recent network-resilience work provides two important challenges to a centrality-only ranking:

- A 2026 intermodal-freight study compares random failures with static and dynamically recalculated degree/betweenness, uses flow-weighted centralities and partial node degradation, and evaluates both giant-component size and flow-weighted network efficiency. This is a useful methodology comparator, not a direct fungal dataset. <https://doi.org/10.1177/03611981261437033>
- A 2025 study of attack uncertainty reports that assuming every targeted attack succeeds can alter the ranking of attack strategies; its model suggests peripheral attacks can become more damaging when attack execution is uncertain. This motivates a separate attack-delivery uncertainty track instead of treating all interventions as certain. <https://doi.org/10.1016/j.ress.2025.111231>

The deterministic plans model **structural intervention outcomes conditional on successful removals**. The implemented adaptive arm recalculates degree after each successful junction removal. The optional delivery profile models the probability that an attempted removal is enacted, but only as a synthetic per-attempt simulation parameter. Partial edge-capacity degradation, cascades, repair, adversarial learning and dynamically recalculated path-load/betweenness are not implemented. Those require explicit calibrated parameters or clearly labelled synthetic stress models.

## 6. Attack-delivery uncertainty simulation

The current deterministic attack plans specify which entities an attacker attempts to remove. The new optional delivery profile distinguishes an attempted removal from a successfully enacted removal.

API:
- `MycelialFailureDeliveryConfig`
- `build_mycelial_failure_delivery_profile(graph, demands, plan, config)`

Configuration:
- trial count per attack scenario;
- fixed-point per-entity removal-success probability from 0 to 1,000,000;
- deterministic seed.

For each scenario and trial, each attempted segment/junction removal is independently enacted with the configured probability. A versioned SplitMix64 draw is derived from the profile seed, scenario ID and trial index. Every trial stores its exact realized failed segment/junction IDs plus its structural/demand results. Per-scenario summaries include mean realized failure count, probability of at least one enacted failure, weighted reachability mean/P05/P50/P95/min/max and directness/path-length means among trials where demand remains reachable.

The profile validator:
- checks every scenario has the configured number of trials;
- recomputes the exact realized failed-ID lists from the PRNG and declared seed;
- re-runs the weighted graph evaluator for **every trial** from those realized failed IDs and compares reachability, directness, path length, component count and cycle rank;
- checks the trial ordering and uniqueness;
- recomputes all distribution summaries from the stored trials;
- binds source graph, demand inventory, attack-plan digest and probability/seed configuration.

Bounds:
- at most 1,000 trials per scenario;
- at most 10,000 total scenario-trial evaluations per profile.

### Conformance boundary

- 0% success probability must realize no removals and reproduce baseline reachability.
- 100% success probability must enact each attempted removal and match the deterministic attack evaluation.
- intermediate probabilities are a **simulation assumption**, not an empirical estimate, unless calibrated from observed delivery/failure data.
- current draws are independent across attempted entities; common-cause or correlated intervention delivery is not implemented.
- this profile models attack enactment uncertainty, not stochastic biological failure, organism survival or physical transport.
- a high failure probability setting does not mean a hypha has that probability of breaking; it means the hypothetical intervention has that chance of being enacted in the model.

The synthetic benchmark runs a fixed 0%, 25%, 50%, 75% and 100% delivery grid under common seed/plan inputs. The deterministic 100%-success arm remains the reference baseline.

## 7. Bounds

| Contract | Bound |
|---|---:|
| Junctions | 256 |
| Hypha segments | 512 |
| Demand pairs | 256 |
| Distinct demand sources | 32 |
| Failure scenarios | 768 |
| Attack-delivery trials per scenario | 1,000 |
| Attack-delivery scenario-trial evaluations per profile | 10,000 |
| Total integer demand weight | (2^{53}) |
| ID length | 128 bytes |

The single-failure sweep contains at most (V+E) scenarios: one per junction and one per segment. These bounds are intended for conformance/small-to-medium graph qualification. A larger-scale implementation should use a separately qualified incremental-dynamic-connectivity or batched shortest-path algorithm with equivalent receipts.

## 6. Synthetic ring-versus-tree control

The unit test compares a three-junction ring with a two-edge tree under removal of the same edge:

- the ring retains all declared demand pairs through the alternate route;
- the tree loses the demand pairs that depended on the removed bridge.

This is a deterministic structural conformance control, not a biological result. It demonstrates why cycle presence is only a hypothesis until the network is subjected to a defined failure.

The synthetic example compares a three-junction ring with the corresponding two-edge tree under the same bridge-edge failure. The ring retains all declared demand-pair reachability by using the alternate route; the tree disconnects the pairs that depended on the removed edge.

Run the synthetic benchmark:

```bash
cargo run -p symthaea-canvas --example mycelial_failure_resilience_bench -- 100
```

It reports the baseline and every single-junction/single-segment deletion, the weighted demand reachability/directness, profile digest and P50/P95 runtime.

## 7. Research benchmarks to pursue

The first source benchmark is the exact single-failure sweep. The broader programme should compare:

1. Random single-edge failures (distribution over seeds).
2. Random single-junction failures.
3. Degree-targeted and betweenness-targeted removal.
4. Articulation/min-cut-targeted removal.
5. Spatially correlated region failure, once the graph adapter retains calibrated curve geometry in a qualified coordinate frame.
6. Multiple simultaneous failures and common-cause events.
7. Recovery/repair: minimum added cost to restore a declared target reachability or directness threshold.
8. Temporal disruption: only after a time-respecting graph/flow model is measured.

Compare against static cycle rank, edge/node connectivity, articulation counts, minimum cuts, global directness, MST and Delaunay reference structures where assumptions match. Add both hub-targeted and peripheral/leaf-targeted removals so resilience is not judged only by centrality attacks.

All policies must use the same frozen source graph, demand pairs, failure events and failure seeds. Report baseline-to-failure deltas and full distributions, not one best-case robustness score.

## 8. Acceptance gates

Proposed, not achieved:

| Metric | Initial target |
|---|---:|
| Deterministic profile digest on frozen graph/demand/scenario inputs | 100% |
| Missing/duplicate failure IDs accepted | 0 |
| Demand pairs omitted from scenario outcomes | 0 |
| Weighted aggregates inconsistent with per-demand outcomes | 0 |
| Network path accepted below Euclidean endpoint distance | 0 |
| Source graph digest changes after derived failure evaluation | 0 |
| Independent small-graph reachability oracle disagreement | 0 |
| Attack plan replay/ranking agreement on frozen graph and demand inputs | 100% |
| Nested budget violations or unexplained scenario omissions | 0 |

## 9. Claim boundaries

The deterministic attack-family benchmark evaluates structural reachability and geometric shortest paths after fully enacted removals. The optional delivery-uncertainty profile evaluates simulated intervention-success uncertainty. Neither mode measures an actual biological failure probability. The programme does not yet simulate or measure:
- cytoplasmic flow;
- pressure/flow conductance;
- nutrient transport;
- biological organism viability;
- adaptation or regrowth;
- repair cost or repair success;
- spatiotemporal correlated damage.

A network can preserve all demand-pair connectivity while its true transport capacity collapses. Functional biological resilience therefore requires calibrated edge conductance or measured transport, independent intervention outcomes and recovery observations. The structural graph is a necessary but insufficient layer.
