# CMAQ-014: Dual-Control Probe Value Under Collateral State Disturbance

**Status:** protocol v0.1 + executable Rust synthetic benchmark; draft  
**Tracking issue:** [#9159](https://github.com/Luminous-Dynamics/symthaea/issues/9159)  
**Claim ceiling:** synthetic active-intervention stress test only; no physical safety, production or SOTA claim

## Research question

A probe can improve the agent's knowledge and simultaneously change the system being diagnosed. Should the agent maximize information gain, or choose a measurement whose information value is worth its cost and physical externality?

CMAQ-009 treated probes as independently authorized and parallel to ordinary decisions. That is useful for isolating information-channel value, but it omits a central property of physical and interactive perception: the sensing action itself can move, deform, damage or otherwise disturb the target. CMAQ-014 makes this externality explicit.

## Synthetic environment

Each episode has a hidden binary task state and a hidden asset-fragility indicator. The agent begins with no information about the binary state. It chooses one probe, receives a noisy observation, updates its decision, and executes the terminal task action.

Three lanes set the prior probability of a fragile asset to:
- robust assets: 1%;
- mixed assets: 20%;
- fragile assets: 60%.

The true per-episode fragility is not exposed to non-oracle policies. Probe modes trade signal accuracy (and its corresponding binary mutual information under a 50/50 prior) against direct cost and disturbance:

| Probe | Signal accuracy | Cost | Disturbance if robust | Disturbance if fragile |
|---|---:|---:|---:|---:|
| none | 50% | 0 | 0 | 0 |
| visual | 62% | 0.010 | 0 | 0 |
| gentle contact | 72% | 0.020 | 0.002 | 0.040 |
| force contact | 90% | 0.030 | 0.020 | 0.200 |
| invasive contact | 98% | 0.040 | 0.040 | 0.350 |

Disturbance at or above 0.12 causes terminal task failure. The risk-constrained policy uses a maximum predicted probability of such damage of 5% per episode. These values define a transparent synthetic benchmark; they are not empirical properties of real robots or materials.

## Policies

1. **No probe:** random prior decision; no sensing cost or disturbance.
2. **Visual only:** low-information, non-contact measurement.
3. **Information greedy:** chooses the most accurate probe without accounting for physical disturbance.
4. **Cost only:** maximizes mutual-information bits minus direct probe cost, still ignoring collateral disturbance.
5. **Disturbance aware:** chooses the probe with the highest expected terminal utility under the lane's public fragility prior, including damage probability and disturbance penalty.
6. **Risk constrained:** maximizes expected terminal utility subject to a hard 5% probability limit on crossing the damage threshold.
7. **Oracle fragility aware:** evaluator-only upper bound that knows the hidden per-episode fragility.

Each policy sees the same deterministic episode tape within a lane. The policy may know the lane prior, but only the oracle sees the current episode's hidden fragility.

## Metrics

Per lane and policy, report:
- terminal task success and raw decision accuracy;
- net utility after task outcome, probe cost and disturbance;
- mean collateral disturbance and direct probe cost;
- damage rate and declared risk-budget violation rate;
- probe accuracy and counts by probe type.

Mean binary mutual information (bits) is reported separately from empirical accuracy. Decision accuracy and task success are intentionally separate. The information-greedy policy may infer the hidden state correctly but still damage a fragile target enough to make the task fail.

## Prior art and research connection

- [Trajectory Planning for Safe Dual Control with Active Exploration](https://arxiv.org/abs/2604.15507) (April 2026) studies information-seeking actions under safety and mission-cost budgets. Its dual-gatekeeper approach motivates separating a soft information objective from hard risk and budget feasibility.
- [MS-MEM: Multi-Skill Manipulation-Enhanced Mapping via Uncertainty- and Disturbance-Aware Action Selection](https://www.max-robotics.com/en/research/publications/cmu42t65n00gvpd42fry8s5bt) (2026) combines viewpoint selection, pushing and grasping while constraining collateral scene disturbance. This is a particularly close conceptual comparator, though CMAQ-014 is not a reproduction.
- [Active Visual Perception: Opportunities and Challenges](https://www.mdpi.com/1424-8220/26/20/6544) (2026) discusses the risk that interactive sensing can alter or damage the scene and the need to evaluate the value of the acquired observation in relation to downstream task outcomes.
- PERSCO (October 2026) demonstrates active learning for perceived-safe robot control, using less obtrusive human feedback as part of the adaptation loop: [paper](https://link.springer.com/article/10.1007/s10514-026-10262-7).

## Falsifiable hypotheses

- Information-greedy probing achieves higher raw decision accuracy but can reduce terminal task success in fragile environments.
- Direct-cost-aware selection alone does not control collateral disturbance.
- Disturbance-aware selection improves net utility on fragile lanes at an explicit information-accuracy cost.
- The hard risk-constrained policy reports zero violations of its declared synthetic damage-probability bound.

## Limitations

- One probe and one terminal decision per episode; no multi-step belief-space planning yet.
- Probe signal accuracy, cost and disturbance distributions are known to the policies as fixed model parameters.
- The risk budget is evaluated against this synthetic fragility prior; it is not a formal physical safety guarantee.
- No correlated probe error, uncertain damage model, recovery action, irreversible asset value or physical contact dynamics.
- The oracle uses hidden fragility and is an evaluation ceiling, not a deployable policy.

## Run

    cargo test --locked
    cargo run --quiet --locked

A green focused workflow validates the deterministic synthetic contract only. The next step is an adapter to a contact-rich manipulation or active-mapping simulator, measuring actual scene-state changes, task completion and safety violations under matched probe budgets.
