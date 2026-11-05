# Robust query disambiguation under set-valued observation uncertainty V1

**Tracking:** [#9987 — value of next evidence / scientific review prioritization](https://github.com/Luminous-Dynamics/symthaea/issues/9987)  
**Prior deterministic planner:** [#10505 — deterministic query-disambiguation set cover](https://github.com/Luminous-Dynamics/symthaea/pull/10505).  
**Related:** [#9344 — query robustness across plausible reconstructions](https://github.com/Luminous-Dynamics/symthaea/issues/9344), [#9255 — query-scoped reliability certificates](https://github.com/Luminous-Dynamics/symthaea/issues/9255).  
**Tool:** connectome-robust-query-disambiguation-audit.  
**Status:** standalone Rust exact planner, schema, known-answer fixtures, independent oracle and isolated CI authored. This is deterministic set-valued robustness, not probabilistic VOI.

## 1. Why V1 differs from deterministic point-outcome planning

The deterministic planner in #10505 assumes one declared categorical outcome per world/action pair. That is useful for a frozen observation rubric with deterministic predictions, but microscopy and adjudication often have ambiguity, measurement-resolution limits, multiple valid interpretations or indeterminate states.

A point prediction can falsely suggest discrimination. Example:
- World A predicts edge-present.
- World B predicts edge-absent.

That looks decisive if the observation model is deterministic. But if either world can also produce indeterminate, the actual observation may not resolve the hypotheses.

This contract replaces each point outcome with a set of all permitted outcome categories under a frozen, conservative observation model.

## 2. Robust separation rule

For action a and world w, let O_a(w) be the declared set of possible categorical outcomes.

An action robustly separates worlds w_i and w_j only when:

    O_a(w_i) intersect O_a(w_j) = empty set

If the sets overlap, at least one outcome category is compatible with both worlds; the action cannot guarantee a distinction and receives no coverage credit for that pair.

The query-conflict universe remains all pairs (w_i,w_j) where the frozen query decision differs.

The exact planner finds the minimum-effort action set that robustly separates every pair in this conflict universe. V1 uses bounded subset enumeration: at most 20 observation actions and 24 worlds (276 possible unordered pairs, at most 144 opposing-decision pairs).

This is intentionally conservative. A selected set separates a pair when at least one selected action is individually robustly separating for that pair. We do not infer that several overlapping noisy outcomes jointly identify a world through an unmodelled correlation structure.

## 3. Result states

- MINIMUM_COST_ROBUST_DISAMBIGUATION_PLAN: every opposing-decision pair has at least one selected action whose possible-outcome sets are disjoint, and minimum effort is within budget.
- MINIMUM_COST_ROBUST_DISAMBIGUATION_EXCEEDS_BUDGET: a complete robust plan exists, but exceeds the declared effort budget.
- NO_ROBUST_DISAMBIGUATION_WITHIN_DECLARED_ACTIONS: at least one opposing-decision pair has overlapping outcome sets for every supplied action. The pair is explicitly returned.
- NO_DECISION_DISAGREEMENT_WITHIN_DECLARED_SCENARIOS: all submitted worlds agree on the query decision. This is scoped to the submitted world set.

No result is a probability, posterior, entropy or expected-value claim. A no-plan result means only that robust separation is unavailable using these declared action models and worlds.

## 4. Evidence and measurement contract

Every action binds:
- stable action ID, action kind and positive integer effort;
- source-evidence digest;
- observation-model digest;
- shared evidence/dependence ancestry groups;
- exactly one non-empty possible-outcome set for every declared scenario.

Possible-outcome sets are not likelihoods. The observation-model digest identifies the frozen declaration; it does not prove that the model captures every real measurement outcome. The action and source digests do not prove acquisition provenance, reviewer compliance or biological validity. Shared ancestry groups are retained but no statistical independence is inferred.

If the true measurement support set is underdeclared, the computed robust-separation guarantee is unsound. Thus each set-valued observation model requires independent scientific review and challenge cases before biological use.

## 5. Known-answer fixture

Three worlds are declared: alpha supports the query; beta and gamma do not. There are two decision-conflict pairs.

- Synapse inspection has disjoint possible outcomes for alpha-beta, but alpha and gamma share a possible outcome, so it cannot robustly separate alpha-gamma.
- Morphology review has overlapping possible outcomes for alpha-beta, but disjoint sets for alpha-gamma.
- An orthogonal measurement has disjoint sets for both pairs but costs five effort units.
- A low-resolution decoy has identical support sets for every world.

Two minimum plans therefore combine synapse inspection and morphology review, for a total effort of four units. There are two tied plans at this cost (one alternative synapse method plus morphology). The five-unit orthogonal measurement is a valid but non-minimal single-action plan.

This demonstrates why point-outcome discrimination and robust set-valued discrimination are not interchangeable.

## 6. Negative control

The no-plan fixture contains a positive and negative world. The only available image inspection admits an indeterminate outcome under both worlds:

- supports-world: {edge-present, indeterminate}
- refutes-world: {edge-absent, indeterminate}

The sets overlap, so the action does not robustly separate the pair. The planner must return NO_ROBUST_DISAMBIGUATION_WITHIN_DECLARED_ACTIONS and list the pair.

## 7. Independent verification

The Python standard-library oracle:
- recomputes scenario-set and action-set digests;
- rebuilds opposing-decision world pairs;
- checks that every possible-outcome set is non-empty, unique and complete;
- recomputes action coverage using set disjointness;
- enumerates subsets with itertools.combinations rather than Rust bit masks;
- verifies exact minimum effort, tied plans, budget state and undiscriminable pairs;
- rejects deliberately corrupted costs, coverage, plan membership and no-plan status.

## 8. Benchmarks and promotion criteria (proposed, not results)

| Metric | Proposed target |
|---|---:|
| Exact robust-plan recovery on finite cases (≤20 actions) | 100% |
| Regret against independent exact oracle | 0 |
| False robust separation under overlapping possible-outcome sets | 0 |
| Missed opposing-decision pairs | 0 |
| Undiscriminable pairs explicitly reported | 100% |
| Deterministic output under action/world/outcome ordering permutations | 100% |
| Unsupported probability / independence claims | 0 |

Compare deterministic point-outcome planning (#10505) with robust support-set planning on the same worlds and actions. The robust planner should never claim more pair separation than the point planner unless a separate joint observation model explicitly justifies it. Measure coverage lost to ambiguity, effort increase needed for robust plans, and frequency of NO_ROBUST_DISAMBIGUATION under broadened support sets.

A later probabilistic extension must introduce calibrated likelihoods, observation noise, dependence across actions, and a declared scientific decision-loss function. It must not reinterpret support-set membership as equal probability.

## 9. Integration boundaries

- #9987: robust deterministic next-evidence discrimination under ambiguous outcome support.
- #10505: point-outcome deterministic planner; this tool is a conservative set-valued complement.
- #9344: supplies plausible reconstruction worlds and query result variations.
- #10502: supplies query-flipping witnesses that can motivate what observations to model.
- #9255: a future certificate may bind query/scenario/action/model digests, robust pair coverage, minimum plan and verifier receipt.
- #10406, #10435 and #10453: evidence dependence, frame closure and execution-route observability remain distinct conjunctive gates.

## 10. Non-claims

The tool does not prove that possible-outcome sets are complete, that measurement models are calibrated, that actions are independent, that an observation will be performed, or that submitted reconstruction worlds exhaust biological alternatives. It computes robust separation only relative to the exact supplied finite support sets and action universe. No probability or biological claim is produced.
