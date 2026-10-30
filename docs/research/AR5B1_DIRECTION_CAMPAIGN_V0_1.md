# AR-5B1 — matched-budget direction campaign v0.1

**Status:** initial synthetic campaign implementation authored; not compiled, executed, or statistically qualified.  
**Tracking issue:** [#9235](https://github.com/Luminous-Dynamics/symthaea/issues/9235)  
**Implementation:** `crates/domains/symthaea-architecture-engineering/src/direction_board_benchmark.rs`  
**Dependency:** [#9233](https://github.com/Luminous-Dynamics/symthaea/pull/9233) and [#8890](https://github.com/Luminous-Dynamics/symthaea/pull/8890)

## Research question

Does a verifier-grounded research-direction allocator improve discovery under a fixed evaluation budget, particularly when a promising direction has delayed payoff or the best mechanism is outside the current grammar?

This is an implementation-discrimination instrument. It is not a neural-network training benchmark, a real semantic novelty verifier, or evidence that Symthaea discovers new architectures better than existing systems.

## v0.1 campaign

The harness runs six policies over four evaluator-owned synthetic landscapes:

| Policy | Meaning |
|---|---|
| Score-only winner-takes-all | Continue with the direction holding the best observed score |
| Fixed round-robin | Rotate through existing directions and NEW |
| Random | Uniform seeded arm selection |
| Global UCB | Conventional UCB using global mean reward |
| Verifier-grounded direction board | Symthaea's direction-local record and explicit NEW arm |
| Oracle upper bound | Reads the complete synthetic payoff schedule; never deployable |

| Landscape | Falsifiable stress |
|---|---|
| Delayed breakthrough | A strong early leader saturates; a weak-starting direction pays off later |
| No-new-direction | The initial grammar already contains the best eligible mechanism; NEW exploration has a measurable opportunity cost |
| New-operator breakthrough | The strongest mechanism is outside the initial grammar and requires an admitted NEW direction |
| Rank reversal | The initial score leader degrades while a delayed direction becomes strongest |

All policies receive the same three-arm bootstrap and the same payoff schedule for a given landscape. Thereafter, they choose from the same legal action space. The unit action budget includes both a direction evaluation and a NEW proposal. Every v0.1 action costs one synthetic unit, so `action_count == total_cost_units == action_budget` is required for every run.

The exact deterministic payoff schedule is intentionally separated from policy randomness. This makes the oracle ground truth exact; only arm selection is seeded. Each run records the complete action trace, scores, NEW dispositions, cost, target-hit slot, simple regret and final direction-board digest.

## Paired comparisons

`run_direction_campaign_matrix` executes the policy × landscape × seed matrix. `aggregate_direction_campaign_runs` produces per-policy summaries. `compare_paired_direction_campaign_runs` compares policies by the exact same landscape, action budget and seed; missing or mismatched policy/seed cells are rejected rather than silently compared as independent samples.

Primary reporting fields are best verified synthetic score at fixed action budget, simple regret, target hit rate, actions to target (with non-hits retained as censored runs), and NEW-direction admissions. These are not collapsed into a single scalar leaderboard.

## What v0.1 does not yet test

- Variable evaluation costs or strict heterogeneous compute/energy budgets.
- Thompson sampling, discounted/sliding-window UCB, MAP-Elites/CVT-MAP-Elites, or a faithful Coral Discovery reproduction.
- More than one payoff template per family; four named families are only the first smoke surface.
- A genuinely independent semantic-equivalence verifier. The synthetic evaluator applies a planted rule to admit/reject a proposal; it does not infer novelty from code or mechanistic evidence.
- Multi-agent proposal-generation quality. Number of agents, proposal quality, model, prompts, tool access and compute must be independent factors in any later campaign.
- Real neural architecture training, held-out task transfer, or scientific-mechanism discovery.

The board's current scheduler also allocates evaluation-count slots, not heterogeneous-cost units. Cost-normalized yield is recorded but does not make the v0.1 allocation cost-optimal.

## Next confirmatory tranche

Extend the landscape generator to include early saturation, delayed breakthrough, steady-middle payoffs, rank reversal, descriptor aliases, true semantic novelty, cross-task tradeoffs, no-new nulls, grammar ceilings, non-identifiability and task shifts. Generate multiple independent payoff templates for each family.

Compare the current policies against:
- discounted/sliding-window UCB for nonstationary payoffs;
- Thompson sampling with explicit reward-distribution assumptions;
- MAP-Elites or CVT-MAP-Elites where behavior descriptors are predeclared;
- a pinned Coral Discovery-style direction-allocation implementation, documenting deviations from its paper;
- score-only, fixed, random, global UCB and an oracle upper bound.

Use at least 100 paired landscape/payoff instances across at least six independent families before confirmatory comparative claims. Determine the final seed count through a development-only power/variance study, then lock generators, seeds and analysis code. Resample at the landscape/payoff-template level; attempts within one schedule are repeated measures, not independent samples. Report paired uncertainty intervals, every failed/censored run, worst-family performance, NEW admission precision, false novelty, and opportunity cost in the no-new null.

## Provisional engineering targets (not achieved results)

These are proposed qualification goals for a later preregistered real-search campaign, not measurements from v0.1:

1. **Discovery efficiency:** at equal wall-clock/compute budget, target at least a 10% relative improvement in independently verified held-out capability or at least a 20% reduction in median cost to the first verified breakthrough across a majority of preregistered challenge families.
2. **Robustness:** do not accept a policy that wins on delayed-payoff tasks by causing an unacceptable loss on no-new-direction, rank-reversal or cross-task-regression families. Report the worst-family result.
3. **Novelty quality:** target at least 95% precision for admitted directions on a blinded semantic-equivalence challenge, and no more than 1% false admissions in adversarial alias controls. Calibrate these numbers with verifier reliability before locking them.
4. **Evidence integrity:** 100% reconciliation of action, proposal, verifier receipt, candidate/semantic-delta digests and declared cost; tampering, replay and missing-cost controls fail closed.
5. **Reproducibility:** identical pinned inputs and seeds reproduce byte-identical logical traces and results. Hardware performance claims additionally need repeated independent hardware runs and environment manifests.

A policy should not be called superior because it wins the planted delayed-payoff test alone. The no-new-direction control must expose exploration tax; a grammar-ceiling task must expose inability to leave the current search grammar; and a held-out task must test whether the new mechanism transfers rather than merely fitting the synthetic schedule.

## Relation to current research systems

Coral Discovery motivates direction-local records, explicit allocation across directions and a NEW arm. Its October 2026 preprint reports six wins, one tie and two losses against its strongest open CORAL comparison on nine matched tasks. That is a paper-reported result, not an independently replicated result here; this module is not an exact reproduction.

ShinkaEvolve, CodeEvolve and CORAL add complementary mechanisms: novelty-aware evolutionary search and proposal selection; island-based evolution and quality-diversity archives; and persistent shared memory with asynchronous multi-agent research. Symthaea should compare these at matched proposal, evaluator, compute and wall-clock budgets rather than infer superiority from different benchmark setups.

FIRE-Bench provides a useful complementary standard: evaluate whether an agent can formulate a research question, implement an experiment, execute it and produce an evidence-backed conclusion, not merely generate plausible code. NAS-Bench-201 zero-cost-proxy results are useful for ranking-prediction comparisons, but their proxy correlations on a fixed search space are not equivalent to open-ended architecture or paradigm discovery.

## Claim ceiling

A successful v0.1 run can demonstrate that the implementation obeys its synthetic action budget and produces deterministic audit traces. A successful confirmatory synthetic campaign could support a claim about allocation behavior on declared synthetic landscape families. Neither establishes better real neural architectures, scientific discovery, AGI, consciousness or universal research superiority.

The next real-search stage must connect direction-level proposals to typed architecture/operator candidates, causal interventions, independent held-out evaluators, resource receipts and reproducible transfer tests. Search must remain isolated from production cognition and repository merge authority.
