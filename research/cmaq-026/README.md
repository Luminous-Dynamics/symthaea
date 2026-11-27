# CMAQ-026 — Decision-Sufficient Active Probing

**Status:** finite Rust evaluator and deterministic tests authored; exact-head format/tests/Clippy pending.  
**Relationship:** decision-utility extension to CMAQ-021/023/025; standalone dependency-free crate.  
**Claim ceiling:** finite deterministic minimax-regret witnesses. No calibrated probability, optimal active-learning, external benchmark, or physical-safety claim.

## Research question

CMAQ-023 asks the candidate to choose a probe without evaluator hints. CMAQ-025 makes observational aliasing and response identifiability explicit. CMAQ-026 asks when a remaining distinction is worth resolving for the decision currently being made.

Two worlds can disagree about future response but still share the same invariant-safe optimal action. Distinguishing them has no decision value. Conversely, a small hypothesis set can be decision-critical when its worlds prefer different actions and an irreversible boundary is near.

## Candidate/evaluator separation

The candidate request is one of:
- Act(action ID);
- Probe(probe ID, action policy keyed by the probe's candidate-visible observation);
- Unknown(safe fallback action).

The evaluator-only world records contain action losses, invariant admissibility, environment-family and independent-cluster identifiers, and the observation each registered probe would return in each world. CandidateDecision has no world ID or family ID field. A probe policy must cover exactly the candidate-visible observation values; it cannot key on hidden world identity.

All action losses and probe costs use one explicit non-negative loss unit defined by the frozen contract. The first version uses no probabilities.

## Decision rule

The no-probe baseline is the minimum over single actions of their worst-case regret across worlds, restricted to actions that satisfy hard invariants in every world.

The v0.1 scorer requires at least one common invariant-safe action across the finite world set (normally an explicit no-mutation/hold fallback). If none exists, it returns `NoCommonInvariantSafeAction` rather than inventing a safe baseline or averaging away the conflict. This is a fail-closed boundary; modelling emergency-stop dynamics and cases with no common safe action is a later extension.

For a candidate probe, compute the worst-case regret of its observation-conditioned action policy and add the probe cost:

`probe_adjusted_regret = worst_case_policy_regret + probe_cost`

Then:

`decision_value = best_no_probe_worst_case_regret - probe_adjusted_regret`

Positive decision value means the selected probe policy improves over the best robust no-probe action after paying its stated cost. Zero or negative value means the probe did not justify its cost under this finite contract. If the candidate violates any invariant, candidate regret/value is unavailable and the violation count remains a hard failure.

The score also reports whether all worlds share one optimal action, the number of probe-observation partitions, and whether every observation bucket has a common optimal action across its worlds.

This is a deterministic finite regret calculation, not expected value of information under a probability distribution.

## Deterministic witnesses

1. High outcome uncertainty but same optimal action: best no-probe regret is zero; probing has negative value after cost.
2. Two possible worlds with different optimal actions: a one-unit probe resolves the distinction and yields positive net decision value.
3. A probe can produce three observation partitions and still be decision-irrelevant.
4. A cheap non-separating probe is charged and does not resolve action disagreement.
5. A two-partition inexpensive probe can dominate a higher-information three-partition probe when both yield the same action regret.
6. Any invariant violation is surfaced separately and invalidates candidate regret aggregation.
7. Probe policies are indexed by observed values, not evaluator-only world IDs.

## Known limitations

- The scorer enumerates a finite evaluator world set; it does not establish that the set is complete.
- Probe outcomes are deterministic in v0.1. No observation noise or probabilistic calibration is implied.
- The regret unit combines action loss and probe cost by explicit contract declaration; a deployment must justify that common scale.
- This crate scores a submitted candidate decision/policy. It does not implement probe execution, a live controller, or action authorization.
- Digests are syntax-checked only; independent byte-level recomputation and authenticated receipts remain outside this crate.
- Independent environment/mechanism clusters are the statistical unit; the deterministic fixtures are software witnesses, not statistical evidence.

## Next integration

Connect CMAQ-026 to CMAQ-023 blind probe acquisition and CMAQ-025 identifiability output. Compare random, entropy-greedy, partition-count, evaluator-hinted and decision-regret-per-cost acquisition at matched probe budgets. Then use CMAQ-016 to test how planner search changes selected-action regret under controlled optimistic model errors.
