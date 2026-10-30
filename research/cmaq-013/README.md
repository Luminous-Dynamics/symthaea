# CMAQ-013: Causal Advantage-of-Prediction Benchmark

**Status:** protocol v0.1 + dependency-free Rust synthetic harness; draft
**Tracking issue:** [#9289](https://github.com/Luminous-Dynamics/symthaea/issues/9289)
**Claim ceiling:** finite-state synthetic measurement study only; not an external benchmark, production result or SOTA claim

## Why this benchmark exists

A world model can have low prediction error and still fail to improve decisions. Conversely, a direct policy may perform well without any predictive mechanism. CMAQ-013 isolates the *causal contribution* of the transition model by comparing matched policies, the same candidate actions and paired episodes under learned, sham and oracle transition tables.

This operationalizes the advantage-of-prediction gap identified by the 2026 robotics benchmark survey: among 160 web-verified benchmarks, 11 contain explicit VLA/world-model contrasts and four turn predictions into executed actions. Those counts describe that survey's corpus, not every benchmark.

## Environment

Three deterministic finite-state tasks share the same five-state, three-action interface:

- **Delayed reward:** the highest immediate reward is not the best finite-horizon choice.
- **Trap avoidance:** a tempting immediate reward leads to a costly state.
- **Action coupling:** the value of an action depends on the next state's available actions.

The evaluator knows the reward table in v0.1. Only the transition table is predicted or corrupted, isolating the value of transition prediction. A future version should separately introduce reward-model error, partial observation, stochastic transitions and continuous control.

Each task has 12 nonterminal state/action transition cells. The learned transition table corrupts exactly 0%, 25%, 50% or 75% of those cells for each deterministic episode tape. All policies receive the same task, corrupted model tape, start state and reward table.

## Policy arms

1. **Direct greedy:** no transition model; chooses the action with the highest immediate reward.
2. **One-step model negative control:** consults the model but plans only one step. It should choose exactly the same action as direct greedy because the next state cannot affect a one-step return.
3. **Learned-model planner:** plans over the finite horizon using the corrupted transition table.
4. **Sham-model planner:** uses a deliberately action-permuted transition table. It is a negative control, not a candidate predictor.
5. **Oracle-model planner:** uses the true environment transition table; evaluator upper reference only.

Every method receives the same three candidate actions. Model-assisted policies use identical horizon and discount. No method receives extra reward information or a privileged action set.

## Required measures

Mean paired return delta versus direct greedy and a paired 95% interval; mean/CVaR95/max return regret versus the oracle planner; root action regret against true environment rollouts; learned-model prediction error; predicted-versus-realized optimism gap; fraction of decisions changed relative to direct greedy; and transition-model queries per decision.

Prediction quality and control value are intentionally distinct. The model can be accurate without producing a better action, and a model planner can make decisions worse when its model is misspecified.

## Falsifiable controls

- With zero transition error, learned-model planning must choose the same action and return as the oracle planner.
- The one-step model arm must match direct greedy exactly.
- The delayed-reward task must show a case where the perfect model improves return over direct greedy.
- Error fractions must be exact over the 12 transition cells.
- All policy arms must share paired model tapes and seed manifests.
- The sham model is reported separately and can never be used as evidence that learned prediction helped.

## Limitations

This is a deterministic finite-state test fixture with known rewards and synthetic corrupted transitions. The direct greedy policy is intentionally a small no-model control, not a tuned TD7/SAC/PPO competitor. External MRS.Q, TD-MPC2, DreamerV3 and TD7 comparisons belong in CMAQ-011 and require matched reproductions. A green focused workflow validates the synthetic measurement contract only.

## Next

Extend to noisy transition observations and learned reward models; add a development/hidden qualification split; port the matched action-set intervention to MuJoCo/DMC; then compare world-model planning against a strong model-free controller under identical interaction, compute and candidate-action budgets.