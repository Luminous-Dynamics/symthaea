# Prefix-Only Action Trajectory Replay V1

**Status:** first implementation slice for [#9790](https://github.com/Luminous-Dynamics/symthaea/issues/9790). It is a deterministic replay boundary and baseline harness, not a causal effect estimator or a full intervention benchmark.

## Boundary

`ActionTrajectory` is the sealed full trace. It contains all ordered action receipts and terminal outcome/evaluator/verifier receipts. It is held by the replay engine and evaluator.

`PrefixView` is a separate value containing only the observed action history, cumulative spend, last observed action cost, current pre-state/frontier digest and features, current candidate-pool digest and admissible operation families, and current remaining budget. It has no suffix actions, suffix costs, eventual terminal type, evaluator receipt, or completion verifier.

`PrefixReplayEngine` constructs a `PrefixView`, calls `PrefixForecaster::predict`, and only after the forecast is returned derives a `PrefixOutcomeLabel` from the sealed trace suffix. This interface makes suffix leakage harder to introduce accidentally; it is not a cryptographic sandbox against a malicious forecaster implementation.

## Trace invariants

- Action sequence is contiguous and one-based.
- Each action has a non-empty operation ID and nonzero candidate-pool, pre-state, post-state, frontier, and meter-receipt digests.
- Candidate operation-family IDs are sorted, unique, and contain the selected operation family.
- State digests form an exact post-state-to-next-pre-state chain.
- Proof-frontier features pass the typed frontier validator.
- A sufficient-proof terminal requires an evaluator receipt, completion verifier receipt, and a final independent-verifier action.
- Censored, competing, and pending traces do not fabricate a successful verifier record.
- Resource accumulation uses checked arithmetic; budget exhaustion is explicit and does not wrap.

## Outcome labels

- `VerifiedRemainingCost`: full measured suffix cost through the final independent verifier action, with verifier digest.
- `RightCensoredLowerBound`: measured suffix work up to timeout/interruption/budget censor, with censor reason.
- `CompetingTerminal`: measured suffix work to permission revocation/abort/abandonment, not a proof-completion label.
- `Pending`: observed suffix work so far, with no terminal label.

## Initial baseline

`LastObservedActionForecaster` uses the most recent action cost, or a declared cold-start cost, multiplied by unresolved obligation count plus mandatory verifier operations. q90 and q95 are fixed declared multiples. It is intentionally simple and uncalibrated. Its purpose is to establish an executable comparator and to test the replay contract.

`HistoricalCostBaseline` also implements `PrefixForecaster`, allowing the disjoint task-family historical model from #9752 to run through exactly the same prefix interface.

## Required leakage controls

1. Change a suffix action cost while preserving the prefix pre-state; the earlier `PrefixView` and forecast must remain identical, while its offline label changes.
2. Change the eventual terminal type while preserving the same observed prefix; the earlier `PrefixView` remains identical while its offline label changes.
3. Verify that successful remaining-cost labels include the final independent-verifier action cost.
4. Reject broken state chains, noncanonical candidate pools, missing meter receipts, missing verifier receipts, and invalid terminal/action combinations.

The deterministic CLI matrix emits three `TRAJECTORY_REPLAY_ROW` rows: verified prefix count/cost; suffix-cost leakage negative control; and terminal-label isolation. Unit tests include replay-engine, budget, label, state-chain, candidate-pool, and arithmetic checks.

## Scope limits

This does not yet provide counterfactual estimates for alternative operations. It records the admissible candidate pool for each observed action, but V1 contains only the action actually taken. Causal operation-effect comparisons require matched pre-action states and either randomized assignment or a defensible conditional exchangeability design. Observational differences must not be described as causal.

The first baseline's multipliers are engineering placeholders, not calibrated quantiles. The historical baseline's empirical quantiles remain descriptive. The feature branch's deterministic trace tests establish the interface invariants, not prediction performance.

## Next increments

1. Add trajectory-level aggregates for cost-to-verified-proof, proof progress, censoring, and competing events.
2. Add paired alternative-action simulator runs from identical pre-state digests.
3. Add a frozen action-effect prediction interface and compare predicted resource/proof-progress deltas to paired outcomes.
4. Add full rollout then prefix replay over multi-step, repeated-seed task families.
5. Compare historical, last-observed-action, frontier-linear, split-vector conformal, selected-policy/CAP where valid, and oracle lower bound on identical prefixes.
6. Add confidence intervals, worst-family reporting, forecast overhead, and APB-style broken-tool/extraneous-tool/unsolvable-task strata.