# ROB-DECISION-SUFFICIENCY-HORIZON-001

Owner: Symthaea issue #11646. Parent contract: #11581.

This standalone crate evaluates a finite sequence of evidence-bound expected-loss intervals and returns the longest contiguous prefix for which a previously selected action remains sufficiently good and robustly admissible.

## Core rule

For the previously selected action a0 and each future tick t, consume:

- an upper bound U0(t) on its expected task loss;
- a lower bound L_i(t) for each robustly admissible alternative;
- explicit admissibility evidence for every candidate action;
- a complete action-set assertion;
- a matching loss scale and loss-model digest;
- a calibration identity, independent validation digest, coverage level and expiry.

The conservative regret upper bound is:

    R_UCB(t) = max(0, U0(t) - min_i L_i(t))

When there is no admissible alternative, regret is zero only if the action set is complete and a0 is proven admissible. Unknown alternative admissibility never means that the alternative set is empty.

The tick remains decision-sufficient when a0 is robustly admissible and R_UCB(t) is no greater than the task-specific epsilon. The horizon stops at the first insufficient or unknown tick. Tick indices must be contiguous.

The output is a decision-sufficiency measurement. It is not an action-prefix physical bound, a controller permit, a runtime-assurance certificate or actuator authority. Compose its resulting prefix with the existing hard-bound rule in #11581:

    N_commit = min(N_hard, N_decision)

## Fixed-point units

Loss values are unsigned integer units within one declared loss-scale identity. The caller chooses and versions the unit scale (for example, micro-loss units). This reference avoids floating-point ordering and makes the threshold boundary exact.

## Evidence limitations

This crate does not estimate beliefs, predict states, calculate expected losses, calibrate confidence bounds or prove robust admissibility. The caller supplies these values and their evidence identities. Synthetic tests qualify only the deterministic evaluator semantics. Independent held-out calibration, physical admissibility and viability remain separate obligations.

## Run

    cargo fmt --manifest-path research/rob-decision-sufficiency-horizon-001/Cargo.toml -- --check
    cargo test --locked --manifest-path research/rob-decision-sufficiency-horizon-001/Cargo.toml
    cargo clippy --locked --manifest-path research/rob-decision-sufficiency-horizon-001/Cargo.toml --all-targets -- -D warnings

## Benchmark controls

Compare fixed-K, fixed-AoI, progress-triggered replanning, PACE-style phase/kinematic execution, the existing #11581 hard-bound minimum, decision-sufficiency alone, and the combined hard-bound plus decision-sufficiency policy. Keep EVSI (value of a new observation) separate from stale-action regret (cost of continuing the old action).

Required counterfactuals include an old but decision-equivalent observation, a fresh but decision-changing observation, an admissible but regretful action, an incomplete alternative set, and an unknown calibration. Report false-continue, false-replan, realised regret, upper-bound coverage, replanning/inference rate, safe-success and all adverse outcomes separately.
