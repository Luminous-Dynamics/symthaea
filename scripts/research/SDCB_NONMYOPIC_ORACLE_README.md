# SDCB exact non-myopic oracle

Standalone exact adaptive finite-horizon policy oracle for the DSSI decision model.

## Build and run

```bash
rustc --edition=2024 --test scripts/research/sdcb_nonmyopic_oracle.rs -o /tmp/sdcb-nonmyopic
/tmp/sdcb-nonmyopic
```

The source includes sdcb_decision_value.rs and therefore also runs its one-step reference tests.

## Exact search limits

- At most 8 candidate experiments.
- At most 4 experiments in a policy.
- At most 4 outcomes per experiment.
- Each experiment can be used at most once.
- The policy can stop early and choose a different next experiment after each outcome.
- Experiment cost is charged at every node where an experiment is selected.

This is an exhaustive small-instance oracle, not a scalable general-purpose planner. Its purpose is to provide exact values for measuring approximation regret on small tasks.

## Key control

Two experiments each have zero standalone EVSI under prior [0.4, 0.4, 0.2]. A one-step EVSI planner abstains. A two-step adaptive policy achieves expected terminal utility 0.85 instead of stopping at 0.80. With budget 1, it cannot execute both and the value stays 0.80.

10 additional Rust tests are authored. Rust compilation/execution has not been performed in this environment.