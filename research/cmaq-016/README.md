# CMAQ-016: Adaptive Search Exploitability

**Status:** synthetic mechanism benchmark v0.1; draft  
**Tracking issue:** [#9396](https://github.com/Luminous-Dynamics/symthaea/issues/9396)  
**Claim ceiling:** synthetic optimizer-exploitation study only; no trained model, external simulator, robot or SOTA result

## Research question

Can an adaptive optimizer systematically discover and exploit narrow model-error regions that are rare under behavior-policy evaluation, and can conservative search reduce this gap without collapsing to direct fallback?

The benchmark isolates an optimizer-in-the-loop effect. It is not another aggregate prediction score and does not replace CMAQ-013's selected-action calibration work.

## Mechanism

The hidden one-step environment has a smooth true reward landscape with a known optimum. In the exploit lane, the learned model adds a narrow optimism pocket away from the true optimum:

```
true return:          smooth, maximum at x = 0.20
predicted return:     true return + narrow positive bias near x = 0.72
```

Random-action prediction error can remain small because the pocket occupies little action-space volume. A greedy optimizer can nevertheless find it and select an action with worse real return. This creates a concrete counterexample to treating global prediction MSE as a sufficient world-model quality measure.

The first implementation uses:
- `benign_iid_noise`: small deterministic residuals, no structured optimism pocket.
- `narrow_optimism_pocket`: localized predicted-return overestimation.
- `random_search`: uniform candidate proposals; select the best model score.
- `adaptive_cem`: iterative cross-entropy-style proposal updates toward high model scores.
- `pessimistic_ensemble_cem`: same adaptive optimizer using the minimum of five deliberately decorrelated error-pocket heads.
- `direct_policy`: fixed non-optimized baseline action.
- `oracle`: evaluator-only true optimum.

The five ensemble heads are explicitly synthetic with separated error-pocket centers; that is not evidence that arbitrary learned ensembles are independent or epistemically calibrated.

## Metrics

For every scenario × candidate budget × policy, the report emits:

- mean true selected return and regret from the known optimum;
- mean reference-model selected return (the central point-predictor head);
- selected-action optimism gap;
- selected-action error;
- fraction of episodes selecting the predeclared exploit pocket;
- total candidate query budget and mean queries per episode (zero for direct/oracle controls), plus episode count.

The generator also records the scenario's uniform-grid prediction MSE as a descriptive diagnostic. It is not substituted for selected-action error or realized task utility.

## Execution

```sh
cd research/cmaq-016
cargo test --locked
cargo run --release --locked > report.json
```

The crate is standalone (`[workspace]`) and has no external Rust dependencies. Rust 1.96.0 is pinned in the workflow.

## Interpretation limits

This is a deterministic scalar mechanism test. It does not implement physical dynamics, reward learning, multi-step state transitions, CEM/MPPI production implementations, a real ensemble-training recipe, or a MRS.Q reproduction. The first-stage harness asks whether an adaptive optimizer can exploit a known localized optimistic error. Follow-up versions must cross error families with temporal correlation, horizon, state-conditioned pockets, adaptive drift and actual external simulators.

A benchmark pass means the harness compiles, replays deterministically and checks the expected synthetic controls. It does not mean Symthaea outperforms an industry baseline or is safe for physical control.
