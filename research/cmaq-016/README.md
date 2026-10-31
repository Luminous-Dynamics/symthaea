# CMAQ-016: Adaptive Search Exploitability

**Status:** synthetic mechanism benchmark v0.1; draft  
**Tracking issue:** [#9396](https://github.com/Luminous-Dynamics/symthaea/issues/9396)  
**Claim ceiling:** synthetic optimizer-exploitation study only; no trained model, external simulator, robot or SOTA result

## Research question

Can an adaptive optimizer systematically discover and exploit narrow model-error regions that are rare under behavior-policy evaluation, and can conservative search reduce this gap without collapsing to direct fallback?

This isolates an optimizer-in-the-loop effect. It complements CMAQ-013's selected-action calibration tests: CMAQ-013 asks whether uncertainty estimates cover selected actions; CMAQ-016 measures the realized utility loss when adaptive search finds and selects a model-error pocket.

## Mechanism

The hidden one-step environment has a smooth true reward landscape with a known optimum at `x = 0.20`. In the exploit lane, the point-predictor adds a narrow optimistic prediction bump near `x = 0.72`:

```
true return:       maximum at x = 0.20
point prediction:  true return + narrow optimism near x = 0.72
```

The pocket occupies little action-space volume, so uniform-grid prediction MSE can remain deceptively small. The optimizer is nevertheless incentivized to find it.

Two deterministic lanes:
- `benign_iid_noise`: small residual noise and no structured optimism.
- `narrow_optimism_pocket`: a narrow point-model overestimate away from the true optimum.

Policy arms:
- `direct_policy`: fixed non-optimized action; no model queries.
- `random_search`: uniform candidate proposals; select the best point-model score.
- `adaptive_cem`: iterative cross-entropy-style proposal updates toward high point-model scores.
- `pessimistic_ensemble_cem`: the same adaptive optimizer using the minimum of five synthetic heads with deliberately separated error-pocket centers.
- `oracle`: evaluator-only true optimum.

The point predictor is the central synthetic head. The other four heads exist only for the pessimistic comparator. Their deliberately shifted error pockets are an ablation, not evidence that arbitrary learned ensemble heads are independent or epistemically calibrated.

## Budgets and metrics

Each policy is evaluated at candidate-query budgets `8, 32, 128, 512` over 256 deterministic paired episodes per scenario. Search arms use exactly the declared candidate budget; direct and oracle controls use zero model candidate queries.

For each scenario × budget × policy, the JSON report emits:
- mean true selected return and regret from the known optimum;
- mean reference-model selected return (the central point-predictor head);
- optimizer optimism gap and selected-action error;
- fraction of episodes selecting the predeclared exploit pocket;
- candidate-query count per episode and uniform-grid prediction MSE.

The prediction MSE is descriptive only. It is never substituted for selected-action error, realized utility or regret.

## Execution

```sh
cd research/cmaq-016
cargo test --locked
cargo run --release --locked > report.json
```

This standalone crate has no external Rust dependencies and pins its lockfile. Rust 1.96.0 is selected by the focused workflow.

## Interpretation limits

This is a deterministic scalar mechanism test. It does not implement physical dynamics, reward learning, multi-step state transitions, production CEM/MPPI implementations, a real ensemble-training recipe or a MRS.Q reproduction. The first-stage harness asks whether an adaptive optimizer can exploit a known localized optimistic error.

Follow-up versions should cross error families with temporal correlation, horizon, state-conditioned pockets, changing regimes, reward/termination misspecification and pinned external simulators. A synthetic mechanism pass does not establish physical safety, SOTA or external-environment performance.
