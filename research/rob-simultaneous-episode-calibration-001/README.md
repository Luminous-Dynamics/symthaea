# ROB-SIMULTANEOUS-EPISODE-CALIBRATION-001

Owner: Symthaea issue #11655. Consumer: decision-sufficiency reference in #11646.

This crate implements finite-sample split-conformal calibration of expected-loss intervals over a **complete frozen action-domain × decision-horizon episode**.

## Why the calibration unit is the episode

The decision-sufficiency evaluator compares a selected action with the minimum lower bound among all admissible alternatives, and then keeps a prefix over future ticks. Marginal per-action or per-tick intervals do not automatically retain their nominal coverage after those selection operations.

The calibration reference therefore:
1. freezes one task/loss/model/action-domain/horizon/stratum scope;
2. requires every calibration episode to provide every candidate action at every tick;
3. calculates one maximum interval nonconformity score per independent episode;
4. takes a finite-sample split-conformal order statistic across episodes;
5. expands every future action/tick interval by the same calibrated amount.

For episode e, candidate action a and tick t, with predicted interval [l,u] and an independently adjudicated reference expected-loss interval [r-,r+]:

    S_e = max_(t,a) max(0, l-r-, r+-u)

The reference target must be an exact finite-oracle expectation or a separately qualified interval that contains the expected task loss. One noisy realized outcome is not sufficient ground truth for an expected-loss predictor.

For n exchangeable complete calibration episodes and miscoverage alpha:

    k = ceil((n + 1) * (1 - alpha))
    q = S_(k), the k-th smallest episode score

If k exceeds n, no finite conformal expansion is available. The reference returns an explicit insufficient-sample error; it never manufactures q=0.

Under exchangeability of **complete independent episodes**, the resulting interval expansion has finite-sample marginal coverage at least 1-alpha simultaneously across the frozen action set and horizon. Dependence within an episode is handled by the maximum score; it is not treated as additional calibration sample size.

## Fixed-point and scope rules

- Expected task loss uses caller-declared unsigned integer units.
- The action IDs are an exact frozen manifest; every action/tick cell appears exactly once per episode.
- Horizon ticks are an exact contiguous interval.
- Candidate action count, horizon, action-domain digest, loss model, task-loss scale, stratum and dataset split are bound by the supplied scope digest.
- The implementation checks digest identity but does not cryptographically compute or attest the upstream digests.
- One complete episode contributes one score, regardless of its number of ticks or actions.
- Different morphology, task, loss, hardware, policy or environment strata require separate scopes when exchangeability is not defensible.

## Use and limitations

The resulting SimultaneousCalibrationReceipt carries its action-domain digest, loss-model and loss-scale identities, scope and split digests, sample count, alpha/coverage, exact action count, horizon length, order-statistic rank and expansion quantile.

The interval-expansion function refuses a different scope, action or tick. Lower bounds are floored at zero. Upper-bound addition uses saturating arithmetic and explicitly reports when the representable u64 maximum is reached; callers must treat that as an extremely wide bound, never as a precise finite prediction.

The reference does not establish exchangeability, verify independent ground truth, estimate task loss, train a model, prove action admissibility, validate a physical plant or authorize actuation. A held-out coverage study is required before deployment.

## Run

    cargo fmt --manifest-path research/rob-simultaneous-episode-calibration-001/Cargo.toml -- --check
    cargo test --locked --manifest-path research/rob-simultaneous-episode-calibration-001/Cargo.toml
    cargo clippy --locked --manifest-path research/rob-simultaneous-episode-calibration-001/Cargo.toml --all-targets -- -D warnings
