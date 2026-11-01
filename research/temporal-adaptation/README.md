# Temporal Adaptation (candidate side)

**Package:** `symthaea-temporal-adaptation`  
**Tracking:** NEURO-SILICON-010 / #9872  
**Status:** research implementation; ten Rust test functions authored, not executed.

## Resource shadow prices

The package consumes the same twelve-dimensional resource vector as `symthaea-temporal-joint-control` and maintains a separate bounded shadow price for each dimension. It uses normalized utilization, configurable EWMA smoothing and a projected subgradient-style update:

`price[i] <- clamp(price[i] + step * (smoothed_utilization[i] - target[i]), 0, max_price)`

All arithmetic is integer/fixed-point. Configured units are parts per million. Each resource has its own target utilization. The maximum observed utilization is capped to bound single-episode influence and arithmetic.

A price is a resource-specific pressure signal. Do not sum prices across bytes, microseconds, nanojoules or operation counts. A subsystem may consume only its matching coordinate when generating candidate alternatives. The joint plan selector must still enforce every hard ceiling.

Zero budget is explicit: zero usage produces zero utilization; positive usage saturates the pressure signal. It never divides by zero.

## Persistence

`snapshot()` exports the complete bounded state. `restore()` validates persisted prices and smoothed utilization against the active configuration before use. A production caller must bind snapshots to the evaluator's signed receipt/checkpoint stream; this crate does not sign or authenticate snapshots.

## Limits

This is a bounded online control heuristic, not a proof of convergence or an optimal dual solution. Stability must be qualified across workload regimes and step sizes. Prices are advisory and never authorize a hard-budget overrun.

## Local checks

```sh
cargo test --manifest-path research/temporal-adaptation/Cargo.toml --locked
```
