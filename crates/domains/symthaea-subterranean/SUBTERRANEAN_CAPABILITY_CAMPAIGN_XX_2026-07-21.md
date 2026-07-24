# Subterranean Capability Campaign XX

## Graceful degradation under competing obligations

Campaign XX extends resource-conflict accounting with explicit liveness recovery. The previous campaign could detect scarcity and service debt, but repeated allocations could remain internally valid while making no operational progress.

## Patch scope

- Bounded material-progress frames.
- Hysteretic reserve rebalancing.
- Persistent arbitration-deadlock detection.
- Safety-monotonic mission shedding.
- Return-or-hold recovery authority.
- Same-frame deployed command constraint.
- Checkpoint schema 13.
- Five deterministic validation contracts.
- Independent-review evidence bundle.

## Key rule

Protected objectives are never sacrificed to manufacture liveness. When progress stalls, discretionary work is shed first. Persistent deadlock then selects protected return when feasible, otherwise a review hold.

## Remaining qualification

The standalone archive does not contain the real Symthaea core and FEP dependencies. Authoritative acceptance requires Rust 1.94 workspace compilation, Clippy, complete tests, calibrated progress thresholds, long-horizon simulation, and HIL deadlock campaigns.
