# Electric Power Series Oracle

A dependency-free Rust reference for integrating simultaneous electrical power channels over a frozen mission or test interval.

The oracle consumes **power samples** that have already been acquired and calibrated by an upstream measurement system. It does not derive power from raw voltage/current, synchronize independent clocks, interpolate asynchronous channels, repair missing samples or model component heat.

## Contract

Every channel carries:
- unique load ID and physical meter-boundary ID;
- typed load category;
- one declared clock-domain ID and clock uncertainty bound;
- source evidence-manifest digest;
- at least two power samples with monotonic integer-nanosecond timestamps;
- finite non-negative power and absolute uncertainty at every sample.

All channels in one evaluation must have the same clock domain and exactly the same timestamp grid. Clock uncertainty and inter-sample gaps are checked against caller-declared maximum bounds. A mismatch or unsupported gap fails closed; no resampling or gap fill occurs.

## Integration

Power and its supplied absolute uncertainty envelope are integrated with the trapezoidal rule. For adjacent samples:

`E_i = ((P_i + P_(i+1))/2) * delta_t`

`U_i = ((u_i + u_(i+1))/2) * delta_t`

The output contains nominal energy, absolute energy uncertainty, lower/upper energy bounds, average power and peak simultaneous channel power. Per-channel results preserve evidence, clock and meter identities.

The uncertainty is a deterministic absolute envelope under the supplied piecewise-linear sample model. It is not a Gaussian confidence interval. Clock uncertainty is gated and reported but is not turned into a statistical energy interval.

## Metering mode

A single battery-terminal-total channel is exclusive. It cannot be added to propulsion/TMS/auxiliary branch channels. In branch mode each supplied physical meter boundary must occur once. Category totals are optional: a missing category is unknown, not a measured zero.

## Important mission distinction

Integrated energy and peak power are separate outputs. Average power cannot establish that a battery, inverter or motor can satisfy a short climb or transition peak. The mission solver must retain instantaneous peak and duration-limited capability checks.

Electrical TMS power is a mission load. This crate does not convert it into battery heat. Physical heat-source allocation remains the responsibility of the thermal-loss ledger and the reusable thermal network.

## Verification

```bash
cargo fmt --manifest-path tools/electric-power-series-oracle/Cargo.toml -- --check
cargo test --locked --manifest-path tools/electric-power-series-oracle/Cargo.toml
cargo clippy --locked --manifest-path tools/electric-power-series-oracle/Cargo.toml --all-targets -- -D warnings
```

All fixtures are synthetic. Passing software tests does not validate sensor synchronization, calibration, physical energy measurement, component capability, aircraft performance, safety or airworthiness.
