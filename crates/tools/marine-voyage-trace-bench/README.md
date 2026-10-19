# Marine voyage trace benchmark (V0)

This dependency-free Rust crate converts a canonical synchronized trace of propulsion and auxiliary electrical input power into separate and total energy bands. It is the data-collection bridge for [MARINE-TRACE-001](https://github.com/Luminous-Dynamics/symthaea/issues/8102) and is intended to compose with [MARINE-ENERGY-BUDGET-001](https://github.com/Luminous-Dynamics/symthaea/issues/8092).

## Run

    cargo test --locked --offline --manifest-path crates/tools/marine-voyage-trace-bench/Cargo.toml

    cargo run --locked --offline --manifest-path crates/tools/marine-voyage-trace-bench/Cargo.toml --example synthetic

The example uses synthetic values only. No result from it is physical evidence.

## What it does

- Requires synchronized propulsion and auxiliary electrical input power in watts at one declared energy-boundary convention.
- Requires start/end coverage, strictly increasing integer-millisecond timestamps, a declared maximum sample gap, a source-trace SHA-256 identifier, mission/condition identifiers, and an explicit absolute-bound method identifier.
- Integrates between adjacent samples using the trapezoidal rule (piecewise-linear interpolation).
- Carries nominal and conservative lower/upper watt-hour totals for propulsion, auxiliaries and the whole electrical boundary.
- Returns Indeterminate for missing channel measurements, incomplete endpoints, or gaps larger than the declared limit; returns Invalid for malformed metadata, invalid numbers or nonmonotonic timestamps.

The trace digest is only a caller-supplied identifier. This crate does not authenticate the source file or sensor, and it does not correct measured data for wind, waves, current, depth, draft or loading.

## Energy and uncertainty contract

For consecutive sample powers P0 and P1, duration delta-t in milliseconds, energy is:

    E_Wh = ((P0 + P1) / 2) * (delta-t / 3,600,000)

The absolute per-sample bound is propagated by integrating its lower and upper power envelopes. Lower power is bounded at zero because this schema represents nonnegative electrical consumption. These are declared arithmetic bounds, not probability distributions or calibrated confidence intervals. Correlation, unobserved transients and sensor bias are not inferred by the crate.

A missing auxiliary channel is never silently interpreted as zero. An over-limit gap is never integrated across. The maximum gap must be set from the acquisition plan before looking at results.

## How this fits marine validation practice

The current ITTC procedure for ship speed/power trials is [7.5-04-01-01.1, Revision 08 (effective 2024)](https://www.ittc.info/media/11972/75-04-01-011.pdf). ISO 15016:2025 defines preparation, execution, reporting and correction of speed-trial data for displacement ships of 50–500 m length between perpendiculars ([official ISO record](https://www.iso.org/standard/82080.html)). ISO 19030-1:2016, reviewed and confirmed current in 2022, provides general principles and indicators for measuring hull/propeller performance changes on the same ship over time ([official ISO record](https://www.iso.org/standard/63774.html)).

This crate implements none of those standards. It supplies only a strict trace arithmetic/quality contract. Raw voyage traces, corrected speed-trial records, simulation output and synthetic fixtures must remain separately labelled. Equivalent mission, condition, energy-boundary and uncertainty semantics are required before energy totals can be compared using issue #8092.

## Claim ceiling

A Complete result means only that the canonical trace passed this crate's metadata, time-coverage, numeric and integration gates. It does not validate the measurement system, prove a hydrodynamic improvement, show corrected speed-power performance, establish lifecycle emissions, demonstrate vessel safety, or certify compliance.
