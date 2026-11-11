# Electric Propulsion Observation Oracle

A dependency-free Rust reference for validating raw propulsion-test channel identity, SI units, calibration provenance, sample ordering and bounded timestamp synchronization.

It selects actual observations close to a declared anchor time. It never interpolates, extrapolates, holds the last value or synthesizes a missing sample.

## Run and channel identity

Every run binds:
- run, asset/specimen and exact configuration/context identities;
- procedure revision and evidence-manifest digest;
- one declared time domain (monotonic, TAI or UTC);
- maximum channel clock uncertainty and maximum allowed synchronized-sample time error.

Every channel binds:
- channel and physical boundary IDs;
- quantity enum, SI unit enum and sign convention;
- exact run/asset/configuration/context identity;
- calibration ID, certificate digest and current/expired/unknown state;
- raw-data digest, clock source and clock uncertainty;
- source sequence, timestamp, value, absolute uncertainty and quality per sample.

Different channels may share a physical boundary (for example, voltage and current at the same terminals). Channel IDs must be unique.

## Synchronization semantics

For each required channel:
1. Validate quantity/unit compatibility, calibration, provenance, sample order and finite values/uncertainties.
2. Consider only samples explicitly marked Valid.
3. Select the nearest actual sample to the requested anchor timestamp. Equal-distance ties select the earlier timestamp, then the lower source sequence.
4. Calculate the absolute sample-to-anchor offset and add the declared channel clock uncertainty.
5. Reject if clock uncertainty exceeds the run limit, or the combined time error exceeds the run skew limit.

The result retains original timestamps, signed offsets, source sequences, values, uncertainties, calibration and raw-data identities. It also records observed maximum sample skew, clock uncertainty and combined timing error.

No interpolation, extrapolation, forward-fill, backward-fill, imputation or cross-run joining is performed. Suspect/invalid samples are never promoted as valid.

All timestamps use the run manifest's declared time domain. This crate does not convert between clock domains or establish UTC/TAI synchronization.

## Supported quantities and SI units

- Torque — N m
- Rotational speed — RPM
- Thrust — N
- Voltage — V
- Current — A
- Electrical power — W
- Temperature — K
- Pressure — Pa
- Coolant mass flow — kg/s
- Sound pressure — Pa RMS
- Vibration acceleration — m/s²

Sign convention is explicit metadata. The crate does not infer sign from sample values or change a sign convention.

## Verification

```bash
cargo fmt --manifest-path tools/electric-propulsion-observation-oracle/Cargo.toml -- --check
cargo test --locked --manifest-path tools/electric-propulsion-observation-oracle/Cargo.toml
cargo clippy --locked --manifest-path tools/electric-propulsion-observation-oracle/Cargo.toml --all-targets -- -D warnings
```

Synthetic synchronization tests validate only this software contract. They do not calibrate an instrument, establish metrology traceability, validate a motor map, prove physical powertrain performance or qualify aircraft/flight evidence.
