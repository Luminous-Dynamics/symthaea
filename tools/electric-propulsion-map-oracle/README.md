# Electric Propulsion Map Oracle

A dependency-free, independent reference for validating and querying a two-dimensional motor efficiency map over a declared motor torque / speed grid.

The v0.1 map is bound to one exact map-content digest, one exact configuration digest, one exact operating-context digest and one exact evidence-manifest digest. The context digest must identify the frozen voltage, temperature, cooling, control/firmware, installation and environmental context for the map slice. The map-content digest identifies the exact serialized map artifact; the evidence-manifest digest identifies the source measurements, calibration and transformations. This crate does not calculate or verify these digests; the upstream evidence owner supplies them. Every lookup result carries both digests.

## Lookup contract

- Motor torque axis is in N m, strictly positive for this motoring-only map, finite, strictly increasing, and contains at least two points.
- Motor speed axis is in RPM, strictly positive, finite, strictly increasing, and contains at least two points.
- Nodes are row-major with speed as the outer index and torque varying fastest.
- Each node contains efficiency in (0, 1] and a finite non-negative absolute uncertainty bound no greater than 1.
- A separate non-negative map-level systematic absolute uncertainty bound is added linearly.
- Query identity must exactly match both map configuration and context digests.
- Query torque and speed must be finite, strictly positive and inside the closed characterized rectangle. Zero-torque efficiency and reverse/regenerative quadrants are outside this motoring-only contract.
- Interpolation is bilinear. Out-of-domain queries return a typed error; there is no extrapolation, nearest-neighbour fallback, or coordinate clamping.

The uncertainty result is a conservative absolute bound under the supplied nodal and systematic bounds: weighted nodal absolute bounds plus the systematic bound. It is not a Gaussian confidence interval and does not replace a covariance-aware uncertainty model. The reported physical interval is clipped to the physical efficiency range [0, 1]; the query coordinates and map support are never clipped.

The map is one map-content-, evidence-manifest- and context-specific motoring slice. This v0.1 does not interpolate across bus voltage, temperature, coolant flow, ageing, firmware or different hardware configurations. A separate map slice and context digest are required for each distinct characterized context.

## Example

For a 2 × 2 grid with torque axis [1, 10] N m, speed axis [1,000, 2,000] RPM, and row-major efficiencies:

- 1,000 RPM: [0.70, 0.80]
- 2,000 RPM: [0.90, 0.95]

A query at 3.25 N m and 1,500 RPM has bilinear weights [0.375, 0.125, 0.375, 0.125] and efficiency 0.81875.

These values are synthetic analytical fixtures, not measured motor performance.

## Verification

```bash
cargo fmt --manifest-path tools/electric-propulsion-map-oracle/Cargo.toml -- --check
cargo test --locked --manifest-path tools/electric-propulsion-map-oracle/Cargo.toml
cargo clippy --locked --manifest-path tools/electric-propulsion-map-oracle/Cargo.toml --all-targets -- -D warnings
```

A passing synthetic test establishes only map data validation and interpolation semantics. It does not characterize a motor, validate a physical efficiency map, establish continuous/peak limits, or provide aircraft safety or airworthiness evidence.


## Exact map artifact identity (v0.2)

The map-content digest is distinct from the source evidence-manifest digest. The same raw observations and calibration evidence can yield different map artifacts after changes to filtering, fitting, grid selection or map-generation code. The map digest must identify the exact serialized axes, nodes, uncertainty values and map metadata. The source evidence-manifest digest must identify the underlying observations and transformations. The lookup result preserves both; the consumer must not substitute one for the other.

This crate requires the caller to supply both exact digests. It does not calculate a canonical serialization or cryptographic hash internally.
