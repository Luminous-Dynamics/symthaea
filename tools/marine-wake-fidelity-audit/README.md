# Marine propeller-plane wake fidelity audit

A standalone Rust audit for comparing a target propeller-plane wake field with the wake realized in a cavitation tunnel or numerical setup. It is a grid/coordinate/uncertainty audit, not a CFD solver or cavitation-noise predictor.

## Run

~~~sh
cargo test --manifest-path tools/marine-wake-fidelity-audit/Cargo.toml
cargo run --manifest-path tools/marine-wake-fidelity-audit/Cargo.toml -- \
  docs/research/marine/fixtures/synthetic-marine-wake-fidelity-v0.1.0.json
~~~

The synthetic fixture must return INCOMPLETE as physical evidence. Exit codes:
- 0: FULL_VECTOR_COMPARABLE (declared field arithmetic/tolerance comparison only);
- 2: INVALID or INCOMPARABLE input;
- 3: INCOMPLETE or AXIAL_ONLY_DESCRIPTIVE;
- 64: invocation or JSON decoding failure.

## Wake contract

- Axial, radial and tangential velocities are normalized by an explicit common reference velocity.
- Target/realized fields carry their own coordinate frame, angular origin, positive azimuth and axial-positive convention. Any mismatch is INCOMPARABLE.
- Target/realized polar grid, area-weight method and per-cell weights must match exactly within declared numeric tolerances. No interpolation or silent azimuth rotation.
- Each field carries instrument/solver ID, calibration/validation digest, raw data digest, and separate area-quadrature method/evidence.
- Component bias, MAE, RMSE and P95 are area-weighted on the polar grid; unweighted diagnostics are retained. Cell-level residuals use conservative sums of expanded uncertainties.
- Axial nominal wake fraction is area-integrated using explicitly declared normalized polar weights. It is not an unweighted average of arbitrary sample points.
- Fourier amplitude/phase diagnostics are produced only for complete uniform azimuth rings with adequate anti-aliasing coverage. Phase difference is circular; harmonic metrics are descriptive and are not currently a standalone acceptance gate.
- Missing radial/tangential components remain missing. Wire-screen-derived axial-only data cannot claim full-vector fidelity.

## Tunnel blockage

- Wire screen: propeller disk area divided by test-section area.
- Dummy/full hull: maximum projected hull frontal area divided by test-section area.
- Other setups require explicit projected area and basis description.
- The ITTC <20% recommendation is a marker, not a correction or proof of zero acoustic confinement. At or above 20%, the audit requires a separate reviewed blockage assessment digest; absent review is INCOMPLETE, not INVALID.
- No universal blockage correction is applied to acoustic noise.

## Interpretation

A passing result establishes consistency of the declared grid, quadrature and component-tolerance arithmetic only. It does not establish wake scaling, Reynolds/Froude similarity, solver validation, cavitation behavior, acoustic noise, propeller performance or ecological benefit. Hashes are evidence identifiers, not authentication.
