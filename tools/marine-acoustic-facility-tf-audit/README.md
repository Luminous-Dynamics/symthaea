# Marine acoustic facility transfer-function audit

A standalone Rust evidence audit for facility transfer-function calibrations measured with a known acoustic source and receiving hydrophone(s). It is not a free-field propagation solver or a universal facility correction.

## Run

~~~sh
cargo test --manifest-path tools/marine-acoustic-facility-tf-audit/Cargo.toml
cargo run --manifest-path tools/marine-acoustic-facility-tf-audit/Cargo.toml -- \
  docs/research/marine/fixtures/synthetic-marine-acoustic-facility-tf-v0.1.0.json
~~~

The synthetic fixture remains INCOMPLETE as physical evidence. Exit codes:
- 0: REPEATABLE_POSITION_TF (a position/frequency-specific result only);
- 2: INVALID, SPATIALLY_VARIABLE or REPEATABILITY_FAILED;
- 3: INCOMPLETE or PROVISIONAL_SOURCE_BASIS;
- 64: invocation or JSON decoding failure.

## Arithmetic and eligibility

The transfer function is computed per sample and frequency band as:

`TF = SPL_received - SL_source`

The declared source-level basis, bandwidth/quantity, projector/transmitter chain, facility/test-section revision, hydrophone/calibration, acquisition chain, water/flow/free-surface state and source/receiver coordinates are preserved. Expanded bounds are conservatively combined by addition; no statistical independence assumption is made.

Manufacturer free-field TVR used in a closed test section requires a separately declared source-basis uncertainty and remains PROVISIONAL_SOURCE_BASIS in v0.1, even when a numeric nominal TF can be computed. Unknown TVR/source-level basis never emits a numeric TF.

Repeatability is checked at each exact source/receiver position pair and frequency band. Same-position repeat scatter is not conflated with spatial variation across different positions. Spatial ranges are compared with conservative uncertainty bounds. Sparse spatial samples are not interpolated or extrapolated, and a facility-wide representative factor is never asserted by v0.1.

Power-spectral-density values and one-third-octave band SPL are never pooled under one band identity. Different bandwidth basis or centre frequency with a repeated band ID is invalid.

## Interpretation limits

A result qualifies only the declared measured input/output relationship at the stated source/receiver position, frequency, signal and facility condition. It does not prove a free-field environment, validate every projector/hydrophone position, correct arbitrary cavitating propeller measurements, validate model-to-full-scale transfer, or establish an environmental/noise benefit. Hashes are evidence identifiers, not authentication.
