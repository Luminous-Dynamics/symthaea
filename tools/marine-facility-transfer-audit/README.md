# Marine facility transfer calibration audit

Standalone Rust audit for spatially resolved acoustic transfer-function calibration in cavitation facilities. It audits the calibration campaign which produces the facility TF; it does not solve the acoustic field or predict propeller noise.

## Run

    cargo test --manifest-path tools/marine-facility-transfer-audit/Cargo.toml
    cargo run --manifest-path tools/marine-facility-transfer-audit/Cargo.toml -- docs/research/marine/fixtures/synthetic-marine-facility-transfer-v0.1.0.json

The committed fixture is synthetic and must return INCOMPLETE. Exit codes: 0 for a repeatable position TF for the declared measurements/conditions only; 2 for invalid calibration records; 3 for incomplete or provisional calibration; 64 for invocation or JSON decoding failure.

## Measurement arithmetic and sampling

For each repeat and frequency band:

    TF = output received SPL - declared input source level

This follows ITTC 7.5-02-03-03.9 Rev.03 section 2.6, Equation 10. The source-level basis is explicit: independently measured, manufacturer free-field TVR, nominal-unverified or unknown. Manufacturer free-field TVR in a closed facility requires additional declared uncertainty and remains provisional; it is not silently treated as an in-situ source level.

Position means are computed first, then the declared aggregate is an equal-weight arithmetic mean of position-mean TF values in dB. Position spread, spatial standard deviation, within-position repeatability standard deviation and a maximum pairwise measured gradient are reported separately. The audit does not interpolate between unmeasured positions.

ITTC recommends at least three positions to assess variability and typically five to ten when considering the whole propeller disc. This contract requires five for whole-disc scope and three for restricted-sector scope, with at least two repeats at each position. Whole-disc coverage also requires three distinct azimuth quadrants and three positions at radial ratio >= 0.7; this is an explicit programme sampling rule, not an additional ITTC requirement.

## Condition and calibration boundaries

The record binds facility/test-section revision and geometry, coordinate frame, source/projector identity and signal, transmitter drive chain, TVR/source-level basis, receiver/hydrophone calibration, acquisition chain, source/receiver coordinates, repeat timebase and raw trace digests.

The calibration water condition carries temperature, pressure, flow, gas content and/or cavitation-susceptibility measure, and free-surface state. ITTC notes that transfer functions are normally measured with water at rest and without depressurization, which differs from cavitation testing with flow, pressure changes and traveling bubbles. V0 therefore marks still-water/other/unknown calibration conditions as provisional even when an extra uncertainty is supplied. A future validated condition-transfer model is not silently inferred.

## Outputs and claim ceiling

Per band: aggregate TF, position minimum/maximum, spatial range/standard deviation, within-position repeatability SD, maximum pairwise gradient and conservative expanded uncertainty. Per position: measured coordinates and repeat statistics.

A repeatable-position result is limited to the declared measured source positions, receiver geometry, bands, facility revision, source/projector and calibration conditions. It does not prove free-field behaviour, a facility-wide continuous TF map, correction for unmeasured positions, propeller radiated-noise prediction, model-to-full-scale transfer or ecological benefit.

SHA-256 values are provenance identifiers, not proof of calibration authenticity. Independent review and raw-byte verification remain required.

# References

- ITTC 7.5-02-03-03.9 (2024), Revision 03, sections 2.6.1–2.6.7. In particular, the standard explains spatial position dependence, recommends 5–10 positions for whole-disc coverage and at least 3 for variability assessment, and identifies source/projector and water-condition caveats.
- ITTC Nawigator XXI Cavitation & Noise Round Robin: use the geometry and test protocol only under the official terms and data-access rules.