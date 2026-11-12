# Marine acoustic source-state contrast audit

A standalone Rust eligibility audit for deciding whether an observed underwater received-level change can be associated with a predeclared change in one source state while the declared alternative source states and service/environment boundary remain controlled.

## Run

~~~sh
cargo test --manifest-path tools/marine-acoustic-source-contrast-audit/Cargo.toml
cargo run --manifest-path tools/marine-acoustic-source-contrast-audit/Cargo.toml -- \
  docs/research/marine/fixtures/synthetic-marine-acoustic-source-contrast-v0.1.0.json
~~~

The synthetic fixture intentionally returns INCOMPLETE as physical evidence, even when its declared source-state contrast is structurally identifiable. Exit codes:
- 0: contrast eligibility only (not physical source decomposition);
- 2: invalid or incomparable inputs;
- 3: incomplete, unresolved-mixture or partially identifiable evidence;
- 64: invocation or JSON decoding failure.

## V0 eligibility rules

- The intervention is predeclared and randomized or counterbalanced.
- The target source state changes and every other declared source state remains unchanged.
- The full source inventory is present on both sides: propeller cavitation, propeller hydrodynamics, prime mover, shaft/gearbox, auxiliary machinery, hull/appendage flow and ambient background.
- No unobserved source contributors are declared.
- Baseline and candidate share exact service and environment fingerprints, hydrophone identity, calibration-certificate digest and timebase receipt.
- Each source state has an independent observable identity and evidence digest.
- Received-SPL bands, frequency centres and expanded-uncertainty confidence levels are matched.

A same-state passive spectrum is UNRESOLVED_MIXTURE. A changed non-target source, service or environment boundary is INCOMPARABLE. Matched repeated (non-randomized/non-counterbalanced) contrasts are only PARTIALLY_IDENTIFIABLE.

## Interpretation boundary

The audit reports the observed candidate-minus-baseline change in **total received sound pressure level** with a conservative uncertainty interval using the sum of expanded uncertainties. It does not estimate additive propeller/machinery source contributions, source level, propagation loss, ecological exposure, biological response, quiet-ship compliance or causal effect beyond the declared contrast design.

The SHA-256 values are identifiers, not authentication. V0 requires the same hydrophone/calibration/timebase across both observations; it does not perform sensor authentication, clock-transfer correction, multi-hydrophone beamforming, blind source separation or spectral deconvolution.
