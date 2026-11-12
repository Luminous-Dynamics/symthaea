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

- The intervention is predeclared; its protocol and experimental-design/randomization evidence are bound by separate digests.
- Randomized or counterbalanced design is required for contrast eligibility. A single baseline/candidate contrast is only an eligibility record; it is not, by itself, a powered causal experiment or causal effect estimate.
- The target source state changes by more than the sum of its baseline/candidate expanded uncertainties.
- Every non-target source has a numeric observable, explicit unit, expanded uncertainty and predeclared equivalence tolerance. A non-target is considered stable only when the observed difference plus both expanded uncertainties remains within that tolerance; a definite violation is INCOMPARABLE and an uncertainty interval crossing the tolerance is INCOMPLETE.
- The full source inventory is present on both sides: propeller cavitation, propeller hydrodynamics, prime mover, shaft/gearbox, auxiliary machinery, hull/appendage flow and ambient background.
- No unobserved source contributors are declared.
- Baseline and candidate share exact service and environment fingerprints, hydrophone identity, calibration-certificate digest and timebase receipt.
- Each source state has an independent observable identity and evidence digest, with unchanged observable identity, units and equivalence tolerances across the contrast.
- Received-SPL bands, frequency centres and expanded-uncertainty confidence levels are matched.

A same-state passive spectrum is UNRESOLVED_MIXTURE. A changed non-target source, service or environment boundary is INCOMPARABLE. Matched repeated (non-randomized/non-counterbalanced) contrasts are only PARTIALLY_IDENTIFIABLE.

## Interpretation boundary

The audit reports the observed candidate-minus-baseline change in **total received sound pressure level** with a conservative uncertainty interval using the sum of expanded uncertainties. It checks eligibility of the declared controlled contrast; one pair does not establish statistical power, a causal effect estimate or transportability. It does not estimate additive propeller/machinery source contributions, source level, propagation loss, ecological exposure, biological response or quiet-ship compliance.

The SHA-256 values are identifiers, not authentication. V0 requires the same hydrophone/calibration/timebase across both observations; it does not perform sensor authentication, clock-transfer correction, multi-hydrophone beamforming, blind source separation or spectral deconvolution.
