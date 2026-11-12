# Marine acoustic scale-transfer audit

Standalone Rust arithmetic and evidence-contract audit for model-to-full-scale propeller cavitation-noise transfer.

## Run

~~~sh
cargo test --manifest-path tools/marine-acoustic-scale-transfer-audit/Cargo.toml
cargo run --manifest-path tools/marine-acoustic-scale-transfer-audit/Cargo.toml -- \
  docs/research/marine/fixtures/synthetic-marine-acoustic-scale-transfer-v0.1.0.json
~~~

The input schema is docs/research/marine/marine-acoustic-scale-transfer-v0.1.0.schema.json.
The synthetic fixture is intentionally rejected as measured validation. Exit codes:

- 0: computational comparison contract complete (not physical qualification);
- 2: invalid input or explicitly non-comparable evidence;
- 3: incomplete evidence or prediction-only;
- 4: measured full-scale comparison falls outside the propagated uncertainty interval;
- 64: invocation, file or JSON decoding failure.

## Implemented arithmetic

- ITTC 7.5-02-03-03.9 (2024, Revision 03), Eq. 15: frequency mapping.
- The same procedure, Eq. 16: SPL scaling with distinct exponent sets for high/low frequency and proportional/constant bandwidth.
- Background correction by subtraction in the linear-energy domain.
- Model source level from corrected SPL minus the measured facility transfer function.
- Conservative interval propagation through background correction, facility transfer and scale-transfer uncertainty. No statistical independence assumption is made.

## Evidence gates

The audit receipt repeats the source dataset ID/digest, evidence kind, scaling method/bandwidth/spectrum representation, per-band frequency-regime rationale, model/full-scale operating-point receipt digests, raw-trace digests and shared timebase receipt. This makes the computed result self-describing without claiming those supplied digests authenticate the physical source. It requires explicit nD-referenced cavitation numbers, frequency-band identity, a declared high/low-frequency regime and rationale per band, and declarations for wake, cavitation pattern, gas/nuclei state, Reynolds-scale review and facility transfer function.

A research transfer comparison requires at least two distinct model-scale propeller rotation rates. Full-scale observations are compared only at the corresponding ITTC-mapped frequencies and for the same declared band IDs. The model and full-scale source-level records must use the same reference distance (normally 1 m).

## Interpretation and limitations

computational_comparison_only means only that the supplied numerical comparison and declared metadata passed this audit. It is not a validation of physical similarity, CFD, model-scale-to-full-scale generalization, an ISO/ITTC measurement, ecological impact, or improved propeller performance.

The ITTC procedure cautions that scaling laws do not correct for reverberation or dissimilar cavitation patterns/dynamics. It recommends at least two model-scale rotation rates for estimating transfer reproducibility. Its often-cited 4.2–7.1 dB combined uncertainty is questionnaire-derived and is not installed as a universal uncertainty floor or acceptance threshold. The caller must supply project-specific expanded uncertainty components; zero is rejected rather than treated as unknown.

V0 implements the facility-transfer-function route. Bandwidth convention selects the ITTC exponent table; the declared power-spectrum versus power-spectral-density representation is reported as separate metadata and is not used to infer a bandwidth convention. Onboard sensor source-strength estimation, Doppler correction, band regridding, spectral-shape scoring, correlated uncertainty models and tip-vortex-specific scaling methods are deliberately not inferred or silently approximated. Hashes are identifiers, not proof of publisher identity, sensor authenticity, rights or calibration validity. Declared review booleans/states are data, not cryptographic attestations.
