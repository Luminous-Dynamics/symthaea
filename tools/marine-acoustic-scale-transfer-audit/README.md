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

## Measurement qualification

Background handling follows ITTC Rev.03 §2.5.3 per spectral band:
- below 3 dB signal/background margin: background-dominated; do not scale the band;
- 3 dB through 10 dB inclusive: subtract background in the linear-energy domain;
- above 10 dB: no background correction is required.

Each model-scale run declares a background measurement method (bare hub or pressure-suppressed) and a separate background raw-trace digest. Unknown methods and malformed trace provenance block comparison. Uncertainty endpoint propagation reapplies these thresholds; when the lower signal/background margin becomes unusable, the predicted interval remains unresolved instead of silently treating that band as valid.

## Implemented arithmetic

- ITTC 7.5-02-03-03.9 (2024, Revision 03), Eq. 15: frequency mapping.
- The same procedure, Eq. 16: SPL scaling with distinct exponent sets for high/low frequency and proportional/constant bandwidth.
- Background correction by subtraction in the linear-energy domain.
- Model source level from corrected SPL minus the measured facility transfer function.
- Conservative interval propagation through background correction, facility transfer and per-band scale-transfer uncertainty. No statistical independence assumption is made.

## Evidence gates

The audit receipt repeats the source dataset ID/digest, evidence kind, scaling method/bandwidth/spectrum representation, per-band frequency-regime rationale, per-band transfer uncertainty, model/full-scale operating-point receipt digests, raw-trace digests and separate model-scale/full-scale timebase receipt digests. Model and full-scale trials are separate campaigns and must not be forced to share one clock identity. This makes the computed result self-describing without claiming those supplied digests authenticate the physical source. It requires explicit nD-referenced cavitation numbers, frequency-band identity, a declared high/low-frequency regime and rationale per band, and declarations for wake, cavitation pattern, gas/nuclei state, Reynolds-scale review and facility transfer function.

A research transfer comparison requires at least two model-scale propeller rotation-rate intervals that do not overlap after declared RPM expanded uncertainty is applied. Full-scale observations are compared only at the corresponding ITTC-mapped frequencies and for the same declared band IDs. The model and full-scale source-level records must use the same reference distance (normally 1 m).

## Cavitation-pattern and sensor-provenance gates

Every model/full-scale pair declares a cavitation-pattern class and an explicit comparability state. SHEET, CLOUD and BUBBLE are calculable only when the pattern is declared MATCHED. TIP_VORTEX and MIXED remain INCOMPLETE because the additional Reynolds-number acoustic scaling treatment is not implemented/validated here; UNKNOWN or PARTIAL comparability also remains INCOMPLETE. An explicit pattern mismatch is NOT_COMPARABLE.

The receipt preserves model/full-scale hydrophone instrument IDs, calibration-certificate digests and the model facility-transfer-function evidence digest, in addition to timebase and raw-trace digests. These are identity/provenance references only: v0.1 checks digest syntax but does not authenticate certificates, verify raw trace bytes, resolve the calibration registry or consume the shared calibration/timebase binding receipt from #11641. That registry integration remains a prerequisite for real campaign use.

## Interpretation and limitations

computational_comparison_only means only that the supplied numerical comparison and declared metadata passed this audit. It is not a validation of physical similarity, CFD, model-scale-to-full-scale generalization, an ISO/ITTC measurement, ecological impact, or improved propeller performance.

The ITTC procedure cautions that scaling laws do not correct for reverberation or dissimilar cavitation patterns/dynamics. It recommends at least two model-scale rotation rates for estimating transfer reproducibility. Its often-cited 4.2–7.1 dB combined uncertainty is questionnaire-derived and is not installed as a universal uncertainty floor or acceptance threshold. The caller must supply project-specific expanded uncertainty components; zero is rejected rather than treated as unknown.

V0 implements the facility-transfer-function route. ITTC Rev.03 Table 1 permits proportional bandwidth with power-spectrum representation; Table 2 permits constant bandwidth with either power spectrum or power spectral density. The declared representation is preserved in the receipt and the prohibited proportional-bandwidth / power-density combination is rejected. Bandwidth convention selects the exponent table. Onboard sensor source-strength estimation, Doppler correction, band regridding, spectral-shape scoring, correlated uncertainty models and tip-vortex-specific scaling methods are deliberately not inferred or silently approximated. Hashes are identifiers, not proof of publisher identity, sensor authenticity, rights or calibration validity. Declared review booleans/states are data, not cryptographic attestations.
