# Marine hydrophone–accelerometer vibration coherence audit

A standalone Rust contract and arithmetic audit for the narrowband cross-spectral relationship between one hydrophone pressure channel and one registered accelerometer reference.

## Run

~~~sh
cargo test --manifest-path tools/marine-vibration-coherence-audit/Cargo.toml
cargo run --manifest-path tools/marine-vibration-coherence-audit/Cargo.toml -- \
  docs/research/marine/fixtures/synthetic-marine-vibration-coherence-v0.1.0.json
~~~

The synthetic fixture must return INCOMPLETE as physical evidence.

## Defined spectral quantities

For pressure p and acceleration a, with the declared convention S_pa = E[P A*]:

- Magnitude-squared coherence: gamma² = |S_pa|² / (S_pp S_aa)
- H1 transfer estimate: H1 = S_pa / S_aa
- Vibration-reference-coherent pressure PSD: S_coh = |S_pa|² / S_aa = gamma² S_pp
- Non-coherent residual: S_res = S_pp - S_coh

Input spectral units are pressure PSD in Pa²/Hz, acceleration PSD in (m/s²)²/Hz, and cross PSD in Pa·(m/s²)/Hz. Reported pressure levels use dB re 1 µPa²/Hz.

## Measurement contract

The input contains frequency-resolved, externally computed Welch cross-spectral estimates. This v0 audit **does not calculate Welch spectra from raw samples**; it validates the estimator metadata, calibration/timebase lineage, spectral-matrix arithmetic and declared marginal uncertainty envelopes. The source raw-trace digests are references to the original aligned pressure/acceleration traces.

Both channels must have distinct instrument/channel identities and calibration/raw-trace digests, while sharing the exact versioned timebase, sample rate, sample count and measurement window. The alignment offset must already have been applied in the CSD estimator. The remaining alignment uncertainty is converted to a frequency-dependent phase uncertainty bound: 2π f U_t, added conservatively to the declared cross-spectrum phase uncertainty.

The H1 transfer phase is withheld when that bound exceeds the predeclared maximum, while magnitude coherence and power estimates remain available. The cross-spectrum sign convention is fixed as pressure times conjugate acceleration, so H1 phase has an unambiguous sign convention.

The spectral matrix must be positive semidefinite: |S_pa|² <= S_pp S_aa. A violation is INVALID; it is never clamped into a pass. Uncertainty envelopes are conservative marginal bounds. The tool does not assert joint coverage or independent errors; when the residual interval would permit negative pressure power, uncertainty is marked unresolved.

## Interpretation boundary

The result describes a linear spectral component coherent with the declared accelerometer reference and a non-coherent residual. Neither component uniquely identifies a physical source. The coherent component may include common-cause excitation; the residual contains every component not linearly coherent with this one reference plus estimator error. The residual is not "cavitation-only", and the coherent part is not automatically "machinery noise".

This is not a multichannel partial-coherence solver, beamformer, source-separation model, causal estimator, acoustic propagation model or regulatory qualification. The coherence interpretation threshold is project-declared before analysis; no universal threshold is embedded.

The audit binds operating-point and timebase receipt digests from the marine trial evidence system. Those hashes are identifiers, not authentication or resolution of the underlying receipt.
