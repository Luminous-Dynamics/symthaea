# symthaea-acoustics

Physical acoustics for Symthaea — speed of sound, wavelength/frequency, decibel
combination, and the Doppler effect. Complements `symthaea-dsp` (digital signals)
with the physical-sound layer. The dependency-free response-curve utilities also support strict input validation, log-frequency interpolation, logarithmic-band means, and log-frequency RMS comparison against an explicit target curve.

Pure `std`, zero deps, no `symthaea-core` link. Checked vs textbook values.

- `speed_of_sound_air(temp_celsius)` — ≈343.2 m/s at 20 °C.
- `wavelength` / `frequency` — `λ = c/f`.
- `combine_decibels` — incoherent SPL sum (two 60 dB → 63 dB).
- `doppler_frequency` — observed pitch under relative motion.
- `response_curve::validate_response_curve` — validates finite, positive, strictly increasing frequency points.
- `response_curve::interpolate_log_frequency` — log-axis interpolation with extrapolation rejected.
- `response_curve::log_frequency_mean_db` — declared-band mean level weighted uniformly in log frequency.
- `response_curve::log_frequency_rms_deviation_db` — target-curve deviation over a declared band and mixed frequency grids.

Response-curve summaries are mathematical metrics, not CTA-2034 compliance, calibrated-measurement evidence, or a human-preference prediction. Raw measurement/calibration provenance remains in FIELD.

```bash
cargo test -p symthaea-acoustics
```
