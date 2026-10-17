# symthaea-acoustics

Physical acoustics for Symthaea — speed of sound, wavelength/frequency, decibel
combination, and the Doppler effect. Complements `symthaea-dsp` (digital signals)
with the physical-sound layer.

Pure `std`, zero deps, no `symthaea-core` link. Checked vs textbook values.

- `speed_of_sound_air(temp_celsius)` — ≈343.2 m/s at 20 °C.
- `wavelength` / `frequency` — `λ = c/f`.
- `combine_decibels` — incoherent SPL sum (two 60 dB → 63 dB).
- `doppler_frequency` — observed pitch under relative motion.
- `response_curve::validate_response_curve` — validates finite, positive, strictly increasing frequency points.
- `response_curve::interpolate_log_frequency` — log-axis interpolation with extrapolation rejected.
- `response_curve::log_frequency_mean_db` — declared-band mean level weighted uniformly in log frequency.
- `response_curve::log_frequency_rms_deviation_db` — target-curve deviation over a declared band and mixed frequency grids.
- `pareto::ObjectiveSpec::{minimize,maximize}` — declares the preferred direction of each objective.
- `pareto::ParetoCandidate` — binds a stable candidate ID to ordered objective values.
- `pareto::pareto_front_indices` — deterministically returns the non-dominated candidate indices, preserving input order.

## Design-search boundary

Use the Pareto filter to preserve trade-offs among separately named objectives such as
response error, directivity error, maximum linear output, BOM cost, enclosure volume,
efficiency, and repairability. It intentionally produces no composite “hi-fi score”
and applies no implicit weighting. Values must be finite, fully scored, and ordered
exactly as the objective specifications. Apply hard feasibility constraints and
measurement/simulation validity gates before ranking; retain rejected and unscored
designs in the calling benchmark's evaluation record.

These utilities analyze derived inputs. They do not ingest raw measurements, establish
microphone calibration, prove simulation validity, implement CTA-2034/AES75/IEC
compliance, or predict human preference. Raw measurement and calibration provenance
remains in FIELD. A response-curve metric or Pareto ranking is not a universal sound
quality judgment.

```bash
cargo test -p symthaea-acoustics
```
