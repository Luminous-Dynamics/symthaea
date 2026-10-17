# symthaea-acoustics

Physical acoustics for Symthaea — speed of sound, wavelength/frequency, decibel
combination, and the Doppler effect. Complements symthaea-dsp (digital signals)
with the physical-sound layer.

Pure std, zero dependencies, no symthaea-core link.

- speed_of_sound_air(temp_celsius) — approximately 343.2 m/s at 20 °C.
- wavelength / frequency — λ = c/f.
- combine_decibels — incoherent SPL sum.
- doppler_frequency — observed frequency under relative motion.
- response_curve::validate_response_curve — validates finite, positive, strictly increasing frequency points.
- response_curve::interpolate_log_frequency — log-axis interpolation with extrapolation rejected.
- response_curve::log_frequency_mean_db — declared-band mean level weighted uniformly over log frequency.
- response_curve::log_frequency_rms_deviation_db — target-curve deviation over a declared band and mixed frequency grids.
- pareto::pareto_front_indices — deterministic unweighted Pareto-front extraction for point estimates.
- robust_pareto::robust_pareto_front_indices — conservative Pareto-front extraction from uncertainty intervals under one declared profile.

## Multi-objective design search

Keep response error, directivity error, maximum output, distortion, cost, size,
efficiency, energy use and repairability as separately named objectives. Do not
silently collapse them into one weighted “hi-fi score”. Apply hard feasibility,
physical safety and evidence-validity gates before ranking candidates.

The point-estimate Pareto module rejects malformed vectors and preserves trade-offs.
The robust module accepts finite lower/upper bounds, validates profile identity,
and treats overlapping intervals as unresolved. When intervals are point-valued,
its front must exactly match the point-estimate Pareto front. The intervals are
not automatically confidence intervals: their statistical or metrological
meaning must be justified and frozen in the caller's profile. Interval ordering
does not by itself model correlations or joint uncertainty.

These utilities analyze derived inputs; they do not ingest raw measurements,
establish microphone calibration, establish simulation validity, implement
CTA-2034/AES75/IEC compliance, guarantee safety, or predict human preference.
Raw observation and calibration provenance remains owned by FIELD. Objective
measurements, model predictions, room response and listening-study outcomes
must remain distinct evidence classes.

Run local tests with:

    cargo test -p symthaea-acoustics
