# symthaea-acoustics

Physical acoustics for Symthaea — speed of sound, wavelength/frequency, decibel
combination, and the Doppler effect. Complements symthaea-dsp (digital signals)
with the physical-sound layer.

Pure std, zero dependencies, no symthaea-core link.

- speed_of_sound_air(temp_celsius) — approximately 343.2 m/s at 20 °C.
- wavelength / frequency — λ = c/f.
- combine_decibels — incoherent SPL sum.
- doppler_frequency — observed frequency under relative motion.
- response_curve — validated response curves, log-frequency interpolation, band summaries and target deviation.
- pareto — deterministic unweighted Pareto-front extraction for point estimates.
- robust_pareto — conservative Pareto-front extraction from objective intervals under one declared profile.
- experiment_selection — deterministic baselines for ranking already-eligible design × fidelity experiments.

## Multi-objective acoustic design

Keep response error, directivity error, maximum output, distortion, cost, size,
efficiency, energy use and repairability as separately named objectives. Do not
silently collapse them into one weighted hi-fi score. Apply hard feasibility,
physical safety and evidence-validity gates before ranking candidates.

The point-estimate Pareto module rejects malformed vectors and preserves trade-offs.
The robust module accepts finite lower/upper bounds, validates profile identity,
and treats overlapping intervals as unresolved. When intervals are point-valued,
its front must exactly match the point-estimate Pareto front. Intervals are not
automatically confidence intervals, and interval ordering alone does not model
correlations or joint uncertainty.

The experiment-selection module compares four explicit baselines: cheapest first,
frontier relevance first, estimated uncertainty reduction per cost, and a seeded
pseudo-random order. Ranking scores are caller-supplied heuristics frozen by an
evaluation profile; they are not probabilistic guarantees or claims that a
measurement will resolve a question. Persist the profile, cost unit, policy,
option IDs, budget and actual evaluation receipts for a benchmark replay.
Only pre-screened, eligible options enter the ranking. This module does not
authorize hardware actuation or bypass safety, calibration, or evidence controls.

These utilities analyze derived inputs. They do not ingest raw measurements,
establish microphone calibration, establish simulation validity, implement
CTA-2034/AES75/IEC compliance, guarantee safety, or predict human preference.
Raw observation and calibration provenance remains owned by FIELD. Objective
measurements, model predictions, room response and listening-study outcomes
must remain distinct evidence classes.

Run tests locally with:

    cargo test -p symthaea-acoustics
