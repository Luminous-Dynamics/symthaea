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
- experiment_selection — deterministic baselines and budgeted plans for already-eligible design × fidelity experiments.

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
pseudo-random order. Ranking inputs are caller-supplied heuristics frozen by an
evaluation profile; they are not probabilities and do not promise a measurement
will resolve a question. Each option must identify its evaluation profile and
common cost unit; mismatches fail closed.

The ranker returns a full option-order and profile metadata. Budget planning
consumes that ranking only when the entire option snapshot is unchanged; this
prevents stale indices being reused after costs, fidelity or relevance estimates
change. The budget planner skips options that do not fit the remaining budget
so a cheaper later option can still be selected. It returns selected and skipped
IDs, estimated spend, remaining budget, profile, unit and policy as a replayable
planning receipt. This receipt is not authorization to execute physical tests.

These utilities analyze derived inputs; they do not ingest raw measurements,
establish microphone calibration, establish simulation validity, implement
CTA-2034/AES75/IEC compliance, guarantee safety, or predict human preference.
Raw observation and calibration provenance remains owned by FIELD. Objective
measurements, model predictions, room response and listening-study outcomes
must remain distinct evidence classes.

Run tests locally with:

    cargo test -p symthaea-acoustics
