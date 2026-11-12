# Marine cavitation intermittency audit

A standalone Rust audit for repeated-window, condition-labelled model-scale cavitation-noise measurements. It reports distributions and cavitation occupancy rather than collapsing a campaign into a single arithmetic dB mean.

## Run

~~~sh
cargo test --manifest-path tools/marine-cavitation-intermittency-audit/Cargo.toml
cargo run --manifest-path tools/marine-cavitation-intermittency-audit/Cargo.toml -- \
  docs/research/marine/fixtures/synthetic-marine-cavitation-intermittency-v0.1.0.json
~~~

The synthetic fixture must return INCOMPLETE as physical evidence. Exit codes:
- 0: distributional summary only;
- 2: invalid input;
- 3: incomplete evidence or synthetic fixture;
- 64: invocation/JSON decoding failure.

## V0 measurements

- Fixed band, declared service fingerprint and declared window duration.
- Per-run timebase, service/environment, hydrophone, calibration and background-trace receipts; each window carries distinct acoustic and cavitation-observation digests and is repeated in the report with an inclusion/exclusion reason and corrected level only when eligible.
- Each run also records total gas content or cavitation-susceptibility alpha, unit, expanded uncertainty, equivalence tolerance, method, instrument, calibration and measurement-evidence digest. Runs are compared only when water condition remains within uncertainty-bounded tolerance; unexplained drift is INCOMPARABLE and unresolved equivalence is INCOMPLETE. Nuclei number/size-distribution evidence coverage is reported separately.
- Per-window ITTC Rev.03 §2.5.3 treatment: below 3 dB unusable; 3 dB <= margin < 10 dB energy-corrected; above 10 dB unadjusted; exactly 10 dB unresolved because Rev.03 does not state a disposition.
- Per-run usable/excluded windows, occupancy, observed false-to-true onset count, left-censored active segments at acquisition starts/gaps, right-censored active segments at run end, observed onsets per covered hour, energy-equivalent mean, median, p05/p95, and cavitation-present/absent conditioned energy means.
- Explicit flag when recorded acquisition duration does not meet the ITTC recommendation of greater than 20 seconds; this is advisory context, not a hard rejection threshold.
- Independent runs remain separate; the case-level run-to-run dispersion is computed from each run's energy-equivalent mean.

The energy-equivalent mean is computed as:

`10 log10(mean(10^(L_i/10)))`

It is not the arithmetic mean of dB values. Empirical quantiles are descriptive, not confidence intervals. The audit does not infer statistical independence of adjacent windows and does not bootstrap them as independent replicates.

## Interpretation limits

A distributional summary does not identify an additive acoustic source contribution, establish a causal effect, validate CFD, extrapolate model-scale levels to full scale, estimate ecological exposure or certify a quiet ship. Cavitation observations and sensor digests are evidence references; v0.1 does not authenticate cameras/sensors or verify raw bytes.

Water susceptibility is a comparability control, not a substitute for controlling pressure, temperature, dissolved gas and nuclei populations. A measured alpha or gas-content value may fail to capture changes in the nuclei number/size distribution; that distribution remains separately referenced and its missing/partial coverage is explicit.

The ITTC Nawigator XXI Cavitation & Noise Round Robin is a natural external target for comparing occupancy, event rates, spectral distributions and inter-facility repeatability. The 2026 Nawigator XXI model-scale study specifically reports cavitation intermittency associated with insufficient nuclei and investigates post-processing to mitigate its effect.
