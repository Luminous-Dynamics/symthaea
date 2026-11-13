# Marine cavitation intermittency audit

A standalone Rust audit for repeated-window, condition-labelled model-scale cavitation-noise measurements. It reports distributions and the fraction of aligned acquisition windows with any cavitation presence rather than collapsing a campaign into a single arithmetic dB mean.

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
- Per-run timebase, operating-point, acoustic/video synchronization, service/environment, hydrophone, calibration and background-trace receipts; each window carries distinct acoustic and cavitation-observation digests, an alignment offset and expanded alignment uncertainty, and is repeated in the report with an inclusion/exclusion reason and corrected level only when eligible.
- Each run records total gas content or cavitation-susceptibility alpha, unit, expanded uncertainty, equivalence tolerance, method, instrument, calibration and measurement-evidence digest. It also carries an optional structured cavitation-nuclei size distribution: measured diameter bins, number concentration per mL, per-bin expanded uncertainty and a declared marginal coverage level, sample volume, instrument/method/calibration and raw evidence digest.
- When every run supplies a compatible nuclei distribution, the audit compares both total number concentration and distribution shape. Shape is compared using the first Wasserstein (earth-mover) distance over the declared diameter-bin midpoints; lower/upper bounds are conservative envelopes derived from the declared per-bin marginal uncertainty intervals; the audit does not claim that the same coverage level applies jointly to the aggregate Wasserstein/total-concentration interval. The comparison receipt preserves measurement IDs, method, instrument, calibration and both raw evidence digests. Different binning or method/instrument/calibration is INCOMPARABLE; uncertainty that straddles either predeclared equivalence tolerance is INCOMPLETE. Nuclei distributions are not silently interpolated or normalized into a single scalar.
- Missing nuclei-distribution evidence across the entire cohort is reported as an advisory gap; partial cohort coverage is INCOMPLETE. Gas content or susceptibility alpha is not treated as a substitute for the number/size distribution.
- Per-window ITTC Rev.03 §2.5.3 treatment: below 3 dB unusable; 3 dB <= margin < 10 dB energy-corrected; above 10 dB unadjusted; exactly 10 dB unresolved because Rev.03 does not state a disposition.
- Per-run usable/excluded windows, cavitation window-presence fraction, observed false-to-true window-transition count, left-censored active window segments after acquisition starts/gaps, right-censored active window segments at acquisition gaps and separately at run end, observed window transitions per covered hour, energy-equivalent mean, median, p05/p95, and cavitation-present/absent conditioned energy means.
- Every included window preserves its corrected SPL and conservative expanded lower/upper measurement bounds. A background-dominated window is not silently discarded: it remains excluded from the corrected distribution but carries a conservative upper limit for its unknown cavitation component (total received SPL plus its expanded uncertainty), consistent with ITTC's permission to report such measurements as upper limits. Upper-limit observations are not imputed into the mean or quantiles.
- The window-presence fraction is the fraction of aligned, annotated acquisition windows in which any cavitation is observed; it is not the fraction of time cavitation occupies within each window. A camera/video observation is eligible for this window classification only when |alignment offset| + expanded alignment uncertainty is within the predeclared maximum. Out-of-tolerance windows remain usable for the acoustic distribution but their cavitation annotation is treated as unknown; window-presence and conditioned comparisons become INCOMPLETE. A synchronization receipt binds the camera/video and hydrophone clocks; raw timestamps alone are not synchronization evidence.
- Explicit flag when recorded acquisition duration does not meet the ITTC recommendation of greater than 20 seconds; this is advisory context, not a hard rejection threshold.
- Independent runs remain separate; the case-level run-to-run dispersion is computed from each run's energy-equivalent mean.

The energy-equivalent mean is computed as:

`10 log10(mean(10^(L_i/10)))`

It is not the arithmetic mean of dB values. Empirical quantiles are descriptive, not confidence intervals. The uncertainty interval for nuclei-distribution comparison is an envelope from declared binwise marginal uncertainty intervals; no joint confidence level is asserted. The audit does not infer statistical independence of adjacent windows and does not bootstrap them as independent replicates.

## Interpretation limits

The report requires an explicit reference run for water/nuclei comparability; input array order is not used to select the reference. A distributional summary does not identify an additive acoustic source contribution, establish a causal effect, validate CFD, extrapolate model-scale levels to full scale, estimate ecological exposure or certify a quiet ship. Cavitation observations and sensor digests are evidence references; v0.1 does not authenticate cameras/sensors or verify raw bytes.

Water susceptibility is a comparability control, not a substitute for controlling pressure, temperature, dissolved gas and nuclei populations. A measured alpha or gas-content value may fail to capture changes in the nuclei number/size distribution; that distribution remains separately referenced and its missing/partial coverage is explicit.

The ITTC Nawigator XXI Cavitation & Noise Round Robin is a natural external target for comparing occupancy, event rates, spectral distributions and inter-facility repeatability. The 2026 Nawigator XXI model-scale study specifically reports cavitation intermittency associated with insufficient nuclei and investigates post-processing to mitigate its effect.
