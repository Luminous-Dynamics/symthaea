# Marine cavitation-conditioned vibration coherence audit

A strict window-level join and condition-stratified summary over two independently generated evidence receipts:

- #11713 cavitation-window/intermittency evidence, including raw cavitation annotation, alignment eligibility, video/acoustic trace identities and acquisition times;
- #11828 hydrophone–accelerometer coherence estimates, including per-frequency coherence, reference-coherent pressure PSD, residual pressure PSD, calibration/timebase and uncertainty.

## Run

~~~sh
cargo test --manifest-path tools/marine-cavitation-conditioned-coherence-audit/Cargo.toml
cargo run --manifest-path tools/marine-cavitation-conditioned-coherence-audit/Cargo.toml -- \
  docs/research/marine/fixtures/synthetic-marine-cavitation-conditioned-coherence-v0.1.0.json
~~~

The synthetic fixture must return INCOMPLETE as physical evidence.

## Join requirements

Every joined window must match on:
- run/window ID, start/end time and acoustic raw-trace SHA-256;
- cavitation-observation SHA-256, raw annotation, conditioning state and declared alignment offset/expanded uncertainty;
- operating-point receipt, vessel/propeller revision, service/environment fingerprints;
- hydrophone instrument/channel/calibration, pressure raw trace, vibration instrument/channel/location/orientation/calibration;
- shared timebase ID, timebase-binding receipt, synchronization receipt;
- coherence estimator fingerprint, cross-spectrum convention, sample rate/count, exact band-ID set and frequency grid.

The tool recomputes whether the acoustic/video alignment is eligible. It never turns an unaligned or unknown label into cavitation-absent.

## Statistics and interpretation

For each run and band, the audit reports cavitation-present and cavitation-absent empirical distributions of:
- magnitude-squared coherence (dimensionless);
- vibration-reference-coherent pressure PSD (linear Pa²/Hz);
- non-coherent residual pressure PSD (linear Pa²/Hz).

The aggregate first computes each run's group means and present-minus-absent deltas, then weights independent runs equally. It also reports between-run sample dispersion. It does not pool adjacent windows as independent vessel/facility replicates, and reports no p-values or confidence intervals.

Per-window marginal lower/upper envelopes are averaged and explicitly labelled **not a joint confidence interval**. Magnitude coherence remains usable when H1 phase is unresolved; no phase or source attribution is inferred from that condition.

## Claim ceiling

This result is a descriptive association between cavitation-observed window state and pressure power linearly coherent with one declared accelerometer reference, within the supplied operating envelope. Coherence does not identify a unique physical source: common-cause excitation can also create coherence. The residual is not a cavitation-only component.

No causal effect, unique source identity, acoustic source level, model-to-full-scale transfer, ecological impact, quiet-ship compliance or noise reduction is claimed. SHA-256 strings are evidence identifiers, not authentication or raw-byte verification.
