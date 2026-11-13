# Marine coherent-noise audit

A deterministic, single-reference cross-spectral calculation and evidence audit for a synchronized shaft/machinery vibration channel and a hydrophone.

## Run

~~~sh
cargo test --manifest-path tools/marine-coherent-noise-audit/Cargo.toml
cargo run --manifest-path tools/marine-coherent-noise-audit/Cargo.toml -- \
  docs/research/marine/fixtures/synthetic-marine-coherent-noise-v0.1.0.json
~~~

The synthetic fixture must return INCOMPLETE as physical evidence.

## Calculation

The declared convention is Gxy = X · conjugate(Y), where X is the reference signal and Y is the hydrophone pressure signal. For each frequency bin:

- magnitude-squared coherence: gamma² = |Gxy|² / (Gxx Gyy)
- reference-coherent hydrophone PSD: Gyy,coh = |Gxy|² / Gxx
- residual PSD: Gyy,res = Gyy - Gyy,coh

All partition arithmetic is performed in linear spectral power. PSDs must form a positive-semidefinite 2×2 spectral matrix. The coherent/residual pressure levels are reported relative to 1 µPa²/Hz. Zero-power component levels and zero-cross-spectrum phase are null, not negative infinity or an invented phase.

## Eligibility

The two channels must use identical sample rates, record lengths, segment/FFT sizes, overlap, window and average counts. Their raw-trace and calibration identities, sensor locations/orientations, a shared timebase receipt, an alignment receipt/uncertainty and an operating-point receipt are retained. Frequency-bin indices must match the declared FFT resolution. A predeclared minimum number of averages and coherence threshold are required.

The output disposition COHERENCE_SCREENED means only that the declared spectral matrices and acquisition contract passed the audit. Per-bin COHERENT_WITH_DECLARED_REFERENCE means the linear component coherent with that reference exceeded the declared threshold. A low coherence value is not evidence that the reference source is absent.

## Claim ceiling

Coherence is statistical linear association with the selected reference, not unique causal attribution. Shared drivers, structural coupling and correlated sources may all produce coherence. The residual contains everything not linearly coherent with this one reference. It is not automatically propeller noise or ambient noise.

V0 does not implement multi-reference partial coherence, beamforming, operational transfer-path analysis, blind source separation, causal inference, source-level transfer, ecological exposure or regulatory qualification. No statistical confidence interval is claimed for coherence or the power partition; the reported average count and estimator configuration are preserved so a later estimator-specific uncertainty method can be added.

The synthetic known-answer fixture contains a bin with Gxx=4, Gyy=19, |Gxy|=8, giving gamma²=16/19, coherent pressure PSD=16 and residual PSD=3, plus a low-coherence negative-control bin. These are arithmetic controls, not physical results.
