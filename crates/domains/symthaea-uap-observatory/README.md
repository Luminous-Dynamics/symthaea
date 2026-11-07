# symthaea-uap-observatory

A std-only first milestone for calibration-aware, provenance-preserving
sensor evidence. It is intentionally a measurement-quality layer, not an
object classifier or a theory of UAP propulsion.

## First milestone

- Typed sensor modality and coordinate frame.
- Distinct measurement payloads for bearings, range-bearing, ECEF position,
  RF spectra and magnetic vectors.
- Explicit measurement uncertainty; rejects NaN, infinity, non-positive
  standard deviations, negative range and invalid elevation.
- TAI nanosecond timestamps and explicit per-sensor clock offset / uncertainty.
- Calibration validity windows, modality/frame matching and nonzero SHA-256
  digests for the raw calibration artefact.
- Raw observation SHA-256 references and duplicate sequence rejection.
- Evidence completeness summary that distinguishes uncalibrated,
  mixed-calibration, single-sensor and multi-sensor-calibrated bundles.

Uncalibrated records are retained and marked as such. They are not deleted,
silently repaired or converted into anomalous-object claims.

## Run

```bash
cargo test -p symthaea-uap-observatory
```

## Epistemic boundary

A multi-sensor-calibrated bundle only means that its records satisfy the
structural calibration contract. It does not establish correct triangulation,
track association, extraordinary motion, object identity or non-human origin.
The next milestones must implement calibrated frame transforms, timestamp
synchronization, uncertainty propagation, blind known-object benchmarks and
conventional-explanation controls before any anomaly conclusion.
