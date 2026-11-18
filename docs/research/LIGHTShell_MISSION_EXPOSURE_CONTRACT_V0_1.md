# Atelier Lightshell — Mission-Profiled Textile Exposure Contract v0.1

## Purpose

Convert approved robot operating profiles and calibrated mission observations into reproducible textile test exposures. The test should represent how a garment zone is actually loaded, not an arbitrary generic cycle count.

This is an adapter over the existing JointLab, robot ODD, endurance, measurement and textile test owners. It is not a telemetry archive, ODD authority, reliability model, test executor, or robot lifetime certificate.

## Exposure types

- Joint flex/articulation: range, rate, dwell and cycle distribution.
- Vibration: frequency-binned acceleration profile, including peaks and high-exposure bands; scalar RMS alone is insufficient.
- Thermal and humidity: distributions and dwell intervals rather than only one maximum.
- Abrasion/contact: event/cycle counts, interface and exposure distribution.
- Cleaning: cycle count and cleaning-agent concentration where measured.
- Repair/service: intervention counts and repair-duration distributions.

Retain min/max, median, p95/p99 and sample counts. Preserve the source traces and frequency bands. Never replace missing, saturated, drifting or unsynchronized measurements with zero or a convenient nominal value.

## Provenance chain

A profiled mission exposure binds exact platform article revision and geometry digest, approved ODD profile, mission mix and service horizon. Every physical source run binds exact robot revision, time window, clock synchronization, instrument and calibration identity, raw-data digest, data-quality disposition and missing-data fraction.

Only measured or calibrated-derived observations can support a profiled plan. Estimates, simulation outputs and synthetic fixtures remain useful for software testing but cannot be promoted into physical mission evidence. Derived exposures must cite the transformation/derivation profile and uncertainty model.

## Sequence matters

Textile degradation is often interaction-dependent. A wash followed by flex may not behave like the sum of an isolated wash test and an isolated flex test. Thermal dwell plus vibration, wet abrasion, repeated impact/contact, and repair followed by re-use are separate test hypotheses.

Represent ordered test blocks and named interaction groups. Compare isolated and combined sequences where the actual mission suggests coupling. Preserve the raw/source references so the test sequence can be regenerated.

## Acceleration and lifetime claims

An accelerated test can be useful for screening, but it does not establish a field-life equivalence by naming a multiplier. Lifetime equivalence requires a validated acceleration model and representative correlation evidence showing that the accelerated exposure preserves the relevant failure mechanism and ranking. Until then, accelerated loads are explicitly unvalidated hypotheses and cannot support a service-life claim.

A coupon plan is not an integrated robot lifetime qualification. Coupon, transition, module and integrated-robot evidence remain separate.

## P0 adversarial controls

Authored tests cover missing robot/ODD identity, article-revision mismatch, missing/saturated/unreviewed source, missing-data substitution, bounded data gaps, simulation-to-physical promotion, non-monotone quantiles, missing vibration spectrum, overlapping frequency bins, duplicate block sequence, unvalidated acceleration, missing sequence group and distinct exposure requirements for sequence-interaction blocks.

## Benchmarks

1. Zone exposure coverage against approved ODD requirements.
2. Distribution and tail preservation against raw measurement traces.
3. Exact replay/reproducibility from source digests.
4. Lab exposure deviation against held-out mission runs.
5. Detection of high-severity tail events and correlated exposures.
6. Correlation of accelerated tests against representative unaccelerated reference tests.
7. Additional failure modes exposed by sequenced tests but missed by isolated tests.

Zero silent missing-data substitution, zero unsupported measured claims and zero unvalidated acceleration-factor promotion are hard gates. Numeric coverage and uncertainty targets must be preregistered after the article, sensors, ODD and sample strategy are frozen.

## Integration

- JointLab / robot physical source: Symthaea #4814.
- Robot operating domain: Symtropy #3505.
- Endurance and maintainability: Symtropy #3628.
- Zone requirements: Symthaea #12742.
- Coupon evidence: Symthaea #12751.
- Method applicability: Symthaea #12759.
- This adapter: Symthaea #12761.

## Current state

The fixture is synthetic and proposed. There is no physical robot mission profile, actual JointLab run, measured frequency spectrum or service-life prediction assigned. Rust tests are authored but not yet compiled or run.