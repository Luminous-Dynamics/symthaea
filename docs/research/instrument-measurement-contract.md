# Instrument Measurement Contract

**Status:** research engineering contract; not a medical device or clinical validation.  
**Crate:** `symthaea-instrumentation`  
**Scope:** scalar measurement envelopes, raw-data and calibration-evidence resolution, freshness, quality gates, and stream ordering.

## Why this is a separate crate

Ultrasound is the first use case, but calibrated measurements are cross-cutting. Keeping the contract in a dependency-light crate avoids coupling a generic evidence envelope to acoustic equations or the robotics physics backend. Domain crates should translate their native data to this envelope at the adapter boundary instead of duplicating provenance rules.

## Envelope requirements

Each measurement binds:

- instrument and channel identifiers (not patient identity); the assessment receipt retains this identity and the named clock epoch;
- monotonically sequenced acquisition, a monotonic timestamp, and an explicit `ClockDomainId` naming the clock origin/epoch;
- explicit quantity and unit, checked for dimensional compatibility;
- finite value and non-negative standard uncertainty in the declared unit;
- a version/digest identifying the processing chain;
- optional raw-data and calibration references, each identified by an ID and SHA-256 digest;
- explicit acquisition-quality flags.

The initial unit set covers common engineering signals and research-healthcare prototypes: distance, time, frequency, acoustic pressure, temperature, electrical potential/current, heart rate, respiratory rate, dimensionless percentages and explicitly typed oxygen-saturation percentages, speed, acceleration, angular velocity, sound-pressure level referenced to 20 µPa, and dimensionless values. Values are never silently converted. Extending the unit vocabulary requires an explicit quantity mapping and tests.

## Fail-closed quantitative gate

The strict assessment checks sequence/time ordering using a caller-maintained `MeasurementStreamGuard` and rejects:

- mismatched clock-domain context, future-dated or stale data;
- any declared quality flag (saturation, motion artifact, lead-off, missing samples, unsynchronized clock, failed self-test, out-of-range values, unknown signal quality, etc.);
- missing raw-data or calibration references;
- raw acquisition bytes that cannot be resolved, are empty, or do not match the envelope's artifact ID/digest;
- unresolved calibration evidence;
- resolver results whose calibration record ID or evidence artifact differs from the measurement reference;
- calibration evidence for a different instrument/channel, quantity/unit, validity interval, or calibrated value range;
- measurement or calibration uncertainty above the consumer's explicit limits.

The raw-data and calibration resolvers are explicit trust boundaries. The raw-data resolver must actually retrieve the referenced bytes, compute/verify their digest, and return non-empty artifact metadata; merely looking up a database row or echoing the supplied reference is insufficient. The calibration resolver must resolve and review artifact bytes/digests, the review receipt, exact instrument/channel identity, supported quantity/unit, validity dates, inclusive calibrated value range, calibration chain and uncertainty contributions. The strict gate rejects a calibration record for another channel or a measured value outside the stated range. The crate checks returned IDs/digests against the observation, but cannot prove either resolver is independent or authoritative by itself. Configure and test both implementations independently. The returned `MeasurementAssessment` is read-only through its public API and retains the instrument, timestamp domain, observed value, calibration evidence/review references, calibrated range, raw artifact metadata, and uncertainty limits that passed. It is not a cryptographically signed receipt or clinical authorization.

Measurement and calibration standard uncertainties are reported separately. The crate does not combine them by root-sum-square because doing so requires explicit assumptions about independence and all material uncertainty contributors. Traceability is a property of a measurement result connected through an unbroken, documented calibration chain in which each link contributes to uncertainty; a certificate reference alone does not establish traceability.

## Stream ordering

`MeasurementStreamGuard` keeps the highest consumed sequence and the last accepted non-future timestamp per instrument/channel. Repeated or lower sequences are rejected without changing state. A newer sequence with a backward or future timestamp is consumed, but the last accepted timestamp remains unchanged; retrying that sequence cannot turn the rejected sample into an accepted replay, and a future-dated sample cannot poison the timestamp high-water mark. Every envelope carries a `ClockDomainId`; the gate requires the caller's current-time reference to use the exact same ID, so monotonic values are not compared across unrelated clock origins. A reboot/reset should get a new clock-domain ID, and an existing in-memory stream guard rejects a domain change rather than silently comparing the new epoch with old timestamps. An epoch transition therefore requires controlled reconstruction of the guard from an authenticated checkpoint or a newly established stream identity; simply clearing state is not evidence of a safe reset. Callers must use documented, appropriately synchronized clock sources; the ID check is not a clock synchronization service. The quantitative-use API requires the guard and advances its sequence state as soon as an envelope reaches the gate—even if subsequent quality or calibration-evidence checks reject it. This is intentionally conservative and means the source must send a new sequence after any rejection. The guard is in-memory, not a tamper-proof replay ledger; safety-sensitive deployments must persist/reconcile stream checkpoints across restarts and authenticate the source.

## Validation strategy

1. Unit tests reject bad dimensions, non-finite values, negative uncertainty, malformed identifiers and digests.
2. Gate tests exercise freshness, quality flags, clock epochs, absent provenance, unresolved raw/calibration artifacts, evidence mismatch, instrument/channel mismatch, wrong units, validity windows, calibrated-range enforcement and uncertainty limits.
3. Clock-domain tests verify mismatched caller clocks fail before consuming stream state and unexpected source epoch changes are rejected by an existing guard. Raw-evidence tests exercise empty artifacts, unresolved acquisition bytes and artifact-ID/digest mismatch; calibration tests exercise certificate/review separation and applicability.
4. Stream tests exercise duplicate/replayed sequence numbers, time reversal and rejection of future timestamps without poisoning the accepted timestamp floor.
5. In the next integration step, connect this contract to synthetic biomedical signals and ultrasound simulator output, then use independently generated raw-acquisition and calibration-review fixtures. The current crate does not acquire hardware or claim a clinical use is safe.

Run in the workspace:

```sh
cargo test -p symthaea-instrumentation
cargo clippy -p symthaea-instrumentation --all-targets -- -D warnings
```

A command being queued is not a pass. Preserve the exact commit and the completed job logs when reporting qualification.

## Research and standards foundations

- [NIST — Metrological Traceability](https://www.nist.gov/metrology/metrological-traceability): traceability requires a documented unbroken calibration chain, with uncertainty contributions at each link. NIST also cautions that traceability alone does not guarantee fitness for purpose.
- [NIST — Measurement Uncertainty](https://www.nist.gov/glossary-term/39291): uncertainty is a non-negative parameter describing dispersion attributable to a measurement result.
- [DICOM PS3.17 — Specification of Standard Measurements](https://dicom.nema.org/medical/dicom/current/output/chtml/part17/sect_DDDD.2.html): standard measurement definitions need enough detail for consistent acquisition and interpretation. This contract is not a DICOM implementation.
- [IMDRF — Good Machine Learning Practice for Medical Device Development](https://www.imdrf.org/documents/good-machine-learning-practice-medical-device-development-guiding-principles): relevant lifecycle and data-quality principles for future ML-enabled features.
- [FDA — Quality Management System Regulation](https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr): became effective on 2 February 2026 and incorporates ISO 13485:2016 by reference for the U.S. device quality-system framework. This note is engineering guidance, not a jurisdiction-specific compliance determination.
- [SAHPRA — Medical Device and IVD Guidance](https://www.sahpra.org.za/medical-devices/): South African regulatory context must be checked against the specific intended use and current applicable guidance before any clinical or commercial deployment.

## Privacy boundary

No patient identifier is part of the envelope. Raw acquisition references can still resolve to sensitive medical data; real integrations must apply access control, encryption, retention limits, audit logging, and appropriate de-identification/consent controls before research reuse. The test fixtures use synthetic identifiers and values, not patient records.

## Explicit non-claims

This contract does not establish diagnostic accuracy, sensor calibration authenticity, metrological traceability by itself, acoustic exposure safety, image quality, patient safety, regulatory conformity, or clinical efficacy. Quantitative assessment is only one evidence gate in a larger intended-use and safety case.
