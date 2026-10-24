# Regenerative Health: Sensing-Layer Qualification v0.1

## Purpose

A regenerative health system must distinguish a physical anomaly from an anomaly in the measurement path.

A sensor fault can resemble structural damage. A stale or configuration-mismatched measurement can also look internally coherent while being operationally invalid. The sensing layer is therefore an evidence boundary upstream of structural diagnosis and recovery qualification.

This contract is deliberately deterministic and platform-neutral.

## Pipeline

Physical state -> sensor field -> sensor-health gate -> structural-health observation -> regenerative-health gate -> intervention -> independent recovery verification.

The sensor-health gate does not diagnose the physical component and does not repair a sensor. It answers a narrower question:

> Is this measurement admissible as evidence for downstream health reasoning?

## States

- Trusted: observation is fresh, internally valid, configuration-compatible, and below the degraded threshold.
- Degraded: observation is usable only with an explicit degraded-sensing interpretation.
- Untrusted: the sensor residual exceeds the configured untrusted threshold.
- Quarantined: evidence is invalid, stale, future-dated, missing, or configuration-mismatched.

Only Trusted sensor evidence may be admitted through admit_health_observation.

## Threats explicitly covered

1. Sensor drift / bias - represented as increasing normalized disagreement with the admissible sensing manifold.
2. Dropout / missing evidence - missing evidence identifiers are quarantined.
3. Stale telemetry - age is bounded deterministically.
4. Future-dated telemetry - impossible temporal evidence is quarantined.
5. Configuration substitution - expected and observed configuration digests must match when an expected digest is supplied.
6. Severe sensing divergence - observations beyond the untrusted threshold cannot silently become trusted structural evidence.

## Why this boundary exists

The 2026 aerospace sensing-digital-twin literature explicitly identifies sensor-health assessment as a missing layer in many SHM pipelines: structural degradation and sensor faults can produce similar measurement anomalies. The reported approach uses physics-aware classification, signal correction, and per-sensor reliability estimation before downstream structural-health logic. This crate adopts the architectural lesson without importing that paper's ML implementation.

The goal is not to make Symthaea's health gate more intelligent; it is to make the evidence boundary harder to fool.

## Non-goals

This module does not:

- claim a vehicle is physically safe;
- certify a repair;
- replace regulated engineering authority;
- infer a physical failure mechanism from a sensor residual;
- treat sensor correction as equivalent to independent verification;
- require network connectivity.

## Future extension

A later adapter can derive SensorObservation from real strain, vibration, pressure, temperature, current, optical-fiber, or other sensing channels and preserve the raw evidence digest in Mycelix. Multi-sensor corroboration can then be added without changing the core structural recovery contract.
