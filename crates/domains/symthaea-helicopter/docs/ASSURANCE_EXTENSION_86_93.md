# Symthaea Helicopter Assurance Extension 86–93

This extension closes eight remaining operational assurance gaps after the
78–85 series. The additions are deliberately fail closed and preserve the
boundary between deterministic software evidence and independently validated
physical, cryptographic, regulatory, and human evidence.

## Patch 86 — Digital-twin divergence

Adds runtime normalized-residual monitoring across declared aircraft signals.
Warnings and unsafe outcomes require persistence; stale, sparse, or unsupported
samples are incomplete rather than treated as model agreement.

## Patch 87 — Electrical power distribution

Adds source capacity, independent failure domains, bus availability, voltage,
priority allocation, brownout, load shedding, and essential-load continuity.
The module is a bounded allocation model, not a detailed circuit solver.

## Patch 88 — Network partition supervision

Adds authenticated-link grace periods, bounded onboard autonomy, return and
landing behavior, local evidence preservation, and reconciliation before remote
commands can resume.

## Patch 89 — Maintenance trend evidence

Adds deterministic slope, residual-noise, sampling-gap, and abrupt-level-shift
gates. It intentionally reports no remaining-useful-life prediction.

## Patch 90 — Fleet anomaly detection

Adds configuration-bound robust median/MAD outlier detection and independent
qualified bounds so common-mode fleet degradation cannot disappear into the
cohort median.

## Patch 91 — Incident reconstruction

Adds one ordered timeline across commands, sensors, faults, alerts, modes,
updates, and maintenance. Explicit causal annotations remain candidate links;
causation is never declared proven by this module.

## Patch 92 — Assurance delta analysis

Compares baseline and candidate assurance artifacts, applies declared impact
rules, and requires fresh revalidation evidence for every affected artifact
kind. Critical artifact removal is rejected.

## Patch 93 — Progressive fleet rollout

Adds shadow, canary, limited-fleet, and broad-fleet gates with minimum cohort,
dwell, incident, divergence, drift, evidence, rollback-target, and restriction
criteria.

## Claim boundary

These modules improve deterministic analysis, operational controls, and evidence
structure. They do not independently establish airworthiness, validated model
accuracy, electrical certification, maintenance prognostic validity,
cryptographic authenticity, regulatory approval, or safe autonomous operation.
