# Regenerative Vehicle Health Architecture v0.1

This document defines the cross-domain architecture for machines that can detect degradation, constrain operation, invoke a qualified intervention, and verify recovery.

## Canonical ownership

- **Symthaea**: sensing representation, temporal inference, prediction, digital-twin residuals, candidate interventions.
- **Mycelix**: asset/component identity, provenance, authorization, lifecycle attestations, repair records, evidence exchange.
- **Deterministic health gate**: converts declared evidence and policy into reproducible health states.
- **Physical system**: sensing, materials, actuators, repair mechanisms, redundancy.
- **External authority**: remains authoritative for regulated operation; software inference does not establish certification.

## Core invariant

Repair is not recovery.

A system may enter Healing after an authorized intervention begins, but it may enter Recovered only when fresh, provenance-bearing, independently verified post-intervention evidence satisfies the declared recovery policy.

## Health lifecycle

Nominal -> Suspect -> Restricted -> RepairPending -> Healing -> VerificationPending -> Recovered

Quarantined is entered whenever required evidence is absent, stale, future-dated, contradictory, or outside the qualified envelope.

## Digital twin boundary

A twin can propose diagnosis or intervention. It cannot establish physical recovery by itself.

The existing helicopter divergence machinery already implements bounded residuals, persistence, freshness, evidence requirements, and canonical reporting. The vehicle contract should reuse those semantics rather than create a second incompatible assurance model.

## Vehicle-domain specialization

The same contract should support:
- road vehicles: batteries, bearings, tires, motors, thermal systems;
- marine vehicles: hulls, propulsion, corrosion, leaks;
- rotorcraft: rotor/drivetrain/structural health;
- fixed-wing aircraft: structural fatigue, propulsion and control systems;
- remote/space systems: pressure, thermal, radiation and subsystem health.

## Research crucible

Before physical self-repair, execute deterministic simulation scenarios for:
- bearing degradation;
- battery thermal anomaly;
- tire/rotor imbalance;
- hull leak;
- composite delamination;
- sensor dropout;
- stale telemetry;
- contradictory evidence;
- successful intervention;
- partial intervention;
- false recovery claim.

Acceptance is evidence that the architecture distinguishes damage, intervention, and verified recovery.

## PR sequence

1. **Symthaea / vehicle health gate** — establish executable state/evidence contract.
2. **Symthaea / twin-health adapter** — translate digital-twin divergence reports into the regenerative gate without duplicating policy logic.
3. **Symthaea / damage-intervention crucible** — deterministic scenario matrix and evidence capsules.
4. **Mycelix / asset provenance schema** — model component identity, installation, intervention authority, repair event and verification attestations.
5. **Mycelix / transport integration** — connect the contract to the existing transport commons only after the provenance semantics are stable.
6. **Standalone platform adapters** — marine and aerospace adapters consume the same semantic contract; no bespoke per-platform truth model.
7. **Physical prototype** — only after simulation and evidence paths are deterministic.

## Standalone-repository rule

The standalone repositories are canonical development surfaces. The Luminous Dynamics monorepo should mirror/integrate the contract rather than become the sole source of truth.

Cross-repository changes should carry:
- canonical schema version;
- compatibility statement;
- source commit SHA;
- evidence-capsule reference;
- explicit owner/repository;
- migration notes where semantics change.
