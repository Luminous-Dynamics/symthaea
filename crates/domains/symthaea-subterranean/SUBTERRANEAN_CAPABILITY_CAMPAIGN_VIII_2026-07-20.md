# Subterranean Capability Campaign VIII

**Date:** 2026-07-20  
**Theme:** Field survivability under partial failure  
**Baseline:** Capability Campaign VII / patch 63

## Campaign objective

Close the field-readiness gap between a system that can identify hazards and one that remains causally truthful when parts of its sensing, actuation, power, thermal, or communications stack degrade.

## Delivered patch sets

### 64 — Redundant sensor quorum

Adds bounded multi-source observations, source replay protection, physical-range validation, median fusion, disagreement-derived source reliability, isolation, and critical-channel quorum.

### 65 — Actuator response isolation

Adds command/response monitoring for ten physical actuators. Persistent non-response removes only the failed actuator's authority and requires explicit service to clear.

### 66 — Power and thermal envelope

Adds continuous command caps and cooling floors derived from battery, temperature, coolant, and maintenance truth.

### 67 — Capability profile

Composes sensing, actuator availability, power, thermal, and maintenance into full-mission, reduced-work, return-only, or hold-for-recovery dispositions.

### 68 — Runtime integration

Places redundant fusion before hazard assessment, field envelopes before physical derating, actuator isolation immediately before plant actuation, and response observation after the plant step.

### 69 — Checkpoint persistence

Advances operational checkpoint schema to v3 and persists sensor replay/reliability, actuator isolation, and envelope state.

### 70 — Survivability evidence

Adds bounded frame and summary evidence for sensor quorum, source reliability, actuator isolation, power/thermal margins, and capability state.

### 71 — Partition recovery

Adds bounded grace, local-autonomy, return-to-mesh, hold-and-beacon, and reconciling modes.

### 72 — Partition-safe runtime authority

Removes team authority during reconciliation, constrains ordinary motion, persists partition state, and records partition evidence.

### 73 — Acceptance contracts

Adds deterministic contracts for critical quorum, robust median fusion, actuator isolation, thermal prioritization, graceful return, partition reconciliation, and checkpoint replay continuity.

### 74 — Cooling-authority regression closure

Fixes a full-suite finding where a late thermal rule could reintroduce pump demand after maintenance declared cooling unavailable.

### 75 — Field survivability protocol

Documents authority ordering, persistence, evidence, and explicit non-claims.

### 76 — Mechanical formatting normalization

Applies repository-wide Rust formatting as an isolated non-functional patch.

## Acceptance results

Offline API-compatible workspace:

- warning-denied type check: required before packaging;
- unit tests: 187 passed, 0 failed, 1 ignored hardware timing benchmark;
- deterministic survivability contract suite: all seven contracts passed;
- exact patch-tree reconstruction: required for incremental and full-series bundles.

The offline environment does not contain the complete Rust 1.94 workspace, real HDC/FEP crates, production Serde configuration, authenticated hardware adapters, or controlled hardware. Those remain authoritative integration gates.

## Highest-value next campaign

A further campaign should leave software-only simulation and establish calibrated hardware-in-the-loop fault campaigns:

- correlated sensor failures and common-mode power faults;
- measured actuator response envelopes;
- thermal-soak and battery-sag profiles;
- radio partition traces through representative geology;
- power-loss checkpoint interruption;
- real 200 Hz latency under logging, fusion, and reconciliation load.
