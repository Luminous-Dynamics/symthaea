# Lifecycle Assurance and Decommissioning Protocol

**System:** `symthaea-subterranean`
**Campaign:** XIII
**Date:** 2026-07-20
**Status:** deterministic software protocol; physical qualification remains external

## 1. Purpose

This protocol governs long-horizon degradation forecasting, predictive maintenance,
assurance-evidence freshness, learned-policy succession, tunnel-site closure, and
terminal machine decommissioning. It exists to prevent a system that is safe on one
mission from silently becoming unsafe across repeated missions, stale evidence,
maintenance deferral, controller replacement, or end-of-life handling.

The lifecycle layer is safety-monotonic. It may remove productive or mobility
authority, require return, require hold, or make decommissioning terminal. It cannot
create actuator authority or override physical hazards, operator locks, return
reserve protection, field envelopes, actuator isolation, or runtime invariants.

## 2. Authority ordering

The relevant authority order is:

1. malformed input and hardware-frame rejection;
2. physical hazard assessment and latching;
3. return-path and reserve protection;
4. operator, degraded-operation, and partition constraints;
5. field survivability and actuator isolation;
6. lifecycle assurance restrictions;
7. nominal learned or reflex control;
8. final independent invariant enforcement;
9. physical actuation and evidence recording.

Lifecycle assurance therefore constrains nominal intent before actuation, while the
final invariant monitor remains independent of it.

## 3. Degradation forecasting

The degradation forecaster observes persistent component health and estimates:

- the critical component;
- a bounded service horizon;
- a later abort-risk horizon;
- confidence in the estimate;
- whether the system is warming up, stable, due for service, or at abort risk.

A forecast is advisory only while uncertainty is high. Once the service horizon is
credible, the maintenance-window planner may finish only bounded work, require a
return to service, require a hold, or refer the asset for retirement review.
Forecasting never restores a failed component and never substitutes for physical
inspection.

## 4. Assurance-evidence freshness

Release evidence is represented as deployment-bound credentials for calibration,
hardware-in-the-loop qualification, environmental qualification, field trials,
certification, and governance. Each credential has a generated step and an expiry
step.

When enforcement is enabled:

- missing, stale, or deployment-mismatched evidence removes productive authority;
- an underground asset returns when a return remains feasible;
- an asset already at a surface or service location holds;
- renewal-due evidence is visible but may retain productive authority until expiry;
- a stricter missing or stale condition cannot be masked by a renewal warning.

The crate validates metadata and authority effects. It does not authenticate the
external evidence artifact or accredited laboratory.

## 5. Policy succession

A successor policy must be registered in shadow, enter canary under the existing
promotion controls, and accumulate a configured healthy overlap while the
predecessor remains active. Cutover authority is revoked immediately if a later
overlap frame becomes unhealthy. A terminal or malformed succession plan cannot
bypass policy identity, lineage, or lifecycle validation.

Succession readiness does not itself promote a policy. The existing role-separated
promotion and governance systems retain that authority.

## 6. Site closure stewardship

Every non-surface node in the current tunnel graph is synchronized into a bounded
closure ledger. Existing closure records are never overwritten by later graph
snapshots. A newly discovered underground node therefore prevents a previously empty
ledger from falsely claiming the site is ready for machine decommissioning.

A node may advance through survey, stabilization, monitoring, and closure only when:

- survey confidence is adequate;
- hazards are contained;
- no work remains active;
- no agents remain present;
- closure evidence is identified.

Closed nodes are terminal. Blocked-node counts are per record rather than per failure
condition, preventing evidence summaries from double-counting one site.

## 7. Machine decommissioning

Decommissioning is a two-stage ceremony:

1. **Request:** two distinct hardware-backed identities, including a Supervisor and
   Safety Officer, authorize one nonzero ceremony ID.
2. **Completion:** the crate recomputes physical conditions from live state.

Completion requires:

- surface or service-bay location;
- no active work;
- clear physical hazards and valid sensor state;
- a zero final actuator command;
- site-closure readiness;
- an external assertion that required evidence has been preserved.

After completion, the asset is terminally `Decommissioned`. Checkpoint loading,
controller replacement, reset, or later operator intent cannot restore motion.

## 8. Persistence

Operational checkpoint schema v6 persists the complete lifecycle supervisor,
including degradation state, maintenance-window policy, evidence freshness,
succession state, asset lifecycle, site-closure records, and last assessment.
Checkpoint validation occurs before live mutation. Invalid lifecycle state rejects
the checkpoint.

## 9. Evidence and release gates

Lifecycle evidence records:

- forecast disposition and service/abort horizons;
- critical component and forecast confidence;
- maintenance-window decision;
- evidence-freshness disposition and missing/stale counts;
- policy-succession state;
- site-closure readiness and synchronization;
- asset lifecycle state;
- productive-work and return authority.

The release validator executes deterministic contracts for forecasting,
freshness-based authority removal, healthy succession overlap, terminal
decommissioning, occupancy-safe site closure, checkpoint validity, command-level
authority, and terminal checkpoint continuity.

A self-consistent lifecycle evidence bundle binds deployment identity, source tree,
build identity, live supervisor state, cached assessment, validation report, and the
five canonical lifecycle requirements. Cached assessments are recomputed against the
stored supervisor and cannot overstate authority.

## 10. Explicit non-claims

This protocol does not establish:

- accredited remaining-useful-life prediction;
- physical component lifetime or maintenance intervals;
- cryptographic evidence authenticity;
- legal sufficiency of evidence retention;
- environmental remediation completeness;
- regulatory permission to close a mine or tunnel;
- safe physical de-energization, dismantling, recycling, or disposal;
- production readiness without real workspace, HIL, chamber, and field validation.

## 11. Production qualification still required

Before deployment, the real Rust 1.94 workspace must pass formatting, Clippy,
full tests, HIL fault injection, calibrated degradation studies, long-duration
thermal and wear campaigns, power-loss checkpoint tests, cryptographic provenance,
independent site-closure review, and jurisdiction-specific decommissioning and
environmental approval.
