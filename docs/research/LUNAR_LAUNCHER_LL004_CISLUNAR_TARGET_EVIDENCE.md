# LL-004 Cislunar Moving-Target Production Slice — Evidence Boundary

## Purpose

This tranche connects launcher studies to moving cislunar receivers while reusing LETN-002A's explicit ephemeris/frame/timescale provenance. It does **not** create a second online ephemeris service or assume a fixed NRHO altitude.

## Implemented contract

`cislunar_target` adds:

- provenance-bound moving target tracks;
- strictly ordered time-tagged states;
- a declared maximum reference-sample gap;
- cubic-Hermite interpolation using supplied position and velocity;
- an explicit inertial pod release state;
- exact frame/timescale equality between release and target track;
- exact central-body-origin matching in the v0 two-body model;
- no target-track extrapolation;
- Moon/central-body two-body RK4 pod propagation;
- bounded closest-approach search with refinement;
- miss distance, relative speed, encounter epoch, pod state, target state, and lineage in every result;
- an optional study-only threshold that preserves `NoEncounterWithinWindow` rather than laundering a distant passage into a rendezvous.

## Independent evidence

Independent draft PR #1596 contains a standard-library Python oracle that imports no Symthaea code. Its executed self-test:

- exactly reproduces constant-velocity Hermite interpolation;
- recovers a closed-form moving-target intercept at 10 s with effectively zero numerical miss;
- demonstrates a target-phase/timing perturbation changing the encounter;
- replays a positive-mu central-gravity fixture deterministically.

The production Rust tests add frame mismatch and explicit no-encounter behavior.

## External context

JPL Horizons provides state/ephemeris products for spacecraft and Earth-Moon dynamical points, but users must explicitly select/verify time scale, reference frame, origin, and output conventions. The planned runtime path therefore consumes checked-in reference tracks with exact provenance rather than depending on live Horizons access.

NASA's Gateway NRHO varies substantially over its roughly 6.5-day orbit, from about 1,500 km to roughly 70,000 km from the Moon. That reinforces the architectural requirement that NRHO/Gateway-like receivers are moving state histories, not fixed altitudes.

## Current model limitations

The v0 production propagator intentionally omits:

- lunar physical rotation/orientation and surface-site -> inertial conversion;
- Earth/Sun third-body perturbations;
- nonspherical lunar gravity/GRAIL harmonics;
- target-track uncertainty/covariance;
- navigation errors;
- finite-burn pod correction;
- catcher geometry/capture dynamics;
- autonomous target extrapolation;
- actual Gateway/NRHO ephemeris data;
- launch or capture authority.

The central-body origin must match the declared frame origin exactly; this conservative v0 rule prevents accidentally applying a Moon-centered two-body model to barycentric/SSB states.

## Promotion path

1. independent analytic/synthetic oracle parity;
2. compiler/test evidence for this Rust tranche;
3. explicit timing-sensitivity fixtures;
4. checked-in external target tracks under LETN-002 provenance;
5. surface-site -> inertial release adapter with lunar orientation provenance;
6. compare two-body encounter results with higher-fidelity cislunar propagation;
7. feed target-state uncertainty into LL-005/006 and catcher covariance;
8. only then connect the evidence to LL-008 hard-gate study admission.

## Non-claims

A small closest-approach distance is **not** a rendezvous, capture, safe corridor, or release authorization. No real catcher speed/distance threshold is established here.
