# LL-004 Moving-Target Encounter Oracle — Evidence Boundary

## Purpose

This artifact independently checks the first moving-receiver mathematics needed by the lunar launcher/catcher Phase-0 program. It is deliberately separate from `symthaea-orbital` and imports no Symthaea code.

## Executed synthetic evidence

`python scripts/ll-cislunar-moving-target-oracle.py` passed before commit.

The self-test covers:

- exact cubic-Hermite reproduction of a constant-velocity target track;
- a closed-form zero-gravity moving-target intercept at exactly 10 s with numerical miss below `1e-8` in the synthetic units;
- an explicit target-phase/timing shift that removes the zero-miss intercept;
- deterministic replay of a positive-mu central-gravity fixture.

The zero-gravity mode exists only to provide exact analytic closest-approach fixtures. Operational lunar studies must use a declared physical dynamics model.

## Model hierarchy

The intended LL-004 evidence ladder is:

1. analytic inertial moving-target fixtures (`mu = 0`) for exact closest-approach geometry;
2. synthetic Moon-centered two-body pod/target studies;
3. production parity against `symthaea-orbital` LL-004 implementation;
4. checked-in, provenance-bound target tracks generated from external ephemeris sources under LETN-002 frame/time contracts;
5. higher-fidelity cislunar dynamics only where they materially change the required trade-study answer.

## External context

JPL Horizons provides ephemerides for spacecraft and Earth-Moon dynamical points, but frame, time scale, origin, and output conventions must be explicit. Runtime network access is not part of the planned hot path; reference tracks should be generated externally and checked in with exact provenance.

Gateway's operational NRHO is strongly time-varying, so it must be represented by a moving target state/track rather than a single altitude or fixed point.

## Non-claims

Passing these fixtures does **not** establish:

- a real Gateway/NRHO trajectory;
- acceptable catcher relative speed;
- surface-site to inertial release accuracy;
- lunar orientation/rotation fidelity;
- three-body or n-body accuracy;
- navigation or timing accuracy;
- terminal guidance performance;
- safe release or capture authority.

A small closest-approach distance is only a study result. It is not a rendezvous, capture, or safety claim.
