# LL-004E Site -> Inertial Release Oracle

## Purpose

Provide an implementation-independent reference for transforming a declared lunar surface-site release into an inertial release state for LL-004 moving-target studies.

The oracle intentionally does **not** contain a lunar orientation model. It consumes an externally supplied body-fixed -> inertial rotation and angular velocity with declared provenance.

## Executed synthetic evidence

The checked-in standard-library Python oracle self-test was executed before commit and passed.

It covers:
- identity body-fixed -> inertial rotation with zero spin;
- exact +90 degree Z rotation;
- pure surface rotation, verifying `v = omega x r` with zero site-relative release speed;
- finite orthonormal north/east/up basis at a near-polar site;
- rotation inverse/symmetry fixture;
- fail-closed malformed rotation matrix.

These are synthetic mathematical fixtures only.

## Lunar frame policy

For high-fidelity lunar studies, do not silently default to a simple `IAU_MOON` rotation model. NAIF treats Earth and Moon as special cases and provides DE-specific lunar Principal Axes and Mean Earth frames (`MOON_PA_DExxx`, `MOON_ME_DExxx`) plus generic aliases. A future external snapshot generator should bind the exact PCK/FK/kernel lineage, epoch/timescale, rotation, and angular velocity used by the study.

Runtime SPICE/network access is not required in the physics hot path. Checked-in orientation snapshots are preferred for reproducible studies.

## Non-claims

This artifact does not:
- select a real South-Pole site;
- establish the correct real-site radius/elevation;
- implement SPICE;
- qualify a DE440 lunar frame;
- include LOLA terrain in the site radius;
- establish launcher pointing/nav tolerances;
- authorize release.

It is a Phase-0 frame-transform oracle only.
