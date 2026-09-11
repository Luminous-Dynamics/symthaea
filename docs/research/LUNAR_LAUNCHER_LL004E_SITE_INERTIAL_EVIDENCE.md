# LL-004E Lunar Site -> Inertial Release Adapter

## Purpose

Bridge a declared lunar surface launch site into the inertial `InertialReleaseState` consumed by LL-004 moving-target encounter studies without embedding a hidden lunar orientation model.

## Architecture

The production adapter consumes:
- a body-fixed site/release specification;
- an externally generated body-fixed -> inertial rotation matrix;
- the body-fixed frame angular velocity relative to inertial, expressed in the inertial frame;
- exact epoch/frame/source/evidence provenance.

It constructs a planetocentric north/east/up basis, maps local azimuth/elevation/speed into the body-fixed frame, rotates the state into inertial coordinates, then adds the surface-rotation term `omega x r` exactly once.

## Independent oracle

Draft PR #1600 is a standard-library Python implementation that imports no Symthaea code. Its self-test was executed before commit and passed identity rotation, +90-degree Z rotation, pure `omega x r`, near-pole basis, inverse/symmetry, and malformed-rotation fixtures.

## Lunar orientation policy

High-fidelity studies should use provenance-bound orientation snapshots generated from an appropriate lunar SPICE frame/kernel lineage. NAIF documents DE-specific lunar Principal Axes and Mean Earth frames (`MOON_PA_DExxx`, `MOON_ME_DExxx`) and warns that Earth/Moon are special cases where generic IAU body-fixed frames should not be blindly used for best accuracy.

No runtime SPICE or network access is required by this adapter. Checked-in orientation snapshots are preferred for reproducibility.

## Current non-claims

This tranche does not:
- select a real South-Pole launch site;
- generate SPICE orientation snapshots;
- bind LOLA terrain/elevation into `site_radius_km`;
- include navigation or pointing uncertainty;
- validate a real DE440 Moon orientation against an external state yet;
- authorize physical launch.

It is a deterministic Phase-0 frame adapter only.
