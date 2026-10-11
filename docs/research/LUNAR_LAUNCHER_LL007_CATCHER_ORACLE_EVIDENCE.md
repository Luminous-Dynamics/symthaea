# LL-007 Catcher Momentum / Energy Oracle — Evidence Note

Status: independent first-order reference accounting. Not capture hardware qualification.

## Purpose

Phase 0 must not let a launcher architecture appear efficient by treating the receiver as a free momentum and energy sink. This oracle makes the first-order catcher ledger explicit.

## Model

For one capture event the caller supplies:

- pod mass;
- catcher mass;
- incoming and declared final relative speed magnitudes;
- capture duration;
- regeneration efficiency;
- electrical-storage efficiency;
- storage energy acceptance limit;
- fraction of capture impulse transferred to catcher translation;
- a 1-D traffic direction sign.

The oracle reports:

- pod momentum and capture impulse;
- momentum transferred to the catcher and to an explicitly external sink;
- catcher delta-v and stationkeeping impulse needed to restore that transferred momentum;
- kinetic energy removed from the pod;
- average force and constant-deceleration reference distance;
- average mechanical power;
- mechanically regenerable energy;
- stored regenerated electricity limited by storage acceptance;
- residual energy that must be dissipated/deformed/handled elsewhere.

A fleet ledger separately reports gross transferred impulse and net catcher momentum, so opposite-direction traffic may cancel net momentum only when explicitly represented; gross handling load never disappears.

## Executed synthetic fixture

The independent self-test was executed before commit for a synthetic 100 kg pod arriving at 200 m/s relative to a 10,000 kg catcher over 10 s, with 80% mechanical regeneration, 90% storage conversion, unlimited synthetic storage, and full momentum transfer.

Reference arithmetic:

- impulse: `20,000 N*s`;
- kinetic energy removed: `2,000,000 J`;
- average force: `2,000 N`;
- constant-deceleration distance: `1,000 m`;
- average mechanical power: `200,000 W`;
- regenerable mechanical energy: `1,600,000 J`;
- stored electrical energy: `1,440,000 J`;
- residual energy: `560,000 J`;
- catcher delta-v: `2 m/s`;
- stationkeeping impulse to restore: `20,000 N*s`.

Additional executed fixtures verify storage-cap clipping, monotonic reduction of residual energy with higher regeneration efficiency, explicit external momentum sinks, equal/opposite traffic net cancellation, and fail-closed attempts to use the capture model as an accelerator.

## Evidence boundary

The model is scalar/1-D and first-order. It does not model:

- structural stress;
- electromagnetic field geometry;
- contact dynamics;
- catcher attitude;
- 3-D orbital momentum vectors;
- stationkeeping propulsion performance;
- thermal rejection hardware;
- power-electronics peak limits;
- storage charge-rate limits;
- capture probability or dispersion;
- catcher availability/downtime;
- safe-miss consequences.

No numeric fixture is a proposed lunar catcher specification.

## Promotion path

1. production implementation agrees with this oracle;
2. extend momentum to 3-D vectors and orbital frames;
3. connect to LL-003/004 arrival states and LL-005 dispersion;
4. add explicit thermal/storage-rate limits;
5. add stationkeeping propulsion model;
6. compare passive, active, regenerative, catcher-tug, LLO-fleet, and L2 architectures;
7. feed lifecycle energy/propellant/mass/reliability into LETN Phase-0 Pareto studies.

Tracks #1575.
