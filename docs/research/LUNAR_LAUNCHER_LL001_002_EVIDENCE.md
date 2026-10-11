# Lunar Launcher LL-001 / LL-002 Evidence Note

## Scope

This tranche establishes a small launcher/catcher vocabulary and a transparent
constant-acceleration reference model for Phase-0 trade studies.

It does **not** establish a launch trajectory, electromagnetic implementation,
thermal design, structural design, catcher design, safe-release kernel, or
hardware authority.

## LL-001 semantics

`symthaea-lunar-launcher` intentionally lives above/beside the neutral transport
vocabulary rather than expanding `TransportMode` with every launcher subtype.

The crate distinguishes:

- surface-hop, regional-surface, and surface-to-space launcher roles;
- passive/active L2, distributed low-lunar-orbit, surface capture-rail, and
  catcher-tug architecture families;
- reusable smart-pod properties;
- stable site/frame references;
- evidence class and evidence references;
- explicit safe-miss/contingency references.

`LauncherSite::max_axial_acceleration_m_s2` is the declared **worst-case cargo
acceleration envelope for that site/profile description**, not the machine's
ultimate theoretical acceleration capability. A future operating-profile type
may expose lower requested acceleration without redefining the site hardware.

No type in this tranche can release a payload, energize a launcher, steer a pod,
or actuate a catcher.

## LL-002 equations

The analytic reference case assumes constant longitudinal acceleration and uses:

- `v^2 = v0^2 + 2 a L`;
- `delta_KE = 1/2 m (v^2 - v0^2)`;
- `E_electrical = delta_KE / eta`;
- `t = (v - v0) / a`;
- long-run average electrical power = `E_electrical / launch_interval`.

The result reports minimum idealized track length, acceleration time, force,
payload kinetic-energy gain, electrical input energy, in-section average power,
long-run average power, cadence, and electrical energy per launched kilogram.

These relations deliberately omit:

- electromagnetic field/current/coil geometry;
- switching and power-electronics detail;
- cryogenic/superconducting systems;
- rail/guide structural loads and thermal rejection;
- drag/friction/levitation losses beyond one aggregate efficiency;
- lunar gravity along the launcher track;
- trajectory dynamics after release;
- release-error covariance;
- catcher momentum/energy;
- safe-miss geometry;
- pod terminal propulsion;
- qualification or human transport.

## Independent oracle

`scripts/ll-launcher-analytic-oracle.py` imports no Symthaea code and mirrors
only the published LL-002 equations. Its self-tests cover:

1. a basic zero-initial-speed closed-form case;
2. a non-zero initial-speed case;
3. the invariant that ideal kinetic/electrical energy for a fixed velocity
   change is independent of chosen constant acceleration, while track length
   and in-section power change.

The Rust implementation also rejects malformed/non-finite inputs and refuses an
analytic pod case when requested acceleration exceeds the pod's declared axial
limit.

## Evidence state

All launcher/catcher numeric examples in unit tests are **reference arithmetic**,
not proposed lunar hardware specifications.

Promotion requires:

1. Rust compile/test execution;
2. independent oracle agreement;
3. LL-003 surface-ballistic validation;
4. LETN-002-backed cislunar trajectory validation for orbital use;
5. LL-005 uncertainty/dispersion work;
6. LL-007 catcher momentum/energy closure;
7. LL-008 deterministic safe-release policy;
8. common network/economic comparison under LETN Phase-0 assumptions.
