# AIRROAD pinned propulsion integration oracle

This standalone Rust crate is a **synthetic software-composition fixture**. It
pins and composes four independently owned Symthaea reference crates:

| Role | Package | Immutable source revision |
|---|---|---|
| Calibrated synchronized observations | `electric-propulsion-observation-oracle` 0.3.0 | `d81f5b4f1ee050d677c3bae1c7733d5bfe73fa20` |
| Strict coordinate-aware context-bound motor map | `electric-propulsion-map-oracle` 0.3.0 | `b99e8cb896cbeec34698cb71e5807b033fb77c07` |
| Gearbox-aware and motor-shaft coordinate drive envelope | `electric-rotorcraft-drive-oracle` 0.6.0 | `c8441438f0e00cceeb1e951ea3c4d88a9aaf3b80` |
| Electrical and thermal loss ledgers | `electric-thermal-loss-ledger-oracle` 0.2.0 | `2865e86a755e92a99bc534a5ea6dd13c272d8dc9` |

Dependencies are pinned by Git commit in `Cargo.toml` and repeated in
`Cargo.lock`. The harness deliberately does not depend on branch names or
require those source PRs to be merged.

## Frozen one-substep fixture

- Rotor-shaft power: 10 kW
- Rotor speed: 3,000 RPM
- Motor-to-rotor speed ratio: 5:1
- Transmission efficiency: 90%
- Inverter efficiency: 98%
- DC bus: 400 V
- Duration: 12 s
- Motor efficiency: synthetic nominal map value 95.75%; the metrology-aware
  coordinate envelope is approximately 95.2966%–96.2037%

Synthetic torque and RPM observations are generated from the drive oracle's
motor-shaft kinematics, passed through the observation synchronizer, and
projected into a motor-shaft operating point. The map query uses that exact
observed point. The map lookup is explicitly adapted into the drive oracle's
map-evidence contract; map identity, cell, weights, configuration, context,
and source-manifest identity are retained.

The v0.3 integration propagates the complete rectangular torque/RPM
metrology set through every intersected bilinear map cell, including rectangles
that cross a grid line. It then passes the coordinate and efficiency bounds to
the motor-shaft demand consumer to calculate minimum, nominal and upper DC-bus
scenarios. Any partial excursion outside map support remains OOD.

The drive-demand input is explicitly at the motor-shaft output boundary.
Gearbox ratio and transmission efficiency are not applied a second time.
The upper demand is used in the synthetic propulsion electrical-load ledger;
the transmission loss, motor loss and inverter loss are separately represented
in the thermal-source ledger for that same upper-bound scenario.

The minimum, nominal, and upper DC-demand cases are independently evaluated.
The upper-bound case feeds a branch-meter ledger containing propulsion,
thermal-management, auxiliary, and distribution loads. A separate loss ledger
allocates transmission, motor, and inverter losses for that same upper-bound
scenario to named thermal nodes while preserving intentionally unallocated
fractions. TMS electrical demand is not
silently reclassified as battery heat, and conversion losses are not added to
electrical demand twice.

## Qualification boundary

Passing these tests demonstrates only that these pinned Rust interfaces compose
for this deterministic synthetic one-substep fixture and its negative controls.
It does **not** validate physical measurements, sensor calibration, a real motor
map, the battery/pack, thermal transfer, a complete aircraft mission, acoustic
performance, safety, airworthiness, or certification.

No physical experiment is authorized by this crate. Measurement proposals remain
subject to the applicable test procedure, facility controls, safety review, and
authorized engineer approval.

## Run

```sh
cargo fmt --manifest-path tools/airroad-propulsion-integration-oracle/Cargo.toml --check
cargo test --manifest-path tools/airroad-propulsion-integration-oracle/Cargo.toml --locked
cargo clippy --manifest-path tools/airroad-propulsion-integration-oracle/Cargo.toml --all-targets --locked -- -D warnings
```


## Coordinate-aware metrology-to-bus envelope (v0.3 draft)

This follow-on composes the coordinate-bound map and motor-shaft demand APIs:

1. Calibrated, synchronized motor-shaft torque/RPM observations produce a nominal operating point plus absolute coordinate bounds.
2. The map oracle enumerates every closed bilinear cell intersecting that full rectangle, rejects any partial out-of-domain rectangle, and returns the global efficiency bounds plus cell/witness provenance. Unlike the earlier same-cell four-corner adapter, a rectangle crossing a torque or speed grid line is now supported by explicit per-cell corner enumeration.
3. The drive oracle consumes the same nominal point and coordinate bounds at the **motor-shaft output boundary**, propagating them with the map efficiency envelope to minimum/nominal/upper DC-bus demand. It does not apply gearbox efficiency again; the rotor-side to motor-side conversion and transmission loss remain separately accounted from the gearbox kinematics.
4. The electrical ledger uses the upper DC-demand scenario for the conservative synthetic propulsion load; transmission, motor and inverter heat sources remain separately identified in the heat-allocation ledger.

The lower/upper combinations use rectangular absolute bounds, not a probabilistic distribution. They may be conservative where torque, speed and efficiency co-vary. The current APIs use f64 interpolation and interval endpoint arithmetic; they are not yet directed-rounded formal floating-point enclosures. No physical validation is implied.

The exact Git revisions for the coordinate-aware map and drive APIs are pinned in Cargo.toml and Cargo.lock. The fixture remains synthetic and single-substep; it is not a complete mission electro-thermal validation.
