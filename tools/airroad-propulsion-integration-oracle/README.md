# AIRROAD pinned propulsion integration oracle

This standalone Rust crate is a **synthetic software-composition fixture**. It
pins and composes four independently owned Symthaea reference crates:

| Role | Package | Immutable source revision |
|---|---|---|
| Calibrated synchronized observations | `electric-propulsion-observation-oracle` 0.3.0 | `d81f5b4f1ee050d677c3bae1c7733d5bfe73fa20` |
| Strict context-bound motor map | `electric-propulsion-map-oracle` 0.2.0 | `45d6060cc6de03cc2511060b7c9f36a583c00fec` |
| Gearbox-aware drive/map envelope | `electric-rotorcraft-drive-oracle` 0.5.0 | `745d1776cd17b3af3007a38d80f5022077e4ad12` |
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
- Motor efficiency: synthetic map value 95%, with 0.5 percentage-point
  absolute uncertainty (0.4 points nodal plus 0.1 point systematic)

Synthetic torque and RPM observations are generated from the drive oracle's
motor-shaft kinematics, passed through the observation synchronizer, and
projected into a motor-shaft operating point. The map query uses that exact
observed point. The map lookup is explicitly adapted into the drive oracle's
map-evidence contract; map identity, cell, weights, configuration, context,
and source-manifest identity are retained.

The observation's absolute torque/RPM and shaft-power bounds remain available
in the returned observation. **This version does not yet propagate those
coordinate bounds through the motor map and drive model.** Its adverse,
nominal, and favourable DC-demand scenarios vary the map-efficiency bound only.
Do not interpret the resulting envelope as the complete metrology-to-bus
uncertainty envelope.

The adverse, nominal, and favourable efficiency cases are independently
evaluated. The adverse case feeds a branch-meter ledger containing propulsion,
thermal-management, auxiliary, and distribution loads. A separate loss ledger
allocates transmission, motor, and inverter losses to named thermal nodes while
preserving intentionally unallocated fractions. TMS electrical demand is not
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
