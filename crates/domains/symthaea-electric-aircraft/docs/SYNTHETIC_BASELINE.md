# Synthetic mission baseline v0.1

**Evidence class:** synthetic software regression fixture.  
**Model:** `electric-fixed-wing-energy-v0.1.0`.  
**Status:** independently recalculated from the stated equations; the Rust test suite has not yet been executed in a Rust toolchain.

This fixture is intended to detect arithmetic regressions. It is not a real aircraft configuration, range prediction, performance claim, or sizing recommendation. None of these values should be compared with competitor aircraft as if they described a Luminous design.

## Input configuration

| Parameter | Value |
|---|---:|
| Empty mass (battery excluded) | 350 kg |
| Payload | 50 kg |
| Battery mass | 240 kg |
| Calculated all-up mass | 640 kg |
| MTOW limit used in fixture | 700 kg |
| Pack nominal energy | 40,000 Wh |
| Pack mass-specific energy (synthetic) | 166.7 Wh/kg |
| Nominal open-circuit voltage | 400 V |
| Pack DC resistance | 0.04 Ω |
| Continuous power ceiling | 75 kW |
| Continuous current ceiling | 250 A |
| Initial SOC | 95% |
| State of health | 95% |
| Protected reserve SOC | 20% |
| Motor / inverter / propeller efficiencies | 94% / 97% / 82% |
| Screening uncertainty margin | 20% |

The 166.7 Wh/kg fixture value is deliberately just a test input. It is not a proposed battery target or Luminous product specification.

## Mission segments

| Segment | Duration | True airspeed | Supplied L/D | Climb rate | Auxiliary load |
|---|---:|---:|---:|---:|---:|
| Climb | 120 s | 35 m/s | 12 | +3 m/s | 1,000 W |
| Cruise | 1,200 s | 55 m/s | 15 | 0 m/s | 800 W |
| Descent | 300 s | 40 m/s | 12 | -2 m/s | 700 W |

The model credits no regenerative energy during descent.

## Independently recalculated outputs

| Output | Result |
|---|---:|
| Terminal mission energy | 14.605 kWh |
| Equivalent chemical energy draw | 14.729 kWh |
| Energy requirement with 20% screening margin | 17.675 kWh |
| Energy available above protected reserve | 28.500 kWh |
| Calculated energy margin | 10.825 kWh |
| Final SOC | 56.24% |
| Peak battery power | 50.667 kW |
| Peak battery current | 128.313 A |
| Minimum terminal voltage | 394.867 V |
| Total equivalent resistive heat | 0.124 kWh |
| Final lumped pack temperature | 295.06 K |

The energy accounting identity for this equivalent-circuit fixture is:

`chemical energy = terminal energy + I²R heat`

within floating-point rounding.

## Sensitivity checks

All values below are independently recalculated from this same synthetic baseline, changing one input at a time.

| Change | Energy margin |
|---|---:|
| Baseline | 10.825 kWh |
| All segment airspeeds +10% | 9.188 kWh |
| Battery SOH reduced to 85% | 7.825 kWh |
| Protected reserve increased to 30% SOC | 7.025 kWh |
| Motor efficiency reduced to 90% | 10.052 kWh |

These are directional screening checks, not uncertainty intervals or validated sensitivity of a real airframe. The Rust tests include corresponding monotonicity assertions.

## Industry context — not a like-for-like comparison

A November 2026 IEEE Spectrum report describes BETA's ALIA CX300 pack architecture as five packs of roughly 45 kWh and 255 kg each. The derived rated pack-level specific energy is approximately:

`225,000 Wh / 1,275 kg ≈ 176.5 Wh/kg`

That is a reported competitor architecture and derived ratio, not an independent cell test, not the usable energy fraction, and not a direct design comparison with this synthetic 40 kWh fixture. BETA also reports a 75 kg / 640 kW H500A motor and 97.5% internal-test efficiency; those values have different evidence status from independently audited, aircraft-level mission efficiency.

The correct next comparison requires the same mission, payload, reserve policy, usable pack energy, ambient conditions, state of health, and measurement boundary. It must include pack, BMS, thermal management, cooling, wiring, inverter, motor, propeller, structural integration, and charging losses.

## Reproduction status

The Python calculation used to independently check the arithmetic is separate from the Rust source. It does not compile or execute the Rust crate. A Rust-enabled run must still perform:

```sh
cargo fmt --manifest-path crates/domains/symthaea-electric-aircraft/Cargo.toml -- --check
cargo test --locked -p symthaea-electric-aircraft
cargo clippy --locked -p symthaea-electric-aircraft --all-targets -- -D warnings
```

Any disagreement between the Rust implementation and this fixture should fail review until explained and independently resolved.
