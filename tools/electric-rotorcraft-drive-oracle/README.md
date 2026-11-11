# Electric Rotorcraft Drive-Chain Reference Oracle

A dependency-free Rust analytical reference for one positive-power drive path:

rotor torque and RPM -> transmission -> motor -> inverter -> DC bus.

The rotor-side mechanical power is torque multiplied by angular velocity. The oracle then applies explicitly supplied transmission, motor and inverter efficiencies to compute the required power at each boundary and the resulting DC-bus current at an explicit operating voltage.

## Important boundaries

- No default or guessed efficiency values are provided.
- Direct drive must explicitly use transmission efficiency = 1.0.
- Each component conversion loss is reported separately.
- Loss energy is not assigned to a battery or thermal node by this crate; the physical thermal architecture must own that allocation.
- DC bus voltage is an explicit operating-point input. Battery OCV, SOC, SOH, internal resistance, voltage sag and battery current limits remain separate battery-model responsibilities.
- This v0.1 crate is motoring-only. Regeneration and reverse power flow are outside the model.
- The operating-point efficiencies must already be valid for the torque, RPM, temperature and configuration being evaluated. This crate does not create, interpolate or extrapolate motor/inverter maps.

This is not a motor electromagnetic model, inverter switching model, battery model, rotor aerodynamic model, aircraft solver, BMS or protection controller. Synthetic reference PASS is not a physical qualification or airworthiness result.

## Exact-path verification

cargo fmt --manifest-path tools/electric-rotorcraft-drive-oracle/Cargo.toml -- --check

cargo test --locked --manifest-path tools/electric-rotorcraft-drive-oracle/Cargo.toml

cargo clippy --locked --manifest-path tools/electric-rotorcraft-drive-oracle/Cargo.toml --all-targets -- -D warnings

All numerical cases are synthetic analytical fixtures. No physical motor, inverter, gearbox or rotor data is embedded.


## Gearbox-aware motor operating point (additive kinematic extension)

The v0.1 scalar report remains unchanged. A separate API,
`evaluate_drive_with_gear_ratio`, additionally calculates the motor-side
speed and torque for an explicitly supplied positive motoring speed ratio:

`motor_to_rotor_speed_ratio = motor_speed / rotor_speed`

For a reduction gearbox whose motor turns faster than the rotor, the ratio is
greater than one. The function derives motor RPM, motor angular velocity and
motor shaft torque from the already-accounted motor mechanical output power.
The caller must supply the ratio; no direct-drive or gearbox default is used.

A direct-drive configuration explicitly supplies both:
- `motor_to_rotor_speed_ratio = 1.0`
- `transmission_efficiency = 1.0`

This is only a kinematic operating-point calculation. It does not establish
that the motor can produce the requested torque at that speed, does not check
field weakening or map limits, and does not add motor/inverter efficiency maps.
It also does not support reverse operation or regeneration.

The additive reference tests include a synthetic 10 kW rotor point at
3,000 RPM with a 5:1 motor-to-rotor speed ratio. All expected values are
analytical fixtures, not measured hardware data.

The function and its tests are introduced in the stacked development branch
for AIRROAD-DRIVE-KINEMATICS-001. Exact-head formatting, test and Clippy
results must be recorded independently; source inspection is not a test pass.
