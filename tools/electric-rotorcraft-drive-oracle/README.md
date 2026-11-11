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


## Gearbox kinematics (v0.2)

The v0.2 additive API keeps `evaluate_drive` and its rotor-side power contract intact. Use `evaluate_drive_with_kinematics` when the motor-side shaft operating point is needed.

The caller explicitly supplies `TransmissionKinematics { motor_to_rotor_speed_ratio }`, with the positive motoring convention:

`g = motor_speed / rotor_speed`

The function reports motor angular velocity, RPM and shaft output torque. Motor mechanical power remains the transmission-input power already computed by the drive-chain reference, while transmission efficiency remains a separate input in `DriveConfig`.

The speed ratio is not an efficiency. Neither value is inferred. This extension is still a motoring-only analytical reference and does not validate a motor torque-speed map, gearbox physical limits, regeneration, hardware behavior or airworthiness.

### Synthetic known-answer example

For 10 kW at the rotor shaft, 3,000 rotor RPM, a 5:1 motor-to-rotor speed ratio and 90% transmission efficiency:

- Rotor torque: approximately 31.831 N m.
- Motor speed: 15,000 RPM.
- Motor shaft mechanical power: approximately 11,111.111 W.
- Motor shaft output torque: approximately 7.074 N m.

With 95% motor efficiency, 98% inverter efficiency and a 400 V DC bus, the chain requires approximately 11,934.598 W and 29.836 A. These are synthetic analytical values, not measured hardware data.

Additional negative controls reject zero, negative, non-finite and numerically unrepresentable ratios.
