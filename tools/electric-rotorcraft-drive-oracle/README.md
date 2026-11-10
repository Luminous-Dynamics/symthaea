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
