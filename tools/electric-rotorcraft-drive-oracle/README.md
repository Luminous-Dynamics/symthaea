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
- This v0.2 crate is motoring-only. Regeneration and reverse power flow are outside the model.
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


## Map-qualified efficiency envelope (v0.3)

The v0.3 additive API connects a caller-supplied, already-qualified motor efficiency interval to the independent drive-chain calculation without introducing a dependency on a map library.

1. `derive_motor_shaft_operating_point` computes motor torque and speed from rotor demand, the explicit gearbox speed ratio and transmission efficiency. It does not require motor or inverter efficiency, so it can be used before a motor-map query.
2. The caller constructs `MotorEfficiencyEvidence` from the map lookup output, including map ID, configuration digest, context digest, evidence-manifest digest, exact motor torque/RPM query and lower/nominal/upper efficiency bounds.
3. `evaluate_drive_with_map_evidence` verifies the expected configuration/context, requires the map query point to match the current derived motor shaft point exactly, and independently evaluates all three efficiency values.

The result contains:
- **Adverse efficiency scenario:** lower motor efficiency, therefore highest DC-bus demand.
- **Nominal efficiency scenario:** nominal map value.
- **Favourable efficiency scenario:** upper motor efficiency, therefore lowest DC-bus demand.

Each scenario retains its own `Result<DriveReport, DriveError>`. A nominal PASS cannot hide an adverse-case bus power/current failure. The result carries map, configuration, context and evidence-manifest identities unchanged.

The efficiency interval is only as meaningful as the upstream map/evidence process. The current map oracle's weighted absolute bounds are not Gaussian confidence intervals. No joint uncertainty or correlation is inferred for transmission, inverter, battery, thermal system or rotor aerodynamics. These must be evaluated by their respective owners.

All fixtures remain synthetic analytical cases; no physical motor efficiency map is embedded. The envelope does not establish a continuous rating, motor limit, safe operating point, aircraft mission capability or airworthiness.
