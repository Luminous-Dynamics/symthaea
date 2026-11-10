# Electric Rotorcraft Hover Reference Oracle

A dependency-free Rust reference implementation of the ideal steady-hover actuator-disk equations.

For total rotor thrust T, local air density rho, and declared effective non-overlapping disk area A:

- disk loading = T / A
- induced velocity = sqrt(T / (2 rho A))
- ideal induced power = T * induced velocity

An optional, explicitly supplied hover figure of merit (FM) provides:

- rotor shaft power estimate = ideal induced power / FM

The figure of merit is never inferred or defaulted. Its evidence, applicability and uncertainty remain the caller's responsibility.

## What it is

- A small independent numerical oracle for ideal momentum-theory hover.
- A lower-bound comparison point for more detailed helicopter and multirotor models.
- A deterministic regression fixture for disk-loading and density scaling.
- A std-only crate that can be built independently of the Symthaea workspace.

## What it is not

It does not predict total installed aircraft electrical power or size a passenger aircraft. It omits blade profile power, tail-rotor/anti-torque power, transmission, motor, inverter/ESC and wiring losses, cooling loads, rotor-rotor/rotor-airframe interactions, ground effect, translational lift, wind, climb/descent, vortex-ring state, compressibility, vibration, acoustics, control allocation, failure behavior and transition aerodynamics.

The effective disk area must represent the stated rotor geometry and must not double-count overlapping/coaxial disks. This simple actuator-disk formula does not calculate interference.

An ideal-hover PASS is not vehicle feasibility, battery qualification, structural qualification, safety evidence, airworthiness or certification.

## Exact-path verification

Run cargo fmt --manifest-path tools/electric-rotorcraft-hover-oracle/Cargo.toml -- --check, cargo test --locked --manifest-path tools/electric-rotorcraft-hover-oracle/Cargo.toml, and cargo clippy --locked --manifest-path tools/electric-rotorcraft-hover-oracle/Cargo.toml --all-targets -- -D warnings.

All numerical cases are synthetic analytical fixtures. No physical rotor or aircraft data is embedded.
