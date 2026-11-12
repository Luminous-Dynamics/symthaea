# symthaea-circuits

Electrical-circuit analysis for Symthaea, completing basic engineering alongside
`symthaea-structural` (mechanical statics). The workspace had no electrical work.

Pure `std`, zero dependencies, no `symthaea-core` link. All results closed-form,
checked against textbook values.

## Capabilities

| Area | API |
|------|-----|
| DC | `dc::{current, voltage, power, series_resistance, parallel_resistance, voltage_divider}` |
| Transients | `transient::{rc_time_constant, rc_charging_voltage, rc_discharging_voltage, rl_time_constant}` |
| DC-bus stability | `dc_bus_stability::analyze` — reduced Thevenin / constant-power-load equilibrium and phase-line reference; near-fold cases fail closed as numerically indeterminate |
| AC | `ac::{capacitive_reactance, inductive_reactance, resonant_frequency, series_rlc_impedance}` |

## Example

```rust
use symthaea_circuits::{dc, ac};
assert!((dc::current(12.0, 4.0) - 3.0).abs() < 1e-12);               // 3 A
assert!((ac::resonant_frequency(1e-3, 1e-6) - 5032.92).abs() < 0.1); // Hz
```

```bash
cargo test -p symthaea-circuits
```


## DC-bus stability research reference

`dc_bus_stability` implements a deliberately narrow, one-state model for a
Thevenin source, lumped DC-link capacitor and constant-power load. It reports
equilibrium roots, local eigenvalue signs, a dimensionless fold-distance
diagnostic and initial-voltage basin classification.

This is a synthetic analytical reference only. The binary64 ambiguity guard is
not formal interval arithmetic. The module does not model converter control
loops, line inductance, battery electrochemistry, switching, protection,
aircraft dynamics or certification.

The frozen contract and machine-readable known-answer vectors are tracked in
[ AIRROAD-DC-DYNAMIC-STABILITY-001 #11712 ](https://github.com/Luminous-Dynamics/symthaea/pull/11712).
