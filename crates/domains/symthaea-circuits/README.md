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
