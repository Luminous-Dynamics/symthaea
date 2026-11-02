# symthaea-circuits

Electrical-circuit analysis for Symthaea, completing basic engineering alongside
`symthaea-structural` (mechanics). This crate is pure `std`, has zero runtime
dependencies, and does not link `symthaea-core`.

## Capabilities

| Area | API | Scope |
|---|---|---|
| DC | `dc::{current, voltage, power, series_resistance, parallel_resistance, voltage_divider}` | Ideal resistive networks |
| Transients | `transient::{rc_time_constant, rc_charging_voltage, rc_discharging_voltage, rl_time_constant}` | First-order RC/RL responses |
| AC | `ac::{capacitive_reactance, inductive_reactance, resonant_frequency, series_rlc_impedance}` | Scalar textbook helpers |
| Resonance | `resonance::{SeriesRlc, CoupledSeriesRlc}` | Validated series-RLC metrics, complex impedance, mutual-inductance coupled modes |

## Validated resonance references

`SeriesRlc::new(R, L, C)` rejects non-finite, non-positive or numerically
non-representable inputs. It provides natural resonant frequency, series Q,
ideal half-power bandwidth, and complex steady-state impedance at a positive
frequency.

`CoupledSeriesRlc::new(first, second, k)` uses
`M = k * sqrt(L1 * L2)` and requires `abs(k) < 1`. Its mode-frequency
calculation solves the two-resonator lossless natural-mode equation and uses
scaled arithmetic to reduce avoidable overflow. The sign of mutual inductance
changes the mode shape; the natural-mode frequencies depend on its magnitude.

These are analytical reference models, not a general circuit simulator. They
assume linear, lumped, passive components. They do not model distributed
transmission-line effects, radiation, nonlinear components, core loss,
frequency-dependent material properties or omitted parasitics. Simulation
evidence must be supplied separately by a qualified backend.

## Example

```rust
use symthaea_circuits::resonance::{CoupledSeriesRlc, SeriesRlc};

let primary = SeriesRlc::new(50.0, 1e-3, 1e-6)?;
let secondary = SeriesRlc::new(50.0, 1e-3, 1e-6)?;

let metrics = primary.metrics();
let coupled = CoupledSeriesRlc::new(primary, secondary, 0.25)?;
let modes = coupled.mode_frequencies_hz()?;

# Ok::<(), Box<dyn std::error::Error>>(())
```

## Qualification

The resonance module includes independent known-answer checks for:

- resonance frequency, Q and half-power bandwidth;
- resistive impedance at resonance;
- both half-power points;
- invalid physical/numerical inputs;
- identical-resonator mode splitting;
- recovery of independent resonances at zero coupling;
- coupling-sign invariance of the natural-mode frequencies.

Run locally with:

```bash
cargo test -p symthaea-circuits
```

The benchmark tests establish analytical software behaviour only. They do not
establish physical validation, ngspice qualification, component accuracy or
measured resonator performance.
