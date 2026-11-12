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
| DC-bus stability | `dc_bus_stability::analyze` — reduced Thevenin / constant-power-load equilibrium and phase-line reference |
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
[AIRROAD-DC-DYNAMIC-STABILITY-001 #11712](https://github.com/Luminous-Dynamics/symthaea/pull/11712).


## Distributed-electric propulsion DC-bus interaction reference

The `dc_bus_interaction` module contains a reduced-order two-branch model:
one finite-impedance Thevenin source, a common DC-link capacitor, two resistive feeders, one input capacitor per branch, and two constant-power loads. It provides:

- a KCL-checked operating-point witness;
- the three-state small-signal Jacobian and cubic characteristic coefficients;
- isolated branch poles and a conservative Routh-Hurwitz disposition;
- frequency-domain branch admittances, source impedance and a common-mode return ratio.

This is a synthetic RC/CPL reference, not a converter-control model, measured impedance fit, switching simulation, aircraft stability proof, or flight-safety authority. The companion exact-rational case and independent verifier are under `docs/research/AIRROAD_PDCM_IMPEDANCE_001_V0_1.md` and `tools/airroad-pdcm-impedance-qualifier/`.


### Numerical-stability disposition

The cubic Routh-Hurwitz test normalizes its coefficients by a characteristic pole-rate before applying the floating-point guard. The reported `normalized_routh_hurwitz_delta` is dimensionless and is invariant to uniform rescaling of all capacitances (a uniform time-scale change). It remains a numerical screening result for the declared model, not a robust uncertainty bound or physical stability guarantee.


## Two-port MIMO hidden-dynamics reference

The `dc_bus_mimo_reference` module provides a narrow exact 2×2 cross-port return-ratio example. It retains both off-diagonal terms, computes `det(I + L(jω))`, carries active open-loop right-half-plane channel-pole metadata, and derives the cleared characteristic polynomial when both cross channels are present.

It demonstrates that diagonal-only SISO projections can omit a coupled unstable mode, and that determinant cancellation must not erase an active realization pole. This is a synthetic benchmark/reference, not a generalized-Nyquist solver or physical converter/aircraft model. See `docs/research/AIRROAD_PDCM_MIMO_001_V0_1.md`.


## MIMO impedance matrix evidence passport

The `mimo_impedance_passport` module validates complete multi-port impedance sweeps. Each matrix column is one injected-current port; every output voltage port must be observed at every exact frequency knot. The sweep binds configuration, operating-point and sweep digests, explicit port voltage/current sign conventions, column excitation/payload digests, uncertainty bounds, and measured/simulated/analytical evidence origin.

Measured sweeps require instrument, calibration-certificate, channel-map and synchronization digests. Matrix lookup is exact-knot only; interpolation and extrapolation are not authorized. Missing cross-port responses are errors, never implicit zeros. Derived impedances are point estimates; supplied voltage/current error bounds are retained but not propagated into a formal complex interval enclosure.

See `docs/research/AIRROAD_PDCM_MIMO_PASSPORT_001_V0_1.md`.
