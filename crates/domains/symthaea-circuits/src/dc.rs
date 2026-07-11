// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! DC circuit analysis: Ohm's law, power, series/parallel resistance, dividers.

/// Ohm's law: current `I = V/R` (A).
pub fn current(voltage: f64, resistance: f64) -> f64 {
    voltage / resistance
}

/// Ohm's law: voltage `V = I·R` (V).
pub fn voltage(current: f64, resistance: f64) -> f64 {
    current * resistance
}

/// Dissipated power `P = V·I` (W).
pub fn power(voltage: f64, current: f64) -> f64 {
    voltage * current
}

/// Dissipated power from current and resistance: `P = I²·R` (W).
pub fn power_ir(current: f64, resistance: f64) -> f64 {
    current * current * resistance
}

/// Total resistance of resistors in series (Ω).
pub fn series_resistance(resistors: &[f64]) -> f64 {
    resistors.iter().sum()
}

/// Total resistance of resistors in parallel (Ω); 0 if any resistor is 0.
pub fn parallel_resistance(resistors: &[f64]) -> f64 {
    let mut recip = 0.0;
    for &r in resistors {
        if r == 0.0 {
            return 0.0;
        }
        recip += 1.0 / r;
    }
    if recip == 0.0 { 0.0 } else { 1.0 / recip }
}

/// Output of a two-resistor voltage divider: `Vout = Vin·R2/(R1+R2)`.
pub fn voltage_divider(v_in: f64, r1: f64, r2: f64) -> f64 {
    v_in * r2 / (r1 + r2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ohms_law() {
        assert!((current(12.0, 4.0) - 3.0).abs() < 1e-12);
        assert!((voltage(3.0, 4.0) - 12.0).abs() < 1e-12);
        assert!((power(12.0, 3.0) - 36.0).abs() < 1e-12);
        assert!((power_ir(3.0, 4.0) - 36.0).abs() < 1e-12);
    }

    #[test]
    fn series_and_parallel() {
        assert!((series_resistance(&[100.0, 220.0, 330.0]) - 650.0).abs() < 1e-9);
        assert!((parallel_resistance(&[10.0, 10.0]) - 5.0).abs() < 1e-12);
        // Parallel is always ≤ the smallest resistor.
        assert!(parallel_resistance(&[100.0, 200.0, 300.0]) < 100.0);
    }

    #[test]
    fn divider_halves_at_equal_resistors() {
        assert!((voltage_divider(10.0, 1000.0, 1000.0) - 5.0).abs() < 1e-12);
    }
}
