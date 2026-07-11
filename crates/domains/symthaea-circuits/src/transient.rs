// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! First-order RC / RL transient response.

/// RC time constant `τ = R·C` (s).
pub fn rc_time_constant(resistance: f64, capacitance: f64) -> f64 {
    resistance * capacitance
}

/// Capacitor voltage while charging toward `v_source`:
/// `V(t) = V_source·(1 − e^(−t/RC))`.
pub fn rc_charging_voltage(v_source: f64, t: f64, resistance: f64, capacitance: f64) -> f64 {
    let tau = rc_time_constant(resistance, capacitance);
    v_source * (1.0 - (-t / tau).exp())
}

/// Capacitor voltage while discharging from `v_initial`:
/// `V(t) = V_initial·e^(−t/RC)`.
pub fn rc_discharging_voltage(v_initial: f64, t: f64, resistance: f64, capacitance: f64) -> f64 {
    let tau = rc_time_constant(resistance, capacitance);
    v_initial * (-t / tau).exp()
}

/// RL time constant `τ = L/R` (s).
pub fn rl_time_constant(inductance: f64, resistance: f64) -> f64 {
    inductance / resistance
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn time_constant() {
        assert!((rc_time_constant(1000.0, 1e-6) - 1e-3).abs() < 1e-15); // 1 ms
    }

    #[test]
    fn charges_to_63_percent_in_one_tau() {
        let tau = rc_time_constant(1000.0, 1e-6);
        let v = rc_charging_voltage(5.0, tau, 1000.0, 1e-6);
        assert!((v - 5.0 * 0.632120).abs() < 1e-4, "v={v}");
    }

    #[test]
    fn charge_and_discharge_sum_to_source() {
        // At any t, charging(V) + discharging(V) = V (symmetry of e^-x, 1-e^-x).
        let (r, c, t) = (1000.0, 1e-6, 0.7e-3);
        let up = rc_charging_voltage(5.0, t, r, c);
        let down = rc_discharging_voltage(5.0, t, r, c);
        assert!((up + down - 5.0).abs() < 1e-9);
    }

    #[test]
    fn rl_constant() {
        assert!((rl_time_constant(2.0, 1000.0) - 2e-3).abs() < 1e-15);
    }
}
