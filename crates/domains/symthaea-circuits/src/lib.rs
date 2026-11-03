// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-circuits
//!
//! A self-contained electrical-circuits layer for Symthaea, completing basic
//! engineering alongside `symthaea-structural` (which covers mechanical
//! statics). The workspace had no electrical/circuit analysis.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Closed-form
//! references are checked against textbook/analytical known answers.
//!
//! ## Scope
//!
//! - DC: Ohm's law, power, series/parallel resistance, voltage divider.
//! - Transients: RC/RL time constants, charging/discharging curves.
//! - AC: capacitive/inductive reactance, LC resonance, series-RLC impedance.
//! - Resonance: validated series-RLC metrics and coupled-inductor natural modes.
//!
//! ## Example
//!
//! ```
//! use symthaea_circuits::{dc, ac, resonance::SeriesRlc};
//! assert!((dc::current(12.0, 4.0) - 3.0).abs() < 1e-12);
//! assert!((ac::resonant_frequency(1e-3, 1e-6) - 5032.92).abs() < 0.1);
//! let resonator = SeriesRlc::new(50.0, 1e-3, 1e-6).unwrap();
//! assert!((resonator.metrics().resonant_frequency_hz - 5032.92).abs() < 0.1);
//! ```
//!
//! The resonance module is an analytical reference model, not a circuit
//! simulator or evidence of physical performance. It assumes linear,
//! lumped, passive components and states its validity limits.

pub mod ac;
pub mod dc;
pub mod resonance;
pub mod transient;

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn led_series_resistor_current() {
        // 5 V supply, 2 V LED drop across a 150 Ω resistor → ~20 mA.
        let i = dc::current(5.0 - 2.0, 150.0);
        assert!((i - 0.02).abs() < 1e-9);
    }

    #[test]
    fn rc_filter_reaches_steady_state() {
        // After ~5 time constants the capacitor is essentially fully charged.
        let (r, c) = (1000.0, 1e-6);
        let v = transient::rc_charging_voltage(5.0, 5.0 * dc_tau(r, c), r, c);
        assert!(v > 4.96); // >99.3%
    }

    fn dc_tau(r: f64, c: f64) -> f64 {
        transient::rc_time_constant(r, c)
    }
}
