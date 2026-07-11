// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! P-f and Q-V droop control, the standard grid-forming inverter control law
//! for islanded/microgrid operation (Chandorkar, Divan & Adapa, "Control of
//! parallel connected inverters in standalone AC supply systems", IEEE
//! Trans. Industry Applications, 1993 — the foundational inverter-droop
//! paper; the underlying principle traces back to synchronous-generator
//! governor droop).
//!
//! Droop control's defining property, and the one this module's tests
//! validate rather than assume: with no communication between sources, two
//! droop-controlled sources sharing a common bus automatically split load
//! changes in inverse proportion to their droop gains.

use serde::{Deserialize, Serialize};

/// Real-power/frequency droop: `f = f0 - kp * (P - P0)`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FrequencyDroop {
    /// Nominal (no-load) frequency, Hz (e.g. 60.0 or 50.0).
    pub nominal_frequency_hz: f64,
    /// Nominal real-power setpoint, kW.
    pub nominal_power_kw: f64,
    /// Droop gain, Hz per kW. Must be > 0 for stable sharing.
    pub droop_hz_per_kw: f64,
}

impl FrequencyDroop {
    /// Frequency this source would settle at while delivering `power_kw`
    /// (grid-forming causality: power determines frequency).
    pub fn frequency_for_power(&self, power_kw: f64) -> f64 {
        self.nominal_frequency_hz - self.droop_hz_per_kw * (power_kw - self.nominal_power_kw)
    }

    /// Power this source would deliver at system frequency `frequency_hz`
    /// (grid-following causality: frequency determines power).
    pub fn power_for_frequency(&self, frequency_hz: f64) -> f64 {
        self.nominal_power_kw - (frequency_hz - self.nominal_frequency_hz) / self.droop_hz_per_kw
    }
}

/// Reactive-power/voltage droop: `V = V0 - kq * (Q - Q0)`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct VoltageDroop {
    /// Nominal (no-load) voltage, volts.
    pub nominal_voltage_v: f64,
    /// Nominal reactive-power setpoint, kVAR.
    pub nominal_power_kvar: f64,
    /// Droop gain, volts per kVAR. Must be > 0 for stable sharing.
    pub droop_v_per_kvar: f64,
}

impl VoltageDroop {
    pub fn voltage_for_reactive_power(&self, reactive_power_kvar: f64) -> f64 {
        self.nominal_voltage_v
            - self.droop_v_per_kvar * (reactive_power_kvar - self.nominal_power_kvar)
    }

    pub fn reactive_power_for_voltage(&self, voltage_v: f64) -> f64 {
        self.nominal_power_kvar - (voltage_v - self.nominal_voltage_v) / self.droop_v_per_kvar
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frequency_droop_at_nominal_power_gives_nominal_frequency() {
        let droop = FrequencyDroop {
            nominal_frequency_hz: 60.0,
            nominal_power_kw: 50.0,
            droop_hz_per_kw: 0.01,
        };
        assert_eq!(droop.frequency_for_power(50.0), 60.0);
    }

    #[test]
    fn test_frequency_droop_falls_as_power_increases() {
        let droop = FrequencyDroop {
            nominal_frequency_hz: 60.0,
            nominal_power_kw: 50.0,
            droop_hz_per_kw: 0.01,
        };
        // +50kW above nominal -> frequency drops by 0.01*50 = 0.5 Hz
        assert!((droop.frequency_for_power(100.0) - 59.5).abs() < 1e-9);
    }

    #[test]
    fn test_power_for_frequency_is_exact_inverse_of_frequency_for_power() {
        let droop = FrequencyDroop {
            nominal_frequency_hz: 60.0,
            nominal_power_kw: 50.0,
            droop_hz_per_kw: 0.01,
        };
        for p in [0.0, 25.0, 50.0, 73.4, 120.0] {
            let f = droop.frequency_for_power(p);
            let round_trip_p = droop.power_for_frequency(f);
            assert!(
                (round_trip_p - p).abs() < 1e-9,
                "p={p} round-tripped to {round_trip_p}"
            );
        }
    }

    #[test]
    fn test_voltage_droop_falls_as_reactive_power_increases() {
        let droop = VoltageDroop {
            nominal_voltage_v: 480.0,
            nominal_power_kvar: 0.0,
            droop_v_per_kvar: 0.5,
        };
        assert!((droop.voltage_for_reactive_power(10.0) - 475.0).abs() < 1e-9);
    }

    /// The defining physical property of droop control: two sources with NO
    /// communication, sharing a common bus frequency, automatically split a
    /// load change in inverse proportion to their droop gains
    /// (ΔP1/ΔP2 = k2/k1). This is what makes droop control useful — it is
    /// not an incidental consequence, it is the entire point of the method.
    #[test]
    fn test_two_sources_share_load_inversely_proportional_to_droop_gain() {
        // ΔP = -Δf/kp for a shared bus frequency deviation Δf, so a SMALLER
        // droop gain kp yields a LARGER |ΔP| -- as kp -> 0 the source holds
        // frequency near-perfectly and must absorb whatever power is needed
        // to do so (an infinitely stiff / slack-bus limit). So source_2
        // below (smaller kp) is the stiffer one and picks up MORE power;
        // source_1 (larger kp) is softer and picks up less.
        let source_1 = FrequencyDroop {
            nominal_frequency_hz: 60.0,
            nominal_power_kw: 50.0,
            droop_hz_per_kw: 0.02, // softer (more Hz drop per kW)
        };
        let source_2 = FrequencyDroop {
            nominal_frequency_hz: 60.0,
            nominal_power_kw: 50.0,
            droop_hz_per_kw: 0.01, // stiffer (half the droop gain of source_1)
        };

        // Both sources see the same system frequency (that's the physical
        // mechanism — frequency is a single shared quantity on an AC bus).
        // Pick an arbitrary common operating frequency below nominal,
        // representing the bus responding to an increased load.
        let common_frequency_hz = 59.4;

        let p1 = source_1.power_for_frequency(common_frequency_hz);
        let p2 = source_2.power_for_frequency(common_frequency_hz);
        let delta_p1 = p1 - source_1.nominal_power_kw;
        let delta_p2 = p2 - source_2.nominal_power_kw;

        // ΔP1 / ΔP2 should equal k2/k1 = 0.01/0.02 = 0.5 — the stiffer
        // source (source_2, smaller droop gain) picks up proportionally
        // MORE power, so source_1's delta is half of source_2's.
        let ratio = delta_p1 / delta_p2;
        let expected_ratio = source_2.droop_hz_per_kw / source_1.droop_hz_per_kw;
        assert!(
            (ratio - expected_ratio).abs() < 1e-9,
            "got ratio {ratio}, expected {expected_ratio}"
        );
        // Concretely: source_2 (stiffer, smaller droop gain) should pick up
        // more delta power than source_1 (softer, larger droop gain).
        assert!(delta_p2 > delta_p1);
    }
}
