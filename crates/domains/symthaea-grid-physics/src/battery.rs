// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Battery energy-storage model in real physical units (kW, kWh).
//!
//! Field semantics follow the reusable schema in
//! `sol-atlas/terra-atlas-mvp/lib/drizzle/schema-energy.ts`'s `batteryStorage`
//! table (power/energy/duration, round-trip efficiency, cycles, degradation),
//! per PLANETARY_ENERGY_COORDINATION_PLAN_2026-07-06.md's reuse map — this is
//! the physics-layer counterpart of that data model, not a reimplementation
//! of it.

use serde::{Deserialize, Serialize};

/// A grid-scale battery energy storage system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Battery {
    /// Usable energy capacity at 100% state of health (kWh).
    pub capacity_kwh: f64,
    /// Maximum charge/discharge power (kW). Symmetric charge/discharge rating.
    pub power_rating_kw: f64,
    /// Round-trip efficiency in \[0, 1\] (AC-to-AC). Applied as sqrt(eta) on
    /// each of charge and discharge, the standard convention so that a full
    /// charge+discharge cycle loses exactly `1 - eta` of the energy moved.
    pub round_trip_efficiency: f64,
    /// Current state of charge, fraction of *current* (degraded) capacity, in \[0, 1\].
    soc: f64,
    /// State of health: fraction of original capacity remaining, in \[0, 1\].
    state_of_health: f64,
    /// Cumulative equivalent full cycles (one full 0->100->0 traversal = 1.0).
    equivalent_full_cycles: f64,
    /// Fractional capacity fade per equivalent full cycle (e.g. 0.0002 = 0.02%/cycle).
    pub degradation_per_cycle: f64,
}

/// Errors from an attempted battery operation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BatteryError {
    /// Requested power exceeds `power_rating_kw`.
    PowerExceedsRating,
    /// `dt_hours` was not finite/positive.
    InvalidDuration,
}

impl Battery {
    /// Construct a new battery at the given state of charge (fraction of
    /// nameplate capacity, \[0, 1\]), starting at 100% state of health.
    pub fn new(capacity_kwh: f64, power_rating_kw: f64, round_trip_efficiency: f64) -> Self {
        Self {
            capacity_kwh,
            power_rating_kw,
            round_trip_efficiency: round_trip_efficiency.clamp(0.0, 1.0),
            soc: 0.5,
            state_of_health: 1.0,
            equivalent_full_cycles: 0.0,
            degradation_per_cycle: 0.0002,
        }
    }

    pub fn with_soc(mut self, soc: f64) -> Self {
        self.soc = soc.clamp(0.0, 1.0);
        self
    }

    pub fn with_degradation_per_cycle(mut self, rate: f64) -> Self {
        self.degradation_per_cycle = rate.max(0.0);
        self
    }

    /// Current state of charge, fraction of *current* (degraded) capacity.
    pub fn soc(&self) -> f64 {
        self.soc
    }

    /// Fraction of original nameplate capacity remaining.
    pub fn state_of_health(&self) -> f64 {
        self.state_of_health
    }

    /// Cumulative equivalent full cycles.
    pub fn equivalent_full_cycles(&self) -> f64 {
        self.equivalent_full_cycles
    }

    /// Usable energy capacity at current state of health (kWh).
    pub fn effective_capacity_kwh(&self) -> f64 {
        self.capacity_kwh * self.state_of_health
    }

    /// Energy currently stored (kWh), at current state of health.
    pub fn stored_energy_kwh(&self) -> f64 {
        self.soc * self.effective_capacity_kwh()
    }

    /// One-way (charge or discharge) efficiency: sqrt of round-trip efficiency,
    /// the standard convention so a full charge+discharge cycle loses exactly
    /// `1 - round_trip_efficiency` of the energy moved.
    fn one_way_efficiency(&self) -> f64 {
        self.round_trip_efficiency.sqrt()
    }

    /// Charge the battery at `power_kw` (AC-side, before efficiency losses)
    /// for `dt_hours`. Returns the energy actually accepted (kWh, DC-side,
    /// after efficiency), which may be less than requested if the battery
    /// reaches full SoC mid-interval.
    pub fn charge(&mut self, power_kw: f64, dt_hours: f64) -> Result<f64, BatteryError> {
        if power_kw.abs() > self.power_rating_kw + f64::EPSILON {
            return Err(BatteryError::PowerExceedsRating);
        }
        if !dt_hours.is_finite() || dt_hours < 0.0 {
            return Err(BatteryError::InvalidDuration);
        }
        let requested_ac_kwh = power_kw.max(0.0) * dt_hours;
        let requested_dc_kwh = requested_ac_kwh * self.one_way_efficiency();
        let headroom_kwh = (1.0 - self.soc) * self.effective_capacity_kwh();
        let accepted_dc_kwh = requested_dc_kwh.min(headroom_kwh);

        if self.effective_capacity_kwh() > 0.0 {
            self.soc += accepted_dc_kwh / self.effective_capacity_kwh();
            self.soc = self.soc.clamp(0.0, 1.0);
        }
        self.accumulate_cycles(accepted_dc_kwh);
        Ok(accepted_dc_kwh)
    }

    /// Discharge the battery at `power_kw` (AC-side, after efficiency losses,
    /// i.e. what's delivered to the grid) for `dt_hours`. Returns the AC
    /// energy actually delivered (kWh), which may be less than requested if
    /// the battery reaches empty mid-interval.
    pub fn discharge(&mut self, power_kw: f64, dt_hours: f64) -> Result<f64, BatteryError> {
        if power_kw.abs() > self.power_rating_kw + f64::EPSILON {
            return Err(BatteryError::PowerExceedsRating);
        }
        if !dt_hours.is_finite() || dt_hours < 0.0 {
            return Err(BatteryError::InvalidDuration);
        }
        let requested_ac_kwh = power_kw.max(0.0) * dt_hours;
        let one_way_eff = self.one_way_efficiency();
        let requested_dc_kwh = if one_way_eff > 0.0 {
            requested_ac_kwh / one_way_eff
        } else {
            0.0
        };
        let available_dc_kwh = self.stored_energy_kwh();
        let delivered_dc_kwh = requested_dc_kwh.min(available_dc_kwh);
        let delivered_ac_kwh = delivered_dc_kwh * one_way_eff;

        if self.effective_capacity_kwh() > 0.0 {
            self.soc -= delivered_dc_kwh / self.effective_capacity_kwh();
            self.soc = self.soc.clamp(0.0, 1.0);
        }
        self.accumulate_cycles(delivered_dc_kwh);
        Ok(delivered_ac_kwh)
    }

    /// Accumulate equivalent-full-cycle count and apply capacity fade.
    /// `energy_moved_kwh` is DC-side energy moved in or out (always >= 0).
    fn accumulate_cycles(&mut self, energy_moved_kwh: f64) {
        if self.capacity_kwh <= 0.0 {
            return;
        }
        // A full cycle moves `2 * capacity_kwh` (charge + discharge, at
        // nameplate). Half that per charge-only or discharge-only event.
        let cycle_fraction = energy_moved_kwh / (2.0 * self.capacity_kwh);
        self.equivalent_full_cycles += cycle_fraction;
        self.state_of_health =
            (1.0 - self.equivalent_full_cycles * self.degradation_per_cycle).clamp(0.0, 1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_charge_moves_expected_energy_at_full_efficiency() {
        let mut b = Battery::new(100.0, 50.0, 1.0).with_soc(0.0);
        let accepted = b.charge(50.0, 1.0).unwrap();
        assert!((accepted - 50.0).abs() < 1e-9);
        assert!((b.soc() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_charge_respects_efficiency_loss() {
        // round_trip_efficiency = 0.81 -> one-way efficiency = 0.9
        let mut b = Battery::new(100.0, 50.0, 0.81).with_soc(0.0);
        let accepted = b.charge(50.0, 1.0).unwrap(); // 50 AC kWh requested
        assert!((accepted - 45.0).abs() < 1e-9, "got {accepted}");
        assert!((b.soc() - 0.45).abs() < 1e-9);
    }

    #[test]
    fn test_full_cycle_loses_exactly_one_minus_round_trip_efficiency() {
        // Charge with headroom to spare (100kWh capacity, only 50 AC kWh
        // requested) so acceptance is governed by efficiency alone, not the
        // headroom cap -- isolates the round-trip-efficiency property this
        // test is about from the separate capping behavior already covered
        // by test_charge_caps_at_headroom. Degradation is zeroed for the
        // same reason: it's real behavior (see test_degradation_accumulates_with_cycles)
        // but shrinks effective_capacity_kwh between the charge and discharge
        // calls by enough to blow the 1e-6 round-trip tolerance below.
        let mut b = Battery::new(100.0, 1000.0, 0.81)
            .with_soc(0.0)
            .with_degradation_per_cycle(0.0);
        let ac_kwh_in = 50.0;
        let charged_dc = b.charge(500.0, 0.1).unwrap(); // 500kW * 0.1h = 50 AC kWh requested
        assert!((charged_dc - 45.0).abs() < 1e-9, "got {charged_dc}"); // 50 * sqrt(0.81)
        // Discharge with plenty of requested power/duration to fully drain
        // the 45kWh DC stored, so delivery is governed by availability, not
        // by the requested-power cap.
        let delivered_ac = b.discharge(500.0, 1.0).unwrap();
        // Round trip: one-way eff = 0.9 both ways -> delivered = charged_dc * 0.9
        assert!((delivered_ac - charged_dc * 0.9).abs() < 1e-6);
        // Overall round-trip ratio should equal round_trip_efficiency.
        let round_trip_ratio = delivered_ac / ac_kwh_in;
        assert!(
            (round_trip_ratio - 0.81).abs() < 1e-6,
            "got {round_trip_ratio}"
        );
    }

    #[test]
    fn test_charge_caps_at_headroom() {
        let mut b = Battery::new(100.0, 50.0, 1.0).with_soc(0.9);
        let accepted = b.charge(50.0, 1.0).unwrap(); // would be 50, headroom is 10
        assert!((accepted - 10.0).abs() < 1e-9, "got {accepted}");
        assert!((b.soc() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_discharge_caps_at_available_energy() {
        let mut b = Battery::new(100.0, 50.0, 1.0).with_soc(0.1);
        let delivered = b.discharge(50.0, 1.0).unwrap(); // would be 50, only 10 stored
        assert!((delivered - 10.0).abs() < 1e-9, "got {delivered}");
        assert!((b.soc() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_power_exceeding_rating_rejected() {
        let mut b = Battery::new(100.0, 50.0, 1.0);
        assert_eq!(b.charge(51.0, 1.0), Err(BatteryError::PowerExceedsRating));
        assert_eq!(
            b.discharge(51.0, 1.0),
            Err(BatteryError::PowerExceedsRating)
        );
    }

    #[test]
    fn test_negative_duration_rejected() {
        let mut b = Battery::new(100.0, 50.0, 1.0);
        assert_eq!(b.charge(10.0, -1.0), Err(BatteryError::InvalidDuration));
    }

    #[test]
    fn test_degradation_accumulates_with_cycles() {
        let mut b = Battery::new(100.0, 1000.0, 1.0)
            .with_soc(0.0)
            .with_degradation_per_cycle(0.001);
        assert!((b.state_of_health() - 1.0).abs() < 1e-9);
        for _ in 0..10 {
            b.charge(1000.0, 0.1).unwrap();
            b.discharge(1000.0, 0.1).unwrap();
        }
        // 10 full cycles at 0.1% fade/cycle -> ~1% fade
        assert!(b.equivalent_full_cycles() > 9.0);
        assert!(b.state_of_health() < 1.0);
        assert!(b.state_of_health() > 0.98);
    }

    #[test]
    fn test_effective_capacity_shrinks_with_state_of_health() {
        let mut b = Battery::new(100.0, 1000.0, 1.0)
            .with_soc(0.0)
            .with_degradation_per_cycle(0.5); // aggressive, for a fast test
        for _ in 0..3 {
            b.charge(1000.0, 0.1).unwrap();
            b.discharge(1000.0, 0.1).unwrap();
        }
        assert!(b.effective_capacity_kwh() < b.capacity_kwh);
    }

    #[test]
    fn test_zero_efficiency_battery_cannot_discharge() {
        let mut b = Battery::new(100.0, 50.0, 0.0).with_soc(1.0);
        let delivered = b.discharge(10.0, 1.0).unwrap();
        assert_eq!(delivered, 0.0);
    }
}
