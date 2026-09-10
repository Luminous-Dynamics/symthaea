// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic electrical-energy accounting for the exoskeleton domain.
//!
//! The power bus is deliberately small: it converts requested actuator mechanical
//! power plus auxiliary loads into electrical draw, enforces a continuous power
//! ceiling and finite stored energy, and returns the fraction of requested actuator
//! work that can actually be supplied. It does not model a fictional shield or any
//! other downstream game technology.

use serde::{Deserialize, Serialize};

/// Grounded aggregate power-system parameters.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PowerBusConfig {
    /// Usable stored electrical energy at full charge, joules.
    pub capacity_j: f64,
    /// Electronics/controller load that exists whenever the bus is active, watts.
    pub idle_power_w: f64,
    /// Additional compute/control load, watts.
    pub control_power_w: f64,
    /// Aggregate motor+drive efficiency from electrical input to mechanical output.
    pub actuator_efficiency: f64,
    /// Maximum electrical power the bus may continuously deliver, watts.
    pub max_continuous_power_w: f64,
}

impl Default for PowerBusConfig {
    fn default() -> Self {
        Self {
            // 1 kWh usable pack. This is a simulation default, not a hardware claim.
            capacity_j: 3_600_000.0,
            idle_power_w: 20.0,
            control_power_w: 15.0,
            actuator_efficiency: 0.82,
            max_continuous_power_w: 1_500.0,
        }
    }
}

impl PowerBusConfig {
    pub fn is_valid(&self) -> bool {
        self.capacity_j.is_finite()
            && self.capacity_j > 0.0
            && self.idle_power_w.is_finite()
            && self.idle_power_w >= 0.0
            && self.control_power_w.is_finite()
            && self.control_power_w >= 0.0
            && self.actuator_efficiency.is_finite()
            && self.actuator_efficiency > 0.0
            && self.actuator_efficiency <= 1.0
            && self.max_continuous_power_w.is_finite()
            && self.max_continuous_power_w > 0.0
    }
}

/// Result of one deterministic power-allocation step.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PowerAllocation {
    /// Fraction [0,1] of requested actuator mechanical power that can be supplied.
    pub actuator_scale: f64,
    /// Electrical power delivered by the bus this step.
    pub electrical_power_w: f64,
    /// Requested actuator mechanical power before curtailment.
    pub requested_mechanical_power_w: f64,
    /// Mechanical actuator power that the bus can support after curtailment.
    pub supplied_mechanical_power_w: f64,
    /// State of charge after the step.
    pub state_of_charge: f64,
}

/// Finite stored-energy source shared by exoskeleton actuators and grounded auxiliaries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PowerBus {
    config: PowerBusConfig,
    energy_j: f64,
    cumulative_draw_j: f64,
}

impl PowerBus {
    pub fn new(config: PowerBusConfig) -> Self {
        assert!(config.is_valid(), "PowerBusConfig must be finite and physically bounded");
        Self {
            energy_j: config.capacity_j,
            cumulative_draw_j: 0.0,
            config,
        }
    }

    pub fn state_of_charge(&self) -> f64 {
        (self.energy_j / self.config.capacity_j).clamp(0.0, 1.0)
    }

    pub fn remaining_energy_j(&self) -> f64 {
        self.energy_j
    }

    pub fn cumulative_draw_j(&self) -> f64 {
        self.cumulative_draw_j
    }

    pub fn config(&self) -> PowerBusConfig {
        self.config
    }

    /// Allocate finite bus power for a simulation step.
    ///
    /// `requested_mechanical_power_w` is the aggregate magnitude of actuator
    /// mechanical power requested by the current controller. `auxiliary_power_w`
    /// represents grounded electrical loads such as sensing or communications.
    /// Regenerative capture is intentionally out of scope for v0.1.
    pub fn allocate(
        &mut self,
        requested_mechanical_power_w: f64,
        auxiliary_power_w: f64,
        dt: f64,
    ) -> PowerAllocation {
        assert!(requested_mechanical_power_w.is_finite() && requested_mechanical_power_w >= 0.0);
        assert!(auxiliary_power_w.is_finite() && auxiliary_power_w >= 0.0);
        assert!(dt.is_finite() && dt > 0.0);

        if self.energy_j <= 0.0 {
            return PowerAllocation {
                actuator_scale: 0.0,
                electrical_power_w: 0.0,
                requested_mechanical_power_w,
                supplied_mechanical_power_w: 0.0,
                state_of_charge: 0.0,
            };
        }

        let fixed_load_w = self.config.idle_power_w + self.config.control_power_w + auxiliary_power_w;
        let requested_actuator_electrical_w =
            requested_mechanical_power_w / self.config.actuator_efficiency;
        let requested_total_w = fixed_load_w + requested_actuator_electrical_w;

        let energy_limited_w = self.energy_j / dt;
        let available_total_w = self
            .config
            .max_continuous_power_w
            .min(energy_limited_w)
            .max(0.0);
        let delivered_total_w = requested_total_w.min(available_total_w);
        let actuator_electrical_w = (delivered_total_w - fixed_load_w).max(0.0);
        let supplied_mechanical_power_w =
            (actuator_electrical_w * self.config.actuator_efficiency)
                .min(requested_mechanical_power_w);
        let actuator_scale = if requested_mechanical_power_w > 0.0 {
            (supplied_mechanical_power_w / requested_mechanical_power_w).clamp(0.0, 1.0)
        } else {
            1.0
        };

        let draw_j = (delivered_total_w * dt).min(self.energy_j);
        self.energy_j -= draw_j;
        self.cumulative_draw_j += draw_j;

        PowerAllocation {
            actuator_scale,
            electrical_power_w: delivered_total_w,
            requested_mechanical_power_w,
            supplied_mechanical_power_w,
            state_of_charge: self.state_of_charge(),
        }
    }

    pub fn reset_full(&mut self) {
        self.energy_j = self.config.capacity_j;
        self.cumulative_draw_j = 0.0;
    }
}

impl Default for PowerBus {
    fn default() -> Self {
        Self::new(PowerBusConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn higher_actuator_work_draws_more_energy() {
        let config = PowerBusConfig {
            capacity_j: 100_000.0,
            max_continuous_power_w: 10_000.0,
            ..Default::default()
        };
        let mut idle = PowerBus::new(config);
        let mut working = PowerBus::new(config);
        idle.allocate(0.0, 0.0, 1.0);
        working.allocate(500.0, 0.0, 1.0);
        assert!(working.remaining_energy_j() < idle.remaining_energy_j());
    }

    #[test]
    fn continuous_limit_curtailed_actuator_work() {
        let config = PowerBusConfig {
            capacity_j: 100_000.0,
            idle_power_w: 0.0,
            control_power_w: 0.0,
            actuator_efficiency: 1.0,
            max_continuous_power_w: 100.0,
        };
        let mut bus = PowerBus::new(config);
        let allocation = bus.allocate(400.0, 0.0, 1.0);
        assert!((allocation.actuator_scale - 0.25).abs() < 1e-12);
        assert!((allocation.supplied_mechanical_power_w - 100.0).abs() < 1e-12);
    }

    #[test]
    fn empty_bus_cannot_supply_actuation() {
        let config = PowerBusConfig {
            capacity_j: 10.0,
            idle_power_w: 0.0,
            control_power_w: 0.0,
            actuator_efficiency: 1.0,
            max_continuous_power_w: 100.0,
        };
        let mut bus = PowerBus::new(config);
        bus.allocate(100.0, 0.0, 1.0);
        assert_eq!(bus.state_of_charge(), 0.0);
        let allocation = bus.allocate(1.0, 0.0, 1.0);
        assert_eq!(allocation.actuator_scale, 0.0);
        assert_eq!(allocation.electrical_power_w, 0.0);
    }

    #[test]
    fn deterministic_replay_matches_exactly() {
        let mut a = PowerBus::default();
        let mut b = PowerBus::default();
        for i in 0..1_000 {
            let requested = ((i % 37) as f64) * 11.0;
            assert_eq!(a.allocate(requested, 7.0, 0.005), b.allocate(requested, 7.0, 0.005));
        }
        assert_eq!(a, b);
    }
}
