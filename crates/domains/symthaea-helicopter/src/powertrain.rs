// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fuel, power-delivery, and reserve accounting for the reduced-order helicopter.
//!
//! The flight simulator previously treated engine power as inexhaustible.
//! This model turns rotor power into fuel burn, applies continuous/takeoff and
//! thermal limits, and exposes a fail-closed return/land reserve decision.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PowertrainConfig {
    pub fuel_capacity_kg: f64,
    pub fuel_lower_heating_value_j_per_kg: f64,
    pub engine_efficiency: f64,
    pub idle_fuel_flow_kg_s: f64,
    pub max_continuous_power_w: f64,
    pub max_takeoff_power_w: f64,
    pub thermal_time_constant_s: f64,
    pub ambient_temperature_k: f64,
    pub maximum_temperature_k: f64,
    pub shutdown_temperature_k: f64,
    pub reserve_fraction: f64,
}

impl Default for PowertrainConfig {
    fn default() -> Self {
        Self {
            fuel_capacity_kg: 100.0,
            fuel_lower_heating_value_j_per_kg: 43.0e6,
            engine_efficiency: 0.28,
            idle_fuel_flow_kg_s: 0.002,
            max_continuous_power_w: 350_000.0,
            max_takeoff_power_w: 420_000.0,
            thermal_time_constant_s: 90.0,
            ambient_temperature_k: 288.15,
            maximum_temperature_k: 470.0,
            shutdown_temperature_k: 520.0,
            reserve_fraction: 0.20,
        }
    }
}

impl PowertrainConfig {
    pub fn validate(&self) -> bool {
        let positive = [
            self.fuel_capacity_kg,
            self.fuel_lower_heating_value_j_per_kg,
            self.max_continuous_power_w,
            self.max_takeoff_power_w,
            self.thermal_time_constant_s,
            self.ambient_temperature_k,
            self.maximum_temperature_k,
            self.shutdown_temperature_k,
        ];
        positive.iter().all(|v| v.is_finite() && *v > 0.0)
            && self.max_takeoff_power_w >= self.max_continuous_power_w
            && self.maximum_temperature_k > self.ambient_temperature_k
            && self.shutdown_temperature_k > self.maximum_temperature_k
            && self.engine_efficiency.is_finite()
            && self.engine_efficiency > 0.0
            && self.engine_efficiency <= 1.0
            && self.idle_fuel_flow_kg_s.is_finite()
            && self.idle_fuel_flow_kg_s >= 0.0
            && self.reserve_fraction.is_finite()
            && (0.0..1.0).contains(&self.reserve_fraction)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PowertrainState {
    pub fuel_kg: f64,
    pub engine_temperature_k: f64,
    pub delivered_power_w: f64,
    pub requested_power_w: f64,
    pub delivery_fraction: f64,
    pub fuel_flow_kg_s: f64,
    pub cumulative_energy_j: f64,
    pub reserve_violated: bool,
    pub engine_shutdown: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FuelReserveAction {
    ContinueMission,
    ReturnToBase,
    LandAsSoonAsPracticable,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FuelReserveAssessment {
    pub action: FuelReserveAction,
    pub fuel_fraction: f64,
    pub estimated_return_fuel_kg: f64,
    pub protected_reserve_kg: f64,
    pub margin_kg: f64,
}

#[derive(Debug, Clone)]
pub struct PowertrainModel {
    config: PowertrainConfig,
    state: PowertrainState,
}

impl Default for PowertrainModel {
    fn default() -> Self {
        Self::new()
    }
}

impl PowertrainModel {
    pub fn new() -> Self {
        let config = PowertrainConfig::default();
        Self {
            state: PowertrainState {
                fuel_kg: config.fuel_capacity_kg,
                engine_temperature_k: config.ambient_temperature_k + 60.0,
                delivered_power_w: 0.0,
                requested_power_w: 0.0,
                delivery_fraction: 1.0,
                fuel_flow_kg_s: 0.0,
                cumulative_energy_j: 0.0,
                reserve_violated: false,
                engine_shutdown: false,
            },
            config,
        }
    }

    pub fn with_config(config: PowertrainConfig) -> Option<Self> {
        if !config.validate() {
            return None;
        }
        let mut model = Self::new();
        model.config = config;
        model.reset();
        Some(model)
    }

    pub fn config(&self) -> PowertrainConfig {
        self.config
    }

    pub fn state(&self) -> PowertrainState {
        self.state
    }

    pub fn fuel_fraction(&self) -> f64 {
        (self.state.fuel_kg / self.config.fuel_capacity_kg).clamp(0.0, 1.0)
    }

    pub fn set_fuel_fraction(&mut self, fraction: f64) -> bool {
        if !fraction.is_finite() || !(0.0..=1.0).contains(&fraction) {
            return false;
        }
        self.state.fuel_kg = fraction * self.config.fuel_capacity_kg;
        self.state.engine_shutdown = self.state.fuel_kg <= 0.0;
        self.state.delivery_fraction = if self.state.engine_shutdown { 0.0 } else { 1.0 };
        true
    }

    /// Fraction of requested rotor power available on the next physics step.
    pub fn available_power_fraction(&self) -> f64 {
        if self.state.engine_shutdown || self.state.fuel_kg <= 0.0 {
            0.0
        } else {
            self.state.delivery_fraction.clamp(0.0, 1.0)
        }
    }

    pub fn step(&mut self, requested_power_w: f64, dt: f64) -> PowertrainState {
        if !requested_power_w.is_finite() || !dt.is_finite() || dt <= 0.0 {
            self.state.engine_shutdown = true;
            self.state.delivery_fraction = 0.0;
            return self.state;
        }

        self.state.requested_power_w = requested_power_w.max(0.0);
        if self.state.fuel_kg <= 0.0
            || self.state.engine_temperature_k >= self.config.shutdown_temperature_k
        {
            self.state.fuel_kg = self.state.fuel_kg.max(0.0);
            self.state.delivered_power_w = 0.0;
            self.state.delivery_fraction = 0.0;
            self.state.fuel_flow_kg_s = 0.0;
            self.state.engine_shutdown = true;
            return self.state;
        }

        let thermal_derate = if self.state.engine_temperature_k <= self.config.maximum_temperature_k
        {
            1.0
        } else {
            ((self.config.shutdown_temperature_k - self.state.engine_temperature_k)
                / (self.config.shutdown_temperature_k - self.config.maximum_temperature_k))
                .clamp(0.0, 1.0)
        };
        let allowed_power_w = self.config.max_takeoff_power_w * thermal_derate;
        let delivered_power_w = self.state.requested_power_w.min(allowed_power_w);
        let fuel_flow_kg_s = if delivered_power_w > 0.0 {
            self.config.idle_fuel_flow_kg_s
                + delivered_power_w
                    / (self.config.fuel_lower_heating_value_j_per_kg
                        * self.config.engine_efficiency)
        } else {
            0.0
        };

        self.state.fuel_kg = (self.state.fuel_kg - fuel_flow_kg_s * dt).max(0.0);
        self.state.delivered_power_w = if self.state.fuel_kg > 0.0 {
            delivered_power_w
        } else {
            0.0
        };
        self.state.delivery_fraction = if self.state.requested_power_w > 1.0 {
            (self.state.delivered_power_w / self.state.requested_power_w).clamp(0.0, 1.0)
        } else if self.state.fuel_kg > 0.0 {
            1.0
        } else {
            0.0
        };
        self.state.fuel_flow_kg_s = fuel_flow_kg_s;
        self.state.cumulative_energy_j += self.state.delivered_power_w * dt;
        self.state.reserve_violated = self.fuel_fraction() <= self.config.reserve_fraction;
        self.state.engine_shutdown = self.state.fuel_kg <= 0.0;

        // First-order thermal model. Full takeoff power tends toward shutdown
        // temperature; zero power tends toward ambient.
        let normalized_power =
            (self.state.delivered_power_w / self.config.max_takeoff_power_w).clamp(0.0, 1.0);
        let target_temperature = self.config.ambient_temperature_k
            + normalized_power
                * (self.config.shutdown_temperature_k - self.config.ambient_temperature_k);
        let alpha = 1.0 - (-dt / self.config.thermal_time_constant_s).exp();
        self.state.engine_temperature_k +=
            alpha * (target_temperature - self.state.engine_temperature_k);
        self.state
    }

    /// Estimate whether remaining fuel protects the configured reserve after a
    /// direct return to base at the supplied groundspeed and measured burn rate.
    pub fn assess_return_reserve(
        &self,
        distance_to_base_m: f64,
        expected_groundspeed_mps: f64,
    ) -> FuelReserveAssessment {
        let protected_reserve_kg = self.config.fuel_capacity_kg * self.config.reserve_fraction;
        let valid = distance_to_base_m.is_finite()
            && distance_to_base_m >= 0.0
            && expected_groundspeed_mps.is_finite()
            && expected_groundspeed_mps > 0.0;
        let estimated_return_fuel_kg = if valid {
            let return_time_s = distance_to_base_m / expected_groundspeed_mps;
            let burn = self
                .state
                .fuel_flow_kg_s
                .max(self.config.idle_fuel_flow_kg_s);
            burn * return_time_s
        } else {
            f64::INFINITY
        };
        let margin_kg = self.state.fuel_kg - estimated_return_fuel_kg - protected_reserve_kg;
        let action =
            if !valid || self.state.engine_shutdown || self.state.fuel_kg <= protected_reserve_kg {
                FuelReserveAction::LandAsSoonAsPracticable
            } else if margin_kg <= 0.0 {
                FuelReserveAction::ReturnToBase
            } else {
                FuelReserveAction::ContinueMission
            };
        FuelReserveAssessment {
            action,
            fuel_fraction: self.fuel_fraction(),
            estimated_return_fuel_kg,
            protected_reserve_kg,
            margin_kg,
        }
    }

    pub fn reset(&mut self) {
        self.state = PowertrainState {
            fuel_kg: self.config.fuel_capacity_kg,
            engine_temperature_k: self.config.ambient_temperature_k + 60.0,
            delivered_power_w: 0.0,
            requested_power_w: 0.0,
            delivery_fraction: 1.0,
            fuel_flow_kg_s: 0.0,
            cumulative_energy_j: 0.0,
            reserve_violated: false,
            engine_shutdown: false,
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn power_use_burns_fuel_and_records_energy() {
        let mut model = PowertrainModel::new();
        let before = model.state().fuel_kg;
        model.step(300_000.0, 10.0);
        assert!(model.state().fuel_kg < before);
        assert!(model.state().cumulative_energy_j > 0.0);
        assert!(model.state().fuel_flow_kg_s > 0.0);
    }

    #[test]
    fn empty_fuel_removes_engine_power() {
        let mut model = PowertrainModel::new();
        assert!(model.set_fuel_fraction(0.0));
        model.step(300_000.0, 0.1);
        assert_eq!(model.available_power_fraction(), 0.0);
        assert!(model.state().engine_shutdown);
    }

    #[test]
    fn reserve_policy_orders_return_before_protected_reserve_is_consumed() {
        let mut model = PowertrainModel::new();
        model.step(300_000.0, 1.0);
        assert!(model.set_fuel_fraction(0.25));
        let assessment = model.assess_return_reserve(20_000.0, 40.0);
        assert_ne!(assessment.action, FuelReserveAction::ContinueMission);
    }

    #[test]
    fn invalid_return_geometry_fails_to_land_action() {
        let model = PowertrainModel::new();
        assert_eq!(
            model.assess_return_reserve(f64::NAN, 0.0).action,
            FuelReserveAction::LandAsSoonAsPracticable
        );
    }
}
