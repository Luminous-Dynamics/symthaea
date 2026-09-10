// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic aggregate thermal model for the exoskeleton powertrain.
//!
//! This model turns electrical loss power into stored heat, removes heat through
//! a simple ambient-dependent cooling term, and exposes a bounded assistance
//! derating factor. Defaults are simulation parameters, not certified hardware data.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThermalConfig {
    /// Reference ambient temperature in degrees Celsius.
    pub ambient_c: f64,
    /// Lumped thermal capacitance of the modeled powered system, J/K.
    pub thermal_capacity_j_per_k: f64,
    /// Passive/active aggregate cooling conductance, W/K above ambient.
    pub cooling_w_per_k: f64,
    /// Temperature at which assistance begins to derate.
    pub derate_start_c: f64,
    /// Temperature at which powered assistance is forced to zero.
    pub shutdown_c: f64,
}

impl Default for ThermalConfig {
    fn default() -> Self {
        Self {
            ambient_c: 25.0,
            thermal_capacity_j_per_k: 20_000.0,
            cooling_w_per_k: 8.0,
            derate_start_c: 60.0,
            shutdown_c: 80.0,
        }
    }
}

impl ThermalConfig {
    pub fn is_valid(&self) -> bool {
        self.ambient_c.is_finite()
            && self.thermal_capacity_j_per_k.is_finite()
            && self.thermal_capacity_j_per_k > 0.0
            && self.cooling_w_per_k.is_finite()
            && self.cooling_w_per_k >= 0.0
            && self.derate_start_c.is_finite()
            && self.shutdown_c.is_finite()
            && self.shutdown_c > self.derate_start_c
            && self.derate_start_c >= self.ambient_c
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ThermalStep {
    pub generated_heat_w: f64,
    pub rejected_heat_w: f64,
    pub temperature_c: f64,
    pub assistance_scale: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ThermalModel {
    config: ThermalConfig,
    temperature_c: f64,
    cumulative_generated_j: f64,
    cumulative_rejected_j: f64,
}

impl ThermalModel {
    pub fn new(config: ThermalConfig) -> Self {
        assert!(config.is_valid(), "ThermalConfig must be finite and ordered");
        Self {
            temperature_c: config.ambient_c,
            cumulative_generated_j: 0.0,
            cumulative_rejected_j: 0.0,
            config,
        }
    }

    pub fn temperature_c(&self) -> f64 {
        self.temperature_c
    }

    pub fn assistance_scale(&self) -> f64 {
        if self.temperature_c <= self.config.derate_start_c {
            1.0
        } else if self.temperature_c >= self.config.shutdown_c {
            0.0
        } else {
            let span = self.config.shutdown_c - self.config.derate_start_c;
            1.0 - (self.temperature_c - self.config.derate_start_c) / span
        }
    }

    pub fn cumulative_generated_j(&self) -> f64 {
        self.cumulative_generated_j
    }

    pub fn cumulative_rejected_j(&self) -> f64 {
        self.cumulative_rejected_j
    }

    pub fn step(&mut self, generated_heat_w: f64, dt: f64) -> ThermalStep {
        assert!(generated_heat_w.is_finite() && generated_heat_w >= 0.0);
        assert!(dt.is_finite() && dt > 0.0);

        let delta_above_ambient = (self.temperature_c - self.config.ambient_c).max(0.0);
        let rejected_heat_w = self.config.cooling_w_per_k * delta_above_ambient;
        let net_heat_j = (generated_heat_w - rejected_heat_w) * dt;
        self.temperature_c += net_heat_j / self.config.thermal_capacity_j_per_k;
        self.temperature_c = self.temperature_c.max(self.config.ambient_c);
        self.cumulative_generated_j += generated_heat_w * dt;
        self.cumulative_rejected_j += rejected_heat_w * dt;

        ThermalStep {
            generated_heat_w,
            rejected_heat_w,
            temperature_c: self.temperature_c,
            assistance_scale: self.assistance_scale(),
        }
    }

    pub fn reset_to_ambient(&mut self) {
        self.temperature_c = self.config.ambient_c;
        self.cumulative_generated_j = 0.0;
        self.cumulative_rejected_j = 0.0;
    }
}

impl Default for ThermalModel {
    fn default() -> Self {
        Self::new(ThermalConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sustained_heat_raises_temperature() {
        let mut thermal = ThermalModel::default();
        let start = thermal.temperature_c();
        for _ in 0..1_000 {
            thermal.step(500.0, 0.1);
        }
        assert!(thermal.temperature_c() > start);
    }

    #[test]
    fn hot_system_derates_monotonically() {
        let config = ThermalConfig {
            ambient_c: 0.0,
            thermal_capacity_j_per_k: 100.0,
            cooling_w_per_k: 0.0,
            derate_start_c: 10.0,
            shutdown_c: 20.0,
        };
        let mut thermal = ThermalModel::new(config);
        thermal.step(1_500.0, 1.0); // 15 C
        let mid = thermal.assistance_scale();
        assert!(mid > 0.0 && mid < 1.0);
        thermal.step(1_000.0, 1.0); // >= 20 C
        assert_eq!(thermal.assistance_scale(), 0.0);
    }

    #[test]
    fn cooling_cannot_drive_below_ambient() {
        let mut thermal = ThermalModel::default();
        for _ in 0..1_000 {
            thermal.step(1_000.0, 0.1);
        }
        for _ in 0..100_000 {
            thermal.step(0.0, 0.1);
        }
        assert!(thermal.temperature_c() >= ThermalConfig::default().ambient_c);
    }

    #[test]
    fn deterministic_replay_matches_exactly() {
        let mut a = ThermalModel::default();
        let mut b = ThermalModel::default();
        for i in 0..10_000 {
            let heat = ((i % 53) as f64) * 17.0;
            assert_eq!(a.step(heat, 0.005), b.step(heat, 0.005));
        }
        assert_eq!(a, b);
    }
}
