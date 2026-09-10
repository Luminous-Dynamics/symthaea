// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Human metabolic reference model for EVA trade studies.
//!
//! This is a research estimator, not a medical device. It maps external
//! mechanical workload into metabolic power, O2 demand, CO2 production, and
//! metabolic heat using explicit coefficients and uncertainty.

use serde::{Deserialize, Serialize};

use crate::space_exosuit::ExosuitEvidenceLevel;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MetabolicModelConfig {
    /// Baseline resting/standing metabolic power, W.
    pub basal_power_w: f64,
    /// Positive external mechanical efficiency in (0, 1].
    pub positive_efficiency: f64,
    /// Fractional cost multiplier for negative mechanical work.
    pub negative_work_cost: f64,
    /// Approximate energy liberated per litre of oxygen consumed, J/L.
    pub joules_per_liter_o2: f64,
    /// Respiratory exchange ratio VCO2/VO2 used for the reference estimate.
    pub respiratory_exchange_ratio: f64,
    /// Relative 1-sigma uncertainty applied to derived rates.
    pub relative_uncertainty: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl Default for MetabolicModelConfig {
    fn default() -> Self {
        Self {
            basal_power_w: 100.0,
            positive_efficiency: 0.24,
            negative_work_cost: 0.35,
            joules_per_liter_o2: 20_100.0,
            respiratory_exchange_ratio: 0.85,
            relative_uncertainty: 0.20,
            evidence: ExosuitEvidenceLevel::Simulation,
        }
    }
}

impl MetabolicModelConfig {
    pub fn is_valid(&self) -> bool {
        self.basal_power_w.is_finite()
            && self.basal_power_w >= 0.0
            && self.positive_efficiency.is_finite()
            && self.positive_efficiency > 0.0
            && self.positive_efficiency <= 1.0
            && self.negative_work_cost.is_finite()
            && self.negative_work_cost >= 0.0
            && self.joules_per_liter_o2.is_finite()
            && self.joules_per_liter_o2 > 0.0
            && self.respiratory_exchange_ratio.is_finite()
            && self.respiratory_exchange_ratio >= 0.0
            && self.relative_uncertainty.is_finite()
            && self.relative_uncertainty >= 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HumanWorkload {
    /// Positive mechanical power delivered by the human to the environment.
    pub positive_mechanical_power_w: f64,
    /// Magnitude of negative/eccentric mechanical power, W.
    pub negative_mechanical_power_w: f64,
}

impl HumanWorkload {
    pub fn is_valid(&self) -> bool {
        self.positive_mechanical_power_w.is_finite()
            && self.positive_mechanical_power_w >= 0.0
            && self.negative_mechanical_power_w.is_finite()
            && self.negative_mechanical_power_w >= 0.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MetabolicEstimate {
    pub metabolic_power_w: f64,
    pub oxygen_l_min: f64,
    pub co2_l_min: f64,
    pub metabolic_heat_w: f64,
    pub uncertainty_fraction: f64,
    pub evidence: ExosuitEvidenceLevel,
}

impl MetabolicEstimate {
    pub fn upper_metabolic_power_w(&self) -> f64 {
        self.metabolic_power_w * (1.0 + self.uncertainty_fraction)
    }

    pub fn upper_oxygen_l_min(&self) -> f64 {
        self.oxygen_l_min * (1.0 + self.uncertainty_fraction)
    }

    pub fn upper_co2_l_min(&self) -> f64 {
        self.co2_l_min * (1.0 + self.uncertainty_fraction)
    }

    pub fn upper_metabolic_heat_w(&self) -> f64 {
        self.metabolic_heat_w * (1.0 + self.uncertainty_fraction)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HumanMetabolicModel {
    config: MetabolicModelConfig,
}

impl HumanMetabolicModel {
    pub fn new(config: MetabolicModelConfig) -> Option<Self> {
        config.is_valid().then_some(Self { config })
    }

    pub fn reference() -> Self {
        Self {
            config: MetabolicModelConfig::default(),
        }
    }

    pub fn estimate(&self, workload: HumanWorkload) -> Option<MetabolicEstimate> {
        if !workload.is_valid() || !self.config.is_valid() {
            return None;
        }

        let positive_cost = workload.positive_mechanical_power_w / self.config.positive_efficiency;
        let negative_cost = workload.negative_mechanical_power_w * self.config.negative_work_cost;
        let metabolic_power_w = self.config.basal_power_w + positive_cost + negative_cost;

        let oxygen_l_s = metabolic_power_w / self.config.joules_per_liter_o2;
        let oxygen_l_min = oxygen_l_s * 60.0;
        let co2_l_min = oxygen_l_min * self.config.respiratory_exchange_ratio;

        // Mechanical work leaving the body is not retained as metabolic heat.
        // Negative work is treated conservatively as internal metabolic cost.
        let metabolic_heat_w =
            (metabolic_power_w - workload.positive_mechanical_power_w).max(0.0);

        Some(MetabolicEstimate {
            metabolic_power_w,
            oxygen_l_min,
            co2_l_min,
            metabolic_heat_w,
            uncertainty_fraction: self.config.relative_uncertainty,
            evidence: self.config.evidence,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn more_positive_work_increases_o2_co2_and_heat() {
        let model = HumanMetabolicModel::reference();
        let low = model
            .estimate(HumanWorkload {
                positive_mechanical_power_w: 25.0,
                negative_mechanical_power_w: 0.0,
            })
            .unwrap();
        let high = model
            .estimate(HumanWorkload {
                positive_mechanical_power_w: 150.0,
                negative_mechanical_power_w: 0.0,
            })
            .unwrap();
        assert!(high.metabolic_power_w > low.metabolic_power_w);
        assert!(high.oxygen_l_min > low.oxygen_l_min);
        assert!(high.co2_l_min > low.co2_l_min);
        assert!(high.metabolic_heat_w > low.metabolic_heat_w);
    }

    #[test]
    fn exosuit_work_savings_can_reduce_metabolic_load() {
        let model = HumanMetabolicModel::reference();
        let unassisted = model
            .estimate(HumanWorkload {
                positive_mechanical_power_w: 120.0,
                negative_mechanical_power_w: 0.0,
            })
            .unwrap();
        let assisted = model
            .estimate(HumanWorkload {
                positive_mechanical_power_w: 70.0,
                negative_mechanical_power_w: 0.0,
            })
            .unwrap();
        assert!(assisted.oxygen_l_min < unassisted.oxygen_l_min);
        assert!(assisted.metabolic_heat_w < unassisted.metabolic_heat_w);
    }

    #[test]
    fn uncertainty_is_explicit_and_conservative() {
        let model = HumanMetabolicModel::reference();
        let estimate = model
            .estimate(HumanWorkload {
                positive_mechanical_power_w: 100.0,
                negative_mechanical_power_w: 20.0,
            })
            .unwrap();
        assert!(estimate.upper_oxygen_l_min() >= estimate.oxygen_l_min);
        assert_eq!(estimate.evidence, ExosuitEvidenceLevel::Simulation);
    }

    #[test]
    fn malformed_workload_is_rejected() {
        let model = HumanMetabolicModel::reference();
        assert!(model
            .estimate(HumanWorkload {
                positive_mechanical_power_w: f64::NAN,
                negative_mechanical_power_w: 0.0,
            })
            .is_none());
    }
}
