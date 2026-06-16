// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Artificial placenta (ECMO-like oxygenation and nutrient exchange).
//!
//! Models a pumpless arteriovenous extracorporeal circuit that provides
//! gas exchange (O2 in, CO2 out), nutrient delivery, and waste clearance.
//! Parameters adapt to the growing metabolic demands of each gestational week.

use serde::{Deserialize, Serialize};

use crate::types::{FetalMetrics, GestationalWeek, OxygenationParams};

/// Artificial placenta providing oxygenation and nutrient exchange.
///
/// Inspired by ECMO (extracorporeal membrane oxygenation) circuits adapted
/// for fetal physiology: lower pressures, fetal hemoglobin O2 affinity,
/// and gradually increasing flow to match metabolic demands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtificialPlacenta {
    /// Oxygenation circuit parameters.
    pub oxygenation: OxygenationParams,
    /// Nutrient delivery rate (relative units, 0-1 scale).
    pub nutrient_delivery_rate: f64,
    /// Waste clearance rate (relative units, 0-1 scale).
    pub waste_clearance_rate: f64,
}

impl ArtificialPlacenta {
    /// Create a new artificial placenta with default early-gestation parameters.
    pub fn new() -> Self {
        Self {
            oxygenation: OxygenationParams {
                fio2: 0.21,             // Ambient air fraction
                flow_rate_ml_min: 50.0, // Low initial flow
                pao2_target: 30.0,      // Fetal PaO2 target (lower than adult)
                paco2_target: 40.0,     // Standard
            },
            nutrient_delivery_rate: 0.3,
            waste_clearance_rate: 0.3,
        }
    }

    /// Adjust parameters for the current gestational week.
    ///
    /// Flow rate, FiO2, and nutrient delivery increase with metabolic demand.
    pub fn adjust_for_week(&mut self, week: GestationalWeek) {
        let w = week.0 as f64;

        // Flow rate scales with fetal mass (roughly exponential)
        self.oxygenation.flow_rate_ml_min = 50.0 + w * 8.0;

        // FiO2 increases slightly in third trimester
        self.oxygenation.fio2 = if w > 28.0 {
            0.21 + (w - 28.0) * 0.005
        } else {
            0.21
        };

        // Nutrient and waste rates scale with gestational progress
        self.nutrient_delivery_rate = (0.3 + w * 0.0175).min(1.0);
        self.waste_clearance_rate = (0.3 + w * 0.0175).min(1.0);

        // PaO2 target gradually increases as lungs mature
        self.oxygenation.pao2_target = 25.0 + w * 0.5;
    }

    /// Compute oxygen delivery (mL O2/min) based on current parameters.
    pub fn oxygen_delivery(&self) -> f64 {
        // Simplified: O2 delivery = flow_rate * FiO2 * extraction_efficiency
        let extraction_efficiency = 0.25; // Fetal hemoglobin has higher O2 affinity
        self.oxygenation.flow_rate_ml_min * self.oxygenation.fio2 * extraction_efficiency
    }

    /// Check whether oxygenation and nutrition are adequate for the given metrics.
    ///
    /// Returns true if the placenta can meet the fetus's metabolic demands.
    pub fn is_adequate(&self, metrics: &FetalMetrics) -> bool {
        let w = metrics.week.0 as f64;

        // Check flow rate is sufficient for weight
        let min_flow = metrics.weight_grams * 0.05; // ~50 mL/min per kg
        if self.oxygenation.flow_rate_ml_min < min_flow {
            return false;
        }

        // Check nutrient delivery scales with gestational week
        let min_nutrient = (w / 40.0).min(0.9);
        if self.nutrient_delivery_rate < min_nutrient {
            return false;
        }

        // Check waste clearance
        if self.waste_clearance_rate < min_nutrient * 0.8 {
            return false;
        }

        true
    }
}

impl Default for ArtificialPlacenta {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_placenta() {
        let p = ArtificialPlacenta::new();
        assert!((p.oxygenation.fio2 - 0.21).abs() < 0.01);
        assert!(p.oxygenation.flow_rate_ml_min > 0.0);
    }

    #[test]
    fn test_adjust_increases_flow() {
        let mut p = ArtificialPlacenta::new();
        let flow_early = p.oxygenation.flow_rate_ml_min;
        p.adjust_for_week(GestationalWeek::new(30));
        assert!(
            p.oxygenation.flow_rate_ml_min > flow_early,
            "Flow should increase with gestation"
        );
    }

    #[test]
    fn test_adjust_increases_nutrients() {
        let mut p = ArtificialPlacenta::new();
        let nut_early = p.nutrient_delivery_rate;
        p.adjust_for_week(GestationalWeek::new(30));
        assert!(
            p.nutrient_delivery_rate > nut_early,
            "Nutrient delivery should increase"
        );
    }

    #[test]
    fn test_oxygen_delivery_positive() {
        let p = ArtificialPlacenta::new();
        assert!(p.oxygen_delivery() > 0.0);
    }

    #[test]
    fn test_adequate_early() {
        let mut p = ArtificialPlacenta::new();
        p.adjust_for_week(GestationalWeek::new(10));
        let metrics = FetalMetrics::normative(GestationalWeek::new(10));
        assert!(
            p.is_adequate(&metrics),
            "Should be adequate for early gestation"
        );
    }

    #[test]
    fn test_inadequate_when_starved() {
        let mut p = ArtificialPlacenta::new();
        p.nutrient_delivery_rate = 0.01;
        p.oxygenation.flow_rate_ml_min = 1.0;
        let metrics = FetalMetrics::normative(GestationalWeek::new(30));
        assert!(
            !p.is_adequate(&metrics),
            "Should be inadequate with minimal resources"
        );
    }

    #[test]
    fn test_placenta_serde_roundtrip() {
        let p = ArtificialPlacenta::new();
        let json = serde_json::to_string(&p).unwrap();
        let deser: ArtificialPlacenta = serde_json::from_str(&json).unwrap();
        assert!((deser.oxygenation.fio2 - p.oxygenation.fio2).abs() < 1e-9);
    }
}
