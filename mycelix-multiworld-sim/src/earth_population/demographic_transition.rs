// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Demographic transition model: 5-stage fertility decline driven by development.
//!
//! CITATION: Notestein, F. (1945) "Population — The Long View".
//! Updated by: Lee, R. (2003) "The Demographic Transition: Three Centuries of
//! Fundamental Change", Journal of Economic Perspectives 17(4).
//! van de Kaa, D. (1987) "Europe's Second Demographic Transition".
//!
//! The Total Fertility Rate (TFR) follows a logistic decline as a function of
//! a composite development index (GDP, education, infrastructure).
//!
//! LIMITATION: This is a reduced-form model. Real demographic transitions involve
//! cultural factors (religion, gender norms, urbanization) that are not captured
//! by the development index alone.

use crate::earth_regions::EarthRegion;
use serde::{Deserialize, Serialize};

/// Demographic transition parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemographicTransition {
    /// Maximum TFR (pre-transition).
    pub tfr_ceiling: f64,
    /// Minimum TFR (post-transition floor, e.g., South Korea 0.72).
    /// We use 1.2 as a structural floor (below replacement but not collapse).
    pub tfr_floor: f64,
    /// Development index at which TFR = midpoint between ceiling and floor.
    pub midpoint: f64,
    /// Steepness of the logistic transition curve.
    pub steepness: f64,
    /// Maximum TFR change per year (demographic momentum constraint).
    pub max_annual_change: f64,
}

impl Default for DemographicTransition {
    fn default() -> Self {
        Self {
            tfr_ceiling: 7.0,
            tfr_floor: 1.2,
            midpoint: 0.45, // Lower midpoint: transition starts earlier in development
            steepness: 10.0, // Steeper: faster decline once started
            max_annual_change: 0.08, // Faster max change (Iran dropped 1.5 TFR in a decade)
        }
    }
}

impl DemographicTransition {
    /// Compute composite development index for a region.
    ///
    /// Weighted average of GDP (log-scaled), education, and infrastructure.
    /// Range: [0, 1] where 0 = least developed, 1 = most developed.
    ///
    /// CITATION: UNDP Human Development Index methodology (modified).
    pub fn development_index(&self, region: &EarthRegion) -> f64 {
        // GDP component: log scale from $400 (ln≈6) to $60,000 (ln≈11)
        let gdp_component = ((region.gdp_per_capita.max(400.0).ln() - 6.0) / 5.0).clamp(0.0, 1.0);
        let edu_component = region.education_index;
        let infra_component = region.infrastructure_level;

        (0.4 * gdp_component + 0.35 * edu_component + 0.25 * infra_component).clamp(0.0, 1.0)
    }

    /// Target TFR for a given development index (logistic function).
    ///
    /// ```text
    /// TFR(d) = floor + (ceiling - floor) / (1 + exp(steepness × (d - midpoint)))
    /// ```
    pub fn target_tfr(&self, development_index: f64) -> f64 {
        self.tfr_floor
            + (self.tfr_ceiling - self.tfr_floor)
                / (1.0 + (self.steepness * (development_index - self.midpoint)).exp())
    }

    /// Current demographic transition stage based on TFR.
    pub fn stage(&self, tfr: f64) -> DemographicTransitionStage {
        if tfr > 5.0 {
            DemographicTransitionStage::PreTransition
        } else if tfr > 4.0 {
            DemographicTransitionStage::EarlyTransition
        } else if tfr > 2.1 {
            DemographicTransitionStage::LateTransition
        } else if tfr > 1.5 {
            DemographicTransitionStage::PostTransition
        } else {
            DemographicTransitionStage::SecondTransition
        }
    }
}

/// Demographic transition stage.
/// CITATION: Notestein (1945), updated by Lee (2003), van de Kaa (1987).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DemographicTransitionStage {
    /// Pre-transition: high birth, high death. TFR > 5.0.
    PreTransition,
    /// Early transition: death rate falls, birth rate still high. TFR 4.0-5.0.
    EarlyTransition,
    /// Late transition: birth rate falls. TFR 2.1-4.0.
    LateTransition,
    /// Post-transition: low birth, low death. TFR 1.5-2.1.
    PostTransition,
    /// Second demographic transition: very low fertility. TFR < 1.5.
    SecondTransition,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target_tfr_decreases_with_development() {
        let dt = DemographicTransition::default();

        let tfr_low = dt.target_tfr(0.1);
        let tfr_mid = dt.target_tfr(0.5);
        let tfr_high = dt.target_tfr(0.9);

        assert!(
            tfr_low > tfr_mid,
            "TFR should decrease: {tfr_low} vs {tfr_mid}"
        );
        assert!(
            tfr_mid > tfr_high,
            "TFR should decrease: {tfr_mid} vs {tfr_high}"
        );
        assert!(tfr_low > 4.0, "Low dev should have high TFR: {tfr_low}");
        assert!(tfr_high < 2.5, "High dev should have low TFR: {tfr_high}");
    }

    #[test]
    fn test_target_tfr_bounded() {
        let dt = DemographicTransition::default();
        let tfr_zero = dt.target_tfr(0.0);
        let tfr_one = dt.target_tfr(1.0);

        assert!(tfr_zero <= dt.tfr_ceiling, "Should not exceed ceiling");
        assert!(tfr_one >= dt.tfr_floor, "Should not go below floor");
    }

    #[test]
    fn test_stage_classification() {
        let dt = DemographicTransition::default();
        assert_eq!(dt.stage(6.0), DemographicTransitionStage::PreTransition);
        assert_eq!(dt.stage(4.5), DemographicTransitionStage::EarlyTransition);
        assert_eq!(dt.stage(3.0), DemographicTransitionStage::LateTransition);
        assert_eq!(dt.stage(1.8), DemographicTransitionStage::PostTransition);
        assert_eq!(dt.stage(1.2), DemographicTransitionStage::SecondTransition);
    }

    #[test]
    fn test_development_index_range() {
        let dt = DemographicTransition::default();

        // Poorest region: South Asia ($3000, edu 0.55, infra 0.50)
        let poor = EarthRegion {
            name: "test".into(),
            population: 100.0,
            growth_rate_annual: 0.01,
            gdp_per_capita: 3000.0,
            education_index: 0.55,
            phi_distribution_mean: 0.2,
            phi_distribution_sd: 0.1,
            economic_specialization: [0.125; 8],
            infrastructure_level: 0.50,
            climate_vulnerability: 0.5,
            cultural_profile: crate::world::CulturalProfile {
                harmony_weights: [0.125; 8],
                individualism: 0.5,
                risk_tolerance: 0.5,
                xenophilia: 0.5,
                traditionalism: 0.5,
                language_divergence: 0.0,
                ritual_count: 0,
                founding_mythology: String::new(),
            },
            spaceport_access: false,
            urbanization: 0.3,
        };

        // Richest region: North America ($55000, edu 0.90, infra 0.95)
        let rich = EarthRegion {
            gdp_per_capita: 55000.0,
            education_index: 0.90,
            infrastructure_level: 0.95,
            ..poor.clone()
        };

        let dev_poor = dt.development_index(&poor);
        let dev_rich = dt.development_index(&rich);

        assert!(dev_poor > 0.0 && dev_poor < 1.0, "Poor dev: {dev_poor}");
        assert!(
            dev_rich > dev_poor,
            "Rich should have higher dev: {dev_rich} vs {dev_poor}"
        );
        assert!(dev_rich > 0.7, "Rich should be >0.7: {dev_rich}");
    }
}
