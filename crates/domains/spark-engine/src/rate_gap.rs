// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Rate Gap Calculator (Option D)
//!
//! Quantitative tools for analyzing the gap between observed and predicted
//! fusion rates in LCF experiments. This is the central mystery: NASA observed
//! ~10³ n/s while standard Gamow physics predicts ~10⁻⁵⁰.

use crate::constants::*;
use crate::physics::GamowResult;
use serde::{Deserialize, Serialize};

/// Complete rate gap analysis for an LCF observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateGapAnalysis {
    /// Experimental conditions
    pub conditions: ExperimentalConditions,
    /// Observed neutron rate (n/s)
    pub observed_rate: f64,
    /// Predicted rate from standard Gamow (n/s)
    pub gamow_predicted_rate: f64,
    /// Gap factor (observed / predicted)
    pub gap_factor: f64,
    /// Log10 of gap factor (orders of magnitude)
    pub gap_orders: f64,
    /// Breakdown of what could explain the gap
    pub gap_breakdown: GapBreakdown,
    /// Assessment
    pub assessment: GapAssessment,
}

/// Experimental conditions for rate calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentalConditions {
    /// Lattice temperature (K)
    pub temperature_k: f64,
    /// Host material (affects screening)
    pub host_material: HostMaterial,
    /// D/Pd loading ratio
    pub loading_ratio: f64,
    /// Active volume (cm³)
    pub active_volume_cm3: f64,
    /// Trigger type
    pub trigger: TriggerType,
    /// Trigger power/intensity
    pub trigger_intensity: f64,
}

impl Default for ExperimentalConditions {
    fn default() -> Self {
        Self {
            temperature_k: 300.0,
            host_material: HostMaterial::Palladium,
            loading_ratio: 0.7,
            active_volume_cm3: 0.01,
            trigger: TriggerType::XRay,
            trigger_intensity: 1.0,
        }
    }
}

/// Host material for D loading.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HostMaterial {
    Palladium,
    Titanium,
    Nickel,
    Erbium,
}

impl HostMaterial {
    /// Electron screening energy (eV) - measured values.
    pub fn screening_ev(&self) -> f64 {
        match self {
            Self::Palladium => 309.0, // Raiola 2004
            Self::Titanium => 65.0,   // Estimated
            Self::Nickel => 95.0,     // Estimated
            Self::Erbium => 800.0,    // NASA Steinetz 2020
        }
    }

    /// D atom density at full loading (atoms/cm³).
    pub fn d_density_full(&self) -> f64 {
        match self {
            Self::Palladium => 6.8e22, // D/Pd = 1
            Self::Titanium => 9.0e22,  // D/Ti = 2 possible
            Self::Nickel => 5.0e22,    // Lower solubility
            Self::Erbium => 1.5e23,    // ErD3 possible
        }
    }

    /// Phonon energy (keV).
    pub fn phonon_kev(&self) -> f64 {
        match self {
            Self::Palladium => 0.056,
            Self::Titanium => 0.045,
            Self::Nickel => 0.040,
            Self::Erbium => 0.025,
        }
    }
}

/// Trigger mechanism type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TriggerType {
    /// Bremsstrahlung X-rays (NASA approach)
    XRay,
    /// Gamma irradiation
    Gamma,
    /// Acoustic (ultrasonic) drive
    Acoustic,
    /// Electrochemical loading
    Electrochemical,
    /// Thermal cycling
    Thermal,
    /// None (equilibrium conditions)
    None,
}

/// Breakdown of gap into contributing factors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapBreakdown {
    /// Enhancement from electron screening (Ue)
    pub screening_enhancement: f64,
    /// Enhancement from phonon-assisted tunneling
    pub phonon_enhancement: f64,
    /// Enhancement from trigger-induced hot spots
    pub hot_spot_enhancement: f64,
    /// Remaining unexplained gap
    pub unexplained_factor: f64,
    /// Log10 of unexplained factor
    pub unexplained_orders: f64,
}

/// Assessment of the rate gap.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapAssessment {
    /// Primary classification
    pub classification: GapClassification,
    /// Is this gap explainable by known physics?
    pub explainable_by_known_physics: bool,
    /// Confidence in observation (0-1)
    pub observation_confidence: f64,
    /// Recommended next steps
    pub recommendations: Vec<String>,
}

/// Classification of gap magnitude.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GapClassification {
    /// < 10×: within measurement uncertainty
    WithinUncertainty,
    /// 10-1000×: significant but potentially explainable
    Significant,
    /// 1000-10^10×: anomalous
    Anomalous,
    /// > 10^10×: requires new physics or systematic error
    Extraordinary,
}

/// Calculator for rate gaps.
pub struct RateGapCalculator;

impl RateGapCalculator {
    /// Perform complete rate gap analysis.
    pub fn analyze(conditions: &ExperimentalConditions, observed_rate: f64) -> RateGapAnalysis {
        // Calculate D density from loading
        let n_d = conditions.host_material.d_density_full() * conditions.loading_ratio;

        // Get screening energy
        let ue = conditions.host_material.screening_ev();

        // Calculate Gamow-predicted rate
        let gamow = crate::physics::GamowIntegration::dd_rate(
            conditions.temperature_k,
            ue,
            0, // No phonon enhancement in baseline
        );

        let gamow_predicted = gamow.to_neutron_rate(n_d, conditions.active_volume_cm3);

        // Calculate gap
        let gap_factor = if gamow_predicted > 0.0 {
            observed_rate / gamow_predicted
        } else {
            f64::INFINITY
        };

        let gap_orders = if gap_factor > 0.0 && gap_factor.is_finite() {
            gap_factor.log10()
        } else {
            f64::INFINITY
        };

        // Break down the gap
        let gap_breakdown = Self::breakdown_gap(conditions, &gamow, gap_factor);

        // Assess the gap
        let assessment = Self::assess_gap(gap_orders, &gap_breakdown, conditions);

        RateGapAnalysis {
            conditions: conditions.clone(),
            observed_rate,
            gamow_predicted_rate: gamow_predicted,
            gap_factor,
            gap_orders,
            gap_breakdown,
            assessment,
        }
    }

    /// Analyze NASA Steinetz 2020 results.
    pub fn analyze_nasa_steinetz() -> RateGapAnalysis {
        let conditions = ExperimentalConditions {
            temperature_k: 300.0,
            host_material: HostMaterial::Erbium,
            loading_ratio: 0.7,      // ErD2 approximate
            active_volume_cm3: 0.01, // ~1 cm² × 25 μm
            trigger: TriggerType::XRay,
            trigger_intensity: 1.0,
        };

        Self::analyze(&conditions, NASA_OBSERVED_RATE)
    }

    /// Break down the gap into contributing factors.
    fn breakdown_gap(
        conditions: &ExperimentalConditions,
        gamow: &GamowResult,
        total_gap: f64,
    ) -> GapBreakdown {
        // Screening enhancement (already included in Gamow calc)
        // Compare to unscreened calculation
        let unscreened = crate::physics::GamowIntegration::dd_rate(
            conditions.temperature_k,
            0.0, // No screening
            0,
        );
        let screening_enhancement = if unscreened.sigma_v_cm3_s > 0.0 {
            gamow.sigma_v_cm3_s / unscreened.sigma_v_cm3_s
        } else {
            1.0
        };

        // Phonon enhancement (estimate based on modes)
        let phonon_enhancement = match conditions.trigger {
            TriggerType::Acoustic => 1e3, // Up to 3 coherent phonon modes
            TriggerType::XRay => 1e2,     // Some phonon excitation
            _ => 1.0,
        };

        // Hot spot enhancement (trigger-dependent)
        let hot_spot_enhancement = match conditions.trigger {
            TriggerType::XRay => 1e10,           // Transient heating to ~1000K
            TriggerType::Gamma => 1e8,           // Less localized
            TriggerType::Electrochemical => 1e5, // Local current heating
            _ => 1.0,
        };

        // Unexplained factor
        let explained_enhancement =
            screening_enhancement * phonon_enhancement * hot_spot_enhancement;
        let unexplained = total_gap / explained_enhancement;
        let unexplained_orders = if unexplained > 0.0 && unexplained.is_finite() {
            unexplained.log10()
        } else {
            f64::INFINITY
        };

        GapBreakdown {
            screening_enhancement,
            phonon_enhancement,
            hot_spot_enhancement,
            unexplained_factor: unexplained,
            unexplained_orders,
        }
    }

    /// Assess the gap and provide recommendations.
    fn assess_gap(
        gap_orders: f64,
        breakdown: &GapBreakdown,
        conditions: &ExperimentalConditions,
    ) -> GapAssessment {
        let classification = if gap_orders < 1.0 {
            GapClassification::WithinUncertainty
        } else if gap_orders < 3.0 {
            GapClassification::Significant
        } else if gap_orders < 10.0 {
            GapClassification::Anomalous
        } else {
            GapClassification::Extraordinary
        };

        // Is it explainable by known physics extensions?
        let explainable = breakdown.unexplained_orders < 10.0;

        // Observation confidence based on trigger type and conditions
        let observation_confidence = match conditions.trigger {
            TriggerType::XRay => 0.8,            // NASA used good diagnostics
            TriggerType::Electrochemical => 0.4, // Historically contentious
            TriggerType::Gamma => 0.7,
            _ => 0.5,
        };

        // Recommendations
        let mut recommendations = vec![];

        match classification {
            GapClassification::WithinUncertainty => {
                recommendations.push("Gap is within uncertainty - improve statistics".to_string());
            }
            GapClassification::Significant => {
                recommendations
                    .push("Verify neutron energy spectrum (2.45 MeV for D-D)".to_string());
                recommendations.push("Run H control to rule out non-fusion sources".to_string());
            }
            GapClassification::Anomalous => {
                recommendations.push("Independent replication at different lab".to_string());
                recommendations.push("Time-resolved diagnostics during trigger".to_string());
                recommendations.push("Test multiple host materials".to_string());
            }
            GapClassification::Extraordinary => {
                recommendations.push("CRITICAL: Independent replication required".to_string());
                recommendations.push("Extensive background characterization".to_string());
                recommendations.push("Consider systematic error sources".to_string());
                recommendations.push("Theoretical engagement for new physics models".to_string());
            }
        }

        if breakdown.unexplained_orders > 20.0 {
            recommendations.push(
                "Gap too large for any known physics extension - check for systematic errors"
                    .to_string(),
            );
        }

        GapAssessment {
            classification,
            explainable_by_known_physics: explainable,
            observation_confidence,
            recommendations,
        }
    }

    /// Calculate minimum screening energy needed to explain observed rate.
    pub fn required_screening_for_rate(
        conditions: &ExperimentalConditions,
        target_rate: f64,
    ) -> f64 {
        // Binary search for required screening
        let n_d = conditions.host_material.d_density_full() * conditions.loading_ratio;
        let volume = conditions.active_volume_cm3;

        let mut ue_low = 0.0;
        let mut ue_high = 100000.0; // 100 keV max

        for _ in 0..100 {
            let ue_mid = (ue_low + ue_high) / 2.0;
            let gamow =
                crate::physics::GamowIntegration::dd_rate(conditions.temperature_k, ue_mid, 0);
            let rate = gamow.to_neutron_rate(n_d, volume);

            if rate < target_rate {
                ue_low = ue_mid;
            } else {
                ue_high = ue_mid;
            }
        }

        (ue_low + ue_high) / 2.0
    }

    /// Calculate minimum temperature needed to explain observed rate.
    pub fn required_temperature_for_rate(
        conditions: &ExperimentalConditions,
        target_rate: f64,
    ) -> f64 {
        let n_d = conditions.host_material.d_density_full() * conditions.loading_ratio;
        let volume = conditions.active_volume_cm3;
        let ue = conditions.host_material.screening_ev();

        let mut t_low = 100.0;
        let mut t_high = 1e9; // 1 billion K max

        for _ in 0..100 {
            let t_mid = (t_low + t_high) / 2.0;
            let gamow = crate::physics::GamowIntegration::dd_rate(t_mid, ue, 0);
            let rate = gamow.to_neutron_rate(n_d, volume);

            if rate < target_rate {
                t_low = t_mid;
            } else {
                t_high = t_mid;
            }
        }

        (t_low + t_high) / 2.0
    }

    /// Generate sensitivity analysis: how rate varies with each parameter.
    pub fn sensitivity_analysis(base_conditions: &ExperimentalConditions) -> SensitivityAnalysis {
        let base_rate = {
            let n_d =
                base_conditions.host_material.d_density_full() * base_conditions.loading_ratio;
            let ue = base_conditions.host_material.screening_ev();
            let gamow =
                crate::physics::GamowIntegration::dd_rate(base_conditions.temperature_k, ue, 0);
            gamow.to_neutron_rate(n_d, base_conditions.active_volume_cm3)
        };

        // Temperature sensitivity: +10K
        let rate_t_plus = {
            let n_d =
                base_conditions.host_material.d_density_full() * base_conditions.loading_ratio;
            let ue = base_conditions.host_material.screening_ev();
            let gamow = crate::physics::GamowIntegration::dd_rate(
                base_conditions.temperature_k + 10.0,
                ue,
                0,
            );
            gamow.to_neutron_rate(n_d, base_conditions.active_volume_cm3)
        };
        let temp_sensitivity = if base_rate > 0.0 {
            (rate_t_plus / base_rate - 1.0) / 10.0 // per K
        } else {
            0.0
        };

        // Screening sensitivity: +10 eV
        let rate_ue_plus = {
            let n_d =
                base_conditions.host_material.d_density_full() * base_conditions.loading_ratio;
            let ue = base_conditions.host_material.screening_ev() + 10.0;
            let gamow =
                crate::physics::GamowIntegration::dd_rate(base_conditions.temperature_k, ue, 0);
            gamow.to_neutron_rate(n_d, base_conditions.active_volume_cm3)
        };
        let screening_sensitivity = if base_rate > 0.0 {
            (rate_ue_plus / base_rate - 1.0) / 10.0 // per eV
        } else {
            0.0
        };

        // Loading sensitivity: +0.1
        let rate_loading_plus = {
            let n_d = base_conditions.host_material.d_density_full()
                * (base_conditions.loading_ratio + 0.1);
            let ue = base_conditions.host_material.screening_ev();
            let gamow =
                crate::physics::GamowIntegration::dd_rate(base_conditions.temperature_k, ue, 0);
            gamow.to_neutron_rate(n_d, base_conditions.active_volume_cm3)
        };
        let loading_sensitivity = if base_rate > 0.0 {
            (rate_loading_plus / base_rate - 1.0) / 0.1 // per 0.1 loading
        } else {
            0.0
        };

        SensitivityAnalysis {
            base_rate,
            temp_sensitivity_per_k: temp_sensitivity,
            screening_sensitivity_per_ev: screening_sensitivity,
            loading_sensitivity_per_01: loading_sensitivity,
            dominant_factor: if temp_sensitivity.abs() > screening_sensitivity.abs() {
                if temp_sensitivity.abs() > loading_sensitivity.abs() {
                    "temperature".to_string()
                } else {
                    "loading".to_string()
                }
            } else if screening_sensitivity.abs() > loading_sensitivity.abs() {
                "screening".to_string()
            } else {
                "loading".to_string()
            },
        }
    }
}

/// Sensitivity analysis results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityAnalysis {
    /// Base reaction rate (n/s)
    pub base_rate: f64,
    /// Fractional rate change per K temperature increase
    pub temp_sensitivity_per_k: f64,
    /// Fractional rate change per eV screening increase
    pub screening_sensitivity_per_ev: f64,
    /// Fractional rate change per 0.1 loading increase
    pub loading_sensitivity_per_01: f64,
    /// Which parameter has the largest effect
    pub dominant_factor: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nasa_analysis() {
        let analysis = RateGapCalculator::analyze_nasa_steinetz();

        // Should show extraordinary gap
        assert!(analysis.gap_orders > 40.0);
        assert_eq!(
            analysis.assessment.classification,
            GapClassification::Extraordinary
        );
    }

    #[test]
    fn test_gap_breakdown() {
        let conditions = ExperimentalConditions::default();
        let analysis = RateGapCalculator::analyze(&conditions, 1e3);

        // Screening enhancement should be > 1
        assert!(analysis.gap_breakdown.screening_enhancement >= 1.0);

        // Hot spot enhancement should be significant for X-ray trigger
        assert!(analysis.gap_breakdown.hot_spot_enhancement > 1.0);
    }

    #[test]
    fn test_required_screening() {
        let conditions = ExperimentalConditions::default();
        let required_ue = RateGapCalculator::required_screening_for_rate(&conditions, 1e3);

        // Should require massive screening to explain NASA rate
        assert!(required_ue > 1000.0); // keV-scale screening needed
    }

    #[test]
    fn test_required_temperature() {
        let conditions = ExperimentalConditions::default();
        let required_t = RateGapCalculator::required_temperature_for_rate(&conditions, 1e3);

        // Should require significantly elevated temperature (well above room temp)
        // The exact value depends on the screening energy and volume,
        // but should be much higher than 300K
        assert!(required_t > 1000.0); // At least 1000K needed
    }

    #[test]
    fn test_sensitivity() {
        let conditions = ExperimentalConditions::default();
        let sensitivity = RateGapCalculator::sensitivity_analysis(&conditions);

        // Temperature should be important factor
        assert!(sensitivity.temp_sensitivity_per_k > 0.0);
    }

    #[test]
    fn test_host_materials() {
        // Erbium has highest screening
        assert!(HostMaterial::Erbium.screening_ev() > HostMaterial::Palladium.screening_ev());

        // Erbium has highest D density
        assert!(HostMaterial::Erbium.d_density_full() > HostMaterial::Palladium.d_density_full());
    }
}
