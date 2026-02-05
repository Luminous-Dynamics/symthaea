//! # Symthaea Bridge
//!
//! Integration module for connecting Spark Engine (LCF physics) with
//! Symthaea (tokamak plasma physics). Provides unified interfaces for
//! fusion physics calculations across different confinement approaches.
//!
//! ## Fusion Approaches
//!
//! - **Tokamak (Symthaea):** Hot plasma at 10⁸ K, magnetic confinement, Q > 1 achievable
//! - **LCF (Spark Engine):** Room temperature, lattice confinement, Q << 1 but anomalous observations
//!
//! This bridge allows systems to query both approaches through a unified API.

use crate::physics::{GamowIntegration, QFactor, QFactorParams};
use crate::rate_gap::{RateGapCalculator, ExperimentalConditions};
use serde::{Deserialize, Serialize};

/// Unified fusion calculation interface.
///
/// Provides a common API for both LCF and tokamak fusion calculations,
/// allowing downstream systems to query either approach.
#[derive(Debug, Clone)]
pub struct UnifiedFusionCalculator {
    /// LCF conditions
    pub lcf_conditions: Option<ExperimentalConditions>,
    /// Tokamak temperature (keV) - placeholder for Symthaea integration
    pub tokamak_temp_kev: Option<f64>,
}

impl UnifiedFusionCalculator {
    /// Create calculator for LCF only.
    pub fn lcf(conditions: ExperimentalConditions) -> Self {
        Self {
            lcf_conditions: Some(conditions),
            tokamak_temp_kev: None,
        }
    }

    /// Create calculator for tokamak only.
    pub fn tokamak(temp_kev: f64) -> Self {
        Self {
            lcf_conditions: None,
            tokamak_temp_kev: Some(temp_kev),
        }
    }

    /// Create calculator for both approaches.
    pub fn both(conditions: ExperimentalConditions, tokamak_temp_kev: f64) -> Self {
        Self {
            lcf_conditions: Some(conditions),
            tokamak_temp_kev: Some(tokamak_temp_kev),
        }
    }

    /// Calculate fusion rates for configured approaches.
    pub fn calculate(&self) -> UnifiedFusionResult {
        let lcf_result = self.lcf_conditions.as_ref().map(|conditions| {
            let screening = conditions.host_material.screening_ev();
            let gamow = GamowIntegration::dd_rate(conditions.temperature_k, screening, 0);
            let q_params = QFactorParams::lcf_typical();
            let (q_factor, _) = QFactor::compute(&gamow, &q_params);

            LcfFusionResult {
                sigma_v_cm3_s: gamow.sigma_v_cm3_s,
                q_factor,
                gamow_peak_kev: gamow.gamow_peak_kev,
                screening_enhancement: gamow.screening_enhancement,
                energy_gain_possible: q_factor > 1.0,
                conditions: conditions.clone(),
            }
        });

        let tokamak_result = self.tokamak_temp_kev.map(|temp_kev| {
            // Convert keV to K: 1 keV = 1.16 × 10⁷ K
            let temp_k = temp_kev * 1.16e7;
            let gamow = GamowIntegration::dd_rate(temp_k, 0.0, 0); // No screening in plasma

            // Tokamak Q factor estimate (simplified)
            // Real tokamak Q depends on confinement time, density, etc.
            // This is just a placeholder for integration testing
            let q_factor = gamow.sigma_v_cm3_s * 1e20; // Very rough estimate

            TokamakFusionResult {
                sigma_v_cm3_s: gamow.sigma_v_cm3_s,
                temp_kev,
                temp_k,
                q_factor_estimate: q_factor,
                energy_gain_possible: temp_kev > 10.0, // Typically need >10 keV for Q>1
            }
        });

        UnifiedFusionResult {
            lcf: lcf_result,
            tokamak: tokamak_result,
            comparison: self.compare_approaches(),
        }
    }

    /// Compare LCF and tokamak approaches.
    fn compare_approaches(&self) -> Option<ApproachComparison> {
        match (&self.lcf_conditions, &self.tokamak_temp_kev) {
            (Some(lcf), Some(tok_kev)) => {
                let tok_kev = *tok_kev;
                let lcf_screening = lcf.host_material.screening_ev();
                let lcf_gamow = GamowIntegration::dd_rate(lcf.temperature_k, lcf_screening, 0);

                let tok_temp_k = tok_kev * 1.16e7;
                let tok_gamow = GamowIntegration::dd_rate(tok_temp_k, 0.0, 0);

                let rate_ratio = if lcf_gamow.sigma_v_cm3_s > 0.0 {
                    tok_gamow.sigma_v_cm3_s / lcf_gamow.sigma_v_cm3_s
                } else {
                    f64::INFINITY
                };

                Some(ApproachComparison {
                    tokamak_rate_advantage: rate_ratio,
                    lcf_simplicity_advantage: "No magnets, room temperature, no tritium".to_string(),
                    tokamak_energy_advantage: tok_kev > 10.0,
                    lcf_compactness_advantage: true,
                    recommendation: if tok_kev > 10.0 {
                        "Tokamak for energy production, LCF for compact neutron sources".to_string()
                    } else {
                        "Neither approach achieves Q > 1 at these conditions".to_string()
                    },
                })
            }
            _ => None,
        }
    }
}

/// Unified fusion calculation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedFusionResult {
    /// LCF result (if calculated)
    pub lcf: Option<LcfFusionResult>,
    /// Tokamak result (if calculated)
    pub tokamak: Option<TokamakFusionResult>,
    /// Comparison between approaches
    pub comparison: Option<ApproachComparison>,
}

/// LCF-specific fusion result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LcfFusionResult {
    /// Reaction rate <σv>
    pub sigma_v_cm3_s: f64,
    /// Q factor
    pub q_factor: f64,
    /// Gamow peak energy
    pub gamow_peak_kev: f64,
    /// Screening enhancement
    pub screening_enhancement: f64,
    /// Whether energy gain is possible
    pub energy_gain_possible: bool,
    /// Conditions used
    pub conditions: ExperimentalConditions,
}

/// Tokamak-specific fusion result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokamakFusionResult {
    /// Reaction rate <σv>
    pub sigma_v_cm3_s: f64,
    /// Temperature in keV
    pub temp_kev: f64,
    /// Temperature in K
    pub temp_k: f64,
    /// Estimated Q factor (simplified)
    pub q_factor_estimate: f64,
    /// Whether energy gain is possible
    pub energy_gain_possible: bool,
}

/// Comparison between LCF and tokamak approaches.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApproachComparison {
    /// How much faster tokamak rate is
    pub tokamak_rate_advantage: f64,
    /// LCF simplicity advantages
    pub lcf_simplicity_advantage: String,
    /// Whether tokamak achieves energy gain
    pub tokamak_energy_advantage: bool,
    /// Whether LCF is more compact
    pub lcf_compactness_advantage: bool,
    /// Overall recommendation
    pub recommendation: String,
}

/// Anomaly flag for reactor designs.
///
/// When integrating with symthaea reactor models, this flag indicates
/// whether the design relies on unvalidated physics enhancements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyFlag {
    /// Whether design assumes anomalous enhancement
    pub assumes_anomaly: bool,
    /// Assumed enhancement factor
    pub assumed_enhancement: f64,
    /// Gap from standard physics
    pub physics_gap_orders: f64,
    /// Warning message
    pub warning: String,
    /// Recommendations
    pub recommendations: Vec<String>,
}

impl AnomalyFlag {
    /// Check if a claimed rate requires anomalous physics.
    pub fn check(claimed_rate: f64, conditions: &ExperimentalConditions) -> Self {
        let analysis = RateGapCalculator::analyze(conditions, claimed_rate);

        let assumes_anomaly = analysis.gap_orders > 3.0; // More than 1000× gap
        let warning = if assumes_anomaly {
            format!(
                "Design assumes {:.0} orders of magnitude enhancement over standard Gamow physics",
                analysis.gap_orders
            )
        } else {
            "Design is consistent with standard physics (within uncertainty)".to_string()
        };

        let mut recommendations = vec![];
        if assumes_anomaly {
            recommendations.push("Validate claimed rate with independent measurement".to_string());
            recommendations.push("Include H control to confirm nuclear origin".to_string());
            recommendations.push("Specify assumed enhancement mechanism".to_string());
        }

        Self {
            assumes_anomaly,
            assumed_enhancement: analysis.gap_factor,
            physics_gap_orders: analysis.gap_orders,
            warning,
            recommendations,
        }
    }

    /// Create flag for designs that explicitly assume anomaly.
    pub fn explicit_anomaly(enhancement: f64, mechanism: &str) -> Self {
        Self {
            assumes_anomaly: true,
            assumed_enhancement: enhancement,
            physics_gap_orders: enhancement.log10(),
            warning: format!(
                "Design explicitly assumes {:.1e}× enhancement via {}",
                enhancement, mechanism
            ),
            recommendations: vec![
                format!("Validate {} mechanism experimentally", mechanism),
                "Include fallback for standard physics case".to_string(),
            ],
        }
    }
}

/// Physics-honest reactor assessment.
///
/// Provides honest assessment of any reactor design claiming LCF-based operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HonestReactorAssessment {
    /// Claimed power output (W)
    pub claimed_power_w: f64,
    /// Physics-predicted power (W)
    pub physics_power_w: f64,
    /// Anomaly flag
    pub anomaly_flag: AnomalyFlag,
    /// Q factor assessment
    pub q_factor: f64,
    /// Energy gain possible?
    pub energy_gain_possible: bool,
    /// Overall verdict
    pub verdict: String,
}

impl HonestReactorAssessment {
    /// Assess a reactor design.
    pub fn assess(
        claimed_power_w: f64,
        conditions: &ExperimentalConditions,
        active_volume_cm3: f64,
    ) -> Self {
        // Calculate physics-predicted power
        let screening = conditions.host_material.screening_ev();
        let gamow = GamowIntegration::dd_rate(conditions.temperature_k, screening, 0);

        let n_d = conditions.host_material.d_density_full() * conditions.loading_ratio;
        let neutron_rate = gamow.to_neutron_rate(n_d, active_volume_cm3);

        // Power from D-D fusion: each reaction releases 3.65 MeV average
        // 50% go to neutron channel (3.27 MeV), 50% to proton channel (4.03 MeV)
        let reactions_per_s = neutron_rate * 2.0; // Neutrons are 50%
        let physics_power_w = reactions_per_s * 3.65e6 * 1.602e-19; // MeV to J

        // Q factor
        let input_power_w = 1.0; // Assume 1W trigger
        let q_factor = physics_power_w / input_power_w;

        // Anomaly flag
        let claimed_neutron_rate = claimed_power_w / (3.65e6 * 1.602e-19) / 2.0;
        let anomaly_flag = AnomalyFlag::check(claimed_neutron_rate, conditions);

        let energy_gain_possible = q_factor > 1.0;

        let verdict = if claimed_power_w > 0.0 && physics_power_w < 1e-100 {
            "IMPOSSIBLE under standard physics - requires unvalidated enhancements".to_string()
        } else if anomaly_flag.assumes_anomaly {
            format!(
                "Requires {:.0} orders of magnitude enhancement - UNVALIDATED",
                anomaly_flag.physics_gap_orders
            )
        } else if energy_gain_possible {
            "Consistent with physics - energy gain possible".to_string()
        } else {
            "Consistent with physics - net energy consumer (Q < 1)".to_string()
        };

        Self {
            claimed_power_w,
            physics_power_w,
            anomaly_flag,
            q_factor,
            energy_gain_possible,
            verdict,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_calculator_lcf() {
        let conditions = ExperimentalConditions::default();
        let calc = UnifiedFusionCalculator::lcf(conditions);
        let result = calc.calculate();

        assert!(result.lcf.is_some());
        assert!(result.tokamak.is_none());

        let lcf = result.lcf.unwrap();
        assert!(lcf.q_factor < 1.0); // Q < 1 at room temp
    }

    #[test]
    fn test_unified_calculator_tokamak() {
        let calc = UnifiedFusionCalculator::tokamak(10.0); // 10 keV
        let result = calc.calculate();

        assert!(result.lcf.is_none());
        assert!(result.tokamak.is_some());

        let tok = result.tokamak.unwrap();
        assert!(tok.sigma_v_cm3_s > 1e-30); // Should have measurable rate
    }

    #[test]
    fn test_anomaly_flag() {
        let conditions = ExperimentalConditions::default();

        // Check NASA-level rate (should flag as anomaly)
        let flag = AnomalyFlag::check(1000.0, &conditions);
        assert!(flag.assumes_anomaly);
        assert!(flag.physics_gap_orders > 30.0);

        // Check very low rate (should not flag)
        let flag_low = AnomalyFlag::check(1e-50, &conditions);
        assert!(!flag_low.assumes_anomaly);
    }

    #[test]
    fn test_honest_assessment() {
        let conditions = ExperimentalConditions::default();

        // Assess unrealistic power claim
        let assessment = HonestReactorAssessment::assess(1000.0, &conditions, 1.0);

        assert!(assessment.claimed_power_w > assessment.physics_power_w);
        assert!(assessment.anomaly_flag.assumes_anomaly);
        assert!(assessment.verdict.contains("UNVALIDATED") || assessment.verdict.contains("IMPOSSIBLE"));
    }
}
