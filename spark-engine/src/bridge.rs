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

// ============================================================================
// Physics Discovery Integration
// ============================================================================

/// Experimental observation for HDC encoding.
///
/// Captures a single experimental measurement for pattern analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LcfObservation {
    /// Measurement timestamp (seconds from start)
    pub timestamp_s: f64,
    /// Temperature (K)
    pub temperature_k: f64,
    /// D/Pd loading ratio
    pub loading_ratio: f64,
    /// Measured neutron rate (n/s)
    pub neutron_rate: f64,
    /// Trigger energy (keV)
    pub trigger_energy_kev: Option<f64>,
    /// Host material screening energy (eV)
    pub screening_ev: f64,
    /// Expected rate from Gamow physics
    pub gamow_predicted_rate: f64,
    /// Enhancement factor (observed/predicted)
    pub enhancement_factor: f64,
    /// Is this an anomaly?
    pub is_anomaly: bool,
}

impl LcfObservation {
    /// Create observation from experimental conditions and measured rate.
    pub fn from_measurement(
        conditions: &ExperimentalConditions,
        neutron_rate: f64,
        timestamp_s: f64,
        trigger_energy_kev: Option<f64>,
    ) -> Self {
        let screening = conditions.host_material.screening_ev();
        let gamow = GamowIntegration::dd_rate(conditions.temperature_k, screening, 0);

        let n_d = conditions.host_material.d_density_full() * conditions.loading_ratio;
        let volume_cm3 = 1.0; // Normalize to 1 cm³
        let predicted_rate = gamow.to_neutron_rate(n_d, volume_cm3);

        let enhancement_factor = if predicted_rate > 0.0 {
            neutron_rate / predicted_rate
        } else {
            f64::INFINITY
        };

        let is_anomaly = enhancement_factor > 1e3; // >1000× is anomalous

        Self {
            timestamp_s,
            temperature_k: conditions.temperature_k,
            loading_ratio: conditions.loading_ratio,
            neutron_rate,
            trigger_energy_kev,
            screening_ev: screening,
            gamow_predicted_rate: predicted_rate,
            enhancement_factor,
            is_anomaly,
        }
    }
}

/// Experimental data pipeline for physics discovery.
///
/// Ingests experimental data and converts it to formats suitable for
/// HDC-based anomaly detection and pattern recognition.
#[derive(Debug, Clone)]
pub struct ExperimentalDataPipeline {
    /// Collected observations
    pub observations: Vec<LcfObservation>,
    /// Temperature series for pattern analysis
    pub temperature_series: Vec<(f64, f64)>,
    /// Neutron rate series
    pub rate_series: Vec<(f64, f64)>,
    /// Enhancement factor series
    pub enhancement_series: Vec<(f64, f64)>,
}

impl ExperimentalDataPipeline {
    /// Create new empty pipeline.
    pub fn new() -> Self {
        Self {
            observations: Vec::new(),
            temperature_series: Vec::new(),
            rate_series: Vec::new(),
            enhancement_series: Vec::new(),
        }
    }

    /// Add observation to pipeline.
    pub fn add_observation(&mut self, obs: LcfObservation) {
        let t = obs.timestamp_s;
        self.temperature_series.push((t, obs.temperature_k));
        self.rate_series.push((t, obs.neutron_rate));
        self.enhancement_series.push((t, obs.enhancement_factor));
        self.observations.push(obs);
    }

    /// Add measurement directly from conditions.
    pub fn add_measurement(
        &mut self,
        conditions: &ExperimentalConditions,
        neutron_rate: f64,
        timestamp_s: f64,
        trigger_energy_kev: Option<f64>,
    ) {
        let obs = LcfObservation::from_measurement(
            conditions,
            neutron_rate,
            timestamp_s,
            trigger_energy_kev,
        );
        self.add_observation(obs);
    }

    /// Load NASA LCF baseline data.
    ///
    /// Creates synthetic observations based on NASA's published results.
    /// Steinetz et al. (2020): ~10³ n/s from PdAu under X-ray irradiation
    pub fn load_nasa_baseline(&mut self) {
        use crate::rate_gap::TriggerType;

        // NASA baseline conditions (Steinetz et al. 2020)
        let conditions = ExperimentalConditions {
            temperature_k: 300.0,
            loading_ratio: 0.7,
            host_material: crate::rate_gap::HostMaterial::Palladium,
            active_volume_cm3: 0.01,
            trigger: TriggerType::XRay,
            trigger_intensity: 1e12, // photons/s/cm²
        };

        // Synthetic NASA observations based on published data
        // These represent the typical observed neutron rates
        let nasa_data = [
            (0.0, 1000.0),      // Baseline observation
            (3600.0, 1200.0),   // Hour 1
            (7200.0, 950.0),    // Hour 2
            (10800.0, 1100.0),  // Hour 3
            (14400.0, 1050.0),  // Hour 4
        ];

        for (timestamp_s, neutron_rate) in nasa_data {
            self.add_measurement(
                &conditions,
                neutron_rate,
                timestamp_s,
                Some(12.0), // 12 keV X-rays
            );
        }
    }

    /// Identify anomalies using statistical analysis.
    pub fn detect_anomalies(&self) -> Vec<&LcfObservation> {
        self.observations.iter()
            .filter(|o| o.is_anomaly)
            .collect()
    }

    /// Calculate statistics on enhancement factors.
    pub fn enhancement_statistics(&self) -> EnhancementStats {
        if self.observations.is_empty() {
            return EnhancementStats::default();
        }

        let enhancements: Vec<f64> = self.observations.iter()
            .map(|o| o.enhancement_factor.log10())
            .filter(|e| e.is_finite())
            .collect();

        if enhancements.is_empty() {
            return EnhancementStats::default();
        }

        let n = enhancements.len() as f64;
        let mean = enhancements.iter().sum::<f64>() / n;
        let variance = enhancements.iter()
            .map(|e| (e - mean).powi(2))
            .sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let mut sorted = enhancements.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len()/2 - 1] + sorted[sorted.len()/2]) / 2.0
        } else {
            sorted[sorted.len()/2]
        };

        let min = sorted.first().copied().unwrap_or(0.0);
        let max = sorted.last().copied().unwrap_or(0.0);

        EnhancementStats {
            mean_log10: mean,
            std_dev_log10: std_dev,
            median_log10: median,
            min_log10: min,
            max_log10: max,
            n_observations: self.observations.len(),
            n_anomalies: self.observations.iter().filter(|o| o.is_anomaly).count(),
        }
    }

    /// Search for correlations between variables.
    pub fn find_correlations(&self) -> Vec<Correlation> {
        let mut correlations = Vec::new();

        if self.observations.len() < 3 {
            return correlations;
        }

        // Temperature vs enhancement
        let temp_corr = self.pearson_correlation(
            &self.observations.iter().map(|o| o.temperature_k).collect::<Vec<_>>(),
            &self.observations.iter().map(|o| o.enhancement_factor.log10()).collect::<Vec<_>>(),
        );

        if let Some(r) = temp_corr {
            correlations.push(Correlation {
                variable_x: "temperature_k".to_string(),
                variable_y: "enhancement_factor".to_string(),
                pearson_r: r,
                significant: r.abs() > 0.5,
            });
        }

        // Loading ratio vs enhancement
        let loading_corr = self.pearson_correlation(
            &self.observations.iter().map(|o| o.loading_ratio).collect::<Vec<_>>(),
            &self.observations.iter().map(|o| o.enhancement_factor.log10()).collect::<Vec<_>>(),
        );

        if let Some(r) = loading_corr {
            correlations.push(Correlation {
                variable_x: "loading_ratio".to_string(),
                variable_y: "enhancement_factor".to_string(),
                pearson_r: r,
                significant: r.abs() > 0.5,
            });
        }

        correlations
    }

    fn pearson_correlation(&self, x: &[f64], y: &[f64]) -> Option<f64> {
        if x.len() != y.len() || x.len() < 2 {
            return None;
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y: f64 = y.iter().filter(|v| v.is_finite()).sum::<f64>()
            / y.iter().filter(|v| v.is_finite()).count() as f64;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for (xi, yi) in x.iter().zip(y.iter()) {
            if !yi.is_finite() { continue; }
            cov += (xi - mean_x) * (yi - mean_y);
            var_x += (xi - mean_x).powi(2);
            var_y += (yi - mean_y).powi(2);
        }

        if var_x > 0.0 && var_y > 0.0 {
            Some(cov / (var_x.sqrt() * var_y.sqrt()))
        } else {
            None
        }
    }

    /// Generate report summarizing the experimental data.
    pub fn generate_report(&self) -> DataPipelineReport {
        let stats = self.enhancement_statistics();
        let correlations = self.find_correlations();
        let anomalies = self.detect_anomalies();

        let conclusion = if stats.n_anomalies > 0 {
            format!(
                "{} of {} observations show >1000× enhancement over Gamow prediction. \
                 Mean enhancement: 10^{:.1} ({:.1e}×). This gap requires explanation.",
                stats.n_anomalies,
                stats.n_observations,
                stats.mean_log10,
                10f64.powf(stats.mean_log10)
            )
        } else {
            "All observations consistent with standard Gamow physics.".to_string()
        };

        DataPipelineReport {
            n_observations: stats.n_observations,
            n_anomalies: stats.n_anomalies,
            enhancement_stats: stats,
            correlations,
            anomaly_timestamps: anomalies.iter().map(|o| o.timestamp_s).collect(),
            conclusion,
        }
    }
}

impl Default for ExperimentalDataPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Enhancement factor statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnhancementStats {
    /// Mean enhancement (log10)
    pub mean_log10: f64,
    /// Standard deviation (log10)
    pub std_dev_log10: f64,
    /// Median enhancement (log10)
    pub median_log10: f64,
    /// Minimum enhancement (log10)
    pub min_log10: f64,
    /// Maximum enhancement (log10)
    pub max_log10: f64,
    /// Number of observations
    pub n_observations: usize,
    /// Number flagged as anomalies
    pub n_anomalies: usize,
}

/// Correlation between variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Correlation {
    /// X variable name
    pub variable_x: String,
    /// Y variable name
    pub variable_y: String,
    /// Pearson correlation coefficient
    pub pearson_r: f64,
    /// Whether correlation is statistically significant
    pub significant: bool,
}

/// Data pipeline analysis report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPipelineReport {
    /// Total observations
    pub n_observations: usize,
    /// Anomalous observations
    pub n_anomalies: usize,
    /// Enhancement statistics
    pub enhancement_stats: EnhancementStats,
    /// Variable correlations
    pub correlations: Vec<Correlation>,
    /// Timestamps of anomalies
    pub anomaly_timestamps: Vec<f64>,
    /// Summary conclusion
    pub conclusion: String,
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

    #[test]
    fn test_lcf_observation() {
        let conditions = ExperimentalConditions::default();
        let obs = LcfObservation::from_measurement(&conditions, 1000.0, 0.0, Some(12.0));

        assert!(obs.is_anomaly); // 1000 n/s is anomalous
        assert!(obs.enhancement_factor > 1e30); // Huge enhancement
        assert_eq!(obs.temperature_k, 300.0);
    }

    #[test]
    fn test_data_pipeline() {
        let mut pipeline = ExperimentalDataPipeline::new();

        // Add some observations
        let conditions = ExperimentalConditions::default();
        pipeline.add_measurement(&conditions, 1000.0, 0.0, Some(12.0));
        pipeline.add_measurement(&conditions, 1200.0, 3600.0, Some(12.0));
        pipeline.add_measurement(&conditions, 800.0, 7200.0, Some(12.0));

        assert_eq!(pipeline.observations.len(), 3);

        let anomalies = pipeline.detect_anomalies();
        assert_eq!(anomalies.len(), 3); // All should be anomalies

        let stats = pipeline.enhancement_statistics();
        assert!(stats.mean_log10 > 30.0); // Enhancement > 10^30
    }

    #[test]
    fn test_data_pipeline_report() {
        let mut pipeline = ExperimentalDataPipeline::new();

        let conditions = ExperimentalConditions::default();
        for i in 0..5 {
            pipeline.add_measurement(&conditions, 1000.0 + i as f64 * 100.0, i as f64 * 3600.0, Some(12.0));
        }

        let report = pipeline.generate_report();
        assert_eq!(report.n_observations, 5);
        assert!(report.n_anomalies > 0);
        assert!(report.conclusion.contains("enhancement"));
    }

    #[test]
    fn test_enhancement_statistics() {
        let mut pipeline = ExperimentalDataPipeline::new();

        let conditions = ExperimentalConditions::default();
        pipeline.add_measurement(&conditions, 1e3, 0.0, None);
        pipeline.add_measurement(&conditions, 1e4, 1.0, None);
        pipeline.add_measurement(&conditions, 1e5, 2.0, None);

        let stats = pipeline.enhancement_statistics();
        assert_eq!(stats.n_observations, 3);
        assert!(stats.std_dev_log10 > 0.0); // Should have variance
    }
}
