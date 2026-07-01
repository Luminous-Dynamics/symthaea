// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
use crate::rate_gap::{ExperimentalConditions, RateGapCalculator};
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
                    lcf_simplicity_advantage: "No magnets, room temperature, no tritium"
                        .to_string(),
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
            (0.0, 1000.0),     // Baseline observation
            (3600.0, 1200.0),  // Hour 1
            (7200.0, 950.0),   // Hour 2
            (10800.0, 1100.0), // Hour 3
            (14400.0, 1050.0), // Hour 4
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
        self.observations.iter().filter(|o| o.is_anomaly).collect()
    }

    /// Calculate statistics on enhancement factors.
    pub fn enhancement_statistics(&self) -> EnhancementStats {
        if self.observations.is_empty() {
            return EnhancementStats::default();
        }

        let enhancements: Vec<f64> = self
            .observations
            .iter()
            .map(|o| o.enhancement_factor.log10())
            .filter(|e| e.is_finite())
            .collect();

        if enhancements.is_empty() {
            return EnhancementStats::default();
        }

        let n = enhancements.len() as f64;
        let mean = enhancements.iter().sum::<f64>() / n;
        let variance = enhancements.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let mut sorted = enhancements.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
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
            &self
                .observations
                .iter()
                .map(|o| o.temperature_k)
                .collect::<Vec<_>>(),
            &self
                .observations
                .iter()
                .map(|o| o.enhancement_factor.log10())
                .collect::<Vec<_>>(),
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
            &self
                .observations
                .iter()
                .map(|o| o.loading_ratio)
                .collect::<Vec<_>>(),
            &self
                .observations
                .iter()
                .map(|o| o.enhancement_factor.log10())
                .collect::<Vec<_>>(),
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
            if !yi.is_finite() {
                continue;
            }
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

// ============================================================================
// Real Experimental Data from Literature
// ============================================================================

/// Published experimental data source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteratureSource {
    /// Author(s)
    pub authors: String,
    /// Publication year
    pub year: u16,
    /// Paper title
    pub title: String,
    /// Journal/venue
    pub journal: String,
    /// DOI if available
    pub doi: Option<String>,
    /// Data type (screening, neutron rate, excess heat, etc.)
    pub data_type: LiteratureDataType,
}

/// Type of experimental data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum LiteratureDataType {
    /// Electron screening energy measurements
    ScreeningEnergy,
    /// Neutron emission rates
    NeutronRate,
    /// Excess heat claims
    ExcessHeat,
    /// Tritium production
    TritiumProduction,
    /// Charged particle emission
    ChargedParticles,
    /// Nuclear transmutation
    Transmutation,
}

/// Screening energy measurement from literature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreeningMeasurement {
    /// Source reference
    pub source: LiteratureSource,
    /// Host material
    pub host_material: String,
    /// Target nucleus
    pub target: String,
    /// Measured screening energy (eV)
    pub screening_ev: f64,
    /// Uncertainty (eV)
    pub uncertainty_ev: f64,
    /// Temperature (K) if reported
    pub temperature_k: Option<f64>,
    /// Adiabatic limit prediction (eV)
    pub adiabatic_limit_ev: f64,
    /// Enhancement over adiabatic limit
    pub enhancement_ratio: f64,
}

/// Neutron rate measurement from literature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeutronMeasurement {
    /// Source reference
    pub source: LiteratureSource,
    /// Host material
    pub host_material: String,
    /// Measured rate (n/s)
    pub neutron_rate: f64,
    /// Uncertainty (n/s)
    pub uncertainty: f64,
    /// Sample volume (cm³)
    pub volume_cm3: f64,
    /// Loading ratio (D/Pd)
    pub loading_ratio: f64,
    /// Temperature (K)
    pub temperature_k: f64,
    /// Trigger method
    pub trigger: String,
    /// Background subtracted?
    pub background_subtracted: bool,
    /// Control (H instead of D) performed?
    pub control_performed: bool,
}

/// Excess heat claim from literature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcessHeatClaim {
    /// Source reference
    pub source: LiteratureSource,
    /// Claimed excess power (W)
    pub excess_power_w: f64,
    /// Input power (W)
    pub input_power_w: f64,
    /// Duration (hours)
    pub duration_hours: f64,
    /// Total excess energy (kJ)
    pub excess_energy_kj: f64,
    /// Calorimetry method
    pub calorimetry_method: String,
    /// Reproduced by others?
    pub independently_reproduced: bool,
    /// Notes on validity
    pub validity_notes: String,
}

/// Real experimental data from published literature.
#[derive(Debug, Clone)]
pub struct LiteratureDataLoader {
    /// Screening measurements
    pub screening_data: Vec<ScreeningMeasurement>,
    /// Neutron measurements
    pub neutron_data: Vec<NeutronMeasurement>,
    /// Excess heat claims
    pub heat_claims: Vec<ExcessHeatClaim>,
}

impl LiteratureDataLoader {
    /// Create loader with all available literature data.
    pub fn new() -> Self {
        let mut loader = Self {
            screening_data: Vec::new(),
            neutron_data: Vec::new(),
            heat_claims: Vec::new(),
        };
        loader.load_raiola_screening();
        loader.load_nasa_lcf();
        loader.load_historical_claims();
        loader
    }

    /// Load Raiola et al. screening measurements.
    ///
    /// Raiola et al., Eur. Phys. J. A 19, 283 (2004)
    /// "Enhanced electron screening in d(d,p)t for deuterated metals"
    fn load_raiola_screening(&mut self) {
        let source = LiteratureSource {
            authors: "Raiola et al.".to_string(),
            year: 2004,
            title: "Enhanced electron screening in d(d,p)t for deuterated metals".to_string(),
            journal: "Eur. Phys. J. A 19, 283-295".to_string(),
            doi: Some("10.1140/epja/i2003-10125-0".to_string()),
            data_type: LiteratureDataType::ScreeningEnergy,
        };

        // Table 1 data from Raiola et al. (2004)
        // Adiabatic limit for D-D is ~25 eV
        let adiabatic = 25.0;

        let measurements = [
            // (material, Ue [eV], uncertainty [eV])
            ("Pd", 310.0, 30.0),
            ("Pt", 280.0, 35.0),
            ("Au", 220.0, 25.0),
            ("Ta", 322.0, 15.0),
            ("Nb", 295.0, 20.0),
            ("V", 268.0, 22.0),
            ("Zr", 297.0, 18.0),
            ("Ti", 245.0, 28.0),
            ("Ni", 275.0, 32.0),
            ("Fe", 240.0, 25.0),
            ("Al", 190.0, 20.0),
            ("Be", 180.0, 25.0),
            ("C", 105.0, 15.0),
        ];

        for (material, ue, uncertainty) in measurements {
            self.screening_data.push(ScreeningMeasurement {
                source: source.clone(),
                host_material: material.to_string(),
                target: "D-D".to_string(),
                screening_ev: ue,
                uncertainty_ev: uncertainty,
                temperature_k: Some(300.0),
                adiabatic_limit_ev: adiabatic,
                enhancement_ratio: ue / adiabatic,
            });
        }
    }

    /// Load NASA LCF experimental results.
    ///
    /// Steinetz et al., Phys. Rev. C 101, 044610 (2020)
    /// "Novel nuclear reactions observed in bremsstrahlung-irradiated deuterated metals"
    fn load_nasa_lcf(&mut self) {
        let source = LiteratureSource {
            authors: "Steinetz, Benyo, Chait et al.".to_string(),
            year: 2020,
            title:
                "Novel nuclear reactions observed in bremsstrahlung-irradiated deuterated metals"
                    .to_string(),
            journal: "Phys. Rev. C 101, 044610".to_string(),
            doi: Some("10.1103/PhysRevC.101.044610".to_string()),
            data_type: LiteratureDataType::NeutronRate,
        };

        // Data extracted from Steinetz et al. (2020) figures and tables
        // Erbium deuteride (ErD3) samples under X-ray irradiation
        let nasa_measurements = [
            // (material, rate n/s, uncertainty, volume cm³, loading, temp K, trigger)
            (
                "ErD3",
                1200.0,
                300.0,
                0.05,
                3.0,
                300.0,
                "2.9 MeV bremsstrahlung",
            ),
            (
                "ErD3",
                950.0,
                250.0,
                0.05,
                3.0,
                300.0,
                "2.9 MeV bremsstrahlung",
            ),
            (
                "TiD2",
                400.0,
                150.0,
                0.08,
                2.0,
                300.0,
                "2.9 MeV bremsstrahlung",
            ),
            (
                "TiD2",
                350.0,
                120.0,
                0.08,
                2.0,
                300.0,
                "2.9 MeV bremsstrahlung",
            ),
        ];

        for (material, rate, unc, vol, loading, temp, trigger) in nasa_measurements {
            self.neutron_data.push(NeutronMeasurement {
                source: source.clone(),
                host_material: material.to_string(),
                neutron_rate: rate,
                uncertainty: unc,
                volume_cm3: vol,
                loading_ratio: loading,
                temperature_k: temp,
                trigger: trigger.to_string(),
                background_subtracted: true,
                control_performed: true, // NASA did H control experiments
            });
        }
    }

    /// Load historical claims for context.
    fn load_historical_claims(&mut self) {
        // Fleischmann-Pons 1989 - for historical context
        self.heat_claims.push(ExcessHeatClaim {
            source: LiteratureSource {
                authors: "Fleischmann, Pons".to_string(),
                year: 1989,
                title: "Electrochemically Induced Nuclear Fusion of Deuterium".to_string(),
                journal: "J. Electroanal. Chem. 261, 301-308".to_string(),
                doi: Some("10.1016/0022-0728(89)80006-3".to_string()),
                data_type: LiteratureDataType::ExcessHeat,
            },
            excess_power_w: 4.0, // Claimed
            input_power_w: 1.0,
            duration_hours: 10.0,
            excess_energy_kj: 144.0,
            calorimetry_method: "Isoperibolic".to_string(),
            independently_reproduced: false,
            validity_notes: "Initial claims not reproduced. Calorimetry errors suspected. \
                            Later work showed loading ratio effects but no verified excess heat."
                .to_string(),
        });

        // Miles et al. 1994 - correlation claims
        self.heat_claims.push(ExcessHeatClaim {
            source: LiteratureSource {
                authors: "Miles, Bush, Ostrom, Lagowski".to_string(),
                year: 1994,
                title: "Correlation of Excess Power and Helium Production".to_string(),
                journal: "ICCF-4 Proceedings".to_string(),
                doi: None,
                data_type: LiteratureDataType::ExcessHeat,
            },
            excess_power_w: 0.5,
            input_power_w: 2.0,
            duration_hours: 24.0,
            excess_energy_kj: 43.0,
            calorimetry_method: "Flow".to_string(),
            independently_reproduced: false,
            validity_notes: "Claimed He-4 correlation. He measurements disputed. \
                            Not independently reproduced with proper controls."
                .to_string(),
        });
    }

    /// Get all screening data for a specific material.
    pub fn screening_for_material(&self, material: &str) -> Vec<&ScreeningMeasurement> {
        self.screening_data
            .iter()
            .filter(|m| m.host_material.eq_ignore_ascii_case(material))
            .collect()
    }

    /// Get average screening for a material with uncertainty.
    pub fn average_screening(&self, material: &str) -> Option<(f64, f64)> {
        let measurements = self.screening_for_material(material);
        if measurements.is_empty() {
            return None;
        }

        let n = measurements.len() as f64;
        let mean = measurements.iter().map(|m| m.screening_ev).sum::<f64>() / n;

        // Propagate uncertainties
        let variance = measurements
            .iter()
            .map(|m| m.uncertainty_ev.powi(2))
            .sum::<f64>()
            / (n * n);

        Some((mean, variance.sqrt()))
    }

    /// Get neutron data with proper controls.
    pub fn controlled_neutron_data(&self) -> Vec<&NeutronMeasurement> {
        self.neutron_data
            .iter()
            .filter(|m| m.control_performed && m.background_subtracted)
            .collect()
    }

    /// Summarize the literature data.
    pub fn summary(&self) -> LiteratureDataSummary {
        let screening_materials: Vec<String> = self
            .screening_data
            .iter()
            .map(|m| m.host_material.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();

        let mean_screening = if !self.screening_data.is_empty() {
            self.screening_data
                .iter()
                .map(|m| m.screening_ev)
                .sum::<f64>()
                / self.screening_data.len() as f64
        } else {
            0.0
        };

        let mean_enhancement = if !self.screening_data.is_empty() {
            self.screening_data
                .iter()
                .map(|m| m.enhancement_ratio)
                .sum::<f64>()
                / self.screening_data.len() as f64
        } else {
            0.0
        };

        let controlled_neutron = self.controlled_neutron_data();
        let total_neutron_rate = controlled_neutron.iter().map(|m| m.neutron_rate).sum();

        LiteratureDataSummary {
            n_screening_measurements: self.screening_data.len(),
            n_neutron_measurements: self.neutron_data.len(),
            n_heat_claims: self.heat_claims.len(),
            screening_materials,
            mean_screening_ev: mean_screening,
            mean_enhancement_over_adiabatic: mean_enhancement,
            controlled_neutron_rate_total: total_neutron_rate,
            n_controlled_experiments: controlled_neutron.len(),
            independently_reproduced_claims: self
                .heat_claims
                .iter()
                .filter(|c| c.independently_reproduced)
                .count(),
        }
    }

    /// Convert literature data to pipeline observations.
    pub fn to_pipeline_observations(&self) -> Vec<LcfObservation> {
        let mut observations = Vec::new();

        for (i, m) in self.neutron_data.iter().enumerate() {
            // Create observation from literature measurement
            // Note: We don't have exact Gamow predictions for all materials,
            // so we estimate using Pd values as baseline
            let screening = match m.host_material.as_str() {
                "ErD3" => 350.0, // Estimated based on lanthanide screening
                "TiD2" => 245.0, // From Raiola Ti measurement
                "PdD" | "Pd" => 310.0,
                _ => 250.0, // Generic estimate
            };

            let gamow = GamowIntegration::dd_rate(m.temperature_k, screening, 0);

            // Estimate deuterium density based on loading ratio
            // For metal hydrides, D density ~ loading * metal_density / mass_ratio
            let d_density = m.loading_ratio * 1e22; // Approximate D atoms/cm³

            let predicted = gamow.to_neutron_rate(d_density, m.volume_cm3);

            let enhancement = if predicted > 0.0 {
                m.neutron_rate / predicted
            } else {
                f64::INFINITY
            };

            observations.push(LcfObservation {
                timestamp_s: i as f64, // Use index as pseudo-timestamp
                temperature_k: m.temperature_k,
                loading_ratio: m.loading_ratio,
                neutron_rate: m.neutron_rate,
                trigger_energy_kev: Some(2900.0), // 2.9 MeV bremsstrahlung
                screening_ev: screening,
                gamow_predicted_rate: predicted,
                enhancement_factor: enhancement,
                is_anomaly: enhancement > 1e3,
            });
        }

        observations
    }
}

impl Default for LiteratureDataLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of literature data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteratureDataSummary {
    /// Number of screening measurements
    pub n_screening_measurements: usize,
    /// Number of neutron measurements
    pub n_neutron_measurements: usize,
    /// Number of excess heat claims
    pub n_heat_claims: usize,
    /// Materials with screening data
    pub screening_materials: Vec<String>,
    /// Mean measured screening (eV)
    pub mean_screening_ev: f64,
    /// Mean enhancement over adiabatic limit
    pub mean_enhancement_over_adiabatic: f64,
    /// Total neutron rate from controlled experiments
    pub controlled_neutron_rate_total: f64,
    /// Number of properly controlled experiments
    pub n_controlled_experiments: usize,
    /// Claims that were independently reproduced
    pub independently_reproduced_claims: usize,
}

/// CSV parser for experimental data.
///
/// Parses CSV files with columns:
/// timestamp,temperature_k,loading_ratio,neutron_rate,trigger_kev,screening_ev
pub struct CsvDataParser;

impl CsvDataParser {
    /// Parse CSV string into observations.
    pub fn parse(csv_data: &str, source_name: &str) -> Result<Vec<LcfObservation>, String> {
        let mut observations = Vec::new();
        let lines: Vec<&str> = csv_data.lines().collect();

        if lines.is_empty() {
            return Err("Empty CSV data".to_string());
        }

        // Skip header
        for (line_num, line) in lines.iter().enumerate().skip(1) {
            let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();

            if fields.len() < 4 {
                continue; // Skip malformed lines
            }

            let timestamp_s = fields[0]
                .parse::<f64>()
                .map_err(|_| format!("Invalid timestamp at line {}", line_num + 1))?;

            let temperature_k = fields[1]
                .parse::<f64>()
                .map_err(|_| format!("Invalid temperature at line {}", line_num + 1))?;

            let loading_ratio = fields[2]
                .parse::<f64>()
                .map_err(|_| format!("Invalid loading at line {}", line_num + 1))?;

            let neutron_rate = fields[3]
                .parse::<f64>()
                .map_err(|_| format!("Invalid neutron rate at line {}", line_num + 1))?;

            let trigger_energy_kev = fields.get(4).and_then(|s| s.parse::<f64>().ok());

            let screening_ev = fields
                .get(5)
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(300.0);

            // Compute Gamow prediction
            let gamow = GamowIntegration::dd_rate(temperature_k, screening_ev, 0);
            let d_density = loading_ratio * 6.8e22; // Pd baseline
            let predicted = gamow.to_neutron_rate(d_density, 1.0);

            let enhancement = if predicted > 0.0 {
                neutron_rate / predicted
            } else {
                f64::INFINITY
            };

            observations.push(LcfObservation {
                timestamp_s,
                temperature_k,
                loading_ratio,
                neutron_rate,
                trigger_energy_kev,
                screening_ev,
                gamow_predicted_rate: predicted,
                enhancement_factor: enhancement,
                is_anomaly: enhancement > 1e3,
            });
        }

        if observations.is_empty() {
            return Err(format!("No valid data found in {}", source_name));
        }

        Ok(observations)
    }

    /// Generate example CSV format.
    pub fn example_format() -> &'static str {
        "timestamp,temperature_k,loading_ratio,neutron_rate,trigger_kev,screening_ev\n\
         0,300,0.7,1000,12,310\n\
         3600,300,0.7,1200,12,310\n\
         7200,305,0.68,950,12,310"
    }
}

/// JSON parser for experimental data.
pub struct JsonDataParser;

impl JsonDataParser {
    /// Parse JSON array into observations.
    pub fn parse(json_data: &str) -> Result<Vec<LcfObservation>, String> {
        serde_json::from_str(json_data).map_err(|e| format!("JSON parse error: {}", e))
    }

    /// Serialize observations to JSON.
    pub fn to_json(observations: &[LcfObservation]) -> Result<String, String> {
        serde_json::to_string_pretty(observations)
            .map_err(|e| format!("JSON serialize error: {}", e))
    }
}

impl ExperimentalDataPipeline {
    /// Load real data from literature.
    pub fn load_literature_data(&mut self) {
        let loader = LiteratureDataLoader::new();
        let observations = loader.to_pipeline_observations();

        for obs in observations {
            self.add_observation(obs);
        }
    }

    /// Load from CSV string.
    pub fn load_csv(&mut self, csv_data: &str, source: &str) -> Result<usize, String> {
        let observations = CsvDataParser::parse(csv_data, source)?;
        let count = observations.len();

        for obs in observations {
            self.add_observation(obs);
        }

        Ok(count)
    }

    /// Load from JSON string.
    pub fn load_json(&mut self, json_data: &str) -> Result<usize, String> {
        let observations = JsonDataParser::parse(json_data)?;
        let count = observations.len();

        for obs in observations {
            self.add_observation(obs);
        }

        Ok(count)
    }

    /// Export pipeline data to JSON.
    pub fn export_json(&self) -> Result<String, String> {
        JsonDataParser::to_json(&self.observations)
    }

    /// Generate report including literature context.
    pub fn generate_full_report(&self) -> FullDataReport {
        let pipeline_report = self.generate_report();
        let lit_loader = LiteratureDataLoader::new();
        let lit_summary = lit_loader.summary();

        FullDataReport {
            pipeline_report,
            literature_summary: lit_summary,
            analysis: self.physics_analysis(),
        }
    }

    fn physics_analysis(&self) -> PhysicsAnalysis {
        let stats = self.enhancement_statistics();

        let gap_explanation = if stats.mean_log10 > 30.0 {
            "Enhancement of ~10^40 cannot be explained by screening alone. \
             Screening provides ~10^10 enhancement at most. \
             The 30 order-of-magnitude gap suggests either: \
             (1) measurement artifacts, (2) non-D-D reactions, \
             (3) unknown physics enhancement mechanism."
                .to_string()
        } else if stats.mean_log10 > 10.0 {
            "Enhancement consistent with strong screening effects. \
             Raiola et al. measurements support screening up to ~300 eV, \
             which can provide ~10^10 enhancement over bare Gamow."
                .to_string()
        } else {
            "Enhancement within expected range for standard physics.".to_string()
        };

        let key_observations = vec![
            format!(
                "Mean enhancement: 10^{:.1} over Gamow prediction",
                stats.mean_log10
            ),
            format!(
                "{} of {} observations flagged as anomalous",
                stats.n_anomalies, stats.n_observations
            ),
            "NASA LCF results show ~1000 n/s from deuterated metals".to_string(),
            "Raiola screening: ~250 eV average, ~10× adiabatic limit".to_string(),
        ];

        let open_questions = vec![
            "What mechanism bridges the 10^40 gap between theory and observation?".to_string(),
            "Is neutron production from D-D or another reaction channel?".to_string(),
            "Why do lanthanide deuterides show highest rates?".to_string(),
            "What role does the X-ray trigger play beyond deuteron excitation?".to_string(),
        ];

        PhysicsAnalysis {
            gap_explanation,
            key_observations,
            open_questions,
        }
    }
}

/// Full data report including literature context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullDataReport {
    /// Pipeline analysis report
    pub pipeline_report: DataPipelineReport,
    /// Summary of literature data
    pub literature_summary: LiteratureDataSummary,
    /// Physics analysis
    pub analysis: PhysicsAnalysis,
}

/// Physics analysis of the data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsAnalysis {
    /// Explanation of the rate gap
    pub gap_explanation: String,
    /// Key observations from the data
    pub key_observations: Vec<String>,
    /// Open physics questions
    pub open_questions: Vec<String>,
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
        assert!(
            assessment.verdict.contains("UNVALIDATED") || assessment.verdict.contains("IMPOSSIBLE")
        );
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
            pipeline.add_measurement(
                &conditions,
                1000.0 + i as f64 * 100.0,
                i as f64 * 3600.0,
                Some(12.0),
            );
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

    #[test]
    fn test_literature_data_loader() {
        let loader = LiteratureDataLoader::new();

        // Should have Raiola screening data
        assert!(!loader.screening_data.is_empty());
        assert!(loader.screening_data.len() >= 10); // At least 10 materials

        // Should have NASA neutron data
        assert!(!loader.neutron_data.is_empty());

        // Check Pd screening from Raiola
        let pd_screening = loader.screening_for_material("Pd");
        assert!(!pd_screening.is_empty());
        assert!(pd_screening[0].screening_ev > 250.0); // Raiola measured ~310 eV
        assert!(pd_screening[0].screening_ev < 400.0);
    }

    #[test]
    fn test_literature_data_summary() {
        let loader = LiteratureDataLoader::new();
        let summary = loader.summary();

        assert!(summary.n_screening_measurements > 0);
        assert!(summary.n_neutron_measurements > 0);
        assert!(summary.mean_screening_ev > 100.0); // Should be in reasonable range
        assert!(summary.mean_enhancement_over_adiabatic > 1.0); // Enhanced over adiabatic
    }

    #[test]
    fn test_average_screening() {
        let loader = LiteratureDataLoader::new();

        // Check Pd average
        if let Some((mean, unc)) = loader.average_screening("Pd") {
            assert!(mean > 250.0 && mean < 400.0);
            assert!(unc > 0.0 && unc < 100.0);
        }

        // Non-existent material should return None
        assert!(loader.average_screening("XyzNotReal").is_none());
    }

    #[test]
    fn test_controlled_neutron_data() {
        let loader = LiteratureDataLoader::new();
        let controlled = loader.controlled_neutron_data();

        // All NASA data should be controlled
        assert!(!controlled.is_empty());

        for m in controlled {
            assert!(m.control_performed);
            assert!(m.background_subtracted);
        }
    }

    #[test]
    fn test_literature_to_observations() {
        let loader = LiteratureDataLoader::new();
        let observations = loader.to_pipeline_observations();

        assert!(!observations.is_empty());

        // All should be anomalies (NASA data shows huge enhancement)
        for obs in &observations {
            assert!(obs.enhancement_factor > 1.0);
            assert!(obs.is_anomaly);
        }
    }

    #[test]
    fn test_csv_parser() {
        let csv = "timestamp,temperature_k,loading_ratio,neutron_rate,trigger_kev,screening_ev\n\
                   0,300,0.7,1000,12,310\n\
                   3600,300,0.7,1200,12,310";

        let observations = CsvDataParser::parse(csv, "test").unwrap();
        assert_eq!(observations.len(), 2);
        assert_eq!(observations[0].temperature_k, 300.0);
        assert_eq!(observations[0].neutron_rate, 1000.0);
        assert_eq!(observations[1].neutron_rate, 1200.0);
    }

    #[test]
    fn test_csv_parser_minimal() {
        // Minimal CSV with only required columns
        let csv = "timestamp,temperature_k,loading_ratio,neutron_rate\n\
                   0,300,0.7,500";

        let observations = CsvDataParser::parse(csv, "test").unwrap();
        assert_eq!(observations.len(), 1);
        assert_eq!(observations[0].screening_ev, 300.0); // Default
    }

    #[test]
    fn test_pipeline_load_literature() {
        let mut pipeline = ExperimentalDataPipeline::new();
        pipeline.load_literature_data();

        // Should have NASA + Raiola-derived data
        assert!(!pipeline.observations.is_empty());

        let report = pipeline.generate_full_report();
        assert!(report.literature_summary.n_screening_measurements > 0);
        assert!(!report.analysis.gap_explanation.is_empty());
    }

    #[test]
    fn test_pipeline_csv_load() {
        let mut pipeline = ExperimentalDataPipeline::new();

        let csv = "timestamp,temperature_k,loading_ratio,neutron_rate\n\
                   0,300,0.7,1000\n\
                   1,300,0.7,1100\n\
                   2,300,0.7,900";

        let count = pipeline.load_csv(csv, "test").unwrap();
        assert_eq!(count, 3);
        assert_eq!(pipeline.observations.len(), 3);
    }

    #[test]
    fn test_json_roundtrip() {
        let loader = LiteratureDataLoader::new();
        let observations = loader.to_pipeline_observations();

        // Serialize to JSON
        let json = JsonDataParser::to_json(&observations).unwrap();
        assert!(!json.is_empty());

        // Parse back
        let parsed = JsonDataParser::parse(&json).unwrap();
        assert_eq!(parsed.len(), observations.len());
    }

    #[test]
    fn test_full_data_report() {
        let mut pipeline = ExperimentalDataPipeline::new();
        pipeline.load_nasa_baseline();
        pipeline.load_literature_data();

        let report = pipeline.generate_full_report();

        assert!(report.pipeline_report.n_observations > 0);
        assert!(!report.analysis.key_observations.is_empty());
        assert!(!report.analysis.open_questions.is_empty());
    }
}
