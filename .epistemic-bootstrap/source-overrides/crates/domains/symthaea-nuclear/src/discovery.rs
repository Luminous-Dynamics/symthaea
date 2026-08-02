// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Nuclear Discovery Engine — finding where new physics hides.
//!
//! The key insight: regions where our model is *most uncertain* are where
//! discoveries are most likely. This module:
//!
//! 1. Calibrates SEMF + shell model against known binding energies
//! 2. Computes a residual- and distance-based uncertainty scale for each isotope
//! 3. Flags anomalies where prediction deviates from neighbors by >2σ
//! 4. Identifies "discovery candidates" — isotopes in uncertain regions
//!    that show enhanced stability relative to model expectations
//!
//! The reported `sigma` is an explicit heuristic model-error scale, not a
//! calibrated Bayesian posterior standard deviation or confidence interval.
//!
//! Reference: Bayesian nuclear mass predictions — Neufcourt et al. (2019).

use crate::mass_formula::SemiEmpiricalMassFormula;
use crate::shell_model::ShellModel;
use serde::{Deserialize, Serialize};

const Z_CORRELATION_LENGTH: f64 = 10.0;
const N_CORRELATION_LENGTH: f64 = 15.0;
const MIN_SIGMA_MEV: f64 = 1.0;
const UNCALIBRATED_SIGMA_MEV: f64 = 15.0;
const MIN_WEIGHT_SUM: f64 = 1.0e-12;

/// A known nuclear binding energy (from AME2020 or experiment).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasuredNucleus {
    pub z: u16,
    pub n: u16,
    pub binding_energy_mev: f64,
    /// Whether this is measured (true) or extrapolated (false)
    pub is_measured: bool,
}

/// Prediction with uncertainty for a single isotope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionWithUncertainty {
    pub z: u16,
    pub n: u16,
    pub a: u16,
    /// SEMF binding energy prediction (MeV)
    pub predicted_be: f64,
    /// Shell correction (MeV)
    pub shell_correction: f64,
    /// Total predicted binding energy (MeV)
    pub total_predicted: f64,
    /// Measured binding energy if available (MeV)
    pub measured_be: Option<f64>,
    /// Residual: measured - predicted (MeV), if measured
    pub residual: Option<f64>,
    /// Heuristic model-error scale (MeV), not a calibrated posterior σ.
    pub sigma: f64,
    /// Anomaly score: |residual|/σ (only if measured)
    pub anomaly_score: Option<f64>,
    /// Discovery interest: how interesting is this isotope for new physics?
    /// High when: σ is large AND shell correction is negative AND no measurement exists
    pub discovery_interest: f64,
}

/// Discovery candidate — an isotope flagged for potential new physics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryCandidate {
    pub z: u16,
    pub n: u16,
    pub a: u16,
    /// Why this isotope is interesting
    pub reason: String,
    /// Discovery interest score (0-1)
    pub interest_score: f64,
    /// Model uncertainty at this point (MeV)
    pub sigma: f64,
    /// Shell correction (MeV)
    pub shell_correction: f64,
    /// Predicted alpha-decay half-life if available
    pub predicted_half_life: Option<f64>,
}

/// Full discovery report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NuclearDiscoveryReport {
    /// All predictions with uncertainty
    pub predictions: Vec<PredictionWithUncertainty>,
    /// Top discovery candidates
    pub candidates: Vec<DiscoveryCandidate>,
    /// Model calibration statistics
    pub calibration: ModelCalibration,
    /// Z range analyzed
    pub z_range: (u16, u16),
    /// N range analyzed
    pub n_range: (u16, u16),
}

/// Model calibration statistics from comparison with known data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCalibration {
    /// Number of isotopes with measured binding energies used
    pub n_calibration: usize,
    /// Mean residual (MeV) — systematic bias
    pub mean_residual: f64,
    /// RMS residual (MeV) — overall accuracy
    pub rms_residual: f64,
    /// Maximum absolute residual (MeV)
    pub max_residual: f64,
    /// Fraction of predictions within 1σ of measurement
    pub fraction_within_1sigma: f64,
}

/// The nuclear discovery engine.
pub struct NuclearDiscoveryEngine {
    semf: SemiEmpiricalMassFormula,
    shell: ShellModel,
    known_nuclei: Vec<MeasuredNucleus>,
}

impl NuclearDiscoveryEngine {
    /// Create with default models and no calibration data.
    pub fn new() -> Self {
        Self {
            semf: SemiEmpiricalMassFormula::default(),
            shell: ShellModel::default(),
            known_nuclei: Vec::new(),
        }
    }

    /// Add known binding energies for calibration.
    pub fn add_known_nuclei(&mut self, nuclei: Vec<MeasuredNucleus>) {
        self.known_nuclei.extend(nuclei);
    }

    /// Add AME2020 reference nuclei for calibration (~70 nuclei).
    ///
    /// Uses the curated AME2020/NUBASE2020 dataset spanning H-2 to Og-294.
    /// Includes all doubly-magic nuclei, iron peak, beta-stable at A≈20 intervals,
    /// benchmark actinides, and all synthesized superheavy elements.
    pub fn add_reference_nuclei(&mut self) {
        self.add_known_nuclei(crate::ame2020::ame2020_reference_nuclei());
    }

    /// Compute prediction with uncertainty for a single isotope.
    fn predict(&self, z: u16, n: u16) -> PredictionWithUncertainty {
        let a = z + n;
        let predicted_be = self.semf.binding_energy(a, z);
        let shell_corr = self.shell.shell_correction_energy(a, z);
        let total = predicted_be + shell_corr;

        // Find measured value if available
        let measured = self
            .known_nuclei
            .iter()
            .find(|m| m.is_measured && m.z == z && m.n == n);
        let measured_be = measured.map(|m| m.binding_energy_mev);
        let residual = measured_be.map(|m| m - total);

        // Estimate uncertainty from measured residuals, calibration density, and
        // distance beyond the measured domain. Exact-target observations are
        // excluded to avoid using a residual to certify itself.
        let sigma = self.estimate_local_sigma(z, n);

        let anomaly_score = residual.map(|r| r.abs() / sigma.max(0.1));

        // Discovery interest: high when uncertain AND shell-enhanced AND unmeasured
        let unmeasured_bonus = if measured_be.is_none() { 1.0 } else { 0.3 };
        let shell_bonus = if shell_corr < -3.0 {
            (-shell_corr / 10.0).min(1.0)
        } else {
            0.0
        };
        let uncertainty_bonus = (sigma / 10.0).min(1.0);
        let discovery_interest =
            (unmeasured_bonus * 0.4 + shell_bonus * 0.35 + uncertainty_bonus * 0.25).min(1.0);

        PredictionWithUncertainty {
            z,
            n,
            a,
            predicted_be,
            shell_correction: shell_corr,
            total_predicted: total,
            measured_be,
            residual,
            sigma,
            anomaly_score,
            discovery_interest,
        }
    }

    /// Estimate a conservative model-error scale at `(Z, N)`.
    ///
    /// The estimate combines three independently visible contributions:
    ///
    /// - distance-weighted RMS residuals in the measured calibration set,
    /// - a sparsity penalty when little calibration weight is nearby, and
    /// - an extrapolation penalty that grows monotonically with distance to the
    ///   nearest measured nucleus.
    ///
    /// Extrapolated reference values never calibrate the estimator. If no measured
    /// calibration data exist at all, the result is the explicit uncalibrated prior
    /// `UNCALIBRATED_SIGMA_MEV` rather than a claim inferred from absent evidence.
    fn estimate_local_sigma(&self, z: u16, n: u16) -> f64 {
        let residuals: Vec<(f64, f64)> = self
            .known_nuclei
            .iter()
            .filter(|m| m.is_measured)
            .filter(|m| m.z != z || m.n != n)
            .filter_map(|m| {
                let a = m.z + m.n;
                let prediction = self.semf.binding_energy(a, m.z)
                    + self.shell.shell_correction_energy(a, m.z);
                let residual = m.binding_energy_mev - prediction;
                if !residual.is_finite() {
                    return None;
                }

                let dz = f64::from(m.z.abs_diff(z)) / Z_CORRELATION_LENGTH;
                let dn = f64::from(m.n.abs_diff(n)) / N_CORRELATION_LENGTH;
                let scaled_distance = (dz.mul_add(dz, dn * dn)).sqrt();
                Some((residual, scaled_distance))
            })
            .collect();

        if residuals.is_empty() {
            return UNCALIBRATED_SIGMA_MEV;
        }

        let global_mse = residuals
            .iter()
            .map(|(residual, _)| residual.powi(2))
            .sum::<f64>()
            / residuals.len() as f64;
        let global_rms = global_mse.sqrt().max(MIN_SIGMA_MEV);
        let residual_envelope = residuals
            .iter()
            .map(|(residual, _)| residual.abs())
            .fold(global_rms, f64::max);
        let nearest_distance = residuals
            .iter()
            .map(|(_, distance)| *distance)
            .fold(f64::INFINITY, f64::min);

        let (weighted_square_error, weight_sum) = residuals.iter().fold(
            (0.0, 0.0),
            |(weighted_error, total_weight), (residual, distance)| {
                let weight = (-0.5 * distance.powi(2)).exp();
                (
                    weighted_error + weight * residual.powi(2),
                    total_weight + weight,
                )
            },
        );

        let local_rms = if weight_sum > MIN_WEIGHT_SUM {
            (weighted_square_error / weight_sum).sqrt()
        } else {
            global_rms
        };
        let density_penalty = global_rms / weight_sum.max(1.0).sqrt();
        let extrapolation_penalty = residual_envelope * nearest_distance;

        local_rms
            .powi(2)
            .mul_add(
                1.0,
                density_penalty
                    .powi(2)
                    .mul_add(1.0, extrapolation_penalty.powi(2)),
            )
            .sqrt()
            .max(MIN_SIGMA_MEV)
    }

    /// Run the full discovery analysis on a region of the nuclear chart.
    pub fn discover(
        &self,
        z_min: u16,
        z_max: u16,
        n_min: u16,
        n_max: u16,
    ) -> NuclearDiscoveryReport {
        // Generate all predictions
        let mut predictions = Vec::new();
        for z in z_min..=z_max {
            for n in n_min..=n_max {
                predictions.push(self.predict(z, n));
            }
        }

        // Calibration statistics
        let calibrated: Vec<_> = predictions
            .iter()
            .filter(|p| p.residual.is_some())
            .collect();
        let n_cal = calibrated.len();
        let mean_res = if n_cal > 0 {
            calibrated.iter().map(|p| p.residual.unwrap()).sum::<f64>() / n_cal as f64
        } else {
            0.0
        };
        let rms_res = if n_cal > 0 {
            (calibrated
                .iter()
                .map(|p| p.residual.unwrap().powi(2))
                .sum::<f64>()
                / n_cal as f64)
                .sqrt()
        } else {
            0.0
        };
        let max_res = calibrated
            .iter()
            .map(|p| p.residual.unwrap().abs())
            .fold(0.0f64, f64::max);
        let within_1sigma = if n_cal > 0 {
            calibrated
                .iter()
                .filter(|p| p.anomaly_score.map(|a| a < 1.0).unwrap_or(false))
                .count() as f64
                / n_cal as f64
        } else {
            0.0
        };

        let calibration = ModelCalibration {
            n_calibration: n_cal,
            mean_residual: mean_res,
            rms_residual: rms_res,
            max_residual: max_res,
            fraction_within_1sigma: within_1sigma,
        };

        // Find discovery candidates (top-20 by interest score)
        let mut candidates: Vec<DiscoveryCandidate> = predictions
            .iter()
            .filter(|p| p.discovery_interest > 0.5)
            .map(|p| {
                let reason = if p.measured_be.is_none() && p.shell_correction < -3.0 {
                    format!(
                        "Unmeasured, strong shell enhancement ({:.1} MeV), high uncertainty (σ={:.1} MeV)",
                        p.shell_correction, p.sigma
                    )
                } else if p.measured_be.is_none() {
                    format!("Unmeasured region, σ={:.1} MeV", p.sigma)
                } else if p.anomaly_score.map(|a| a > 2.0).unwrap_or(false) {
                    format!(
                        "Anomalous: measured deviates by {:.1}σ from prediction",
                        p.anomaly_score.unwrap()
                    )
                } else {
                    "Moderate interest".to_string()
                };

                DiscoveryCandidate {
                    z: p.z,
                    n: p.n,
                    a: p.a,
                    reason,
                    interest_score: p.discovery_interest,
                    sigma: p.sigma,
                    shell_correction: p.shell_correction,
                    predicted_half_life: self.semf.geiger_nuttall_half_life(p.a, p.z),
                }
            })
            .collect();

        candidates.sort_by(|a, b| {
            b.interest_score
                .partial_cmp(&a.interest_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(20);

        NuclearDiscoveryReport {
            predictions,
            candidates,
            calibration,
            z_range: (z_min, z_max),
            n_range: (n_min, n_max),
        }
    }

    /// Print a discovery report summary.
    pub fn print_report(report: &NuclearDiscoveryReport) -> String {
        let mut out = String::new();
        out.push_str("=== Nuclear Discovery Report ===\n");
        out.push_str(&format!(
            "Region: Z={}-{}, N={}-{} ({} isotopes)\n",
            report.z_range.0,
            report.z_range.1,
            report.n_range.0,
            report.n_range.1,
            report.predictions.len()
        ));
        out.push_str(&format!(
            "Calibration: {} known nuclei, RMS={:.2} MeV, {:.0}% within 1σ\n\n",
            report.calibration.n_calibration,
            report.calibration.rms_residual,
            report.calibration.fraction_within_1sigma * 100.0
        ));

        out.push_str(&format!(
            "Top {} Discovery Candidates:\n",
            report.candidates.len()
        ));
        out.push_str("  Z   N    A  | Interest | σ (MeV) | Shell (MeV) | Reason\n");
        out.push_str("------|----------|---------|-------------|-------\n");

        for c in &report.candidates {
            out.push_str(&format!(
                "{:>3} {:>3} {:>4} |   {:.2}   |  {:>5.1}  |    {:>7.1}  | {}\n",
                c.z, c.n, c.a, c.interest_score, c.sigma, c.shell_correction, c.reason
            ));
        }

        out
    }
}

impl Default for NuclearDiscoveryEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_measurement(
        engine: &NuclearDiscoveryEngine,
        z: u16,
        n: u16,
        residual_mev: f64,
        is_measured: bool,
    ) -> MeasuredNucleus {
        let a = z + n;
        let prediction = engine.semf.binding_energy(a, z)
            + engine.shell.shell_correction_energy(a, z);
        MeasuredNucleus {
            z,
            n,
            binding_energy_mev: prediction + residual_mev,
            is_measured,
        }
    }

    #[test]
    fn test_discovery_with_reference_nuclei() {
        let mut engine = NuclearDiscoveryEngine::new();
        engine.add_reference_nuclei();

        // Run discovery on a small region
        let report = engine.discover(110, 116, 175, 185);

        assert!(!report.predictions.is_empty());
        assert!(report.calibration.n_calibration > 0);
        assert!(report.calibration.rms_residual > 0.0);
    }

    #[test]
    fn test_uncertainty_varies_with_data_density() {
        let mut engine = NuclearDiscoveryEngine::new();
        engine.add_reference_nuclei();

        // Near Pb-208 — dense data region (many nearby calibration nuclei)
        let near = engine.predict(82, 126);
        // Far from any data (Z=130, N=220) — well beyond synthesized elements
        let far = engine.predict(130, 220);

        // Both should have finite, positive uncertainty
        assert!(
            near.sigma > 0.0 && near.sigma.is_finite(),
            "Near σ={}",
            near.sigma
        );
        assert!(
            far.sigma > 0.0 && far.sigma.is_finite(),
            "Far σ={}",
            far.sigma
        );

        // The well-measured region should have lower uncertainty than extrapolation
        assert!(
            far.sigma > near.sigma,
            "Far-from-data σ ({}) should be > near-data σ ({})",
            far.sigma,
            near.sigma
        );
    }

    #[test]
    fn test_uncertainty_is_invariant_to_calibration_order() {
        let mut forward = NuclearDiscoveryEngine::new();
        forward.add_reference_nuclei();

        let mut reversed_data = crate::ame2020::ame2020_reference_nuclei();
        reversed_data.reverse();
        let mut reversed = NuclearDiscoveryEngine::new();
        reversed.add_known_nuclei(reversed_data);

        let forward_sigma = forward.predict(120, 184).sigma;
        let reversed_sigma = reversed.predict(120, 184).sigma;
        assert!(
            (forward_sigma - reversed_sigma).abs() <= 1.0e-10,
            "calibration order changed σ: forward={forward_sigma}, reversed={reversed_sigma}"
        );
    }

    #[test]
    fn test_extrapolated_references_do_not_calibrate_uncertainty() {
        let mut engine = NuclearDiscoveryEngine::new();
        let measured = synthetic_measurement(&engine, 82, 126, 2.0, true);
        let extrapolated = synthetic_measurement(&engine, 130, 220, 0.0, false);
        engine.add_known_nuclei(vec![measured, extrapolated]);

        let far = engine.predict(130, 220);
        assert!(
            far.sigma >= UNCALIBRATED_SIGMA_MEV,
            "extrapolated references must not suppress uncertainty: σ={}",
            far.sigma
        );
    }

    #[test]
    fn test_uncalibrated_prior_is_explicit() {
        let engine = NuclearDiscoveryEngine::new();
        let prediction = engine.predict(82, 126);
        assert_eq!(prediction.sigma, UNCALIBRATED_SIGMA_MEV);
    }

    #[test]
    fn test_discovery_candidates_sorted() {
        let mut engine = NuclearDiscoveryEngine::new();
        engine.add_reference_nuclei();

        let report = engine.discover(110, 120, 170, 190);

        // Candidates should be sorted by interest (descending)
        for i in 1..report.candidates.len() {
            assert!(
                report.candidates[i - 1].interest_score >= report.candidates[i].interest_score,
                "Candidates not sorted at position {}: {} < {}",
                i,
                report.candidates[i - 1].interest_score,
                report.candidates[i].interest_score
            );
        }
    }

    #[test]
    fn test_unmeasured_isotopes_have_higher_interest() {
        let mut engine = NuclearDiscoveryEngine::new();
        engine.add_reference_nuclei();

        // Pb-208 is measured → lower discovery interest
        let pb208 = engine.predict(82, 126);
        // Element 120, N=184 is unmeasured → higher interest
        let e120 = engine.predict(120, 184);

        assert!(
            e120.discovery_interest > pb208.discovery_interest,
            "Unmeasured should have higher interest: E120={}, Pb208={}",
            e120.discovery_interest,
            pb208.discovery_interest
        );
    }

    #[test]
    fn test_calibration_statistics() {
        let mut engine = NuclearDiscoveryEngine::new();
        engine.add_reference_nuclei();

        // Use a focused region near heavy nuclei where SEMF is most accurate
        let report = engine.discover(75, 95, 110, 155);

        assert!(
            report.calibration.n_calibration >= 2,
            "Should calibrate against at least 2 nuclei, got {}",
            report.calibration.n_calibration
        );

        // RMS should be finite and positive
        assert!(
            report.calibration.rms_residual > 0.0 && report.calibration.rms_residual.is_finite(),
            "RMS residual should be positive and finite: {}",
            report.calibration.rms_residual
        );
    }
}
