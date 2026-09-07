// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Nuclear residual exploration over simple SEMF + shell-model predictions.
//!
//! This module is a hypothesis-generation surface, not a discovery authority.
//! Local residual spread is reported as an empirical dispersion **proxy**, not a
//! calibrated predictive uncertainty. Large residual ratios are anomaly-ranking
//! proxies, not significance tests. Unmeasured/high-priority coordinates are
//! exploration candidates, not evidence that a nucleus exists or is stable.
//!
//! Only entries explicitly marked `is_measured=true` may enter empirical residual
//! calibration. AME estimated/extrapolated entries are never treated as measured
//! evidence by this module.

use crate::mass_formula::SemiEmpiricalMassFormula;
use crate::shell_model::ShellModel;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

/// A nuclear binding-energy datum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasuredNucleus {
    pub z: u16,
    pub n: u16,
    pub binding_energy_mev: f64,
    /// True only when the source classifies this value as measured rather than
    /// extrapolated/estimated.
    pub is_measured: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ExplorationConfig {
    pub local_radius_z: u16,
    pub local_radius_n: u16,
    pub min_local_measured_points: usize,
    /// Fallback residual-dispersion proxy when local measured support is sparse.
    pub sparse_support_dispersion_proxy_mev: f64,
    /// Numerical floor for standardized-residual proxy calculations.
    pub dispersion_floor_mev: f64,
    /// Scale used only to normalize the dispersion component of the heuristic
    /// exploration-priority score.
    pub priority_dispersion_scale_mev: f64,
    pub candidate_priority_threshold: f64,
    pub max_candidates: usize,
}

impl Default for ExplorationConfig {
    fn default() -> Self {
        Self {
            local_radius_z: 10,
            local_radius_n: 15,
            min_local_measured_points: 2,
            sparse_support_dispersion_proxy_mev: 15.0,
            dispersion_floor_mev: 1.0,
            priority_dispersion_scale_mev: 10.0,
            candidate_priority_threshold: 0.5,
            max_candidates: 20,
        }
    }
}

impl ExplorationConfig {
    fn validate(self) -> Result<(), ExplorationError> {
        if self.local_radius_z == 0 && self.local_radius_n == 0 {
            return Err(ExplorationError::EmptyLocalWindow);
        }
        if self.min_local_measured_points == 0 || self.max_candidates == 0 {
            return Err(ExplorationError::InvalidCountConfig);
        }
        for value in [
            self.sparse_support_dispersion_proxy_mev,
            self.dispersion_floor_mev,
            self.priority_dispersion_scale_mev,
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(ExplorationError::InvalidPositiveScale);
            }
        }
        if !self.candidate_priority_threshold.is_finite()
            || !(0.0..=1.0).contains(&self.candidate_priority_threshold)
        {
            return Err(ExplorationError::InvalidPriorityThreshold);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ExplorationCalibrationDatum {
    pub z: u16,
    pub n: u16,
    pub binding_energy_mev_bits: u64,
}

impl From<&MeasuredNucleus> for ExplorationCalibrationDatum {
    fn from(nucleus: &MeasuredNucleus) -> Self {
        Self {
            z: nucleus.z,
            n: nucleus.n,
            binding_energy_mev_bits: nucleus.binding_energy_mev.to_bits(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExplorationCalibrationReceipt {
    pub config: ExplorationConfig,
    pub exact_measured_data: Vec<ExplorationCalibrationDatum>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExplorationInterpretation {
    HeuristicExplorationOnly,
}

/// One SEMF + shell-model prediction with empirical residual diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationPrediction {
    pub z: u16,
    pub n: u16,
    pub a: u16,
    pub semf_binding_energy_mev: f64,
    pub shell_correction_mev: f64,
    pub total_predicted_binding_energy_mev: f64,
    pub measured_binding_energy_mev: Option<f64>,
    pub residual_mev: Option<f64>,
    /// Population spread of nearby measured residuals, subject to the declared
    /// sparse-support fallback and floor. The target coordinate itself is always
    /// excluded from this support set. This is not a predictive posterior SD.
    pub local_residual_dispersion_proxy_mev: f64,
    pub local_measured_support_count: usize,
    /// |residual| / local residual-dispersion proxy when a measurement exists.
    /// This is a ranking diagnostic, not a calibrated z-score or p-value.
    pub residual_to_local_dispersion_ratio_proxy: Option<f64>,
    /// Transparent 0..1 hypothesis-prioritization heuristic.
    pub exploration_priority_proxy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplorationCandidate {
    pub z: u16,
    pub n: u16,
    pub a: u16,
    pub reason: String,
    pub exploration_priority_proxy: f64,
    pub local_residual_dispersion_proxy_mev: f64,
    pub local_measured_support_count: usize,
    pub shell_correction_mev: f64,
    /// SEMF Geiger-Nuttall output when the simple model returns one. This is an
    /// exploratory model output, not a qualified lifetime prediction.
    pub semf_geiger_nuttall_half_life_proxy_seconds: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualFitSummary {
    pub n_measured_comparisons: usize,
    pub mean_residual_mev: f64,
    pub rms_residual_mev: f64,
    pub max_abs_residual_mev: f64,
    /// Fraction whose |residual| is no larger than the leave-one-coordinate-out
    /// local residual spread proxy. This is deliberately not named coverage or
    /// "within 1 sigma".
    pub fraction_within_local_dispersion_proxy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NuclearExplorationReport {
    pub predictions: Vec<ExplorationPrediction>,
    pub candidates: Vec<ExplorationCandidate>,
    pub residual_fit_summary: ResidualFitSummary,
    pub calibration_receipt: ExplorationCalibrationReceipt,
    pub interpretation: ExplorationInterpretation,
    pub z_range: (u16, u16),
    pub n_range: (u16, u16),
    pub note: String,
}

/// Hypothesis-generation engine over SEMF + a simplified shell correction.
pub struct NuclearExplorationEngine {
    semf: SemiEmpiricalMassFormula,
    shell: ShellModel,
    measured_nuclei: Vec<MeasuredNucleus>,
    config: ExplorationConfig,
}

impl NuclearExplorationEngine {
    pub fn new() -> Self {
        Self::with_config(ExplorationConfig::default())
            .expect("default exploration configuration must remain valid")
    }

    pub fn with_config(config: ExplorationConfig) -> Result<Self, ExplorationError> {
        config.validate()?;
        Ok(Self {
            semf: SemiEmpiricalMassFormula::default(),
            shell: ShellModel::default(),
            measured_nuclei: Vec::new(),
            config,
        })
    }

    /// Add empirical residual-calibration data. Estimated/extrapolated entries
    /// fail closed rather than being silently treated as measurements.
    pub fn try_add_measured_nuclei(
        &mut self,
        nuclei: impl IntoIterator<Item = MeasuredNucleus>,
    ) -> Result<(), ExplorationError> {
        let mut seen = self
            .measured_nuclei
            .iter()
            .map(|nucleus| (nucleus.z, nucleus.n))
            .collect::<BTreeSet<_>>();
        let mut additions = Vec::new();
        for nucleus in nuclei {
            validate_measured_nucleus(&nucleus)?;
            if !seen.insert((nucleus.z, nucleus.n)) {
                return Err(ExplorationError::DuplicateMeasuredNucleus {
                    z: nucleus.z,
                    n: nucleus.n,
                });
            }
            additions.push(nucleus);
        }
        self.measured_nuclei.extend(additions);
        self.measured_nuclei.sort_by_key(|nucleus| (nucleus.z, nucleus.n));
        Ok(())
    }

    /// Load only the measured subset of the in-tree AME2020 table.
    /// Returns the number of measured entries admitted.
    pub fn add_ame2020_measured_reference(&mut self) -> Result<usize, ExplorationError> {
        let measured = crate::ame2020::ame2020_reference_nuclei()
            .into_iter()
            .filter(|nucleus| nucleus.is_measured)
            .collect::<Vec<_>>();
        let count = measured.len();
        self.try_add_measured_nuclei(measured)?;
        Ok(count)
    }

    pub fn measured_calibration_count(&self) -> usize {
        self.measured_nuclei.len()
    }

    pub fn calibration_receipt(&self) -> ExplorationCalibrationReceipt {
        let mut exact_measured_data = self
            .measured_nuclei
            .iter()
            .map(ExplorationCalibrationDatum::from)
            .collect::<Vec<_>>();
        exact_measured_data.sort_unstable();
        ExplorationCalibrationReceipt {
            config: self.config,
            exact_measured_data,
        }
    }

    pub fn predict(&self, z: u16, n: u16) -> Result<ExplorationPrediction, ExplorationError> {
        let a = z
            .checked_add(n)
            .ok_or(ExplorationError::MassNumberOverflow { z, n })?;
        if a == 0 {
            return Err(ExplorationError::InvalidCoordinate { z, n });
        }
        let semf_binding_energy_mev = self.semf.binding_energy(a, z);
        let shell_correction_mev = self.shell.shell_correction_energy(a, z);
        let total_predicted_binding_energy_mev = semf_binding_energy_mev + shell_correction_mev;
        if !semf_binding_energy_mev.is_finite()
            || !shell_correction_mev.is_finite()
            || !total_predicted_binding_energy_mev.is_finite()
        {
            return Err(ExplorationError::NonFinitePrediction { z, n });
        }

        let measured_binding_energy_mev = self
            .measured_nuclei
            .iter()
            .find(|nucleus| nucleus.z == z && nucleus.n == n)
            .map(|nucleus| nucleus.binding_energy_mev);
        let residual_mev =
            measured_binding_energy_mev.map(|measured| measured - total_predicted_binding_energy_mev);
        let (local_residual_dispersion_proxy_mev, local_measured_support_count) =
            self.local_residual_dispersion_proxy(z, n)?;
        let residual_to_local_dispersion_ratio_proxy = residual_mev.map(|residual| {
            residual.abs() / local_residual_dispersion_proxy_mev.max(self.config.dispersion_floor_mev)
        });

        // Explicit heuristic components. None is a probability.
        let unmeasured_component = if measured_binding_energy_mev.is_none() { 1.0 } else { 0.0 };
        let shell_component = if shell_correction_mev < -3.0 {
            (-shell_correction_mev / 10.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let dispersion_component = (local_residual_dispersion_proxy_mev
            / self.config.priority_dispersion_scale_mev)
            .clamp(0.0, 1.0);
        let exploration_priority_proxy =
            (0.40 * unmeasured_component + 0.35 * shell_component + 0.25 * dispersion_component)
                .clamp(0.0, 1.0);

        Ok(ExplorationPrediction {
            z,
            n,
            a,
            semf_binding_energy_mev,
            shell_correction_mev,
            total_predicted_binding_energy_mev,
            measured_binding_energy_mev,
            residual_mev,
            local_residual_dispersion_proxy_mev,
            local_measured_support_count,
            residual_to_local_dispersion_ratio_proxy,
            exploration_priority_proxy,
        })
    }

    fn local_residual_dispersion_proxy(
        &self,
        z: u16,
        n: u16,
    ) -> Result<(f64, usize), ExplorationError> {
        let residuals = self
            .measured_nuclei
            .iter()
            // Leave the target coordinate out of its own reference distribution.
            .filter(|datum| (datum.z, datum.n) != (z, n))
            .filter(|datum| {
                datum.z.abs_diff(z) <= self.config.local_radius_z
                    && datum.n.abs_diff(n) <= self.config.local_radius_n
            })
            .map(|datum| {
                let a = datum
                    .z
                    .checked_add(datum.n)
                    .ok_or(ExplorationError::MassNumberOverflow {
                        z: datum.z,
                        n: datum.n,
                    })?;
                let prediction = self.semf.binding_energy(a, datum.z)
                    + self.shell.shell_correction_energy(a, datum.z);
                let residual = datum.binding_energy_mev - prediction;
                if residual.is_finite() {
                    Ok(residual)
                } else {
                    Err(ExplorationError::NonFiniteCalibrationResidual {
                        z: datum.z,
                        n: datum.n,
                    })
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        let support_count = residuals.len();
        if support_count < self.config.min_local_measured_points {
            return Ok((
                self.config
                    .sparse_support_dispersion_proxy_mev
                    .max(self.config.dispersion_floor_mev),
                support_count,
            ));
        }
        let mean = residuals.iter().sum::<f64>() / support_count as f64;
        let variance = residuals
            .iter()
            .map(|residual| (residual - mean).powi(2))
            .sum::<f64>()
            / support_count as f64;
        let dispersion = variance.sqrt().max(self.config.dispersion_floor_mev);
        if !dispersion.is_finite() {
            return Err(ExplorationError::NonFiniteLocalDispersion { z, n });
        }
        Ok((dispersion, support_count))
    }

    pub fn explore(
        &self,
        z_min: u16,
        z_max: u16,
        n_min: u16,
        n_max: u16,
    ) -> Result<NuclearExplorationReport, ExplorationError> {
        if z_min == 0 || z_min > z_max || n_min > n_max {
            return Err(ExplorationError::InvalidExplorationRegion);
        }

        let mut predictions = Vec::new();
        for z in z_min..=z_max {
            for n in n_min..=n_max {
                predictions.push(self.predict(z, n)?);
            }
        }

        let measured_predictions = predictions
            .iter()
            .filter(|prediction| prediction.residual_mev.is_some())
            .collect::<Vec<_>>();
        let n_measured = measured_predictions.len();
        let mean_residual_mev = if n_measured > 0 {
            measured_predictions
                .iter()
                .filter_map(|prediction| prediction.residual_mev)
                .sum::<f64>()
                / n_measured as f64
        } else {
            0.0
        };
        let rms_residual_mev = if n_measured > 0 {
            (measured_predictions
                .iter()
                .filter_map(|prediction| prediction.residual_mev)
                .map(|residual| residual * residual)
                .sum::<f64>()
                / n_measured as f64)
                .sqrt()
        } else {
            0.0
        };
        let max_abs_residual_mev = measured_predictions
            .iter()
            .filter_map(|prediction| prediction.residual_mev)
            .map(f64::abs)
            .fold(0.0, f64::max);
        let fraction_within_local_dispersion_proxy = if n_measured > 0 {
            measured_predictions
                .iter()
                .filter(|prediction| {
                    prediction
                        .residual_to_local_dispersion_ratio_proxy
                        .is_some_and(|ratio| ratio <= 1.0)
                })
                .count() as f64
                / n_measured as f64
        } else {
            0.0
        };

        let residual_fit_summary = ResidualFitSummary {
            n_measured_comparisons: n_measured,
            mean_residual_mev,
            rms_residual_mev,
            max_abs_residual_mev,
            fraction_within_local_dispersion_proxy,
        };

        let mut candidates = predictions
            .iter()
            .filter(|prediction| {
                prediction.exploration_priority_proxy >= self.config.candidate_priority_threshold
            })
            .map(|prediction| {
                let reason = if prediction.measured_binding_energy_mev.is_none()
                    && prediction.shell_correction_mev < -3.0
                {
                    format!(
                        "unmeasured coordinate; shell-correction proxy {:.1} MeV; local residual-dispersion proxy {:.1} MeV from {} measured neighbors",
                        prediction.shell_correction_mev,
                        prediction.local_residual_dispersion_proxy_mev,
                        prediction.local_measured_support_count
                    )
                } else if prediction.measured_binding_energy_mev.is_none() {
                    format!(
                        "unmeasured coordinate; local residual-dispersion proxy {:.1} MeV from {} measured neighbors",
                        prediction.local_residual_dispersion_proxy_mev,
                        prediction.local_measured_support_count
                    )
                } else if prediction
                    .residual_to_local_dispersion_ratio_proxy
                    .is_some_and(|ratio| ratio > 2.0)
                {
                    format!(
                        "measured residual is {:.1}x the leave-one-out local residual-dispersion proxy",
                        prediction.residual_to_local_dispersion_ratio_proxy.unwrap_or(0.0)
                    )
                } else {
                    "heuristic exploration priority".to_string()
                };
                ExplorationCandidate {
                    z: prediction.z,
                    n: prediction.n,
                    a: prediction.a,
                    reason,
                    exploration_priority_proxy: prediction.exploration_priority_proxy,
                    local_residual_dispersion_proxy_mev: prediction
                        .local_residual_dispersion_proxy_mev,
                    local_measured_support_count: prediction.local_measured_support_count,
                    shell_correction_mev: prediction.shell_correction_mev,
                    semf_geiger_nuttall_half_life_proxy_seconds: self
                        .semf
                        .geiger_nuttall_half_life(prediction.a, prediction.z),
                }
            })
            .collect::<Vec<_>>();
        candidates.sort_by(|left, right| {
            right
                .exploration_priority_proxy
                .total_cmp(&left.exploration_priority_proxy)
                .then(left.z.cmp(&right.z))
                .then(left.n.cmp(&right.n))
        });
        candidates.truncate(self.config.max_candidates);

        Ok(NuclearExplorationReport {
            predictions,
            candidates,
            residual_fit_summary,
            calibration_receipt: self.calibration_receipt(),
            interpretation: ExplorationInterpretation::HeuristicExplorationOnly,
            z_range: (z_min, z_max),
            n_range: (n_min, n_max),
            note: "SEMF + simplified shell-model exploration; leave-one-coordinate-out local residual spread and priority are proxies only and do not establish calibrated uncertainty, statistical significance, nuclear existence/stability, decay lifetime, or discovery".to_string(),
        })
    }

    pub fn format_report(report: &NuclearExplorationReport) -> String {
        let mut output = String::new();
        output.push_str("=== Nuclear Exploration Report ===\n");
        output.push_str(&format!(
            "Region: Z={}-{}, N={}-{} ({} coordinates)\n",
            report.z_range.0,
            report.z_range.1,
            report.n_range.0,
            report.n_range.1,
            report.predictions.len()
        ));
        output.push_str(&format!(
            "Measured residual comparisons: {}, RMS residual={:.2} MeV, {:.0}% within leave-one-out local dispersion proxy\n",
            report.residual_fit_summary.n_measured_comparisons,
            report.residual_fit_summary.rms_residual_mev,
            report.residual_fit_summary.fraction_within_local_dispersion_proxy * 100.0
        ));
        output.push_str(&format!(
            "Interpretation: {:?}\n\nTop {} exploration candidates:\n",
            report.interpretation,
            report.candidates.len()
        ));
        output.push_str("  Z   N    A | Priority | Residual spread proxy | Support | Shell proxy | Reason\n");
        for candidate in &report.candidates {
            output.push_str(&format!(
                "{:>3} {:>3} {:>4} |   {:.2}   |       {:>7.1} MeV | {:>7} | {:>7.1} MeV | {}\n",
                candidate.z,
                candidate.n,
                candidate.a,
                candidate.exploration_priority_proxy,
                candidate.local_residual_dispersion_proxy_mev,
                candidate.local_measured_support_count,
                candidate.shell_correction_mev,
                candidate.reason
            ));
        }
        output
    }
}

impl Default for NuclearExplorationEngine {
    fn default() -> Self {
        Self::new()
    }
}

fn validate_measured_nucleus(nucleus: &MeasuredNucleus) -> Result<(), ExplorationError> {
    if !nucleus.is_measured {
        return Err(ExplorationError::EstimatedDatumEnteredMeasuredCalibration {
            z: nucleus.z,
            n: nucleus.n,
        });
    }
    if nucleus.z == 0 && nucleus.n == 0 {
        return Err(ExplorationError::InvalidCoordinate {
            z: nucleus.z,
            n: nucleus.n,
        });
    }
    if !nucleus.binding_energy_mev.is_finite() {
        return Err(ExplorationError::NonFiniteMeasuredEnergy {
            z: nucleus.z,
            n: nucleus.n,
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExplorationError {
    EmptyLocalWindow,
    InvalidCountConfig,
    InvalidPositiveScale,
    InvalidPriorityThreshold,
    InvalidExplorationRegion,
    EstimatedDatumEnteredMeasuredCalibration { z: u16, n: u16 },
    DuplicateMeasuredNucleus { z: u16, n: u16 },
    InvalidCoordinate { z: u16, n: u16 },
    NonFiniteMeasuredEnergy { z: u16, n: u16 },
    MassNumberOverflow { z: u16, n: u16 },
    NonFinitePrediction { z: u16, n: u16 },
    NonFiniteCalibrationResidual { z: u16, n: u16 },
    NonFiniteLocalDispersion { z: u16, n: u16 },
}

impl fmt::Display for ExplorationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyLocalWindow => write!(f, "exploration local window must span at least one axis"),
            Self::InvalidCountConfig => write!(f, "exploration support/candidate counts must be positive"),
            Self::InvalidPositiveScale => write!(f, "exploration scale parameters must be finite and positive"),
            Self::InvalidPriorityThreshold => write!(f, "exploration priority threshold must be finite in [0,1]"),
            Self::InvalidExplorationRegion => write!(f, "invalid nuclear exploration region"),
            Self::EstimatedDatumEnteredMeasuredCalibration { z, n } => write!(f, "estimated/extrapolated datum entered measured residual calibration at Z={z}, N={n}"),
            Self::DuplicateMeasuredNucleus { z, n } => write!(f, "duplicate measured exploration datum at Z={z}, N={n}"),
            Self::InvalidCoordinate { z, n } => write!(f, "invalid nuclear coordinate Z={z}, N={n}"),
            Self::NonFiniteMeasuredEnergy { z, n } => write!(f, "non-finite measured binding energy at Z={z}, N={n}"),
            Self::MassNumberOverflow { z, n } => write!(f, "mass-number overflow at Z={z}, N={n}"),
            Self::NonFinitePrediction { z, n } => write!(f, "non-finite exploration prediction at Z={z}, N={n}"),
            Self::NonFiniteCalibrationResidual { z, n } => write!(f, "non-finite calibration residual at Z={z}, N={n}"),
            Self::NonFiniteLocalDispersion { z, n } => write!(f, "non-finite local residual-dispersion proxy at Z={z}, N={n}"),
        }
    }
}

impl std::error::Error for ExplorationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn datum(z: u16, n: u16, be: f64, measured: bool) -> MeasuredNucleus {
        MeasuredNucleus {
            z,
            n,
            binding_energy_mev: be,
            is_measured: measured,
        }
    }

    #[test]
    fn estimated_entries_fail_closed_for_empirical_residual_calibration() {
        let mut engine = NuclearExplorationEngine::new();
        assert!(matches!(
            engine.try_add_measured_nuclei([datum(8, 20, 167.6, false)]),
            Err(ExplorationError::EstimatedDatumEnteredMeasuredCalibration { .. })
        ));
        assert_eq!(engine.measured_calibration_count(), 0);
    }

    #[test]
    fn ame_reference_loader_admits_measured_subset_only() {
        let mut engine = NuclearExplorationEngine::new();
        let count = engine.add_ame2020_measured_reference().unwrap();
        assert_eq!(count, engine.measured_calibration_count());
        assert!(count > 0);
        assert_eq!(engine.calibration_receipt().exact_measured_data.len(), count);
    }

    #[test]
    fn local_residual_reference_is_leave_one_coordinate_out() {
        let mut engine = NuclearExplorationEngine::with_config(ExplorationConfig {
            min_local_measured_points: 1,
            ..ExplorationConfig::default()
        })
        .unwrap();
        engine
            .try_add_measured_nuclei([
                datum(8, 8, 127.619, true),
                datum(8, 10, 139.808, true),
            ])
            .unwrap();
        let measured_target = engine.predict(8, 8).unwrap();
        assert_eq!(measured_target.local_measured_support_count, 1);
        assert!(measured_target
            .residual_to_local_dispersion_ratio_proxy
            .is_some());
    }

    #[test]
    fn residual_spread_is_explicit_proxy_not_calibrated_uncertainty() {
        let mut engine = NuclearExplorationEngine::new();
        engine
            .try_add_measured_nuclei([
                datum(8, 8, 127.619, true),
                datum(8, 10, 139.808, true),
                datum(10, 10, 160.645, true),
            ])
            .unwrap();
        let prediction = engine.predict(9, 9).unwrap();
        assert!(prediction.local_residual_dispersion_proxy_mev.is_finite());
        assert!(prediction.local_residual_dispersion_proxy_mev > 0.0);
        assert_eq!(prediction.measured_binding_energy_mev, None);
    }

    #[test]
    fn exploration_report_carries_proxy_only_interpretation() {
        let mut engine = NuclearExplorationEngine::new();
        engine
            .try_add_measured_nuclei([
                datum(6, 6, 92.162, true),
                datum(8, 8, 127.619, true),
                datum(10, 10, 160.645, true),
            ])
            .unwrap();
        let report = engine.explore(6, 10, 6, 12).unwrap();
        assert_eq!(
            report.interpretation,
            ExplorationInterpretation::HeuristicExplorationOnly
        );
        assert_eq!(report.calibration_receipt.exact_measured_data.len(), 3);
        assert!(report
            .predictions
            .iter()
            .all(|prediction| prediction.exploration_priority_proxy.is_finite()));
    }

    #[test]
    fn duplicate_measured_coordinates_fail_closed() {
        let mut engine = NuclearExplorationEngine::new();
        engine
            .try_add_measured_nuclei([datum(8, 8, 127.619, true)])
            .unwrap();
        assert!(matches!(
            engine.try_add_measured_nuclei([datum(8, 8, 127.620, true)]),
            Err(ExplorationError::DuplicateMeasuredNucleus { .. })
        ));
    }

    #[test]
    fn mass_number_overflow_fails_closed() {
        let engine = NuclearExplorationEngine::new();
        assert!(matches!(
            engine.predict(u16::MAX, 1),
            Err(ExplorationError::MassNumberOverflow { .. })
        ));
    }
}
