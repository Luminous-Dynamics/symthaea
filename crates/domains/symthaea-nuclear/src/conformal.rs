// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Leakage-resistant conformal diagnostics for nuclear-mass prediction.
//!
//! The previous exploratory test calibrated a Random-Forest predictor that had
//! already been fitted on the full measured AME2020 corpus, then reported
//! coverage over a corpus overlapping both fit and calibration. That is useful
//! as a smoke test but cannot support an honest uncertainty statement.
//!
//! This module enforces a three-way boundary:
//!
//! ```text
//! frozen blind split
//!      |
//!      +-- training region -- deterministic partition --> fit subset
//!      |                                      `---------> calibration subset
//!      |
//!      `-- structural blind holdout --------------------> OOD diagnostic only
//! ```
//!
//! The conformal quantile is computed only from a calibration subset that the RF
//! fit never sees. The structural holdout remains untouched until evaluation.
//!
//! **Important:** split-conformal finite-sample coverage requires exchangeability
//! between calibration and future examples. A proton/neutron frontier, shell
//! window, or other deliberately hostile structural holdout violates that working
//! assumption by design. Therefore blind-holdout coverage reported here is an
//! **out-of-domain diagnostic, not a conformal coverage guarantee**.

use crate::blind_models::BlindTrainingReceipt;
use crate::blind_validation::{BlindHoldout, BlindValidationError, BlindValidationSplit};
use crate::discovery::MeasuredNucleus;
use crate::ml_mass::{MlMassConfig, MlMassFitError, MlMassPredictor};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt;

/// Deterministic, order-independent partition of the blind training region.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationSplitRule {
    pub modulus: u16,
    pub calibration_remainder: u16,
}

impl Default for CalibrationSplitRule {
    fn default() -> Self {
        Self {
            modulus: 5,
            calibration_remainder: 0,
        }
    }
}

impl CalibrationSplitRule {
    fn validate(self) -> Result<(), ConformalError> {
        if self.modulus < 2 || self.calibration_remainder >= self.modulus {
            return Err(ConformalError::InvalidCalibrationSplitRule);
        }
        Ok(())
    }

    fn is_calibration(self, z: u16, n: u16) -> bool {
        // Stable coordinate mixer; this chooses membership only. It is not a
        // cryptographic hash and carries no security meaning.
        let mixed = (u64::from(z).wrapping_mul(73_856_093))
            ^ (u64::from(n).wrapping_mul(19_349_663));
        mixed % u64::from(self.modulus) == u64::from(self.calibration_remainder)
    }
}

/// Nonconformity score used by the calibration quantile.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ConformalScore {
    /// |prediction - measurement| in MeV.
    AbsoluteResidual,
    /// |prediction - measurement| divided by RF tree-dispersion proxy, with a
    /// declared positive floor. Tree dispersion is not treated as a calibrated
    /// standard deviation.
    TreeDispersionNormalized { floor_mev: f64 },
}

impl ConformalScore {
    fn validate(self) -> Result<(), ConformalError> {
        match self {
            Self::AbsoluteResidual => Ok(()),
            Self::TreeDispersionNormalized { floor_mev }
                if floor_mev.is_finite() && floor_mev > 0.0 =>
            {
                Ok(())
            }
            Self::TreeDispersionNormalized { .. } => Err(ConformalError::InvalidScaleFloor),
        }
    }

    fn scale(self, tree_dispersion_mev: f64) -> f64 {
        match self {
            Self::AbsoluteResidual => 1.0,
            Self::TreeDispersionNormalized { floor_mev } => tree_dispersion_mev.max(floor_mev),
        }
    }
}

/// Exact datum identity used to bind the fit/calibration/holdout partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CalibrationDatum {
    pub z: u16,
    pub n: u16,
    pub binding_energy_mev_bits: u64,
}

impl From<&MeasuredNucleus> for CalibrationDatum {
    fn from(nucleus: &MeasuredNucleus) -> Self {
        Self {
            z: nucleus.z,
            n: nucleus.n,
            binding_energy_mev_bits: nucleus.binding_energy_mev.to_bits(),
        }
    }
}

fn exact_data(nuclei: &[MeasuredNucleus]) -> Vec<CalibrationDatum> {
    let mut data = nuclei.iter().map(CalibrationDatum::from).collect::<Vec<_>>();
    data.sort_unstable();
    data
}

/// Exact partition receipt for one conformal calibration run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CalibrationPartitionReceipt {
    pub protocol: BlindHoldout,
    pub split_rule: CalibrationSplitRule,
    pub rf_config: MlMassConfig,
    pub fit: Vec<CalibrationDatum>,
    pub calibration: Vec<CalibrationDatum>,
    pub blind_holdout: Vec<CalibrationDatum>,
}

impl CalibrationPartitionReceipt {
    fn matches_split(&self, split: &BlindValidationSplit) -> bool {
        let mut parent_training = self.fit.clone();
        parent_training.extend(self.calibration.iter().copied());
        parent_training.sort_unstable();
        parent_training == exact_data(&split.training)
            && self.blind_holdout == exact_data(&split.holdout)
            && self.protocol == split.protocol
    }
}

/// Explicit statement about what statistical guarantee is *not* being made.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlindCoverageInterpretation {
    /// The structural holdout is intentionally distribution-shifted relative to
    /// the fit/calibration region, so empirical coverage is diagnostic only.
    NoExchangeabilityGuarantee,
}

/// One prediction interval.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConformalInterval {
    pub z: u16,
    pub n: u16,
    pub point_prediction_mev: f64,
    pub lower_mev: f64,
    pub upper_mev: f64,
    pub half_width_mev: f64,
    pub tree_dispersion_proxy_mev: f64,
}

/// Empirical coverage measured on the untouched structural blind holdout.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BlindCoverageDiagnostic {
    pub protocol: BlindHoldout,
    pub target_marginal_coverage: f64,
    pub n_holdout: usize,
    pub n_covered: usize,
    pub empirical_coverage: f64,
    pub mean_interval_width_mev: f64,
    pub median_interval_width_mev: f64,
    pub max_interval_width_mev: f64,
    pub interpretation: BlindCoverageInterpretation,
}

/// RF conformal calibrator whose learned fit never sees either calibration or
/// structural blind holdout data.
pub struct RfBlindConformalCalibrator {
    predictor: MlMassPredictor,
    receipt: CalibrationPartitionReceipt,
    score: ConformalScore,
    target_marginal_coverage: f64,
    calibration_quantile: f64,
}

impl RfBlindConformalCalibrator {
    pub fn fit(
        split: &BlindValidationSplit,
        rf_config: MlMassConfig,
        split_rule: CalibrationSplitRule,
        score: ConformalScore,
        target_marginal_coverage: f64,
    ) -> Result<Self, ConformalError> {
        split.validate().map_err(ConformalError::BlindValidation)?;
        // Also exercises #642's duplicate/non-finite receipt invariants before
        // any learned fit is attempted.
        BlindTrainingReceipt::from_split(split).map_err(ConformalError::BlindReceipt)?;
        split_rule.validate()?;
        score.validate()?;
        if !target_marginal_coverage.is_finite()
            || target_marginal_coverage <= 0.0
            || target_marginal_coverage >= 1.0
        {
            return Err(ConformalError::InvalidTargetCoverage);
        }

        let mut fit = Vec::new();
        let mut calibration = Vec::new();
        for nucleus in &split.training {
            if split_rule.is_calibration(nucleus.z, nucleus.n) {
                calibration.push(nucleus.clone());
            } else {
                fit.push(nucleus.clone());
            }
        }
        if fit.is_empty() {
            return Err(ConformalError::EmptyFitSet);
        }
        if calibration.is_empty() {
            return Err(ConformalError::EmptyCalibrationSet);
        }

        // Explicit corpus only: neither calibration nor structural holdout can
        // enter this fit call.
        let predictor = MlMassPredictor::fit_measured_with_config(&fit, rf_config)
            .map_err(ConformalError::MlMassFit)?;

        let mut scores = Vec::with_capacity(calibration.len());
        for nucleus in &calibration {
            let prediction = predictor.predict(nucleus.z, nucleus.n);
            let residual = (prediction.binding_energy - nucleus.binding_energy_mev).abs();
            let scale = score.scale(prediction.uncertainty);
            let nonconformity = residual / scale;
            if !nonconformity.is_finite() || nonconformity < 0.0 {
                return Err(ConformalError::NonFiniteCalibrationScore {
                    z: nucleus.z,
                    n: nucleus.n,
                });
            }
            scores.push(nonconformity);
        }
        scores.sort_by(|left, right| left.total_cmp(right));
        let calibration_quantile = finite_sample_quantile(&scores, target_marginal_coverage)?;

        let receipt = CalibrationPartitionReceipt {
            protocol: split.protocol,
            split_rule,
            rf_config,
            fit: exact_data(&fit),
            calibration: exact_data(&calibration),
            blind_holdout: exact_data(&split.holdout),
        };

        Ok(Self {
            predictor,
            receipt,
            score,
            target_marginal_coverage,
            calibration_quantile,
        })
    }

    pub fn receipt(&self) -> &CalibrationPartitionReceipt {
        &self.receipt
    }

    pub fn calibration_quantile(&self) -> f64 {
        self.calibration_quantile
    }

    pub fn interval(&self, z: u16, n: u16) -> Result<ConformalInterval, ConformalError> {
        let prediction = self.predictor.predict(z, n);
        let scale = self.score.scale(prediction.uncertainty);
        let half_width_mev = self.calibration_quantile * scale;
        if !prediction.binding_energy.is_finite()
            || !half_width_mev.is_finite()
            || half_width_mev < 0.0
        {
            return Err(ConformalError::NonFiniteInterval { z, n });
        }
        Ok(ConformalInterval {
            z,
            n,
            point_prediction_mev: prediction.binding_energy,
            lower_mev: prediction.binding_energy - half_width_mev,
            upper_mev: prediction.binding_energy + half_width_mev,
            half_width_mev,
            tree_dispersion_proxy_mev: prediction.uncertainty,
        })
    }

    /// Empirical coverage on the exact structural blind holdout retained at fit
    /// time. The report always carries `NoExchangeabilityGuarantee`.
    pub fn evaluate_blind_holdout(
        &self,
        split: &BlindValidationSplit,
    ) -> Result<BlindCoverageDiagnostic, ConformalError> {
        split.validate().map_err(ConformalError::BlindValidation)?;
        if !self.receipt.matches_split(split) {
            return Err(ConformalError::SplitDoesNotMatchCalibrationReceipt);
        }

        let mut covered = 0usize;
        let mut widths = Vec::with_capacity(split.holdout.len());
        for nucleus in &split.holdout {
            let interval = self.interval(nucleus.z, nucleus.n)?;
            if nucleus.binding_energy_mev >= interval.lower_mev
                && nucleus.binding_energy_mev <= interval.upper_mev
            {
                covered += 1;
            }
            widths.push(2.0 * interval.half_width_mev);
        }
        if widths.is_empty() {
            return Err(ConformalError::EmptyBlindHoldout);
        }
        widths.sort_by(|left, right| left.total_cmp(right));
        let n = widths.len();
        let mean_interval_width_mev = widths.iter().sum::<f64>() / n as f64;
        let median_interval_width_mev = if n % 2 == 0 {
            (widths[n / 2 - 1] + widths[n / 2]) / 2.0
        } else {
            widths[n / 2]
        };
        let max_interval_width_mev = widths[n - 1];

        Ok(BlindCoverageDiagnostic {
            protocol: split.protocol,
            target_marginal_coverage: self.target_marginal_coverage,
            n_holdout: n,
            n_covered: covered,
            empirical_coverage: covered as f64 / n as f64,
            mean_interval_width_mev,
            median_interval_width_mev,
            max_interval_width_mev,
            interpretation: BlindCoverageInterpretation::NoExchangeabilityGuarantee,
        })
    }
}

/// Standard split-conformal finite-sample quantile using one-based rank
/// `ceil((n + 1) * target_coverage)`, clamped to the available calibration set.
fn finite_sample_quantile(
    sorted_scores: &[f64],
    target_coverage: f64,
) -> Result<f64, ConformalError> {
    if sorted_scores.is_empty() {
        return Err(ConformalError::EmptyCalibrationSet);
    }
    let n = sorted_scores.len();
    let rank = (((n + 1) as f64 * target_coverage).ceil() as usize).clamp(1, n);
    let quantile = sorted_scores[rank - 1];
    if !quantile.is_finite() || quantile < 0.0 {
        return Err(ConformalError::InvalidCalibrationQuantile);
    }
    Ok(quantile)
}

#[derive(Debug, Clone, PartialEq)]
pub enum ConformalError {
    BlindValidation(BlindValidationError),
    BlindReceipt(crate::blind_models::BlindModelError),
    MlMassFit(MlMassFitError),
    InvalidCalibrationSplitRule,
    InvalidScaleFloor,
    InvalidTargetCoverage,
    EmptyFitSet,
    EmptyCalibrationSet,
    EmptyBlindHoldout,
    NonFiniteCalibrationScore { z: u16, n: u16 },
    InvalidCalibrationQuantile,
    NonFiniteInterval { z: u16, n: u16 },
    SplitDoesNotMatchCalibrationReceipt,
}

impl fmt::Display for ConformalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BlindValidation(error) => write!(f, "blind split validation failed: {error}"),
            Self::BlindReceipt(error) => write!(f, "blind training receipt failed: {error}"),
            Self::MlMassFit(error) => write!(f, "RF fit failed: {error}"),
            Self::InvalidCalibrationSplitRule => write!(f, "invalid calibration split rule"),
            Self::InvalidScaleFloor => write!(f, "conformal scale floor must be finite and positive"),
            Self::InvalidTargetCoverage => {
                write!(f, "target marginal coverage must be finite and strictly between 0 and 1")
            }
            Self::EmptyFitSet => write!(f, "conformal fit subset is empty"),
            Self::EmptyCalibrationSet => write!(f, "conformal calibration subset is empty"),
            Self::EmptyBlindHoldout => write!(f, "structural blind holdout is empty"),
            Self::NonFiniteCalibrationScore { z, n } => write!(
                f,
                "non-finite conformal calibration score at Z={z}, N={n}"
            ),
            Self::InvalidCalibrationQuantile => write!(f, "invalid conformal calibration quantile"),
            Self::NonFiniteInterval { z, n } => {
                write!(f, "non-finite conformal interval at Z={z}, N={n}")
            }
            Self::SplitDoesNotMatchCalibrationReceipt => write!(
                f,
                "evaluation split does not match the exact fit/calibration/holdout receipt"
            ),
        }
    }
}

impl std::error::Error for ConformalError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn frontier_split() -> BlindValidationSplit {
        BlindValidationSplit::from_ame2020(BlindHoldout::ProtonFrontier {
            train_z_max: 82,
        })
        .unwrap()
    }

    fn small_rf() -> MlMassConfig {
        MlMassConfig {
            n_trees: 8,
            max_depth: 4,
            min_samples: 2,
            seed: 31415,
        }
    }

    #[test]
    fn calibration_partition_never_contains_structural_holdout() {
        let split = frontier_split();
        let calibrator = RfBlindConformalCalibrator::fit(
            &split,
            small_rf(),
            CalibrationSplitRule::default(),
            ConformalScore::TreeDispersionNormalized { floor_mev: 0.01 },
            0.90,
        )
        .unwrap();

        let fit: BTreeSet<_> = calibrator
            .receipt()
            .fit
            .iter()
            .map(|datum| (datum.z, datum.n))
            .collect();
        let calibration: BTreeSet<_> = calibrator
            .receipt()
            .calibration
            .iter()
            .map(|datum| (datum.z, datum.n))
            .collect();
        assert!(fit.is_disjoint(&calibration));
        assert!(split.holdout.iter().all(|nucleus| {
            !fit.contains(&(nucleus.z, nucleus.n))
                && !calibration.contains(&(nucleus.z, nucleus.n))
        }));
        assert_eq!(
            fit.len() + calibration.len(),
            split.training.len()
        );
    }

    #[test]
    fn finite_sample_quantile_uses_one_based_rank() {
        let scores = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // ceil((5+1)*0.5)=3 => third sorted value.
        assert_eq!(finite_sample_quantile(&scores, 0.5).unwrap(), 3.0);
        // High requested coverage clamps to the largest available score.
        assert_eq!(finite_sample_quantile(&scores, 0.99).unwrap(), 5.0);
    }

    #[test]
    fn structural_holdout_coverage_is_explicitly_ood_diagnostic() {
        let split = frontier_split();
        let calibrator = RfBlindConformalCalibrator::fit(
            &split,
            small_rf(),
            CalibrationSplitRule::default(),
            ConformalScore::AbsoluteResidual,
            0.90,
        )
        .unwrap();
        let report = calibrator.evaluate_blind_holdout(&split).unwrap();

        assert_eq!(report.n_holdout, split.holdout.len());
        assert!(report.empirical_coverage.is_finite());
        assert!((0.0..=1.0).contains(&report.empirical_coverage));
        assert_eq!(
            report.interpretation,
            BlindCoverageInterpretation::NoExchangeabilityGuarantee
        );
    }

    #[test]
    fn evaluation_rejects_changed_holdout_values() {
        let split = frontier_split();
        let calibrator = RfBlindConformalCalibrator::fit(
            &split,
            small_rf(),
            CalibrationSplitRule::default(),
            ConformalScore::AbsoluteResidual,
            0.90,
        )
        .unwrap();

        let mut changed = split.clone();
        changed.holdout[0].binding_energy_mev += 0.001;
        changed.validate().unwrap();
        assert_eq!(
            calibrator.evaluate_blind_holdout(&changed).unwrap_err(),
            ConformalError::SplitDoesNotMatchCalibrationReceipt
        );
    }

    #[test]
    fn invalid_calibration_configuration_fails_closed() {
        let split = frontier_split();
        assert_eq!(
            RfBlindConformalCalibrator::fit(
                &split,
                small_rf(),
                CalibrationSplitRule {
                    modulus: 1,
                    calibration_remainder: 0,
                },
                ConformalScore::AbsoluteResidual,
                0.90,
            )
            .err()
            .unwrap(),
            ConformalError::InvalidCalibrationSplitRule
        );
        assert_eq!(
            RfBlindConformalCalibrator::fit(
                &split,
                small_rf(),
                CalibrationSplitRule::default(),
                ConformalScore::TreeDispersionNormalized { floor_mev: 0.0 },
                0.90,
            )
            .err()
            .unwrap(),
            ConformalError::InvalidScaleFloor
        );
    }
}
