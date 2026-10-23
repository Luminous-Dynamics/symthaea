// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Reference-free detector selection from clean-data self-perturbations.
//!
//! This module is inspired by recent work on calibrated detector aggregation, but is not a
//! reproduction of DecayBench's Alert algorithm and carries no no-regret theorem. It selects
//! detector channels that distinguish clean calibration examples from perturbations generated
//! without deployment labels, orients channels whose scores move in the reverse direction, and
//! produces one nonnegative aggregate score. The aggregate must then be conformally calibrated
//! on a separate, untouched clean split.
//!
//! Selection quality depends on whether the self-generated perturbations resemble operational
//! shifts. The selector may choose the wrong detectors under a mismatch between probe
//! perturbations and future drift. Outer conformal calibration controls marginal false alarms
//! under the exchangeability protocol; it does not guarantee power, meaningful diagnosis, or
//! detection of harmful prediction degradation.

use crate::adequacy_calibration::{
    CalibrationError, ConformalAdequacyCalibrator, ConformalAdequacyResult,
};
use crate::model_adequacy::AdequacyDiagnostic;

/// How selected, oriented detector scores are combined before outer calibration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DetectorPanelAggregation {
    /// Average positive standardized scores; helps when multiple selected channels respond.
    MeanPositiveZ,
    /// Use the strongest positive standardized score; avoids dilution by weaker selected channels.
    MaxPositiveZ,
}

/// Failure to fit or evaluate the reference-free detector panel.
#[derive(Clone, Debug, PartialEq)]
pub enum DetectorPanelError {
    /// There are too few clean or self-perturbed observations to estimate the panel.
    InsufficientSelectionSamples,
    /// Input matrices are empty, have zero columns, or have inconsistent widths.
    InvalidSelectionShape,
    /// At least one selection score is non-finite.
    NonFiniteSelectionScore,
    /// top_m must be between one and the number of detector columns.
    InvalidTopM,
    /// The AUC threshold must be finite and in (0.5, 1].
    InvalidDetectabilityThreshold,
    /// Scale floor must be finite and positive.
    InvalidScaleFloor,
    /// No detector met the predeclared self-detectability threshold.
    NoInformativeDetectors,
    /// Audit vector does not match the detector count or contains a non-finite score.
    InvalidAuditScores,
    /// The shared conformal calibrator rejected the audit contract.
    Calibration(CalibrationError),
}

/// Selected detector summary suitable for an auditable monitor receipt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SelectedDetector {
    /// Column index in the supplied detector-score vectors.
    pub detector_index: usize,
    /// +1 means higher scores aligned with the self-perturbation; -1 means lower scores did.
    pub orientation: i8,
    /// Mean score on clean selector data.
    pub clean_mean: f64,
    /// Population standard deviation on clean selector data.
    pub clean_scale: f64,
    /// AUC of the oriented detector separating clean from self-perturbed selector observations.
    pub detectability_auc: f64,
}

/// Frozen reference-free detector panel. Fit once, then calibrate its aggregate independently.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectorPanelScorer {
    detector_count: usize,
    selected: Vec<SelectedDetector>,
    scale_floor: f64,
    aggregation: DetectorPanelAggregation,
}

impl DetectorPanelScorer {
    /// Select detectors using only clean scores and self-generated perturbation scores.
    ///
    /// Both matrices have the same frozen detector-column order. Perturbations must be built
    /// without access to deployment/evaluation labels. Selection, orientation, clean means/scales,
    /// top_m, detectability threshold, and aggregation are frozen before outer calibration.
    ///
    /// The resulting aggregate is not calibrated by this method. Fit a
    /// ConformalAdequacyCalibrator from independent clean outer-calibration observations scored
    /// with this exact panel before evaluating any deployment audit.
    pub fn fit(
        clean_selection_scores: &[Vec<f64>],
        perturbed_selection_scores: &[Vec<f64>],
        top_m: usize,
        min_detectability_auc: f64,
        scale_floor: f64,
        aggregation: DetectorPanelAggregation,
    ) -> Result<Self, DetectorPanelError> {
        if clean_selection_scores.len() < 2 || perturbed_selection_scores.len() < 2 {
            return Err(DetectorPanelError::InsufficientSelectionSamples);
        }
        if !min_detectability_auc.is_finite()
            || min_detectability_auc <= 0.5
            || min_detectability_auc > 1.0
        {
            return Err(DetectorPanelError::InvalidDetectabilityThreshold);
        }
        if !scale_floor.is_finite() || scale_floor <= 0.0 {
            return Err(DetectorPanelError::InvalidScaleFloor);
        }

        let detector_count = clean_selection_scores
            .first()
            .ok_or(DetectorPanelError::InvalidSelectionShape)?
            .len();
        if detector_count == 0
            || !clean_selection_scores.iter().all(|row| row.len() == detector_count)
            || !perturbed_selection_scores.iter().all(|row| row.len() == detector_count)
        {
            return Err(DetectorPanelError::InvalidSelectionShape);
        }
        if top_m == 0 || top_m > detector_count {
            return Err(DetectorPanelError::InvalidTopM);
        }
        if clean_selection_scores
            .iter()
            .flatten()
            .chain(perturbed_selection_scores.iter().flatten())
            .any(|score| !score.is_finite())
        {
            return Err(DetectorPanelError::NonFiniteSelectionScore);
        }

        let clean_means = column_means(clean_selection_scores, detector_count);
        let clean_scales = column_scales(clean_selection_scores, &clean_means, detector_count);
        let mut candidates = Vec::with_capacity(detector_count);

        for detector_index in 0..detector_count {
            let clean = column_values(clean_selection_scores, detector_index);
            let perturbed = column_values(perturbed_selection_scores, detector_index);
            let raw_auc = auc_perturbed_higher(&clean, &perturbed);
            let (orientation, detectability_auc) = if raw_auc >= 0.5 {
                (1_i8, raw_auc)
            } else {
                (-1_i8, 1.0 - raw_auc)
            };
            if detectability_auc >= min_detectability_auc {
                candidates.push(SelectedDetector {
                    detector_index,
                    orientation,
                    clean_mean: clean_means[detector_index],
                    clean_scale: clean_scales[detector_index].max(scale_floor),
                    detectability_auc,
                });
            }
        }

        if candidates.is_empty() {
            return Err(DetectorPanelError::NoInformativeDetectors);
        }
        candidates.sort_by(|left, right| {
            right
                .detectability_auc
                .total_cmp(&left.detectability_auc)
                .then_with(|| left.detector_index.cmp(&right.detector_index))
        });
        candidates.truncate(top_m);

        Ok(Self {
            detector_count,
            selected: candidates,
            scale_floor,
            aggregation,
        })
    }

    /// Produce a nonconformity score from one score per frozen detector.
    ///
    /// Returns None for a wrong detector width or any non-finite input. Higher values mean
    /// more evidence aligned with the self-generated perturbation, not proof of harmful drift.
    pub fn score(&self, detector_scores: &[f64]) -> Option<f64> {
        if detector_scores.len() != self.detector_count
            || detector_scores.iter().any(|score| !score.is_finite())
        {
            return None;
        }

        let mut positive_z_scores = Vec::with_capacity(self.selected.len());
        for selected in &self.selected {
            let score = detector_scores[selected.detector_index];
            let oriented_z = selected.orientation as f64
                * (score - selected.clean_mean)
                / selected.clean_scale.max(self.scale_floor);
            if !oriented_z.is_finite() {
                return None;
            }
            positive_z_scores.push(oriented_z.max(0.0));
        }

        let aggregate = match self.aggregation {
            DetectorPanelAggregation::MeanPositiveZ => {
                positive_z_scores.iter().sum::<f64>() / positive_z_scores.len() as f64
            }
            DetectorPanelAggregation::MaxPositiveZ => positive_z_scores
                .iter()
                .copied()
                .fold(0.0_f64, f64::max),
        };
        aggregate.is_finite().then_some(aggregate)
    }

    /// Calibrate a panel score against independent outer-clean scores.
    ///
    /// Calibration scores must be generated after panel selection has been frozen. This
    /// API cannot prove that selection data and outer-calibration data are disjoint.
    pub fn assess(
        &self,
        calibrator: &ConformalAdequacyCalibrator,
        detector_scores: &[f64],
        independent_audit: bool,
    ) -> Result<ConformalAdequacyResult, DetectorPanelError> {
        let score = self
            .score(detector_scores)
            .ok_or(DetectorPanelError::InvalidAuditScores)?;
        calibrator
            .assess_audit_score_with_diagnostic(
                score,
                independent_audit,
                AdequacyDiagnostic::QueryDistributionShift,
            )
            .map_err(DetectorPanelError::Calibration)
    }

    /// Indices and orientation statistics of selected detectors, sorted by self-detectability.
    pub fn selected_detectors(&self) -> &[SelectedDetector] {
        &self.selected
    }

    /// Total detector-vector width required by score.
    pub fn detector_count(&self) -> usize {
        self.detector_count
    }

    /// Aggregation rule frozen at fit time.
    pub fn aggregation(&self) -> DetectorPanelAggregation {
        self.aggregation
    }
}

fn column_values(matrix: &[Vec<f64>], column: usize) -> Vec<f64> {
    matrix.iter().map(|row| row[column]).collect()
}

fn column_means(matrix: &[Vec<f64>], width: usize) -> Vec<f64> {
    (0..width)
        .map(|column| {
            matrix.iter().map(|row| row[column]).sum::<f64>() / matrix.len() as f64
        })
        .collect()
}

fn column_scales(matrix: &[Vec<f64>], means: &[f64], width: usize) -> Vec<f64> {
    (0..width)
        .map(|column| {
            let sum = matrix
                .iter()
                .map(|row| {
                    let centered = row[column] - means[column];
                    centered * centered
                })
                .sum::<f64>();
            (sum / matrix.len() as f64).sqrt()
        })
        .collect()
}

/// Exact empirical AUC for perturbed-positive observations against clean-negative observations.
/// Tied scores contribute one half. All scores must already have been validated as finite.
fn auc_perturbed_higher(clean: &[f64], perturbed: &[f64]) -> f64 {
    let mut labelled = clean
        .iter()
        .copied()
        .map(|score| (score, false))
        .chain(perturbed.iter().copied().map(|score| (score, true)))
        .collect::<Vec<_>>();
    labelled.sort_by(|left, right| left.0.total_cmp(&right.0));

    let positive_count = perturbed.len() as f64;
    let negative_count = clean.len() as f64;
    let mut positive_rank_sum = 0.0_f64;
    let mut start = 0_usize;
    while start < labelled.len() {
        let mut end = start + 1;
        while end < labelled.len() && labelled[end].0 == labelled[start].0 {
            end += 1;
        }
        let average_rank = ((start + 1) as f64 + end as f64) / 2.0;
        let positives_in_group = labelled[start..end]
            .iter()
            .filter(|entry| entry.1)
            .count();
        positive_rank_sum += average_rank * positives_in_group as f64;
        start = end;
    }

    let u_statistic = positive_rank_sum - positive_count * (positive_count + 1.0) / 2.0;
    u_statistic / (positive_count * negative_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_adequacy::AdequacyAssessment;

    fn selection_data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let clean = (0..40)
            .map(|index| {
                let x = index as f64 * 0.05;
                let y = index as f64 * 0.04;
                let quiet = (index % 5) as f64 * 0.1;
                vec![x, y, quiet]
            })
            .collect::<Vec<_>>();
        let perturbed = clean
            .iter()
            .map(|row| vec![row[0] + 3.0, row[1] - 2.0, row[2]])
            .collect::<Vec<_>>();
        (clean, perturbed)
    }

    fn fit_panel(aggregation: DetectorPanelAggregation) -> DetectorPanelScorer {
        let (clean, perturbed) = selection_data();
        DetectorPanelScorer::fit(&clean, &perturbed, 2, 0.75, 1e-6, aggregation).unwrap()
    }

    #[test]
    fn selection_orients_reverse_detector_and_rejects_uninformative_channel() {
        let panel = fit_panel(DetectorPanelAggregation::MeanPositiveZ);
        let selected = panel.selected_detectors();
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].detector_index, 0);
        assert_eq!(selected[0].orientation, 1);
        assert_eq!(selected[0].detectability_auc, 1.0);
        assert_eq!(selected[1].detector_index, 1);
        assert_eq!(selected[1].orientation, -1);
        assert_eq!(selected[1].detectability_auc, 1.0);
        assert!(!selected.iter().any(|detector| detector.detector_index == 2));
    }

    #[test]
    fn panel_score_reinforces_complementary_channels_and_rejects_bad_vectors() {
        let panel = fit_panel(DetectorPanelAggregation::MeanPositiveZ);
        let (clean, perturbed) = selection_data();
        let clean_score = panel.score(&clean[20]).unwrap();
        let perturbed_score = panel.score(&perturbed[20]).unwrap();
        assert!(perturbed_score > clean_score + 3.0);
        assert!(panel.score(&[0.0, 0.0]).is_none());
        assert!(panel.score(&[0.0, f64::INFINITY, 0.0]).is_none());

        let max_panel = fit_panel(DetectorPanelAggregation::MaxPositiveZ);
        let max_score = max_panel.score(&perturbed[20]).unwrap();
        assert!(max_score >= perturbed_score);
    }

    #[test]
    fn outer_conformal_calibration_is_separate_and_typed() {
        let panel = fit_panel(DetectorPanelAggregation::MeanPositiveZ);
        let mut outer_clean = Vec::new();
        for index in 0..100 {
            let x = (index % 40) as f64 * 0.05;
            let y = (index % 40) as f64 * 0.04;
            let quiet = ((index * 3) % 5) as f64 * 0.1;
            outer_clean.push(panel.score(&[x, y, quiet]).unwrap());
        }
        let calibrator = ConformalAdequacyCalibrator::new(&outer_clean, 0.05).unwrap();
        let (_, perturbed) = selection_data();
        let result = panel.assess(&calibrator, &perturbed[20], true).unwrap();
        assert!(result.p_value <= 0.05);
        assert!(matches!(
            result.assessment,
            AdequacyAssessment::Alarm {
                diagnostic: AdequacyDiagnostic::QueryDistributionShift,
                calibrated: true,
                independent_audit: true,
            }
        ));
    }

    #[test]
    fn panel_selection_fails_closed_on_bad_contracts() {
        let (clean, perturbed) = selection_data();
        assert!(matches!(
            DetectorPanelScorer::fit(
                &clean,
                &perturbed,
                0,
                0.75,
                1e-6,
                DetectorPanelAggregation::MeanPositiveZ,
            ),
            Err(DetectorPanelError::InvalidTopM)
        ));
        assert!(matches!(
            DetectorPanelScorer::fit(
                &clean,
                &perturbed,
                2,
                0.5,
                1e-6,
                DetectorPanelAggregation::MeanPositiveZ,
            ),
            Err(DetectorPanelError::InvalidDetectabilityThreshold)
        ));
        assert!(matches!(
            DetectorPanelScorer::fit(
                &clean,
                &clean,
                2,
                0.75,
                1e-6,
                DetectorPanelAggregation::MeanPositiveZ,
            ),
            Err(DetectorPanelError::NoInformativeDetectors)
        ));
        let mut malformed = clean.clone();
        malformed[0].pop();
        assert!(matches!(
            DetectorPanelScorer::fit(
                &malformed,
                &perturbed,
                2,
                0.75,
                1e-6,
                DetectorPanelAggregation::MeanPositiveZ,
            ),
            Err(DetectorPanelError::InvalidSelectionShape)
        ));
        let mut nonfinite = clean.clone();
        nonfinite[0][0] = f64::NAN;
        assert!(matches!(
            DetectorPanelScorer::fit(
                &nonfinite,
                &perturbed,
                2,
                0.75,
                1e-6,
                DetectorPanelAggregation::MeanPositiveZ,
            ),
            Err(DetectorPanelError::NonFiniteSelectionScore)
        ));
    }
}
