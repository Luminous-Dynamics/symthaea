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
    /// Finite input scores overflowed while computing normalization statistics.
    NumericalOverflow,
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

/// Selected detector/orientation, including the probe families that selected it.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectedDetector {
    /// Column index in the supplied detector-score vectors.
    pub detector_index: usize,
    /// +1 means higher scores aligned with a probe; -1 means lower scores did.
    pub orientation: i8,
    /// Mean score on clean selector data.
    pub clean_mean: f64,
    /// Population standard deviation on clean selector data.
    pub clean_scale: f64,
    /// Best oriented AUC observed in a family that selected this channel.
    pub detectability_auc: f64,
    /// Indices of individually evaluated probe families that selected this orientation.
    pub probe_family_indices: Vec<usize>,
}

/// Auditable coverage result for one separately evaluated self-perturbation family.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeFamilyCoverage {
    /// Index of the family in the input slice to fit_multi_probe.
    pub probe_family_index: usize,
    /// Best oriented AUC of any detector for this family, including detectors below threshold.
    pub best_detectability_auc: f64,
    /// Number of detector channels meeting the detectability threshold before top-m truncation.
    pub candidates_above_threshold: usize,
    /// Number of selected detector/orientation channels retained for this family.
    pub selected_channels: usize,
    /// True when at least one channel for this probe family survived selection.
    pub covered: bool,
}

/// Frozen reference-free detector panel. Fit once, then calibrate its aggregate independently.
#[derive(Clone, Debug, PartialEq)]
pub struct DetectorPanelScorer {
    detector_count: usize,
    selected: Vec<SelectedDetector>,
    probe_family_coverage: Vec<ProbeFamilyCoverage>,
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
        Self::fit_multi_probe(
            clean_selection_scores,
            &[perturbed_selection_scores.to_vec()],
            top_m,
            min_detectability_auc,
            scale_floor,
            aggregation,
        )
    }

    /// Select detector/orientation channels separately for each predeclared probe family.
    ///
    /// Each family is evaluated independently; scores are never pooled across perturbation types
    /// before measuring AUC. This prevents a strong response in one family from hiding a
    /// specialist that responds in the opposite direction or only to another family. At most
    /// top_m_per_family channels are selected within each family. Equivalent detector/orientation
    /// pairs are merged, retaining every family index that selected the pair.
    ///
    /// A family with no selected channel is reported as uncovered in probe_family_coverage().
    /// At least one family must select a detector overall, but an uncovered family is not treated
    /// as an error: it is a limitation the benchmark caller must resolve or explicitly accept.
    ///
    /// Selection inputs may use self-generated probes only; evaluation labels and confirmatory
    /// shift outcomes must not be used. Outer calibration must remain separate and independent.
    pub fn fit_multi_probe(
        clean_selection_scores: &[Vec<f64>],
        perturbed_selection_families: &[Vec<Vec<f64>>],
        top_m_per_family: usize,
        min_detectability_auc: f64,
        scale_floor: f64,
        aggregation: DetectorPanelAggregation,
    ) -> Result<Self, DetectorPanelError> {
        if clean_selection_scores.len() < 2
            || perturbed_selection_families.is_empty()
            || perturbed_selection_families
                .iter()
                .any(|family| family.len() < 2)
        {
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
            || !clean_selection_scores
                .iter()
                .all(|row| row.len() == detector_count)
            || perturbed_selection_families
                .iter()
                .any(|family| family.iter().any(|row| row.len() != detector_count))
        {
            return Err(DetectorPanelError::InvalidSelectionShape);
        }
        if top_m_per_family == 0 || top_m_per_family > detector_count {
            return Err(DetectorPanelError::InvalidTopM);
        }
        if clean_selection_scores
            .iter()
            .flatten()
            .chain(perturbed_selection_families.iter().flatten().flatten())
            .any(|score| !score.is_finite())
        {
            return Err(DetectorPanelError::NonFiniteSelectionScore);
        }

        let clean_means = column_means(clean_selection_scores, detector_count)
            .ok_or(DetectorPanelError::NumericalOverflow)?;
        let clean_scales = column_scales(clean_selection_scores, &clean_means, detector_count)
            .ok_or(DetectorPanelError::NumericalOverflow)?;
        let clean_columns = (0..detector_count)
            .map(|index| column_values(clean_selection_scores, index))
            .collect::<Vec<_>>();

        let mut selected = Vec::<SelectedDetector>::new();
        let mut probe_family_coverage = Vec::with_capacity(perturbed_selection_families.len());

        for (probe_family_index, perturbed_scores) in
            perturbed_selection_families.iter().enumerate()
        {
            let perturbed_columns = (0..detector_count)
                .map(|index| column_values(perturbed_scores, index))
                .collect::<Vec<_>>();
            let mut candidates = Vec::<SelectedDetector>::with_capacity(detector_count);
            let mut best_detectability_auc = 0.5_f64;

            for detector_index in 0..detector_count {
                let raw_auc = auc_perturbed_higher(
                    &clean_columns[detector_index],
                    &perturbed_columns[detector_index],
                );
                let (orientation, detectability_auc) = if raw_auc >= 0.5 {
                    (1_i8, raw_auc)
                } else {
                    (-1_i8, 1.0 - raw_auc)
                };
                best_detectability_auc = best_detectability_auc.max(detectability_auc);
                if detectability_auc >= min_detectability_auc {
                    candidates.push(SelectedDetector {
                        detector_index,
                        orientation,
                        clean_mean: clean_means[detector_index],
                        clean_scale: clean_scales[detector_index],
                        detectability_auc,
                        probe_family_indices: vec![probe_family_index],
                    });
                }
            }

            let candidates_above_threshold = candidates.len();
            candidates.sort_by(|left, right| {
                right
                    .detectability_auc
                    .total_cmp(&left.detectability_auc)
                    .then_with(|| left.detector_index.cmp(&right.detector_index))
            });
            candidates.truncate(top_m_per_family);
            let selected_channels = candidates.len();

            for candidate in candidates {
                if let Some(existing) = selected.iter_mut().find(|existing| {
                    existing.detector_index == candidate.detector_index
                        && existing.orientation == candidate.orientation
                }) {
                    if !existing.probe_family_indices.contains(&probe_family_index) {
                        existing.probe_family_indices.push(probe_family_index);
                    }
                    existing.detectability_auc =
                        existing.detectability_auc.max(candidate.detectability_auc);
                } else {
                    selected.push(candidate);
                }
            }

            probe_family_coverage.push(ProbeFamilyCoverage {
                probe_family_index,
                best_detectability_auc,
                candidates_above_threshold,
                selected_channels,
                covered: selected_channels > 0,
            });
        }

        if selected.is_empty() {
            return Err(DetectorPanelError::NoInformativeDetectors);
        }
        selected.sort_by(|left, right| {
            right
                .detectability_auc
                .total_cmp(&left.detectability_auc)
                .then_with(|| left.detector_index.cmp(&right.detector_index))
                .then_with(|| left.orientation.cmp(&right.orientation))
        });

        Ok(Self {
            detector_count,
            selected,
            probe_family_coverage,
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
            let oriented_z = selected.orientation as f64 * (score - selected.clean_mean)
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
            DetectorPanelAggregation::MaxPositiveZ => {
                positive_z_scores.iter().copied().fold(0.0_f64, f64::max)
            }
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

    /// Coverage summary for each separately evaluated probe family.
    pub fn probe_family_coverage(&self) -> &[ProbeFamilyCoverage] {
        &self.probe_family_coverage
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

fn column_means(matrix: &[Vec<f64>], width: usize) -> Option<Vec<f64>> {
    if matrix.is_empty() || width == 0 || matrix.iter().any(|row| row.len() != width) {
        return None;
    }
    let mut sums = vec![0.0_f64; width];
    for row in matrix {
        for (sum, value) in sums.iter_mut().zip(row.iter()) {
            *sum += *value;
            if !sum.is_finite() {
                return None;
            }
        }
    }
    let means = sums
        .into_iter()
        .map(|sum| sum / matrix.len() as f64)
        .collect::<Vec<_>>();
    means.iter().all(|mean| mean.is_finite()).then_some(means)
}

fn column_scales(matrix: &[Vec<f64>], means: &[f64], width: usize) -> Option<Vec<f64>> {
    if matrix.is_empty()
        || width == 0
        || means.len() != width
        || matrix.iter().any(|row| row.len() != width)
    {
        return None;
    }
    let mut sums_of_squares = vec![0.0_f64; width];
    for row in matrix {
        for column in 0..width {
            let centered = row[column] - means[column];
            let square = centered * centered;
            if !centered.is_finite() || !square.is_finite() {
                return None;
            }
            sums_of_squares[column] += square;
            if !sums_of_squares[column].is_finite() {
                return None;
            }
        }
    }
    let scales = sums_of_squares
        .into_iter()
        .map(|sum| (sum / matrix.len() as f64).sqrt())
        .collect::<Vec<_>>();
    scales
        .iter()
        .all(|scale| scale.is_finite())
        .then_some(scales)
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
        let mut positives_in_group = 0_usize;
        for (_, positive) in &labelled[start..end] {
            if *positive {
                positives_in_group += 1;
            }
        }
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
    fn self_perturbation_mismatch_can_hide_an_unselected_operational_signal() {
        let panel = fit_panel(DetectorPanelAggregation::MeanPositiveZ);
        let mut operational_scores = vec![0.0; panel.detector_count()];
        for selected in panel.selected_detectors() {
            operational_scores[selected.detector_index] = selected.clean_mean;
        }
        // The probe perturbs only detectors 0 and 1, so it selects those two. This mock
        // operational change affects only detector 2, which selection correctly cannot infer.
        operational_scores[2] = 1000.0;
        assert_eq!(panel.score(&operational_scores), Some(0.0));
        assert!(
            !panel
                .selected_detectors()
                .iter()
                .any(|detector| detector.detector_index == 2)
        );
    }

    #[test]
    fn multi_probe_selection_preserves_specialists_and_reports_uncovered_family() {
        let (clean, _) = selection_data();
        let first_family = clean
            .iter()
            .map(|row| vec![row[0] + 3.0, row[1], row[2]])
            .collect::<Vec<_>>();
        let second_family = clean
            .iter()
            .map(|row| vec![row[0], row[1] - 2.0, row[2]])
            .collect::<Vec<_>>();
        let uninformative_family = clean.clone();
        let pooled_families = first_family
            .iter()
            .chain(second_family.iter())
            .cloned()
            .collect::<Vec<_>>();
        assert!(matches!(
            DetectorPanelScorer::fit(
                &clean,
                &pooled_families,
                1,
                0.8,
                1e-6,
                DetectorPanelAggregation::MeanPositiveZ,
            ),
            Err(DetectorPanelError::NoInformativeDetectors)
        ));

        let panel = DetectorPanelScorer::fit_multi_probe(
            &clean,
            &[first_family, second_family, uninformative_family],
            1,
            0.8,
            1e-6,
            DetectorPanelAggregation::MeanPositiveZ,
        )
        .unwrap();

        let selected = panel.selected_detectors();
        assert!(selected.iter().any(|detector| {
            detector.detector_index == 0
                && detector.orientation == 1
                && detector.probe_family_indices.contains(&0)
        }));
        assert!(selected.iter().any(|detector| {
            detector.detector_index == 1
                && detector.orientation == -1
                && detector.probe_family_indices.contains(&1)
        }));

        let coverage = panel.probe_family_coverage();
        assert_eq!(coverage.len(), 3);
        assert!(coverage[0].covered);
        assert!(coverage[1].covered);
        assert_eq!(coverage[0].selected_channels, 1);
        assert_eq!(coverage[1].selected_channels, 1);
        assert!(!coverage[2].covered);
        assert_eq!(coverage[2].selected_channels, 0);
        assert!((coverage[2].best_detectability_auc - 0.5).abs() < 1e-12);

        let clean_audit_score = panel.score(&clean[20]).unwrap();
        let mut location_audit = clean[20].clone();
        location_audit[0] += 3.0;
        let mut reverse_audit = clean[20].clone();
        reverse_audit[1] -= 2.0;
        assert!(panel.score(&location_audit).unwrap() > clean_audit_score + 1.0);
        assert!(panel.score(&reverse_audit).unwrap() > clean_audit_score + 1.0);
    }

    #[test]
    fn multi_probe_rejects_malformed_family_and_insufficient_family_samples() {
        let (clean, _) = selection_data();
        assert!(matches!(
            DetectorPanelScorer::fit_multi_probe(
                &clean,
                &[],
                1,
                0.75,
                1e-6,
                DetectorPanelAggregation::MeanPositiveZ,
            ),
            Err(DetectorPanelError::InsufficientSelectionSamples)
        ));
        assert!(matches!(
            DetectorPanelScorer::fit_multi_probe(
                &clean,
                &[vec![vec![1.0, 2.0, 3.0]]],
                1,
                0.75,
                1e-6,
                DetectorPanelAggregation::MeanPositiveZ,
            ),
            Err(DetectorPanelError::InsufficientSelectionSamples)
        ));
        let malformed_family = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!(matches!(
            DetectorPanelScorer::fit_multi_probe(
                &clean,
                &[malformed_family],
                1,
                0.75,
                1e-6,
                DetectorPanelAggregation::MeanPositiveZ,
            ),
            Err(DetectorPanelError::InvalidSelectionShape)
        ));
    }

    #[test]
    fn outer_conformal_calibration_is_separate_and_typed() {
        let panel = fit_panel(DetectorPanelAggregation::MeanPositiveZ);
        let mut outer_clean = Vec::new();
        let mut state = 0x61A7_0E5D_93C2_B841_u64;
        for _ in 0..100 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let x = ((state >> 11) as f64 / ((1_u64 << 53) as f64)) * 1.95;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let quiet = ((state >> 11) as f64 / ((1_u64 << 53) as f64)) * 0.4;
            outer_clean.push(panel.score(&[x, x * 0.8, quiet]).unwrap());
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
        assert!(matches!(
            panel.assess(&calibrator, &perturbed[20], false),
            Err(DetectorPanelError::Calibration(
                CalibrationError::AuditNotIndependent
            ))
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
        let overflow = vec![vec![f64::MAX; 3]; 3];
        assert!(matches!(
            DetectorPanelScorer::fit(
                &overflow,
                &perturbed,
                2,
                0.75,
                1e-6,
                DetectorPanelAggregation::MeanPositiveZ,
            ),
            Err(DetectorPanelError::NumericalOverflow)
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
