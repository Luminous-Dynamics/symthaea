// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Split-leakage auditing for BCI window manifests.
//!
//! This module operates on metadata only; it never loads EEG payloads. It is
//! deliberately conservative and is not a substitute for dataset-specific review.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum EvaluationSplit {
    Train,
    Validation,
    Test,
}

impl EvaluationSplit {
    const fn index(self) -> usize {
        match self {
            Self::Train => 0,
            Self::Validation => 1,
            Self::Test => 2,
        }
    }
}

/// Identity and half-open sample range of one model input window.
/// Identifiers must be stable, pseudonymous within a dataset version, and non-empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AssignedWindow {
    pub split: EvaluationSplit,
    pub subject_id: String,
    pub recording_id: String,
    pub start_sample: u64,
    pub end_sample_exclusive: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitAuditConfig {
    /// Require no subject identifier to occur in more than one split.
    pub require_subject_disjoint_across_splits: bool,
    /// Require no (subject, recording/session) pair to occur in more than one split.
    pub require_recording_disjoint_across_splits: bool,
    /// Minimum unused sample gap between different-split windows from the same recording.
    /// This catches leakage caused by overlapping or nearly adjacent filtered windows.
    pub guard_samples: u64,
    pub require_all_splits_nonempty: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SplitAuditError {
    EmptyManifest,
    EmptyIdentity { index: usize, field: &'static str },
    InvalidWindowRange { index: usize, start: u64, end_exclusive: u64 },
    SubjectLeakage { subject_id: String, first_split: EvaluationSplit, second_split: EvaluationSplit },
    RecordingLeakage { subject_id: String, recording_id: String, first_split: EvaluationSplit, second_split: EvaluationSplit },
    TemporalLeakage { earlier_index: usize, later_index: usize, guard_samples: u64 },
    EmptySplit(EvaluationSplit),
}

impl fmt::Display for SplitAuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for SplitAuditError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitAuditReport {
    pub window_count: usize,
    pub unique_subject_count: usize,
    pub unique_recording_count: usize,
    pub train_window_count: usize,
    pub validation_window_count: usize,
    pub test_window_count: usize,
}

/// Audits only manifest metadata; inputs are not modified.
/// Complexity is O(n log n) for n windows.
pub fn audit_split_assignments(
    assignments: &[AssignedWindow],
    config: SplitAuditConfig,
) -> Result<SplitAuditReport, SplitAuditError> {
    if assignments.is_empty() {
        return Err(SplitAuditError::EmptyManifest);
    }

    let mut split_subjects: BTreeMap<&str, EvaluationSplit> = BTreeMap::new();
    let mut split_recordings: BTreeMap<(&str, &str), EvaluationSplit> = BTreeMap::new();
    let mut subjects = BTreeSet::new();
    let mut recordings = BTreeSet::new();
    let mut counts = [0usize; 3];

    for (index, item) in assignments.iter().enumerate() {
        if item.subject_id.trim().is_empty() {
            return Err(SplitAuditError::EmptyIdentity { index, field: "subject_id" });
        }
        if item.recording_id.trim().is_empty() {
            return Err(SplitAuditError::EmptyIdentity { index, field: "recording_id" });
        }
        if item.start_sample >= item.end_sample_exclusive {
            return Err(SplitAuditError::InvalidWindowRange {
                index,
                start: item.start_sample,
                end_exclusive: item.end_sample_exclusive,
            });
        }
        counts[item.split.index()] += 1;
        subjects.insert(item.subject_id.as_str());
        recordings.insert((item.subject_id.as_str(), item.recording_id.as_str()));

        let prior_split = *split_subjects.entry(item.subject_id.as_str()).or_insert(item.split);
        if config.require_subject_disjoint_across_splits && prior_split != item.split {
            return Err(SplitAuditError::SubjectLeakage {
                subject_id: item.subject_id.clone(),
                first_split: prior_split,
                second_split: item.split,
            });
        }

        let rec_key = (item.subject_id.as_str(), item.recording_id.as_str());
        let prior_recording_split = *split_recordings.entry(rec_key).or_insert(item.split);
        if config.require_recording_disjoint_across_splits && prior_recording_split != item.split {
            return Err(SplitAuditError::RecordingLeakage {
                subject_id: item.subject_id.clone(),
                recording_id: item.recording_id.clone(),
                first_split: prior_recording_split,
                second_split: item.split,
            });
        }
    }

    if config.require_all_splits_nonempty {
        for split in [EvaluationSplit::Train, EvaluationSplit::Validation, EvaluationSplit::Test] {
            if counts[split.index()] == 0 {
                return Err(SplitAuditError::EmptySplit(split));
            }
        }
    }

    // Subject-disjoint checks above use the first-seen split, which is sufficient
    // to reject any subsequent different-split assignment. For session/window
    // evaluation, inspect same-record windows in start order and enforce a guard gap.
    let mut order: Vec<(usize, &AssignedWindow)> = assignments.iter().enumerate().collect();
    order.sort_by(|(_, a), (_, b)| {
        (&a.subject_id, &a.recording_id, a.start_sample, a.end_sample_exclusive, a.split)
            .cmp(&(&b.subject_id, &b.recording_id, b.start_sample, b.end_sample_exclusive, b.split))
    });

    let mut current_group: Option<(&str, &str)> = None;
    // For each split, retain the largest end of any earlier window in this recording.
    let mut max_end: [Option<(u64, usize)>; 3] = [None, None, None];
    for (original_index, item) in order {
        let group = (item.subject_id.as_str(), item.recording_id.as_str());
        if current_group != Some(group) {
            current_group = Some(group);
            max_end = [None, None, None];
        }

        for other in [EvaluationSplit::Train, EvaluationSplit::Validation, EvaluationSplit::Test] {
            if other == item.split { continue; }
            if let Some((end, prior_index)) = max_end[other.index()] {
                if config.require_recording_disjoint_across_splits {
                    return Err(SplitAuditError::RecordingLeakage {
                        subject_id: item.subject_id.clone(),
                        recording_id: item.recording_id.clone(),
                        first_split: other,
                        second_split: item.split,
                    });
                }
                let overlaps = end > item.start_sample;
                let gap_too_small = !overlaps
                    && item.start_sample.saturating_sub(end) < config.guard_samples;
                if overlaps || gap_too_small {
                    return Err(SplitAuditError::TemporalLeakage {
                        earlier_index: prior_index,
                        later_index: original_index,
                        guard_samples: config.guard_samples,
                    });
                }
            }
        }

        let slot = &mut max_end[item.split.index()];
        if slot.as_ref().is_none_or(|(end, _)| item.end_sample_exclusive > *end) {
            *slot = Some((item.end_sample_exclusive, original_index));
        }
    }

    Ok(SplitAuditReport {
        window_count: assignments.len(),
        unique_subject_count: subjects.len(),
        unique_recording_count: recordings.len(),
        train_window_count: counts[EvaluationSplit::Train.index()],
        validation_window_count: counts[EvaluationSplit::Validation.index()],
        test_window_count: counts[EvaluationSplit::Test.index()],
    })
}


/// Canonical per-run identity. The harness must serialize this schema canonically
/// and compute SHA-256 externally; this dependency-free core validates field
/// completeness and that a scorecard references the exact supplied digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkRunManifest {
    pub schema_version: u16,
    pub dataset_id: String,
    pub dataset_version: String,
    pub dataset_license_id: String,
    pub task_id: String,
    pub primary_metric_id: String,
    pub dataset_manifest_sha256: [u8; 32],
    pub split_manifest_sha256: [u8; 32],
    pub evaluation_protocol_sha256: [u8; 32],
    pub preprocessing_manifest_sha256: [u8; 32],
    pub model_artifact_sha256: [u8; 32],
    pub environment_manifest_sha256: [u8; 32],
    pub code_revision: String,
    pub seed: u64,
}

impl BenchmarkRunManifest {
    pub fn validate(&self) -> Result<(), BenchmarkArtifactError> {
        if self.schema_version != 1 {
            return Err(BenchmarkArtifactError::UnsupportedManifestSchema(self.schema_version));
        }
        for (field, value) in [
            ("dataset_id", self.dataset_id.as_str()),
            ("dataset_version", self.dataset_version.as_str()),
            ("dataset_license_id", self.dataset_license_id.as_str()),
            ("task_id", self.task_id.as_str()),
            ("primary_metric_id", self.primary_metric_id.as_str()),
            ("code_revision", self.code_revision.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(BenchmarkArtifactError::MissingManifestField(field));
            }
        }
        for (field, digest) in [
            ("dataset_manifest_sha256", &self.dataset_manifest_sha256),
            ("split_manifest_sha256", &self.split_manifest_sha256),
            ("evaluation_protocol_sha256", &self.evaluation_protocol_sha256),
            ("preprocessing_manifest_sha256", &self.preprocessing_manifest_sha256),
            ("model_artifact_sha256", &self.model_artifact_sha256),
            ("environment_manifest_sha256", &self.environment_manifest_sha256),
        ] {
            if digest.iter().all(|byte| *byte == 0) {
                return Err(BenchmarkArtifactError::ZeroDigest(field));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricDirection {
    HigherIsBetter,
    LowerIsBetter,
}

/// Integer-scaled measurement avoids float/NaN ambiguity in canonical scorecards.
/// The reported value is estimate_scaled / scale in the declared unit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkMetric {
    pub metric_id: String,
    pub estimate_scaled: i64,
    pub scale: u64,
    pub unit: String,
    pub sample_count: u64,
    pub direction: MetricDirection,
    pub lower_bound_scaled: Option<i64>,
    pub upper_bound_scaled: Option<i64>,
    /// Required when interval bounds are present; identifies the actual method
    /// and confidence/coverage convention recorded in the run protocol.
    pub uncertainty_method_id: Option<String>,
}

impl BenchmarkMetric {
    fn validate(&self) -> Result<(), BenchmarkArtifactError> {
        if self.metric_id.trim().is_empty() {
            return Err(BenchmarkArtifactError::InvalidMetric("metric_id must be non-empty"));
        }
        if self.unit.trim().is_empty() {
            return Err(BenchmarkArtifactError::InvalidMetric("unit must be non-empty"));
        }
        if self.scale == 0 {
            return Err(BenchmarkArtifactError::InvalidMetric("scale must be non-zero"));
        }
        if self.sample_count == 0 {
            return Err(BenchmarkArtifactError::InvalidMetric("sample_count must be non-zero"));
        }
        match (self.lower_bound_scaled, self.upper_bound_scaled, self.uncertainty_method_id.as_deref()) {
            (Some(lower), Some(upper), Some(method))
                if lower <= self.estimate_scaled && self.estimate_scaled <= upper && !method.trim().is_empty() => {}
            (None, None, None) => {}
            _ => return Err(BenchmarkArtifactError::InvalidMetric(
                "uncertainty bounds must be paired, contain estimate, and name the method",
            )),
        }
        Ok(())
    }
}

/// Scorecard duplicates the critical provenance bindings deliberately: validation
/// rejects a result if any dataset, split, protocol, preprocessing, model, code, or
/// environment identity differs from the manifest used to run it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BenchmarkScorecard {
    pub schema_version: u16,
    pub run_manifest_sha256: [u8; 32],
    pub dataset_manifest_sha256: [u8; 32],
    pub split_manifest_sha256: [u8; 32],
    pub evaluation_protocol_sha256: [u8; 32],
    pub preprocessing_manifest_sha256: [u8; 32],
    pub model_artifact_sha256: [u8; 32],
    pub environment_manifest_sha256: [u8; 32],
    pub code_revision: String,
    pub metrics: Vec<BenchmarkMetric>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BenchmarkArtifactError {
    UnsupportedManifestSchema(u16),
    UnsupportedScorecardSchema(u16),
    MissingManifestField(&'static str),
    ZeroDigest(&'static str),
    InvalidMetric(&'static str),
    DuplicateMetric(String),
    MissingPrimaryMetric(String),
    ManifestDigestMismatch,
    IdentityMismatch(&'static str),
    EmptyMetrics,
}

impl fmt::Display for BenchmarkArtifactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for BenchmarkArtifactError {}

impl BenchmarkScorecard {
    /// `canonical_manifest_sha256` must be SHA-256 of the canonical serialized
    /// manifest bytes, computed by the caller's reviewed serialization/hash layer.
    pub fn validate_against(
        &self,
        manifest: &BenchmarkRunManifest,
        canonical_manifest_sha256: [u8; 32],
    ) -> Result<(), BenchmarkArtifactError> {
        manifest.validate()?;
        if self.schema_version != 1 {
            return Err(BenchmarkArtifactError::UnsupportedScorecardSchema(self.schema_version));
        }
        if canonical_manifest_sha256.iter().all(|byte| *byte == 0)
            || self.run_manifest_sha256 != canonical_manifest_sha256
        {
            return Err(BenchmarkArtifactError::ManifestDigestMismatch);
        }
        for (field, scorecard_digest, manifest_digest) in [
            ("dataset_manifest_sha256", &self.dataset_manifest_sha256, &manifest.dataset_manifest_sha256),
            ("split_manifest_sha256", &self.split_manifest_sha256, &manifest.split_manifest_sha256),
            ("evaluation_protocol_sha256", &self.evaluation_protocol_sha256, &manifest.evaluation_protocol_sha256),
            ("preprocessing_manifest_sha256", &self.preprocessing_manifest_sha256, &manifest.preprocessing_manifest_sha256),
            ("model_artifact_sha256", &self.model_artifact_sha256, &manifest.model_artifact_sha256),
            ("environment_manifest_sha256", &self.environment_manifest_sha256, &manifest.environment_manifest_sha256),
        ] {
            if scorecard_digest != manifest_digest {
                return Err(BenchmarkArtifactError::IdentityMismatch(field));
            }
        }
        if self.code_revision != manifest.code_revision {
            return Err(BenchmarkArtifactError::IdentityMismatch("code_revision"));
        }
        if self.metrics.is_empty() {
            return Err(BenchmarkArtifactError::EmptyMetrics);
        }
        let mut metric_ids = BTreeSet::new();
        let mut has_primary = false;
        for metric in &self.metrics {
            metric.validate()?;
            if !metric_ids.insert(metric.metric_id.as_str()) {
                return Err(BenchmarkArtifactError::DuplicateMetric(metric.metric_id.clone()));
            }
            has_primary |= metric.metric_id == manifest.primary_metric_id;
        }
        if !has_primary {
            return Err(BenchmarkArtifactError::MissingPrimaryMetric(manifest.primary_metric_id.clone()));
        }
        Ok(())
    }
}


/// A participant × session × recording-context cell in a frozen evaluation manifest.
/// IDs must be pseudonymous, stable, and non-empty.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct EvaluationCell {
    pub participant_id: String,
    pub session_id: String,
    pub context_id: String,
}

impl EvaluationCell {
    fn validate(&self) -> Result<(), CellBalancedAccuracyError> {
        for (field, value) in [
            ("participant_id", self.participant_id.as_str()),
            ("session_id", self.session_id.as_str()),
            ("context_id", self.context_id.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(CellBalancedAccuracyError::EmptyCellIdentity(field));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabeledPrediction {
    pub cell: EvaluationCell,
    pub target_class: u16,
    pub predicted_class: u16,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClassRecall {
    pub class_id: u16,
    pub support: u64,
    pub correct: u64,
    /// Recall scaled by 1,000,000, rounded down.
    pub recall_ppm: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellBalancedAccuracy {
    pub cell: EvaluationCell,
    pub observation_count: u64,
    /// Macro average of required-class recalls, scaled by 1,000,000.
    pub balanced_accuracy_ppm: u32,
    pub per_class: Vec<ClassRecall>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellMeanBalancedAccuracyReport {
    pub observation_count: u64,
    pub cell_count: u64,
    /// Equal-weight mean of cell-level balanced accuracies; scale 1,000,000.
    pub cell_mean_balanced_accuracy_ppm: u32,
    pub cells: Vec<CellBalancedAccuracy>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CellBalancedAccuracyError {
    EmptyPredictions,
    EmptyClassSet,
    EmptyExpectedCells,
    EmptyCellIdentity(&'static str),
    DuplicateClass(u16),
    DuplicateExpectedCell(EvaluationCell),
    MissingCell(EvaluationCell),
    UnexpectedCell(EvaluationCell),
    UnknownTargetClass(u16),
    UnknownPredictedClass(u16),
    MissingClassSupport { cell: EvaluationCell, class_id: u16 },
    CounterOverflow,
}

impl fmt::Display for CellBalancedAccuracyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for CellBalancedAccuracyError {}

/// Reproduces an equal-weight cell-mean balanced-accuracy aggregation.
///
/// The caller must supply the complete expected cell inventory from the frozen
/// evaluation manifest. Each cell must contain at least one ground-truth sample
/// for every declared class. This prevents missing cells or classes from silently
/// disappearing from the score. The function is metadata-only and has no dataset
/// loader. Scores are fixed-point parts-per-million for deterministic reporting.
pub fn evaluate_cell_mean_balanced_accuracy(
    predictions: &[LabeledPrediction],
    class_ids: &[u16],
    expected_cells: &[EvaluationCell],
) -> Result<CellMeanBalancedAccuracyReport, CellBalancedAccuracyError> {
    if predictions.is_empty() {
        return Err(CellBalancedAccuracyError::EmptyPredictions);
    }
    if class_ids.is_empty() {
        return Err(CellBalancedAccuracyError::EmptyClassSet);
    }
    if expected_cells.is_empty() {
        return Err(CellBalancedAccuracyError::EmptyExpectedCells);
    }

    let mut class_set = BTreeSet::new();
    for &class_id in class_ids {
        if !class_set.insert(class_id) {
            return Err(CellBalancedAccuracyError::DuplicateClass(class_id));
        }
    }

    let mut expected_set = BTreeSet::new();
    for cell in expected_cells {
        cell.validate()?;
        if !expected_set.insert(cell.clone()) {
            return Err(CellBalancedAccuracyError::DuplicateExpectedCell(cell.clone()));
        }
    }

    // For each cell/class, store (ground-truth support, correct predictions).
    let mut grouped: BTreeMap<EvaluationCell, BTreeMap<u16, (u64, u64)>> = BTreeMap::new();
    for prediction in predictions {
        prediction.cell.validate()?;
        if !class_set.contains(&prediction.target_class) {
            return Err(CellBalancedAccuracyError::UnknownTargetClass(prediction.target_class));
        }
        if !class_set.contains(&prediction.predicted_class) {
            return Err(CellBalancedAccuracyError::UnknownPredictedClass(prediction.predicted_class));
        }
        if !expected_set.contains(&prediction.cell) {
            return Err(CellBalancedAccuracyError::UnexpectedCell(prediction.cell.clone()));
        }

        let per_class = grouped.entry(prediction.cell.clone()).or_insert_with(|| {
            class_ids.iter().map(|id| (*id, (0_u64, 0_u64))).collect()
        });
        let counts = per_class.get_mut(&prediction.target_class)
            .ok_or(CellBalancedAccuracyError::UnknownTargetClass(prediction.target_class))?;
        counts.0 = counts.0.checked_add(1).ok_or(CellBalancedAccuracyError::CounterOverflow)?;
        if prediction.target_class == prediction.predicted_class {
            counts.1 = counts.1.checked_add(1).ok_or(CellBalancedAccuracyError::CounterOverflow)?;
        }
    }

    let observation_count = u64::try_from(predictions.len())
        .map_err(|_| CellBalancedAccuracyError::CounterOverflow)?;
    let class_count = u128::try_from(class_ids.len())
        .map_err(|_| CellBalancedAccuracyError::CounterOverflow)?;
    let mut cells = Vec::with_capacity(expected_set.len());
    let mut total_cell_score_ppm = 0_u128;

    for cell in &expected_set {
        let per_class_counts = grouped.get(cell)
            .ok_or_else(|| CellBalancedAccuracyError::MissingCell(cell.clone()))?;
        let mut class_reports = Vec::with_capacity(class_ids.len());
        let mut sum_recall_ppm = 0_u128;
        let mut cell_observations = 0_u64;

        for &class_id in class_ids {
            let &(support, correct) = per_class_counts.get(&class_id)
                .ok_or(CellBalancedAccuracyError::UnknownTargetClass(class_id))?;
            if support == 0 {
                return Err(CellBalancedAccuracyError::MissingClassSupport {
                    cell: cell.clone(),
                    class_id,
                });
            }
            cell_observations = cell_observations.checked_add(support)
                .ok_or(CellBalancedAccuracyError::CounterOverflow)?;
            let recall_ppm = u32::try_from(
                (u128::from(correct) * 1_000_000_u128) / u128::from(support)
            ).map_err(|_| CellBalancedAccuracyError::CounterOverflow)?;
            sum_recall_ppm = sum_recall_ppm.checked_add(u128::from(recall_ppm))
                .ok_or(CellBalancedAccuracyError::CounterOverflow)?;
            class_reports.push(ClassRecall { class_id, support, correct, recall_ppm });
        }

        let balanced_accuracy_ppm = u32::try_from(sum_recall_ppm / class_count)
            .map_err(|_| CellBalancedAccuracyError::CounterOverflow)?;
        total_cell_score_ppm = total_cell_score_ppm
            .checked_add(u128::from(balanced_accuracy_ppm))
            .ok_or(CellBalancedAccuracyError::CounterOverflow)?;
        cells.push(CellBalancedAccuracy {
            cell: cell.clone(),
            observation_count: cell_observations,
            balanced_accuracy_ppm,
            per_class: class_reports,
        });
    }

    let cell_count = u64::try_from(cells.len()).map_err(|_| CellBalancedAccuracyError::CounterOverflow)?;
    let cell_mean_balanced_accuracy_ppm = u32::try_from(total_cell_score_ppm / u128::from(cell_count))
        .map_err(|_| CellBalancedAccuracyError::CounterOverflow)?;

    Ok(CellMeanBalancedAccuracyReport {
        observation_count,
        cell_count,
        cell_mean_balanced_accuracy_ppm,
        cells,
    })
}


/// Track 02 reference-semantics variant: balanced accuracy over the class set
/// observed in each cell (union of target and predicted labels), then equal-weight
/// mean across the frozen expected cell inventory.
///
/// The organizer's Python metric computes recalls with floating-point arithmetic.
/// This dependency-free helper reports fixed-point ppm, flooring each recall, each
/// cell macro-average, and the final mean; its semantic aggregation matches the
/// referenced objective, but its integer output is not bit-for-bit identical.
pub fn evaluate_track02_reference_compatible_cell_mean(
    predictions: &[LabeledPrediction],
    allowed_class_ids: &[u16],
    expected_cells: &[EvaluationCell],
) -> Result<CellMeanBalancedAccuracyReport, CellBalancedAccuracyError> {
    if predictions.is_empty() {
        return Err(CellBalancedAccuracyError::EmptyPredictions);
    }
    if allowed_class_ids.is_empty() {
        return Err(CellBalancedAccuracyError::EmptyClassSet);
    }
    if expected_cells.is_empty() {
        return Err(CellBalancedAccuracyError::EmptyExpectedCells);
    }

    let mut class_set = BTreeSet::new();
    for &class_id in allowed_class_ids {
        if !class_set.insert(class_id) {
            return Err(CellBalancedAccuracyError::DuplicateClass(class_id));
        }
    }

    let mut expected_set = BTreeSet::new();
    for cell in expected_cells {
        cell.validate()?;
        if !expected_set.insert(cell.clone()) {
            return Err(CellBalancedAccuracyError::DuplicateExpectedCell(cell.clone()));
        }
    }

    let mut grouped: BTreeMap<EvaluationCell, Vec<(u16, u16)>> = BTreeMap::new();
    for item in predictions {
        item.cell.validate()?;
        if !class_set.contains(&item.target_class) {
            return Err(CellBalancedAccuracyError::UnknownTargetClass(item.target_class));
        }
        if !class_set.contains(&item.predicted_class) {
            return Err(CellBalancedAccuracyError::UnknownPredictedClass(item.predicted_class));
        }
        if !expected_set.contains(&item.cell) {
            return Err(CellBalancedAccuracyError::UnexpectedCell(item.cell.clone()));
        }
        grouped.entry(item.cell.clone())
            .or_default()
            .push((item.target_class, item.predicted_class));
    }

    let observation_count = u64::try_from(predictions.len())
        .map_err(|_| CellBalancedAccuracyError::CounterOverflow)?;
    let mut cells = Vec::with_capacity(expected_set.len());
    let mut total_cell_score_ppm = 0_u128;

    for cell in &expected_set {
        let rows = grouped.get(cell)
            .ok_or_else(|| CellBalancedAccuracyError::MissingCell(cell.clone()))?;
        let mut observed_classes = BTreeSet::new();
        let mut class_counts: BTreeMap<u16, (u64, u64)> = BTreeMap::new();

        for &(target, predicted) in rows {
            observed_classes.insert(target);
            observed_classes.insert(predicted);
            let target_counts = class_counts.entry(target).or_insert((0, 0));
            target_counts.0 = target_counts.0.checked_add(1)
                .ok_or(CellBalancedAccuracyError::CounterOverflow)?;
            if target == predicted {
                target_counts.1 = target_counts.1.checked_add(1)
                    .ok_or(CellBalancedAccuracyError::CounterOverflow)?;
            }
            // Ensure predicted-only classes are retained in the per-cell union.
            class_counts.entry(predicted).or_insert((0, 0));
        }

        let class_count = u128::try_from(observed_classes.len())
            .map_err(|_| CellBalancedAccuracyError::CounterOverflow)?;
        if class_count == 0 {
            return Err(CellBalancedAccuracyError::MissingCell(cell.clone()));
        }
        let mut per_class = Vec::with_capacity(observed_classes.len());
        let mut sum_recall_ppm = 0_u128;

        for class_id in observed_classes {
            let (support, correct) = class_counts.get(&class_id).copied().unwrap_or((0, 0));
            let recall_ppm = if support == 0 {
                0
            } else {
                u32::try_from(
                    (u128::from(correct) * 1_000_000_u128) / u128::from(support)
                ).map_err(|_| CellBalancedAccuracyError::CounterOverflow)?
            };
            sum_recall_ppm = sum_recall_ppm
                .checked_add(u128::from(recall_ppm))
                .ok_or(CellBalancedAccuracyError::CounterOverflow)?;
            per_class.push(ClassRecall {
                class_id,
                support,
                correct,
                recall_ppm,
            });
        }

        let balanced_accuracy_ppm = u32::try_from(sum_recall_ppm / class_count)
            .map_err(|_| CellBalancedAccuracyError::CounterOverflow)?;
        total_cell_score_ppm = total_cell_score_ppm
            .checked_add(u128::from(balanced_accuracy_ppm))
            .ok_or(CellBalancedAccuracyError::CounterOverflow)?;
        cells.push(CellBalancedAccuracy {
            cell: cell.clone(),
            observation_count: u64::try_from(rows.len())
                .map_err(|_| CellBalancedAccuracyError::CounterOverflow)?,
            balanced_accuracy_ppm,
            per_class,
        });
    }

    let cell_count = u64::try_from(cells.len())
        .map_err(|_| CellBalancedAccuracyError::CounterOverflow)?;
    let cell_mean_balanced_accuracy_ppm =
        u32::try_from(total_cell_score_ppm / u128::from(cell_count))
            .map_err(|_| CellBalancedAccuracyError::CounterOverflow)?;

    Ok(CellMeanBalancedAccuracyReport {
        observation_count,
        cell_count,
        cell_mean_balanced_accuracy_ppm,
        cells,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window(split: EvaluationSplit, subject: &str, recording: &str, start: u64, end: u64) -> AssignedWindow {
        AssignedWindow {
            split,
            subject_id: subject.into(),
            recording_id: recording.into(),
            start_sample: start,
            end_sample_exclusive: end,
        }
    }

    fn config() -> SplitAuditConfig {
        SplitAuditConfig {
            require_subject_disjoint_across_splits: false,
            require_recording_disjoint_across_splits: false,
            guard_samples: 10,
            require_all_splits_nonempty: true,
        }
    }

    fn run_manifest() -> BenchmarkRunManifest {
        BenchmarkRunManifest {
            schema_version: 1,
            dataset_id: "physionet-eegmmidb".into(),
            dataset_version: "1.0.0".into(),
            dataset_license_id: "physionet-terms-v1".into(),
            task_id: "motor-imagery".into(),
            primary_metric_id: "balanced_accuracy".into(),
            dataset_manifest_sha256: [1; 32],
            split_manifest_sha256: [2; 32],
            evaluation_protocol_sha256: [3; 32],
            preprocessing_manifest_sha256: [4; 32],
            model_artifact_sha256: [5; 32],
            environment_manifest_sha256: [6; 32],
            code_revision: "deadbeef".into(),
            seed: 42,
        }
    }

    fn scorecard(manifest: &BenchmarkRunManifest) -> BenchmarkScorecard {
        BenchmarkScorecard {
            schema_version: 1,
            run_manifest_sha256: [7; 32],
            dataset_manifest_sha256: manifest.dataset_manifest_sha256,
            split_manifest_sha256: manifest.split_manifest_sha256,
            evaluation_protocol_sha256: manifest.evaluation_protocol_sha256,
            preprocessing_manifest_sha256: manifest.preprocessing_manifest_sha256,
            model_artifact_sha256: manifest.model_artifact_sha256,
            environment_manifest_sha256: manifest.environment_manifest_sha256,
            code_revision: manifest.code_revision.clone(),
            metrics: vec![BenchmarkMetric {
                metric_id: "balanced_accuracy".into(),
                estimate_scaled: 765_000,
                scale: 1_000_000,
                unit: "fraction".into(),
                sample_count: 90,
                direction: MetricDirection::HigherIsBetter,
                lower_bound_scaled: Some(710_000),
                upper_bound_scaled: Some(810_000),
                uncertainty_method_id: Some("participant-cluster-bootstrap-95-v1".into()),
            }],
        }
    }


    fn eval_cell(participant: &str, session: &str, context: &str) -> EvaluationCell {
        EvaluationCell {
            participant_id: participant.into(),
            session_id: session.into(),
            context_id: context.into(),
        }
    }

    fn prediction(cell: &EvaluationCell, target: u16, predicted: u16) -> LabeledPrediction {
        LabeledPrediction {
            cell: cell.clone(),
            target_class: target,
            predicted_class: predicted,
        }
    }

    #[test]
    fn track02_reference_semantics_omit_classes_absent_from_target_and_prediction() {
        let cell = eval_cell("sub-01", "session-01", "graz");
        let predictions = vec![
            prediction(&cell, 0, 0),
            prediction(&cell, 1, 1),
            prediction(&cell, 1, 0),
        ];
        let report = evaluate_track02_reference_compatible_cell_mean(
            &predictions, &[0, 1, 2], &[cell]
        ).unwrap();
        // Class 2 is absent from both target and prediction and is not included
        // by the organizer's observed-class union. The active recalls are 1.0, 0.5.
        assert_eq!(report.cell_mean_balanced_accuracy_ppm, 750_000);
        assert_eq!(report.cells[0].per_class.len(), 2);
    }

    #[test]
    fn track02_reference_semantics_count_predicted_only_class_as_zero_recall() {
        let cell = eval_cell("sub-01", "session-01", "brainhero");
        let predictions = vec![
            prediction(&cell, 0, 0),
            prediction(&cell, 0, 2),
        ];
        let report = evaluate_track02_reference_compatible_cell_mean(
            &predictions, &[0, 1, 2], &[cell]
        ).unwrap();
        // Active classes are 0 and 2. Class 0 recall is 0.5; predicted-only
        // class 2 has zero support and zero recall, so the macro score is 0.25.
        assert_eq!(report.cell_mean_balanced_accuracy_ppm, 250_000);
        assert_eq!(report.cells[0].per_class[1].class_id, 2);
        assert_eq!(report.cells[0].per_class[1].support, 0);
        assert_eq!(report.cells[0].per_class[1].recall_ppm, 0);
    }

    #[test]
    fn cell_mean_balanced_accuracy_weights_cells_equally_not_by_window_count() {
        let large = eval_cell("sub-01", "session-01", "interface-a");
        let small = eval_cell("sub-02", "session-02", "interface-b");
        let predictions = vec![
            prediction(&large, 0, 0),
            prediction(&large, 0, 0),
            prediction(&large, 1, 0),
            prediction(&large, 1, 0),
            prediction(&small, 0, 0),
            prediction(&small, 1, 1),
        ];
        let report = evaluate_cell_mean_balanced_accuracy(&predictions, &[0, 1], &[small.clone(), large.clone()]).unwrap();
        assert_eq!(report.cell_count, 2);
        assert_eq!(report.observation_count, 6);
        assert_eq!(report.cell_mean_balanced_accuracy_ppm, 750_000);
        assert_eq!(report.cells[0].cell, large);
        assert_eq!(report.cells[0].balanced_accuracy_ppm, 500_000);
        assert_eq!(report.cells[1].cell, small);
        assert_eq!(report.cells[1].balanced_accuracy_ppm, 1_000_000);
    }

    #[test]
    fn cell_scorer_rejects_missing_classes_and_missing_expected_cells() {
        let first = eval_cell("sub-01", "session-01", "interface-a");
        let missing = eval_cell("sub-02", "session-02", "interface-b");
        let one_class = vec![prediction(&first, 0, 0), prediction(&first, 0, 0)];
        assert!(matches!(
            evaluate_cell_mean_balanced_accuracy(&one_class, &[0, 1], &[first.clone()]),
            Err(CellBalancedAccuracyError::MissingClassSupport { class_id: 1, .. })
        ));
        let all_classes = vec![prediction(&first, 0, 0), prediction(&first, 1, 1)];
        assert_eq!(
            evaluate_cell_mean_balanced_accuracy(&all_classes, &[0, 1], &[first, missing.clone()]),
            Err(CellBalancedAccuracyError::MissingCell(missing))
        );
    }

    #[test]
    fn cell_scorer_rejects_unexpected_cells_duplicate_classes_and_unknown_labels() {
        let expected = eval_cell("sub-01", "session-01", "interface-a");
        let unexpected = eval_cell("sub-02", "session-02", "interface-b");
        let one = vec![prediction(&unexpected, 0, 0), prediction(&unexpected, 1, 1)];
        assert_eq!(
            evaluate_cell_mean_balanced_accuracy(&one, &[0, 1], &[expected.clone()]),
            Err(CellBalancedAccuracyError::UnexpectedCell(unexpected))
        );
        let one = vec![prediction(&expected, 0, 0), prediction(&expected, 1, 1)];
        assert_eq!(
            evaluate_cell_mean_balanced_accuracy(&one, &[0, 0], &[expected.clone()]),
            Err(CellBalancedAccuracyError::DuplicateClass(0))
        );
        let one = vec![prediction(&expected, 2, 0)];
        assert_eq!(
            evaluate_cell_mean_balanced_accuracy(&one, &[0, 1], &[expected]),
            Err(CellBalancedAccuracyError::UnknownTargetClass(2))
        );
    }

    #[test]
    fn cell_score_is_order_invariant_and_requires_complete_cell_identity() {
        let cell = eval_cell("sub-01", "session-01", "interface-a");
        let first = vec![prediction(&cell, 0, 1), prediction(&cell, 1, 1), prediction(&cell, 0, 0), prediction(&cell, 1, 1)];
        let mut reversed = first.clone();
        reversed.reverse();
        let a = evaluate_cell_mean_balanced_accuracy(&first, &[0, 1], &[cell.clone()]).unwrap();
        let b = evaluate_cell_mean_balanced_accuracy(&reversed, &[0, 1], &[cell.clone()]).unwrap();
        assert_eq!(a, b);
        let blank = eval_cell("", "session-01", "interface-a");
        let bad = vec![prediction(&blank, 0, 0), prediction(&blank, 1, 1)];
        assert_eq!(
            evaluate_cell_mean_balanced_accuracy(&bad, &[0, 1], &[blank]),
            Err(CellBalancedAccuracyError::EmptyCellIdentity("participant_id"))
        );
    }

    #[test]
    fn scorecard_binds_to_exact_run_manifest_and_required_metric() {
        let manifest = run_manifest();
        let mut card = scorecard(&manifest);
        assert_eq!(card.validate_against(&manifest, [7; 32]), Ok(()));
        assert_eq!(card.validate_against(&manifest, [8; 32]), Err(BenchmarkArtifactError::ManifestDigestMismatch));
        card.model_artifact_sha256 = [9; 32];
        assert_eq!(card.validate_against(&manifest, [7; 32]), Err(BenchmarkArtifactError::IdentityMismatch("model_artifact_sha256")));
    }

    #[test]
    fn rejects_incomplete_manifest_and_invalid_metric_uncertainty() {
        let mut manifest = run_manifest();
        manifest.dataset_license_id = " ".into();
        assert_eq!(manifest.validate(), Err(BenchmarkArtifactError::MissingManifestField("dataset_license_id")));
        manifest = run_manifest();
        manifest.dataset_manifest_sha256 = [0; 32];
        assert_eq!(manifest.validate(), Err(BenchmarkArtifactError::ZeroDigest("dataset_manifest_sha256")));
        manifest = run_manifest();
        let mut card = scorecard(&manifest);
        card.metrics[0].lower_bound_scaled = Some(800_000);
        assert_eq!(card.validate_against(&manifest, [7; 32]), Err(BenchmarkArtifactError::InvalidMetric("uncertainty bounds must be paired, contain estimate, and name the method")));
    }

    #[test]
    fn rejects_duplicate_metrics_and_missing_primary_metric() {
        let manifest = run_manifest();
        let mut card = scorecard(&manifest);
        card.metrics.push(card.metrics[0].clone());
        assert_eq!(card.validate_against(&manifest, [7; 32]), Err(BenchmarkArtifactError::DuplicateMetric("balanced_accuracy".into())));
        card.metrics = vec![BenchmarkMetric {
            metric_id: "macro_f1".into(),
            estimate_scaled: 800_000,
            scale: 1_000_000,
            unit: "fraction".into(),
            sample_count: 50,
            direction: MetricDirection::HigherIsBetter,
            lower_bound_scaled: None,
            upper_bound_scaled: None,
            uncertainty_method_id: None,
        }];
        assert_eq!(card.validate_against(&manifest, [7; 32]), Err(BenchmarkArtifactError::MissingPrimaryMetric("balanced_accuracy".into())));
    }

    #[test]
    fn accepts_session_split_with_guarded_nonoverlapping_windows() {
        let windows = vec![
            window(EvaluationSplit::Train, "sub-01", "run-01", 0, 100),
            window(EvaluationSplit::Validation, "sub-01", "run-01", 120, 200),
            window(EvaluationSplit::Test, "sub-01", "run-01", 220, 300),
        ];
        let report = audit_split_assignments(&windows, config()).unwrap();
        assert_eq!(report.window_count, 3);
        assert_eq!(report.unique_subject_count, 1);
        assert_eq!(report.train_window_count, 1);
        assert_eq!(report.validation_window_count, 1);
        assert_eq!(report.test_window_count, 1);
    }

    #[test]
    fn rejects_overlapping_windows_across_splits() {
        let windows = vec![
            window(EvaluationSplit::Train, "sub-01", "run-01", 0, 100),
            window(EvaluationSplit::Test, "sub-01", "run-01", 90, 120),
            window(EvaluationSplit::Validation, "sub-02", "run-02", 0, 10),
        ];
        assert!(matches!(audit_split_assignments(&windows, config()), Err(SplitAuditError::TemporalLeakage { .. })));
    }

    #[test]
    fn rejects_windows_inside_guard_band() {
        let windows = vec![
            window(EvaluationSplit::Train, "sub-01", "run-01", 0, 100),
            window(EvaluationSplit::Validation, "sub-01", "run-01", 105, 150),
            window(EvaluationSplit::Test, "sub-02", "run-02", 0, 10),
        ];
        assert!(matches!(audit_split_assignments(&windows, config()), Err(SplitAuditError::TemporalLeakage { guard_samples: 10, .. })));
    }

    #[test]
    fn catches_nested_window_leakage_not_just_adjacent_pair_overlap() {
        let windows = vec![
            window(EvaluationSplit::Train, "sub-01", "run-01", 0, 100),
            window(EvaluationSplit::Train, "sub-01", "run-01", 1, 2),
            window(EvaluationSplit::Test, "sub-01", "run-01", 90, 110),
            window(EvaluationSplit::Validation, "sub-02", "run-02", 0, 10),
        ];
        assert!(matches!(audit_split_assignments(&windows, config()), Err(SplitAuditError::TemporalLeakage { .. })));
    }

    #[test]
    fn rejects_subject_leakage_when_subject_holdout_is_required() {
        let mut c = config();
        c.require_subject_disjoint_across_splits = true;
        let windows = vec![
            window(EvaluationSplit::Train, "sub-01", "run-01", 0, 100),
            window(EvaluationSplit::Validation, "sub-01", "run-02", 0, 100),
            window(EvaluationSplit::Test, "sub-02", "run-01", 0, 100),
        ];
        assert!(matches!(audit_split_assignments(&windows, c), Err(SplitAuditError::SubjectLeakage { .. })));
    }

    #[test]
    fn rejects_recording_leakage_even_when_windows_do_not_overlap() {
        let mut c = config();
        c.require_recording_disjoint_across_splits = true;
        let windows = vec![
            window(EvaluationSplit::Train, "sub-01", "run-01", 0, 100),
            window(EvaluationSplit::Validation, "sub-01", "run-01", 120, 200),
            window(EvaluationSplit::Test, "sub-02", "run-02", 0, 10),
        ];
        assert!(matches!(audit_split_assignments(&windows, c), Err(SplitAuditError::RecordingLeakage { .. })));
    }

    #[test]
    fn rejects_invalid_identities_ranges_and_missing_splits() {
        let mut c = config();
        c.require_all_splits_nonempty = false;
        assert_eq!(audit_split_assignments(&[], c), Err(SplitAuditError::EmptyManifest));
        let mut windows = vec![
            window(EvaluationSplit::Train, "", "run-01", 0, 100),
            window(EvaluationSplit::Validation, "sub-02", "run-02", 0, 10),
            window(EvaluationSplit::Test, "sub-03", "run-03", 0, 10),
        ];
        assert!(matches!(audit_split_assignments(&windows, c), Err(SplitAuditError::EmptyIdentity { field: "subject_id", .. })));
        windows[0] = window(EvaluationSplit::Train, "sub-01", "run-01", 100, 100);
        assert!(matches!(audit_split_assignments(&windows, c), Err(SplitAuditError::InvalidWindowRange { .. })));
        let only_train = vec![window(EvaluationSplit::Train, "sub-01", "run-01", 0, 100)];
        c.require_all_splits_nonempty = true;
        assert_eq!(audit_split_assignments(&only_train, c), Err(SplitAuditError::EmptySplit(EvaluationSplit::Validation)));
    }
}
