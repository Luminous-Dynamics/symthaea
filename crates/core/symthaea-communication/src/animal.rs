//! Grounded animal-communication evaluation and playback governance.

use crate::{CapabilityLevel, CommunicationEvidence, ExpressionAuthorization, ReplicationStatus};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub trait BioacousticEncoder {
    fn provider_id(&self) -> &str;
    fn model_hash(&self) -> &str;
    fn encode(&mut self, mono_audio: &[f32], sample_rate_hz: u32) -> Result<Vec<Vec<f32>>, String>;
}

/// Confusion counts for a single binary detection operating point.
/// Undefined ratios return None instead of silently reporting zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct BinaryDetectionCounts {
    pub true_positives: u64,
    pub false_positives: u64,
    pub true_negatives: u64,
    pub false_negatives: u64,
}

impl BinaryDetectionCounts {
    pub fn from_predictions(truth: &[bool], predicted: &[bool]) -> Result<Self, String> {
        if truth.is_empty() || truth.len() != predicted.len() {
            return Err(
                "binary evaluation requires equal, non-empty truth and prediction lists".into(),
            );
        }

        let mut counts = Self::default();
        for (actual, estimate) in truth.iter().zip(predicted) {
            match (*actual, *estimate) {
                (true, true) => increment(&mut counts.true_positives)?,
                (false, true) => increment(&mut counts.false_positives)?,
                (false, false) => increment(&mut counts.true_negatives)?,
                (true, false) => increment(&mut counts.false_negatives)?,
            }
        }
        Ok(counts)
    }

    pub fn sample_count(&self) -> Option<u64> {
        self.true_positives
            .checked_add(self.false_positives)?
            .checked_add(self.true_negatives)?
            .checked_add(self.false_negatives)
    }

    pub fn precision(&self) -> Option<f64> {
        self.sample_count()?;
        ratio(
            self.true_positives,
            self.true_positives.checked_add(self.false_positives)?,
        )
    }

    pub fn recall(&self) -> Option<f64> {
        self.sample_count()?;
        ratio(
            self.true_positives,
            self.true_positives.checked_add(self.false_negatives)?,
        )
    }

    pub fn specificity(&self) -> Option<f64> {
        self.sample_count()?;
        ratio(
            self.true_negatives,
            self.true_negatives.checked_add(self.false_positives)?,
        )
    }

    pub fn false_positive_rate(&self) -> Option<f64> {
        self.sample_count()?;
        ratio(
            self.false_positives,
            self.false_positives.checked_add(self.true_negatives)?,
        )
    }

    pub fn f1(&self) -> Option<f64> {
        self.sample_count()?;
        let doubled_true_positives = self.true_positives.checked_mul(2)?;
        let denominator = doubled_true_positives
            .checked_add(self.false_positives)?
            .checked_add(self.false_negatives)?;
        ratio(doubled_true_positives, denominator)
    }

    pub fn balanced_accuracy(&self) -> Option<f64> {
        Some((self.recall()? + self.specificity()?) / 2.0)
    }
}

fn increment(value: &mut u64) -> Result<(), String> {
    *value = value
        .checked_add(1)
        .ok_or_else(|| "binary detection count overflow".to_string())?;
    Ok(())
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

/// Brier score (mean squared probability error) for binary outcomes.
/// Lower is better; the result is in [0, 1] for valid probability scores.
pub fn binary_brier_score(truth: &[bool], scores: &[f64]) -> Result<f64, String> {
    if truth.is_empty() || truth.len() != scores.len() {
        return Err("Brier score requires aligned, non-empty labels and scores".into());
    }
    if scores
        .iter()
        .any(|score| !score.is_finite() || !(0.0..=1.0).contains(score))
    {
        return Err("Brier scores must be finite probabilities within [0, 1]".into());
    }

    let squared_error_sum: f64 = truth
        .iter()
        .zip(scores)
        .map(|(actual, score)| {
            let target = if *actual { 1.0 } else { 0.0 };
            (*score - target).powi(2)
        })
        .sum();
    Ok(squared_error_sum / truth.len() as f64)
}

/// False-alarm events per hour of scored continuous audio. The caller must
/// match predicted events to annotations and merge adjacent threshold
/// crossings first; this function does not perform event matching.
pub fn false_alarm_events_per_hour(
    false_alarm_events: u64,
    observation_seconds: f64,
) -> Result<f64, String> {
    if !observation_seconds.is_finite() || observation_seconds <= 0.0 {
        return Err("false-alarm rate requires positive finite observation duration".into());
    }
    let rate = false_alarm_events as f64 / (observation_seconds / 3600.0);
    if !rate.is_finite() {
        return Err("false-alarm rate is not representable as a finite value".into());
    }
    Ok(rate)
}

/// Tie-aware stepwise average precision for binary labels and scores.
/// Scores must be finite values in [0, 1]; a split with no positive labels
/// is rejected because recall and average precision are undefined.
pub fn binary_average_precision(truth: &[bool], scores: &[f64]) -> Result<f64, String> {
    if truth.is_empty() || truth.len() != scores.len() {
        return Err("average precision requires equal, non-empty labels and scores".into());
    }
    if scores.iter().any(|score| !score.is_finite() || !(0.0..=1.0).contains(score)) {
        return Err("average precision scores must be finite and within [0, 1]".into());
    }

    let positive_count = truth.iter().filter(|&&label| label).count();
    if positive_count == 0 {
        return Err("average precision is undefined without positive labels".into());
    }

    let canonical_score = |score: f64| if score == 0.0 { 0.0 } else { score };
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|left, right| {
        canonical_score(scores[*right]).total_cmp(&canonical_score(scores[*left]))
    });

    let mut true_positives = 0usize;
    let mut false_positives = 0usize;
    let mut area = 0.0;
    let mut index = 0;

    while index < order.len() {
        let score = canonical_score(scores[order[index]]);
        let mut group_true_positives = 0usize;
        let mut group_false_positives = 0usize;
        while index < order.len() && canonical_score(scores[order[index]]) == score {
            if truth[order[index]] {
                group_true_positives += 1;
            } else {
                group_false_positives += 1;
            }
            index += 1;
        }
        true_positives += group_true_positives;
        false_positives += group_false_positives;
        let seen = true_positives + false_positives;
        let precision = true_positives as f64 / seen as f64;
        let recall_increment = group_true_positives as f64 / positive_count as f64;
        area += precision * recall_increment;
    }

    Ok(area)
}

/// A score threshold selected on validation data under a false-positive
/// rate constraint. The recorded counts are validation-only evidence and must
/// not be presented as held-out test results.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BinaryThresholdSelection {
    pub threshold: f64,
    pub validation_counts: BinaryDetectionCounts,
}

/// Select the threshold with highest validation recall subject to
/// `false_positive_rate <= maximum_false_positive_rate`.
///
/// Selection must be performed on a declared validation/calibration split;
/// freeze the returned threshold before evaluating on the test split.
/// Both positive and negative examples are required so the FPR constraint is
/// measurable. If no positive prediction satisfies the budget, a threshold
/// just above the highest score is returned and predicts no positives.
pub fn select_threshold_at_max_false_positive_rate(
    truth: &[bool],
    scores: &[f64],
    maximum_false_positive_rate: f64,
) -> Result<BinaryThresholdSelection, String> {
    if truth.is_empty() || truth.len() != scores.len() {
        return Err("threshold selection requires aligned, non-empty labels and scores".into());
    }
    if !maximum_false_positive_rate.is_finite()
        || !(0.0..=1.0).contains(&maximum_false_positive_rate)
    {
        return Err("maximum false-positive rate must be finite and within [0, 1]".into());
    }
    if truth.iter().all(|label| *label) || truth.iter().all(|label| !*label) {
        return Err("threshold selection requires both positive and negative labels".into());
    }
    if scores
        .iter()
        .any(|score| !score.is_finite() || !(0.0..=1.0).contains(score))
    {
        return Err("threshold selection scores must be finite and within [0, 1]".into());
    }

    let canonical_score = |score: f64| if score == 0.0 { 0.0 } else { score };
    let normalized_scores: Vec<_> = scores.iter().map(|score| canonical_score(*score)).collect();
    let maximum_score = normalized_scores
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .ok_or_else(|| "threshold selection received no scores".to_string())?;

    let mut thresholds = normalized_scores.clone();
    thresholds.sort_by(|left, right| right.total_cmp(left));
    thresholds.dedup_by(|left, right| *left == *right);
    // Scores are in [0, 1], so the next representable float is finite and
    // strictly above every supplied score, including the exact value 1.0.
    thresholds.push(f64::from_bits(maximum_score.to_bits() + 1));

    let mut best: Option<BinaryThresholdSelection> = None;
    for threshold in thresholds {
        let predicted: Vec<_> = normalized_scores
            .iter()
            .map(|score| *score >= threshold)
            .collect();
        let counts = BinaryDetectionCounts::from_predictions(truth, &predicted)?;
        let false_positive_rate = counts
            .false_positive_rate()
            .ok_or_else(|| "false-positive rate is undefined for validation data".to_string())?;
        if false_positive_rate > maximum_false_positive_rate {
            continue;
        }
        let recall = counts
            .recall()
            .ok_or_else(|| "recall is undefined for validation data".to_string())?;

        let replace = match best {
            None => true,
            Some(current) => {
                let current_recall = current
                    .validation_counts
                    .recall()
                    .ok_or_else(|| "stored validation recall is undefined".to_string())?;
                let current_fpr = current
                    .validation_counts
                    .false_positive_rate()
                    .ok_or_else(|| "stored validation FPR is undefined".to_string())?;
                recall > current_recall
                    || (recall == current_recall && false_positive_rate < current_fpr)
                    || (recall == current_recall
                        && false_positive_rate == current_fpr
                        && threshold > current.threshold)
            }
        };
        if replace {
            best = Some(BinaryThresholdSelection {
                threshold,
                validation_counts: counts,
            });
        }
    }

    best.ok_or_else(|| "no threshold satisfies the declared false-positive budget".into())
}

/// Apply a frozen threshold to a new split. A threshold greater than 1.0
/// is valid and intentionally predicts no positives when the validation
/// operating point used the no-detections fallback.
pub fn apply_binary_threshold(scores: &[f64], threshold: f64) -> Result<Vec<bool>, String> {
    if !threshold.is_finite() || threshold < 0.0 {
        return Err("binary detection threshold must be finite and non-negative".into());
    }
    if scores
        .iter()
        .any(|score| !score.is_finite() || !(0.0..=1.0).contains(score))
    {
        return Err("binary detection scores must be finite and within [0, 1]".into());
    }
    Ok(scores.iter().map(|score| *score >= threshold).collect())
}

/// Declared source lineage for one dataset split. IDs refer to provenance
/// records, not authenticated facts; this type prevents accidental leakage
/// when the caller supplies complete and truthful source metadata.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnimalSplitLineage {
    pub split_id: String,
    pub sample_ids: Vec<String>,
    pub source_recording_ids: Vec<String>,
    #[serde(default)]
    pub individual_ids: Vec<String>,
    #[serde(default)]
    pub site_ids: Vec<String>,
}

/// Claims the evaluation intends to test through disjoint holdouts.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnimalSplitPolicy {
    pub hold_out_individuals: bool,
    pub hold_out_sites: bool,
}

/// Rejects sample/recording leakage and enforces optional individual/site
/// holdouts. Source-recording separation is always required because clips
/// or crops from one recording must not straddle evaluation splits.
pub fn validate_animal_split_lineage(
    splits: &[AnimalSplitLineage],
    policy: &AnimalSplitPolicy,
) -> Result<(), String> {
    if splits.len() < 2 {
        return Err("animal evaluation requires at least two splits".into());
    }

    let mut split_ids = BTreeSet::new();
    for split in splits {
        if split.split_id.trim().is_empty() || !split_ids.insert(&split.split_id) {
            return Err("split identifiers must be non-empty and unique".into());
        }
        validate_unique_ids(&split.sample_ids, "sample", &split.split_id, true)?;
        validate_unique_ids(
            &split.source_recording_ids,
            "source recording",
            &split.split_id,
            true,
        )?;
        validate_unique_ids(
            &split.individual_ids,
            "individual",
            &split.split_id,
            policy.hold_out_individuals,
        )?;
        validate_unique_ids(
            &split.site_ids,
            "site",
            &split.split_id,
            policy.hold_out_sites,
        )?;
    }

    for (index, left) in splits.iter().enumerate() {
        for right in &splits[index + 1..] {
            reject_overlap(
                &left.sample_ids,
                &right.sample_ids,
                "sample",
                &left.split_id,
                &right.split_id,
            )?;
            reject_overlap(
                &left.source_recording_ids,
                &right.source_recording_ids,
                "source recording",
                &left.split_id,
                &right.split_id,
            )?;
            if policy.hold_out_individuals {
                reject_overlap(
                    &left.individual_ids,
                    &right.individual_ids,
                    "individual",
                    &left.split_id,
                    &right.split_id,
                )?;
            }
            if policy.hold_out_sites {
                reject_overlap(
                    &left.site_ids,
                    &right.site_ids,
                    "site",
                    &left.split_id,
                    &right.split_id,
                )?;
            }
        }
    }
    Ok(())
}

fn validate_unique_ids(
    ids: &[String],
    kind: &str,
    split_id: &str,
    required: bool,
) -> Result<(), String> {
    if required && ids.is_empty() {
        return Err(format!("{kind} lineage is required for split {split_id}"));
    }
    let mut unique = BTreeSet::new();
    for id in ids {
        if id.trim().is_empty() || !unique.insert(id.as_str()) {
            return Err(format!(
                "{kind} IDs must be non-empty and unique in split {split_id}"
            ));
        }
    }
    Ok(())
}

fn reject_overlap(
    left_ids: &[String],
    right_ids: &[String],
    kind: &str,
    left_split: &str,
    right_split: &str,
) -> Result<(), String> {
    let left: BTreeSet<_> = left_ids.iter().map(String::as_str).collect();
    if right_ids.iter().any(|id| left.contains(id.as_str())) {
        return Err(format!(
            "{kind} leakage between splits {left_split} and {right_split}"
        ));
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RepresentationBenchmark {
    pub provider: String,
    pub model_hash: String,
    pub benchmark: String,
    pub score: f64,
    pub pinned_baseline_score: f64,
    pub held_out_tasks: Vec<String>,
}

impl RepresentationBenchmark {
    pub fn passes(&self) -> bool {
        !self.provider.is_empty()
            && self.model_hash.len() >= 16
            && !self.benchmark.is_empty()
            && self.score.is_finite()
            && self.pinned_baseline_score.is_finite()
            && self.score >= self.pinned_baseline_score
            && !self.held_out_tasks.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AnimalControl {
    ShuffledContext,
    IdentityOnly,
    LocationOnly,
    AcousticOnly,
    HeldOutIndividuals,
    HeldOutSites,
    TemporalPermutation,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ContextPredictionEvaluation {
    pub task: String,
    pub primary_score: f64,
    pub control_scores: BTreeMap<AnimalControl, f64>,
    pub required_margin: f64,
}

impl ContextPredictionEvaluation {
    pub fn passes(&self) -> bool {
        // Require every preregistered confound control. A missing control is
        // not equivalent to a control that passed.
        let required_controls = [
            AnimalControl::ShuffledContext,
            AnimalControl::IdentityOnly,
            AnimalControl::LocationOnly,
            AnimalControl::AcousticOnly,
            AnimalControl::HeldOutIndividuals,
            AnimalControl::HeldOutSites,
        ];

        !self.task.trim().is_empty()
            && self.primary_score.is_finite()
            && self.required_margin.is_finite()
            && self.required_margin >= 0.0
            && required_controls
                .iter()
                .all(|control| self.control_scores.contains_key(control))
            && self.control_scores.values().all(|control| {
                control.is_finite() && self.primary_score >= control + self.required_margin
            })
    }
}

pub fn promotable_animal_capability(evidence: &[CommunicationEvidence]) -> CapabilityLevel {
    let intervention = crate::has_grounded_intervention(evidence);
    let independently_replicated = evidence
        .iter()
        .any(|item| item.replication == ReplicationStatus::IndependentlyReplicated);
    if intervention && independently_replicated {
        CapabilityLevel::Reference
    } else {
        CapabilityLevel::Structure
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlaybackProtocol {
    pub species: String,
    pub population: String,
    pub welfare_review: String,
    pub stop_conditions: Vec<String>,
    pub authorization: ExpressionAuthorization,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnimalEvaluationBundle {
    pub representation_baseline: String,
    pub context_prediction: ContextPredictionEvaluation,
    pub segmentation_stability: f64,
    /// Stable sample IDs. Legacy bundles deserialize but fail validation
    /// until these provenance fields are populated.
    #[serde(default)]
    pub train_sample_ids: Vec<String>,
    #[serde(default)]
    pub test_sample_ids: Vec<String>,
    /// Recording-level lineage prevents different crops from leaking across
    /// train and test. IDs are declarations, not source authentication.
    #[serde(default)]
    pub train_source_recording_ids: Vec<String>,
    #[serde(default)]
    pub test_source_recording_ids: Vec<String>,
    pub train_individuals: Vec<String>,
    pub test_individuals: Vec<String>,
    pub train_sites: Vec<String>,
    pub test_sites: Vec<String>,
    pub temporal_permutation_score: f64,
}

impl AnimalEvaluationBundle {
    pub fn validate(&self) -> Result<(), String> {
        let unique_nonempty = |values: &[String]| {
            let unique: std::collections::BTreeSet<_> = values.iter().collect();
            !values.is_empty()
                && values.iter().all(|value| !value.trim().is_empty())
                && unique.len() == values.len()
        };
        let train_individuals: std::collections::BTreeSet<_> =
            self.train_individuals.iter().collect();
        let test_individuals: std::collections::BTreeSet<_> =
            self.test_individuals.iter().collect();
        let train_sites: std::collections::BTreeSet<_> = self.train_sites.iter().collect();
        let test_sites: std::collections::BTreeSet<_> = self.test_sites.iter().collect();
        let lineage = [
            AnimalSplitLineage {
                split_id: "train".into(),
                sample_ids: self.train_sample_ids.clone(),
                source_recording_ids: self.train_source_recording_ids.clone(),
                individual_ids: self.train_individuals.clone(),
                site_ids: self.train_sites.clone(),
            },
            AnimalSplitLineage {
                split_id: "test".into(),
                sample_ids: self.test_sample_ids.clone(),
                source_recording_ids: self.test_source_recording_ids.clone(),
                individual_ids: self.test_individuals.clone(),
                site_ids: self.test_sites.clone(),
            },
        ];
        let lineage_policy = AnimalSplitPolicy {
            hold_out_individuals: true,
            hold_out_sites: true,
        };
        validate_animal_split_lineage(&lineage, &lineage_policy)?;
        if self.representation_baseline.trim().is_empty()
            || !unique_nonempty(&self.train_individuals)
            || !unique_nonempty(&self.test_individuals)
            || !unique_nonempty(&self.train_sites)
            || !unique_nonempty(&self.test_sites)
            || !train_individuals.is_disjoint(&test_individuals)
            || !train_sites.is_disjoint(&test_sites)
            || !(0.0..=1.0).contains(&self.segmentation_stability)
            || !self.temporal_permutation_score.is_finite()
            || !self.context_prediction.passes()
            || self.context_prediction.primary_score
                < self.temporal_permutation_score + self.context_prediction.required_margin
        {
            return Err(
                "animal evaluation lacks disjoint holdouts, controls, or valid scores".into(),
            );
        }
        Ok(())
    }
}

impl PlaybackProtocol {
    pub fn validate(&self) -> Result<(), String> {
        if self.species.is_empty()
            || self.population.is_empty()
            || self.welfare_review.is_empty()
            || self.stop_conditions.is_empty()
            || self.authorization.protocol_id.is_empty()
            || self.authorization.reviewer.is_empty()
        {
            return Err("playback requires species, population, welfare review, stop conditions, and reviewer".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DatasetSplit, EvidenceDomain, EvidenceKind, EvidenceRecord, ReplicationStatus};

    #[test]
    fn binary_detection_counts_perfect_predictions() {
        let counts = BinaryDetectionCounts::from_predictions(
            &[true, true, false, false],
            &[true, true, false, false],
        )
        .unwrap();
        assert_eq!(counts.sample_count(), Some(4));
        assert_eq!(counts.true_positives, 2);
        assert_eq!(counts.true_negatives, 2);
        assert_eq!(counts.precision(), Some(1.0));
        assert_eq!(counts.recall(), Some(1.0));
        assert_eq!(counts.specificity(), Some(1.0));
        assert_eq!(counts.false_positive_rate(), Some(0.0));
        assert_eq!(counts.f1(), Some(1.0));
        assert_eq!(counts.balanced_accuracy(), Some(1.0));
    }

    #[test]
    fn binary_detection_counts_expose_false_alarms_and_misses() {
        let counts = BinaryDetectionCounts::from_predictions(
            &[true, true, false, false],
            &[true, false, true, false],
        )
        .unwrap();
        assert_eq!(counts.true_positives, 1);
        assert_eq!(counts.false_positives, 1);
        assert_eq!(counts.true_negatives, 1);
        assert_eq!(counts.false_negatives, 1);
        assert_eq!(counts.precision(), Some(0.5));
        assert_eq!(counts.recall(), Some(0.5));
        assert_eq!(counts.false_positive_rate(), Some(0.5));
        assert_eq!(counts.f1(), Some(0.5));
        assert_eq!(counts.balanced_accuracy(), Some(0.5));
    }

    #[test]
    fn binary_detection_metrics_mark_missing_class_support_undefined() {
        let counts =
            BinaryDetectionCounts::from_predictions(&[false, false], &[false, false]).unwrap();
        assert_eq!(counts.specificity(), Some(1.0));
        assert_eq!(counts.recall(), None);
        assert_eq!(counts.precision(), None);
        assert_eq!(counts.f1(), None);
        assert_eq!(counts.balanced_accuracy(), None);
    }

    #[test]
    fn threshold_selection_maximizes_recall_under_zero_false_alarm_budget() {
        let selection = select_threshold_at_max_false_positive_rate(
            &[true, true, false, false],
            &[0.9, 0.4, 0.8, 0.1],
            0.0,
        )
        .unwrap();
        assert_eq!(selection.threshold, 0.9);
        assert_eq!(selection.validation_counts.true_positives, 1);
        assert_eq!(selection.validation_counts.false_positives, 0);
        assert_eq!(selection.validation_counts.recall(), Some(0.5));
        assert_eq!(selection.validation_counts.false_positive_rate(), Some(0.0));
    }

    #[test]
    fn threshold_selection_uses_allowed_false_alarm_budget() {
        let selection = select_threshold_at_max_false_positive_rate(
            &[true, true, false, false],
            &[0.9, 0.4, 0.8, 0.1],
            0.5,
        )
        .unwrap();
        assert_eq!(selection.threshold, 0.4);
        assert_eq!(selection.validation_counts.recall(), Some(1.0));
        assert_eq!(selection.validation_counts.false_positive_rate(), Some(0.5));
    }

    #[test]
    fn frozen_validation_threshold_applies_to_held_out_scores() {
        let validation = select_threshold_at_max_false_positive_rate(
            &[true, true, false, false],
            &[0.9, 0.4, 0.8, 0.1],
            0.0,
        )
        .unwrap();
        let test_predictions =
            apply_binary_threshold(&[0.95, 0.89, 0.2], validation.threshold).unwrap();
        assert_eq!(test_predictions, vec![true, false, false]);
        let test_counts =
            BinaryDetectionCounts::from_predictions(&[false, true, false], &test_predictions)
                .unwrap();
        assert_eq!(test_counts.false_positives, 1);
        assert_eq!(test_counts.false_negatives, 1);
    }

    #[test]
    fn no_detection_threshold_is_finite_even_at_maximum_score() {
        let selection = select_threshold_at_max_false_positive_rate(
            &[true, false],
            &[1.0, 1.0],
            0.0,
        )
        .unwrap();
        assert!(selection.threshold.is_finite());
        assert!(selection.threshold > 1.0);
        assert_eq!(
            apply_binary_threshold(&[1.0, 0.0], selection.threshold).unwrap(),
            vec![false, false]
        );
    }

    #[test]
    fn applying_threshold_rejects_invalid_scores_or_threshold() {
        assert!(apply_binary_threshold(&[f64::NAN], 0.5).is_err());
        assert!(apply_binary_threshold(&[1.5], 0.5).is_err());
        assert!(apply_binary_threshold(&[0.5], f64::INFINITY).is_err());
        assert!(apply_binary_threshold(&[0.5], -0.1).is_err());
    }

    #[test]
    fn threshold_selection_handles_ties_conservatively() {
        let selection = select_threshold_at_max_false_positive_rate(
            &[true, false],
            &[0.5, 0.5],
            0.0,
        )
        .unwrap();
        assert!(selection.threshold > 0.5);
        assert_eq!(selection.validation_counts.true_positives, 0);
        assert_eq!(selection.validation_counts.false_positives, 0);
    }

    #[test]
    fn threshold_selection_rejects_invalid_validation_inputs() {
        assert!(
            select_threshold_at_max_false_positive_rate(&[], &[], 0.1).is_err()
        );
        assert!(
            select_threshold_at_max_false_positive_rate(&[true], &[0.8], 0.1).is_err()
        );
        assert!(
            select_threshold_at_max_false_positive_rate(
                &[true, false],
                &[f64::NAN, 0.2],
                0.1,
            )
            .is_err()
        );
        assert!(
            select_threshold_at_max_false_positive_rate(
                &[true, false],
                &[0.8, 0.2],
                f64::NAN,
            )
            .is_err()
        );
    }

    #[test]
    fn binary_brier_score_rewards_calibrated_probabilities() {
        assert_eq!(binary_brier_score(&[true, false], &[1.0, 0.0]).unwrap(), 0.0);
        assert_eq!(binary_brier_score(&[true, false], &[0.5, 0.5]).unwrap(), 0.25);
        assert_eq!(binary_brier_score(&[true, false], &[0.0, 1.0]).unwrap(), 1.0);
    }

    #[test]
    fn binary_brier_score_rejects_invalid_inputs() {
        assert!(binary_brier_score(&[], &[]).is_err());
        assert!(binary_brier_score(&[true], &[]).is_err());
        assert!(binary_brier_score(&[true], &[f64::NAN]).is_err());
        assert!(binary_brier_score(&[false], &[1.1]).is_err());
    }

    #[test]
    fn false_alarm_events_per_hour_uses_recording_exposure() {
        assert_eq!(false_alarm_events_per_hour(2, 1800.0).unwrap(), 4.0);
        assert_eq!(false_alarm_events_per_hour(0, 3600.0).unwrap(), 0.0);
    }

    #[test]
    fn false_alarm_events_per_hour_rejects_invalid_duration() {
        assert!(false_alarm_events_per_hour(1, 0.0).is_err());
        assert!(false_alarm_events_per_hour(1, f64::NAN).is_err());
        assert!(false_alarm_events_per_hour(1, f64::INFINITY).is_err());
    }

    #[test]
    fn binary_average_precision_scores_a_ranked_presence_detector() {
        let score = binary_average_precision(
            &[true, false, true, false],
            &[0.9, 0.8, 0.4, 0.1],
        )
        .unwrap();
        assert!((score - 5.0 / 6.0).abs() < 1e-12);
    }

    #[test]
    fn binary_average_precision_groups_tied_scores_without_order_bias() {
        let score = binary_average_precision(&[true, false], &[0.5, 0.5]).unwrap();
        assert_eq!(score, 0.5);
        assert_eq!(binary_average_precision(&[true, false], &[0.0, -0.0]).unwrap(), 0.5);
    }

    #[test]
    fn binary_average_precision_rejects_invalid_or_positive_free_splits() {
        assert!(binary_average_precision(&[], &[]).is_err());
        assert!(binary_average_precision(&[true], &[]).is_err());
        assert!(binary_average_precision(&[false, false], &[0.8, 0.1]).is_err());
        assert!(binary_average_precision(&[true], &[f64::NAN]).is_err());
        assert!(binary_average_precision(&[true], &[1.1]).is_err());
    }

    #[test]
    fn binary_detection_counts_handle_overflowed_external_counts() {
        let counts = BinaryDetectionCounts {
            true_positives: u64::MAX,
            false_positives: 1,
            true_negatives: 0,
            false_negatives: 0,
        };
        assert_eq!(counts.sample_count(), None);
        assert_eq!(counts.precision(), None);
        assert_eq!(counts.recall(), None);
        assert_eq!(counts.specificity(), None);
        assert_eq!(counts.false_positive_rate(), None);
        assert_eq!(counts.f1(), None);
        assert_eq!(counts.balanced_accuracy(), None);
    }

    #[test]
    fn binary_detection_counts_reject_empty_or_mismatched_inputs() {
        assert!(BinaryDetectionCounts::from_predictions(&[], &[]).is_err());
        assert!(BinaryDetectionCounts::from_predictions(&[true], &[]).is_err());
    }

    fn split(
        split_id: &str,
        samples: &[&str],
        recordings: &[&str],
        individuals: &[&str],
        sites: &[&str],
    ) -> AnimalSplitLineage {
        AnimalSplitLineage {
            split_id: split_id.into(),
            sample_ids: samples.iter().map(|value| (*value).into()).collect(),
            source_recording_ids: recordings.iter().map(|value| (*value).into()).collect(),
            individual_ids: individuals.iter().map(|value| (*value).into()).collect(),
            site_ids: sites.iter().map(|value| (*value).into()).collect(),
        }
    }

    #[test]
    fn animal_lineage_rejects_two_crops_from_same_source_recording() {
        let splits = vec![
            split("train", &["crop-a"], &["recording-1"], &[], &[]),
            split("test", &["crop-b"], &["recording-1"], &[], &[]),
        ];
        assert!(validate_animal_split_lineage(&splits, &AnimalSplitPolicy::default()).is_err());
    }

    #[test]
    fn animal_lineage_rejects_missing_source_recording_provenance() {
        let splits = vec![
            split("train", &["clip-a"], &[], &[], &[]),
            split("test", &["clip-b"], &["recording-b"], &[], &[]),
        ];
        assert!(validate_animal_split_lineage(&splits, &AnimalSplitPolicy::default()).is_err());
    }

    #[test]
    fn animal_lineage_requires_disjoint_individuals_when_claimed() {
        let splits = vec![
            split("train", &["a"], &["rec-a"], &["bird-1"], &["site-a"]),
            split("test", &["b"], &["rec-b"], &["bird-1"], &["site-b"]),
        ];
        let policy = AnimalSplitPolicy {
            hold_out_individuals: true,
            hold_out_sites: false,
        };
        assert!(validate_animal_split_lineage(&splits, &policy).is_err());
    }

    #[test]
    fn animal_lineage_requires_disjoint_sites_when_claimed() {
        let splits = vec![
            split("train", &["a"], &["rec-a"], &["bird-1"], &["site-a"]),
            split("test", &["b"], &["rec-b"], &["bird-2"], &["site-a"]),
        ];
        let policy = AnimalSplitPolicy {
            hold_out_individuals: false,
            hold_out_sites: true,
        };
        assert!(validate_animal_split_lineage(&splits, &policy).is_err());
    }

    #[test]
    fn animal_lineage_allows_known_individual_overlap_when_not_claiming_holdout() {
        let splits = vec![
            split("train", &["a"], &["rec-a"], &["bird-1"], &["site-a"]),
            split("test", &["b"], &["rec-b"], &["bird-1"], &["site-b"]),
        ];
        assert!(validate_animal_split_lineage(&splits, &AnimalSplitPolicy::default()).is_ok());
    }

    #[test]
    fn animal_lineage_rejects_duplicate_and_blank_ids() {
        let duplicate = vec![
            split("train", &["a", "a"], &["rec-a"], &[], &[]),
            split("test", &["b"], &["rec-b"], &[], &[]),
        ];
        assert!(validate_animal_split_lineage(&duplicate, &AnimalSplitPolicy::default()).is_err());

        let blank_source = vec![
            split("train", &["a"], &[" "], &[], &[]),
            split("test", &["b"], &["rec-b"], &[], &[]),
        ];
        assert!(
            validate_animal_split_lineage(&blank_source, &AnimalSplitPolicy::default()).is_err()
        );
    }

    #[test]
    fn animal_lineage_rejects_duplicate_split_ids() {
        let splits = vec![
            split("same", &["a"], &["rec-a"], &[], &[]),
            split("same", &["b"], &["rec-b"], &[], &[]),
        ];
        assert!(validate_animal_split_lineage(&splits, &AnimalSplitPolicy::default()).is_err());
    }

    fn make_context_prediction(primary: f64, margin: f64) -> ContextPredictionEvaluation {
        ContextPredictionEvaluation {
            task: "context".into(),
            primary_score: primary,
            control_scores: BTreeMap::from([
                (AnimalControl::ShuffledContext, primary - margin - 0.01),
                (AnimalControl::IdentityOnly, primary - margin - 0.01),
                (AnimalControl::LocationOnly, primary - margin - 0.01),
                (AnimalControl::AcousticOnly, primary - margin - 0.01),
                (AnimalControl::HeldOutIndividuals, primary - margin - 0.01),
                (AnimalControl::HeldOutSites, primary - margin - 0.01),
            ]),
            required_margin: margin,
        }
    }

    #[test]
    fn empty_controls_never_pass() {
        assert!(
            !ContextPredictionEvaluation {
                task: "context".into(),
                primary_score: 1.0,
                control_scores: BTreeMap::new(),
                required_margin: 0.01
            }
            .passes()
        );
    }

    #[test]
    fn context_prediction_passes_when_all_controls_present() {
        assert!(make_context_prediction(0.80, 0.05).passes());
    }

    #[test]
    fn context_prediction_rejects_non_finite_margin() {
        let mut eval = make_context_prediction(0.80, 0.05);
        eval.required_margin = f64::NAN;
        assert!(!eval.passes());
    }

    #[test]
    fn context_prediction_rejects_empty_task_name() {
        let mut eval = make_context_prediction(0.80, 0.05);
        eval.task = "  ".into();
        assert!(!eval.passes());
    }

    #[test]
    fn missing_identity_or_location_control_fails_closed() {
        let mut eval = make_context_prediction(0.80, 0.05);
        eval.control_scores.remove(&AnimalControl::IdentityOnly);
        assert!(!eval.passes());

        let mut eval = make_context_prediction(0.80, 0.05);
        eval.control_scores.remove(&AnimalControl::LocationOnly);
        assert!(!eval.passes());
    }

    #[test]
    fn context_prediction_fails_when_margin_not_met() {
        // Primary 0.80, control at 0.79, required_margin 0.05 — gap is only 0.01.
        let eval = ContextPredictionEvaluation {
            task: "context".into(),
            primary_score: 0.80,
            control_scores: BTreeMap::from([
                (AnimalControl::ShuffledContext, 0.79),
                (AnimalControl::IdentityOnly, 0.79),
                (AnimalControl::LocationOnly, 0.79),
                (AnimalControl::AcousticOnly, 0.79),
                (AnimalControl::HeldOutIndividuals, 0.79),
                (AnimalControl::HeldOutSites, 0.79),
            ]),
            required_margin: 0.05,
        };
        assert!(!eval.passes());
    }

    #[test]
    fn missing_held_out_individuals_fails() {
        let eval = ContextPredictionEvaluation {
            task: "context".into(),
            primary_score: 0.80,
            // No HeldOutIndividuals or HeldOutSites
            control_scores: BTreeMap::from([
                (AnimalControl::ShuffledContext, 0.70),
                (AnimalControl::AcousticOnly, 0.70),
            ]),
            required_margin: 0.05,
        };
        assert!(!eval.passes());
    }

    fn make_bundle(overlap_identities: bool) -> AnimalEvaluationBundle {
        let (test_individuals, test_sites) = if overlap_identities {
            (
                vec!["individual-A".into(), "individual-B".into()],
                vec!["site-3".into()],
            )
        } else {
            (
                vec!["individual-C".into(), "individual-D".into()],
                vec!["site-3".into()],
            )
        };
        AnimalEvaluationBundle {
            representation_baseline: "cav".into(),
            context_prediction: make_context_prediction(0.80, 0.05),
            segmentation_stability: 0.85,
            train_sample_ids: vec!["train-sample-1".into(), "train-sample-2".into()],
            test_sample_ids: vec!["test-sample-1".into(), "test-sample-2".into()],
            train_source_recording_ids: vec![
                "train-recording-1".into(),
                "train-recording-2".into(),
            ],
            test_source_recording_ids: vec!["test-recording-1".into()],
            train_individuals: vec!["individual-A".into(), "individual-B".into()],
            test_individuals,
            train_sites: vec!["site-1".into(), "site-2".into()],
            test_sites,
            temporal_permutation_score: 0.60, // well below 0.80 - 0.05 margin
        }
    }

    #[test]
    fn animal_bundle_passes_when_all_conditions_met() {
        assert!(make_bundle(false).validate().is_ok());
    }

    #[test]
    fn animal_bundle_rejects_shared_source_recording_across_splits() {
        let mut bundle = make_bundle(false);
        bundle.test_source_recording_ids = vec!["train-recording-1".into()];
        assert!(bundle.validate().is_err());
    }

    #[test]
    fn animal_bundle_rejects_missing_sample_or_recording_lineage() {
        let mut no_samples = make_bundle(false);
        no_samples.test_sample_ids.clear();
        assert!(no_samples.validate().is_err());

        let mut no_recordings = make_bundle(false);
        no_recordings.train_source_recording_ids.clear();
        assert!(no_recordings.validate().is_err());
    }

    #[test]
    fn animal_bundle_fails_when_train_test_individuals_overlap() {
        assert!(make_bundle(true).validate().is_err());
    }

    #[test]
    fn animal_bundle_rejects_non_finite_temporal_permutation() {
        let mut bundle = make_bundle(false);
        bundle.temporal_permutation_score = f64::NAN;
        assert!(bundle.validate().is_err());
    }

    #[test]
    fn animal_bundle_rejects_duplicate_or_empty_split_ids() {
        let mut duplicate = make_bundle(false);
        duplicate.train_sites.push("site-1".into());
        assert!(duplicate.validate().is_err());

        let mut empty = make_bundle(false);
        empty.test_individuals.push(" ".into());
        assert!(empty.validate().is_err());
    }

    #[test]
    fn promotable_capability_requires_intervention_and_replication() {
        fn make_evidence(kind: EvidenceKind, replicated: bool) -> crate::CommunicationEvidence {
            let mut e = crate::CommunicationEvidence {
                id: String::new(),
                dataset_uri: "local:test".into(),
                dataset_hash: "d".into(),
                model_hash: "m".into(),
                lineage: vec![],
                split: DatasetSplit::default(),
                evidence_records: vec![EvidenceRecord {
                    kind,
                    description: "test".into(),
                    result_hash: "r".into(),
                    preregistered: true,
                    independently_replicated: replicated,
                }],
                preregistration_uri: None,
                replication: if replicated {
                    ReplicationStatus::IndependentlyReplicated
                } else {
                    ReplicationStatus::Unreplicated
                },
                calibration: vec![],
                experimental: false,
                domain: EvidenceDomain::AnimalCommunication,
            };
            e.id = e.computed_id().unwrap();
            e
        }
        // Only preregistered intervention + independent replication reaches Reference.
        let evidence = vec![make_evidence(EvidenceKind::Intervention, true)];
        assert_eq!(
            promotable_animal_capability(&evidence),
            CapabilityLevel::Reference
        );

        // Observational correlation alone stays at Structure.
        let evidence = vec![make_evidence(EvidenceKind::ObservationalCorrelation, false)];
        assert_eq!(
            promotable_animal_capability(&evidence),
            CapabilityLevel::Structure
        );

        // Intervention without independent replication stays at Structure.
        let evidence = vec![make_evidence(EvidenceKind::Intervention, false)];
        assert_eq!(
            promotable_animal_capability(&evidence),
            CapabilityLevel::Structure
        );
    }
}
