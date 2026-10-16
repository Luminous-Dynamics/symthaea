// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic event-level scoring for precomputed bioacoustic predictions.
//!
//! This evaluator uses the corpus manifest as its annotation source. It scores
//! the Test partition only and requires an explicit assertion that annotations
//! exhaustively cover the target labels on every Test recording. That assertion
//! is metadata, not an independently authenticated fact. This module does not
//! run a model or verify the source audio bytes; use bioacoustic-asset-audit first.

use crate::bioacoustic_corpus::{
    BioacousticCorpusManifest, BioacousticSplit, BioacousticSplitPolicy,
    validate_bioacoustic_corpus,
};
use crate::TimeSpan;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const BIOACOUSTIC_EVENT_PREDICTION_SCHEMA_VERSION: u32 = 1;
pub const BIOACOUSTIC_EVENT_REPORT_SCHEMA_VERSION: u32 = 2;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EventMatchRule {
    /// One-to-one match when the absolute start-time error is within tolerance.
    /// Suitable for point-like events such as click onset.
    StartTolerance { max_start_error_s: f64 },
    /// One-to-one match when temporal intersection-over-union meets the minimum.
    /// Suitable for annotated segments such as whistles or songs.
    TemporalIou { min_iou: f64 },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PredictedBioacousticEvent {
    pub prediction_id: String,
    pub sample_id: String,
    pub interval: TimeSpan,
    pub label: String,
    /// Probability-like confidence score in [0, 1] from the frozen model.
    pub score: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BioacousticEventPredictionBundle {
    pub schema_version: u32,
    pub dataset_id: String,
    pub dataset_revision: String,
    /// Must equal the canonical BLAKE3 identity computed from the corpus manifest.
    pub corpus_manifest_blake3: String,
    pub task_id: String,
    pub model_id: String,
    /// SHA-256 of the exact model artifact, not merely a model name/version.
    pub model_sha256: String,
    pub target_labels: Vec<String>,
    pub matching_rule: EventMatchRule,
    /// Threshold selected on Validation, frozen before the Test set was scored.
    pub frozen_score_threshold: f64,
    /// Must enumerate every Test sample, including samples with no target events.
    /// This records an annotator/protocol assertion; code cannot prove annotation completeness.
    pub exhaustive_annotation_sample_ids: Vec<String>,
    /// A declared split policy. Source-recording lineage is mandatory for evaluation.
    pub split_policy: BioacousticSplitPolicy,
    pub predictions: Vec<PredictedBioacousticEvent>,
}

impl BioacousticEventPredictionBundle {
    /// Stable BLAKE3 identity for the prediction bundle, independent of list
    /// order where order has no scoring semantics.
    pub fn computed_bundle_blake3(&self) -> Result<String, String> {
        let mut canonical = self.clone();
        canonical.target_labels.sort();
        canonical.exhaustive_annotation_sample_ids.sort();
        canonical.corpus_manifest_blake3 =
            canonical.corpus_manifest_blake3.to_ascii_lowercase();
        canonical.model_sha256 = canonical.model_sha256.to_ascii_lowercase();
        canonical.predictions.sort_by(|left, right| left.prediction_id.cmp(&right.prediction_id));
        let bytes = serde_json::to_vec(&canonical)
            .map_err(|error| format!("cannot serialize prediction bundle: {error}"))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BioacousticEventMetrics {
    pub true_positives: u64,
    pub false_positives: u64,
    pub false_negatives: u64,
    pub ground_truth_events: u64,
    pub predictions_after_threshold: u64,
    pub scored_duration_seconds: f64,
    pub unscored_duration_seconds: f64,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1: Option<f64>,
    /// Micro event average precision across all scored Test recordings.
    pub average_precision: Option<f64>,
    /// False-positive event rate per hour of non-overlapping audio included in this score.
    pub false_alarms_per_scored_hour: Option<f64>,
}

impl BioacousticEventMetrics {
    fn from_counts(
        true_positives: usize,
        false_positives: usize,
        false_negatives: usize,
        ground_truth_events: usize,
        predictions_after_threshold: usize,
        scored_duration_seconds: f64,
        unscored_duration_seconds: f64,
    ) -> Self {
        let precision = ratio(true_positives, true_positives + false_positives);
        let recall = ratio(true_positives, true_positives + false_negatives);
        let f1_denominator = 2 * true_positives + false_positives + false_negatives;
        let f1 = ratio(2 * true_positives, f1_denominator);
        let false_alarms_per_hour = (scored_duration_seconds.is_finite()
            && scored_duration_seconds > 0.0)
            .then(|| false_positives as f64 * 3600.0 / scored_duration_seconds);
        Self {
            true_positives: true_positives as u64,
            false_positives: false_positives as u64,
            false_negatives: false_negatives as u64,
            ground_truth_events: ground_truth_events as u64,
            predictions_after_threshold: predictions_after_threshold as u64,
            scored_duration_seconds,
            unscored_duration_seconds,
            precision,
            recall,
            f1,
            average_precision: None,
            false_alarms_per_scored_hour: false_alarms_per_hour,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BioacousticPerSampleMetrics {
    pub sample_id: String,
    pub predictions_excluded_by_unscored_regions: u64,
    pub metrics: BioacousticEventMetrics,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BioacousticEventEvaluationReport {
    pub status: String,
    pub schema_version: u32,
    pub corpus_schema_version: u32,
    pub dataset_id: String,
    pub dataset_revision: String,
    pub corpus_manifest_blake3: String,
    pub prediction_bundle_blake3: String,
    pub scored_split: BioacousticSplit,
    pub task_id: String,
    pub model_id: String,
    pub model_sha256: String,
    pub target_labels: Vec<String>,
    pub matching_rule: EventMatchRule,
    pub frozen_score_threshold: f64,
    pub exhaustive_test_annotation_coverage_asserted: bool,
    pub split_policy: BioacousticSplitPolicy,
    pub test_recording_count: usize,
    /// Mean of per-recording thresholded F1, excluding recordings with no
    /// truth events and no scored predictions (F1 is undefined for that case).
    pub macro_f1_over_recordings: Option<f64>,
    pub macro_f1_recording_count: usize,
    pub predictions_excluded_by_unscored_regions: u64,
    pub metrics: BioacousticEventMetrics,
    pub per_sample: Vec<BioacousticPerSampleMetrics>,
    pub audio_bytes_verified_by_this_evaluator: bool,
    pub annotation_truth_authenticated_by_this_evaluator: bool,
    pub model_executed_by_this_evaluator: bool,
    pub prediction_provenance_authenticated_by_this_evaluator: bool,
    pub validation_threshold_selection_authenticated_by_this_evaluator: bool,
}

#[derive(Clone, Debug)]
struct Event {
    id: String,
    sample_id: String,
    interval: TimeSpan,
    label: String,
    score: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    tp: usize,
    fp: usize,
    fn_: usize,
    ground_truth: usize,
    predictions: usize,
}

pub fn evaluate_bioacoustic_events(
    manifest: &BioacousticCorpusManifest,
    bundle: &BioacousticEventPredictionBundle,
) -> Result<BioacousticEventEvaluationReport, String> {
    if bundle.schema_version != BIOACOUSTIC_EVENT_PREDICTION_SCHEMA_VERSION {
        return Err("unsupported event-evaluation bundle schema version".into());
    }
    if bundle.dataset_id != manifest.dataset_id
        || bundle.dataset_revision != manifest.dataset_revision
    {
        return Err("prediction bundle dataset ID/revision does not match corpus manifest".into());
    }
    let computed_manifest_hash = manifest.computed_manifest_blake3()?;
    let computed_bundle_hash = bundle.computed_bundle_blake3()?;
    if !bundle
        .corpus_manifest_blake3
        .eq_ignore_ascii_case(&computed_manifest_hash)
    {
        return Err("prediction bundle corpus-manifest BLAKE3 does not match manifest".into());
    }
    if bundle.task_id.trim().is_empty() || bundle.model_id.trim().is_empty() {
        return Err("task ID and model ID must be non-empty".into());
    }
    if !is_sha256_hex(&bundle.model_sha256) {
        return Err("model SHA-256 must contain exactly 64 hexadecimal characters".into());
    }
    if !bundle.frozen_score_threshold.is_finite()
        || !(0.0..=1.0).contains(&bundle.frozen_score_threshold)
    {
        return Err("frozen score threshold must be finite and within [0, 1]".into());
    }
    if bundle.target_labels.is_empty() {
        return Err("at least one target label is required".into());
    }
    let mut target_labels = BTreeSet::new();
    if bundle.target_labels.iter().any(|label| {
        label.trim().is_empty()
            || label.trim() != label
            || !target_labels.insert(label.trim().to_owned())
    }) {
        return Err("target labels must be non-empty and unique".into());
    }
    validate_match_rule(&bundle.matching_rule)?;

    if !bundle.split_policy.require_source_recording_ids {
        return Err(
            "evaluation requires source-recording lineage; enable require_source_recording_ids"
                .into(),
        );
    }
    validate_bioacoustic_corpus(manifest, bundle.split_policy)?;

    let mut test_recordings: Vec<_> = manifest
        .recordings
        .iter()
        .filter(|recording| recording.split == BioacousticSplit::Test)
        .collect();
    test_recordings.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));
    if test_recordings.is_empty() {
        return Err("corpus manifest has no Test recordings".into());
    }
    validate_non_overlapping_source_coverage(&test_recordings)?;
    let test_sample_ids: BTreeSet<_> = test_recordings
        .iter()
        .map(|recording| recording.sample_id.as_str())
        .collect();
    let test_recording_by_id: std::collections::BTreeMap<_, _> = test_recordings
        .iter()
        .map(|recording| (recording.sample_id.as_str(), *recording))
        .collect();

    let mut exhaustive_ids = BTreeSet::new();
    if bundle.exhaustive_annotation_sample_ids.iter().any(|id| {
        id.trim().is_empty() || !exhaustive_ids.insert(id.as_str())
    }) {
        return Err("exhaustive annotation sample IDs must be non-empty and unique".into());
    }
    if exhaustive_ids != test_sample_ids {
        return Err(
            "exhaustive_annotation_sample_ids must exactly cover every Test sample, including empty-event recordings"
                .into(),
        );
    }

    let mut prediction_ids = BTreeSet::new();
    let mut predictions_by_sample: std::collections::BTreeMap<&str, Vec<Event>> =
        std::collections::BTreeMap::new();
    for prediction in &bundle.predictions {
        if prediction.prediction_id.trim().is_empty()
            || !prediction_ids.insert(prediction.prediction_id.as_str())
        {
            return Err("prediction IDs must be non-empty and unique".into());
        }
        if !test_sample_ids.contains(prediction.sample_id.as_str()) {
            return Err(format!(
                "prediction {} references non-Test or unknown sample {}",
                prediction.prediction_id, prediction.sample_id
            ));
        }
        if prediction.label.trim().is_empty()
            || prediction.label.trim() != prediction.label
            || !target_labels.contains(prediction.label.trim())
        {
            return Err(format!(
                "prediction {} label is not one of the declared target labels",
                prediction.prediction_id
            ));
        }
        if !valid_interval(prediction.interval) {
            return Err(format!("prediction {} has an invalid interval", prediction.prediction_id));
        }
        if !prediction.score.is_finite() || !(0.0..=1.0).contains(&prediction.score) {
            return Err(format!(
                "prediction {} score must be finite and in [0, 1]",
                prediction.prediction_id
            ));
        }
        let recording = test_recording_by_id
            .get(prediction.sample_id.as_str())
            .expect("membership in test_sample_ids implies a test recording");
        let duration_s = recording.sample_count as f64 / recording.sample_rate_hz as f64;
        if prediction.interval.start_s < 0.0 || prediction.interval.end_s > duration_s {
            return Err(format!(
                "prediction {} interval falls outside sample {} duration",
                prediction.prediction_id, prediction.sample_id
            ));
        }
        predictions_by_sample
            .entry(prediction.sample_id.as_str())
            .or_default()
            .push(Event {
                id: prediction.prediction_id.clone(),
                sample_id: prediction.sample_id.clone(),
                interval: prediction.interval,
                label: prediction.label.clone(),
                score: prediction.score,
            });
    }

    let mut per_sample = Vec::with_capacity(test_recordings.len());
    let mut total = Counts::default();
    let mut total_duration = 0.0_f64;
    let mut total_unscored_duration = 0.0_f64;
    let mut all_test_truth = Vec::new();
    let mut all_test_predictions = Vec::new();
    let mut total_predictions_excluded_by_unscored_regions = 0usize;

    for recording in &test_recordings {
        let duration_s = recording.sample_count as f64 / recording.sample_rate_hz as f64;
        if !duration_s.is_finite() || duration_s <= 0.0 {
            return Err(format!("sample {} has invalid duration", recording.sample_id));
        }
        let unscored_duration_s: f64 = recording
            .unscored_regions
            .iter()
            .map(|region| region.interval.end_s - region.interval.start_s)
            .sum();
        let scored_duration_s = duration_s - unscored_duration_s;
        if !scored_duration_s.is_finite() || scored_duration_s < 0.0 {
            return Err(format!(
                "sample {} has invalid unscored-region duration",
                recording.sample_id
            ));
        }
        let truth: Vec<_> = recording
            .annotations
            .iter()
            .filter(|annotation| target_labels.contains(annotation.label.trim()))
            .map(|annotation| Event {
                id: annotation.annotation_id.clone(),
                sample_id: recording.sample_id.clone(),
                interval: annotation.interval,
                label: annotation.label.trim().to_owned(),
                score: 1.0,
            })
            .collect();
        let mut all_predictions = predictions_by_sample
            .remove(recording.sample_id.as_str())
            .unwrap_or_default();
        let predictions_before_unscored_filter = all_predictions.len();
        all_predictions.retain(|prediction| {
            !prediction_in_unscored_region(
                prediction,
                &recording.unscored_regions,
                &bundle.matching_rule,
            )
        });
        let excluded_by_unscored_regions =
            predictions_before_unscored_filter - all_predictions.len();
        total_predictions_excluded_by_unscored_regions += excluded_by_unscored_regions;
        let mut predicted = Vec::new();
        for prediction in &all_predictions {
            if prediction.score >= bundle.frozen_score_threshold {
                predicted.push(prediction.clone());
            }
        }
        let (tp, fp, fn_) = match_events(&truth, &predicted, &bundle.matching_rule);
        let counts = Counts {
            tp,
            fp,
            fn_,
            ground_truth: truth.len(),
            predictions: predicted.len(),
        };
        total.tp += counts.tp;
        total.fp += counts.fp;
        total.fn_ += counts.fn_;
        total.ground_truth += counts.ground_truth;
        total.predictions += counts.predictions;
        total_duration += scored_duration_s;
        total_unscored_duration += unscored_duration_s;
        let mut sample_metrics = BioacousticEventMetrics::from_counts(
            counts.tp,
            counts.fp,
            counts.fn_,
            counts.ground_truth,
            counts.predictions,
            scored_duration_s,
            unscored_duration_s,
        );
        sample_metrics.average_precision =
            event_average_precision(&truth, &all_predictions, &bundle.matching_rule);
        all_test_truth.extend(truth);
        all_test_predictions.extend(all_predictions);
        per_sample.push(BioacousticPerSampleMetrics {
            sample_id: recording.sample_id.clone(),
            predictions_excluded_by_unscored_regions: excluded_by_unscored_regions as u64,
            metrics: sample_metrics,
        });
    }

    if !total_duration.is_finite() || total_duration <= 0.0 {
        return Err("event evaluation has no scorable audio after excluding unscored regions".into());
    }
    let mut metrics = BioacousticEventMetrics::from_counts(
        total.tp,
        total.fp,
        total.fn_,
        total.ground_truth,
        total.predictions,
        total_duration,
        total_unscored_duration,
    );
    metrics.average_precision =
        event_average_precision(&all_test_truth, &all_test_predictions, &bundle.matching_rule);

    let macro_f1_values: Vec<f64> = per_sample
        .iter()
        .filter(|sample| {
            sample.metrics.ground_truth_events > 0
                || sample.metrics.predictions_after_threshold > 0
        })
        .filter_map(|sample| sample.metrics.f1)
        .collect();
    let macro_f1_over_recordings = if macro_f1_values.is_empty() {
        None
    } else {
        Some(
            macro_f1_values.iter().sum::<f64>()
                / macro_f1_values.len() as f64,
        )
    };

    Ok(BioacousticEventEvaluationReport {
        status: "EVALUATED".into(),
        schema_version: BIOACOUSTIC_EVENT_REPORT_SCHEMA_VERSION,
        corpus_schema_version: manifest.schema_version,
        dataset_id: manifest.dataset_id.clone(),
        dataset_revision: manifest.dataset_revision.clone(),
        corpus_manifest_blake3: computed_manifest_hash,
        prediction_bundle_blake3: computed_bundle_hash,
        scored_split: BioacousticSplit::Test,
        task_id: bundle.task_id.clone(),
        model_id: bundle.model_id.clone(),
        model_sha256: bundle.model_sha256.to_ascii_lowercase(),
        target_labels: bundle.target_labels.clone(),
        matching_rule: bundle.matching_rule.clone(),
        frozen_score_threshold: bundle.frozen_score_threshold,
        exhaustive_test_annotation_coverage_asserted: true,
        split_policy: bundle.split_policy,
        test_recording_count: test_recordings.len(),
        macro_f1_over_recordings,
        macro_f1_recording_count: macro_f1_values.len(),
        predictions_excluded_by_unscored_regions: total_predictions_excluded_by_unscored_regions as u64,
        metrics,
        per_sample,
        audio_bytes_verified_by_this_evaluator: false,
        annotation_truth_authenticated_by_this_evaluator: false,
        model_executed_by_this_evaluator: false,
        prediction_provenance_authenticated_by_this_evaluator: false,
        validation_threshold_selection_authenticated_by_this_evaluator: false,
    })
}

fn prediction_in_unscored_region(
    prediction: &Event,
    regions: &[crate::bioacoustic_corpus::BioacousticUnscoredRegion],
    rule: &EventMatchRule,
) -> bool {
    match rule {
        EventMatchRule::StartTolerance { .. } => {
            // Region metadata is validated as sorted and non-overlapping, so
            // only the last region beginning at/before the point can contain it.
            let after_start = regions.partition_point(|region| {
                region.interval.start_s <= prediction.interval.start_s
            });
            after_start > 0
                && prediction.interval.start_s < regions[after_start - 1].interval.end_s
        }
        EventMatchRule::TemporalIou { min_iou } => {
            // Here min_iou is reused as the predeclared minimum fraction of the
            // predicted duration covered by unscored regions (not a region IoU).
            let duration =
                prediction.interval.end_s - prediction.interval.start_s;
            if duration <= 0.0 {
                return false;
            }
            let first = regions.partition_point(|region| {
                region.interval.end_s <= prediction.interval.start_s
            });
            let last = regions.partition_point(|region| {
                region.interval.start_s < prediction.interval.end_s
            });
            let overlap_seconds: f64 = regions[first..last]
                .iter()
                .map(|region| {
                    (prediction.interval.end_s.min(region.interval.end_s)
                        - prediction.interval.start_s.max(region.interval.start_s))
                        .max(0.0)
                })
                .sum();
            overlap_seconds / duration >= *min_iou
        }
    }
}

type SourceAssetCoverage<'a> = (
    &'a crate::bioacoustic_corpus::BioacousticRecording,
    f64,
    f64,
);

fn validate_non_overlapping_source_coverage(
    test_recordings: &[&crate::bioacoustic_corpus::BioacousticRecording],
) -> Result<(), String> {
    let mut by_source: BTreeMap<&str, Vec<SourceAssetCoverage<'_>>> = BTreeMap::new();

    for recording in test_recordings {
        let source_id = recording
            .source_recording_id
            .as_deref()
            .ok_or_else(|| format!("sample {} has no source-recording ID", recording.sample_id))?;
        let duration = recording.sample_count as f64 / recording.sample_rate_hz as f64;
        let start = recording.source_start_s.unwrap_or(0.0);
        let end = start + duration;
        by_source
            .entry(source_id)
            .or_default()
            .push((*recording, start, end));
    }

    for (source_id, assets) in &mut by_source {
        if assets.len() > 1 && assets.iter().any(|(recording, _, _)| recording.source_start_s.is_none()) {
            return Err(format!(
                "multiple Test assets share source {source_id}; every crop needs source_start_s to audit overlap"
            ));
        }
        assets.sort_by(|left, right| {
            left.1.total_cmp(&right.1)
                .then_with(|| left.0.sample_id.cmp(&right.0.sample_id))
        });
        for pair in assets.windows(2) {
            let (previous, _, previous_end) = pair[0];
            let (current, current_start, _) = pair[1];
            if current_start < previous_end {
                return Err(format!(
                    "overlapping Test assets within source {source_id}: {} and {}",
                    previous.sample_id, current.sample_id
                ));
            }
        }
    }
    Ok(())
}

fn validate_match_rule(rule: &EventMatchRule) -> Result<(), String> {
    match rule {
        EventMatchRule::StartTolerance { max_start_error_s } => {
            if !max_start_error_s.is_finite() || *max_start_error_s < 0.0 {
                return Err("start-time tolerance must be finite and non-negative".into());
            }
        }
        EventMatchRule::TemporalIou { min_iou } => {
            if !min_iou.is_finite() || !(0.0..=1.0).contains(min_iou) || *min_iou == 0.0 {
                return Err("minimum temporal IoU must be finite and within (0, 1]".into());
            }
        }
    }
    Ok(())
}

fn match_events(
    truth: &[Event],
    predictions: &[Event],
    rule: &EventMatchRule,
) -> (usize, usize, usize) {
    let adjacency = build_match_adjacency(truth, predictions, rule);
    let mut order: Vec<usize> = (0..predictions.len()).collect();
    order.sort_by(|left, right| {
        predictions[*right]
            .score
            .total_cmp(&predictions[*left].score)
            .then_with(|| predictions[*left].id.cmp(&predictions[*right].id))
    });
    let mut truth_to_prediction = vec![None; truth.len()];
    let mut true_positives = 0usize;
    for prediction_index in order {
        let mut seen = vec![false; truth.len()];
        if augment_match(prediction_index, &adjacency, &mut seen, &mut truth_to_prediction) {
            true_positives += 1;
        }
    }
    (
        true_positives,
        predictions.len() - true_positives,
        truth.len() - true_positives,
    )
}

fn event_average_precision(
    truth: &[Event],
    predictions: &[Event],
    rule: &EventMatchRule,
) -> Option<f64> {
    if truth.is_empty() {
        return None;
    }
    let adjacency = build_match_adjacency(truth, predictions, rule);
    let mut order: Vec<usize> = (0..predictions.len()).collect();
    order.sort_by(|left, right| {
        predictions[*right]
            .score
            .total_cmp(&predictions[*left].score)
            .then_with(|| predictions[*left].id.cmp(&predictions[*right].id))
    });

    let mut truth_to_prediction = vec![None; truth.len()];
    let mut matched_total = 0usize;
    let mut seen_predictions = 0usize;
    let mut average_precision = 0.0_f64;
    let mut cursor = 0usize;
    while cursor < order.len() {
        let score = predictions[order[cursor]].score;
        let mut group_true_positives = 0usize;
        let mut group_predictions = 0usize;
        while cursor < order.len() && predictions[order[cursor]].score == score {
            let prediction_index = order[cursor];
            let mut seen_truth = vec![false; truth.len()];
            if augment_match(
                prediction_index,
                &adjacency,
                &mut seen_truth,
                &mut truth_to_prediction,
            ) {
                matched_total += 1;
                group_true_positives += 1;
            }
            group_predictions += 1;
            cursor += 1;
        }
        seen_predictions += group_predictions;
        if group_true_positives > 0 {
            let precision = matched_total as f64 / seen_predictions as f64;
            let recall_increment = group_true_positives as f64 / truth.len() as f64;
            average_precision += precision * recall_increment;
        }
    }
    Some(average_precision)
}

fn build_match_adjacency(
    truth: &[Event],
    predictions: &[Event],
    rule: &EventMatchRule,
) -> Vec<Vec<usize>> {
    let mut truth_by_sample_label: std::collections::BTreeMap<(&str, &str), Vec<usize>> =
        std::collections::BTreeMap::new();
    for (truth_index, actual) in truth.iter().enumerate() {
        truth_by_sample_label
            .entry((actual.sample_id.as_str(), actual.label.as_str()))
            .or_default()
            .push(truth_index);
    }
    for indices in truth_by_sample_label.values_mut() {
        indices.sort_by(|left, right| {
            truth[*left]
                .interval
                .start_s
                .total_cmp(&truth[*right].interval.start_s)
                .then_with(|| truth[*left].id.cmp(&truth[*right].id))
        });
    }
    // Prefix maximum ends make interval-overlap queries logarithmic before
    // candidate checks, avoiding a scan over every earlier event for every
    // prediction in a dense/long recording.
    let prefix_max_end_by_key: std::collections::BTreeMap<(&str, &str), Vec<f64>> =
        truth_by_sample_label
            .iter()
            .map(|(key, indices)| {
                let mut maximum_end = f64::NEG_INFINITY;
                let prefix = indices
                    .iter()
                    .map(|truth_index| {
                        maximum_end = maximum_end.max(truth[*truth_index].interval.end_s);
                        maximum_end
                    })
                    .collect();
                (*key, prefix)
            })
            .collect();

    let mut adjacency = vec![Vec::new(); predictions.len()];
    for (prediction_index, prediction) in predictions.iter().enumerate() {
        let key = (prediction.sample_id.as_str(), prediction.label.as_str());
        let Some(candidates) = truth_by_sample_label.get(&key) else {
            continue;
        };
        let (first, last) = match rule {
            EventMatchRule::StartTolerance { max_start_error_s } => {
                let lower = prediction.interval.start_s - max_start_error_s;
                let upper = prediction.interval.start_s + max_start_error_s;
                (
                    candidates.partition_point(|truth_index| {
                        truth[*truth_index].interval.start_s < lower
                    }),
                    candidates.partition_point(|truth_index| {
                        truth[*truth_index].interval.start_s <= upper
                    }),
                )
            }
            EventMatchRule::TemporalIou { .. } => {
                let prefix_max_ends = prefix_max_end_by_key
                    .get(&key)
                    .expect("every truth group has a prefix-max-end index");
                (
                    prefix_max_ends.partition_point(|max_end| {
                        *max_end <= prediction.interval.start_s
                    }),
                    candidates.partition_point(|truth_index| {
                        truth[*truth_index].interval.start_s < prediction.interval.end_s
                    }),
                )
            }
        };
        for &truth_index in &candidates[first..last] {
            let actual = &truth[truth_index];
            if events_match(prediction.interval, actual.interval, rule) {
                adjacency[prediction_index].push(truth_index);
            }
        }
        adjacency[prediction_index].sort_by(|left, right| match rule {
            EventMatchRule::StartTolerance { .. } => {
                start_error(prediction.interval, truth[*left].interval)
                    .total_cmp(&start_error(prediction.interval, truth[*right].interval))
                    .then_with(|| truth[*left].id.cmp(&truth[*right].id))
            }
            EventMatchRule::TemporalIou { .. } => {
                temporal_iou(prediction.interval, truth[*right].interval)
                    .total_cmp(&temporal_iou(prediction.interval, truth[*left].interval))
                    .then_with(|| truth[*left].id.cmp(&truth[*right].id))
            }
        });
    }
    adjacency
}

fn augment_match(
    prediction_index: usize,
    adjacency: &[Vec<usize>],
    seen_truth: &mut [bool],
    truth_to_prediction: &mut [Option<usize>],
) -> bool {
    for &truth_index in &adjacency[prediction_index] {
        if seen_truth[truth_index] {
            continue;
        }
        seen_truth[truth_index] = true;
        let previous_prediction = truth_to_prediction[truth_index];
        if previous_prediction.is_none()
            || augment_match(
                previous_prediction.expect("checked as Some"),
                adjacency,
                seen_truth,
                truth_to_prediction,
            )
        {
            truth_to_prediction[truth_index] = Some(prediction_index);
            return true;
        }
    }
    false
}

fn events_match(predicted: TimeSpan, actual: TimeSpan, rule: &EventMatchRule) -> bool {
    match rule {
        EventMatchRule::StartTolerance { max_start_error_s } => {
            start_error(predicted, actual) <= *max_start_error_s
        }
        EventMatchRule::TemporalIou { min_iou } => temporal_iou(predicted, actual) >= *min_iou,
    }
}

fn start_error(left: TimeSpan, right: TimeSpan) -> f64 {
    (left.start_s - right.start_s).abs()
}

fn temporal_iou(left: TimeSpan, right: TimeSpan) -> f64 {
    let intersection = (left.end_s.min(right.end_s) - left.start_s.max(right.start_s)).max(0.0);
    let union = (left.end_s.max(right.end_s) - left.start_s.min(right.start_s)).max(0.0);
    if union > 0.0 { intersection / union } else { 0.0 }
}

fn valid_interval(interval: TimeSpan) -> bool {
    interval.start_s.is_finite()
        && interval.end_s.is_finite()
        && interval.start_s >= 0.0
        && interval.end_s > interval.start_s
}

fn ratio(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

fn is_sha256_hex(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bioacoustic_corpus::{
        BioacousticAnnotation, BioacousticIntendedUse, BioacousticRecording,
        BioacousticRightsEvidence, RightsAdmissionDecision,
    };

    fn recording(
        sample_id: &str,
        split: BioacousticSplit,
        source: &str,
        annotations: Vec<BioacousticAnnotation>,
    ) -> BioacousticRecording {
        BioacousticRecording {
            sample_id: sample_id.into(),
            asset_uri: format!("audio/{sample_id}.wav"),
            content_sha256: match sample_id {
                "train-1" => "a".repeat(64),
                "valid-1" => "b".repeat(64),
                "test-1" => "c".repeat(64),
                _ => "d".repeat(64),
            },
            split,
            sample_rate_hz: 10,
            channels: 1,
            sample_count: 100,
            source_recording_id: Some(source.into()),
            source_start_s: None,
            session_id: Some(source.into()),
            site_id: Some(format!("site-{source}")),
            individual_ids: vec![format!("animal-{source}")],
            annotations,
            unscored_regions: Vec::new(),
        }
    }

    fn annotation(id: &str, start_s: f64, end_s: f64, label: &str) -> BioacousticAnnotation {
        BioacousticAnnotation {
            annotation_id: id.into(),
            interval: TimeSpan { start_s, end_s },
            label: label.into(),
            annotation_source: "expert-adjudicated-v1".into(),
            frequency_min_hz: None,
            frequency_max_hz: None,
        }
    }

    fn fixture() -> (BioacousticCorpusManifest, BioacousticEventPredictionBundle) {
        let manifest = BioacousticCorpusManifest {
            schema_version: crate::bioacoustic_corpus::BIOACOUSTIC_CORPUS_SCHEMA_VERSION,
            dataset_id: "pilot".into(),
            dataset_revision: "rev-a".into(),
            source_uri: "https://example.org/pilot".into(),
            intended_use: BioacousticIntendedUse::NonCommercialResearch,
            rights: BioacousticRightsEvidence {
                license_id: "TEST".into(),
                license_uri: "https://example.org/license".into(),
                source_asset_terms_uri: "https://example.org/terms".into(),
                decision_reference: "review-1".into(),
                intended_use: BioacousticIntendedUse::NonCommercialResearch,
                decision: RightsAdmissionDecision::Approved,
            },
            recordings: vec![
                recording(
                    "train-1",
                    BioacousticSplit::Train,
                    "src-train",
                    vec![annotation("train-event", 1.0, 2.0, "whistle")],
                ),
                recording(
                    "valid-1",
                    BioacousticSplit::Validation,
                    "src-valid",
                    vec![annotation("valid-event", 1.0, 2.0, "whistle")],
                ),
                recording(
                    "test-1",
                    BioacousticSplit::Test,
                    "src-test-1",
                    vec![
                        annotation("truth-1", 1.0, 2.0, "whistle"),
                        annotation("truth-2", 3.0, 4.0, "whistle"),
                    ],
                ),
                recording("test-2", BioacousticSplit::Test, "src-test-2", vec![]),
            ],
        };
        let bundle = BioacousticEventPredictionBundle {
            schema_version: BIOACOUSTIC_EVENT_PREDICTION_SCHEMA_VERSION,
            dataset_id: "pilot".into(),
            dataset_revision: "rev-a".into(),
            corpus_manifest_blake3: manifest.computed_manifest_blake3().unwrap(),
            task_id: "whistle-detection".into(),
            model_id: "baseline-v1".into(),
            model_sha256: "b".repeat(64),
            target_labels: vec!["whistle".into()],
            matching_rule: EventMatchRule::TemporalIou { min_iou: 0.5 },
            frozen_score_threshold: 0.5,
            exhaustive_annotation_sample_ids: vec!["test-1".into(), "test-2".into()],
            split_policy: BioacousticSplitPolicy {
                require_source_recording_ids: true,
                hold_out_sessions: true,
                hold_out_sites: true,
                hold_out_individuals: true,
                require_annotations: false,
            },
            predictions: vec![
                PredictedBioacousticEvent {
                    prediction_id: "pred-1".into(),
                    sample_id: "test-1".into(),
                    interval: TimeSpan { start_s: 1.1, end_s: 2.0 },
                    label: "whistle".into(),
                    score: 0.9,
                },
                PredictedBioacousticEvent {
                    prediction_id: "pred-duplicate".into(),
                    sample_id: "test-1".into(),
                    interval: TimeSpan { start_s: 1.1, end_s: 2.0 },
                    label: "whistle".into(),
                    score: 0.8,
                },
                PredictedBioacousticEvent {
                    prediction_id: "pred-low".into(),
                    sample_id: "test-1".into(),
                    interval: TimeSpan { start_s: 3.0, end_s: 4.0 },
                    label: "whistle".into(),
                    score: 0.1,
                },
            ],
        };
        (manifest, bundle)
    }

    #[test]
    fn prediction_bundle_identity_is_order_and_hex_case_invariant() {
        let (_, mut bundle) = fixture();
        let original = bundle.computed_bundle_blake3().unwrap();
        bundle.predictions.reverse();
        bundle.target_labels.reverse();
        bundle.exhaustive_annotation_sample_ids.reverse();
        bundle.model_sha256 = bundle.model_sha256.to_ascii_uppercase();
        bundle.corpus_manifest_blake3 = bundle.corpus_manifest_blake3.to_ascii_uppercase();
        assert_eq!(original, bundle.computed_bundle_blake3().unwrap());
    }

    #[test]
    fn scores_test_only_with_one_to_one_matching_and_frozen_threshold() {
        let (manifest, bundle) = fixture();
        let report = evaluate_bioacoustic_events(&manifest, &bundle).unwrap();
        assert_eq!(
            report.schema_version,
            BIOACOUSTIC_EVENT_REPORT_SCHEMA_VERSION
        );
        assert_eq!(report.scored_split, BioacousticSplit::Test);
        assert_eq!(report.test_recording_count, 2);
        assert_eq!(report.metrics.true_positives, 1);
        assert_eq!(report.metrics.false_positives, 1);
        assert_eq!(report.metrics.false_negatives, 1);
        assert_eq!(report.metrics.ground_truth_events, 2);
        assert_eq!(report.metrics.predictions_after_threshold, 2);
        assert_eq!(report.macro_f1_recording_count, 1);
        assert_eq!(report.macro_f1_over_recordings, Some(0.5));
        assert!((report.metrics.average_precision.unwrap() - 5.0 / 6.0).abs() < 1e-9);
        assert!(report.exhaustive_test_annotation_coverage_asserted);
        assert!(!report.audio_bytes_verified_by_this_evaluator);
    }

    #[test]
    fn predictions_in_explicitly_unscored_regions_are_not_counted_as_false_alarms() {
        let (mut manifest, mut bundle) = fixture();
        manifest.recordings[2].unscored_regions.push(
            crate::bioacoustic_corpus::BioacousticUnscoredRegion {
                interval: TimeSpan {
                    start_s: 5.0,
                    end_s: 6.0,
                },
                reason: "ambiguous overlapping caller".into(),
                annotation_source: "expert-reviewed-uncertainty-v1".into(),
            },
        );
        bundle.predictions.push(PredictedBioacousticEvent {
            prediction_id: "pred-ambiguous".into(),
            sample_id: "test-1".into(),
            interval: TimeSpan {
                start_s: 5.1,
                end_s: 5.8,
            },
            label: "whistle".into(),
            score: 0.99,
        });
        bundle.corpus_manifest_blake3 = manifest.computed_manifest_blake3().unwrap();

        let report = evaluate_bioacoustic_events(&manifest, &bundle).unwrap();
        assert_eq!(report.metrics.predictions_after_threshold, 2);
        assert_eq!(report.predictions_excluded_by_unscored_regions, 1);
        assert_eq!(report.per_sample[0].predictions_excluded_by_unscored_regions, 1);
        assert_eq!(report.metrics.false_positives, 1);
        assert_eq!(report.metrics.unscored_duration_seconds, 1.0);
        assert_eq!(report.metrics.scored_duration_seconds, 19.0);
        assert!(
            (report.metrics.false_alarms_per_scored_hour.unwrap() - (3600.0 / 19.0)).abs()
                < 1e-9
        );
        assert!((report.metrics.average_precision.unwrap() - 5.0 / 6.0).abs() < 1e-9);
    }

    #[test]
    fn rejects_incomplete_exhaustive_annotation_declaration() {
        let (manifest, mut bundle) = fixture();
        bundle.exhaustive_annotation_sample_ids.pop();
        assert!(evaluate_bioacoustic_events(&manifest, &bundle).is_err());
    }

    #[test]
    fn rejects_train_or_validation_predictions() {
        let (manifest, mut bundle) = fixture();
        bundle.predictions.push(PredictedBioacousticEvent {
            prediction_id: "leak".into(),
            sample_id: "train-1".into(),
            interval: TimeSpan { start_s: 1.0, end_s: 2.0 },
            label: "whistle".into(),
            score: 0.9,
        });
        assert!(evaluate_bioacoustic_events(&manifest, &bundle).is_err());
    }

    #[test]
    fn rejects_changed_manifest_and_missing_lineage_policy() {
        let (manifest, mut bundle) = fixture();
        bundle.corpus_manifest_blake3 = "0".repeat(64);
        assert!(evaluate_bioacoustic_events(&manifest, &bundle).is_err());

        let (manifest, mut bundle) = fixture();
        bundle.split_policy.require_source_recording_ids = false;
        assert!(evaluate_bioacoustic_events(&manifest, &bundle).is_err());
    }

    #[test]
    fn maximum_matching_does_not_double_count_truth_events() {
        let rule = EventMatchRule::StartTolerance { max_start_error_s: 0.2 };
        let truth = vec![
            Event {
                id: "a".into(),
                sample_id: "s".into(),
                interval: TimeSpan {
                    start_s: 1.0,
                    end_s: 1.1,
                },
                label: "click".into(),
                score: 1.0,
            },
            Event {
                id: "b".into(),
                sample_id: "s".into(),
                interval: TimeSpan {
                    start_s: 1.15,
                    end_s: 1.25,
                },
                label: "click".into(),
                score: 1.0,
            },
        ];
        let predictions = vec![
            Event {
                id: "p1".into(),
                sample_id: "s".into(),
                interval: TimeSpan {
                    start_s: 1.1,
                    end_s: 1.2,
                },
                label: "click".into(),
                score: 0.9,
            },
            Event {
                id: "p2".into(),
                sample_id: "s".into(),
                interval: TimeSpan {
                    start_s: 0.85,
                    end_s: 0.95,
                },
                label: "click".into(),
                score: 0.8,
            },
        ];
        let (tp, fp, fn_) = match_events(&truth, &predictions, &rule);
        assert_eq!((tp, fp, fn_), (2, 0, 0));
    }

    #[test]
    fn rejects_overlapping_test_crops_before_reporting_false_alarms_per_hour() {
        let (mut manifest, mut bundle) = fixture();
        manifest.recordings[2].source_start_s = Some(0.0);
        manifest.recordings[3].source_recording_id = Some("src-test-1".into());
        manifest.recordings[3].source_start_s = Some(5.0);
        bundle.corpus_manifest_blake3 = manifest.computed_manifest_blake3().unwrap();

        let error = evaluate_bioacoustic_events(&manifest, &bundle).unwrap_err();
        assert!(error.contains("overlapping Test assets"));
    }

    #[test]
    fn accepts_non_overlapping_test_crops_and_uses_scored_duration() {
        let (mut manifest, mut bundle) = fixture();
        manifest.recordings[2].source_start_s = Some(0.0);
        manifest.recordings[3].source_recording_id = Some("src-test-1".into());
        manifest.recordings[3].session_id = Some("src-test-1".into());
        manifest.recordings[3].site_id = Some("site-src-test-1".into());
        manifest.recordings[3].individual_ids = vec!["animal-src-test-1".into()];
        manifest.recordings[3].source_start_s = Some(10.0);
        bundle.corpus_manifest_blake3 = manifest.computed_manifest_blake3().unwrap();

        let report = evaluate_bioacoustic_events(&manifest, &bundle).unwrap();
        assert_eq!(report.metrics.scored_duration_seconds, 20.0);
        assert_eq!(report.metrics.false_alarms_per_scored_hour, Some(180.0));
    }

    #[test]
    fn rejects_duplicate_prediction_ids_and_invalid_scores() {
        let (manifest, mut bundle) = fixture();
        bundle.predictions[1].prediction_id = bundle.predictions[0].prediction_id.clone();
        assert!(evaluate_bioacoustic_events(&manifest, &bundle).is_err());

        let (manifest, mut bundle) = fixture();
        bundle.predictions[0].score = f64::NAN;
        assert!(evaluate_bioacoustic_events(&manifest, &bundle).is_err());
    }
}
