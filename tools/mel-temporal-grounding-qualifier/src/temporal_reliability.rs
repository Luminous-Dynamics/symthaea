//! Corpus-level inter-annotator reliability for musical temporal grounding.
//!
//! Nominal disposition reliability is computed with a transparent,
//! missing-rater-tolerant Krippendorff alpha calculation. Event-set and
//! boundary agreement remain separate and reuse the production temporal
//! grounding scorer at a preregistered IoU-threshold sweep.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::temporal_consensus::IndependentTemporalAnnotationV1;
use crate::temporal_grounding::{
    AudioArtifactClockV1, GroundTruthStateV1, GroundingPredictionStateV1,
    TEMPORAL_GROUNDING_SCHEMA_V1, TemporalGroundTruthV1, TemporalGroundingPlanV1,
    TemporalGroundingPredictionV1,
    score_temporal_grounding, validate_temporal_ground_truth,
    validate_temporal_grounding_plan,
};

const CORPUS_SCHEMA_V1: &str = "mel-temporal-annotation-reliability-corpus-v1";
const REPORT_SCHEMA_V1: &str = "mel-temporal-annotation-reliability-report-v1";
const RELIABILITY_EVALUATOR_V1: &str = "mel-temporal-annotation-reliability-v1";
const MAX_ANNOTATIONS_PER_UNIT_V1: usize = 256;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalReliabilityUnitV1 {
    pub unit_id: String,
    /// Opaque stable source-work lineage digest. Crops and edits of one work
    /// must share this value so they cannot be mistaken for independent works.
    pub source_lineage_sha256: String,
    pub query_id: String,
    pub query_text: String,
    pub artifact: AudioArtifactClockV1,
    pub annotation_manifest_sha256: String,
    pub annotations: Vec<IndependentTemporalAnnotationV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TemporalReliabilityCorpusV1 {
    pub schema: String,
    pub annotation_policy_version: String,
    pub minimum_annotations_per_unit: u16,
    /// Frozen evaluator identity and all other scoring-plan fields.
    pub plan: TemporalGroundingPlanV1,
    /// Must be strictly increasing, unique, and in (0, 1].
    pub interval_iou_thresholds: Vec<f64>,
    pub units: Vec<TemporalReliabilityUnitV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct NominalKrippendorffAlphaV1 {
    pub metric: &'static str,
    pub items: usize,
    pub ratings: usize,
    pub category_counts: BTreeMap<String, usize>,
    /// Observed pairwise disagreement under the Krippendorff coincidence matrix.
    pub observed_disagreement: Option<f64>,
    /// Expected disagreement from the pooled marginal category distribution.
    pub expected_disagreement: Option<f64>,
    pub alpha: Option<f64>,
    pub status_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TemporalPairwiseReliabilityV1 {
    pub threshold: f64,
    pub present_present_pairs: usize,
    pub symmetric_event_f1_mean: Option<f64>,
    pub matched_interval_iou_mean: Option<f64>,
    pub exact_occurrence_count_agreement_rate: Option<f64>,
    /// Boundary errors use one direction only to avoid double-counting each pair.
    pub start_boundary_mae_frames: Option<f64>,
    pub end_boundary_mae_frames: Option<f64>,
    pub start_boundary_p95_abs_error_frames: Option<f64>,
    pub end_boundary_p95_abs_error_frames: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TemporalReliabilityUnitSummaryV1 {
    pub unit_id: String,
    pub source_lineage_sha256: String,
    pub annotation_count: usize,
    pub disposition_pair_count: usize,
    pub disposition_agreement_rate: Option<f64>,
    pub pairwise_by_iou_threshold: Vec<TemporalPairwiseReliabilityV1>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TemporalReliabilityReportV1 {
    pub schema: &'static str,
    pub evaluator_version: &'static str,
    pub annotation_policy_version: String,
    pub item_count: usize,
    pub source_work_clusters: usize,
    pub total_annotation_records: usize,
    pub minimum_annotations_per_unit: u16,
    /// Formal alpha applies only to the nominal four-state disposition.
    pub disposition_alpha: NominalKrippendorffAlphaV1,
    /// Macro average across query items, so a heavily annotated item cannot
    /// dominate a sparsely annotated item.
    pub pairwise_by_iou_threshold: Vec<TemporalPairwiseReliabilityV1>,
    pub unit_summaries: Vec<TemporalReliabilityUnitSummaryV1>,
    pub raw_annotation_records_preserved: bool,
    pub consensus_ground_truth_created: bool,
    pub model_performance_scored: bool,
    pub population_confidence_interval_computed: bool,
    pub source_work_bootstrap_computed: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct CategoryCounts {
    present: usize,
    absent: usize,
    ambiguous: usize,
    not_estimable: usize,
}

impl CategoryCounts {
    fn increment(&mut self, state: GroundTruthStateV1) {
        match state {
            GroundTruthStateV1::Present => self.present += 1,
            GroundTruthStateV1::Absent => self.absent += 1,
            GroundTruthStateV1::Ambiguous => self.ambiguous += 1,
            GroundTruthStateV1::NotEstimable => self.not_estimable += 1,
        }
    }

    fn values(self) -> [usize; 4] {
        [self.present, self.absent, self.ambiguous, self.not_estimable]
    }

    fn as_map(self) -> BTreeMap<String, usize> {
        BTreeMap::from([
            ("Present".into(), self.present),
            ("Absent".into(), self.absent),
            ("Ambiguous".into(), self.ambiguous),
            ("NotEstimable".into(), self.not_estimable),
        ])
    }
}

pub fn audit_reliability_corpus_json(input: &str) -> Result<TemporalReliabilityReportV1, String> {
    let corpus: TemporalReliabilityCorpusV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid temporal reliability corpus JSON: {error}"))?;
    audit_reliability_corpus(&corpus)
}

pub fn audit_reliability_corpus(
    corpus: &TemporalReliabilityCorpusV1,
) -> Result<TemporalReliabilityReportV1, String> {
    validate_corpus(corpus)?;
    let mut global_counts = CategoryCounts::default();
    let mut total_ratings = 0_usize;
    let mut observed_disagreement_total = 0.0_f64;
    let mut source_lineages = BTreeSet::new();
    let mut total_annotation_records = 0_usize;
    let mut unit_summaries = Vec::with_capacity(corpus.units.len());

    for unit in &corpus.units {
        source_lineages.insert(unit.source_lineage_sha256.as_str());
        let mut counts = CategoryCounts::default();
        for annotation in &unit.annotations {
            counts.increment(annotation.state);
        }
        let item_ratings = unit.annotations.len();
        let sum_squares = counts.values().iter().try_fold(0_usize, |sum, count| {
            let square = count
                .checked_mul(*count)
                .ok_or_else(|| "item category count square overflow".to_string())?;
            sum.checked_add(square)
                .ok_or_else(|| "item category count sum overflow".to_string())
        })?;
        let all_ordered_pairs = item_ratings
            .checked_mul(item_ratings)
            .ok_or_else(|| "item ordered-pair count overflow".to_string())?;
        let ordered_disagreements = all_ordered_pairs
            .checked_sub(sum_squares)
            .ok_or_else(|| "item coincidence disagreement underflow".to_string())?;
        let item_denominator = item_ratings
            .checked_sub(1)
            .ok_or_else(|| "item has fewer than two ratings".to_string())?;
        observed_disagreement_total +=
            ordered_disagreements as f64 / item_denominator as f64;
        total_ratings = total_ratings
            .checked_add(item_ratings)
            .ok_or_else(|| "corpus rating count overflow".to_string())?;
        total_annotation_records = total_annotation_records
            .checked_add(item_ratings)
            .ok_or_else(|| "corpus annotation record count overflow".to_string())?;
        global_counts.present = global_counts
            .present
            .checked_add(counts.present)
            .ok_or_else(|| "global Present count overflow".to_string())?;
        global_counts.absent = global_counts
            .absent
            .checked_add(counts.absent)
            .ok_or_else(|| "global Absent count overflow".to_string())?;
        global_counts.ambiguous = global_counts
            .ambiguous
            .checked_add(counts.ambiguous)
            .ok_or_else(|| "global Ambiguous count overflow".to_string())?;
        global_counts.not_estimable = global_counts
            .not_estimable
            .checked_add(counts.not_estimable)
            .ok_or_else(|| "global NotEstimable count overflow".to_string())?;
        unit_summaries.push(summarize_unit(unit, &corpus.plan, &corpus.interval_iou_thresholds)?);
    }

    let observed_disagreement = (total_ratings > 0)
        .then_some(observed_disagreement_total / total_ratings as f64);
    let global_sum_squares = global_counts.values().iter().try_fold(
        0_usize,
        |sum, count| {
            let square = count
                .checked_mul(*count)
                .ok_or_else(|| "global category count square overflow".to_string())?;
            sum.checked_add(square)
                .ok_or_else(|| "global category count sum overflow".to_string())
        },
    )?;
    let expected_disagreement = if total_ratings > 1 {
        let all_pairs = total_ratings
            .checked_mul(total_ratings)
            .ok_or_else(|| "global ordered-pair count overflow".to_string())?;
        let denominator = total_ratings
            .checked_mul(total_ratings - 1)
            .ok_or_else(|| "global coincidence denominator overflow".to_string())?;
        let numerator = all_pairs
            .checked_sub(global_sum_squares)
            .ok_or_else(|| "global coincidence disagreement underflow".to_string())?;
        Some(numerator as f64 / denominator as f64)
    } else {
        None
    };
    let (alpha, alpha_status_reason) = match (observed_disagreement, expected_disagreement) {
        (Some(_observed), Some(expected)) if expected > f64::EPSILON => {
            let value = 1.0 - _observed / expected;
            (Some(value), None)
        }
        (Some(_), Some(_)) => (
            None,
            Some("expected nominal disagreement is zero; all ratings share one category".into()),
        ),
        _ => (
            None,
            Some("insufficient ratings to calculate nominal Krippendorff alpha".into()),
        ),
    };

    let disposition_alpha = NominalKrippendorffAlphaV1 {
        metric: "krippendorff_alpha_nominal",
        items: corpus.units.len(),
        ratings: total_ratings,
        category_counts: global_counts.as_map(),
        observed_disagreement,
        expected_disagreement,
        alpha,
        status_reason: alpha_status_reason,
    };

    let pairwise_by_iou_threshold =
        aggregate_threshold_summaries(&unit_summaries, &corpus.interval_iou_thresholds);

    Ok(TemporalReliabilityReportV1 {
        schema: REPORT_SCHEMA_V1,
        evaluator_version: RELIABILITY_EVALUATOR_V1,
        annotation_policy_version: corpus.annotation_policy_version.clone(),
        item_count: corpus.units.len(),
        source_work_clusters: source_lineages.len(),
        total_annotation_records,
        minimum_annotations_per_unit: corpus.minimum_annotations_per_unit,
        disposition_alpha,
        pairwise_by_iou_threshold,
        unit_summaries,
        raw_annotation_records_preserved: true,
        consensus_ground_truth_created: false,
        model_performance_scored: false,
        population_confidence_interval_computed: false,
        source_work_bootstrap_computed: false,
    })
}

fn validate_corpus(corpus: &TemporalReliabilityCorpusV1) -> Result<(), String> {
    if corpus.schema != CORPUS_SCHEMA_V1 {
        return Err(format!("unsupported reliability corpus schema: {}", corpus.schema));
    }
    require_text(&corpus.annotation_policy_version, "annotation_policy_version")?;
    if corpus.minimum_annotations_per_unit < 2 {
        return Err("minimum_annotations_per_unit must be at least two".into());
    }
    let plan_issues = validate_temporal_grounding_plan(&corpus.plan);
    if !plan_issues.is_empty() {
        return Err(format!("invalid frozen temporal grounding plan: {plan_issues:?}"));
    }
    if corpus.units.len() < 2 {
        return Err("formal reliability requires at least two query-items".into());
    }
    if corpus.interval_iou_thresholds.is_empty() {
        return Err("at least one interval IoU threshold is required".into());
    }
    let mut previous_threshold = 0.0;
    for threshold in &corpus.interval_iou_thresholds {
        if !threshold.is_finite() || *threshold <= previous_threshold || *threshold > 1.0 {
            return Err("IoU thresholds must be finite, unique, strictly increasing and in (0,1]".into());
        }
        previous_threshold = *threshold;
    }

    let mut unit_ids = BTreeSet::new();
    let mut annotation_digests = BTreeSet::new();
    for (index, unit) in corpus.units.iter().enumerate() {
        require_text(&unit.unit_id, &format!("units[{index}].unit_id"))?;
        require_text(&unit.query_id, &format!("units[{index}].query_id"))?;
        require_text(&unit.query_text, &format!("units[{index}].query_text"))?;
        validate_digest(&unit.source_lineage_sha256, "source_lineage_sha256")?;
        validate_digest(&unit.artifact.audio_sha256, "artifact.audio_sha256")?;
        validate_digest(&unit.annotation_manifest_sha256, "annotation_manifest_sha256")?;
        if !unit_ids.insert(unit.unit_id.as_str()) {
            return Err(format!("duplicate reliability unit_id: {}", unit.unit_id));
        }
        if unit.annotations.len() < corpus.minimum_annotations_per_unit as usize {
            return Err(format!(
                "unit {} has {} annotations, below minimum {}",
                unit.unit_id,
                unit.annotations.len(),
                corpus.minimum_annotations_per_unit
            ));
        }
        if unit.annotations.len() > MAX_ANNOTATIONS_PER_UNIT_V1 {
            return Err(format!(
                "unit {} exceeds the maximum of {} independent annotations",
                unit.unit_id,
                MAX_ANNOTATIONS_PER_UNIT_V1
            ));
        }
        let mut annotator_ids = BTreeSet::new();
        for annotation in &unit.annotations {
            validate_digest(&annotation.annotator_id_sha256, "annotator_id_sha256")?;
            validate_digest(&annotation.annotation_sha256, "annotation_sha256")?;
            if !annotator_ids.insert(annotation.annotator_id_sha256.as_str()) {
                return Err(format!(
                    "unit {} repeats an annotator ID",
                    unit.unit_id
                ));
            }
            if !annotation_digests.insert(annotation.annotation_sha256.as_str()) {
                return Err("annotation record digest is reused across the corpus".into());
            }
            let truth = annotation_truth(corpus, unit, annotation);
            let issues = validate_temporal_ground_truth(&truth);
            if !issues.is_empty() {
                return Err(format!(
                    "unit {} annotation rejected by production validator: {issues:?}",
                    unit.unit_id
                ));
            }
        }
    }
    Ok(())
}

fn summarize_unit(
    unit: &TemporalReliabilityUnitV1,
    plan: &TemporalGroundingPlanV1,
    thresholds: &[f64],
) -> Result<TemporalReliabilityUnitSummaryV1, String> {
    let count = unit.annotations.len();
    let pair_count = count.saturating_mul(count.saturating_sub(1)) / 2;
    let mut agreement_pairs = 0_usize;
    let mut present_pairs = 0_usize;
    for left in 0..count {
        for right in (left + 1)..count {
            let a = &unit.annotations[left];
            let b = &unit.annotations[right];
            if a.state == b.state {
                agreement_pairs += 1;
            }
            if a.state == GroundTruthStateV1::Present && b.state == GroundTruthStateV1::Present {
                present_pairs += 1;
            }
        }
    }

    let mut pairwise_by_iou_threshold = Vec::with_capacity(thresholds.len());
    for threshold in thresholds {
        let mut f1_values = Vec::new();
        let mut iou_values = Vec::new();
        let mut count_agreement = Vec::new();
        let mut start_errors = Vec::new();
        let mut end_errors = Vec::new();
        for left in 0..count {
            for right in (left + 1)..count {
                let a = &unit.annotations[left];
                let b = &unit.annotations[right];
                if a.state != GroundTruthStateV1::Present
                    || b.state != GroundTruthStateV1::Present
                {
                    continue;
                }
                let mut threshold_plan = plan.clone();
                threshold_plan.minimum_interval_iou = *threshold;
                let left_truth = annotation_truth_for_unit(unit, a, &plan.evaluator_version);
                let right_truth = annotation_truth_for_unit(unit, b, &plan.evaluator_version);
                let left_prediction = annotation_prediction(unit, a);
                let right_prediction = annotation_prediction(unit, b);
                let left_score = score_temporal_grounding(
                    &left_truth,
                    &right_prediction,
                    &threshold_plan,
                )
                .map_err(|issues| format!("left→right event agreement failed: {issues:?}"))?;
                let right_score = score_temporal_grounding(
                    &right_truth,
                    &left_prediction,
                    &threshold_plan,
                )
                .map_err(|issues| format!("right→left event agreement failed: {issues:?}"))?;
                if let (Some(left_f1), Some(right_f1)) = (left_score.f1, right_score.f1) {
                    f1_values.push((left_f1 + right_f1) / 2.0);
                }
                if let (Some(left_iou), Some(right_iou)) = (
                    left_score.matched_interval_iou_mean,
                    right_score.matched_interval_iou_mean,
                ) {
                    iou_values.push((left_iou + right_iou) / 2.0);
                }
                count_agreement.push(if a.intervals.len() == b.intervals.len() { 1.0 } else { 0.0 });
                for matched in &left_score.matches {
                    start_errors.push(matched.start_error_frames.unsigned_abs() as f64);
                    end_errors.push(matched.end_error_frames.unsigned_abs() as f64);
                }
            }
        }
        pairwise_by_iou_threshold.push(TemporalPairwiseReliabilityV1 {
            threshold: *threshold,
            present_present_pairs: f1_values.len(),
            symmetric_event_f1_mean: mean(&f1_values),
            matched_interval_iou_mean: mean(&iou_values),
            exact_occurrence_count_agreement_rate: mean(&count_agreement),
            start_boundary_mae_frames: mean(&start_errors),
            end_boundary_mae_frames: mean(&end_errors),
            start_boundary_p95_abs_error_frames: percentile(&start_errors, 0.95),
            end_boundary_p95_abs_error_frames: percentile(&end_errors, 0.95),
        });
    }

    Ok(TemporalReliabilityUnitSummaryV1 {
        unit_id: unit.unit_id.clone(),
        source_lineage_sha256: unit.source_lineage_sha256.clone(),
        annotation_count: count,
        disposition_pair_count: pair_count,
        disposition_agreement_rate: rate(agreement_pairs, pair_count),
        pairwise_by_iou_threshold,
    })
}

fn aggregate_threshold_summaries(
    units: &[TemporalReliabilityUnitSummaryV1],
    thresholds: &[f64],
) -> Vec<TemporalPairwiseReliabilityV1> {
    thresholds
        .iter()
        .enumerate()
        .map(|(index, threshold)| {
            let values: Vec<_> = units
                .iter()
                .map(|unit| &unit.pairwise_by_iou_threshold[index])
                .collect();
            TemporalPairwiseReliabilityV1 {
                threshold: *threshold,
                present_present_pairs: values
                    .iter()
                    .map(|value| value.present_present_pairs)
                    .sum(),
                symmetric_event_f1_mean: mean(
                    &values
                        .iter()
                        .filter_map(|value| value.symmetric_event_f1_mean)
                        .collect::<Vec<_>>(),
                ),
                matched_interval_iou_mean: mean(
                    &values
                        .iter()
                        .filter_map(|value| value.matched_interval_iou_mean)
                        .collect::<Vec<_>>(),
                ),
                exact_occurrence_count_agreement_rate: mean(
                    &values
                        .iter()
                        .filter_map(|value| value.exact_occurrence_count_agreement_rate)
                        .collect::<Vec<_>>(),
                ),
                start_boundary_mae_frames: mean(
                    &values
                        .iter()
                        .filter_map(|value| value.start_boundary_mae_frames)
                        .collect::<Vec<_>>(),
                ),
                end_boundary_mae_frames: mean(
                    &values
                        .iter()
                        .filter_map(|value| value.end_boundary_mae_frames)
                        .collect::<Vec<_>>(),
                ),
                start_boundary_p95_abs_error_frames: percentile(
                    &values
                        .iter()
                        .filter_map(|value| value.start_boundary_p95_abs_error_frames)
                        .collect::<Vec<_>>(),
                    0.95,
                ),
                end_boundary_p95_abs_error_frames: percentile(
                    &values
                        .iter()
                        .filter_map(|value| value.end_boundary_p95_abs_error_frames)
                        .collect::<Vec<_>>(),
                    0.95,
                ),
            }
        })
        .collect()
}

fn annotation_truth(
    corpus: &TemporalReliabilityCorpusV1,
    unit: &TemporalReliabilityUnitV1,
    annotation: &IndependentTemporalAnnotationV1,
) -> TemporalGroundTruthV1 {
    TemporalGroundTruthV1 {
        schema_version: TEMPORAL_GROUNDING_SCHEMA_V1.into(),
        query_id: unit.query_id.clone(),
        query_text: unit.query_text.clone(),
        artifact: unit.artifact.clone(),
        annotation_policy_version: corpus.annotation_policy_version.clone(),
        annotation_set_sha256: annotation.annotation_sha256.clone(),
        annotator_count: 1,
        state: annotation.state,
        intervals: annotation.intervals.clone(),
        status_reason: annotation.status_reason.clone(),
    }
}

fn annotation_truth_for_unit(
    unit: &TemporalReliabilityUnitV1,
    annotation: &IndependentTemporalAnnotationV1,
    annotation_policy_version: &str,
) -> TemporalGroundTruthV1 {
    TemporalGroundTruthV1 {
        schema_version: TEMPORAL_GROUNDING_SCHEMA_V1.into(),
        query_id: unit.query_id.clone(),
        query_text: unit.query_text.clone(),
        artifact: unit.artifact.clone(),
        annotation_policy_version: annotation_policy_version.into(),
        annotation_set_sha256: annotation.annotation_sha256.clone(),
        annotator_count: 1,
        state: annotation.state,
        intervals: annotation.intervals.clone(),
        status_reason: annotation.status_reason.clone(),
    }
}

fn annotation_prediction(
    unit: &TemporalReliabilityUnitV1,
    annotation: &IndependentTemporalAnnotationV1,
) -> TemporalGroundingPredictionV1 {
    TemporalGroundingPredictionV1 {
        schema_version: TEMPORAL_GROUNDING_SCHEMA_V1.into(),
        query_id: unit.query_id.clone(),
        artifact: unit.artifact.clone(),
        state: GroundingPredictionStateV1::Occurrences {
            intervals: annotation.intervals.clone(),
        },
    }
}

fn validate_digest(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(format!("{label} must be a lowercase SHA-256 digest"));
    }
    Ok(())
}

fn require_text(value: &str, label: &str) -> Result<(), String> {
    if value.trim().is_empty() {
        return Err(format!("{label} must not be empty"));
    }
    Ok(())
}

fn rate(numerator: usize, denominator: usize) -> Option<f64> {
    (denominator > 0).then(|| numerator as f64 / denominator as f64)
}

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn percentile(values: &[f64], quantile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let index = ((sorted.len() - 1) as f64 * quantile).ceil() as usize;
    sorted.get(index).copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_consensus::IndependentTemporalAnnotationV1;
    use crate::temporal_grounding::{
        FrameIntervalV1, TEMPORAL_GROUNDING_EVALUATOR_V1,
    };

    const AUDIO_A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const AUDIO_B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const LINEAGE_A: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const LINEAGE_B: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

    fn digest(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    fn artifact(audio: &str) -> AudioArtifactClockV1 {
        AudioArtifactClockV1 {
            audio_sha256: audio.into(),
            sample_rate_hz: 48_000,
            channels: 2,
            total_frames: 48_000 * 60,
        }
    }

    fn annotation(
        id: char,
        receipt: char,
        state: GroundTruthStateV1,
        intervals: Vec<FrameIntervalV1>,
        reason: Option<&str>,
    ) -> IndependentTemporalAnnotationV1 {
        IndependentTemporalAnnotationV1 {
            annotator_id_sha256: digest(id),
            annotation_sha256: digest(receipt),
            state,
            intervals,
            status_reason: reason.map(str::to_owned),
        }
    }

    fn unit(
        id: &str,
        lineage: &str,
        audio: &str,
        annotations: Vec<IndependentTemporalAnnotationV1>,
    ) -> TemporalReliabilityUnitV1 {
        TemporalReliabilityUnitV1 {
            unit_id: id.into(),
            source_lineage_sha256: lineage.into(),
            query_id: format!("query-{id}"),
            query_text: "Every piano entrance".into(),
            artifact: artifact(audio),
            annotation_manifest_sha256: digest(if id == "item-1" { 'e' } else { 'f' }),
            annotations,
        }
    }

    fn corpus(units: Vec<TemporalReliabilityUnitV1>) -> TemporalReliabilityCorpusV1 {
        TemporalReliabilityCorpusV1 {
            schema: CORPUS_SCHEMA_V1.into(),
            annotation_policy_version: "music-temporal-annotation-v1".into(),
            minimum_annotations_per_unit: 2,
            plan: TemporalGroundingPlanV1 {
                evaluator_version: TEMPORAL_GROUNDING_EVALUATOR_V1.into(),
                minimum_interval_iou: 0.5,
            },
            interval_iou_thresholds: vec![0.25, 0.5, 0.75, 0.9],
            units,
        }
    }

    #[test]
    fn nominal_alpha_is_one_for_perfect_state_agreement_across_items() {
        let units = vec![
            unit(
                "item-1",
                LINEAGE_A,
                AUDIO_A,
                vec![
                    annotation('1', '1', GroundTruthStateV1::Present, vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }], None),
                    annotation('2', '2', GroundTruthStateV1::Present, vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }], None),
                ],
            ),
            unit(
                "item-2",
                LINEAGE_B,
                AUDIO_B,
                vec![
                    annotation('1', '3', GroundTruthStateV1::Absent, Vec::new(), None),
                    annotation('2', '4', GroundTruthStateV1::Absent, Vec::new(), None),
                ],
            ),
        ];
        let result = audit_reliability_corpus(&corpus(units)).unwrap();
        assert_eq!(result.disposition_alpha.ratings, 4);
        assert_eq!(result.disposition_alpha.observed_disagreement, Some(0.0));
        assert_eq!(result.disposition_alpha.alpha, Some(1.0));
        assert_eq!(result.source_work_clusters, 2);
        assert!(!result.consensus_ground_truth_created);
        assert!(!result.population_confidence_interval_computed);
    }

    #[test]
    fn nominal_alpha_can_be_negative_when_annotators_systematically_disagree() {
        let units = vec![
            unit(
                "item-1",
                LINEAGE_A,
                AUDIO_A,
                vec![
                    annotation('1', '1', GroundTruthStateV1::Present, vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }], None),
                    annotation('2', '2', GroundTruthStateV1::Absent, Vec::new(), None),
                ],
            ),
            unit(
                "item-2",
                LINEAGE_B,
                AUDIO_B,
                vec![
                    annotation('1', '3', GroundTruthStateV1::Present, vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }], None),
                    annotation('2', '4', GroundTruthStateV1::Absent, Vec::new(), None),
                ],
            ),
        ];
        let result = audit_reliability_corpus(&corpus(units)).unwrap();
        assert!(result.disposition_alpha.alpha.unwrap() < 0.0);
    }

    #[test]
    fn degenerate_single_category_panel_returns_none_with_reason() {
        let units = vec![
            unit(
                "item-1",
                LINEAGE_A,
                AUDIO_A,
                vec![
                    annotation('1', '1', GroundTruthStateV1::Present, vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }], None),
                    annotation('2', '2', GroundTruthStateV1::Present, vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }], None),
                ],
            ),
            unit(
                "item-2",
                LINEAGE_B,
                AUDIO_B,
                vec![
                    annotation('1', '3', GroundTruthStateV1::Present, vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }], None),
                    annotation('2', '4', GroundTruthStateV1::Present, vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }], None),
                ],
            ),
        ];
        let result = audit_reliability_corpus(&corpus(units)).unwrap();
        assert_eq!(result.disposition_alpha.alpha, None);
        assert!(result.disposition_alpha.status_reason.is_some());
    }

    #[test]
    fn iou_threshold_sweep_separates_similar_from_exact_boundary_match() {
        let units = vec![
            unit(
                "item-1",
                LINEAGE_A,
                AUDIO_A,
                vec![
                    annotation('1', '1', GroundTruthStateV1::Present, vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }], None),
                    annotation('2', '2', GroundTruthStateV1::Present, vec![FrameIntervalV1 { start_frame: 120, end_frame: 220 }], None),
                ],
            ),
            unit(
                "item-2",
                LINEAGE_B,
                AUDIO_B,
                vec![
                    annotation('1', '3', GroundTruthStateV1::Present, vec![FrameIntervalV1 { start_frame: 300, end_frame: 400 }], None),
                    annotation('2', '4', GroundTruthStateV1::Present, vec![FrameIntervalV1 { start_frame: 300, end_frame: 400 }], None),
                ],
            ),
        ];
        let result = audit_reliability_corpus(&corpus(units)).unwrap();
        assert_eq!(result.pairwise_by_iou_threshold.len(), 4);
        assert_eq!(result.pairwise_by_iou_threshold[0].present_present_pairs, 2);
        assert!(result.pairwise_by_iou_threshold[0].symmetric_event_f1_mean.unwrap() > 0.99);
        assert!(result.pairwise_by_iou_threshold[2].symmetric_event_f1_mean.unwrap() < 1.0);
        assert!(result.pairwise_by_iou_threshold[2].start_boundary_mae_frames.is_some());
    }

    #[test]
    fn mixed_missingness_is_supported_but_each_item_needs_two_annotations() {
        let units = vec![
            unit(
                "item-1",
                LINEAGE_A,
                AUDIO_A,
                vec![
                    annotation('1', '1', GroundTruthStateV1::Present, vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }], None),
                    annotation('2', '2', GroundTruthStateV1::Absent, Vec::new(), None),
                    annotation('3', '3', GroundTruthStateV1::Ambiguous, Vec::new(), Some("soft onset")),
                ],
            ),
            unit(
                "item-2",
                LINEAGE_B,
                AUDIO_B,
                vec![
                    annotation('1', '4', GroundTruthStateV1::NotEstimable, Vec::new(), Some("masked by percussion")),
                    annotation('3', '5', GroundTruthStateV1::NotEstimable, Vec::new(), Some("masked by percussion")),
                ],
            ),
        ];
        let result = audit_reliability_corpus(&corpus(units)).unwrap();
        assert_eq!(result.total_annotation_records, 5);
        assert_eq!(result.disposition_alpha.category_counts["Present"], 1);
        assert_eq!(result.disposition_alpha.category_counts["Absent"], 1);
        assert_eq!(result.disposition_alpha.category_counts["Ambiguous"], 1);
        assert_eq!(result.disposition_alpha.category_counts["NotEstimable"], 2);
    }

    #[test]
    fn duplicate_unit_or_annotator_and_invalid_thresholds_are_rejected() {
        let good = vec![
            unit(
                "item-1",
                LINEAGE_A,
                AUDIO_A,
                vec![
                    annotation('1', '1', GroundTruthStateV1::Present, vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }], None),
                    annotation('2', '2', GroundTruthStateV1::Present, vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }], None),
                ],
            ),
            unit(
                "item-2",
                LINEAGE_B,
                AUDIO_B,
                vec![
                    annotation('1', '3', GroundTruthStateV1::Absent, Vec::new(), None),
                    annotation('2', '4', GroundTruthStateV1::Absent, Vec::new(), None),
                ],
            ),
        ];
        assert!(audit_reliability_corpus(&corpus(vec![good[0].clone(), good[0].clone()])).is_err());
        let mut duplicate_annotator = good.clone();
        duplicate_annotator[0].annotations[1].annotator_id_sha256 =
            duplicate_annotator[0].annotations[0].annotator_id_sha256.clone();
        assert!(audit_reliability_corpus(&corpus(duplicate_annotator)).is_err());
        let mut invalid_thresholds = corpus(good);
        invalid_thresholds.interval_iou_thresholds = vec![0.5, 0.5];
        assert!(audit_reliability_corpus(&invalid_thresholds).is_err());
    }
}
