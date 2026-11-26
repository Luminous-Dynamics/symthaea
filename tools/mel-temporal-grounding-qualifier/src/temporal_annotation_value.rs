//! Paired before/after annotation information and effort audit.
//!
//! This module evaluates the marginal effect of exactly one additional
//! independent annotation. It preserves disagreement and reports separate
//! changes in panel entropy, event-set concordance, boundary dispersion,
//! frozen panel-decision state, and human effort. It does not reduce these
//! dimensions to a single reward or claim that lower disagreement is always
//! better.

use serde::{Deserialize, Serialize};

use crate::temporal_consensus::{
    audit_multi_annotator_case, MultiAnnotatorGroundingReportV1,
    MultiAnnotatorTemporalCaseV1,
};

const INPUT_SCHEMA_V1: &str = "mel-temporal-annotation-marginal-value-v1";
const REPORT_SCHEMA_V1: &str = "mel-temporal-annotation-marginal-value-report-v1";
const EVALUATOR_VERSION_V1: &str = "mel-temporal-annotation-marginal-value-v1";
const PPM_SCALE: f64 = 1_000_000.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnnotationSelectionStreamV1 {
    ActiveRefinement,
    ProbabilityAudit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PanelDecisionStateV1 {
    SufficientlySupportedPresent,
    SufficientlySupportedAbsent,
    NeedsReview,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrozenPanelDecisionProfileV1 {
    pub profile_id: String,
    pub minimum_eligible_annotators: u16,
    pub minimum_present_support_ppm: u32,
    pub minimum_absent_support_ppm: u32,
    pub maximum_ambiguous_not_estimable_share_ppm: u32,
    pub minimum_pairwise_present_event_f1_ppm: u32,
    pub minimum_pairwise_present_count_agreement_ppm: u32,
    pub maximum_event_boundary_spread_ms: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnnotationEffortReceiptV1 {
    /// Opaque session receipt reference; not authenticated by this evaluator.
    pub session_receipt_sha256: String,
    /// Actual active listening, excluding idle/background wall time.
    pub active_listening_seconds: u64,
    /// Total focused review activity, including listening and comparison.
    pub active_review_seconds: u64,
    /// Session elapsed wall time, including pauses and interruptions.
    pub elapsed_wall_seconds: u64,
    pub replay_count: u32,
    pub seek_count: u32,
    /// Optional self-report, fixed-point 0..=1,000,000.
    pub self_reported_difficulty_ppm: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnnotationMarginalValueInputV1 {
    pub schema: String,
    pub source_work_id_sha256: String,
    pub selection_stream: AnnotationSelectionStreamV1,
    /// Receipt emitted by the source-balanced annotation acquisition planner.
    pub selection_receipt_sha256: String,
    pub decision_profile: FrozenPanelDecisionProfileV1,
    pub before_panel: MultiAnnotatorTemporalCaseV1,
    /// Must contain all before records unchanged, followed by exactly one new record.
    pub after_panel: MultiAnnotatorTemporalCaseV1,
    pub effort: AnnotationEffortReceiptV1,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PanelInformationSnapshotV1 {
    pub annotations: usize,
    pub eligible_present_absent_annotations: usize,
    pub present_annotations: usize,
    pub absent_annotations: usize,
    pub ambiguous_annotations: usize,
    pub not_estimable_annotations: usize,
    pub disposition_entropy_bits: Option<f64>,
    pub present_occurrence_count_entropy_bits: Option<f64>,
    pub pairwise_disposition_agreement_rate: Option<f64>,
    pub pairwise_present_symmetric_event_f1: Option<f64>,
    pub pairwise_present_mean_matched_iou: Option<f64>,
    pub pairwise_present_exact_count_agreement: Option<f64>,
    pub maximum_event_boundary_spread_ms: Option<f64>,
    pub model_fully_correct_rate_against_panel: Option<f64>,
    pub frozen_decision_state: PanelDecisionStateV1,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PerMinuteMetricDeltaV1 {
    /// Post-panel metric minus pre-panel metric, divided by active review minutes.
    /// Positive/negative values are descriptive; no sign is universally better.
    pub disposition_entropy_bits_per_active_review_minute: Option<f64>,
    pub occurrence_count_entropy_bits_per_active_review_minute: Option<f64>,
    pub pairwise_present_event_f1_per_active_review_minute: Option<f64>,
    pub pairwise_present_matched_iou_per_active_review_minute: Option<f64>,
    pub pairwise_count_agreement_per_active_review_minute: Option<f64>,
    /// Negative values indicate a smaller boundary spread; not automatically better.
    pub maximum_boundary_spread_ms_per_active_review_minute: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AnnotationMarginalValueReportV1 {
    pub schema: &'static str,
    pub evaluator_version: &'static str,
    pub source_work_id_sha256: String,
    pub query_id: String,
    pub audio_sha256: String,
    pub selection_stream: AnnotationSelectionStreamV1,
    pub selection_receipt_sha256: String,
    pub before_annotation_manifest_sha256: String,
    pub after_annotation_manifest_sha256: String,
    pub newly_added_annotator_id_sha256: String,
    pub newly_added_annotation_sha256: String,
    pub before: PanelInformationSnapshotV1,
    pub after: PanelInformationSnapshotV1,
    pub disposition_entropy_delta_bits: Option<f64>,
    pub occurrence_count_entropy_delta_bits: Option<f64>,
    pub pairwise_present_event_f1_delta: Option<f64>,
    pub pairwise_present_matched_iou_delta: Option<f64>,
    pub pairwise_present_count_agreement_delta: Option<f64>,
    pub maximum_event_boundary_spread_delta_ms: Option<f64>,
    pub decision_transition: String,
    pub effort: AnnotationEffortReceiptV1,
    pub active_listening_minutes: f64,
    pub active_review_minutes: f64,
    pub elapsed_wall_minutes: f64,
    pub per_minute_deltas: PerMinuteMetricDeltaV1,
    /// Human/model alignment changes when the panel composition changes; this
    /// is not a model improvement or a measure of annotation truth.
    pub model_fully_correct_rate_delta_against_panel: Option<f64>,
    pub annotation_record_count_delta: usize,
    /// The active stream is adaptive; its results are descriptive only.
    pub active_stream_is_probability_sample: bool,
    /// This item-level paired adapter does not compute population estimates.
    pub population_estimate_computed: bool,
    pub selection_receipt_authenticated: bool,
    pub session_receipt_authenticated: bool,
    pub annotation_records_authenticated: bool,
    pub synthetic_consensus_ground_truth_created: bool,
    pub single_marginal_value_score_created: bool,
}

pub fn audit_annotation_marginal_value_json(
    input: &str,
) -> Result<AnnotationMarginalValueReportV1, String> {
    let case: AnnotationMarginalValueInputV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid annotation marginal-value JSON: {error}"))?;
    audit_annotation_marginal_value(&case)
}

pub fn audit_annotation_marginal_value(
    input: &AnnotationMarginalValueInputV1,
) -> Result<AnnotationMarginalValueReportV1, String> {
    validate_input(input)?;

    let before_report = audit_multi_annotator_case(&input.before_panel)
        .map_err(|error| format!("before annotation panel failed validation: {error}"))?;
    let after_report = audit_multi_annotator_case(&input.after_panel)
        .map_err(|error| format!("after annotation panel failed validation: {error}"))?;

    let before = snapshot(&before_report, &input.decision_profile)?;
    let after = snapshot(&after_report, &input.decision_profile)?;

    let disposition_entropy_delta_bits = delta(
        before.disposition_entropy_bits,
        after.disposition_entropy_bits,
    );
    let occurrence_count_entropy_delta_bits = delta(
        before.present_occurrence_count_entropy_bits,
        after.present_occurrence_count_entropy_bits,
    );
    let pairwise_present_event_f1_delta = delta(
        before.pairwise_present_symmetric_event_f1,
        after.pairwise_present_symmetric_event_f1,
    );
    let pairwise_present_matched_iou_delta = delta(
        before.pairwise_present_mean_matched_iou,
        after.pairwise_present_mean_matched_iou,
    );
    let pairwise_present_count_agreement_delta = delta(
        before.pairwise_present_exact_count_agreement,
        after.pairwise_present_exact_count_agreement,
    );
    let maximum_event_boundary_spread_delta_ms = delta(
        before.maximum_event_boundary_spread_ms,
        after.maximum_event_boundary_spread_ms,
    );
    let model_fully_correct_rate_delta_against_panel = delta(
        before.model_fully_correct_rate_against_panel,
        after.model_fully_correct_rate_against_panel,
    );

    let active_listening_minutes = input.effort.active_listening_seconds as f64 / 60.0;
    let active_review_minutes = input.effort.active_review_seconds as f64 / 60.0;
    let elapsed_wall_minutes = input.effort.elapsed_wall_seconds as f64 / 60.0;

    let per_minute_deltas = PerMinuteMetricDeltaV1 {
        disposition_entropy_bits_per_active_review_minute: per_minute(
            disposition_entropy_delta_bits,
            active_review_minutes,
        ),
        occurrence_count_entropy_bits_per_active_review_minute: per_minute(
            occurrence_count_entropy_delta_bits,
            active_review_minutes,
        ),
        pairwise_present_event_f1_per_active_review_minute: per_minute(
            pairwise_present_event_f1_delta,
            active_review_minutes,
        ),
        pairwise_present_matched_iou_per_active_review_minute: per_minute(
            pairwise_present_matched_iou_delta,
            active_review_minutes,
        ),
        pairwise_count_agreement_per_active_review_minute: per_minute(
            pairwise_present_count_agreement_delta,
            active_review_minutes,
        ),
        maximum_boundary_spread_ms_per_active_review_minute: per_minute(
            maximum_event_boundary_spread_delta_ms,
            active_review_minutes,
        ),
    };

    let new_annotation = input
        .after_panel
        .annotations
        .last()
        .ok_or_else(|| "after panel must retain the newly added annotation".to_string())?;

    Ok(AnnotationMarginalValueReportV1 {
        schema: REPORT_SCHEMA_V1,
        evaluator_version: EVALUATOR_VERSION_V1,
        source_work_id_sha256: input.source_work_id_sha256.clone(),
        query_id: input.before_panel.query_id.clone(),
        audio_sha256: input.before_panel.artifact.audio_sha256.clone(),
        selection_stream: input.selection_stream,
        selection_receipt_sha256: input.selection_receipt_sha256.clone(),
        before_annotation_manifest_sha256: input.before_panel.annotation_manifest_sha256.clone(),
        after_annotation_manifest_sha256: input.after_panel.annotation_manifest_sha256.clone(),
        newly_added_annotator_id_sha256: new_annotation.annotator_id_sha256.clone(),
        newly_added_annotation_sha256: new_annotation.annotation_sha256.clone(),
        before,
        after,
        disposition_entropy_delta_bits,
        occurrence_count_entropy_delta_bits,
        pairwise_present_event_f1_delta,
        pairwise_present_matched_iou_delta,
        pairwise_present_count_agreement_delta,
        maximum_event_boundary_spread_delta_ms,
        decision_transition: decision_transition(
            before_report.summary.panel_size,
            after_report.summary.panel_size,
            &input.decision_profile,
        ),
        effort: input.effort.clone(),
        active_listening_minutes,
        active_review_minutes,
        elapsed_wall_minutes,
        per_minute_deltas,
        model_fully_correct_rate_delta_against_panel,
        annotation_record_count_delta: 1,
        active_stream_is_probability_sample: false,
        population_estimate_computed: false,
        selection_receipt_authenticated: false,
        session_receipt_authenticated: false,
        annotation_records_authenticated: false,
        synthetic_consensus_ground_truth_created: false,
        single_marginal_value_score_created: false,
    })
}

fn validate_input(input: &AnnotationMarginalValueInputV1) -> Result<(), String> {
    if input.schema != INPUT_SCHEMA_V1 {
        return Err(format!("unsupported annotation marginal-value schema: {}", input.schema));
    }
    validate_digest(&input.source_work_id_sha256, "source_work_id_sha256")?;
    validate_digest(&input.selection_receipt_sha256, "selection_receipt_sha256")?;
    require_text(&input.decision_profile.profile_id, "decision_profile.profile_id")?;
    validate_decision_profile(&input.decision_profile)?;
    validate_effort(&input.effort)?;

    let before = &input.before_panel;
    let after = &input.after_panel;
    if before.query_id != after.query_id
        || before.query_text != after.query_text
        || before.artifact != after.artifact
        || before.annotation_policy_version != after.annotation_policy_version
        || before.minimum_annotator_count != after.minimum_annotator_count
        || before.plan != after.plan
        || before.prediction != after.prediction
    {
        return Err("before/after panels differ in query, audio, policy, minimum panel, scorer plan or model prediction".into());
    }
    if before.annotation_manifest_sha256 == after.annotation_manifest_sha256 {
        return Err("before/after annotation manifests must have distinct digests".into());
    }
    if after.annotations.len() != before.annotations.len().saturating_add(1) {
        return Err("after panel must contain exactly one more annotation than before panel".into());
    }
    if after.annotations[..before.annotations.len()] != before.annotations[..] {
        return Err("after panel rewrites, reorders or removes an existing annotation record".into());
    }

    let new_annotation = after.annotations.last().unwrap();
    if before
        .annotations
        .iter()
        .any(|annotation| annotation.annotator_id_sha256 == new_annotation.annotator_id_sha256)
    {
        return Err("new annotation must come from a distinct independent annotator".into());
    }
    if before
        .annotations
        .iter()
        .any(|annotation| annotation.annotation_sha256 == new_annotation.annotation_sha256)
    {
        return Err("new annotation record digest duplicates a previous record".into());
    }
    Ok(())
}

fn validate_decision_profile(profile: &FrozenPanelDecisionProfileV1) -> Result<(), String> {
    if profile.minimum_eligible_annotators < 2 {
        return Err("decision profile requires at least two eligible annotators".into());
    }
    for (label, value) in [
        ("minimum_present_support_ppm", profile.minimum_present_support_ppm),
        ("minimum_absent_support_ppm", profile.minimum_absent_support_ppm),
        (
            "maximum_ambiguous_not_estimable_share_ppm",
            profile.maximum_ambiguous_not_estimable_share_ppm,
        ),
        (
            "minimum_pairwise_present_event_f1_ppm",
            profile.minimum_pairwise_present_event_f1_ppm,
        ),
        (
            "minimum_pairwise_present_count_agreement_ppm",
            profile.minimum_pairwise_present_count_agreement_ppm,
        ),
    ] {
        if value > 1_000_000 {
            return Err(format!("decision profile {label} exceeds one million ppm"));
        }
    }
    Ok(())
}

fn validate_effort(effort: &AnnotationEffortReceiptV1) -> Result<(), String> {
    validate_digest(&effort.session_receipt_sha256, "session_receipt_sha256")?;
    if effort.active_listening_seconds > effort.active_review_seconds
        || effort.active_review_seconds > effort.elapsed_wall_seconds
    {
        return Err("effort durations must satisfy listening <= review <= elapsed wall time".into());
    }
    if effort.self_reported_difficulty_ppm.is_some_and(|value| value > 1_000_000) {
        return Err("self_reported_difficulty_ppm must be in [0, 1,000,000]".into());
    }
    Ok(())
}

fn snapshot(
    report: &MultiAnnotatorGroundingReportV1,
    profile: &FrozenPanelDecisionProfileV1,
) -> Result<PanelInformationSnapshotV1, String> {
    let states: Vec<_> = report.annotator_results.iter().map(|item| item.state).collect();
    let state_entropy = categorical_entropy(states.iter().map(|state| match state {
        crate::temporal_grounding::GroundTruthStateV1::Present => 0,
        crate::temporal_grounding::GroundTruthStateV1::Absent => 1,
        crate::temporal_grounding::GroundTruthStateV1::Ambiguous => 2,
        crate::temporal_grounding::GroundTruthStateV1::NotEstimable => 3,
    }));

    let present_counts: Vec<_> = report
        .annotator_results
        .iter()
        .filter(|item| item.state == crate::temporal_grounding::GroundTruthStateV1::Present)
        .map(|item| item.intervals.len())
        .collect();
    let count_entropy = categorical_entropy(present_counts.iter().copied());

    let summary = &report.summary;
    let boundary_spread = report
        .event_reference_boundary_distributions
        .iter()
        .map(|item| item.start_spread_frames.max(item.end_spread_frames))
        .max()
        .map(|frames| frames as f64 * 1_000.0 / report.prediction.artifact.sample_rate_hz as f64);

    let decision = decide_panel_state(report, profile, boundary_spread);

    Ok(PanelInformationSnapshotV1 {
        annotations: summary.panel_size,
        eligible_present_absent_annotations: summary.present_annotators + summary.absent_annotators,
        present_annotations: summary.present_annotators,
        absent_annotations: summary.absent_annotators,
        ambiguous_annotations: summary.ambiguous_exclusions,
        not_estimable_annotations: summary.not_estimable_exclusions,
        disposition_entropy_bits: state_entropy,
        present_occurrence_count_entropy_bits: count_entropy,
        pairwise_disposition_agreement_rate: summary.pairwise_disposition_agreement_rate,
        pairwise_present_symmetric_event_f1: summary.pairwise_present_symmetric_event_f1,
        pairwise_present_mean_matched_iou: summary.pairwise_present_mean_matched_iou,
        pairwise_present_exact_count_agreement: summary.pairwise_present_exact_count_agreement,
        maximum_event_boundary_spread_ms: boundary_spread,
        model_fully_correct_rate_against_panel: summary.fully_correct_rate,
        frozen_decision_state: decision,
    })
}

fn decide_panel_state(
    report: &MultiAnnotatorGroundingReportV1,
    profile: &FrozenPanelDecisionProfileV1,
    boundary_spread_ms: Option<f64>,
) -> PanelDecisionStateV1 {
    let summary = &report.summary;
    let eligible = summary.present_annotators + summary.absent_annotators;
    if eligible < profile.minimum_eligible_annotators as usize || report.summary.panel_size == 0 {
        return PanelDecisionStateV1::NeedsReview;
    }
    let ambiguous_share = (summary.ambiguous_exclusions + summary.not_estimable_exclusions) as f64
        / summary.panel_size as f64;
    if ambiguous_share > profile.maximum_ambiguous_not_estimable_share_ppm as f64 / PPM_SCALE {
        return PanelDecisionStateV1::NeedsReview;
    }

    let present_support = summary.present_annotators as f64 / eligible as f64;
    let absent_support = summary.absent_annotators as f64 / eligible as f64;

    if summary.present_annotators > 0
        && present_support >= profile.minimum_present_support_ppm as f64 / PPM_SCALE
        && summary.pairwise_present_symmetric_event_f1.is_some_and(|value| {
            value >= profile.minimum_pairwise_present_event_f1_ppm as f64 / PPM_SCALE
        })
        && summary.pairwise_present_exact_count_agreement.is_some_and(|value| {
            value >= profile.minimum_pairwise_present_count_agreement_ppm as f64 / PPM_SCALE
        })
        && boundary_spread_ms.is_some_and(|value| value <= profile.maximum_event_boundary_spread_ms as f64)
    {
        return PanelDecisionStateV1::SufficientlySupportedPresent;
    }
    if summary.present_annotators == 0
        && absent_support >= profile.minimum_absent_support_ppm as f64 / PPM_SCALE
    {
        return PanelDecisionStateV1::SufficientlySupportedAbsent;
    }
    PanelDecisionStateV1::NeedsReview
}

fn categorical_entropy(values: impl Iterator<Item = usize>) -> Option<f64> {
    let mut counts = std::collections::BTreeMap::<usize, usize>::new();
    let mut total = 0_usize;
    for value in values {
        *counts.entry(value).or_default() += 1;
        total += 1;
    }
    if total == 0 {
        return None;
    }
    let total_f = total as f64;
    Some(
        counts
            .values()
            .map(|count| {
                let probability = *count as f64 / total_f;
                -probability * probability.log2()
            })
            .sum(),
    )
}

fn decision_transition(before_count: usize, after_count: usize, _profile: &FrozenPanelDecisionProfileV1) -> String {
    format!("panel_size:{before_count}->{after_count}")
}

fn delta(before: Option<f64>, after: Option<f64>) -> Option<f64> {
    before.zip(after).map(|(before, after)| after - before)
}

fn per_minute(delta: Option<f64>, active_review_minutes: f64) -> Option<f64> {
    delta.zip((active_review_minutes > 0.0).then_some(active_review_minutes))
        .map(|(delta, minutes)| delta / minutes)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_consensus::IndependentTemporalAnnotationV1;
    use crate::temporal_grounding::{
        AudioArtifactClockV1, FrameIntervalV1, GroundTruthStateV1, GroundingPredictionStateV1,
        TemporalGroundingPlanV1, TemporalGroundingPredictionV1, TEMPORAL_GROUNDING_EVALUATOR_V1,
        TEMPORAL_GROUNDING_SCHEMA_V1,
    };

    const SOURCE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const WORK: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const SELECTION: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";

    fn digest(ch: char) -> String { ch.to_string().repeat(64) }

    fn artifact() -> AudioArtifactClockV1 {
        AudioArtifactClockV1 {
            audio_sha256: SOURCE.into(),
            sample_rate_hz: 48_000,
            channels: 2,
            total_frames: 48000 * 60,
        }
    }

    fn plan() -> TemporalGroundingPlanV1 {
        TemporalGroundingPlanV1 {
            evaluator_version: TEMPORAL_GROUNDING_EVALUATOR_V1.into(),
            minimum_interval_iou: 0.5,
        }
    }

    fn annotation(id: char, record: char, state: crate::temporal_grounding::GroundTruthStateV1) -> IndependentTemporalAnnotationV1 {
        IndependentTemporalAnnotationV1 {
            annotator_id_sha256: digest(id),
            annotation_sha256: digest(record),
            state,
            intervals: if state == GroundTruthStateV1::Present {
                vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }]
            } else { Vec::new() },
            status_reason: None,
        }
    }

    fn panel(records: Vec<IndependentTemporalAnnotationV1>, manifest: char) -> MultiAnnotatorTemporalCaseV1 {
        let art = artifact();
        MultiAnnotatorTemporalCaseV1 {
            schema: "mel-multi-annotator-temporal-grounding-v1".into(),
            query_id: "query-1".into(),
            query_text: "Every piano entrance".into(),
            artifact: art.clone(),
            annotation_policy_version: "policy-v1".into(),
            annotation_manifest_sha256: digest(manifest),
            minimum_annotator_count: 2,
            plan: plan(),
            prediction: TemporalGroundingPredictionV1 {
                schema_version: TEMPORAL_GROUNDING_SCHEMA_V1.into(),
                query_id: "query-1".into(),
                artifact: art,
                state: GroundingPredictionStateV1::Occurrences {
                    intervals: vec![FrameIntervalV1 { start_frame: 100, end_frame: 200 }],
                },
            },
            annotations: records,
        }
    }

    fn input(before: MultiAnnotatorTemporalCaseV1, after: MultiAnnotatorTemporalCaseV1, review_seconds: u64) -> AnnotationMarginalValueInputV1 {
        AnnotationMarginalValueInputV1 {
            schema: INPUT_SCHEMA_V1.into(),
            source_work_id_sha256: WORK.into(),
            selection_stream: AnnotationSelectionStreamV1::ActiveRefinement,
            selection_receipt_sha256: SELECTION.into(),
            decision_profile: FrozenPanelDecisionProfileV1 {
                profile_id: "profile-v1".into(),
                minimum_eligible_annotators: 4,
                minimum_present_support_ppm: 900_000,
                minimum_absent_support_ppm: 900_000,
                maximum_ambiguous_not_estimable_share_ppm: 0,
                minimum_pairwise_present_event_f1_ppm: 900_000,
                minimum_pairwise_present_count_agreement_ppm: 900_000,
                maximum_event_boundary_spread_ms: 50,
            },
            before_panel: before,
            after_panel: after,
            effort: AnnotationEffortReceiptV1 {
                session_receipt_sha256: digest('d'),
                active_listening_seconds: review_seconds / 2,
                active_review_seconds: review_seconds,
                elapsed_wall_seconds: review_seconds + 30,
                replay_count: 2,
                seek_count: 1,
                self_reported_difficulty_ppm: Some(300_000),
            },
        }
    }

    #[test]
    fn aligned_additional_annotation_can_resolve_minimum_panel_gate() {
        let first = vec![
            annotation('1', '1', GroundTruthStateV1::Present),
            annotation('2', '2', GroundTruthStateV1::Present),
            annotation('3', '3', GroundTruthStateV1::Present),
        ];
        let before = panel(first.clone(), 'e');
        let mut after_records = first;
        after_records.push(annotation('4', '4', GroundTruthStateV1::Present));
        let after = panel(after_records, 'f');
        let report = audit_annotation_marginal_value(&input(before, after, 120)).unwrap();
        assert_eq!(report.before.frozen_decision_state, PanelDecisionStateV1::NeedsReview);
        assert_eq!(report.after.frozen_decision_state, PanelDecisionStateV1::SufficientlySupportedPresent);
        assert_eq!(report.annotation_record_count_delta, 1);
        assert_eq!(report.active_review_minutes, 2.0);
        assert_eq!(report.effort.active_listening_seconds, 60);
        assert_eq!(report.disposition_entropy_delta_bits, Some(0.0));
        assert!(!report.single_marginal_value_score_created);
    }

    #[test]
    fn added_ambiguous_annotation_is_information_not_an_absent_label_or_reward_failure() {
        let first = vec![
            annotation('1', '1', GroundTruthStateV1::Present),
            annotation('2', '2', GroundTruthStateV1::Present),
            annotation('3', '3', GroundTruthStateV1::Present),
        ];
        let before = panel(first.clone(), 'e');
        let mut after_records = first;
        after_records.push(annotation('4', '4', GroundTruthStateV1::Ambiguous));
        let after = panel(after_records, 'f');
        let report = audit_annotation_marginal_value(&input(before, after, 180)).unwrap();
        assert!(report.after.disposition_entropy_bits.unwrap() > report.before.disposition_entropy_bits.unwrap());
        assert_eq!(report.after.ambiguous_annotations, 1);
        assert_eq!(report.after.absent_annotations, 0);
        assert_eq!(report.after.frozen_decision_state, PanelDecisionStateV1::NeedsReview);
    }

    #[test]
    fn rewriting_an_existing_annotation_fails_closed() {
        let first = vec![
            annotation('1', '1', GroundTruthStateV1::Present),
            annotation('2', '2', GroundTruthStateV1::Present),
        ];
        let before = panel(first.clone(), 'e');
        let mut after_records = first;
        after_records[0].intervals[0].start_frame = 120;
        after_records.push(annotation('3', '3', GroundTruthStateV1::Present));
        let after = panel(after_records, 'f');
        assert!(audit_annotation_marginal_value(&input(before, after, 120)).is_err());
    }

    #[test]
    fn zero_active_review_time_keeps_per_minute_metrics_undefined() {
        let first = vec![
            annotation('1', '1', GroundTruthStateV1::Present),
            annotation('2', '2', GroundTruthStateV1::Present),
        ];
        let before = panel(first.clone(), 'e');
        let mut after_records = first;
        after_records.push(annotation('3', '3', GroundTruthStateV1::Present));
        let after = panel(after_records, 'f');
        let mut paired = input(before, after, 0);
        paired.effort.active_listening_seconds = 0;
        paired.effort.elapsed_wall_seconds = 30;
        let report = audit_annotation_marginal_value(&paired).unwrap();
        assert_eq!(report.per_minute_deltas.disposition_entropy_bits_per_active_review_minute, None);
    }
}
