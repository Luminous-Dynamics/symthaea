//! Deterministic paired draft-to-final revision audit over the production
//! temporal-grounding scorer. It reuses the canonical one-to-one interval
//! matcher and never computes a second temporal alignment.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::temporal_grounding::{
    score_temporal_grounding, validate_temporal_ground_truth,
    validate_temporal_grounding_prediction, AudioArtifactClockV1, FrameIntervalV1,
    GroundTruthStateV1, GroundingPredictionStateV1, TemporalGroundingOutcomeV1,
    TemporalGroundingPlanV1, TemporalGroundingPredictionV1, TemporalGroundingScoreV1,
    TemporalGroundTruthV1, TEMPORAL_GROUNDING_SCHEMA_V1,
};

const REVISION_SCHEMA: &str = "mel-temporal-revision-case-v1";
const EPSILON: f64 = 1e-12;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceConditionV1 {
    NoEvidence,
    LocalRetrieved,
    YokedRetrieved,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RevisionActionV1 {
    NoChange,
    AddOccurrence,
    RemoveOccurrence,
    SplitMergedOccurrence,
    MergeDuplicateOccurrence,
    AdjustStartBoundary,
    AdjustEndBoundary,
    Other,
}

/// Self-contained V2 input: the evaluator receives raw versioned scorer
/// inputs and computes draft/final scores itself. Callers cannot inject or
/// forge derived metric summaries into the revision comparison.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevisionCaseV2 {
    pub schema: String,
    pub case_id: String,
    /// Shared stable identifier for the same query/model trial across evidence arms.
    pub experiment_unit_id: String,
    /// Stable source-work lineage identifier for held-out/source-disjoint analysis.
    pub source_lineage_id: String,
    pub model_identity_sha256: String,
    pub prompt_sha256: String,
    pub adapter_sha256: String,
    /// Immutable identity of the executable/runtime/container/weights environment.
    pub runtime_identity_sha256: String,
    pub decoding_config_sha256: String,
    pub max_output_tokens: u32,
    pub actual_output_tokens: u32,
    pub max_wall_time_ms: u64,
    pub elapsed_wall_time_ms: u64,
    pub evidence_condition: EvidenceConditionV1,
    pub evidence: Option<RevisionEvidenceV1>,
    pub actions: Vec<RevisionActionV1>,
    pub truth: TemporalGroundTruthV1,
    pub plan: TemporalGroundingPlanV1,
    pub draft_prediction: TemporalGroundingPredictionV1,
    pub final_prediction: TemporalGroundingPredictionV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevisionEvidenceV1 {
    pub source_audio_sha256: String,
    pub source_sample_rate_hz: u32,
    pub evidence_payload_sha256: String,
    pub retrieval_config_sha256: String,
    pub start_frame: u64,
    pub end_frame: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RevisionCaseV1 {
    pub schema: String,
    pub case_id: String,
    pub experiment_unit_id: String,
    pub model_identity_sha256: String,
    pub prompt_sha256: String,
    pub adapter_sha256: String,
    pub evidence_condition: EvidenceConditionV1,
    pub evidence: Option<RevisionEvidenceV1>,
    pub actions: Vec<RevisionActionV1>,
    pub draft: TemporalGroundingScoreV1,
    pub final_score: TemporalGroundingScoreV1,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RevisionScoreSnapshotV1 {
    pub predicted_occurrences: usize,
    pub ground_truth_occurrences: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub f1: Option<f64>,
    pub exact_occurrence_count: Option<bool>,
    pub count_gated_f1: Option<f64>,
    pub correct_absence_claim: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RevisionReportV1 {
    pub case_id: String,
    pub experiment_unit_id: String,
    pub source_lineage_id: Option<String>,
    pub runtime_identity_sha256: Option<String>,
    pub decoding_config_sha256: Option<String>,
    pub max_output_tokens: Option<u32>,
    pub actual_output_tokens: Option<u32>,
    pub max_wall_time_ms: Option<u64>,
    pub elapsed_wall_time_ms: Option<u64>,
    pub evaluator_version: String,
    pub query_id: String,
    pub audio_sha256: String,
    pub evidence_condition: EvidenceConditionV1,
    pub actions: Vec<RevisionActionV1>,
    pub draft: RevisionScoreSnapshotV1,
    pub final_score: RevisionScoreSnapshotV1,
    pub f1_delta: Option<f64>,
    pub count_gated_f1_delta: Option<f64>,
    pub helpful_revision: Option<bool>,
    pub harmful_revision: Option<bool>,
    pub draft_was_perfect: bool,
    pub preservation_violation: bool,
    pub absence_improved: Option<bool>,
    pub absence_regressed: Option<bool>,
}

/// Paired evidence-ablation result for one frozen query/model unit.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MatchedEvidenceContrastV1 {
    pub experiment_unit_id: String,
    pub no_evidence_case_id: String,
    pub local_evidence_case_id: String,
    pub yoked_evidence_case_id: String,
    pub local_minus_no_evidence_f1: Option<f64>,
    pub local_minus_yoked_f1: Option<f64>,
    pub local_minus_no_evidence_count_gated_f1: Option<f64>,
    pub local_minus_yoked_count_gated_f1: Option<f64>,
    pub local_revision_gain: Option<f64>,
    pub yoked_revision_gain: Option<f64>,
    pub local_output_tokens: u32,
    pub no_evidence_output_tokens: u32,
    pub yoked_output_tokens: u32,
    pub local_elapsed_wall_time_ms: u64,
    pub no_evidence_elapsed_wall_time_ms: u64,
    pub yoked_elapsed_wall_time_ms: u64,
    pub local_beats_yoked: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RevisionAggregateV1 {
    pub evaluator_version: String,
    pub cases: usize,
    pub positive_cases: usize,
    pub absent_cases: usize,
    pub excluded_cases: usize,
    pub helpful_revisions: usize,
    pub harmful_revisions: usize,
    pub unchanged_positive_revisions: usize,
    pub preservation_violations: usize,
    pub mean_draft_f1: Option<f64>,
    pub mean_final_f1: Option<f64>,
    pub mean_f1_delta: Option<f64>,
    pub one_to_many_cases: usize,
    pub mean_draft_count_gated_f1: Option<f64>,
    pub mean_final_count_gated_f1: Option<f64>,
    pub mean_count_gated_f1_delta: Option<f64>,
    pub absent_claims_scored: usize,
    pub correct_absence_before: usize,
    pub correct_absence_after: usize,
    pub absence_improvements: usize,
    pub absence_regressions: usize,
}

/// Compare one no-evidence, one local-evidence and one yoked-evidence
/// case per experiment unit. The three cases must share the same raw draft,
/// frozen truth, scoring plan, model, prompt and adapter. This is the key
/// control for attributing incremental value to retrieval rather than to a
/// changed draft or a different inference configuration.
pub fn compare_matched_evidence_arms(
    cases: &[RevisionCaseV2],
) -> Result<Vec<MatchedEvidenceContrastV1>, String> {
    let mut grouped: BTreeMap<&str, Vec<&RevisionCaseV2>> = BTreeMap::new();
    for case in cases {
        require_text(&case.experiment_unit_id, "experiment_unit_id")?;
        grouped
            .entry(case.experiment_unit_id.as_str())
            .or_default()
            .push(case);
    }

    let mut results = Vec::with_capacity(grouped.len());
    for (unit_id, group) in grouped {
        if group.len() != 3 {
            return Err(format!(
                "experiment unit {unit_id} must have exactly three evidence arms"
            ));
        }
        let mut no_evidence = None;
        let mut local = None;
        let mut yoked = None;
        let mut case_ids = BTreeSet::new();
        for case in &group {
            if !case_ids.insert(case.case_id.as_str()) {
                return Err(format!("experiment unit {unit_id} repeats a case_id"));
            }
            let slot = match case.evidence_condition {
                EvidenceConditionV1::NoEvidence => &mut no_evidence,
                EvidenceConditionV1::LocalRetrieved => &mut local,
                EvidenceConditionV1::YokedRetrieved => &mut yoked,
            };
            if slot.replace(*case).is_some() {
                return Err(format!("experiment unit {unit_id} repeats an evidence condition"));
            }
        }
        let no_evidence = no_evidence
            .ok_or_else(|| format!("experiment unit {unit_id} lacks the no-evidence arm"))?;
        let local = local
            .ok_or_else(|| format!("experiment unit {unit_id} lacks the local-evidence arm"))?;
        let yoked = yoked
            .ok_or_else(|| format!("experiment unit {unit_id} lacks the yoked-evidence arm"))?;

        for arm in [local, yoked] {
            if arm.experiment_unit_id != no_evidence.experiment_unit_id
                || arm.truth != no_evidence.truth
                || arm.plan != no_evidence.plan
                || arm.draft_prediction != no_evidence.draft_prediction
                || arm.model_identity_sha256 != no_evidence.model_identity_sha256
                || arm.prompt_sha256 != no_evidence.prompt_sha256
                || arm.adapter_sha256 != no_evidence.adapter_sha256
                || arm.source_lineage_id != no_evidence.source_lineage_id
                || arm.runtime_identity_sha256 != no_evidence.runtime_identity_sha256
                || arm.decoding_config_sha256 != no_evidence.decoding_config_sha256
                || arm.max_output_tokens != no_evidence.max_output_tokens
                || arm.max_wall_time_ms != no_evidence.max_wall_time_ms
            {
                return Err(format!(
                    "experiment unit {unit_id} is not a matched evidence-ablation triplet"
                ));
            }
        }

        let no_report = audit_revision_v2(no_evidence)?;
        let local_report = audit_revision_v2(local)?;
        let yoked_report = audit_revision_v2(yoked)?;
        let local_minus_no_evidence_f1 = paired_delta(
            local_report.final_score.f1,
            no_report.final_score.f1,
        );
        let local_minus_yoked_f1 =
            paired_delta(local_report.final_score.f1, yoked_report.final_score.f1);
        let local_minus_no_evidence_count_gated_f1 = paired_delta(
            local_report.final_score.count_gated_f1,
            no_report.final_score.count_gated_f1,
        );
        let local_minus_yoked_count_gated_f1 = paired_delta(
            local_report.final_score.count_gated_f1,
            yoked_report.final_score.count_gated_f1,
        );
        let local_revision_gain = local_report.f1_delta;
        let yoked_revision_gain = yoked_report.f1_delta;
        let local_beats_yoked = match (
            local_revision_gain,
            yoked_revision_gain,
            local_minus_yoked_f1,
        ) {
            (Some(local_gain), Some(yoked_gain), Some(final_delta)) => {
                Some(local_gain > yoked_gain + EPSILON && final_delta > EPSILON)
            }
            _ => None,
        };

        results.push(MatchedEvidenceContrastV1 {
            experiment_unit_id: unit_id.to_owned(),
            no_evidence_case_id: no_evidence.case_id.clone(),
            local_evidence_case_id: local.case_id.clone(),
            yoked_evidence_case_id: yoked.case_id.clone(),
            local_minus_no_evidence_f1,
            local_minus_yoked_f1,
            local_minus_no_evidence_count_gated_f1,
            local_minus_yoked_count_gated_f1,
            local_revision_gain,
            yoked_revision_gain,
            local_output_tokens: local.actual_output_tokens,
            no_evidence_output_tokens: no_evidence.actual_output_tokens,
            yoked_output_tokens: yoked.actual_output_tokens,
            local_elapsed_wall_time_ms: local.elapsed_wall_time_ms,
            no_evidence_elapsed_wall_time_ms: no_evidence.elapsed_wall_time_ms,
            yoked_elapsed_wall_time_ms: yoked.elapsed_wall_time_ms,
            local_beats_yoked,
        });
    }
    Ok(results)
}

fn paired_delta(left: Option<f64>, right: Option<f64>) -> Option<f64> {
    left.zip(right).map(|(left, right)| left - right)
}

/// Parse and score a self-contained revision case using the exact
/// production temporal-grounding scorer.
pub fn audit_revision_case_v2(input: &str) -> Result<RevisionReportV1, String> {
    let case: RevisionCaseV2 = serde_json::from_str(input)
        .map_err(|error| format!("invalid temporal revision V2 case JSON: {error}"))?;
    audit_revision_v2(&case)
}

/// Unlike V1's summary-input API, V2 never accepts caller-supplied derived
/// score summaries. Both predictions are scored against one frozen truth,
/// exact audio clock and evaluator plan.
pub fn audit_revision_v2(case: &RevisionCaseV2) -> Result<RevisionReportV1, String> {
    if case.schema != "mel-temporal-revision-case-v2" {
        return Err(format!("unsupported temporal revision V2 schema: {}", case.schema));
    }
    require_text(&case.case_id, "case_id")?;
    require_text(&case.experiment_unit_id, "experiment_unit_id")?;
    require_text(&case.source_lineage_id, "source_lineage_id")?;
    validate_digest(&case.model_identity_sha256, "model_identity_sha256")?;
    validate_digest(&case.prompt_sha256, "prompt_sha256")?;
    validate_digest(&case.adapter_sha256, "adapter_sha256")?;
    validate_digest(&case.runtime_identity_sha256, "runtime_identity_sha256")?;
    validate_digest(&case.decoding_config_sha256, "decoding_config_sha256")?;
    validate_budget(case)?;
    validate_actions(&case.actions)?;

    let truth_issues = validate_temporal_ground_truth(&case.truth);
    if !truth_issues.is_empty() {
        return Err(format!("invalid revision ground truth: {truth_issues:?}"));
    }
    for (label, prediction) in [
        ("draft_prediction", &case.draft_prediction),
        ("final_prediction", &case.final_prediction),
    ] {
        let issues = validate_temporal_grounding_prediction(prediction);
        if !issues.is_empty() {
            return Err(format!("invalid {label}: {issues:?}"));
        }
    }
    if case.actions.as_slice() == [RevisionActionV1::NoChange]
        && case.draft_prediction != case.final_prediction
    {
        return Err("NoChange action is incompatible with different draft/final predictions".into());
    }

    let draft_score = score_temporal_grounding(
        &case.truth,
        &case.draft_prediction,
        &case.plan,
    )
    .map_err(|issues| format!("draft score rejected by production evaluator: {issues:?}"))?;
    let final_score = score_temporal_grounding(
        &case.truth,
        &case.final_prediction,
        &case.plan,
    )
    .map_err(|issues| format!("final score rejected by production evaluator: {issues:?}"))?;
    let legacy_case = RevisionCaseV1 {
        schema: REVISION_SCHEMA.into(),
        case_id: case.case_id.clone(),
        experiment_unit_id: case.experiment_unit_id.clone(),
        model_identity_sha256: case.model_identity_sha256.clone(),
        prompt_sha256: case.prompt_sha256.clone(),
        adapter_sha256: case.adapter_sha256.clone(),
        evidence_condition: case.evidence_condition,
        evidence: case.evidence.clone(),
        actions: case.actions.clone(),
        draft: draft_score,
        final_score,
    };
    validate_v2_evidence(case)?;
    let mut report = audit_revision(&legacy_case)?;
    report.source_lineage_id = Some(case.source_lineage_id.clone());
    report.runtime_identity_sha256 = Some(case.runtime_identity_sha256.clone());
    report.decoding_config_sha256 = Some(case.decoding_config_sha256.clone());
    report.max_output_tokens = Some(case.max_output_tokens);
    report.actual_output_tokens = Some(case.actual_output_tokens);
    report.max_wall_time_ms = Some(case.max_wall_time_ms);
    report.elapsed_wall_time_ms = Some(case.elapsed_wall_time_ms);
    Ok(report)
}

fn validate_v2_evidence(case: &RevisionCaseV2) -> Result<(), String> {
    let scored_audio = &case.truth.artifact;
    match (case.evidence_condition, &case.evidence) {
        (EvidenceConditionV1::NoEvidence, None) => Ok(()),
        (EvidenceConditionV1::NoEvidence, Some(_)) => {
            Err("NoEvidence condition must not carry an evidence reference".into())
        }
        (_, None) => Err("retrieved/yoked evidence condition requires an evidence reference".into()),
        (condition, Some(evidence)) => {
            validate_digest(&evidence.source_audio_sha256, "evidence.source_audio_sha256")?;
            validate_digest(
                &evidence.evidence_payload_sha256,
                "evidence.evidence_payload_sha256",
            )?;
            validate_digest(
                &evidence.retrieval_config_sha256,
                "evidence.retrieval_config_sha256",
            )?;
            if evidence.source_sample_rate_hz == 0 || evidence.end_frame <= evidence.start_frame {
                return Err("evidence sample rate and half-open frame window must be valid".into());
            }
            let same_audio_digest = evidence.source_audio_sha256 == scored_audio.audio_sha256;
            let same_audio_clock = same_audio_digest
                && evidence.source_sample_rate_hz == scored_audio.sample_rate_hz;
            match condition {
                EvidenceConditionV1::LocalRetrieved if !same_audio_clock => {
                    Err("local evidence must bind the exact scored audio digest and sample rate".into())
                }
                EvidenceConditionV1::LocalRetrieved
                    if evidence.end_frame > scored_audio.total_frames =>
                {
                    Err("local evidence window exceeds the scored recording".into())
                }
                EvidenceConditionV1::YokedRetrieved if same_audio_digest => {
                    Err("yoked evidence must originate from a different audio artifact".into())
                }
                _ => Ok(()),
            }
        }
    }
}

pub fn audit_revision_case(input: &str) -> Result<RevisionReportV1, String> {
    let case: RevisionCaseV1 = serde_json::from_str(input)
        .map_err(|error| format!("invalid temporal revision case JSON: {error}"))?;
    audit_revision(&case)
}

pub fn audit_revision(case: &RevisionCaseV1) -> Result<RevisionReportV1, String> {
    if case.schema != REVISION_SCHEMA {
        return Err(format!("unsupported temporal revision schema: {}", case.schema));
    }
    require_text(&case.case_id, "case_id")?;
    validate_digest(&case.model_identity_sha256, "model_identity_sha256")?;
    validate_digest(&case.prompt_sha256, "prompt_sha256")?;
    validate_digest(&case.adapter_sha256, "adapter_sha256")?;
    validate_actions(&case.actions)?;
    validate_evidence(case)?;

    validate_score(&case.draft, "draft")?;
    validate_score(&case.final_score, "final_score")?;
    validate_score_pair(&case.draft, &case.final_score)?;
    if case.actions.as_slice() == [RevisionActionV1::NoChange]
        && case.draft != case.final_score
    {
        return Err("NoChange action is incompatible with different draft/final score records".into());
    }

    let draft = snapshot(&case.draft);
    let final_score = snapshot(&case.final_score);
    let positive = case.draft.outcome == TemporalGroundingOutcomeV1::ScoredPresentQuery;
    let absent = case.draft.outcome == TemporalGroundingOutcomeV1::ScoredAbsentQuery;
    let f1_delta = positive.then(|| {
        final_score.f1.unwrap_or(0.0) - draft.f1.unwrap_or(0.0)
    });
    let count_gated_f1_delta = (positive && case.draft.ground_truth_occurrences > 1).then(|| {
        final_score.count_gated_f1.unwrap_or(0.0)
            - draft.count_gated_f1.unwrap_or(0.0)
    });
    let helpful_revision = f1_delta.map(|delta| delta > EPSILON);
    let harmful_revision = f1_delta.map(|delta| delta < -EPSILON);
    let draft_was_perfect = is_perfect(&case.draft);
    let preservation_violation = draft_was_perfect && !is_perfect(&case.final_score);
    let absence_improved = absent.then(|| {
        case.final_score.correct_absence_claim == Some(true)
            && case.draft.correct_absence_claim != Some(true)
    });
    let absence_regressed = absent.then(|| {
        case.draft.correct_absence_claim == Some(true)
            && case.final_score.correct_absence_claim != Some(true)
    });

    Ok(RevisionReportV1 {
        case_id: case.case_id.clone(),
        experiment_unit_id: case.experiment_unit_id.clone(),
        source_lineage_id: None,
        runtime_identity_sha256: None,
        decoding_config_sha256: None,
        max_output_tokens: None,
        actual_output_tokens: None,
        max_wall_time_ms: None,
        elapsed_wall_time_ms: None,
        evaluator_version: case.draft.evaluator_version.clone(),
        query_id: case.draft.query_id.clone(),
        audio_sha256: case.draft.audio_sha256.clone(),
        evidence_condition: case.evidence_condition,
        actions: case.actions.clone(),
        draft,
        final_score,
        f1_delta,
        count_gated_f1_delta,
        helpful_revision,
        harmful_revision,
        draft_was_perfect,
        preservation_violation,
        absence_improved,
        absence_regressed,
    })
}

pub fn aggregate_revision_reports(
    reports: &[RevisionReportV1],
) -> Result<RevisionAggregateV1, String> {
    let evaluator_version = reports
        .first()
        .map(|report| report.evaluator_version.clone())
        .unwrap_or_default();
    let mut case_ids = BTreeSet::new();
    for report in reports {
        require_text(&report.case_id, "report.case_id")?;
        if report.evaluator_version != evaluator_version {
            return Err("revision aggregate mixes temporal-grounding evaluator versions".into());
        }
        if !case_ids.insert(report.case_id.as_str()) {
            return Err(format!("duplicate revision case_id: {}", report.case_id));
        }
    }

    let positive: Vec<_> = reports
        .iter()
        .filter(|report| report.f1_delta.is_some())
        .collect();
    let one_to_many: Vec<_> = reports
        .iter()
        .filter(|report| report.count_gated_f1_delta.is_some())
        .collect();
    let absent: Vec<_> = reports
        .iter()
        .filter(|report| report.absence_improved.is_some())
        .collect();

    let draft_f1: Vec<_> = positive
        .iter()
        .map(|report| report.draft.f1.unwrap_or(0.0))
        .collect();
    let final_f1: Vec<_> = positive
        .iter()
        .map(|report| report.final_score.f1.unwrap_or(0.0))
        .collect();
    let f1_deltas: Vec<_> = positive
        .iter()
        .map(|report| report.f1_delta.unwrap_or(0.0))
        .collect();
    let draft_gated: Vec<_> = one_to_many
        .iter()
        .map(|report| report.draft.count_gated_f1.unwrap_or(0.0))
        .collect();
    let final_gated: Vec<_> = one_to_many
        .iter()
        .map(|report| report.final_score.count_gated_f1.unwrap_or(0.0))
        .collect();
    let gated_deltas: Vec<_> = one_to_many
        .iter()
        .map(|report| report.count_gated_f1_delta.unwrap_or(0.0))
        .collect();

    Ok(RevisionAggregateV1 {
        evaluator_version,
        cases: reports.len(),
        positive_cases: positive.len(),
        absent_cases: absent.len(),
        excluded_cases: reports.len() - positive.len() - absent.len(),
        helpful_revisions: positive
            .iter()
            .filter(|report| report.helpful_revision == Some(true))
            .count(),
        harmful_revisions: positive
            .iter()
            .filter(|report| report.harmful_revision == Some(true))
            .count(),
        unchanged_positive_revisions: positive
            .iter()
            .filter(|report| {
                report.helpful_revision == Some(false) && report.harmful_revision == Some(false)
            })
            .count(),
        preservation_violations: reports
            .iter()
            .filter(|report| report.preservation_violation)
            .count(),
        mean_draft_f1: mean(&draft_f1),
        mean_final_f1: mean(&final_f1),
        mean_f1_delta: mean(&f1_deltas),
        one_to_many_cases: one_to_many.len(),
        mean_draft_count_gated_f1: mean(&draft_gated),
        mean_final_count_gated_f1: mean(&final_gated),
        mean_count_gated_f1_delta: mean(&gated_deltas),
        absent_claims_scored: absent
            .iter()
            .filter(|report| report.draft.correct_absence_claim.is_some()
                || report.final_score.correct_absence_claim.is_some())
            .count(),
        correct_absence_before: absent
            .iter()
            .filter(|report| report.draft.correct_absence_claim == Some(true))
            .count(),
        correct_absence_after: absent
            .iter()
            .filter(|report| report.final_score.correct_absence_claim == Some(true))
            .count(),
        absence_improvements: absent
            .iter()
            .filter(|report| report.absence_improved == Some(true))
            .count(),
        absence_regressions: absent
            .iter()
            .filter(|report| report.absence_regressed == Some(true))
            .count(),
    })
}

fn validate_budget(case: &RevisionCaseV2) -> Result<(), String> {
    if case.max_output_tokens == 0 || case.max_wall_time_ms == 0 {
        return Err("revision case must declare positive token and wall-time budgets".into());
    }
    if case.actual_output_tokens > case.max_output_tokens {
        return Err("actual output tokens exceed the declared maximum".into());
    }
    if case.elapsed_wall_time_ms > case.max_wall_time_ms {
        return Err("elapsed wall time exceeds the declared maximum".into());
    }
    Ok(())
}

fn validate_actions(actions: &[RevisionActionV1]) -> Result<(), String> {
    if actions.is_empty() {
        return Err("revision actions must contain at least one typed action".into());
    }
    if actions.len() > 8 {
        return Err("revision actions exceed the bounded maximum of eight".into());
    }
    if actions.contains(&RevisionActionV1::NoChange) && actions.len() != 1 {
        return Err("NoChange cannot be combined with other revision actions".into());
    }
    let mut seen = BTreeSet::new();
    for action in actions {
        if !seen.insert(*action) {
            return Err("revision actions must not repeat".into());
        }
    }
    Ok(())
}

fn validate_evidence(case: &RevisionCaseV1) -> Result<(), String> {
    match (case.evidence_condition, &case.evidence) {
        (EvidenceConditionV1::NoEvidence, None) => Ok(()),
        (EvidenceConditionV1::NoEvidence, Some(_)) => {
            Err("NoEvidence condition must not carry an evidence reference".into())
        }
        (_, None) => Err("retrieved/yoked evidence condition requires an evidence reference".into()),
        (condition, Some(evidence)) => {
            validate_digest(&evidence.source_audio_sha256, "evidence.source_audio_sha256")?;
            validate_digest(
                &evidence.evidence_payload_sha256,
                "evidence.evidence_payload_sha256",
            )?;
            validate_digest(
                &evidence.retrieval_config_sha256,
                "evidence.retrieval_config_sha256",
            )?;
            if evidence.source_sample_rate_hz == 0 || evidence.end_frame <= evidence.start_frame {
                return Err("evidence sample rate and half-open frame window must be valid".into());
            }
            let same_audio_digest = evidence.source_audio_sha256 == case.draft.audio_sha256;
            let same_audio_clock = same_audio_digest
                && evidence.source_sample_rate_hz == case.draft.sample_rate_hz;
            match condition {
                EvidenceConditionV1::LocalRetrieved if !same_audio_clock => {
                    Err("local evidence must bind the exact scored audio digest and sample rate".into())
                }
                EvidenceConditionV1::YokedRetrieved if same_audio_digest => {
                    Err("yoked evidence must originate from a different audio artifact".into())
                }
                _ => Ok(()),
            }
        }
    }
}

fn validate_score(
    score: &TemporalGroundingScoreV1,
    label: &str,
) -> Result<(), String> {
    require_text(&score.evaluator_version, &format!("{label}.evaluator_version"))?;
    require_text(&score.query_id, &format!("{label}.query_id"))?;
    validate_digest(&score.audio_sha256, &format!("{label}.audio_sha256"))?;
    if score.sample_rate_hz == 0 {
        return Err(format!("{label}.sample_rate_hz must be positive"));
    }
    for (metric, value) in [
        ("precision", score.precision),
        ("recall", score.recall),
        ("f1", score.f1),
        ("matched_interval_iou_mean", score.matched_interval_iou_mean),
    ] {
        if value.is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value)) {
            return Err(format!("{label}.{metric} must be finite and in [0,1]"));
        }
    }
    if score.outcome == TemporalGroundingOutcomeV1::ScoredPresentQuery {
        if score.true_positives.checked_add(score.false_positives)
            != Some(score.predicted_occurrences)
        {
            return Err(format!("{label} prediction count does not equal TP + FP"));
        }
        if score.true_positives.checked_add(score.false_negatives)
            != Some(score.ground_truth_occurrences)
        {
            return Err(format!("{label} truth count does not equal TP + FN"));
        }
        if score.count_absolute_error
            != Some(score.predicted_occurrences.abs_diff(score.ground_truth_occurrences))
            || score.exact_occurrence_count
                != Some(score.predicted_occurrences == score.ground_truth_occurrences)
        {
            return Err(format!("{label} occurrence-count diagnostics are inconsistent"));
        }
        if score.matches.len() != score.true_positives {
            return Err(format!("{label} one-to-one match count differs from true_positives"));
        }
        let mut prediction_indices = BTreeSet::new();
        let mut annotation_indices = BTreeSet::new();
        for matched in &score.matches {
            if matched.prediction_index >= score.predicted_occurrences
                || matched.annotation_index >= score.ground_truth_occurrences
                || !prediction_indices.insert(matched.prediction_index)
                || !annotation_indices.insert(matched.annotation_index)
                || !matched.interval_iou.is_finite()
                || !(0.0..=1.0).contains(&matched.interval_iou)
            {
                return Err(format!("{label} match vector is not a valid one-to-one interval assignment"));
            }
        }
    }
    if score.outcome == TemporalGroundingOutcomeV1::ScoredAbsentQuery
        && (score.ground_truth_occurrences != 0
            || score.predicted_occurrences != 0
            || score.true_positives != 0
            || score.false_positives != 0
            || score.false_negatives != 0
            || !score.matches.is_empty())
    {
        return Err(format!("{label} absent-query score carries positive occurrence counts"));
    }
    match score.outcome {
        TemporalGroundingOutcomeV1::ScoredPresentQuery if !score.scored || score.f1.is_none() => {
            Err(format!("{label} present query must be scored with F1"))
        }
        TemporalGroundingOutcomeV1::ScoredAbsentQuery if !score.scored => {
            Err(format!("{label} absent query must be scored"))
        }
        TemporalGroundingOutcomeV1::ExcludedAmbiguousGroundTruth
        | TemporalGroundingOutcomeV1::ExcludedNotEstimableGroundTruth
            if score.scored || score.f1.is_some() =>
        {
            Err(format!("{label} excluded truth must not be scored"))
        }
        _ => Ok(()),
    }
}

fn validate_score_pair(
    draft: &TemporalGroundingScoreV1,
    final_score: &TemporalGroundingScoreV1,
) -> Result<(), String> {
    if draft.evaluator_version != final_score.evaluator_version {
        return Err("draft/final scores use different evaluator versions".into());
    }
    if draft.query_id != final_score.query_id {
        return Err("draft/final scores reference different query IDs".into());
    }
    if draft.audio_sha256 != final_score.audio_sha256 {
        return Err("draft/final scores reference different audio artifacts".into());
    }
    if draft.sample_rate_hz != final_score.sample_rate_hz {
        return Err("draft/final scores use different sample rates".into());
    }
    if draft.outcome != final_score.outcome
        || draft.scored != final_score.scored
        || draft.ground_truth_occurrences != final_score.ground_truth_occurrences
    {
        return Err("draft/final score cohort or ground-truth cardinality changed".into());
    }
    Ok(())
}

fn snapshot(score: &TemporalGroundingScoreV1) -> RevisionScoreSnapshotV1 {
    let count_gated_f1 = if score.outcome == TemporalGroundingOutcomeV1::ScoredPresentQuery
        && score.ground_truth_occurrences > 1
    {
        Some(if score.exact_occurrence_count == Some(true) {
            score.f1.unwrap_or(0.0)
        } else {
            0.0
        })
    } else {
        None
    };
    RevisionScoreSnapshotV1 {
        predicted_occurrences: score.predicted_occurrences,
        ground_truth_occurrences: score.ground_truth_occurrences,
        true_positives: score.true_positives,
        false_positives: score.false_positives,
        false_negatives: score.false_negatives,
        f1: score.f1,
        exact_occurrence_count: score.exact_occurrence_count,
        count_gated_f1,
        correct_absence_claim: score.correct_absence_claim,
    }
}

fn is_perfect(score: &TemporalGroundingScoreV1) -> bool {
    match score.outcome {
        TemporalGroundingOutcomeV1::ScoredPresentQuery => {
            score.f1 == Some(1.0)
                && score.false_positives == 0
                && score.false_negatives == 0
                && score.exact_occurrence_count == Some(true)
                && !score.parse_failure
                && !score.abstained
        }
        TemporalGroundingOutcomeV1::ScoredAbsentQuery => {
            score.correct_absence_claim == Some(true) && !score.abstained && !score.parse_failure
        }
        TemporalGroundingOutcomeV1::ExcludedAmbiguousGroundTruth
        | TemporalGroundingOutcomeV1::ExcludedNotEstimableGroundTruth => false,
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

fn mean(values: &[f64]) -> Option<f64> {
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::temporal_grounding::{
        GroundingPredictionKindV1, TemporalGroundingOutcomeV1, TemporalIntervalMatchV1,
    };

    const EVALUATOR: &str = "mel-temporal-grounding-v1";
    const AUDIO: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const MODEL: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const PROMPT: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
    const ADAPTER: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";
    const EVIDENCE: &str = "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee";
    const RETRIEVAL: &str = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff";

    fn interval(start_frame: u64, end_frame: u64) -> FrameIntervalV1 {
        FrameIntervalV1 {
            start_frame,
            end_frame,
        }
    }

    fn raw_prediction(
        audio: &AudioArtifactClockV1,
        query_id: &str,
        intervals: Vec<FrameIntervalV1>,
    ) -> TemporalGroundingPredictionV1 {
        TemporalGroundingPredictionV1 {
            schema_version: TEMPORAL_GROUNDING_SCHEMA_V1.into(),
            query_id: query_id.into(),
            artifact: audio.clone(),
            state: GroundingPredictionStateV1::Occurrences { intervals },
        }
    }

    fn revision_case_v2(
        truth_intervals: Vec<FrameIntervalV1>,
        draft_intervals: Vec<FrameIntervalV1>,
        final_intervals: Vec<FrameIntervalV1>,
    ) -> RevisionCaseV2 {
        let audio = AudioArtifactClockV1 {
            audio_sha256: AUDIO.into(),
            sample_rate_hz: 48_000,
            channels: 2,
            total_frames: 48_000 * 60,
        };
        let query_id = "query-v2";
        RevisionCaseV2 {
            schema: "mel-temporal-revision-case-v2".into(),
            case_id: "case-v2".into(),
            experiment_unit_id: "unit-v2".into(),
            source_lineage_id: "source-lineage-001".into(),
            model_identity_sha256: MODEL.into(),
            prompt_sha256: PROMPT.into(),
            adapter_sha256: ADAPTER.into(),
            runtime_identity_sha256: EVIDENCE.into(),
            decoding_config_sha256: RETRIEVAL.into(),
            max_output_tokens: 512,
            actual_output_tokens: 128,
            max_wall_time_ms: 30_000,
            elapsed_wall_time_ms: 4_000,
            evidence_condition: EvidenceConditionV1::NoEvidence,
            evidence: None,
            actions: vec![RevisionActionV1::Other],
            truth: TemporalGroundTruthV1 {
                schema_version: TEMPORAL_GROUNDING_SCHEMA_V1.into(),
                query_id: query_id.into(),
                query_text: "Every piano entrance".into(),
                artifact: audio.clone(),
                annotation_policy_version: "annot-v1".into(),
                annotation_set_sha256: EVIDENCE.into(),
                annotator_count: 2,
                state: GroundTruthStateV1::Present,
                intervals: truth_intervals,
                status_reason: None,
            },
            plan: TemporalGroundingPlanV1 {
                evaluator_version: "mel-temporal-grounding-evaluator-v1".into(),
                minimum_interval_iou: 0.5,
            },
            draft_prediction: raw_prediction(&audio, query_id, draft_intervals),
            final_prediction: raw_prediction(&audio, query_id, final_intervals),
        }
    }

    fn positive_score(
        f1: f64,
        tp: usize,
        fp: usize,
        fn_: usize,
    ) -> TemporalGroundingScoreV1 {
        let predicted = tp + fp;
        let truth = tp + fn_;
        let precision = if predicted == 0 { 0.0 } else { tp as f64 / predicted as f64 };
        let recall = if truth == 0 { 0.0 } else { tp as f64 / truth as f64 };
        TemporalGroundingScoreV1 {
            evaluator_version: EVALUATOR.into(),
            query_id: "query-1".into(),
            audio_sha256: AUDIO.into(),
            sample_rate_hz: 48_000,
            outcome: TemporalGroundingOutcomeV1::ScoredPresentQuery,
            prediction_kind: GroundingPredictionKindV1::Occurrences,
            scored: true,
            abstained: false,
            parse_failure: false,
            ground_truth_occurrences: truth,
            predicted_occurrences: predicted,
            ambiguous_candidate_count: 0,
            true_positives: tp,
            false_positives: fp,
            false_negatives: fn_,
            precision: Some(precision),
            recall: Some(recall),
            f1: Some(f1),
            matched_interval_iou_mean: Some(f1),
            count_absolute_error: Some(fp.abs_diff(fn_)),
            exact_occurrence_count: Some(fp == fn_),
            correct_absence_claim: None,
            matches: (0..tp)
                .map(|index| TemporalIntervalMatchV1 {
                    prediction_index: index,
                    annotation_index: index,
                    interval_iou: f1,
                    start_error_frames: 0,
                    end_error_frames: 0,
                })
                .collect(),
        }
    }

    fn case(draft: TemporalGroundingScoreV1, final_score: TemporalGroundingScoreV1) -> RevisionCaseV1 {
        RevisionCaseV1 {
            schema: REVISION_SCHEMA.into(),
            case_id: "case-1".into(),
            experiment_unit_id: "unit-legacy".into(),
            model_identity_sha256: MODEL.into(),
            prompt_sha256: PROMPT.into(),
            adapter_sha256: ADAPTER.into(),
            evidence_condition: EvidenceConditionV1::NoEvidence,
            evidence: None,
            actions: vec![RevisionActionV1::Other],
            draft,
            final_score,
        }
    }

    fn matched_evidence_triplet() -> [RevisionCaseV2; 3] {
        let truth = vec![interval(100, 200), interval(500, 600)];
        let draft = vec![interval(100, 200), interval(800, 900)];
        let mut no_evidence = revision_case_v2(
            truth.clone(),
            draft.clone(),
            draft.clone(),
        );
        no_evidence.case_id = "unit-v2-no-evidence".into();

        let mut local = no_evidence.clone();
        local.case_id = "unit-v2-local".into();
        local.evidence_condition = EvidenceConditionV1::LocalRetrieved;
        local.evidence = Some(RevisionEvidenceV1 {
            source_audio_sha256: AUDIO.into(),
            source_sample_rate_hz: 48_000,
            evidence_payload_sha256: EVIDENCE.into(),
            retrieval_config_sha256: RETRIEVAL.into(),
            start_frame: 0,
            end_frame: 48_000,
        });
        local.final_prediction =
            raw_prediction(&local.truth.artifact, "query-v2", truth.clone());

        let mut yoked = no_evidence.clone();
        yoked.case_id = "unit-v2-yoked".into();
        yoked.evidence_condition = EvidenceConditionV1::YokedRetrieved;
        yoked.evidence = Some(RevisionEvidenceV1 {
            source_audio_sha256:
                "1212121212121212121212121212121212121212121212121212121212121212".into(),
            source_sample_rate_hz: 48_000,
            evidence_payload_sha256: RETRIEVAL.into(),
            retrieval_config_sha256: EVIDENCE.into(),
            start_frame: 0,
            end_frame: 48_000,
        });
        [no_evidence, local, yoked]
    }

    #[test]
    fn matched_evidence_triplet_isolates_local_retrieval_gain() {
        let cases = matched_evidence_triplet();
        let result = compare_matched_evidence_arms(&cases).unwrap();
        assert_eq!(result.len(), 1);
        let contrast = &result[0];
        assert_eq!(contrast.experiment_unit_id, "unit-v2");
        assert_eq!(contrast.local_minus_no_evidence_f1, Some(0.5));
        assert_eq!(contrast.local_minus_yoked_f1, Some(0.5));
        assert_eq!(
            contrast.local_minus_no_evidence_count_gated_f1,
            Some(1.0)
        );
        assert_eq!(
            contrast.local_minus_yoked_count_gated_f1,
            Some(1.0)
        );
        assert_eq!(contrast.local_beats_yoked, Some(true));
        assert_eq!(contrast.local_output_tokens, 128);
        assert_eq!(contrast.no_evidence_output_tokens, 128);
        assert_eq!(contrast.local_elapsed_wall_time_ms, 4_000);
    }

    #[test]
    fn matched_evidence_triplet_rejects_changed_initial_draft() {
        let mut cases = matched_evidence_triplet();
        cases[2].draft_prediction = raw_prediction(
            &cases[2].truth.artifact,
            "query-v2",
            vec![interval(100, 200), interval(700, 800)],
        );
        assert!(compare_matched_evidence_arms(&cases).is_err());
    }

    #[test]
    fn matched_evidence_triplet_requires_each_arm_exactly_once() {
        let cases = matched_evidence_triplet();
        assert!(compare_matched_evidence_arms(&cases[..2]).is_err());
        let mut duplicate = cases;
        duplicate[2].evidence_condition = EvidenceConditionV1::LocalRetrieved;
        assert!(compare_matched_evidence_arms(&duplicate).is_err());
    }

    #[test]
    fn v2_computes_draft_and_final_scores_from_raw_predictions() {
        let case = revision_case_v2(
            vec![interval(100, 200), interval(500, 600)],
            vec![interval(100, 200), interval(800, 900)],
            vec![interval(100, 200), interval(500, 600)],
        );
        let report = audit_revision_v2(&case).unwrap();
        assert_eq!(report.draft.f1, Some(0.5));
        assert_eq!(report.final_score.f1, Some(1.0));
        assert_eq!(report.draft.count_gated_f1, Some(0.0));
        assert_eq!(report.final_score.count_gated_f1, Some(1.0));
        assert_eq!(report.count_gated_f1_delta, Some(1.0));
        assert_eq!(report.helpful_revision, Some(true));
    }

    #[test]
    fn v2_rejects_no_change_when_raw_predictions_differ_even_if_scores_could_match() {
        let mut case = revision_case_v2(
            vec![interval(100, 200), interval(500, 600)],
            vec![interval(100, 200), interval(800, 900)],
            vec![interval(100, 200), interval(700, 800)],
        );
        case.actions = vec![RevisionActionV1::NoChange];
        assert!(audit_revision_v2(&case).is_err());
    }

    #[test]
    fn v2_rejects_predictions_from_a_different_audio_artifact() {
        let mut case = revision_case_v2(
            vec![interval(100, 200), interval(500, 600)],
            vec![interval(100, 200), interval(800, 900)],
            vec![interval(100, 200), interval(500, 600)],
        );
        case.final_prediction.artifact.audio_sha256 =
            "1212121212121212121212121212121212121212121212121212121212121212".into();
        assert!(audit_revision_v2(&case).is_err());
    }

    #[test]
    fn yoked_evidence_cannot_relabel_the_same_audio_with_a_different_sample_rate() {
        let mut case = revision_case_v2(
            vec![interval(100, 200), interval(500, 600)],
            vec![interval(100, 200), interval(800, 900)],
            vec![interval(100, 200), interval(500, 600)],
        );
        case.evidence_condition = EvidenceConditionV1::YokedRetrieved;
        case.evidence = Some(RevisionEvidenceV1 {
            source_audio_sha256: AUDIO.into(),
            source_sample_rate_hz: 44_100,
            evidence_payload_sha256: EVIDENCE.into(),
            retrieval_config_sha256: RETRIEVAL.into(),
            start_frame: 100,
            end_frame: 200,
        });
        assert!(audit_revision_v2(&case).is_err());
    }

    #[test]
    fn v2_local_evidence_must_stay_inside_the_scored_recording() {
        let mut case = revision_case_v2(
            vec![interval(100, 200), interval(500, 600)],
            vec![interval(100, 200), interval(800, 900)],
            vec![interval(100, 200), interval(500, 600)],
        );
        case.evidence_condition = EvidenceConditionV1::LocalRetrieved;
        case.evidence = Some(RevisionEvidenceV1 {
            source_audio_sha256: AUDIO.into(),
            source_sample_rate_hz: 48_000,
            evidence_payload_sha256: EVIDENCE.into(),
            retrieval_config_sha256: RETRIEVAL.into(),
            start_frame: 47_000,
            end_frame: 49_000,
        });
        assert!(audit_revision_v2(&case).is_err());
        case.evidence.as_mut().unwrap().end_frame = 48_000;
        assert!(audit_revision_v2(&case).is_ok());
    }

    #[test]
    fn v2_rejects_inference_budget_overrun() {
        let mut case = revision_case_v2(
            vec![interval(100, 200), interval(500, 600)],
            vec![interval(100, 200), interval(800, 900)],
            vec![interval(100, 200), interval(500, 600)],
        );
        case.actual_output_tokens = case.max_output_tokens + 1;
        assert!(audit_revision_v2(&case).is_err());

        case.actual_output_tokens = 100;
        case.elapsed_wall_time_ms = case.max_wall_time_ms + 1;
        assert!(audit_revision_v2(&case).is_err());
    }

    #[test]
    fn matched_triplet_rejects_different_runtime_or_decoding_identity() {
        let mut cases = matched_evidence_triplet();
        cases[2].runtime_identity_sha256 =
            "1212121212121212121212121212121212121212121212121212121212121212".into();
        assert!(compare_matched_evidence_arms(&cases).is_err());

        let mut cases = matched_evidence_triplet();
        cases[2].decoding_config_sha256 =
            "1212121212121212121212121212121212121212121212121212121212121212".into();
        assert!(compare_matched_evidence_arms(&cases).is_err());
    }

    #[test]
    fn v2_json_parser_uses_raw_scorer_inputs_not_injected_scores() {
        let case = revision_case_v2(
            vec![interval(100, 200), interval(500, 600)],
            vec![interval(100, 200), interval(800, 900)],
            vec![interval(100, 200), interval(500, 600)],
        );
        let encoded = serde_json::to_string(&case).unwrap();
        let report = audit_revision_case_v2(&encoded).unwrap();
        assert_eq!(report.final_score.f1, Some(1.0));
    }

    #[test]
    fn better_final_revision_reports_gain_without_claiming_causality() {
        let draft = positive_score(0.5, 1, 1, 1);
        let final_score = positive_score(1.0, 2, 0, 0);
        let report = audit_revision(&case(draft, final_score)).unwrap();
        assert_eq!(report.f1_delta, Some(0.5));
        assert_eq!(report.helpful_revision, Some(true));
        assert_eq!(report.harmful_revision, Some(false));
        assert_eq!(report.draft.count_gated_f1, Some(0.0));
        assert_eq!(report.final_score.count_gated_f1, Some(1.0));
        assert_eq!(report.count_gated_f1_delta, Some(1.0));
    }

    #[test]
    fn revision_that_damages_a_perfect_draft_is_a_preservation_violation() {
        let draft = positive_score(1.0, 2, 0, 0);
        let final_score = positive_score(0.5, 1, 1, 1);
        let report = audit_revision(&case(draft, final_score)).unwrap();
        assert!(report.draft_was_perfect);
        assert!(report.preservation_violation);
        assert_eq!(report.harmful_revision, Some(true));
    }

    #[test]
    fn cross_query_or_cross_audio_draft_final_pairs_are_rejected() {
        let draft = positive_score(0.5, 1, 1, 1);
        let mut final_score = positive_score(1.0, 2, 0, 0);
        final_score.audio_sha256 = "1212121212121212121212121212121212121212121212121212121212121212".into();
        assert!(audit_revision(&case(draft, final_score)).is_err());
    }

    #[test]
    fn declared_no_change_cannot_hide_a_different_final_score() {
        let mut revision = case(
            positive_score(0.5, 1, 1, 1),
            positive_score(1.0, 2, 0, 0),
        );
        revision.actions = vec![RevisionActionV1::NoChange];
        assert!(audit_revision(&revision).is_err());
    }

    #[test]
    fn absence_revision_reports_improvement_without_relabeling_abstention_as_absence() {
        let absent = audit_revision(&case(
            absent_score(Some(false)),
            absent_score(Some(true)),
        ))
        .unwrap();
        assert_eq!(absent.absence_improved, Some(true));
        assert_eq!(absent.absence_regressed, Some(false));
        assert_eq!(absent.f1_delta, None);

        let abstained = audit_revision(&case(
            absent_score(Some(false)),
            absent_score(None),
        ))
        .unwrap();
        assert_eq!(abstained.absence_improved, Some(false));
        assert_eq!(abstained.absence_regressed, Some(false));
        assert_eq!(abstained.final_score.correct_absence_claim, None);
    }

    #[test]
    fn valid_yoked_evidence_requires_a_distinct_source_recording() {
        let mut yoked = case(positive_score(0.5, 1, 1, 1), positive_score(1.0, 2, 0, 0));
        yoked.evidence_condition = EvidenceConditionV1::YokedRetrieved;
        yoked.evidence = Some(RevisionEvidenceV1 {
            source_audio_sha256: "1212121212121212121212121212121212121212121212121212121212121212".into(),
            source_sample_rate_hz: 44_100,
            evidence_payload_sha256: EVIDENCE.into(),
            retrieval_config_sha256: RETRIEVAL.into(),
            start_frame: 100,
            end_frame: 200,
        });
        assert!(audit_revision(&yoked).is_ok());
    }

    #[test]
    fn local_and_yoked_evidence_have_distinct_source_binding_rules() {
        let mut local = case(positive_score(0.5, 1, 1, 1), positive_score(1.0, 2, 0, 0));
        local.evidence_condition = EvidenceConditionV1::LocalRetrieved;
        local.evidence = Some(RevisionEvidenceV1 {
            source_audio_sha256: AUDIO.into(),
            source_sample_rate_hz: 48_000,
            evidence_payload_sha256: EVIDENCE.into(),
            retrieval_config_sha256: RETRIEVAL.into(),
            start_frame: 100,
            end_frame: 200,
        });
        assert!(audit_revision(&local).is_ok());
        local.evidence_condition = EvidenceConditionV1::YokedRetrieved;
        assert!(audit_revision(&local).is_err());
    }

    #[test]
    fn empty_experiment_unit_identity_is_rejected() {
        let mut input = case(
            positive_score(0.5, 1, 1, 1),
            positive_score(1.0, 2, 0, 0),
        );
        input.experiment_unit_id.clear();
        assert!(audit_revision(&input).is_err());
    }

    #[test]
    fn aggregate_reports_helpful_harmful_and_preservation_denominators_separately() {
        let improved = audit_revision(&case(
            positive_score(0.5, 1, 1, 1),
            positive_score(1.0, 2, 0, 0),
        )).unwrap();
        let harmed = audit_revision(&case(
            positive_score(1.0, 2, 0, 0),
            positive_score(0.5, 1, 1, 1),
        )).unwrap();
        let mut harmed = harmed;
        harmed.case_id = "case-2".into();
        let aggregate = aggregate_revision_reports(&[improved, harmed]).unwrap();
        assert_eq!(aggregate.cases, 2);
        assert_eq!(aggregate.positive_cases, 2);
        assert_eq!(aggregate.helpful_revisions, 1);
        assert_eq!(aggregate.harmful_revisions, 1);
        assert_eq!(aggregate.preservation_violations, 1);
        assert_eq!(aggregate.mean_f1_delta, Some(0.0));
    }
}
